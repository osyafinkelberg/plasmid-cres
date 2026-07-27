from pathlib import Path
import sys
import numpy as np
import polars as pl
from Bio import SeqIO
from tqdm import tqdm

sys.path.insert(0, "..")
from plasmidtools import crest


# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
COMBINED_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"

ENCODINGS_PARQUET = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
PREDS_PARQUET = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
CHECKPOINT_DIR = ADDGENE_DIR / "crest_checkpoints"

CHECKPOINT_CHUNK_SIZE = 50_000  # save to disk after every 50k tiles


def tokenize_dataset(gbk_path: Path, tile_sz: int = 200) -> tuple[list, dict]:
    """
    Phase 1: Scan all plasmids, deduplicate sequences into a unique token map,
    and encode each plasmid as an integer array of token IDs.
    """
    REV_COMP_TABLE = str.maketrans("ATCGN", "TAGCN")

    unique_tile_to_id = {}
    id_to_unique_tile = []
    token_counter = 0
    plasmid_blueprints = {}  

    # Parse dataset
    update_every = 1000
    pbar = tqdm(total=N_PLASMIDS, desc="Generate unique tile dataset")

    for record_idx, record in enumerate(SeqIO.parse(gbk_path, "genbank")):
        gbk_name = record.name or record.id
        seq_str = str(record.seq).upper()
        L = len(seq_str)

        if L < tile_sz:
            continue

        half_sz = tile_sz // 2
        # Extend by full tile_sz to safely accommodate wrap-around from modulo math
        extended_seq = seq_str + seq_str[:tile_sz]

        # Pre-allocate integer arrays for this plasmid's structural layout
        plasmid_ids = np.zeros(L, dtype=np.int32)
        plasmid_is_rev = np.zeros(L, dtype=np.bool_)

        for i in range(L):
            # Shift the start index backward so the tile is centered at i
            start_idx = (i - half_sz) % L
            fwd_tile = extended_seq[start_idx : start_idx + tile_sz]
            rev_tile = fwd_tile[::-1].translate(REV_COMP_TABLE)

            if fwd_tile <= rev_tile:
                canonical, is_rev = fwd_tile, False
            else:
                canonical, is_rev = rev_tile, True

            if canonical not in unique_tile_to_id:
                unique_tile_to_id[canonical] = token_counter
                id_to_unique_tile.append(canonical)
                token_counter += 1

            plasmid_ids[i] = unique_tile_to_id[canonical]
            plasmid_is_rev[i] = is_rev

        plasmid_blueprints[gbk_name] = {
            "tile_ids": plasmid_ids,
            "is_rev": plasmid_is_rev,
            "length": L
        }

        if (record_idx + 1) % update_every == 0 or record_idx == N_PLASMIDS - 1:
            pbar.update(update_every)

    return id_to_unique_tile, plasmid_blueprints


def save_plasmid_encodings(blueprints: dict, output_path: Path) -> None:
    """Packages the primitive structural arrays into a compact Parquet file."""

    df = pl.DataFrame({
        "gbk_name": list(blueprints.keys()),
        "tile_ids": [bp["tile_ids"].tolist() for bp in blueprints.values()],
        "is_rev": [bp["is_rev"].tolist() for bp in blueprints.values()]
    })

    df.write_parquet(output_path, compression="snappy")


if __name__ == "__main__":  # 19 hours on L40S GPU

    # 1. Configuration & Metadata
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    plasmid_meta = pl.read_csv(ADDGENE_DIR / "mammalian_plasmids.tsv", separator="\t")
    N_PLASMIDS = plasmid_meta.filter(pl.col("download_status") == "200").height

    # 2. Preprocessing: Tokenize and Map
    id_to_unique_tile, plasmid_blueprints = tokenize_dataset(COMBINED_GBK, tile_sz=200)

    # Save the structural encodings safely to disk
    if not ENCODINGS_PARQUET.exists():
        save_plasmid_encodings(plasmid_blueprints, ENCODINGS_PARQUET)

    # 3. Checkpointing logic: Identify what has already been computed
    completed_ids = set()
    checkpoint_files = list(CHECKPOINT_DIR.glob("preds_chunk_*.parquet"))

    if checkpoint_files:
        for ckpt in checkpoint_files:
            completed_ids.update(pl.read_parquet(ckpt, columns=["tile_ID"])["tile_ID"].to_list())

    # Filter for pending tasks
    pending_tasks = [
        (tile_id, seq) for tile_id, seq in enumerate(id_to_unique_tile) 
        if tile_id not in completed_ids
    ]
    n_pending = len(pending_tasks)

    # 4. Inference loop with Chunked Saves
    if n_pending > 0:
        pbar = tqdm(total=n_pending, desc="Predicting CREs")

        # Process in isolated chunks to ensure safe memory offloading
        for chunk_idx in range(0, n_pending, CHECKPOINT_CHUNK_SIZE):
            chunk = pending_tasks[chunk_idx : chunk_idx + CHECKPOINT_CHUNK_SIZE]

            # Re-instantiate the predictor to keep state clean per chunk
            cre_predictor = crest.CREST(crest.BATCH_SIZE)
            for tile_id, tile in chunk:
                cre_predictor.update(f"{tile_id}", tile)

            valid_ids, cre_preds = cre_predictor.get_predictions()
            chunk_df = pl.DataFrame(
                {"tile_ID": valid_ids.astype(np.int32)} |
                {cell: cre_preds[:, cell_idx] for cell_idx, cell in enumerate(crest.CREST_LABELS)}
            )

            # Save chunk to disk
            chunk_path = CHECKPOINT_DIR / f"preds_chunk_{chunk_idx}_{chunk_idx + len(chunk)}.parquet"
            chunk_df.write_parquet(chunk_path, compression="snappy")

            pbar.update(len(chunk))

        pbar.close()

    # 5. Final Aggregation
    all_chunks = [pl.read_parquet(p) for p in CHECKPOINT_DIR.glob("preds_chunk_*.parquet")]
    if all_chunks:
        final_predictions = pl.concat(all_chunks).sort("tile_ID")
        final_predictions.write_parquet(PREDS_PARQUET, compression="snappy")

import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, "..")
from plasmidtools import helpers

# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
ELEMENT_FILE = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
PRIMERS_FILE = ADDGENE_DIR / "mammalian_plasmids_primers.parquet"
REPR_SEQ_FASTA = ADDGENE_DIR / "element_representative_sequences.fasta"
CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"
CRE_TSS_FILE = ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet"
STATS_FILE = ADDGENE_DIR / "mammalian_plasmids_statistics.parquet"
ELEMENT_OVERLAPS_OUT = ADDGENE_DIR / "mammalian_plasmids_element_cre_overlaps.parquet"
PRIMERS_OVERLAPS_OUT = ADDGENE_DIR / "mammalian_plasmids_primers_cre_overlaps.parquet"
REPR_SEQ_OVERLAPS = ADDGENE_DIR / "element_representative_sequences_cre_overlaps.parquet"
TSS_FLANK_SIZE = 50


def crest_cells(columns) -> list[str]:
    """CREST cell-line names present as `CREST (<cell>)` columns, in file order."""
    prefix = "CREST ("
    return [col[len(prefix):-1] for col in columns if col.startswith(prefix)]


def cre_column(metric: str, cell: str) -> str:
    """Name of a per-cell-line CRE metric column."""
    return f"{metric} ({cell})"


def get_midpoints(intervals: list, plasmid_length: int) -> np.ndarray:
    """Calculates midpoints for a list of [start, end] intervals, handling origin wraps."""
    if not intervals:
        return np.array([])

    mids = []
    for s, e in intervals:
        if s <= e:
            mids.append((s + e) / 2.0)
        else:
            # handle wrapping across the circular boundary (e.g., origin)
            mids.append(((s + e + plasmid_length) / 2.0) % plasmid_length)
    return np.array(mids)


def get_interval_indices(intervals: list, L: int) -> np.ndarray:
    """
    Expands a list of [start, end] intervals into a flattened array 
    of unique sequence indices, handling circular wrap-around.
    """
    if intervals is None:
        return np.array([], dtype=int)

    indices = []
    for s, e in intervals:
        if s <= e:
            indices.extend(range(s, e))
        else:  # Wrap around origin
            indices.extend(range(s, L))
            indices.extend(range(e))

    return np.unique(np.array(indices) % L)


def safe_nanmean(arr: np.ndarray, idxs: np.ndarray) -> float:
    """Safely calculates nanmean, returning np.nan if array is empty or all NaNs."""
    if len(idxs) == 0:
        return np.nan

    subset = arr[idxs]
    if np.isnan(subset).all():
        return np.nan

    return float(np.nanmean(subset))


def calculate_overlap_statistics(elements_path: Path, output_path: Path) -> None:
    # 1. Load Datasets
    elements_df = pl.read_parquet(elements_path)
    cre_tss_df = pl.read_parquet(CRE_TSS_FILE)
    stats_df = pl.read_parquet(STATS_FILE)

    cells = crest_cells(cre_tss_df.columns)
    print(f"CRE statistics for {len(cells)} CREST cell line(s): {', '.join(cells)}")

    # Pre-load CREST global arrays to allow avg_signal extraction, one row per cell line.
    tile_encoding = pl.read_parquet(CREST_TILE_ENCOD)
    cre_predictions = pl.read_parquet(CREST_TILE_PREDS)
    n_crest_tiles = tile_encoding["tile_ids"].list.max().max() + 1
    pred_tile_ids = cre_predictions["tile_ID"].to_numpy()
    crest_tile_preds = np.full((len(cells), n_crest_tiles), np.nan)
    for cell_idx, cell in enumerate(cells):
        crest_tile_preds[cell_idx, pred_tile_ids] = cre_predictions[cell].to_numpy()

    # optimization: flat int32 tile-id array plus per-plasmid offsets. Holding
    # the same 442M ids as Python lists costs ~15 GB; this costs ~1.8 GB.
    tile_ids_flat = tile_encoding["tile_ids"].explode().to_numpy().astype(np.int32)
    tile_offsets = np.concatenate(([0], np.cumsum(tile_encoding["tile_ids"].list.len().to_numpy().astype(np.int64))))
    crest_tile_span = {
        gbk_name: (int(tile_offsets[i]), int(tile_offsets[i + 1]))
        for i, gbk_name in enumerate(tile_encoding["gbk_name"].to_list())
    }

    # 2. Plasmid-level lookups.
    # Only the raw interval lists are held for every plasmid (a few hundred bytes
    # each). Expanding them into index sets is what costs memory - roughly 6 GB
    # per cell line across the whole collection - so that is done one plasmid at
    # a time in the loop below, which keeps peak memory flat as cell lines are
    # added rather than growing 8-fold.
    plasmid_length = dict(stats_df.select(["gbk_name", "plasmid_length"]).iter_rows())
    cre_track_names = [f"CREST ({cell})" for cell in cells]
    cre_intervals = {}
    for row in cre_tss_df.iter_rows(named=True):
        cre_intervals[row["gbk_name"]] = (
            [row[track] or [] for track in cre_track_names],
            row["Puffin (FANTOM_CAGE_fwd)"] or [],
            row["Puffin (FANTOM_CAGE_rev)"] or [],
        )

    def empty_metrics() -> dict:
        """Metric block for an element whose body maps to zero base pairs."""
        record = {
            "tss_hits": 0, "tss_fwd_hits": 0, "tss_rev_hits": 0,
            "tss_fwd_avg_signal": np.nan, "tss_rev_avg_signal": np.nan,
            "fraction_tss_fwd_bp": 0.0, "fraction_tss_rev_bp": 0.0,
        }
        for cell in cells:
            record[cre_column("cre_hits", cell)] = 0
            record[cre_column("cre_avg_signal", cell)] = np.nan
            record[cre_column("fraction_cre_bp", cell)] = 0.0
        return record

    # 3. Iterate through elements to count overlaps
    overlap_records = []
    custom_types = {"backbone_spacer", "putative_orf", "putative_noncoding"}

    with h5py.File(PUFFIN_PREDS, "r") as h5f:
        puffin_feat_names = h5f.attrs["features"]
        puffin_fwd_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE fwd")[0, 0])
        puffin_rev_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE rev")[0, 0])

        def plasmid_state(gbk_name: str) -> dict | None:
            """Expand one plasmid's intervals into midpoints, index sets and signals."""
            L = plasmid_length.get(gbk_name)
            if L is None or gbk_name not in cre_intervals or gbk_name not in crest_tile_span:
                return None

            cre_iv, fwd_iv, rev_iv = cre_intervals[gbk_name]
            span_start, span_end = crest_tile_span[gbk_name]
            puffin_preds = h5f[gbk_name][:]

            return {
                "L": L,
                "crest_signals": crest_tile_preds[:, tile_ids_flat[span_start:span_end]],
                "puffin_fwd": puffin_preds[puffin_fwd_idx],
                "puffin_rev": puffin_preds[puffin_rev_idx],
                "cre_mids_int": [np.floor(get_midpoints(iv, L)).astype(int).tolist() for iv in cre_iv],
                "cre_idx_set": [set(get_interval_indices(iv, L)) for iv in cre_iv],
                "fwd_mids_int": np.floor(get_midpoints(fwd_iv, L)).astype(int).tolist(),
                "rev_mids_int": np.floor(get_midpoints(rev_iv, L)).astype(int).tolist(),
                "fwd_idx_set": set(get_interval_indices(fwd_iv, L)),
                "rev_idx_set": set(get_interval_indices(rev_iv, L)),
            }

        update_every = 25_000
        N_ELEMENT_INSTANCES = elements_df.height
        pbar = tqdm(total=N_ELEMENT_INSTANCES, desc="Mapping overlaps")

        # optimization: element rows are stored contiguously per plasmid, so a
        # single-plasmid cache replaces the dictionaries that previously grew to
        # hold every plasmid's Puffin and CREST tracks at once.
        current_gbk, sd = None, None

        for row_idx, row in enumerate(elements_df.iter_rows(named=True)):
            gbk_name = row["gbk_name"]
            if gbk_name != current_gbk:
                current_gbk, sd = gbk_name, plasmid_state(gbk_name)

            if sd is None:
                if (row_idx + 1) % update_every == 0 or row_idx == N_ELEMENT_INSTANCES - 1:
                    pbar.update(update_every)
                continue

            L = sd["L"]
            element_intervals = row["intervals"]
            strand = row["strand"]
            e_type = row["element_type"]
            e_name = row["element_name"]

            if e_type in custom_types:
                if e_type == "backbone_spacer":
                    out_type, out_name = "backbone", "backbone_spacer"
                elif e_type == "putative_orf":
                    out_type, out_name = "putative_insert", "putative_orf"
                elif e_type == "putative_noncoding":
                    out_type, out_name = "putative_insert", "putative_noncoding"
            else:
                out_type, out_name = e_type, e_name

            # --- Map the Element Body and Flanks ---
            body_indices = []
            for s, e in element_intervals:
                if s <= e:
                    body_indices.extend(range(s, e))
                else:
                    body_indices.extend(range(s, L))
                    body_indices.extend(range(e))

            body_idx = np.array(body_indices) % L

            # log empty stats and skip the rest of the loop for this element
            if len(body_idx) == 0:
                overlap_records.append({"type": out_type, "name": out_name, **empty_metrics()})
                continue

            genomic_start = body_indices[0] % L
            genomic_end = (body_indices[-1] + 1) % L

            genomic_left = np.arange(genomic_start - TSS_FLANK_SIZE, genomic_start) % L
            genomic_right = np.arange(genomic_end, genomic_end + TSS_FLANK_SIZE) % L

            puffin_fwd = sd["puffin_fwd"]
            puffin_rev = sd["puffin_rev"]

            # --- Strand-aware Signal, Midpoint, and Index Resolution ---
            if strand == -1:
                feat_downstream_idx = genomic_left
                feat_upstream_idx = genomic_right

                signal_fwd = puffin_rev
                signal_rev = puffin_fwd

                mids_fwd = sd["rev_mids_int"]
                mids_rev = sd["fwd_mids_int"]

                full_fwd_set = sd["rev_idx_set"]
                full_rev_set = sd["fwd_idx_set"]
            else:
                feat_downstream_idx = genomic_right
                feat_upstream_idx = genomic_left

                signal_fwd = puffin_fwd
                signal_rev = puffin_rev

                mids_fwd = sd["fwd_mids_int"]
                mids_rev = sd["rev_mids_int"]

                full_fwd_set = sd["fwd_idx_set"]
                full_rev_set = sd["rev_idx_set"]

            # --- Calculate Sets ---
            body_set = set(body_idx)
            # optimization: Native python union is much faster than np.concatenate + set()
            fwd_set = body_set.union(feat_downstream_idx)
            rev_set = body_set.union(feat_upstream_idx)

            # --- Calculate Counts (Based on midpoints) ---
            # Using generator sum avoids destroying duplicate midpoints while keeping O(1) set lookups
            tss_fwd_hits = sum(1 for m in mids_fwd if m in fwd_set)
            tss_rev_hits = sum(1 for m in mids_rev if m in rev_set)

            # --- Calculate Averages & Fractions (Based on full interval intersections) ---
            tss_fwd_overlap_idx = np.array(list(full_fwd_set & fwd_set), dtype=int)
            tss_rev_overlap_idx = np.array(list(full_rev_set & rev_set), dtype=int)

            # Signal extraction 
            tss_fwd_avg_signal = safe_nanmean(signal_fwd, tss_fwd_overlap_idx)
            tss_rev_avg_signal = safe_nanmean(signal_rev, tss_rev_overlap_idx)

            # Fraction calculations
            len_body = len(body_idx)
            len_fwd = len(fwd_set)
            len_rev = len(rev_set)

            frac_tss_fwd = len(tss_fwd_overlap_idx) / len_fwd if len_fwd > 0 else 0.0
            frac_tss_rev = len(tss_rev_overlap_idx) / len_rev if len_rev > 0 else 0.0

            record = {
                "type": out_type,
                "name": out_name,
                "tss_hits": tss_fwd_hits + tss_rev_hits,
                "tss_fwd_hits": tss_fwd_hits,
                "tss_rev_hits": tss_rev_hits,
                "tss_fwd_avg_signal": tss_fwd_avg_signal,
                "tss_rev_avg_signal": tss_rev_avg_signal,
                "fraction_tss_fwd_bp": frac_tss_fwd,
                "fraction_tss_rev_bp": frac_tss_rev,
            }

            # --- Per-cell-line CRE overlap ---
            for cell_idx, cell in enumerate(cells):
                cre_overlap_idx = np.array(list(sd["cre_idx_set"][cell_idx] & body_set), dtype=int)
                record[cre_column("cre_hits", cell)] = sum(1 for m in sd["cre_mids_int"][cell_idx] if m in body_set)
                record[cre_column("cre_avg_signal", cell)] = safe_nanmean(sd["crest_signals"][cell_idx], cre_overlap_idx)
                record[cre_column("fraction_cre_bp", cell)] = len(cre_overlap_idx) / len_body

            overlap_records.append(record)

            if (row_idx + 1) % update_every == 0 or row_idx == N_ELEMENT_INSTANCES - 1:
                pbar.update(update_every)
    pbar.close()

    # 4. Group by type & name, then average across all instances of that element
    if overlap_records:
        res_df = pl.DataFrame(overlap_records)
        aggregations = [
            pl.col("tss_hits").mean().alias("n_tss_midpoints"),
            pl.col("tss_fwd_hits").mean().alias("n_tss_fwd_midpoints"),
            pl.col("tss_rev_hits").mean().alias("n_tss_rev_midpoints"),

            pl.col("tss_fwd_avg_signal").drop_nans().mean().alias("tss_fwd_avg_signal"),
            pl.col("tss_rev_avg_signal").drop_nans().mean().alias("tss_rev_avg_signal"),

            pl.col("fraction_tss_fwd_bp").mean().alias("fraction_tss_fwd_bp"),
            pl.col("fraction_tss_rev_bp").mean().alias("fraction_tss_rev_bp"),
        ]
        for cell in cells:
            aggregations.extend([
                pl.col(cre_column("cre_hits", cell)).mean().alias(cre_column("n_cre_midpoints", cell)),
                pl.col(cre_column("cre_avg_signal", cell)).drop_nans().mean().alias(cre_column("cre_avg_signal", cell)),
                pl.col(cre_column("fraction_cre_bp", cell)).mean().alias(cre_column("fraction_cre_bp", cell)),
            ])

        final_stats = res_df.group_by(["type", "name"]).agg(aggregations)

        final_stats = final_stats.sort(["type", "name"])
        final_stats.write_parquet(output_path)


def extract_representative_sequence_relative_cre_overlaps(output_path: Path) -> pl.DataFrame:
    """
    Extracts relative positions of CRE and TSS elements for representative sequences,
    accounting for sequence flanks, circular wrap-arounds, and strand orientation.
    """
    print("Extracting relative CRE / TSS positions for representative sequences...")

    cre_tss_df = pl.read_parquet(CRE_TSS_FILE)
    plasmid_stats = pl.read_parquet(STATS_FILE)
    representative_map = helpers.load_representative_sequences(REPR_SEQ_FASTA)

    # 1. Create fast lookups for regulatory intervals and plasmid lengths
    cre_lookup = {}
    for row in cre_tss_df.iter_rows(named=True):
        cre_lookup[row["gbk_name"]] = {
            "CREST (HEK293T)": row["CREST (HEK293T)"] or [],
            "Puffin (FANTOM_CAGE_fwd)": row["Puffin (FANTOM_CAGE_fwd)"] or [],
            "Puffin (FANTOM_CAGE_rev)": row["Puffin (FANTOM_CAGE_rev)"] or []
        }

    len_lookup = dict(plasmid_stats.select(["gbk_name", "plasmid_length"]).iter_rows())
    results = []

    # 2. Iterate through representative map and extract local relative overlaps
    for (f_type, f_name), meta in representative_map.items():
        gbk = meta["representative_plasmid_gbk"]
        strand = meta["strand"]
        plasmid_position = meta["plasmid_position"]

        L = len_lookup.get(gbk)
        if not L:
            continue

        gbk_cre = cre_lookup.get(gbk, {
            "CREST (HEK293T)": [], "Puffin (FANTOM_CAGE_fwd)": [], "Puffin (FANTOM_CAGE_rev)": []
        })

        # Reconstruct the exact linear indices of the extracted sequence space
        full_indices = []
        for start, end in plasmid_position:
            full_indices.extend(range(start, end))

        extracted_intervals = {}

        # Process each regulatory track via boolean masking
        for track_name in ["CREST (HEK293T)", "Puffin (FANTOM_CAGE_fwd)", "Puffin (FANTOM_CAGE_rev)"]:
            intervals = gbk_cre[track_name]
            
            # Map global intervals to the circular plasmid length
            mask = np.zeros(L, dtype=bool)
            for c_s, c_e in intervals:
                if c_s < c_e:
                    mask[c_s:c_e] = True
                else:
                    mask[c_s:L] = True
                    mask[0:c_e] = True
                    
            # Slice out the specific window matching the representative sequence
            sliced_mask = mask[full_indices]
            
            # Flip coordinates to maintain 5' -> 3' orientation relative to the element
            if strand == -1:
                sliced_mask = sliced_mask[::-1]
                
            # Collapse the continuous True blocks back into relative [start, end) intervals
            diff = np.diff(np.concatenate(([0], sliced_mask.view(np.int8), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            extracted_intervals[track_name] = [[int(s), int(e)] for s, e in zip(starts, ends)]
            
        # 3. Orient FWD/REV tracks correctly for negative strand instances
        if strand == -1:
            fwd_temp = extracted_intervals["Puffin (FANTOM_CAGE_fwd)"]
            extracted_intervals["Puffin (FANTOM_CAGE_fwd)"] = extracted_intervals["Puffin (FANTOM_CAGE_rev)"]
            extracted_intervals["Puffin (FANTOM_CAGE_rev)"] = fwd_temp

        results.append({
            "element_type": f_type,
            "element_name": f_name,
            "CREST (HEK293T)": extracted_intervals["CREST (HEK293T)"],
            "Puffin (FANTOM_CAGE_fwd)": extracted_intervals["Puffin (FANTOM_CAGE_fwd)"],
            "Puffin (FANTOM_CAGE_rev)": extracted_intervals["Puffin (FANTOM_CAGE_rev)"]
        })

    df_out = pl.DataFrame(results)
    df_out.write_parquet(output_path)
    print(f"Saved relative overlaps to {output_path}")

    return df_out


if __name__ == "__main__":
    calculate_overlap_statistics(ELEMENT_FILE, ELEMENT_OVERLAPS_OUT)  # 20 min, 8 cell lines
    calculate_overlap_statistics(PRIMERS_FILE, PRIMERS_OVERLAPS_OUT)  # 11 min, 8 cell lines
    extract_representative_sequence_relative_cre_overlaps(REPR_SEQ_OVERLAPS)  # 2 sec

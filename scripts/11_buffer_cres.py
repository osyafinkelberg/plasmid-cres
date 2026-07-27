"""
Candidate cryptic CREs in Buffer (non-functional plasmid backone) sequences.
"""
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from tqdm import tqdm

# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
ELEMENT_FILE = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"
CRE_TSS_FILE = ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet"
STATS_FILE = ADDGENE_DIR / "mammalian_plasmids_statistics.parquet"

BUFFER_CRES_OUT = ADDGENE_DIR / "buffer_cres.parquet"

# --- PARAMETERS ---
MIN_FEATURE_EXCLUSION_LEN = 100
CRE_CORE_WINDOW = 400
TSS_CORE_WINDOW = 650


def get_interval_indices(intervals: list, L: int) -> np.ndarray:
    """Expands a list of [start, end] intervals into a flattened array of unique sequence indices."""
    if not intervals:
        return np.array([], dtype=int)
    indices = []
    for s, e in intervals:
        if s <= e:
            indices.extend(range(s, e))
        else:  
            indices.extend(range(s, L))
            indices.extend(range(e))
    return np.unique(np.array(indices) % L)


def safe_nanmean(arr: np.ndarray, idxs: list, metric: str) -> float:
    """Safely calculates nanmean across specific indices."""
    if not len(idxs):
        return np.nan
    subset = arr[idxs]
    if np.isnan(subset).all():
        return np.nan
    if metric == "max":
        return float(np.nanmax(subset))
    return float(np.nanmean(subset))


def extract_buffer_regulatory_elements() -> None:
    print("Loading datasets and mapping structures...")

    # 1. Load Core Datasets
    elements_df = pl.read_parquet(ELEMENT_FILE)
    cre_tss_df = pl.read_parquet(CRE_TSS_FILE)
    stats_df = pl.read_parquet(STATS_FILE)

    # Extract dynamically available tracks
    cre_tss_columns = [c for c in cre_tss_df.columns if c.startswith(("CREST (", "Puffin ("))]

    # Extract lengths
    plasmid_lengths = dict(stats_df.select(["gbk_name", "plasmid_length"]).iter_rows())
    sequence_map = dict(stats_df.select(["gbk_name", "sequence_id"]).iter_rows())

    # 2. Pre-load CREST Global Predictions
    tile_encoding = pl.read_parquet(CREST_TILE_ENCOD)
    cre_predictions = pl.read_parquet(CREST_TILE_PREDS)
    crest_cells = [c for c in cre_predictions.columns if c not in ("tile_ID", "sequence_id", "gbk_name", "sequence")]

    n_crest_tiles = tile_encoding["tile_ids"].explode().max() + 1
    cell_preds = {}
    for cell in crest_cells:
        arr = np.full(n_crest_tiles, np.nan)
        arr[cre_predictions["tile_ID"].to_numpy()] = cre_predictions[cell].to_numpy()
        cell_preds[cell] = arr
  
    crest_tiles_dict = dict(zip(tile_encoding["gbk_name"].to_list(), tile_encoding["tile_ids"].to_list()))

    # 3. Pre-parse Plasmid Features for O(1) Access
    features_by_gbk = {}
    for row in elements_df.iter_rows(named=True):
        gbk = row["gbk_name"]
        if gbk not in features_by_gbk:
            features_by_gbk[gbk] = []
        features_by_gbk[gbk].append({
            "type": row["element_type"],
            "name": row["element_name"],
            "intervals": row["intervals"]
        })

    results = []

    # 4. Global Discovery & Filtering Loop
    with h5py.File(PUFFIN_PREDS, "r") as h5f:
        puffin_feat_names = h5f.attrs["features"]
        puffin_fwd_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE fwd")[0, 0])
        puffin_rev_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE rev")[0, 0])

        for row in tqdm(cre_tss_df.iter_rows(named=True), total=cre_tss_df.height, desc="Processing Plasmids"):
            gbk = row["gbk_name"]
            L = plasmid_lengths.get(gbk)
            if not L:
                continue

            seq_id = sequence_map[gbk]
            
            # --- Compile Exclusion Rules for Current Plasmid ---
            parsed_features = []
            for feat in features_by_gbk.get(gbk, []):
                f_type = feat["type"]

                # Backbone spacers are natively ignored (they are the buffer we want)
                if f_type == "backbone_spacer":
                    continue

                f_idx = set(get_interval_indices(feat["intervals"], L))

                # Assign logic tier
                if f_type in ["putative_orf", "putative_noncoding"]:
                    parsed_features.append({"name": f"{f_type}:{feat['name']}", "indices": f_idx, "tier": "soft"})
                elif len(f_idx) >= MIN_FEATURE_EXCLUSION_LEN:
                    parsed_features.append({"name": f"{f_type}:{feat['name']}", "indices": f_idx, "tier": "hard"})
                # Short known features (< 100bp) are ignored, treated as part of the neutral buffer

            sd_cache = {"crest_cache": {}, "puffin_cache": {}}

            # --- Iterate Over Regulatory Tracks ---
            for track_name in cre_tss_columns:
                intervals_list = row[track_name]
                if not intervals_list:
                    continue

                is_cre = track_name.startswith("CREST")

                for i, interval in enumerate(intervals_list):
                    s, e = interval
                    raw_indices_arr = get_interval_indices([interval], L)
                    raw_len = len(raw_indices_arr)
                    if raw_len == 0:
                        continue

                    raw_idx_set = set(raw_indices_arr)

                    # Calculate true geometric midpoint
                    if s <= e:
                        midpoint = int(np.floor((s + e) / 2.0))
                    else:
                        midpoint = int(np.floor(((s + e + L) / 2.0) % L))

                    # --- Define Geometry Windows ---
                    if is_cre:
                        r_core = CRE_CORE_WINDOW // 2
                    else:
                        r_core = TSS_CORE_WINDOW // 2

                    core_set = set(np.arange(midpoint - r_core, midpoint + r_core) % L)

                    # --- Evaluate Conditions & Functional Overlaps ---
                    drop_element = False
                    flags = []

                    for pf in parsed_features:
                        # isdisjoint() is implemented in C and stops at the first overlap, making it extremely fast
                        overlap_core = not core_set.isdisjoint(pf["indices"])
                        overlap_target = not raw_idx_set.isdisjoint(pf["indices"])

                        if pf["tier"] == "hard":
                            if overlap_core:
                                drop_element = True
                                break
                            elif overlap_target:
                                flags.append(pf["name"])
                        else:  # Soft
                            if overlap_core or overlap_target:
                                flags.append(pf["name"])

                    if drop_element:
                        continue

                    # --- Calculate Activity ---
                    if is_cre:
                        cell_name = track_name.removeprefix("CREST (")[:-1]
                        if gbk not in sd_cache["crest_cache"]:
                            tiles = crest_tiles_dict.get(gbk, [])
                            sd_cache["crest_cache"][gbk] = {c: cell_preds[c][tiles] for c in crest_cells}
                        sig_arr = sd_cache["crest_cache"][gbk][cell_name]
                        activity = safe_nanmean(sig_arr, list(raw_indices_arr), metric="mean")
                    else:
                        if gbk not in sd_cache["puffin_cache"]:
                            sd_cache["puffin_cache"][gbk] = h5f[gbk][:]
                        if track_name == "Puffin (FANTOM_CAGE_fwd)":
                            sig_arr = sd_cache["puffin_cache"][gbk][puffin_fwd_idx]
                        else:
                            sig_arr = sd_cache["puffin_cache"][gbk][puffin_rev_idx]
                        activity = safe_nanmean(sig_arr, list(raw_indices_arr), metric="max")

                    # --- Record Entry ---
                    cre_raw_id = f"{gbk}_{track_name.replace(' ', '_')}_{i}"

                    results.append({
                        "gbk_name": gbk,
                        "sequence_id": seq_id,
                        "cre_raw_id": cre_raw_id,
                        "cre_type": track_name,
                        "cre_position": interval,
                        "cre_length": raw_len,
                        "cre_activity": activity,
                        "overlapped_features": flags
                    })

    # 5. Build Final Dataset
    if results:
        res_df = pl.DataFrame(results)
        res_df.write_parquet(BUFFER_CRES_OUT)
        print(f"\nSaved {res_df.height} buffer elements to {BUFFER_CRES_OUT}")
    else:
        print("\nNo buffer elements survived the exclusion criteria.")


if __name__ == "__main__":
    extract_buffer_regulatory_elements()  # 20 min

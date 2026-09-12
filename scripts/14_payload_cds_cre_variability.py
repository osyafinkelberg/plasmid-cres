from pathlib import Path

import h5py
import numpy as np
import polars as pl
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
ADDGENE_DIR = DATA_DIR / "addgene"
COMBINED_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"
STATS_FILE = ADDGENE_DIR / "mammalian_plasmids_statistics.parquet"

ALIGN_DIR = DATA_DIR / "addgene_alignments"
UNIQUE_SEQUENCE_IDS = ALIGN_DIR / "element_unique_sequence_ids.parquet"
UNIQUE_SEQUENCE_IDS_CDS = ALIGN_DIR / "element_unique_sequence_ids_cds.parquet"

CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"
CRE_TSS_FILE = ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet"

OUT_CDS_CRE_VARIABILITY = ALIGN_DIR / "cds_instance_cres.parquet"
TSS_FLANK_SIZE = 50 


def get_midpoints(intervals: list, plasmid_length: int) -> np.ndarray:
    """Calculates midpoints for a list of [start, end] intervals, handling origin wraps."""
    if not intervals:
        return np.array([])

    mids = []
    for s, e in intervals:
        if s <= e:
            mids.append((s + e) / 2.0)
        else:
            mids.append(((s + e + plasmid_length) / 2.0) % plasmid_length)
    return np.array(mids)


def get_interval_indices(intervals: list, L: int) -> np.ndarray:
    """Expands a list of [start, end] intervals into a flattened array of unique sequence indices."""
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


def circular_bounds(intervals: list, L: int) -> tuple[int, int]:
    """Start and end of an element body on the circular plasmid, as [start, end).

    Parts are merged and the widest gap between them is taken as the outside of
    the element, as `04_addgene_msa.py` does. The stored interval order cannot be
    used instead: Biopython lists minus-strand joins 5' to 3', so the first part's
    start and the last part's end mark an internal junction, not the two ends.
    """
    parts = []
    for s, e in intervals:
        if s > e:  # wraps the origin
            parts.extend([(s, L), (0, e)])
        else:
            parts.append((s, e))
    parts.sort()

    merged = [parts[0]]
    for s, e in parts[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    gaps = [(merged[(i + 1) % len(merged)][0] - merged[i][1]) % L for i in range(len(merged))]
    widest = int(np.argmax(gaps))
    return merged[(widest + 1) % len(merged)][0] % L, merged[widest][1] % L


def safe_nanmean(arr: np.ndarray, idxs: np.ndarray, metric: str) -> float:
    """Safely reduces `arr` over `idxs`, returning np.nan if empty or all NaNs.

    `metric="mean"` averages the overlap, `metric="max"` takes its peak.
    """
    if len(idxs) == 0:
        return np.nan

    subset = arr[idxs]
    if np.isnan(subset).all():
        return np.nan

    if metric == "max":
        return float(np.nanmax(subset))

    return float(np.nanmean(subset))


def owned_peak_indices(peaks: list[set], mids: list[int], window: set) -> np.ndarray:
    """Positions inside `window` belonging to the TSS peaks the window owns.

    A peak is owned when its midpoint falls in the window - the same test the
    midpoint counts use - so an element cannot report activity for a peak it did
    not count. Clipping a neighbouring peak's shoulder contributes nothing:
    without this, an element that catches the tail of a wide peak takes that
    peak's height while counting no midpoint at all.
    """
    owned = [peak for peak, mid in zip(peaks, mids) if mid in window]
    if not owned:
        return np.array([], dtype=int)

    return np.fromiter(set.union(*owned) & window, dtype=int)


def calculate_instance_overlap_statistics() -> None:
    # 1. Load Datasets and Map IDs
    print("Loading datasets and mapping IDs...")

    # Load Nucleotide IDs and rename unique_id
    unique_ids_df = (
        pl.read_parquet(UNIQUE_SEQUENCE_IDS)
        .filter(pl.col("element_type") == "CDS")
        .rename({"unique_id": "unique_nuc_id"})
    )

    # Load AA IDs, explode the mapping list, and rename columns to prepare for join
    aa_ids_df = (
        pl.read_parquet(UNIQUE_SEQUENCE_IDS_CDS)
        .explode("unique_nuc_ids")
        .select([
            pl.col("unique_id").alias("unique_aa_id"),
            pl.col("unique_nuc_ids").alias("unique_nuc_id")
        ])
    )

    # Join AA IDs onto Nucleotide IDs
    mapped_ids_df = unique_ids_df.join(aa_ids_df, on="unique_nuc_id", how="left")

    # Explode the grouped lists to get one row per CDS instance
    instances_df = mapped_ids_df.explode(["gbk_names", "positions", "strands"]).rename({
        "gbk_names": "gbk_name",
        "positions": "position",
        "strands": "strand"
    })

    cre_tss_df = pl.read_parquet(CRE_TSS_FILE)
    stats_df = pl.read_parquet(STATS_FILE)

    # Pre-load CREST global arrays to allow avg_signal extraction
    tile_encoding = pl.read_parquet(CREST_TILE_ENCOD)
    cre_predictions = pl.read_parquet(CREST_TILE_PREDS)
    n_crest_tiles = tile_encoding["tile_ids"].list.max().max() + 1
    hek293t_tile_preds = np.full(n_crest_tiles, np.nan)
    hek293t_tile_preds[cre_predictions["tile_ID"].to_numpy()] = cre_predictions["HEK293T"].to_numpy()

    crest_tiles_dict = dict(zip(tile_encoding["gbk_name"].to_list(), tile_encoding["tile_ids"].to_list()))

    # 2. Pre-calculate midpoints AND full interval indices
    print("Pre-calculating plasmid features...")
    seq_data = {}
    for row in stats_df.iter_rows(named=True):
        seq_data[row["gbk_name"]] = {"L": row["plasmid_length"]}

    for row in cre_tss_df.iter_rows(named=True):
        gbk_name = row["gbk_name"]
        if gbk_name not in seq_data:
            continue
        L = seq_data[gbk_name]["L"]

        seq_data[gbk_name]["cre_mids_int"] = np.floor(get_midpoints(row["CREST (HEK293T)"], L)).astype(int).tolist()
        seq_data[gbk_name]["fwd_mids_int"] = np.floor(get_midpoints(row["Puffin (FANTOM_CAGE_fwd)"], L)).astype(int).tolist()
        seq_data[gbk_name]["rev_mids_int"] = np.floor(get_midpoints(row["Puffin (FANTOM_CAGE_rev)"], L)).astype(int).tolist()

        seq_data[gbk_name]["cre_idx_set"] = set(get_interval_indices(row["CREST (HEK293T)"], L))
        seq_data[gbk_name]["fwd_peak_idx"] = [
            set(get_interval_indices([interval], L)) for interval in (row["Puffin (FANTOM_CAGE_fwd)"] or [])
        ]
        seq_data[gbk_name]["rev_peak_idx"] = [
            set(get_interval_indices([interval], L)) for interval in (row["Puffin (FANTOM_CAGE_rev)"] or [])
        ]

    # 3. Iterate through elements to count overlaps
    overlap_records = []

    with h5py.File(PUFFIN_PREDS, "r") as h5f:
        puffin_feat_names = h5f.attrs["features"]
        puffin_fwd_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE fwd")[0, 0])
        puffin_rev_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE rev")[0, 0])

        N_ELEMENT_INSTANCES = instances_df.height
        update_every = 10_000
        pbar = tqdm(total=N_ELEMENT_INSTANCES, desc="Extracting CDS instance signals")

        puffin_cache = {}
        crest_cache = {}

        for row_idx, row in enumerate(instances_df.iter_rows(named=True)):
            gbk_name = row["gbk_name"] 
            sd = seq_data.get(gbk_name)
            if sd is None:
                continue

            L = sd["L"]
            element_intervals = row["position"]
            strand = row["strand"]
            e_type = row["element_type"]
            e_name = row["element_name"]
            unique_nuc_id = row["unique_nuc_id"]
            unique_aa_id = row["unique_aa_id"]

            # Map the Element Body and Flanks
            body_indices = []
            for s, e in element_intervals:
                if s <= e:
                    body_indices.extend(range(s, e))
                else:
                    body_indices.extend(range(s, L))
                    body_indices.extend(range(e))

            body_idx = np.array(body_indices) % L

            if len(body_idx) == 0:
                overlap_records.append({
                    "feature_type": e_type, "feature_name": e_name, 
                    "unique_nuc_id": unique_nuc_id, "unique_aa_id": unique_aa_id, 
                    "gbk_name": gbk_name, "position": element_intervals, "strand": strand,
                    "length": 0, "n_cre_midpoints": 0, "n_tss_midpoints": 0, 
                    "n_tss_fwd_midpoints": 0, "n_tss_rev_midpoints": 0,
                    "cre_avg_signal": np.nan, "tss_fwd_max_signal": np.nan, 
                    "tss_rev_max_signal": np.nan, "tss_max_signal": np.nan,
                    "fraction_cre_bp": 0.0
                })
                continue

            # True ends, not the stored order: for a minus-strand join that order puts
            # both TSS flanks at an internal junction, inside the body.
            genomic_start, genomic_end = circular_bounds(element_intervals, L)

            genomic_left = np.arange(genomic_start - TSS_FLANK_SIZE, genomic_start) % L
            genomic_right = np.arange(genomic_end, genomic_end + TSS_FLANK_SIZE) % L

            # Source Signals (Using Cache)
            if gbk_name not in crest_cache:
                crest_cache[gbk_name] = hek293t_tile_preds[np.array(crest_tiles_dict[gbk_name])]
            plasmid_crest = crest_cache[gbk_name]

            if gbk_name not in puffin_cache:
                puffin_cache[gbk_name] = h5f[gbk_name][:]
            puffin_preds = puffin_cache[gbk_name]

            puffin_fwd = puffin_preds[puffin_fwd_idx]
            puffin_rev = puffin_preds[puffin_rev_idx]

            # Strand-aware Signal, Midpoint, and Index Resolution
            if strand == -1:
                feat_downstream_idx = genomic_left
                feat_upstream_idx = genomic_right

                signal_fwd = puffin_rev
                signal_rev = puffin_fwd

                mids_fwd = sd["rev_mids_int"]
                mids_rev = sd["fwd_mids_int"]

                peaks_fwd = sd["rev_peak_idx"]
                peaks_rev = sd["fwd_peak_idx"]
            else:
                feat_downstream_idx = genomic_right
                feat_upstream_idx = genomic_left

                signal_fwd = puffin_fwd
                signal_rev = puffin_rev

                mids_fwd = sd["fwd_mids_int"]
                mids_rev = sd["rev_mids_int"]

                peaks_fwd = sd["fwd_peak_idx"]
                peaks_rev = sd["rev_peak_idx"]

            # Calculate Sets
            body_set = set(body_idx)
            fwd_set = body_set.union(feat_downstream_idx)
            rev_set = body_set.union(feat_upstream_idx)

            # Calculate Counts (Based on midpoints)
            n_cre = sum(1 for m in sd["cre_mids_int"] if m in body_set)
            tss_fwd_hits = sum(1 for m in mids_fwd if m in fwd_set)
            tss_rev_hits = sum(1 for m in mids_rev if m in rev_set)

            # Calculate Averages & Fractions
            cre_overlap_idx = np.array(list(sd["cre_idx_set"] & body_set), dtype=int)
            tss_fwd_overlap_idx = owned_peak_indices(peaks_fwd, mids_fwd, fwd_set)
            tss_rev_overlap_idx = owned_peak_indices(peaks_rev, mids_rev, rev_set)

            cre_avg_signal = safe_nanmean(plasmid_crest, cre_overlap_idx, metric="mean")
            # Peak, not mean, matching `08`: a TSS is a point event, so averaging over
            # every TSS-called bp in the element dilutes a sharp initiation site.
            tss_fwd_max_signal = safe_nanmean(signal_fwd, tss_fwd_overlap_idx, metric="max")
            tss_rev_max_signal = safe_nanmean(signal_rev, tss_rev_overlap_idx, metric="max")
            
            if np.isnan(tss_fwd_max_signal) and np.isnan(tss_rev_max_signal):
                tss_max_signal = np.nan
            else:
                tss_max_signal = float(np.nanmax([tss_fwd_max_signal, tss_rev_max_signal]))

            len_body = len(body_idx)
            frac_cre = len(cre_overlap_idx) / len_body if len_body > 0 else 0.0

            overlap_records.append({
                "feature_type": e_type,
                "feature_name": e_name,
                "unique_nuc_id": unique_nuc_id, 
                "unique_aa_id": unique_aa_id,
                "gbk_name": gbk_name,         
                "position": element_intervals,
                "strand": strand,
                "length": len_body,
                "n_cre_midpoints": n_cre,
                "n_tss_midpoints": tss_fwd_hits + tss_rev_hits,
                "n_tss_fwd_midpoints": tss_fwd_hits,
                "n_tss_rev_midpoints": tss_rev_hits,
                "cre_avg_signal": cre_avg_signal,
                "tss_fwd_max_signal": tss_fwd_max_signal,
                "tss_rev_max_signal": tss_rev_max_signal,
                "tss_max_signal": tss_max_signal,
                "fraction_cre_bp": frac_cre
            })

            if (row_idx + 1) % update_every == 0 or row_idx == N_ELEMENT_INSTANCES - 1:
                pbar.update(min(update_every, N_ELEMENT_INSTANCES - row_idx))
        pbar.close()

    # 4. Save the unaggregated instance dataset
    print("Saving output...")
    if overlap_records:
        res_df = pl.DataFrame(overlap_records)
        res_df.write_parquet(OUT_CDS_CRE_VARIABILITY)
        print(f"Saved dataset with {res_df.height} records")


if __name__ == "__main__":
    calculate_instance_overlap_statistics()  # 9 min

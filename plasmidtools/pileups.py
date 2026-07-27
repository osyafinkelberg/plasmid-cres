from pathlib import Path
import h5py
import numpy as np
import polars as pl

from .statplots import ELEMENT_TYPE_PRIORITIES


# METADATA EXTRACTOR
def get_matrix_metadata(
    plasmid_stats: pl.DataFrame, element_positions: pl.DataFrame,
    element_type: str, element_name: str
) -> pl.DataFrame:
    """
    Extracts the geometric bounds and metadata for instances of a specific feature.
    Separated to allow independent coordinate mapping without building the full type matrix.
    """
    target_instances = element_positions.filter(
        (pl.col("element_type") == element_type) & (pl.col("element_name") == element_name)
    )
    lengths_dict = dict(plasmid_stats.select(["sequence_id", "plasmid_length"]).iter_rows())

    matrix_metadata = []
    for row in target_instances.iter_rows(named=True):
        seq_id = row["sequence_id"]
        if seq_id not in lengths_dict:
            continue

        L = lengths_dict[seq_id]
        intervals = row["intervals"]
        strand = row.get("strand", 1)

        # Use max-gap logic to find true genomic bounds of multi-interval features
        parts = []
        for p_s, p_e in intervals:
            if p_s > p_e:
                parts.append((p_s, L))
                parts.append((0, p_e))
            else:
                parts.append((p_s, p_e))

        parts.sort(key=lambda x: x[0])
        merged_parts = [parts[0]]
        for p_s, p_e in parts[1:]:
            prev_s, prev_e = merged_parts[-1]
            if p_s <= prev_e:
                merged_parts[-1] = (prev_s, max(prev_e, p_e))
            else:
                merged_parts.append((p_s, p_e))

        max_gap, max_gap_idx = -1, -1
        n_merged = len(merged_parts)
        for i in range(n_merged):
            curr_e = merged_parts[i][1]
            next_s = merged_parts[(i + 1) % n_merged][0]
            gap = (next_s - curr_e) % L
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i

        genomic_end = merged_parts[max_gap_idx][1]
        genomic_start = merged_parts[(max_gap_idx + 1) % n_merged][0]

        # Determine true continuous span length of the element instance
        if genomic_start <= genomic_end:
            actual_len = genomic_end - genomic_start
        else:
            actual_len = (L - genomic_start) + genomic_end

        matrix_metadata.append({
            "sequence_id": seq_id,
            "plasmid_len": L,
            "instance_start": genomic_start,
            "instance_end": genomic_end,
            "actual_len": actual_len,
            "strand": strand
        })

    # Strict schema ensures downstream matrix functions don't crash if 0 targets found
    schema = {
        "sequence_id": pl.Int64, "plasmid_len": pl.Int64, "instance_start": pl.Int64, 
        "instance_end": pl.Int64, "actual_len": pl.Int64, "strand": pl.Int64
    }
    return pl.DataFrame(matrix_metadata, schema=schema) if matrix_metadata else pl.DataFrame(schema=schema)


# ARCHITECTURE MATRIX
def extract_element_type_matrix(
    plasmid_stats: pl.DataFrame, element_positions: pl.DataFrame,
    element_type: str, element_name: str, element_size: int, flank_size: int,
    matrix_metadata: pl.DataFrame = None
) -> tuple[np.ndarray, pl.DataFrame]:
    """
    Generates a localized, type-encoded matrix centered on instances of a specific feature subtype.
    """
    # 1. Fetch metadata if not provided
    if matrix_metadata is None:
        matrix_metadata = get_matrix_metadata(plasmid_stats, element_positions, element_type, element_name)

    total_window_width = (2 * flank_size) + element_size
    if matrix_metadata.height == 0:
        return np.zeros((0, total_window_width), dtype=np.int8), matrix_metadata

    # 2. Replicate category mapping and priorities
    unique_types = sorted(ELEMENT_TYPE_PRIORITIES.keys())
    type_to_idx = {t: i + 1 for i, t in enumerate(unique_types)}

    target_seq_ids = matrix_metadata["sequence_id"].unique().to_list()
    lengths_dict = dict(matrix_metadata.select(["sequence_id", "plasmid_len"]).unique().iter_rows())

    # Group all features on relevant plasmids to optimize downstream mask generation
    features_by_seq = (
        element_positions.filter(pl.col("sequence_id").is_in(target_seq_ids))
        .group_by("sequence_id")
        .agg([pl.col("element_type"), pl.col("intervals")])
    )

    # 3. Pre-compute painted base-masks for target plasmids
    plasmid_masks = {}
    for row in features_by_seq.iter_rows(named=True):
        seq_id = row["sequence_id"]
        p_len = lengths_dict[seq_id]

        base_mask = np.zeros(p_len, dtype=np.int8)
        priority_mask = np.zeros(p_len, dtype=np.int8)

        for t, intervals in zip(row["element_type"], row["intervals"]):
            val = type_to_idx.get(t, 0)
            if val == 0:
                continue

            prio = ELEMENT_TYPE_PRIORITIES.get(t, 2)

            def update_with_priority(s, e):
                chunk_prio = priority_mask[s:e]
                mask = prio > chunk_prio
                if np.any(mask):
                    base_mask[s:e][mask] = val
                    priority_mask[s:e][mask] = prio

            for start, end in intervals:
                if start < end:
                    update_with_priority(start, end)
                else:
                    update_with_priority(start, p_len)
                    update_with_priority(0, end)

        plasmid_masks[seq_id] = base_mask

    # 4. Extract aligned windows driven purely by the pre-computed metadata
    matrix_rows = []
    for meta in matrix_metadata.iter_rows(named=True):
        seq_id = meta["sequence_id"]
        # In rare cases where a plasmid has target sequences but no features in element_positions
        if seq_id not in plasmid_masks: 
            continue

        base_mask = plasmid_masks[seq_id]
        L = meta["plasmid_len"]
        start = meta["instance_start"]
        end = meta["instance_end"]
        actual_len = meta["actual_len"]
        strand = meta["strand"]

        # Vectorized coordinate mapping
        j_left = np.arange(flank_size)
        pos_left = (start - flank_size + j_left) % L

        j_element = np.arange(element_size)
        pos_element = (start + (j_element * actual_len) // element_size) % L

        j_right = np.arange(flank_size)
        pos_right = (end + j_right) % L

        combined_positions = np.concatenate([pos_left, pos_element, pos_right])
        window_row = base_mask[combined_positions]

        if strand == -1:
            window_row = window_row[::-1]

        matrix_rows.append(window_row)

    if matrix_rows:
        type_matrix = np.vstack(matrix_rows)
    else:
        type_matrix = np.zeros((0, total_window_width), dtype=np.int8)

    return type_matrix, matrix_metadata


# CRE / TSS MATRIX
def extract_regulatory_element_matrix(
    cre_positions: pl.DataFrame, cre_column: str,
    matrix_metadata: pl.DataFrame, element_size: int, flank_size: int, 
) -> np.ndarray:
    """
    Generates a binary matrix matching the layout of the element type matrix, 
    mapping CRE / TSS positional predictions directly onto the localized, 
    element-centered coordinate windows.
    """
    # 1. Map sequence_id to the chosen column's raw interval data for fast lookups
    cre_lookup = dict(cre_positions.select(["sequence_id", cre_column]).iter_rows())

    # Pre-extract unique plasmid lengths from the metadata
    seq_len_lookup = dict(matrix_metadata.select(["sequence_id", "plasmid_len"]).unique().iter_rows())

    # 2. OPTIMIZATION: Pre-compute the CRE masks per unique plasmid
    cre_masks = {}
    for seq_id, p_len in seq_len_lookup.items():
        cre_base_mask = np.zeros(p_len, dtype=np.int8)
        intervals = cre_lookup.get(seq_id, [])

        if intervals:  # Safe check in case intervals is None/empty
            for c_start, c_end in intervals:
                if c_start < c_end:
                    cre_base_mask[c_start:c_end] = 1
                else:  # Handle circular wrap-around boundaries for CRE markers
                    cre_base_mask[c_start:p_len] = 1
                    cre_base_mask[0:c_end] = 1
 
        cre_masks[seq_id] = cre_base_mask

    cre_rows = []
    total_window_width = (2 * flank_size) + element_size

    # 3. Iterate through metadata rows to mirror transformations 1:1
    for meta in matrix_metadata.iter_rows(named=True):
        seq_id = meta["sequence_id"]
        p_len = meta["plasmid_len"]
        start = meta["instance_start"]
        end = meta["instance_end"]
        actual_len = meta["actual_len"]
        strand = meta["strand"]

        # Fetch the pre-computed plasmid mask
        cre_base_mask = cre_masks[seq_id]

        # 4. Apply identical coordinate layout transformation
        j_left = np.arange(flank_size)
        pos_left = (start - flank_size + j_left) % p_len

        j_element = np.arange(element_size)
        pos_element = (start + (j_element * actual_len) // element_size) % p_len

        j_right = np.arange(flank_size)
        pos_right = (end + j_right) % p_len

        combined_positions = np.concatenate([pos_left, pos_element, pos_right])
        window_row = cre_base_mask[combined_positions]

        # 5. Flip coordinate mapping space if on the negative strand
        if strand == -1:
            window_row = window_row[::-1]

        cre_rows.append(window_row)

    if cre_rows:
        return np.vstack(cre_rows)
    else:
        return np.zeros((0, total_window_width), dtype=np.int8)


# SIGNALS PILEUP
def extract_aligned_element_predictions(
    element_positions_path: Path, crest_tile_encoding_path: Path,
    crest_tile_preds_path: Path, puffin_preds_path: Path,
    element_type: str, element_name: str, flank_size: int,
    cell_names: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extracts and aligns CREST and Puffin predictions for a specific element.

    Args:
        element_positions_path: Path to the element positions file.
        crest_tile_encoding_path: Path to the CREST tile encoding file.
        crest_tile_preds_path: Path to the CREST tile predictions file.
        puffin_preds_path: Path to the Puffin predictions file.
        element_type: The type of the element (e.g., 'promoter', 'putative_orf').
        element_name: The specific name of the element.
        flank_size: Number of base pairs to include upstream and downstream.
        cell_names: List of cell types to extract CREST predictions for.

    Returns:
        Tuple of (cre_matrices, fwd_tss_matrix, rev_tss_matrix). 
        - cre_matrices is a dictionary mapping cell names to arrays of shape (N_instances, 2 * flank_size + median_element_length).
        - fwd_tss_matrix and rev_tss_matrix are arrays of the same shape.
    """
    # 1. Load Element Metadata
    plasmid_elements = pl.read_parquet(element_positions_path)
    element_df = plasmid_elements.filter(
        (pl.col("element_type") == element_type) & 
        (pl.col("element_name") == element_name)
    )

    n_instances = element_df.height
    if n_instances == 0:
        raise ValueError(f"No instances found for {element_type}: {element_name}")

    median_len = int(element_df["length"].median())
    aligned_len = 2 * flank_size + median_len

    # 2. Pre-load CREST global arrays for requested cells
    tile_encoding = pl.read_parquet(crest_tile_encoding_path)
    cre_predictions = pl.read_parquet(crest_tile_preds_path)
    n_crest_tiles = tile_encoding["tile_ids"].explode().max() + 1
    
    cell_tile_preds = {}
    for cell in cell_names:
        preds = np.full(n_crest_tiles, np.nan)
        preds[cre_predictions["tile_ID"].to_numpy()] = cre_predictions[cell].to_numpy()
        cell_tile_preds[cell] = preds

    # 3. Initialize Output Matrices
    cre_matrices = {cell: np.zeros((n_instances, aligned_len), dtype=np.float32) for cell in cell_names}
    fwd_matrix = np.zeros((n_instances, aligned_len), dtype=np.float32)
    rev_matrix = np.zeros((n_instances, aligned_len), dtype=np.float32)

    # 4. Iterate and Extract Predictions
    with h5py.File(puffin_preds_path, "r") as h5f:
        puffin_feat_names = h5f.attrs["features"]
        puffin_fwd_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE fwd")[0, 0])
        puffin_rev_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE rev")[0, 0])

        for row_idx, row in enumerate(element_df.iter_rows(named=True)):
            gbk_name = row["gbk_name"]
            strand = row["strand"]
            intervals = row["intervals"]

            # Puffin
            puffin_preds = h5f[gbk_name][:]
            puffin_fwd = puffin_preds[puffin_fwd_idx]
            puffin_rev = puffin_preds[puffin_rev_idx]

            L = len(puffin_fwd)

            # --- Reconstruct Genomic Indices ---
            body_indices = []
            for s, e in intervals:
                if s < e:
                    body_indices.extend(range(s, e))
                else:  # Wrap around origin
                    body_indices.extend(range(s, L))
                    body_indices.extend(range(0, e))

            genomic_start = body_indices[0]
            genomic_end = (body_indices[-1] + 1) % L

            # Flank indices mapped to circular plasmid
            left_idx = np.arange(genomic_start - flank_size, genomic_start) % L
            right_idx = np.arange(genomic_end, genomic_end + flank_size) % L
            body_idx = np.array(body_indices)

            # --- Interpolation & Assembly Helper ---
            def assemble_aligned_signal(signal_array: np.ndarray) -> np.ndarray:
                left_flank = signal_array[left_idx]
                right_flank = signal_array[right_idx]
                body_raw = signal_array[body_idx]

                # Interpolate body to median length
                x_old = np.linspace(0, 1, len(body_raw))
                x_new = np.linspace(0, 1, median_len)
                body_interp = np.interp(x_new, x_old, body_raw)

                return np.concatenate([left_flank, body_interp, right_flank])

            # Process signals linearly
            fwd_genomic = assemble_aligned_signal(puffin_fwd)
            rev_genomic = assemble_aligned_signal(puffin_rev)

            # CREST for all cells
            crest_tiles = tile_encoding.filter(pl.col("gbk_name") == gbk_name)["tile_ids"].to_list()[0]
            crest_tiles_arr = np.array(crest_tiles)

            cre_genomics = {}
            for cell in cell_names:
                plasmid_crest = cell_tile_preds[cell][crest_tiles_arr]
                cre_genomics[cell] = assemble_aligned_signal(plasmid_crest)

            # orientation correction
            if strand == -1:
                for cell in cell_names:
                    cre_matrices[cell][row_idx] = cre_genomics[cell][::-1]
                # swap forward / reverse TSS biologically for negative strand elements
                fwd_matrix[row_idx] = rev_genomic[::-1]
                rev_matrix[row_idx] = fwd_genomic[::-1]
            else:
                for cell in cell_names:
                    cre_matrices[cell][row_idx] = cre_genomics[cell]
                fwd_matrix[row_idx] = fwd_genomic
                rev_matrix[row_idx] = rev_genomic

    return cre_matrices, fwd_matrix, rev_matrix

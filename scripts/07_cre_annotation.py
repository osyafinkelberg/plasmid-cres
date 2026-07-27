import sys
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import polars as pl
import h5py
from tqdm import tqdm

sys.path.insert(0, "..")
from plasmidtools import crest


# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"
CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
CRE_ANNOTATION_OUT = ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet"

DENSITY_THRESH = 0.2
CRE_PEAK_RADIUS = 25
TSS_PEAK_RADIUS = 5

CREST_THRESHOLDS = pl.read_csv(Path().cwd().parent / "../mpra-predictor/data/cre_thresholds_fdr_001.csv")
CREST_CELLS = crest.CREST_LABELS
PUFFIN_CAGE_THRESH = 0.1  # FANTOM CAGE (arbitrary)
CREST_CELL_THRESHOLDS = {
    cell: CREST_THRESHOLDS.filter(pl.col("cell") == cell)["threshold"][0] * 1.15  # strict thresholding
    for cell in CREST_CELLS
}


class CREPeakCaller:
    def __init__(self, summit_threshold: float, density_threshold: float, peak_radius: int):
        self.summit_threshold = summit_threshold
        self.density_threshold = density_threshold
        self.peak_radius = peak_radius

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        L = len(scores)
        if L == 0:
            return np.array([], dtype=int).reshape(0, 2)

        # np.nan values interpreted as 'not active'
        scores = np.nan_to_num(scores, nan=0.0)

        # circular boundary conditions: artificial extension
        flank = self.peak_radius * 2

        # handle edge cases where flank is larger than the plasmid itself
        if flank > L:
            repeats = int(np.ceil(flank / L))
            left_pad = np.tile(scores, repeats)[-flank:]
            right_pad = np.tile(scores, repeats)[:flank]
        else:
            left_pad = scores[-flank:]
            right_pad = scores[:flank]

        padded_scores = np.concatenate([left_pad, scores, right_pad])

        increasing = np.pad(padded_scores[1:] - padded_scores[:-1] >= 0, (1, 0), mode='constant')
        decreasing = np.pad(padded_scores[1:] - padded_scores[:-1] <= 0, (0, 1), mode='constant')
        local_max_mask = increasing & decreasing
        threshold_mask = padded_scores > self.summit_threshold
        mask_padded = np.pad(threshold_mask, (self.peak_radius - 1, self.peak_radius), mode='constant')
        smooth_mask = sliding_window_view(
            mask_padded, window_shape=2 * self.peak_radius, axis=0
        ).mean(1) >= self.density_threshold

        summits_padded = np.argwhere(local_max_mask & smooth_mask).flatten()

        # Extract summits strictly within the primary sequence window [flank, flank + L)
        valid_summits = summits_padded[(summits_padded >= flank) & (summits_padded < flank + L)] - flank

        if len(valid_summits) == 0:
            return np.array([], dtype=int).reshape(0, 2)

        # Project intervals onto a circular boolean mask to seamlessly merge overlaps
        active = np.zeros(L, dtype=bool)
        for s in valid_summits:
            indices = np.arange(s - self.peak_radius, s + self.peak_radius) % L
            active[indices] = True

        # If the entire plasmid is one giant active region
        if active.all():
            return np.array([[0, L]], dtype=int)

        # Extract contiguous active segments
        pad_active = np.concatenate([[False], active, [False]])
        diff = np.diff(pad_active.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Merge boundary-crossing peak if the sequence wraps around the origin
        if active[0] and active[-1]:
            ends[-1] = ends[0]
            starts = starts[1:]
            ends = ends[1:]

        return np.column_stack((starts, ends))


def annotate_cre_and_tss() -> None:

    # Load CREST predictions
    tile_encoding = pl.read_parquet(CREST_TILE_ENCOD)
    cre_predictions = pl.read_parquet(CREST_TILE_PREDS)
    n_crest_tiles = tile_encoding["tile_ids"].list.max().max() + 1

    # Setup tile predictions arrays for all cell types
    crest_tile_preds_all = {}
    for cell in CREST_CELLS:
        preds = np.full(n_crest_tiles, np.nan)
        preds[cre_predictions["tile_ID"].to_numpy()] = cre_predictions[cell].to_numpy()
        crest_tile_preds_all[cell] = preds

    plasmids = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_statistics.parquet")[["gbk_name", "sequence_id", "plasmid_length"]]

    # Setup PeakCallers for all cell types
    crest_peak_callers = {
        cell: CREPeakCaller(CREST_CELL_THRESHOLDS[cell], DENSITY_THRESH, CRE_PEAK_RADIUS)
        for cell in CREST_CELLS
    }
    puffin_peak_caller = CREPeakCaller(PUFFIN_CAGE_THRESH, DENSITY_THRESH, TSS_PEAK_RADIUS)

    results = []
    update_every = 1000
    N_PLASMIDS = plasmids.height

    with h5py.File(PUFFIN_PREDS, "r") as h5f:
        puffin_feat_names = h5f.attrs["features"]
        puffin_fwd_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE fwd")[0, 0])
        puffin_rev_idx = int(np.argwhere(puffin_feat_names == "FANTOM_CAGE rev")[0, 0])

        pbar = tqdm(total=N_PLASMIDS, desc="Annotate CREs / TSSs")
        for plasmid_idx, (gbk_name, sequence_id, plasmid_length) in enumerate(plasmids.rows()):

            # CREST CREs for all cells
            crest_tile_ids = tile_encoding.filter(pl.col("gbk_name") == gbk_name)["tile_ids"][0].to_numpy()

            crest_results = {}
            for i, cell in enumerate(CREST_CELLS):
                plasmid_crest = crest_tile_preds_all[cell][crest_tile_ids]
                if i == 0:  # Only need to assert shape once per plasmid
                    assert plasmid_crest.shape[0] == plasmid_length

                crest_cres = crest_peak_callers[cell](plasmid_crest)
                crest_results[f"CREST ({cell})"] = crest_cres.tolist()

            # Puffin TSSs
            puffin_preds = h5f[gbk_name][:]
            assert puffin_preds.shape[1] == plasmid_length
            puffin_fwd_tss = puffin_peak_caller(puffin_preds[puffin_fwd_idx])
            puffin_rev_tss = puffin_peak_caller(puffin_preds[puffin_rev_idx])

            # Save CREs / TSSs positions dynamically unpacking all cells
            base_results = {
                "gbk_name": gbk_name,
                "sequence_id": sequence_id,
                "Puffin (FANTOM_CAGE_fwd)": puffin_fwd_tss.tolist(),
                "Puffin (FANTOM_CAGE_rev)": puffin_rev_tss.tolist(),
            }
            results.append({**base_results, **crest_results})

            if (plasmid_idx + 1) % update_every == 0 or plasmid_idx == N_PLASMIDS - 1:
                pbar.update(update_every)

        pbar.close()
        
    pl.DataFrame(results).write_parquet(CRE_ANNOTATION_OUT)


if __name__ == "__main__":
    annotate_cre_and_tss()  # 30 min

import sys
from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, "..")
from plasmidtools import crest, pileups


# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
ADDGENE_DIR = DATA_DIR / "addgene"
MANUAL_DIR = DATA_DIR / "manual_annotations"

ELEMENT_POSITIONS = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"
PROMOTER_ACTIVITY_OUT = ADDGENE_DIR / "promoter_activity_predictions.csv"

CREST_CELLS = crest.CREST_LABELS
TSS_FLANK_SIZE = 50


def derive_element_cre_activity(cre_matrix: np.ndarray, pred_matrix: np.ndarray, flank_size: int) -> float:
    cre_slice = cre_matrix[:, flank_size: -flank_size]
    pred_slice = pred_matrix[:, flank_size: -flank_size]
    mask = (cre_slice == 1) & ~np.isnan(pred_slice)
    row_sums = np.sum(np.where(mask, pred_slice, 0), axis=1)
    # row_counts = np.sum(mask, axis=1)
    # row_means = np.divide(
    #     row_sums, 
    #     row_counts, 
    #     out=np.full(row_sums.shape, np.nan, dtype=float), 
    #     where=(row_counts > 0)
    # )
    # return np.nanmedian(row_means)
    return np.nanmedian(row_sums)  # separates promoters better, than average per bp activity


def calculate_crest_cell_specific_promoter_activity() -> None:

    strength_df = []
    for element_type, element_name in tqdm(promoters["element_type", "element_name"].rows()):
        element_size = element_lengths.filter(
            (pl.col("element_type") == element_type) & (pl.col("element_name") == element_name)
        )["element_length"][0]

        matrix_metadata = pileups.get_matrix_metadata(plasmid_stats, element_positions, element_type, element_name)

        crest_matrix_dct, puff_fwd_matrix, puff_rev_matrix = pileups.extract_aligned_element_predictions(
            ELEMENT_POSITIONS, CREST_TILE_ENCOD, CREST_TILE_PREDS, PUFFIN_PREDS,
            element_type, element_name, TSS_FLANK_SIZE, cell_names=CREST_CELLS
        )

        row = {
            "element_type": element_type,
            "element_name": element_name,
            "Puffin (FANTOM_CAGE_fwd)": np.median(puff_fwd_matrix[:, TSS_FLANK_SIZE:].max(1)),
            "Puffin (FANTOM_CAGE_rev)": np.median(puff_rev_matrix[:, :-TSS_FLANK_SIZE].max(1)),
        }

        for cell in CREST_CELLS:
            cre_matrix = pileups.extract_regulatory_element_matrix(
                cre_positions, f"CREST ({cell})", matrix_metadata, element_size, TSS_FLANK_SIZE
            )
            row[f"CREST ({cell})"] = derive_element_cre_activity(cre_matrix, crest_matrix_dct[cell], TSS_FLANK_SIZE)

        strength_df.append(row)

    strength_df = pl.DataFrame(strength_df)
    strength_df.write_csv(PROMOTER_ACTIVITY_OUT)


if __name__ == "__main__":

    # --- 1. Data ---
    element_lengths = (
        pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_elements.parquet")
        .group_by(["element_type", "element_name"])
        .agg(pl.col("length").median().cast(pl.Int64))
        .rename({"length": "element_length"})
    )
    plasmid_stats = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_statistics.parquet")
    element_positions = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_elements.parquet")
    cre_positions = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet")
    promoters = pl.read_csv(MANUAL_DIR / "addgene_promoters_and_enhancers.csv")

    # --- 2. Aggregated CRE activity ---
    calculate_crest_cell_specific_promoter_activity()  # 85 min

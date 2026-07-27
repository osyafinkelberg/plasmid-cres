import sys
from pathlib import Path
import shutil
import polars as pl
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from plasmidtools import helpers, statplots, pileups


# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
PILEUP_FIGURES = FIGURES_DIR / "pileups"
ADDGENE_DIR = DATA_DIR / "addgene"
MANUAL_DIR = DATA_DIR / "manual_annotations"
ELEMENT_POSITIONS = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"

PILEUP_PATH = ADDGENE_DIR / "plasmid_elements_prediction_pileups.h5"
PILEUP_FIGURES.mkdir(parents=True, exist_ok=True)
FLANK_SIZE = 500


def process_element(
    element_type: str, element_name: str, write_to_file: bool, visualize: bool
) -> None:
    element_size = element_lengths.filter((pl.col("element_type") == element_type) & (pl.col("element_name") == element_name))["element_length"][0]

    # Check early to save CPU / IO overhead
    if helpers.has_sufficient_flank(element_type, element_name, FLANK_SIZE, PILEUP_PATH):
        print(f"Skipping pipeline for {element_type}/{element_name}: Already processed with sufficient flank.")

        # LOAD MATRICES
        _, _, type_matrix, cre_matrix, tss_fwd_matrix, tss_rev_matrix, crest_matrix, puff_fwd_matrix, puff_rev_matrix = helpers.load_aligned_predictions_h5(
            element_type, element_name, PILEUP_PATH
        )

    else:  # --- Pile-up Matrix Generation ---
        print(f"Extracting profiles for {element_type}/{element_name}...")
        type_matrix, matrix_metadata = pileups.extract_element_type_matrix(
            plasmid_stats, element_positions,
            element_type, element_name, element_size, FLANK_SIZE
        )
        cre_matrix = pileups.extract_regulatory_element_matrix(cre_positions, "CREST (HEK293T)", matrix_metadata, element_size, FLANK_SIZE)
        tss_fwd_matrix = pileups.extract_regulatory_element_matrix(cre_positions, "Puffin (FANTOM_CAGE_fwd)", matrix_metadata, element_size, FLANK_SIZE)
        tss_rev_matrix = pileups.extract_regulatory_element_matrix(cre_positions, "Puffin (FANTOM_CAGE_rev)", matrix_metadata, element_size, FLANK_SIZE)

        crest_matrix_dct, puff_fwd_matrix, puff_rev_matrix = pileups.extract_aligned_element_predictions(
            ELEMENT_POSITIONS, CREST_TILE_ENCOD, CREST_TILE_PREDS, PUFFIN_PREDS,
            element_type, element_name, FLANK_SIZE, cell_names=["HEK293T"]
        )
        crest_matrix = crest_matrix_dct["HEK293T"]

        if write_to_file:  # SAVE MATRICES
            helpers.save_aligned_predictions_h5(
                element_type, element_name, element_size, FLANK_SIZE,
                type_matrix, cre_matrix, tss_fwd_matrix, tss_rev_matrix,
                crest_matrix, puff_fwd_matrix, puff_rev_matrix,
                PILEUP_PATH
            )

    # --- 3. Visualization ---
    if visualize:
        filestem = f"{helpers.sanitize_filename(element_type)}__{helpers.sanitize_filename(element_name)}"
        temp_dir = PILEUP_FIGURES / f"{filestem}_tmp"
        temp_dir.mkdir(exist_ok=True)

        fig, ax = statplots.element_centered_architecture_heatmap(
            type_matrix=type_matrix[:10000],
            flank_size=FLANK_SIZE,
            element_size=element_size,
            cre_matrix=cre_matrix,
            cre_label="CREST (HEK293T)",
            target_label=f"{element_type} - {element_name}"
        )
        fig.savefig(temp_dir / "01.jpg", dpi=300, format='jpg')
        plt.close(fig)

        fig, ax = statplots.element_centered_architecture_heatmap(
            type_matrix=type_matrix[:10000],
            flank_size=FLANK_SIZE,
            element_size=element_size,
            cre_matrix=tss_fwd_matrix,
            cre_label="Puffin (FANTOM_CAGE_fwd)",
            target_label=f"{element_type} - {element_name}"
        )
        fig.savefig(temp_dir / "02.jpg", dpi=300, format='jpg')
        plt.close(fig)

        fig, ax = statplots.element_centered_architecture_heatmap(
            type_matrix=type_matrix[:10000],
            flank_size=FLANK_SIZE,
            element_size=element_size,
            cre_matrix=tss_rev_matrix,
            cre_label="Puffin (FANTOM_CAGE_rev)",
            target_label=f"{element_type} - {element_name}"
        )
        fig.savefig(temp_dir / "03.jpg", dpi=300, format='jpg')
        plt.close(fig)

        fig, (ax1, ax2) = statplots.combined_prediction_pileups(
            crest_matrix, puff_fwd_matrix, puff_rev_matrix, element_type, element_name, FLANK_SIZE
        )
        fig.savefig(temp_dir / "04.jpg", dpi=300, format='jpg')
        plt.close(fig)

        helpers.convert_jpgs_to_pdf(temp_dir, PILEUP_FIGURES / f"{filestem}.pdf")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":

    # --- 1. Data ---
    plasmid_stats = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_statistics.parquet")
    element_positions = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_elements.parquet")
    element_lengths = (
        pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_elements.parquet")
        .group_by(["element_type", "element_name"])
        .agg(pl.col("length").median().cast(pl.Int64))
        .rename({"length": "element_length"})
    )
    cre_positions = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet")

    # --- 2. Processing Custom Elements ---
    for element_type, element_name, write_to_file, visualize in [
        ["repeat_region", "ITR", True, False],
        ["rep_origin", "ori", True, False],
        ["misc_feature", "Rosa26 left arm", True, False]
    ]:
        process_element(element_type, element_name, write_to_file, visualize)

    # --- 3. Processing Addgene Promoters ---
    promoters = pl.read_csv(MANUAL_DIR / "addgene_promoters_and_enhancers.csv")
    for (element_type, element_name) in promoters[["element_type", "element_name"]].rows():
        process_element(element_type, element_name, True, False)

    # # --- 4. Processing Cryptic CREs ---
    # df_clustered = pl.read_csv(ADDGENE_DIR / "element_cre_overlap_clustering.csv")
    # cryptic_elements = (
    #     df_clustered
    #     .filter(df_clustered["is_cryptic_cre"] & ~df_clustered["type"].is_in(["backbone_spacer", "putative_orf", "putative_noncoding"]))
    #     .select(["type", "name"])
    #     .rows()
    # )
    # for (element_type, element_name) in cryptic_elements:  # ~1.5 hours
    #     process_element(element_type, element_name, False, True)

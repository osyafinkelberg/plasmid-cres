import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, "..")
from plasmidtools import helpers, pileups, statplots

# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
PILEUP_FIGURES = FIGURES_DIR / "pileups"
ADDGENE_DIR = DATA_DIR / "addgene"
MANUAL_DIR = DATA_DIR / "manual_annotations"
ELEMENT_POSITIONS = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
PLASMID_STATS = ADDGENE_DIR / "mammalian_plasmids_statistics.parquet"
CRE_TSS_FILE = ADDGENE_DIR / "mammalian_plasmids_cre_and_tss.parquet"
CLUSTERING_FILE = ADDGENE_DIR / "element_cre_overlap_clustering.csv"
CREST_TILE_ENCOD = ADDGENE_DIR / "mammalian_plasmids_crest_encodings.parquet"
CREST_TILE_PREDS = ADDGENE_DIR / "mammalian_plasmids_crest_preds.parquet"
PUFFIN_PREDS = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"

PILEUP_PATH = ADDGENE_DIR / "plasmid_elements_prediction_pileups.h5"
PILEUP_FIGURES.mkdir(parents=True, exist_ok=True)
FLANK_SIZE = 500

# CREST cell line the pile-ups are drawn for; notebook 03 aliases the same one.
CRE_CELL = statplots.DEFAULT_CRE_CELL


def outdated_stored_elements(element_positions: pl.DataFrame) -> list[tuple[str, str]]:
    """Stored elements whose pile-up groups predate `helpers.PILEUP_FORMAT`.

    Group paths hold sanitized names, so they are matched back against the element
    table rather than un-sanitized.
    """
    if not PILEUP_PATH.exists():
        return []

    with h5py.File(PILEUP_PATH, "r") as h5f:
        outdated = {
            f"{group_type}/{group_name}"
            for group_type in h5f
            for group_name in h5f[group_type]
            if int(h5f[group_type][group_name].attrs.get("pileup_format", 1)) != helpers.PILEUP_FORMAT
        }

    element_keys = element_positions.select(["element_type", "element_name"]).unique().sort(["element_type", "element_name"])
    return [
        (element_type, element_name)
        for element_type, element_name in element_keys.rows()
        if f"{helpers.sanitize_filename(element_type)}/{helpers.sanitize_filename(element_name)}" in outdated
    ]


def process_element(
    element_type: str,
    element_name: str,
    write_to_file: bool,
    visualize: bool,
    plasmid_stats: pl.DataFrame,
    element_positions: pl.DataFrame,
    element_lengths: pl.DataFrame,
    cre_positions: pl.DataFrame,
    tracks: dict,
) -> None:
    # Check early to save CPU / IO overhead
    if helpers.has_sufficient_flank(element_type, element_name, FLANK_SIZE, PILEUP_PATH):
        print(f"Loading {element_type}/{element_name}: already stored with sufficient flank.")

        # LOAD MATRICES - with the stored geometry rather than FLANK_SIZE, since a
        # group kept for its larger flank would otherwise be drawn with wrong bounds.
        (
            element_size, flank_size,
            type_matrix, cre_matrix, tss_fwd_matrix, tss_rev_matrix,
            crest_matrix, puff_fwd_matrix, puff_rev_matrix,
        ) = helpers.load_aligned_predictions_h5(element_type, element_name, PILEUP_PATH)

    else:  # --- Pile-up Matrix Generation ---
        print(f"Extracting profiles for {element_type}/{element_name}...")
        element_size = element_lengths.filter(
            (pl.col("element_type") == element_type) & (pl.col("element_name") == element_name)
        )["element_length"][0]
        flank_size = FLANK_SIZE

        type_matrix, matrix_metadata = pileups.extract_element_type_matrix(
            plasmid_stats, element_positions,
            element_type, element_name, element_size, flank_size
        )
        cre_matrix = pileups.extract_regulatory_element_matrix(
            cre_positions, f"CREST ({CRE_CELL})", matrix_metadata, element_size, flank_size
        )
        tss_fwd_matrix, tss_rev_matrix = pileups.extract_tss_matrices(
            cre_positions, matrix_metadata, element_size, flank_size
        )
        crest_matrix_dct, puff_fwd_matrix, puff_rev_matrix = pileups.extract_aligned_element_predictions(
            matrix_metadata, element_size, flank_size, cell_names=[CRE_CELL], tracks=tracks
        )
        crest_matrix = crest_matrix_dct[CRE_CELL]

        if write_to_file:  # SAVE MATRICES
            helpers.save_aligned_predictions_h5(
                element_type, element_name, element_size, flank_size,
                type_matrix, cre_matrix, tss_fwd_matrix, tss_rev_matrix,
                crest_matrix, puff_fwd_matrix, puff_rev_matrix,
                PILEUP_PATH
            )

    # --- Visualization ---
    if visualize:
        filestem = f"{helpers.sanitize_filename(element_type)}__{helpers.sanitize_filename(element_name)}"
        target_label = f"{element_type} - {element_name}"

        # All three architecture pages overlay the same rows, so cluster them once.
        row_order, row_clusters = statplots.architecture_row_order(
            type_matrix[statplots.architecture_row_sample(type_matrix.shape[0])]
        )
        overlays = [
            (cre_matrix, f"CREST ({CRE_CELL})"),
            (tss_fwd_matrix, "Puffin CAGE, element strand"),
            (tss_rev_matrix, "Puffin CAGE, opposite strand"),
        ]

        # One multi-page vector PDF: no temporary JPGs to stitch, or to leak on a crash.
        with PdfPages(PILEUP_FIGURES / f"{filestem}.pdf") as pdf:
            for overlay, overlay_label in overlays:
                fig, _ = statplots.element_centered_architecture_heatmap(
                    type_matrix=type_matrix,
                    flank_size=flank_size,
                    element_size=element_size,
                    cre_matrix=overlay,
                    cre_label=overlay_label,
                    target_label=target_label,
                    row_order=row_order,
                    row_clusters=row_clusters,
                )
                pdf.savefig(fig, dpi=300, bbox_inches="tight")
                plt.close(fig)

            fig, _ = statplots.combined_prediction_pileups(
                crest_matrix, puff_fwd_matrix, puff_rev_matrix, element_type, element_name, flank_size,
                cre_label=f"CREST ({CRE_CELL})",
            )
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":

    # --- 1. Data ---
    plasmid_stats = pl.read_parquet(PLASMID_STATS)
    element_positions = pl.read_parquet(ELEMENT_POSITIONS)
    element_lengths = (
        element_positions
        .group_by(["element_type", "element_name"])
        .agg(pl.col("length").median().cast(pl.Int64))
        .rename({"length": "element_length"})
    )
    cre_positions = pl.read_parquet(CRE_TSS_FILE)

    # Loaded once for the whole run: per element they cost ~10 s to rebuild, plus a
    # ~50 ms random read from the Puffin file for every instance.
    tracks = pileups.load_prediction_tracks(
        CREST_TILE_ENCOD, CREST_TILE_PREDS, PUFFIN_PREDS, cell_names=[CRE_CELL]
    )
    data = {
        "plasmid_stats": plasmid_stats,
        "element_positions": element_positions,
        "element_lengths": element_lengths,
        "cre_positions": cre_positions,
        "tracks": tracks,
    }

    # # --- 2. Processing Custom Elements ---
    # for element_type, element_name, write_to_file, visualize in [
    #     ["rep_origin", "RSF ori", True, False],
    #     ["LTR", "3' LTR", True, False],
    #     ["misc_feature", "Rosa26 left arm", True, False],
    #     ["mobile_element", "IS1", True, False],
    #     ["rep_origin", "SV40 ori", True, False],
    #     ["rep_origin", "p15A ori", True, False],
    #     ["misc_signal", "Ad5 Psi", True, False],
    #     ["repeat_region", "ITR", True, False],
    #     ["rep_origin", "ori", True, False],
    #     ["intron", "chimeric intron", True, False],
    #     ["CDS", "ABE(7.10)", True, False]
    # ]:
    #     process_element(element_type, element_name, write_to_file, visualize, **data)

    # # --- 3. Processing Addgene Promoters ---
    # promoters = pl.read_csv(MANUAL_DIR / "addgene_promoters_and_enhancers.csv")
    # for (element_type, element_name) in promoters[["element_type", "element_name"]].rows():
    #     process_element(element_type, element_name, True, False, **data)

    # --- 4. Refreshing Outdated Stored Pile-ups ---
    # Groups written before `helpers.PILEUP_FORMAT` took their signal tracks from the
    # stored interval order and their TSS masks from the plasmid strand. Rebuild all
    # of them, so notebooks 02 and 03 never load one; matrices only, no figures.
    for element_type, element_name in outdated_stored_elements(element_positions):
        process_element(element_type, element_name, True, False, **data)

    # --- 5. Processing Active Components ---
    # Same selection as the heatmap view in `03_component_cres.ipynb`: typed elements
    # (length >= 100) that are not inactive and are not themselves annotated
    # mammalian Pol II promoters / enhancers.
    df_clustered = pl.read_csv(CLUSTERING_FILE)
    ACTIVE_FILTER = (df_clustered["cluster_label"] != "inactive") & (~df_clustered["is_reference_element"])
    for element_type, element_name in df_clustered.filter(ACTIVE_FILTER).select(["type", "name"]).rows():
        process_element(element_type, element_name, True, True, **data)

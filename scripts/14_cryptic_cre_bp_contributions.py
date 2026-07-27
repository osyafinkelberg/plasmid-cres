import sys
from pathlib import Path
import shutil
import numpy as np
import polars as pl
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
import plasmidtools

sys.path.insert(0, "../data/puffin")
import puffin


# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
ADDGENE_DIR = DATA_DIR / "addgene"
CONTRIBS_OUTPUT = ADDGENE_DIR / "CCCs_contrib_scores.h5"
CONTRIB_FIGURES = FIGURES_DIR / "contribs"
CONTRIB_FIGURES.mkdir(parents=True, exist_ok=True)

CREST_INDEX = int(np.argwhere(plasmidtools.crest.CREST_LABELS == "HEK293T")[0, 0])
PUFFIN_FLANK = 325


def join_contribution_tiles(
    onehot_tiles: np.ndarray,
    contrib_tiles: np.ndarray,
    valid_idxs: np.ndarray,
    L: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs full-length arrays from sliding windows, gracefully handling
    missing/invalid tile indices by using an overlap-average accumulator.
    """
    assert onehot_tiles.shape == contrib_tiles.shape
    assert len(onehot_tiles.shape) == 3
    assert onehot_tiles.shape[1] == 4

    tile_size = onehot_tiles.shape[-1]

    # initialize full-length target matrix grids
    accum_contrib = np.zeros((4, L), dtype=np.float32)
    realigned_onehot = np.zeros((4, L), dtype=np.float32)
    coverage_counts = np.zeros(L, dtype=np.int32)

    # place each valid window into its absolute genomic coordinate position
    for k, start_pos in enumerate(valid_idxs):
        end_pos = start_pos + tile_size

        accum_contrib[:, start_pos:end_pos] += contrib_tiles[k]
        realigned_onehot[:, start_pos:end_pos] = onehot_tiles[k]  # overwriting matches identical base identities
        coverage_counts[start_pos:end_pos] += 1

    # compute mean contribution per base, safely guarding against zero-coverage regions
    safe_counts = np.where(coverage_counts == 0, 1, coverage_counts)
    realigned_contrib = accum_contrib / safe_counts

    return realigned_onehot, realigned_contrib


def reverse_complement(dna_sequence: str) -> str:
    tab = str.maketrans("ATCGatcg", "TAGCtagc")
    return dna_sequence.translate(tab)[::-1]


def obtain_crest_contribution_scores(element_sequence: str, crest_index: int) -> tuple[np.ndarray, np.ndarray]:
    L = len(element_sequence)
    crest_tiles = [element_sequence[i : i + 200] for i in range(L - 199)]

    interpreter = plasmidtools.crest.CRESTInterpreter(batch_size=plasmidtools.crest.BATCH_SIZE, pred_index=crest_index)
    for idx, tile in enumerate(crest_tiles):
        interpreter.update(f"{idx}", tile)

    valid_idxs, onehots, contribs = interpreter.get_predictions()
    valid_idxs = valid_idxs.astype(np.int32)

    # pass total sequence length L to build the reconstructed shape accurately
    realigned_onehots, realigned_contribs = join_contribution_tiles(onehots, contribs, valid_idxs, L)
    return realigned_onehots, realigned_contribs


def obtain_puffin_contribution_scores(puffin_model: puffin.Puffin, element_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    puff_fwd_interp = {key: np.array(val) for key, val in puffin_model.interpret(element_sequence, targeti="FANTOM_CAGE").T.to_dict('list').items()}
    fwd_contribs = puff_fwd_interp["Basepair contribution score to transcription initiation"]
    puff_rev_interp = {key: np.array(val) for key, val in puffin_model.interpret(reverse_complement(element_sequence), targeti="FANTOM_CAGE").iloc[:, ::-1].T.to_dict('list').items()}
    rev_contribs = puff_rev_interp["Basepair contribution score to transcription initiation"]
    return fwd_contribs, rev_contribs


def calculate_cryptic_cre_bp_contrbutions(output_path: Path) -> None:

    if output_path.exists():
        print("Contributions already calculated. Exiting.")
        return

    # Optimization: Instantiate the Puffin model once globally outside the parsing loop
    puffin_model_instance = puffin.Puffin(use_cuda=(plasmidtools.crest.DEVICE == 'cuda'))

    # Initialize structured HDF5 file to stream data directly to disk storage
    with h5py.File(output_path, "w") as h5f:

        for idx, element in enumerate(tqdm(cryptic_elements, desc="Processing elements")):
            element_type, element_name = element
            element_dct = repr_seqs_dct.get((element_type, element_name), None)

            if element_dct is None:
                continue

            element_sequence = element_dct["sequence"]

            # Calculate interpretability maps
            onehots, crest_contribs = obtain_crest_contribution_scores(element_sequence, CREST_INDEX)
            contribs_fwd, contribs_rev = obtain_puffin_contribution_scores(puffin_model_instance, element_sequence)

            # Sanitize names to prevent HDF5 structural pathway breaks if names have forward slashes
            sanitized_type = element_type.replace("/", "_")
            sanitized_name = element_name.replace("/", "_")

            # Create sub-group structural layout inside HDF5: /element_type/element_name/
            grp = h5f.create_group(f"{sanitized_type}/{sanitized_name}")

            # Save matrix datasets using gzip compression to save significant disk space
            grp.create_dataset("onehot", data=onehots, compression="gzip", compression_opts=4)
            grp.create_dataset("crest", data=crest_contribs, compression="gzip", compression_opts=4)
            grp.create_dataset("puffin_fwd", data=contribs_fwd, compression="gzip", compression_opts=4)
            grp.create_dataset("puffin_rev", data=contribs_rev, compression="gzip", compression_opts=4)

    print(f"Contribution scores calculated and written to {output_path}")


def process_element(
    element_type: str, 
    element_name: str,
    repr_seqs_dct: dict,
    repr_seqs_cre: pl.DataFrame,
    contribs_output_path: Path,
    output_figures_dir: Path,
) -> None:
    """
    Extracts contribution scores for a specific element and generates plots 
    centered on overlapping CRE/TSS regions, saving them to a combined PDF.
    Redundant heavily overlapping TSS regions are skipped.
    """
    print(f"Processing contribution plots for {element_type} - {element_name}...")

    # 1. Setup Data and Directories
    filestem = f"{plasmidtools.helpers.sanitize_filename(element_type)}__{plasmidtools.helpers.sanitize_filename(element_name)}"
    temp_dir = output_figures_dir / f"{filestem}_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    element_seq_dct = repr_seqs_dct[(element_type, element_name)]

    # Extract coordinate lists 
    cre_row = repr_seqs_cre.filter((pl.col("element_type") == element_type) & (pl.col("element_name") == element_name))

    if len(cre_row) == 0:
        print(f"No CRE/TSS mapping found for {element_type}-{element_name}. Skipping.")
        shutil.rmtree(temp_dir)
        return

    cre_positions = cre_row["CREST (HEK293T)"][0].to_list()
    tss_fwd_positions = cre_row["Puffin (FANTOM_CAGE_fwd)"][0].to_list()
    tss_rev_positions = cre_row["Puffin (FANTOM_CAGE_rev)"][0].to_list()

    # Load Contribution Scores
    contrib_scores = plasmidtools.helpers.load_contribution_scores(contribs_output_path, element_type, element_name)

    # Compute Dot Products
    cre_contribs = contrib_scores['onehot'] * contrib_scores['crest']
    fwd_contribs = contrib_scores['onehot'][:, PUFFIN_FLANK: -PUFFIN_FLANK] * contrib_scores['puffin_fwd'][None, :]
    rev_contribs = contrib_scores['onehot'][:, PUFFIN_FLANK: -PUFFIN_FLANK] * contrib_scores['puffin_rev'][None, :]

    flank_size = element_seq_dct["flank_size"]
    element_size = cre_contribs.shape[1] - 2 * flank_size

    img_counter = 1
    
    # Heuristic parameter: min distance between plot centers to avoid redundant windows
    MIN_PLOT_DISTANCE = 200

    # --- 2. Iterate and Plot CREST (HEK293T) ---
    for cre_start, cre_stop in cre_positions:
        mid = (cre_start + cre_stop) // 2
        start = max(0, mid - MIN_PLOT_DISTANCE)
        stop = min(cre_contribs.shape[1], mid + MIN_PLOT_DISTANCE)

        fig, ax = plasmidtools.contribplots.contribution_scores_plot(cre_contribs[:, start:stop])
        fig, ax = plasmidtools.contribplots.apply_element_annotations(
            fig, ax, slice_start=start, slice_end=stop, 
            flank_size=flank_size, element_size=element_size, 
            element_label=f"{element_type}-{element_name}"
        )
        ax.set_ylabel("CREST (HEK293T)", fontsize=25)
        
        fig.savefig(temp_dir / f"{img_counter:03d}.jpg", dpi=300, format='jpg', bbox_inches='tight')
        plt.close(fig)
        img_counter += 1

    # --- 3. Iterate and Plot Puffin (FANTOM_CAGE_fwd) ---
    plotted_fwd_mids = []
    for tss_start, tss_stop in tss_fwd_positions:
        # Align global coordinate mid to the truncated puffin array
        mid_aligned = ((tss_start + tss_stop) // 2) - PUFFIN_FLANK

        # Only plot if the overlap region actually falls within the shortened Puffin window
        if mid_aligned < 0 or mid_aligned > fwd_contribs.shape[1]:
            continue
            
        # Skip if within MIN_PLOT_DISTANCE of an already plotted FWD TSS
        if any(abs(mid_aligned - prev_mid) < MIN_PLOT_DISTANCE for prev_mid in plotted_fwd_mids):
            continue
            
        plotted_fwd_mids.append(mid_aligned)

        start = max(0, mid_aligned - MIN_PLOT_DISTANCE)
        stop = min(fwd_contribs.shape[1], mid_aligned + MIN_PLOT_DISTANCE)

        fig, ax = plasmidtools.contribplots.contribution_scores_plot(fwd_contribs[:, start:stop], y_min=-1, y_max=100)
        fig, ax = plasmidtools.contribplots.apply_element_annotations(
            fig, ax, slice_start=start, slice_end=stop, 
            flank_size=flank_size - PUFFIN_FLANK, element_size=element_size, 
            element_label=f"{element_type}-{element_name}"
        )
        ax.set_ylabel("Puffin (FANTOM_CAGE_fwd)", fontsize=25)

        fig.savefig(temp_dir / f"{img_counter:03d}.jpg", dpi=300, format='jpg', bbox_inches='tight')
        plt.close(fig)
        img_counter += 1

    # --- 4. Iterate and Plot Puffin (FANTOM_CAGE_rev) ---
    plotted_rev_mids = []
    for tss_start, tss_stop in tss_rev_positions:
        # Align global coordinate mid to the truncated puffin array
        mid_aligned = ((tss_start + tss_stop) // 2) - PUFFIN_FLANK

        if mid_aligned < 0 or mid_aligned > rev_contribs.shape[1]:
            continue

        # Skip if within MIN_PLOT_DISTANCE of an already plotted REV TSS
        if any(abs(mid_aligned - prev_mid) < MIN_PLOT_DISTANCE for prev_mid in plotted_rev_mids):
            continue

        plotted_rev_mids.append(mid_aligned)

        start = max(0, mid_aligned - MIN_PLOT_DISTANCE)
        stop = min(rev_contribs.shape[1], mid_aligned + MIN_PLOT_DISTANCE)

        fig, ax = plasmidtools.contribplots.contribution_scores_plot(rev_contribs[:, start:stop], y_min=-1, y_max=100)
        fig, ax = plasmidtools.contribplots.apply_element_annotations(
            fig, ax, slice_start=start, slice_end=stop, 
            flank_size=flank_size - PUFFIN_FLANK, element_size=element_size, 
            element_label=f"{element_type}-{element_name}"
        )
        ax.set_ylabel("Puffin (FANTOM_CAGE_rev)", fontsize=25)

        fig.savefig(temp_dir / f"{img_counter:03d}.jpg", dpi=300, format='jpg', bbox_inches='tight')
        plt.close(fig)
        img_counter += 1

    # --- 5. Compile PDF and Cleanup ---
    if img_counter > 1:
        plasmidtools.helpers.convert_jpgs_to_pdf(temp_dir, output_figures_dir / f"{filestem}.pdf")
    else:
        print(f"No visualizations generated for {element_type}-{element_name}.")

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # -- 1. Load configurations and data sources ---
    df_clustered = pl.read_csv(ADDGENE_DIR / "element_cre_overlap_clustering.csv")
    cryptic_elements = (
        df_clustered
        .filter(df_clustered["is_cryptic_cre"] & ~df_clustered["type"].is_in(["backbone_spacer", "putative_orf", "putative_noncoding"]))
        .select(["type", "name"])
        .rows()
    )
    repr_seqs_dct = plasmidtools.helpers.load_representative_sequences(ADDGENE_DIR / "element_representative_sequences.fasta")
    repr_seqs_cre = pl.read_parquet(ADDGENE_DIR / "element_representative_sequences_cre_overlaps.parquet")

    # 2. --- Calculate & Save CCC full-length contribution scores ---
    calculate_cryptic_cre_bp_contrbutions(CONTRIBS_OUTPUT)  # 5 hours (L40S GPU)

    # 3. --- Plot (Real) Contribution Scores at CRE / TSS regions ---
    for element_type, element_name in cryptic_elements:  # 6 min
        process_element(element_type, element_name, repr_seqs_dct,
        repr_seqs_cre, CONTRIBS_OUTPUT, CONTRIB_FIGURES,
)

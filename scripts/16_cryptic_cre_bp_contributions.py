import hashlib
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.insert(0, "..")
import plasmidtools

sys.path.insert(0, "../data/puffin")
import puffin

# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
ADDGENE_DIR = DATA_DIR / "addgene"
CLUSTERING_FILE = ADDGENE_DIR / "element_cre_overlap_clustering.csv"
REPR_SEQ_FASTA = ADDGENE_DIR / "element_representative_sequences.fasta"
REPR_SEQ_OVERLAPS = ADDGENE_DIR / "element_representative_sequences_cre_overlaps.parquet"
CONTRIBS_OUTPUT = ADDGENE_DIR / "CCCs_contrib_scores.h5"
CONTRIB_FIGURES = FIGURES_DIR / "contribs"
CONTRIB_FIGURES.mkdir(parents=True, exist_ok=True)

# CREST cell line the contributions are computed for; the pile-ups draw the same one.
CRE_CELL = plasmidtools.statplots.DEFAULT_CRE_CELL
CREST_INDEX = int(np.argwhere(plasmidtools.crest.CREST_LABELS == CRE_CELL)[0, 0])
CREST_TILE_SIZE = 200
PUFFIN_FLANK = 325  # Puffin trims this much from each end of the sequence it scores

# Version of the stored contribution groups. Bump it when the computation changes, so
# groups written by the old code are recomputed instead of reused.
CONTRIB_FORMAT = 1

# Plot windows reach this far either side of a CRE / TSS call midpoint, and a call
# closer than this to one already drawn shares its window rather than getting a page.
MIN_PLOT_DISTANCE = 200

# Narrowest y-ranges drawn; `contribution_scores_plot` widens them to fit the data.
# CREST pages share one range, set above the data (stacks peak at 0.88) so the
# letters keep readable proportions rather than being stretched into spikes: 237 of
# 256 pages are drawn on it exactly, 1 is widened to fit and the 18 tallest are
# stretched by the letter-aspect cap, so CRE strength compares across elements.
# Puffin letter stacks span 3.4-185 between pages, which no shared range suits, so
# that one is only a floor and each page fits its own data.
CREST_Y_RANGE = (-0.4, 1.1)
PUFFIN_Y_RANGE = (-5.0, 10.0)
# Per-bp TACS threshold line; the 0.15 was never calibrated on Puffin's scale.
CREST_TACS_THRESHOLD = 0.15
YLABEL_FONTSIZE = 25  # above FONT_SIZES["axis_label"]; kept from the earlier figures


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
    crest_tiles = [element_sequence[i : i + CREST_TILE_SIZE] for i in range(L - CREST_TILE_SIZE + 1)]

    interpreter = plasmidtools.crest.CRESTInterpreter(batch_size=plasmidtools.crest.BATCH_SIZE, pred_index=crest_index)
    for idx, tile in enumerate(crest_tiles):
        interpreter.update(f"{idx}", tile)

    valid_idxs, onehots, contribs = interpreter.get_predictions()
    valid_idxs = valid_idxs.astype(np.int32)

    # pass total sequence length L to build the reconstructed shape accurately
    realigned_onehots, realigned_contribs = join_contribution_tiles(onehots, contribs, valid_idxs, L)
    return realigned_onehots, realigned_contribs


def obtain_puffin_contribution_scores(puffin_model: puffin.Puffin, element_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    # `interpret` returns fields as rows and positions as columns, so reversing the
    # columns of the reverse-complement result puts it back in forward coordinates -
    # identical (r = 1.0) to Puffin's own `reverse_strand=True` output.
    puff_fwd_interp = {key: np.array(val) for key, val in puffin_model.interpret(element_sequence, targeti="FANTOM_CAGE").T.to_dict('list').items()}
    fwd_contribs = puff_fwd_interp["Basepair contribution score to transcription initiation"]
    puff_rev_interp = {key: np.array(val) for key, val in puffin_model.interpret(reverse_complement(element_sequence), targeti="FANTOM_CAGE").iloc[:, ::-1].T.to_dict('list').items()}
    rev_contribs = puff_rev_interp["Basepair contribution score to transcription initiation"]
    return fwd_contribs, rev_contribs


def contrib_group_path(element_type: str, element_name: str) -> str:
    """HDF5 group of an element, as `helpers.load_contribution_scores` looks it up.

    The same sanitization as the pile-up groups in `15_element_pileups.py`, so one
    group path addresses the same element in both files.
    """
    return (
        f"{plasmidtools.helpers.sanitize_filename(element_type)}"
        f"/{plasmidtools.helpers.sanitize_filename(element_name)}"
    )


def sequence_fingerprint(sequence: str) -> str:
    """Stable fingerprint of the sequence a group's scores were computed from."""
    return hashlib.sha1(sequence.encode()).hexdigest()


def calculate_contribution_scores(
    elements: list[tuple[str, str]],
    repr_seqs_dct: dict,
    output_path: Path,
) -> None:
    """CREST and Puffin contribution scores for each element's representative sequence.

    Each element is written as soon as it is done and stamped with a fingerprint of
    its sequence and `CONTRIB_FORMAT`, written last so a group cut short by a killed
    job carries no stamp. A group whose stamp still matches is reused: an interrupted
    job resumes where it stopped, and a rerun recomputes only elements whose
    representative sequence has changed. Groups under names no representative
    sequence carries any more are deleted.
    """
    current = {contrib_group_path(element_type, element_name) for element_type, element_name in repr_seqs_dct}
    puffin_model = None  # loaded only once something actually needs computing

    with h5py.File(output_path, "a") as h5f:
        orphaned = sorted(
            f"{group_type}/{group_name}"
            for group_type in h5f
            for group_name in h5f[group_type]
            if f"{group_type}/{group_name}" not in current
        )
        for group_path in orphaned:
            print(f"Removed {group_path}: no representative sequence carries that name.")
            del h5f[group_path]
        for group_type in [group_type for group_type in h5f if len(h5f[group_type]) == 0]:
            del h5f[group_type]

        for element_type, element_name in tqdm(elements, desc="Contribution scores"):
            element_dct = repr_seqs_dct.get((element_type, element_name))
            if element_dct is None:
                print(f"No representative sequence for {element_type}/{element_name}; skipped.")
                continue

            element_sequence = element_dct["sequence"]
            fingerprint = sequence_fingerprint(element_sequence)
            group_path = contrib_group_path(element_type, element_name)

            if group_path in h5f:
                attrs = h5f[group_path].attrs
                if attrs.get("sequence_sha1") == fingerprint and int(attrs.get("contrib_format", 0)) == CONTRIB_FORMAT:
                    continue
                del h5f[group_path]  # stale or incomplete

            if puffin_model is None:
                puffin_model = puffin.Puffin(use_cuda=(plasmidtools.crest.DEVICE == 'cuda'))

            onehots, crest_contribs = obtain_crest_contribution_scores(element_sequence, CREST_INDEX)
            contribs_fwd, contribs_rev = obtain_puffin_contribution_scores(puffin_model, element_sequence)

            grp = h5f.create_group(group_path)
            grp.create_dataset("onehot", data=onehots, compression="gzip", compression_opts=4)
            grp.create_dataset("crest", data=crest_contribs, compression="gzip", compression_opts=4)
            grp.create_dataset("puffin_fwd", data=contribs_fwd, compression="gzip", compression_opts=4)
            grp.create_dataset("puffin_rev", data=contribs_rev, compression="gzip", compression_opts=4)
            grp.attrs["crest_cell"] = CRE_CELL
            grp.attrs["contrib_format"] = CONTRIB_FORMAT
            grp.attrs["sequence_sha1"] = fingerprint
            h5f.flush()

    print(f"Contribution scores are in {output_path}")


def plot_windows(intervals: list, offset: int, track_length: int) -> list[tuple[int, int]]:
    """Plot windows around call midpoints, in the coordinates of one score track.

    `offset` moves sequence coordinates onto the track (Puffin's is trimmed by
    PUFFIN_FLANK at each end). Midpoints off the track are dropped, and a call closer
    than MIN_PLOT_DISTANCE to one already kept shares that window.
    """
    windows, kept_mids = [], []
    for call_start, call_stop in intervals:
        mid = (call_start + call_stop) // 2 - offset
        if mid < 0 or mid >= track_length:
            continue
        if any(abs(mid - kept) < MIN_PLOT_DISTANCE for kept in kept_mids):
            continue
        kept_mids.append(mid)
        windows.append((max(0, mid - MIN_PLOT_DISTANCE), min(track_length, mid + MIN_PLOT_DISTANCE)))
    return windows


def process_element(
    element_type: str, 
    element_name: str,
    repr_seqs_dct: dict,
    repr_seqs_cre: pl.DataFrame,
    contribs_output_path: Path,
    output_figures_dir: Path,
) -> None:
    """
    Plots an element's contribution scores in windows centred on the CRE and TSS calls
    overlapping its representative sequence, one page per window, into a single PDF.
    """
    print(f"Processing contribution plots for {element_type} - {element_name}...")

    element_seq_dct = repr_seqs_dct[(element_type, element_name)]
    cre_row = repr_seqs_cre.filter((pl.col("element_type") == element_type) & (pl.col("element_name") == element_name))

    if len(cre_row) == 0:
        print(f"No CRE/TSS mapping found for {element_type}-{element_name}. Skipping.")
        return

    contrib_scores = plasmidtools.helpers.load_contribution_scores(contribs_output_path, element_type, element_name)

    # Dot products with the one-hot sequence
    cre_contribs = contrib_scores['onehot'] * contrib_scores['crest']
    fwd_contribs = contrib_scores['onehot'][:, PUFFIN_FLANK: -PUFFIN_FLANK] * contrib_scores['puffin_fwd'][None, :]
    rev_contribs = contrib_scores['onehot'][:, PUFFIN_FLANK: -PUFFIN_FLANK] * contrib_scores['puffin_rev'][None, :]

    flank_size = element_seq_dct["flank_size"]
    element_size = cre_contribs.shape[1] - 2 * flank_size

    # The representative sequence and its TSS calls are both on the element's own
    # strand, so the forward Puffin track is the element strand.
    tracks = [
        (f"CREST ({CRE_CELL})", cre_contribs, cre_row[f"CREST ({CRE_CELL})"][0].to_list(), 0, flank_size, CREST_Y_RANGE, CREST_TACS_THRESHOLD),
        ("Puffin CAGE, element strand", fwd_contribs, cre_row["Puffin (FANTOM_CAGE_fwd)"][0].to_list(), PUFFIN_FLANK, flank_size - PUFFIN_FLANK, PUFFIN_Y_RANGE, None),
        ("Puffin CAGE, opposite strand", rev_contribs, cre_row["Puffin (FANTOM_CAGE_rev)"][0].to_list(), PUFFIN_FLANK, flank_size - PUFFIN_FLANK, PUFFIN_Y_RANGE, None),
    ]
    pages = [
        (label, scores, start, stop, track_flank, y_range, threshold)
        for label, scores, intervals, offset, track_flank, y_range, threshold in tracks
        for start, stop in plot_windows(intervals, offset, scores.shape[1])
    ]
    if not pages:
        print(f"No visualizations generated for {element_type}-{element_name}.")
        return

    filestem = f"{plasmidtools.helpers.sanitize_filename(element_type)}__{plasmidtools.helpers.sanitize_filename(element_name)}"
    with PdfPages(output_figures_dir / f"{filestem}.pdf") as pdf:
        for label, scores, start, stop, track_flank, (y_min, y_max), threshold in pages:
            fig, ax = plasmidtools.contribplots.contribution_scores_plot(
                scores[:, start:stop], y_min=y_min, y_max=y_max, per_pos_threshold=threshold,
            )
            fig, ax = plasmidtools.contribplots.apply_element_annotations(
                fig, ax, slice_start=start, slice_end=stop,
                flank_size=track_flank, element_size=element_size,
                element_label=f"{element_type}-{element_name}"
            )
            ax.set_ylabel(label, fontsize=YLABEL_FONTSIZE)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)


if __name__ == "__main__":
    # --- 1. Load configurations and data sources ---
    # Same selection as `15_element_pileups.py` and the heatmap view in
    # `03_component_cres.ipynb`: typed elements that are not inactive and are not
    # themselves annotated mammalian Pol II promoters / enhancers.
    df_clustered = pl.read_csv(CLUSTERING_FILE)
    ACTIVE_FILTER = (df_clustered["cluster_label"] != "inactive") & (~df_clustered["is_reference_element"])
    active_elements = df_clustered.filter(ACTIVE_FILTER).select(["type", "name"]).rows()

    repr_seqs_dct = plasmidtools.helpers.load_representative_sequences(REPR_SEQ_FASTA)
    repr_seqs_cre = pl.read_parquet(REPR_SEQ_OVERLAPS)

    # --- 2. Calculate & Save full-length contribution scores ---
    calculate_contribution_scores(active_elements, repr_seqs_dct, CONTRIBS_OUTPUT)  # ~10.4 h for 122 elements (L40S GPU)

    # --- 3. Plot contribution scores at CRE / TSS regions ---
    for element_type, element_name in active_elements:
        process_element(
            element_type, element_name, repr_seqs_dct,
            repr_seqs_cre, CONTRIBS_OUTPUT, CONTRIB_FIGURES,
        )

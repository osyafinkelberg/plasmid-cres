import functools
import re
import typing as tp
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from Bio import SeqIO
from PIL import Image

from .crest import CREST_LABELS

PUFFIN_KEYS = [
    'FANTOM_CAGE_fwd', 'ENCODE_CAGE_fwd', 'ENCODE_RAMPAGE_fwd', 'GRO_CAP_fwd', 'PRO_CAP_fwd',
    'FANTOM_CAGE_rev', 'ENCODE_CAGE_rev', 'ENCODE_RAMPAGE_rev', 'GRO_CAP_rev', 'PRO_CAP_rev'
]

# Feature types Addgene's GenBank files never write with `complement()`: 0 of
# 82,168 polyA_signal, 71,740 misc_feature, 43,832 enhancer, 32,807 LTR, 27,504
# regulatory and 11,838 repeat_region features carry one, and intron has 1 of
# 18,504. Biopython reads a location without `complement()` as the plus strand, so
# on these types a strand of +1 records no direction at all and an element sitting
# reversed on a plasmid still reads forward. `04_addgene_msa.py` assigns their
# orientation from sequence instead. 5'UTR is left out: 12 of its 207 features are
# written with `complement()`, so it does carry a direction.
DIRECTION_FREE_TYPES = frozenset({
    "3'UTR", "LTR", "RBS", "enhancer", "exon", "gap", "intron", "misc_feature",
    "misc_recomb", "misc_signal", "mobile_element", "oriT", "polyA_signal",
    "regulatory", "repeat_region",
})

# Version of the pile-up groups `save_aligned_predictions_h5` writes. Bump it when
# the extraction changes what a stored group means, so stale groups are rebuilt
# rather than reused. 2: signal tracks share the type matrix's element bounds, and
# TSS masks are resolved on the element's own strand. 3: direction-free types take
# their strand from sequence, so rows are drawn in the element's own orientation.
PILEUP_FORMAT = 3

# Labels whose recurring SnapGene notes mark different sequences, written by
# `build_note_splits` in `03_addgene_stats.py`. Rebuild it after downloading new
# GenBank files: a note it has not seen falls back to the bare label.
NOTE_SPLITS = Path(__file__).resolve().parents[1] / "data/addgene/element_name_splits.csv"


def base_feature_name(feat) -> str:
    """The feature's label: the first of `label`, `gene`, `note` and `product`."""
    name = None
    for key in ['label', 'gene', 'note', 'product']:
        if key in feat.qualifiers:
            name = feat.qualifiers[key][0]
            break

    if not name:
        name = "unknown"

    return name


def feature_note(feat) -> str:
    """SnapGene's description of the feature, whitespace-normalised; '' when absent."""
    return " ".join(feat.qualifiers.get("note", [""])[0].split())


@functools.cache
def note_splits() -> dict[tuple[str, str, str], str]:
    """`(element_type, label, note)` -> `element_name`, read from NOTE_SPLITS."""
    if not NOTE_SPLITS.exists():
        raise FileNotFoundError(
            f"{NOTE_SPLITS} is missing; build it with `build_note_splits` in `03_addgene_stats.py`"
        )
    table = pl.read_csv(NOTE_SPLITS, infer_schema_length=0, missing_utf8_is_empty_string=True)
    return {
        (element_type, label, note): element_name
        for element_type, label, note, element_name
        in table.select(["element_type", "label", "note", "element_name"]).iter_rows()
    }


def extract_feature_name(feat) -> str:
    """Element name of a GenBank feature.

    The feature's label, qualified by its SnapGene note where one label covers
    different sequences - `chimeric intron` is four unrelated introns, named apart
    as e.g. `chimeric intron [chimera introns from chicken beta-actin rabbit]`.
    Every other label is returned unchanged.
    """
    label = base_feature_name(feat)
    return note_splits().get((feat.type, label, feature_note(feat)), label)


def extract_genbank_record_by_name(input_file: Path, record_name: str, output_file: Path) -> bool:
    """
    Streams lines as raw text. Identifies the target record by checking 
    the exact string match on the LOCUS name token.
    """
    if output_file.exists():
        print("Output file exists, exiting.")
        return True

    with open(input_file, "r") as in_f:
        in_target_record = False
        record_lines = []

        for line in in_f:
            if line.startswith("LOCUS"):
                tokens = line.split()
                # tokens[1] corresponds to the record's 'name' field
                if len(tokens) > 1 and tokens[1] == record_name:
                    in_target_record = True

            if in_target_record:
                record_lines.append(line)
                if line.startswith("//"):
                    # Write the exact text block and exit immediately
                    with open(output_file, "w") as out_f:
                        out_f.writelines(record_lines)
                    return True

    return False  # Record name not found


def load_stap_tracks(path: Path) -> dict[int, tuple[dict[str, str], np.ndarray]]:
    """Parse mapped/PMID-*.txt into {addgene_id: (header fields, per-bp counts)}.

    Two lines per plasmid: a `>key=value|...` header, then one integer per bp
    (position 1 first). Counts are deduplicated forward-strand molecules.
    """
    tracks = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = dict(kv.split("=", 1) for kv in line[1:].strip().split("|"))
            counts = np.array(next(fh).strip().split(","), dtype=np.int64)
            tracks[int(header["plasmid"].split("-")[1])] = (header, counts)
    return tracks


def load_representative_sequences(fasta_path: Path | str) -> dict[tuple[str, str], dict[str, tp.Any]]: 
    """ 
    Parses the representative element sequences FASTA file (output of scripts/04_addgene_msa.py) 
    and builds a metadata lookup dictionary. 

    Args: 
        fasta_path: Path to the FASTA file containing representative sequences. 

    Returns: 
        A dictionary mapping (element_type, element_name) to a nested dictionary  
        containing structural and citation metrics. 
    """ 
    fasta_path = Path(fasta_path) 
    if not fasta_path.exists(): 
        raise FileNotFoundError(f"The file {fasta_path} does not exist.") 

    representative_map = {} 

    # SeqIO automatically strips the leading '>' from the description field 
    for record in SeqIO.parse(fasta_path, "fasta"): 
        header_parts = record.description.split("|") 

        # We now expect 10 fields due to the newly added plasmid_position and strand
        if len(header_parts) < 10: 
            # Skip or log malformed headers if any exist 
            continue 

        element_type = header_parts[0]
        element_name = header_parts[1]
        flank_size = int(header_parts[2])
        n_plasmids = int(header_parts[3])
        n_citations = int(header_parts[4])
        plasmid_frequency = float(header_parts[5])
        citation_frequency = float(header_parts[6])
        raw_intervals = header_parts[7]
        repr_plasmid_gbk = header_parts[8]
        raw_plasmid_position = header_parts[9]
        strand = int(header_parts[10])

        # Convert "50-120,180-340" string into a list of typed tuples: [(50, 120), (180, 340)] 
        element_intervals = [] 
        if raw_intervals: 
            for interval in raw_intervals.split(","): 
                if "-" in interval: 
                    start, end = interval.split("-") 
                    element_intervals.append((int(start), int(end))) 
   
        # Apply the exact same tuple conversion to the raw plasmid position
        plasmid_position = []
        if raw_plasmid_position:
            for interval in raw_plasmid_position.split(","):
                if "-" in interval:
                    start, end = interval.split("-")
                    plasmid_position.append((int(start), int(end)))

        # Map to the structural layout 
        key = (element_type, element_name) 
        representative_map[key] = { 
            "sequence": str(record.seq).upper(),  # Forced uppercase as requested 
            "flank_size": flank_size, 
            "n_plasmids": n_plasmids, 
            "n_citations": n_citations, 
            "plasmid_frequency": plasmid_frequency, 
            "citation_frequency": citation_frequency, 
            "element_intervals": element_intervals,
            "representative_plasmid_gbk": repr_plasmid_gbk,
            "plasmid_position": plasmid_position,
            "strand": strand
        } 

    return representative_map


def load_oriented_elements(elements_path: Path | str, orientation_path: Path | str) -> pl.DataFrame:
    """The element table, with direction-free strands taken from sequence.

    For instances of `DIRECTION_FREE_TYPES` that `04_addgene_msa.py` oriented,
    `strand` becomes the GenBank strand times that sequence orientation. The
    orientation is relative to sequences extracted strand-aware, so the product is
    what places the rare instance these types do write with `complement()` (an
    intron and a 3'UTR); every other one is annotated +1 and simply takes the
    orientation. The GenBank value is kept as `annotated_strand`, and
    `orientation_source` says which of the two each row uses. Annotated types keep their strand, as do features below the
    25 bp extraction threshold (`regulatory` and `RBS` are ~10 bp) and any variant
    too diverged to place.

    Row order is preserved: `08_element_cre_overlap.py` walks the table expecting a
    plasmid's elements to be contiguous.
    """
    elements = pl.read_parquet(elements_path)
    key = ["gbk_name", "element_type", "element_name", "intervals"]
    orientation = pl.read_parquet(orientation_path).select([*key, "sequence_orientation"])

    oriented = (
        elements.join(orientation, on=key, how="left", maintain_order="left")
        .with_columns(
            from_sequence=(
                pl.col("element_type").is_in(list(DIRECTION_FREE_TYPES))
                & pl.col("sequence_orientation").is_not_null()
            )
        )
        .with_columns(
            annotated_strand=pl.col("strand"),
            strand=pl.when(pl.col("from_sequence"))
                     .then(pl.col("strand") * pl.col("sequence_orientation"))
                     .otherwise(pl.col("strand")),
            orientation_source=pl.when(pl.col("from_sequence"))
                                 .then(pl.lit("sequence"))
                                 .otherwise(pl.lit("annotation")),
        )
        .drop(["sequence_orientation", "from_sequence"])
    )

    if oriented.height != elements.height:
        raise ValueError(
            f"the orientation table duplicated element rows: {elements.height} -> {oriented.height}"
        )

    return oriented


def sanitize_filename(name: str) -> str:
    """Removes invalid characters for file and path structures."""
    return re.sub(r'[^\w\-_\.]', '_', name)


def has_sufficient_flank(
    element_type: str, 
    element_name: str, 
    flank_size: int, 
    h5_path: str | Path
) -> bool:
    """
    Checks if an element is already stored in the H5 file with an equal 
    or greater flank size, written in the current `PILEUP_FORMAT`. Safe to run
    before heavy matrix computations.
    """
    if not Path(h5_path).exists():
        return False

    group_path = f"{sanitize_filename(element_type)}/{sanitize_filename(element_name)}"
    with h5py.File(h5_path, "r") as h5f:
        if group_path in h5f:
            group = h5f[group_path]
            if int(group.attrs.get("pileup_format", 1)) != PILEUP_FORMAT:
                return False
            existing_flank = group.attrs.get("flank_size", -1)
            return existing_flank >= flank_size

    return False


def save_aligned_predictions_h5(
    element_type: str,
    element_name: str,
    element_size: int,
    flank_size: int,
    type_matrix: np.ndarray,
    cre_matrix: np.ndarray,
    tss_fwd_matrix: np.ndarray,
    tss_rev_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    pred_fwd_matrix: np.ndarray,
    pred_rev_matrix: np.ndarray,
    h5_path: str | Path
) -> None:
    """
    Saves all structural, binary, and continuous prediction matrices to the H5 file.
    Uses has_sufficient_flank to skip or overwrite.
    """
    if has_sufficient_flank(element_type, element_name, flank_size, h5_path):
        print(f"Skipping storage for {element_type}/{element_name}: Adequate flank already exists.")
        return

    group_path = f"{sanitize_filename(element_type)}/{sanitize_filename(element_name)}"
    with h5py.File(h5_path, "a") as h5f:
        if group_path in h5f:
            print(f"Overwriting {group_path}: stored flank_size or pileup format is out of date.")
            del h5f[group_path]
        else:
            print(f"Saving {group_path}: New element with flank_size {flank_size}.")

        group = h5f.create_group(group_path)
        group.attrs["element_size"] = element_size
        group.attrs["flank_size"] = flank_size
        group.attrs["pileup_format"] = PILEUP_FORMAT

        # Apply GZIP compression chunks for efficient dense array storage
        group.create_dataset("type_matrix", data=type_matrix, compression="gzip", chunks=True)
        group.create_dataset("cre_matrix", data=cre_matrix, compression="gzip", chunks=True)
        group.create_dataset("tss_fwd_matrix", data=tss_fwd_matrix, compression="gzip", chunks=True)
        group.create_dataset("tss_rev_matrix", data=tss_rev_matrix, compression="gzip", chunks=True)
        group.create_dataset("pred_matrix", data=pred_matrix, compression="gzip", chunks=True)
        group.create_dataset("pred_fwd_matrix", data=pred_fwd_matrix, compression="gzip", chunks=True)
        group.create_dataset("pred_rev_matrix", data=pred_rev_matrix, compression="gzip", chunks=True)


def load_aligned_predictions_h5(
    element_type: str,
    element_name: str,
    h5_path: str | Path
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the flank size and all 7 associated matrices for a specific plasmid element.

    Returns:
        A tuple containing:
            - element_size (int)
            - flank_size (int)
            - type_matrix
            - cre_matrix
            - tss_fwd_matrix
            - tss_rev_matrix
            - pred_matrix
            - pred_fwd_matrix
            - pred_rev_matrix
    """
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"H5 database file not found at: {h5_path}")

    group_path = f"{sanitize_filename(element_type)}/{sanitize_filename(element_name)}"

    with h5py.File(h5_path, "r") as h5f:
        if group_path not in h5f:
            raise KeyError(f"No data found for element path: '{group_path}'")

        group = h5f[group_path]

        element_size = int(group.attrs.get("element_size", -1))
        flank_size = int(group.attrs.get("flank_size", -1))

        pileup_format = int(group.attrs.get("pileup_format", 1))
        if pileup_format != PILEUP_FORMAT:
            print(
                f"Warning: {group_path} was written in pileup format {pileup_format}, not "
                f"{PILEUP_FORMAT}; regenerate it with `15_element_pileups.py`."
            )

        # Pull datasets entirely into memory as numpy arrays [:]
        type_matrix = group["type_matrix"][:]
        cre_matrix = group["cre_matrix"][:]
        tss_fwd_matrix = group["tss_fwd_matrix"][:]
        tss_rev_matrix = group["tss_rev_matrix"][:]
        pred_matrix = group["pred_matrix"][:]
        pred_fwd_matrix = group["pred_fwd_matrix"][:]
        pred_rev_matrix = group["pred_rev_matrix"][:]

    return (
        element_size, flank_size, 
        type_matrix, cre_matrix, tss_fwd_matrix, tss_rev_matrix, 
        pred_matrix, pred_fwd_matrix, pred_rev_matrix
    )


def load_contribution_scores(
    h5_path: Path | str, 
    element_type: str, 
    element_name: str
) -> dict[str, np.ndarray]:
    """
    Loads the contribution scores for a specific element from the HDF5 dataset.

    Args:
        h5_path: Path to the .h5 file containing the contribution scores.
        element_type: The raw type of the element (as it appears in your DataFrame).
        element_name: The raw name of the element.

    Returns:
        A dictionary with keys 'onehot', 'crest', 'puffin_fwd', and 'puffin_rev',
        mapping to their respective numpy arrays.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"The dataset file {h5_path} does not exist.")

    # Replicate the sanitization used during writing to ensure path matching. This is
    # `sanitize_filename`, the same scheme the pile-up groups use, so a group path means
    # the same thing in both files; it replaced a bare "/" -> "_" that spelled 349 of the
    # 1,270 element names differently from the pile-ups.
    group_path = f"{sanitize_filename(element_type)}/{sanitize_filename(element_name)}"

    result = {}
    with h5py.File(h5_path, "r") as h5f:
        if group_path not in h5f:
            raise KeyError(f"Element path '{group_path}' not found in the dataset.")

        group = h5f[group_path]

        # Load datasets entirely into memory as numpy arrays using [:]
        result["onehot"] = group["onehot"][:]
        result["crest"] = group["crest"][:]
        result["puffin_fwd"] = group["puffin_fwd"][:]
        result["puffin_rev"] = group["puffin_rev"][:]

    return result


def load_mutant_preds(h5_path: Path, mutant_name: str, crest_tile_size: int = 200) -> dict:
    """
    Loads and reconstructs complete prediction and contribution score tracks 
    for a given mutant sequence from the centralized HDF5 dataset.
    """
    out_dict = {}
    
    with h5py.File(h5_path, "r") as h5f:
        if f"mutants/{mutant_name}" not in h5f:
            raise KeyError(f"Mutant '{mutant_name}' not found in dataset.")
            
        mut_grp = h5f[f"mutants/{mutant_name}"]
        
        # 1. Load Puffin Data
        # Shape (10, L)
        puffin_preds = mut_grp["puffin_preds"][:]
        for i, track_name in enumerate(PUFFIN_KEYS):
            out_dict[f"Puffin_{track_name}"] = puffin_preds[i]
            
        out_dict["Puffin_cs_FANTOM_CAGE_fwd"] = mut_grp["puffin_contrib_fwd"][:]
        out_dict["Puffin_cs_FANTOM_CAGE_rev"] = mut_grp["puffin_contrib_rev"][:]
        
        # 2. Extract CREST tile mapping
        tile_ids = mut_grp["crest_tile_ids"][:]
        L = len(tile_ids) + crest_tile_size - 1
        
        # 3. Efficiently fetch unique tiles from global HDF5 storage
        unique_ids, inv_indices = np.unique(tile_ids, return_inverse=True)
        unique_ids_list = unique_ids.tolist()
        
        fetched_preds = h5f["tiles/preds"][unique_ids_list]
        fetched_onehots = h5f["tiles/onehots"][unique_ids_list]
        fetched_cs_hek = h5f["tiles/contribs_HEK293T"][unique_ids_list]
        fetched_cs_k562 = h5f["tiles/contribs_K562"][unique_ids_list]
        
        # Expand unique tiles back into the original sequence order
        ordered_preds = fetched_preds[inv_indices]
        ordered_onehots = fetched_onehots[inv_indices]
        ordered_cs_hek = fetched_cs_hek[inv_indices]
        ordered_cs_k562 = fetched_cs_k562[inv_indices]
        
    # 4. Reconstruct Full Sequence Arrays via Overlap Accumulation (for one-hot and contribs)
    accum_onehot = np.zeros((4, L), dtype=np.float32)
    accum_cs_hek = np.zeros((4, L), dtype=np.float32)
    accum_cs_k562 = np.zeros((4, L), dtype=np.float32)
    coverage_counts = np.zeros(L, dtype=np.int32)
    
    for i in range(len(tile_ids)):
        start, end = i, i + crest_tile_size
        
        # One-hot can just be overwritten since overlapping bases are identical
        accum_onehot[:, start:end] = ordered_onehots[i]
        
        # Contributions are summed, then averaged
        accum_cs_hek[:, start:end] += ordered_cs_hek[i]
        accum_cs_k562[:, start:end] += ordered_cs_k562[i]
        coverage_counts[start:end] += 1
        
    # Prevent division by zero
    safe_counts = np.where(coverage_counts == 0, 1, coverage_counts)
    
    out_dict["onehot"] = accum_onehot
    out_dict["CREST_cs_HEK293T"] = accum_cs_hek / safe_counts
    out_dict["CREST_cs_K562"] = accum_cs_k562 / safe_counts
    
    # 5. Map CREST Predictions (per bp)
    # Assign each tile's prediction to the center base pair of that tile
    final_preds = np.full((L, len(CREST_LABELS)), np.nan, dtype=np.float32)
    half_sz = crest_tile_size // 2
    final_preds[half_sz : half_sz + len(tile_ids), :] = ordered_preds
    
    for j, cell_name in enumerate(CREST_LABELS):
        out_dict[f"CREST_{cell_name}"] = final_preds[:, j]
        
    return out_dict


def convert_jpgs_to_pdf(images_dir: Path, output_pdf: Path) -> None:
    jpg_files = sorted(images_dir.glob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError(f"No JPG files found in directory: {images_dir}")

    images = [Image.open(fp).convert("RGB") for fp in jpg_files]
    images[0].save(
        output_pdf,
        format="PDF",
        resolution=100.0,
        save_all=True,
        append_images=images[1:]
    )

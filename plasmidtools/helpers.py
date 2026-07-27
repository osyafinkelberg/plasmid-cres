import typing as tp
from pathlib import Path
import h5py
import numpy as np
import re
from Bio import SeqIO
from PIL import Image

from .crest import CREST_LABELS

PUFFIN_KEYS = [
    'FANTOM_CAGE_fwd', 'ENCODE_CAGE_fwd', 'ENCODE_RAMPAGE_fwd', 'GRO_CAP_fwd', 'PRO_CAP_fwd',
    'FANTOM_CAGE_rev', 'ENCODE_CAGE_rev', 'ENCODE_RAMPAGE_rev', 'GRO_CAP_rev', 'PRO_CAP_rev'
]


def extract_feature_name(feat) -> str:
    name = None
    for key in ['label', 'gene', 'note', 'product']:
        if key in feat.qualifiers:
            name = feat.qualifiers[key][0]
            break

    if not name:
        name = "unknown"

    return name


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


def sanitize_filename(name: str) -> str:
    """Removes invalid characters for file and path structures."""
    return re.sub(r'[^\w\-_\.]', '_', name)


def has_sufficient_flank(
    element_type: str, 
    element_name: str, 
    flank_size: int, 
    h5_path: tp.Union[str, Path]
) -> bool:
    """
    Checks if an element is already stored in the H5 file with an equal 
    or greater flank size. Safe to run before heavy matrix computations.
    """
    if not Path(h5_path).exists():
        return False

    group_path = f"{sanitize_filename(element_type)}/{sanitize_filename(element_name)}"
    with h5py.File(h5_path, "r") as h5f:
        if group_path in h5f:
            existing_flank = h5f[group_path].attrs.get("flank_size", -1)
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
    h5_path: tp.Union[str, Path]
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
            print(f"Overwriting {group_path}: Updating to a larger flank_size ({flank_size}).")
            del h5f[group_path]
        else:
            print(f"Saving {group_path}: New element with flank_size {flank_size}.")

        group = h5f.create_group(group_path)
        group.attrs["element_size"] = element_size
        group.attrs["flank_size"] = flank_size

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
    h5_path: tp.Union[str, Path]
) -> tp.Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Replicate the sanitization used during writing to ensure path matching
    sanitized_type = element_type.replace("/", "_")
    sanitized_name = element_name.replace("/", "_")
    group_path = f"{sanitized_type}/{sanitized_name}"

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

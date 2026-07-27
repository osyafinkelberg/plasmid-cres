from pathlib import Path
import numpy as np
import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm


# --- CONFIGURATION ---
CUR_DIR = Path(__file__).resolve()
ADDGENE_DIR = CUR_DIR.parent.parent / "data/addgene"
COMBINED_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"
PLASMID_CITATIONS = ADDGENE_DIR / "citations_addgene.parquet"

ELEMENT_POSITIONS_OUT = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
PRIMERS_POSITIONS_OUT = ADDGENE_DIR / "mammalian_plasmids_primers.parquet"
PLASMID_STATS_OUT = ADDGENE_DIR / "mammalian_plasmids_statistics.parquet"
ELEMENT_CITATIONS_OUT = ADDGENE_DIR / "citations_addgene_elements.parquet"
PRIMERS_CITATIONS_OUT = ADDGENE_DIR / "citations_addgene_primers.parquet"

MIN_ORF_NUC_LENGTH = 300
MIN_INS_NUC_LENTHS = 150
COMMON_MARKERS = ["puro", "bsd", "zeo", "neo", "hygro", "gfp", "yfp", "cfp", "mcherry", "luciferase"]


def extract_feature_name(feat) -> str:
    name = None
    for key in ['label', 'gene', 'note', 'product']:
        if key in feat.qualifiers:
            name = feat.qualifiers[key][0]
            break

    if not name:
        name = "unknown"

    return name


def is_insert_match(feat_name: str, expected_inserts: list) -> bool:
    """
    Fuzzy matches a GenBank feature name against the Addgene metadata inserts list.
    """
    if not feat_name or not expected_inserts:
        return False

    feat_lower = feat_name.lower().replace("-", "")

    for ins in expected_inserts:
        ins_lower = ins.lower().replace("-", "")

        # 1. Direct substring match (e.g., "dcas9" in "fb-tagged dcas9")
        if feat_lower in ins_lower or ins_lower in feat_lower:
            return True

        # 2. Common synthetic biology abbreviations mapping
        for marker in COMMON_MARKERS:
            if marker in feat_lower and marker in ins_lower:
                return True

    return False


def find_all_orfs(sequence: str, min_length: int = MIN_ORF_NUC_LENGTH) -> list:
    """
    Scans the entire global sequence to find all ORFs.
    Returns a list of dictionaries with precise coordinates.
    """
    seq_obj = Seq(sequence)
    orfs = []
    seq_len = len(sequence)

    for strand, nuc_seq in [(1, seq_obj), (-1, seq_obj.reverse_complement())]:
        for frame in range(3):
            # Calculate how many bases to trim to make the slice length a multiple of 3
            trim = (len(nuc_seq) - frame) % 3

            # Avoid slicing with 0 which returns an empty string. 
            # Use explicit length math instead of negative indexing.
            end_slice = len(nuc_seq) - trim
            trans = str(nuc_seq[frame:end_slice].translate())

            aa_start = 0
            peptides = trans.split("*")

            for i, peptide in enumerate(peptides):
                m_idx = peptide.find('M')
                if m_idx != -1:
                    orf_aa_len = len(peptide) - m_idx
                    orf_nuc_len = orf_aa_len * 3

                    if orf_nuc_len >= min_length:
                        nuc_start = frame + ((aa_start + m_idx) * 3)

                        # The last peptide in a split string does NOT end with a stop codon
                        # unless the string itself ended with one (resulting in an empty last peptide).
                        has_stop_codon = (i < len(peptides) - 1)

                        # Only add the 3 base pairs if a stop codon actually exists
                        nuc_end = nuc_start + orf_nuc_len + (3 if has_stop_codon else 0)

                        if strand == 1:
                            orfs.append({
                                "start": nuc_start, 
                                "end": nuc_end, 
                                "strand": strand, 
                                "len": orf_nuc_len
                            })
                        else:
                            # Because we guaranteed nuc_end <= seq_len, these will never be negative
                            orfs.append({
                                "start": seq_len - nuc_end, 
                                "end": seq_len - nuc_start, 
                                "strand": strand, 
                                "len": orf_nuc_len
                            })

                aa_start += len(peptide) + 1 # +1 to account for the split '*'

    return orfs


def collect_plasmid_element_data(
    expected_inserts: dict[str, list[str]],
    insert_min_len: int = MIN_INS_NUC_LENTHS,
    orf_min_len: int = MIN_ORF_NUC_LENGTH,
) -> None:
    feature_data = []

    for record in tqdm(SeqIO.parse(COMBINED_GBK, "genbank"), total=N_PLASMIDS, desc="Mapping precise boundaries"):
        gbk_name = record.name
        seq_id = int(gbk_name.removeprefix("sequence-").removesuffix("-").removesuffix("-e"))

        insert_names = expected_inserts.get(seq_id, [])
        L = len(record.seq)
        annotated_mask = np.zeros(L, dtype=bool)

        # --- STEP 1: Feature Metadata Matching ---
        for feat in record.features:
            if feat.type in ['source', 'primer_bind']: 
                continue  # primers are ignored for structural mask building

            name = extract_feature_name(feat)
            intervals = [(int(p.start), int(p.end)) for p in feat.location.parts]

            # check if this feature is actually an insert
            if is_insert_match(name, insert_names):
                ins_back_state = "insert"
            else:
                ins_back_state = "backbone"
                for start, end in intervals:
                    if start < end:
                        annotated_mask[start: end] = True
                    else:
                        annotated_mask[start:] = True
                        annotated_mask[:end] = True

            feature_data.append({
                "gbk_name": gbk_name, "sequence_id": seq_id,
                "insert_or_backbone": ins_back_state,
                "element_type": feat.type, "element_name": name,
                "length": len(feat), "strand": feat.location.strand, 
                "intervals": intervals 
            })

        # --- STEP 2: Gap & ORF Resolution ---
        unannotated_indices = np.where(~annotated_mask)[0]
        global_orfs = find_all_orfs(str(record.seq), orf_min_len)

        if len(unannotated_indices) > 0:
            breaks = np.where(np.diff(unannotated_indices) > 1)[0]
            blocks = np.split(unannotated_indices, breaks + 1)

            for i, block in enumerate(blocks):
                gap_start, gap_end = int(block[0]), int(block[-1]) + 1
                gap_len = len(block)

                if gap_len < insert_min_len:
                    feature_data.append({
                        "gbk_name": gbk_name, "sequence_id": seq_id,
                        "insert_or_backbone": "backbone",
                        "element_type": "backbone_spacer", "element_name": f"spacer_{i+1}",
                        "length": gap_len, "strand": 0, "intervals": [(gap_start, gap_end)]
                    })
                    continue

                # find all ORFs that overlap this gap
                overlapping_orfs = [
                    orf for orf in global_orfs
                    if max(gap_start, orf["start"]) < min(gap_end, orf["end"])
                ]

                if overlapping_orfs:
                    # grab the largest overlapping ORF
                    largest_orf = max(overlapping_orfs, key=lambda x: x["len"])

                    # add the ORF insert
                    feature_data.append({
                        "gbk_name": gbk_name, "sequence_id": seq_id,
                        "insert_or_backbone": "insert",
                        "element_type": "putative_orf", "element_name": f"insert_orf_{i+1}",
                        "length": largest_orf["len"], "strand": largest_orf["strand"], 
                        "intervals": [(largest_orf["start"], largest_orf["end"])]
                    })

                    # add remaining uncovered gap parts as background sequences (backbone spacers)
                    if gap_start < largest_orf["start"]:
                        left_end = min(gap_end, largest_orf["start"])
                        if left_end > gap_start:
                            feature_data.append({
                                "gbk_name": gbk_name, "sequence_id": seq_id,
                                "insert_or_backbone": "backbone",
                                "element_type": "backbone_spacer", "element_name": f"spacer_{i+1}_left",
                                "length": left_end - gap_start, "strand": 0, 
                                "intervals": [(gap_start, left_end)]
                            })

                    if gap_end > largest_orf["end"]:
                        right_start = max(gap_start, largest_orf["end"])
                        if right_start < gap_end:
                            feature_data.append({
                                "gbk_name": gbk_name, "sequence_id": seq_id,
                                "insert_or_backbone": "backbone",
                                "element_type": "backbone_spacer", "element_name": f"spacer_{i+1}_right",
                                "length": gap_end - right_start, "strand": 0, 
                                "intervals": [(right_start, gap_end)]
                            })

                else:
                    # gap is large but has no ORF (e.g., shRNA, enhancer, or non-coding RNA)
                    feature_data.append({
                        "gbk_name": gbk_name, "sequence_id": seq_id,
                        "insert_or_backbone": "insert",
                        "element_type": "putative_noncoding", "element_name": f"insert_noncoding_{i+1}",
                        "length": gap_len, "strand": 0, "intervals": [(gap_start, gap_end)]
                    })

    pl.DataFrame(feature_data).write_parquet(ELEMENT_POSITIONS_OUT)


def collect_primer_data() -> None:
    """
    Extracts 'primer_bind' features from GenBank records and cross-references 
    them against the previously annotated plasmid elements to find structural overlap.
    """
    # the previously collected elements
    elements_df = pl.read_parquet(ELEMENT_POSITIONS_OUT)

    # group elements by sequence_id for O(1) lookup
    elements_by_seq = {}
    for row in elements_df.iter_rows(named=True):
        seq_id = row["sequence_id"]
        if seq_id not in elements_by_seq:
            elements_by_seq[seq_id] = []
        elements_by_seq[seq_id].append(row)

    primer_data = []
    for record in tqdm(SeqIO.parse(COMBINED_GBK, "genbank"), total=N_PLASMIDS, desc="Mapping primers"):
        gbk_name = record.name
        seq_id = int(gbk_name.removeprefix("sequence-").removesuffix("-").removesuffix("-e"))
        L = len(record.seq)

        # Pre-compute nucleotide index sets for the existing elements on this plasmid
        # We sort by length so that the tightest overlapping feature is prioritized.
        plasmid_elements = elements_by_seq.get(seq_id, [])
        mapped_elements = []
        for elem in plasmid_elements:
            idx_set = set()
            for s, e in elem["intervals"]:
                if s < e:
                    idx_set.update(range(s, e))
                else:
                    # handle circular wrap-around intervals
                    idx_set.update(range(s, L))
                    idx_set.update(range(0, e))

            mapped_elements.append({
                "type": elem["element_type"],
                "name": elem["element_name"],
                "length": elem["length"],
                "indices": idx_set
            })

        mapped_elements.sort(key=lambda x: x["length"])

        # extract and map the primer_bind features
        for feat in record.features:
            if feat.type == 'primer_bind':
                name = extract_feature_name(feat)
                intervals = [(int(p.start), int(p.end)) for p in feat.location.parts]

                # compute nucleotide index set for the primer
                primer_idx = set()
                for s, e in intervals:
                    if s < e:
                        primer_idx.update(range(s, e))
                    else:
                        primer_idx.update(range(s, L))
                        primer_idx.update(range(0, e))

                # check for absolute containment within other features
                overlap_type = ""
                overlap_name = ""
                for elem in mapped_elements:
                    if primer_idx.issubset(elem["indices"]):
                        overlap_type = elem["type"]
                        overlap_name = elem["name"]
                        break  # found the most specific (shortest) containing feature

                primer_data.append({
                    "gbk_name": gbk_name,
                    "sequence_id": seq_id,
                    "element_type": feat.type,
                    "element_name": name,
                    "length": len(feat),
                    "strand": feat.location.strand,
                    "intervals": intervals,
                    "overlap_feature_type": overlap_type,
                    "overlap_feature_name": overlap_name
                })

    pl.DataFrame(primer_data).write_parquet(PRIMERS_POSITIONS_OUT)


def calculate_plasmid_statistics() -> None:
    """
    Reads the previously collected plasmid element data and the full GenBank dataset 
    to calculate true, non-overlapping physical sequence lengths and feature counts,
    discriminating between insert types and computing an aggregate total insert length.
    """
    # pre-calculated in the `collect_plasmid_element_data` function
    df = pl.read_parquet(ELEMENT_POSITIONS_OUT)

    # pre-process: group elements by sequence_id into a dictionary for O(1) lookup
    elements_by_seq = {}
    for row in df.iter_rows(named=True):
        seq_id = row["sequence_id"]
        if seq_id not in elements_by_seq:
            elements_by_seq[seq_id] = []
        elements_by_seq[seq_id].append(row)

    plasmid_metadata = []

    # iterate through GenBank records to determine absolute lengths and calculate statistics
    for record in tqdm(SeqIO.parse(COMBINED_GBK, "genbank"), total=N_PLASMIDS, desc="Calculating statistics"):
        gbk_name = record.name
        seq_id = int(gbk_name.removeprefix("sequence-").removesuffix("-").removesuffix("-e"))
        p_len = len(record.seq)

        # boolean masks to prevent double-counting overlapping elements
        backbone_mask = np.zeros(p_len, dtype=bool)
        ann_insert_mask = np.zeros(p_len, dtype=bool)
        put_orf_mask = np.zeros(p_len, dtype=bool)
        put_nc_mask = np.zeros(p_len, dtype=bool)
        n_feats = 0
        
        # map the intervals to masks
        if seq_id in elements_by_seq:
            for row in elements_by_seq[seq_id]:
                ins_back = row["insert_or_backbone"]
                elem_type = row["element_type"]
                intervals = row["intervals"]

                # 'source' and 'primer_bind' are ignored in the `collect_plasmid_element_data` function
                # skip background spacers to count actual biological features.
                if elem_type != "backbone_spacer":
                    n_feats += 1

                for start, end in intervals:
                    # handle standard vs circular wrap-around intervals
                    if start < end:
                        slices = [(start, end)]
                    else:
                        slices = [(start, p_len), (0, end)]

                    for s_start, s_end in slices:
                        if ins_back == "backbone":
                            backbone_mask[s_start:s_end] = True
                        elif ins_back == "insert":
                            if elem_type == "putative_orf":
                                put_orf_mask[s_start:s_end] = True
                            elif elem_type == "putative_noncoding":
                                put_nc_mask[s_start:s_end] = True
                            else:
                                ann_insert_mask[s_start:s_end] = True

        # combined mask for all insert types to resolve multi-type overlaps
        total_insert_mask = ann_insert_mask | put_orf_mask | put_nc_mask

        plasmid_metadata.append({
            "gbk_name": gbk_name,
            "sequence_id": seq_id,
            "plasmid_length": p_len,
            "backbone_length": int(np.sum(backbone_mask)),
            "total_insert_length": int(np.sum(total_insert_mask)),
            "annotated_insert_length": int(np.sum(ann_insert_mask)),
            "putative_orf_insert_length": int(np.sum(put_orf_mask)),
            "putative_noncoding_insert_length": int(np.sum(put_nc_mask)),
            "n_feats": n_feats
        })

    pl.DataFrame(plasmid_metadata).write_parquet(PLASMID_STATS_OUT)


def calculate_element_citation_statistics(elements_path: Path, output_path: Path) -> None:
    plasmid_download = pl.read_csv(ADDGENE_DIR / "mammalian_plasmids.tsv", separator="\t")
    plasmid_citations = pl.read_parquet(PLASMID_CITATIONS)
    plasmid_elements = pl.read_parquet(elements_path)

    # filter out non-GenBank custom annotated features
    annotated_types = ["backbone_spacer", "putative_orf", "putative_noncoding"]
    genbank_elements = plasmid_elements.filter(~pl.col("element_type").is_in(annotated_types))

    # join elements with metadata map to resolve sequence_id -> plasmid_id
    elements_with_ids = genbank_elements.join(
        plasmid_download.select(["sequence_id", "plasmid_id"]), 
        on="sequence_id", 
        how="inner"
    )

    # left join with citations to preserve features on plasmids with 0 citations
    elements_with_citations = elements_with_ids.join(
        plasmid_citations.select(["plasmid_id", "citing_pmids"]), 
        on="plasmid_id", 
        how="left"
    )

    # explode the PMID list to evaluate element-level union sets efficiently
    exploded_df = elements_with_citations.explode("citing_pmids")

    # group by GenBank feature definitions and calculate metrics
    elements_stats = (
        exploded_df.group_by(["element_type", "element_name"])
        .agg([
            pl.col("sequence_id").n_unique().alias("n_plasmids"),
            pl.col("citing_pmids").drop_nulls().n_unique().alias("n_citations"),
            pl.col("citing_pmids")
            .drop_nulls()
            .unique()
            .cast(pl.String)
            .alias("citing_pmids")
        ])
        .sort(["n_citations", "n_plasmids"], descending=[True, True])
    )

    elements_stats.write_parquet(output_path)


if __name__ == "__main__":
    plasmid_download = pl.read_csv(ADDGENE_DIR / "mammalian_plasmids.tsv", separator="\t")
    expected_inserts = {
        row["sequence_id"]: row["inserts"].split(" ||| ") if row["inserts"] else [] for row in plasmid_download.iter_rows(named=True)
    }
    N_PLASMIDS = plasmid_download.filter(pl.col("download_status") == "200").height

    # # 1.
    collect_plasmid_element_data(expected_inserts, MIN_INS_NUC_LENTHS, MIN_ORF_NUC_LENGTH)

    # # 2.
    collect_primer_data()

    # # 3.
    calculate_plasmid_statistics()

    # # 4.
    calculate_element_citation_statistics(ELEMENT_POSITIONS_OUT, ELEMENT_CITATIONS_OUT)
    calculate_element_citation_statistics(PRIMERS_POSITIONS_OUT, PRIMERS_CITATIONS_OUT)

    pass

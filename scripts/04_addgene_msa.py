from pathlib import Path
import subprocess
import numpy as np
import polars as pl
from Bio import SeqIO
from Bio import AlignIO, motifs
from Bio.Seq import Seq
from tqdm import tqdm
import re
from collections import defaultdict


# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
COMBINED_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"

FASTA_DIR = ADDGENE_DIR / "fasta"
OUT_INDIVIDUAL_DIVERGENCE = ADDGENE_DIR / "element_individual_divergence.parquet"
OUT_AVERAGE_DIVERGENCE = ADDGENE_DIR / "element_average_divergence.parquet"
MSA_ERROR_LOG = ADDGENE_DIR / "element_msa_errors.tsv"
REP_FASTA_OUT = ADDGENE_DIR / "element_representative_sequences.fasta"

SEQ_LEN_THRESH = 25
REPRESENT_FLANK_SIZE = 325


def extract_feature_name(feat) -> str:
    name = None
    for key in ['label', 'gene', 'note', 'product']:
        if key in feat.qualifiers:
            name = feat.qualifiers[key][0]
            break
    return name if name else "unknown"


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\.]', '_', name)


def extract_subtypes_to_fasta() -> None:
    if FASTA_DIR.exists():
        print("Sequences are already extracted (directory exists)")
        return

    FASTA_DIR.mkdir(parents=True)

    # optimization: collect in memory instead of keeping thousands of file handles open
    seq_collections = defaultdict(list)

    for record in tqdm(SeqIO.parse(COMBINED_GBK, "genbank"), total=N_PLASMIDS):
        for feat in record.features:
            if feat.type == 'source':
                continue

            element_type = feat.type
            element_name = extract_feature_name(feat)
            sanitized_id = f"{sanitize_filename(element_type)}__{sanitize_filename(element_name)}"

            seq_id = f"{record.name}|||{element_type}|||{element_name}|||{sanitized_id}"
            sequence = str(feat.extract(record.seq)).upper()

            if len(sequence) < SEQ_LEN_THRESH:
                continue

            seq_collections[sanitized_id].append(f">{seq_id}\n{sequence}\n")

    for sid, seqs in seq_collections.items():
        with open(FASTA_DIR / f"{sid}.fasta", "w") as f:
            f.writelines(seqs)


def make_and_analyze_msa():
    individual_metrics = []
    aggregated_metrics = []
    errors_log = []

    fasta_files = sorted(list(FASTA_DIR.glob("*.fasta")))
    for fasta_file in tqdm(fasta_files, desc="Aligning and generating PWMs"):
        current_file_stem = fasta_file.stem
        msa_file = fasta_file.with_suffix(".aln")

        # 1. Generate MSA using MAFFT
        if not msa_file.exists():
            command = f"module load mafft/7.305 && mafft --auto {str(fasta_file)}"
            try:
                with open(msa_file, "w") as out:
                    subprocess.run(command, shell=True, check=True, stdout=out, stderr=subprocess.DEVNULL)
            except Exception as e:
                errors_log.append({"file_name": current_file_stem, "error_loc": "mafft run", "error": str(e)})
                continue

        # 2. Load the Alignment
        try:
            alignment = AlignIO.read(msa_file, "fasta")
            for record in alignment:
                record.seq = record.seq.upper()
        except Exception as e:
            errors_log.append({"file_name": current_file_stem, "error_loc": "read alignment", "error": str(e)})
            continue

        if len(alignment) < 2:
            continue

        # 3. Create Motif object & Consensus
        m = motifs.create(alignment)
        try:
            consensus_seq = m.counts.calculate_consensus(identity=0.5)
        except Exception as e:
            errors_log.append({"file_name": current_file_stem, "error_loc": "calc consensus", "error": str(e)})
            continue

        consensus_str = str(consensus_seq)

        # optimization: Convert consensus to byte array once for fast vectorized comparisons
        c_arr = np.frombuffer(consensus_str.encode(), dtype='S1')
        identities = []

        # 4. Calculate Divergence Metrics
        for record in alignment:
            header_parts = record.description.split("|||")
            gbk_name = header_parts[0]
            element_type = header_parts[1]
            element_name = header_parts[2]
            file_name = header_parts[3]

            seq_str = str(record.seq)
            s_arr = np.frombuffer(seq_str.encode(), dtype='S1')

            # Vectorized character comparison
            valid_mask = (s_arr != b'-') & (c_arr != b'-')
            matches = np.sum(s_arr[valid_mask] == c_arr[valid_mask])
            valid_positions = np.sum(valid_mask)

            ind_identity = matches / valid_positions if valid_positions > 0 else 0
            identities.append(ind_identity)

            individual_metrics.append({
                "element_type": element_type,
                "element_name": element_name,
                "file_name": file_name,
                "gbk_name": gbk_name,
                "identity_to_consensus": ind_identity
            })

        aggregated_metrics.append({
            "element_type": element_type,  # uses the last extracted type/name, valid as they are grouped by file
            "element_name": element_name,
            "file_name": file_name,
            "consensus_seq": consensus_str,
            "avg_identity": float(np.mean(identities)),
            "n_instances": len(alignment),
            "alignment_length": alignment.get_alignment_length()
        })

    # Save outputs
    pl.DataFrame(individual_metrics).write_parquet(OUT_INDIVIDUAL_DIVERGENCE)
    pl.DataFrame(aggregated_metrics).write_parquet(OUT_AVERAGE_DIVERGENCE)
    pl.DataFrame(errors_log).write_csv(MSA_ERROR_LOG, separator="\t")


def get_representative_sequence(plasmid_citations: pl.DataFrame, flank_size: int) -> None: 
    print(f"Extracting representative sequences with {flank_size}bp flanks...") 

    pmid_map = dict(zip(plasmid_citations["gbk_name"].to_list(), plasmid_citations["citing_pmids"].to_list())) 
    element_records = defaultdict(list) 

    for record in tqdm(SeqIO.parse(COMBINED_GBK, "genbank"), total=N_PLASMIDS): 
        L = len(record.seq) 
        gbk_name = record.name 
        pmid = pmid_map.get(gbk_name, None) 
        seq_str = str(record.seq) 

        for feat in record.features: 
            if feat.type == 'source' or not feat.location: 
                continue 

            e_type = feat.type 
            e_name = extract_feature_name(feat) 

            # 1. Collect all component indices of the feature body 
            parts = [] 
            part_indices_set = set() 
            for p in feat.location.parts: 
                p_s, p_e = int(p.start), int(p.end) 
                if p_s > p_e:  # wrapping part 
                    parts.append((p_s, L)) 
                    parts.append((0, p_e)) 
                    part_indices_set.update(range(p_s, L)) 
                    part_indices_set.update(range(0, p_e)) 
                else: 
                    parts.append((p_s, p_e)) 
                    part_indices_set.update(range(p_s, p_e)) 

            # 2. Sort parts and merge adjacent/overlapping segments 
            parts.sort(key=lambda x: x[0]) 
            merged_parts = [parts[0]] 
            for p_s, p_e in parts[1:]: 
                prev_s, prev_e = merged_parts[-1] 
                if p_s <= prev_e: 
                    merged_parts[-1] = (prev_s, max(prev_e, p_e)) 
                else: 
                    merged_parts.append((p_s, p_e)) 

            # 3. Find largest gap to define true genomic boundaries 
            max_gap = -1 
            max_gap_idx = -1 
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

            # 4. Construct indices for the entire span (including Gaps / Flanks) 
            if genomic_start <= genomic_end: 
                span_idx = list(range(genomic_start, genomic_end)) 
            else: 
                span_idx = list(range(genomic_start, L)) + list(range(0, genomic_end)) 

            left_flank_indices = [(genomic_start - flank_size + i) % L for i in range(flank_size)] 
            right_flank_indices = [(genomic_end + i) % L for i in range(flank_size)] 
            full_indices = left_flank_indices + span_idx + right_flank_indices 

            # Calculate exact plasmid positional coordinates of the fully flanked window
            seq_start = (genomic_start - flank_size) % L
            seq_end = (genomic_end + flank_size) % L

            if seq_start < seq_end:
                plasmid_position = f"{seq_start}-{seq_end}"
            else:
                # Sequence wraps around the circular origin (two intervals required for slice)
                plasmid_position = f"{seq_start}-{L},{0}-{seq_end}"

            # Safely capture strand direction
            strand = feat.location.strand if feat.location.strand is not None else 1

            # 5. Build string with case demarcations (Body = Upper, Gap / Flank = Lower) 
            chars = [] 
            for i in full_indices: 
                c = seq_str[i] 
                if i in part_indices_set: 
                    chars.append(c.upper()) 
                else: 
                    chars.append(c.lower()) 
            flanked_seq = "".join(chars) 

            # 6. Strand adjustments (Native Seq Reverse Complement preserves case demarcations) 
            if strand == -1: 
                flanked_seq = str(Seq(flanked_seq).reverse_complement()) 

            # 7. Extract 0-indexed [start:end] intervals relative to the saved sequence string 
            upper_intervals = [] 
            start_idx = None 
            for i, char in enumerate(flanked_seq): 
                if char.isupper(): 
                    if start_idx is None: 
                        start_idx = i 
                else: 
                    if start_idx is not None: 
                        upper_intervals.append(f"{start_idx}-{i}") 
                        start_idx = None 

            # Catch trailing feature at the very end 
            if start_idx is not None: 
                upper_intervals.append(f"{start_idx}-{len(flanked_seq)}") 

            intervals_str = ",".join(upper_intervals) 

            # Ensure PMID is processed as a list cleanly 
            pmid_list = [] 
            if pmid and pmid == pmid: # Checks for non-null / non-NaN 
                if isinstance(pmid, str): 
                    pmid_list = [p.strip() for p in pmid.split(",") if p.strip()] 
                else: 
                    pmid_list = list(pmid) 

            element_records[(e_type, e_name)].append({ 
                'seq': flanked_seq, 
                'intervals': intervals_str, 
                'gbk': gbk_name, 
                'pmid_list': pmid_list,
                'plasmid_position': plasmid_position,
                'strand': strand
            }) 

    # Analyze frequencies and write representatives 
    with open(REP_FASTA_OUT, "w") as out_fasta: 
        for (e_type, e_name), instances in element_records.items(): 
            if not instances: 
                continue 

            # Total baseline metrics for this element 
            total_plasmids = len(instances) 
            all_element_pmids = set() 

            # Group instances by exact sequence and aggregate global PMIDs 
            seq_to_instances = defaultdict(list) 
            for inst in instances: 
                seq_to_instances[inst['seq']].append(inst) 
                all_element_pmids.update(inst['pmid_list']) 

            total_element_citations = len(all_element_pmids) 

            # Evaluate metrics for each unique sequence 
            seq_metrics = [] 
            for seq, seq_insts in seq_to_instances.items(): 
                seq_pmids = set() 
                for inst in seq_insts: 
                    seq_pmids.update(inst['pmid_list']) 

                n_citations = len(seq_pmids) 
                n_plasmids = len(set([inst['gbk'] for inst in seq_insts])) 

                plasmid_frequency = len(seq_insts) / total_plasmids 
                citation_frequency = (n_citations / total_element_citations) if total_element_citations > 0 else 0.0 

                seq_metrics.append({ 
                    'seq': seq, 
                    'n_plasmids': n_plasmids, 
                    'n_citations': n_citations, 
                    'plasmid_frequency': plasmid_frequency, 
                    'citation_frequency': citation_frequency, 
                    'intervals': seq_insts[0]['intervals'],
                    'repres_plasmid_gbk': seq_insts[0]['gbk'],
                    'plasmid_position': seq_insts[0]['plasmid_position'],
                    'strand': seq_insts[0]['strand'],
                })

            # Sort by citation_frequency first (descending), then plasmid_frequency as tie-breaker 
            seq_metrics.sort(key=lambda x: (x['citation_frequency'], x['plasmid_frequency']), reverse=True) 
            rep = seq_metrics[0] 

            # Header includes all the fields
            header = f">{e_type}|{e_name}|{flank_size}|{rep['n_plasmids']}|{rep['n_citations']}|{rep['plasmid_frequency']:.4f}|{rep['citation_frequency']:.4f}|{rep['intervals']}|{rep['repres_plasmid_gbk']}|{rep['plasmid_position']}|{rep['strand']}"
            out_fasta.write(f"{header}\n{rep['seq']}\n") 

    print(f"Representatives saved to {REP_FASTA_OUT}")


if __name__ == "__main__":
    plasmid_citations = (
        pl.read_csv(ADDGENE_DIR / "mammalian_plasmids.tsv", separator="\t")
        .join(pl.read_parquet(ADDGENE_DIR / "citations_addgene.parquet"), on="plasmid_id")
        .join(pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_statistics.parquet")[["sequence_id", "gbk_name"]], on="sequence_id")
    )
    N_PLASMIDS = plasmid_citations.height

    # extract_subtypes_to_fasta()  # 1 min
    # make_and_analyze_msa()  # 7.5 hours (Gold-6242 CPU) 
    get_representative_sequence(plasmid_citations, REPRESENT_FLANK_SIZE)  # 8 min

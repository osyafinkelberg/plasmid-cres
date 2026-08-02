import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from Bio import AlignIO, SeqIO, motifs
from Bio.Seq import Seq
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
ADDGENE_DIR = DATA_DIR / "addgene"
COMBINED_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"
REP_FASTA_OUT = ADDGENE_DIR / "element_representative_sequences.fasta"

ALIGN_DIR = DATA_DIR / "addgene_alignments"

FASTA_DIR = ALIGN_DIR / "fasta"
OUT_UNIQUE_SEQUENCE_IDS = ALIGN_DIR / "element_unique_sequence_ids.parquet"
OUT_INDIVIDUAL_DIVERGENCE = ALIGN_DIR / "element_individual_divergence.parquet"
OUT_AVERAGE_DIVERGENCE = ALIGN_DIR / "element_average_divergence.parquet"
MSA_ERROR_LOG = ALIGN_DIR / "element_msa_errors.tsv"

FASTA_CDS_DIR = ALIGN_DIR / "fasta_cds"
OUT_UNIQUE_SEQUENCE_IDS_CDS = ALIGN_DIR / "element_unique_sequence_ids_cds.parquet"
OUT_INDIVIDUAL_DIVERGENCE_CDS = ALIGN_DIR / "element_individual_divergence_cds.parquet"
OUT_AVERAGE_DIVERGENCE_CDS = ALIGN_DIR / "element_average_divergence_cds.parquet"
MSA_ERROR_LOG_CDS = ALIGN_DIR / "element_msa_errors_cds.tsv"

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

            sequence = str(feat.extract(record.seq)).upper()

            if len(sequence) < SEQ_LEN_THRESH:
                continue

            position = [[int(p.start), int(p.end)] for p in feat.location.parts]
            strand = feat.location.strand if feat.location.strand is not None else 1
            seq_collections[sanitized_id].append((sequence, record.name, element_type, element_name, position, strand))

    unique_id_rows = []
    for sid, entries in seq_collections.items():
        element_type = entries[0][2]
        element_name = entries[0][3]

        # Group by unique sequence, preserving first-seen order
        seq_to_gbk_names = defaultdict(list)
        for sequence, gbk_name, _, _, position, strand in entries:
            seq_to_gbk_names[sequence].append((gbk_name, position, strand))

        fasta_lines = []
        for idx, (seq, instances) in enumerate(seq_to_gbk_names.items()):
            unique_id = f"{sid}|||{idx}"
            gbk_names = [inst[0] for inst in instances]
            positions = [inst[1] for inst in instances]
            strands = [inst[2] for inst in instances]
            fasta_lines.append(f">{unique_id}\n{seq}\n")
            unique_id_rows.append({
                "element_type": element_type,
                "element_name": element_name,
                "unique_id": unique_id,
                "sanitized_name": sid,
                "gbk_names": gbk_names,
                "positions": positions,
                "strands": strands,
                "n_instances": len(gbk_names)
            })

        with open(FASTA_DIR / f"{sid}.fasta", "w") as f:
            f.writelines(fasta_lines)

    pl.DataFrame(unique_id_rows).write_parquet(OUT_UNIQUE_SEQUENCE_IDS)


def extract_cds_aa_to_fasta() -> None:
    if FASTA_CDS_DIR.exists():
        print("CDS amino acid sequences are already extracted (directory exists)")
        return

    FASTA_CDS_DIR.mkdir(parents=True)

    # Build mapping from (sanitized_name, gbk_name) -> list of nt unique_ids for cross-referencing
    nt_gbk_to_uid = defaultdict(list)
    for row in pl.read_parquet(OUT_UNIQUE_SEQUENCE_IDS).filter(pl.col("element_type") == "CDS").iter_rows(named=True):
        for gbk_name in set(row["gbk_names"]):
            nt_gbk_to_uid[(row["sanitized_name"], gbk_name)].append(row["unique_id"])

    # optimization: collect in memory instead of keeping thousands of file handles open
    seq_collections = defaultdict(list)

    for record in tqdm(SeqIO.parse(COMBINED_GBK, "genbank"), total=N_PLASMIDS):
        for feat in record.features:
            if feat.type != 'CDS':
                continue

            element_type = feat.type
            element_name = extract_feature_name(feat)
            sanitized_id = f"{sanitize_filename(element_type)}__{sanitize_filename(element_name)}"

            nt_sequence = str(feat.extract(record.seq)).upper()

            if len(nt_sequence) < SEQ_LEN_THRESH:
                continue

            # Translate to amino acid sequence
            try:
                aa_sequence = str(Seq(nt_sequence).translate(to_stop=True))
            except Exception:  # noqa: BLE001, S112
                continue

            if not aa_sequence:
                continue

            position = [[int(p.start), int(p.end)] for p in feat.location.parts]
            strand = feat.location.strand if feat.location.strand is not None else 1
            seq_collections[sanitized_id].append((aa_sequence, record.name, element_type, element_name, position, strand))

    unique_id_rows = []
    for sid, entries in seq_collections.items():
        element_type = entries[0][2]
        element_name = entries[0][3]

        # Group by unique sequence, preserving first-seen order
        seq_to_gbk_names = defaultdict(list)
        for aa_sequence, gbk_name, _, _, position, strand in entries:
            seq_to_gbk_names[aa_sequence].append((gbk_name, position, strand))

        fasta_lines = []
        for idx, (seq, instances) in enumerate(seq_to_gbk_names.items()):
            unique_id = f"{sid}|||aa_{idx}"
            gbk_names = [inst[0] for inst in instances]
            positions = [inst[1] for inst in instances]
            strands = [inst[2] for inst in instances]

            # Find all corresponding nucleotide unique IDs (multiple NT seqs may share an AA seq)
            seen_nuc_ids = set()
            unique_nuc_ids = []
            for gbk_name in set(gbk_names):
                for nuc_id in nt_gbk_to_uid.get((sid, gbk_name), []):
                    if nuc_id not in seen_nuc_ids:
                        unique_nuc_ids.append(nuc_id)
                        seen_nuc_ids.add(nuc_id)

            fasta_lines.append(f">{unique_id}\n{seq}\n")
            unique_id_rows.append({
                "element_type": element_type,
                "element_name": element_name,
                "unique_id": unique_id,
                "sanitized_name": sid,
                "gbk_names": gbk_names,
                "positions": positions,
                "strands": strands,
                "n_instances": len(gbk_names),
                "unique_nuc_ids": unique_nuc_ids,
                "n_unique_nuc_ids": len(unique_nuc_ids),
            })

        with open(FASTA_CDS_DIR / f"{sid}.fasta", "w") as f:
            f.writelines(fasta_lines)

    pl.DataFrame(unique_id_rows).write_parquet(OUT_UNIQUE_SEQUENCE_IDS_CDS)


def make_and_analyze_msa():
    individual_metrics = []
    aggregated_metrics = []
    errors_log = []

    # Load unique sequence ID metadata for lookup
    uid_df = pl.read_parquet(OUT_UNIQUE_SEQUENCE_IDS)
    uid_meta = {
        row["unique_id"]: (row["element_type"], row["element_name"], row["n_instances"])
        for row in uid_df.iter_rows(named=True)
    }

    fasta_files = sorted(FASTA_DIR.glob("*.fasta"))
    for fasta_file in tqdm(fasta_files, desc="Aligning and generating PWMs"):
        current_file_stem = fasta_file.stem
        msa_file = fasta_file.with_suffix(".aln")

        # 1. Generate MSA using MAFFT
        if not msa_file.exists():
            command = f"module load mafft/7.305 && mafft --auto {fasta_file!s}"
            try:
                with open(msa_file, "w") as out:
                    subprocess.run(command, shell=True, check=True, stdout=out, stderr=subprocess.DEVNULL)
            except Exception as e:  # noqa: BLE001
                errors_log.append({"file_name": current_file_stem, "error_loc": "mafft run", "error": str(e)})
                continue
            finally:
                if msa_file.exists() and msa_file.stat().st_size == 0:
                    msa_file.unlink()

        # 2. Load the Alignment
        try:
            alignment = AlignIO.read(msa_file, "fasta")
            for record in alignment:
                record.seq = record.seq.upper()
        except Exception as e:  # noqa: BLE001
            errors_log.append({"file_name": current_file_stem, "error_loc": "read alignment", "error": str(e)})
            continue
        finally:
            if msa_file.exists() and msa_file.stat().st_size == 0:
                msa_file.unlink()

        if len(alignment) < 2:
            continue

        # 3. Create Motif object & Consensus
        m = motifs.create(alignment)
        try:
            consensus_seq = m.counts.calculate_consensus(identity=0.5)
        except Exception as e:  # noqa: BLE001
            errors_log.append({"file_name": current_file_stem, "error_loc": "calc consensus", "error": str(e)})
            continue
        finally:
            if msa_file.exists() and msa_file.stat().st_size == 0:
                msa_file.unlink()

        consensus_str = str(consensus_seq)

        # optimization: Convert consensus to byte array once for fast vectorized comparisons
        c_arr = np.frombuffer(consensus_str.encode(), dtype='S1')
        identities = []

        # 4. Calculate Divergence Metrics
        n_instances_total = 0
        for record in alignment:
            unique_id = record.description
            element_type, element_name, n_instances = uid_meta[unique_id]
            file_name = unique_id.split("|||")[0]

            seq_str = str(record.seq)
            s_arr = np.frombuffer(seq_str.encode(), dtype='S1')

            # Vectorized character comparison
            valid_mask = (s_arr != b'-') & (c_arr != b'-')
            matches = np.sum(s_arr[valid_mask] == c_arr[valid_mask])
            valid_positions = np.sum(valid_mask)

            ind_identity = matches / valid_positions if valid_positions > 0 else 0
            identities.append(ind_identity)
            n_instances_total += n_instances

            individual_metrics.append({
                "element_type": element_type,
                "element_name": element_name,
                "file_name": file_name,
                "unique_id": unique_id,
                "n_instances": n_instances,
                "identity_to_consensus": ind_identity
            })

        aggregated_metrics.append({
            "element_type": element_type,  # uses the last extracted type/name, valid as they are grouped by file
            "element_name": element_name,
            "file_name": file_name,
            "consensus_seq": consensus_str,
            "avg_identity": float(np.mean(identities)),
            "n_instances_total": n_instances_total,
            "n_instances_unique": len(alignment),
            "alignment_length": alignment.get_alignment_length()
        })

    # Save outputs
    pl.DataFrame(individual_metrics).write_parquet(OUT_INDIVIDUAL_DIVERGENCE)
    pl.DataFrame(aggregated_metrics).write_parquet(OUT_AVERAGE_DIVERGENCE)
    pl.DataFrame(errors_log).write_csv(MSA_ERROR_LOG, separator="\t")


def make_and_analyze_msa_cds():
    individual_metrics = []
    aggregated_metrics = []
    errors_log = []

    # Load unique AA sequence ID metadata for lookup
    uid_aa_meta = {
        row["unique_id"]: (row["element_type"], row["element_name"], row["n_instances"], row["unique_nuc_ids"], row["n_unique_nuc_ids"])
        for row in pl.read_parquet(OUT_UNIQUE_SEQUENCE_IDS_CDS).iter_rows(named=True)
    }

    # Load all CDS NT sequences into memory for codon-aware divergence computation
    uid_nt_seq = {}
    for fasta_nt_file in sorted(FASTA_DIR.glob("CDS__*.fasta")):
        for record in SeqIO.parse(fasta_nt_file, "fasta"):
            uid_nt_seq[record.id] = str(record.seq).upper()

    fasta_cds_files = sorted(FASTA_CDS_DIR.glob("*.fasta"))
    for fasta_cds_file in tqdm(fasta_cds_files, desc="Aligning CDS and generating PWMs"):
        current_file_stem = fasta_cds_file.stem

        # 1. Amino acid alignment (sequences from FASTA_CDS_DIR)
        aa_aln_file = fasta_cds_file.with_suffix(".aln")

        if not aa_aln_file.exists():
            command = f"module load mafft/7.305 && mafft --auto {fasta_cds_file!s}"
            try:
                with open(aa_aln_file, "w") as out:
                    subprocess.run(command, shell=True, check=True, stdout=out, stderr=subprocess.DEVNULL)
            except Exception as e:  # noqa: BLE001
                errors_log.append({"file_name": current_file_stem, "error_loc": "mafft aa run", "error": str(e)})
            finally:
                if aa_aln_file.exists() and aa_aln_file.stat().st_size == 0:
                    aa_aln_file.unlink()

        if not aa_aln_file.exists():
            continue

        # 2. Load the Alignment
        try:
            alignment = AlignIO.read(aa_aln_file, "fasta")
            for record in alignment:
                record.seq = record.seq.upper()
        except Exception as e:  # noqa: BLE001
            errors_log.append({"file_name": current_file_stem, "error_loc": "read aa alignment", "error": str(e)})
            continue

        if len(alignment) < 2:
            continue

        # 3. Compute AA consensus manually (motifs module is DNA-specific)
        try:
            aln_len = alignment.get_alignment_length()
            consensus_chars = []
            for i in range(aln_len):
                col = [str(rec.seq[i]) for rec in alignment]
                non_gap = [c for c in col if c != '-']
                if not non_gap:
                    consensus_chars.append('-')
                    continue
                aa_counts = defaultdict(int)
                for c in non_gap:
                    aa_counts[c] += 1
                best = max(aa_counts, key=aa_counts.get)
                consensus_chars.append(best if aa_counts[best] / len(non_gap) >= 0.5 else 'X')
            consensus_str = ''.join(consensus_chars)
        except Exception as e:  # noqa: BLE001
            errors_log.append({"file_name": current_file_stem, "error_loc": "calc aa consensus", "error": str(e)})
            continue

        # optimization: Convert consensus to byte array once for fast vectorized comparisons
        c_arr = np.frombuffer(consensus_str.encode(), dtype='S1')
        identities = []
        n_instances_total = 0

        # 4. Calculate Divergence Metrics
        for record in alignment:
            unique_id = record.description
            element_type, element_name, n_instances, unique_nuc_ids, n_unique_nuc_ids = uid_aa_meta[unique_id]
            file_name = unique_id.split("|||")[0]

            seq_str = str(record.seq)
            s_arr = np.frombuffer(seq_str.encode(), dtype='S1')

            # Vectorized character comparison (AA identity to consensus)
            valid_mask = (s_arr != b'-') & (c_arr != b'-')
            matches = np.sum(s_arr[valid_mask] == c_arr[valid_mask])
            valid_positions = np.sum(valid_mask)

            aa_identity = matches / valid_positions if valid_positions > 0 else 0
            identities.append(aa_identity)
            n_instances_total += n_instances

            # 5. Codon-aware NT divergence: NT seqs encoding this AA are implicitly aligned
            # (coding length = 3 * aa_len, identical across all seqs encoding the same AA)
            aa_len = int(np.sum(s_arr != b'-'))
            coding_len = aa_len * 3

            nt_coding_seqs = []
            for nuc_id in unique_nuc_ids:
                nt_seq = uid_nt_seq.get(nuc_id)
                if nt_seq is not None and len(nt_seq) >= coding_len:
                    nt_coding_seqs.append(nt_seq[:coding_len])

            if not nt_coding_seqs:
                nuc_avg_identity = None
            elif len(nt_coding_seqs) == 1:
                nuc_avg_identity = 1.0
            else:
                # Build position-wise consensus of the coding NT sequences
                nt_consensus_chars = []
                for i in range(coding_len):
                    nt_col_counts = defaultdict(int)
                    for s in nt_coding_seqs:
                        nt_col_counts[s[i]] += 1
                    best_nt = max(nt_col_counts, key=nt_col_counts.get)
                    nt_consensus_chars.append(best_nt if nt_col_counts[best_nt] / len(nt_coding_seqs) >= 0.5 else 'N')
                nt_consensus = ''.join(nt_consensus_chars)

                # optimization: Convert NT consensus to byte array once for fast vectorized comparisons
                nt_c_arr = np.frombuffer(nt_consensus.encode(), dtype='S1')
                nt_identities = []
                for nt_seq in nt_coding_seqs:
                    nt_s_arr = np.frombuffer(nt_seq.encode(), dtype='S1')
                    nt_identities.append(float(np.mean(nt_s_arr == nt_c_arr)))
                nuc_avg_identity = float(np.mean(nt_identities))

            individual_metrics.append({
                "element_type": element_type,
                "element_name": element_name,
                "file_name": file_name,
                "unique_id": unique_id,
                "n_instances": n_instances,
                "n_unique_nuc_ids": n_unique_nuc_ids,
                "aa_identity_to_consensus": aa_identity,
                "nuc_avg_identity": nuc_avg_identity
            })

        aggregated_metrics.append({
            "element_type": element_type,  # uses the last extracted type/name, valid as they are grouped by file
            "element_name": element_name,
            "file_name": file_name,
            "consensus_seq": consensus_str,
            "avg_identity": float(np.mean(identities)),
            "n_instances_total": n_instances_total,
            "n_instances_unique": len(alignment),
            "alignment_length": alignment.get_alignment_length()
        })

    # Save outputs
    pl.DataFrame(individual_metrics).write_parquet(OUT_INDIVIDUAL_DIVERGENCE_CDS)
    pl.DataFrame(aggregated_metrics).write_parquet(OUT_AVERAGE_DIVERGENCE_CDS)
    pl.DataFrame(errors_log).write_csv(MSA_ERROR_LOG_CDS, separator="\t")


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
                    part_indices_set.update(range(p_e)) 
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
                span_idx = list(range(genomic_start, L)) + list(range(genomic_end)) 

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
            if pmid and pmid == pmid:  # checks for non-null / non-NaN  # noqa: PLR0124
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
                n_plasmids = len({inst['gbk'] for inst in seq_insts}) 

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

    extract_subtypes_to_fasta()  # 1.5 min
    extract_cds_aa_to_fasta()  # 1.5 min
    make_and_analyze_msa()  # 17 min (Gold-6242 CPU)
    make_and_analyze_msa_cds()  # 4 min (Gold-6242 CPU)
    # get_representative_sequence(plasmid_citations, REPRESENT_FLANK_SIZE)  # 8 min

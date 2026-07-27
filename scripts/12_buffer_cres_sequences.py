import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import polars as pl
from Bio import SeqIO
from tqdm import tqdm

sys.path.insert(0, "..")
from plasmidtools.search import PlasmidSearchSuffixArray

# --- CONFIGURATION ---
DATA_DIR = Path().cwd().parent / "data"
ADDGENE_DIR = DATA_DIR / "addgene"
MANUAL_DIR = DATA_DIR / "manual_annotations"

BUFFER_CRES_FILE = ADDGENE_DIR / "buffer_cres.parquet"
PLASMID_FASTA = ADDGENE_DIR / "mammalian_plasmids_wrapped.fasta"
PLASMID_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"
PLASMIDS_TSV = ADDGENE_DIR / "mammalian_plasmids.tsv"
PLASMID_CITATIONS = ADDGENE_DIR / "citations_addgene.parquet"
GLOBAL_HITS_OUT = ADDGENE_DIR / "buffer_cres_sequences.parquet"

# --- PARAMETERS ---
CRE_ANCHOR_SIZE = 200
TSS_ANCHOR_SIZE = 650
ALLOWED_SHIFT = 20  # max base pairs of shift to consider sequences identical
REV_COMP_TABLE = str.maketrans("ATCGN", "TAGCN")


def get_canonical(seq: str) -> tuple[str, bool]:
    """Returns the lexicographically smaller of a sequence and its reverse complement."""
    rev_seq = seq[::-1].translate(REV_COMP_TABLE)
    if seq <= rev_seq:
        return seq, False  # False means it was already canonical (fwd)
    return rev_seq, True   # True means the reverse complement is the canonical


def load_plasmid_sequences(gbk_path: Path) -> dict[str, str]:
    """Loads all plasmid sequences into memory for fast anchor extraction."""
    seqs = {}
    for record in SeqIO.parse(gbk_path, "genbank"):
        gbk_name = record.name or record.id
        seqs[gbk_name] = str(record.seq).upper()
    return seqs


def load_plasmid_features(gbk_path: Path) -> dict[str, list[dict]]:
    """Loads feature annotations for all plasmids to evaluate functional overlap."""
    features_dict = {}
    for record in SeqIO.parse(gbk_path, "genbank"):
        gbk_name = record.name or record.id
        feats = []
        for f in record.features:
            if f.type == "source":
                continue
            name = f.qualifiers.get("label", f.qualifiers.get("note", f.qualifiers.get("gene", [""])))[0]
            if not name:
                name = "unnamed"
            feat_name = f"{f.type}:{name}"
            
            for part in f.location.parts:
                feats.append({
                    "start": int(part.start),
                    "end": int(part.end),
                    "name": feat_name
                })
        features_dict[gbk_name] = feats
    return features_dict


def build_metadata_maps() -> tuple[dict[str, set], dict[str, set]]:
    """Creates fast mappings from gbk_name to citing PMIDs and vector types."""
    plasmid_ids = pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_statistics.parquet")[["gbk_name", "sequence_id"]]
    plasmid_download = pl.read_csv(PLASMIDS_TSV, separator="\t")
    plasmid_citations = pl.read_parquet(PLASMID_CITATIONS)

    merged = (
        plasmid_ids
        .join(plasmid_download, on="sequence_id", how="left")
        .join(plasmid_citations, on="plasmid_id", how="left")
        .select(["gbk_name", "vector_type", "citing_pmids"])
    )

    # 1. Group citations
    grouped_cit = merged.explode("citing_pmids").drop_nulls("citing_pmids").group_by("gbk_name").agg(
        pl.col("citing_pmids").unique().alias("pmids")
    )
    cit_dict = {row["gbk_name"]: set(row["pmids"]) for row in grouped_cit.iter_rows(named=True)}

    # 2. Group vector types
    grouped_vt = merged.select(["gbk_name", "vector_type"]).drop_nulls("vector_type").unique()
    vt_dict = defaultdict(set)
    for row in grouped_vt.iter_rows(named=True):
        # Addgene vector_types are sometimes comma-separated strings
        vt_str = row["vector_type"]
        if isinstance(vt_str, str):
            for vt in vt_str.split(","):
                vt_dict[row["gbk_name"]].add(vt.strip())
                
    return cit_dict, dict(vt_dict)


def get_known_cres() -> set[str]:
    """Loads a set of manually annotated functional mammalian CREs."""
    promoters = pl.read_csv(MANUAL_DIR / "addgene_promoters_and_enhancers.csv")
    known_cres = [
        f"{t}:{n}" for t, n in 
        promoters.filter(pl.col("active_in_mammalian_cells") & (pl.col("RNA_polymerase") == "RNA Pol II"))[["element_type", "element_name"]].rows()
    ]
    return set(known_cres)


def cluster_sequences_by_overlap(unique_seqs: list[str], min_overlap: int) -> list[list[str]]:
    """
    Groups sequences that are shifted by up to (L - min_overlap) base pairs using a K-mer graph.
    Evaluates both forward and reverse-complement K-mers to resolve canonicalization flips.
    """
    if not unique_seqs:
        return []

    L = len(unique_seqs[0])
    k = min_overlap

    # If sequences are exactly the minimum overlap length (or smaller), exact match is required
    if L <= k:
        return [[s] for s in unique_seqs]

    kmer_to_seqs = defaultdict(list)

    # 1. Map K-mers (Forward AND Reverse-Complement)
    for idx, seq in enumerate(unique_seqs):
        # Forward K-mers
        for i in range(L - k + 1):
            kmer = seq[i:i+k]
            kmer_to_seqs[kmer].append(idx)

        # Reverse-complement K-mers
        rc_seq = seq[::-1].translate(REV_COMP_TABLE)
        for i in range(L - k + 1):
            kmer = rc_seq[i:i+k]
            kmer_to_seqs[kmer].append(idx)

    # 2. Build adjacency graph
    G = nx.Graph()
    G.add_nodes_from(range(len(unique_seqs)))
    for seq_indices in kmer_to_seqs.values():
        unique_indices = list(set(seq_indices))  # Deduplicate to prevent self-loops from palindromes
        if len(unique_indices) > 1:
            # Connect all sequences that share this exact K-mer (in either orientation)
            for i in range(len(unique_indices) - 1):
                G.add_edge(unique_indices[i], unique_indices[i+1])

    # 3. Extract connected components
    clusters = []
    for comp in nx.connected_components(G):
        clusters.append([unique_seqs[i] for i in comp])

    return clusters


def group_buffer_cre_sequences() -> None:
    print("Loading buffer CREs and parsing plasmid sequences/features...")
    buffer_df = pl.read_parquet(BUFFER_CRES_FILE)
    plasmid_seqs = load_plasmid_sequences(PLASMID_GBK)
    plasmid_feats = load_plasmid_features(PLASMID_GBK)
    citation_map, vector_type_map = build_metadata_maps()
    known_cres_set = get_known_cres()
    
    api = PlasmidSearchSuffixArray(PLASMID_FASTA)

    cre_types = buffer_df["cre_type"].unique().to_list()
    final_records = []
    global_anchor_counter = 0

    for c_type in cre_types:
        print(f"\n--- Processing {c_type} ---")
        type_df = buffer_df.filter(pl.col("cre_type") == c_type)
        
        is_cre = c_type.startswith("CREST")
        anchor_len = CRE_ANCHOR_SIZE if is_cre else TSS_ANCHOR_SIZE
        min_overlap = anchor_len - ALLOWED_SHIFT

        # 1. Extract Sequences & Exact Match Deduplication
        exact_seq_to_ids = defaultdict(list)
        cre_id_to_features = {}
        
        for row in tqdm(type_df.iter_rows(named=True), total=type_df.height, desc="Extracting Anchors"):
            gbk = row["gbk_name"]
            if gbk not in plasmid_seqs:
                continue

            seq = plasmid_seqs[gbk]
            L = len(seq)
            s, e = row["cre_position"]

            # Calculate midpoint
            if s <= e:
                mid = (s + e) // 2
            else:
                mid = ((s + e + L) // 2) % L

            # Extract anchor window
            r_anchor = anchor_len // 2
            start_idx = (mid - r_anchor) % L

            # Handle wrap-around sequence extraction
            if start_idx + anchor_len <= L:
                raw_anchor = seq[start_idx : start_idx + anchor_len]
            else:
                raw_anchor = seq[start_idx : L] + seq[0 : (start_idx + anchor_len) % L]

            # Group by exact Canonical sequence
            canonical_seq, _ = get_canonical(raw_anchor)
            exact_seq_to_ids[canonical_seq].append(row["cre_raw_id"])

            # Map raw ID to reduced feature names
            raw_feats = row.get("overlapped_features", [])
            if raw_feats is None:
                raw_feats = []
            elif isinstance(raw_feats, str):
                raw_feats = [raw_feats]

            cleaned_feats = []
            for f in raw_feats:
                if f.startswith(("putative_orf:", "putative_noncoding:")):
                    cleaned_feats.append(f.split(":")[0])
                else:
                    cleaned_feats.append(f)
            cre_id_to_features[row["cre_raw_id"]] = cleaned_feats

        unique_canonicals = list(exact_seq_to_ids.keys())
        print(f"Found {len(unique_canonicals)} exactly unique canonical sequences.")

        # 2. Shift Resolution via K-mer Graph
        print(f"Resolving {ALLOWED_SHIFT}bp shifts...")
        clusters = cluster_sequences_by_overlap(unique_canonicals, min_overlap)
        print(f"Reduced to {len(clusters)} biological sequence clusters.")

        # 3. FM-Index Search & Aggregation
        for cluster_seqs in tqdm(clusters, desc="Global FM-Index Search"):
            global_anchor_counter += 1
            anchor_id = f"ANC_{global_anchor_counter:06d}"

            # Merge original CRE IDs and unique features
            cre_raw_ids = []
            source_overlapped_features_set = set()
            for seq in cluster_seqs:
                ids = exact_seq_to_ids[seq]
                cre_raw_ids.extend(ids)
                for cid in ids:
                    source_overlapped_features_set.update(cre_id_to_features.get(cid, []))

            # Dictionary to deduplicate hits on the same plasmid
            # Keys: gbk_name, Values: list of (position, strand)
            plasmid_hits = defaultdict(list)

            # Extract the actual hit length to calculate coordinates correctly
            hit_len = len(cluster_seqs[0])

            # Query the union of all sequence variations
            for seq_variant in cluster_seqs:
                # Query forward and reverse complement variants (RC handled by api)
                hits = api.search_sequence(seq_variant)
                for h in hits:
                    strand_val = 1 if h["orientation"] == "fwd" else -1
                    plasmid_hits[h["gbk_name"]].append((h["position"], strand_val))

            # 4. Hit Deduplication, Overlap Parsing, & Citation Aggregation
            final_gbk_names = []
            final_positions = []
            final_strands = []
            final_is_buffer = []
            
            all_hit_overlapped_features = set()
            all_vector_types = set()
            
            buffer_gbks = set()
            total_buffer_citations = set()

            for gbk, hits_list in plasmid_hits.items():
                # Cluster hits within ~ALLOWED_SHIFT bp and take the first one
                hits_list.sort(key=lambda x: x[0])

                deduped_hits = []
                for pos, strand in hits_list:
                    if not deduped_hits or (pos - deduped_hits[-1][0]) > ALLOWED_SHIFT * 2:
                        deduped_hits.append((pos, strand))
                        
                p_len = len(plasmid_seqs[gbk])
                p_feats = plasmid_feats.get(gbk, [])

                for pos, strand in deduped_hits:
                    # Circular topology overlap check
                    hit_end = pos + hit_len
                    if hit_end <= p_len:
                        hit_ranges = [(pos, hit_end)]
                    else:
                        hit_ranges = [(pos, p_len), (0, hit_end % p_len)]
                        
                    hit_overlaps = []
                    for f in p_feats:
                        f_start, f_end, f_name = f["start"], f["end"], f["name"]
                        for hs, he in hit_ranges:
                            if hs < f_end and f_start < he: # 0-indexed interval intersection
                                hit_overlaps.append(f_name)
                                break
                    
                    is_buffer_hit = len(hit_overlaps) == 0

                    final_gbk_names.append(gbk)
                    final_positions.append(pos)
                    final_strands.append(strand)
                    final_is_buffer.append(is_buffer_hit)
                    
                    all_hit_overlapped_features.update(hit_overlaps)
                    if gbk in vector_type_map:
                        all_vector_types.update(vector_type_map[gbk])

                    if is_buffer_hit:
                        buffer_gbks.add(gbk)
                        if gbk in citation_map:
                            total_buffer_citations.update(citation_map[gbk])

            final_records.append({
                "anchor_id": anchor_id,
                "cre_type": c_type,
                "cre_raw_ids": cre_raw_ids,
                "cre_overlapped_features": list(source_overlapped_features_set),
                "gbk_names": final_gbk_names,
                "positions": final_positions,
                "strands": final_strands,
                "is_buffer": final_is_buffer,
                "n_plasmids": len(buffer_gbks),
                "n_citations": len(total_buffer_citations),
                "overlapped_features": list(all_hit_overlapped_features),
                "overlaps_known_cre": bool(all_hit_overlapped_features & known_cres_set),
                "vector_types": list(all_vector_types)
            })

    # 5. Output
    if final_records:
        out_df = pl.DataFrame(final_records)
        out_df = out_df.sort(["n_citations", "n_plasmids"], descending=[True, True])
        out_df.write_parquet(GLOBAL_HITS_OUT)
        print(f"\n Saved {out_df.height} unique global backbone anchors to {GLOBAL_HITS_OUT}")
    else:
        print("\nNo anchors generated.")


def sanity_check_test_source_inclusion():
    print("Loading data for sanity check...")
    df_raw = pl.read_parquet(BUFFER_CRES_FILE).select(["cre_raw_id", "gbk_name"])
    df_seq = pl.read_parquet(GLOBAL_HITS_OUT).select(["anchor_id", "cre_raw_ids", "gbk_names", "n_plasmids"])

    df_seq = df_seq.filter(pl.col("n_plasmids") < 2)

    # Create a fast lookup map: cre_raw_id -> source gbk_name
    raw_to_source = {row["cre_raw_id"]: row["gbk_name"] for row in df_raw.iter_rows(named=True)}

    missing_sources = []
    for row in df_seq.iter_rows(named=True):
        anchor_id = row["anchor_id"]
        found_gbks = set(row["gbk_names"])

        for raw_id in row["cre_raw_ids"]:
            source_gbk = raw_to_source.get(raw_id)
            if source_gbk and source_gbk not in found_gbks:
                missing_sources.append({
                    "anchor_id": anchor_id,
                    "cre_raw_id": raw_id,
                    "missing_source_gbk": source_gbk
                })
  
    if not missing_sources:
        print("Sanity Check Passed: All source plasmids were successfully mapped by the aligner.")
    else:
        print(f"Failed: Found {len(missing_sources)} missed source mappings.")
        for err in missing_sources[:10]:
            print(f" - {err['cre_raw_id']} from {err['missing_source_gbk']} missing in {err['anchor_id']}")


if __name__ == "__main__":
    group_buffer_cre_sequences()  # 12 min with Mappy: fuzzy, but incomplete (heuristic) search
                                  # 6 min with Suffix Array: exact, exhaustive search
    sanity_check_test_source_inclusion()

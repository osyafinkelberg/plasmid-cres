import concurrent.futures
import json
import os
from pathlib import Path

import polars as pl
import requests
from Bio import SeqIO
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

CUR_DIR = Path(__file__).resolve()
ADDGENE_DIR = CUR_DIR.parent.parent / "data/addgene"
ADDGENE_JSON_FILE = ADDGENE_DIR / "addgene_plasmids_with_sequences.json"
GBK_OUTPUT = ADDGENE_DIR / "mammalian_plasmids.gbk"
FASTA_OUTPUT = ADDGENE_DIR / "mammalian_plasmids_wrapped.fasta"
TSV_OUTPUT = ADDGENE_DIR / "mammalian_plasmids.tsv"

CIRCULAR_OVERLAP = 2000  # overlap of "wrapped" plasmid sequences in fasta file

load_dotenv()
TOKEN = os.environ.get("ADDGENE_TOKEN")
if not TOKEN:
    raise ValueError("ADDGENE_TOKEN environment variable is not set")


def download_single_plasmid(plasmid, session, master_file, progress_bar):
    v_types = plasmid.get('cloning', {}).get('vector_types', [])
    is_mammalian = any("mammalian" in vt.lower() for vt in v_types)

    if not is_mammalian:
        progress_bar.update(1)
        return None

    p_id = plasmid["id"]
    sequences_dict = plasmid.get("sequences", {})
    full_seq_entries = sequences_dict.get("public_addgene_full_sequences", [])
    res = {
        "plasmid_id": p_id,
        "plasmid_name": plasmid["name"],
        "vector_type": " ||| ".join(v_types),
        "backbone": plasmid["cloning"]["backbone"],
        "inserts": " ||| ".join(insert["name"] for insert in plasmid["inserts"]),
        "n_inserts": len(plasmid["inserts"])
    }

    if not full_seq_entries:
        progress_bar.update(1)
        res.update({"download_status": "no sequence ID"})
        return res

    seq_ids = [entry["genbank_url"].strip('/').split('/')[-1] for entry in full_seq_entries]
    best_seq_id = max(seq_ids)
    url = f'https://api.developers.addgene.org/download/genbank/{best_seq_id}/'

    try:
        response = session.get(url, headers={"Authorization": f"Token {TOKEN}"}, timeout=20)
        if response.status_code == 200:
            content = response.content.decode('utf-8', errors='ignore')
            if "LOCUS       . " in content:
                content = content.replace("LOCUS       . ", f"LOCUS       Addgene_{p_id:<8}")

            # Use a lock if writing to the same file from multiple threads
            # For simplicity here, we return the content to be written by the main thread
            res.update({"sequence_id": best_seq_id, "download_status": "200", "content": content})

        else:
            res.update({"sequence_id": best_seq_id, "download_status": str(response.status_code)})

    except Exception as e:  # noqa: BLE001
        res.update({"sequence_id": best_seq_id, "download_status": f"Error: {e!s}"})

    progress_bar.update(1)
    return res


def batch_download_vectors():
    with open(ADDGENE_JSON_FILE, 'r') as f:
        data = json.load(f)

    # Setup Session with Retries to handle the 'Connection Reset' errors
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=20))

    download_logs = []

    # Use ThreadPoolExecutor for concurrent I/O
    # Max_workers=10 is safe to avoid getting banned by the API
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # noqa: SIM117
        with tqdm(total=len(data["plasmids"]), desc="Concurrent Download") as pbar:
            futures = [executor.submit(download_single_plasmid, p, session, None, pbar) for p in data["plasmids"]]
            with open(GBK_OUTPUT, "wb") as master_file:
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res and "content" in res:
                        master_file.write(res["content"].encode('utf-8'))
                        master_file.write(b"\n")
                        # Remove content before saving to metadata list to save RAM
                        del res["content"]
                        download_logs.append(res)
                    elif res:
                        download_logs.append(res)

    # Save tracking data
    download_logs_df = pl.DataFrame(download_logs)[
        ["plasmid_id", "plasmid_name", "sequence_id", "vector_type", "backbone", "inserts", "download_status"]
    ]
    download_logs_df.write_csv(TSV_OUTPUT, separator="\t")


def create_circular_fasta_database() -> None:
    """
    Parses a GenBank file, linearly doubles the circular junction zones, 
    and saves the output to a FASTA file.
    """    
    with open(FASTA_OUTPUT, "w") as f_out:
        for record in SeqIO.parse(GBK_OUTPUT, "genbank"):
            gbk_name = str(record.name)
            seq_str = str(record.seq).upper()

            L = len(seq_str)
            if L == 0:
                continue

            repeats = (CIRCULAR_OVERLAP // L) + 1
            overlap_seq = (seq_str * repeats)[:CIRCULAR_OVERLAP]
            wrapped_seq = seq_str + overlap_seq
            
            # Append the true biological length to the header
            f_out.write(f">{gbk_name}|{L}\n{wrapped_seq}\n")


if __name__ == "__main__":
    # # Addgene GenBank download (run this only once)
    # batch_download_vectors()

    # # Plasmid sequences in fasta format (wrapping accounts for plasmid circularity when building sequence FM-index)
    create_circular_fasta_database()

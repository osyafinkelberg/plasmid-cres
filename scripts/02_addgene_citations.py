from pathlib import Path
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import json
import os
import polars as pl

# --- CONFIGURATION ---
CUR_DIR = Path(__file__).resolve()
ADDGENE_DIR = CUR_DIR.parent.parent / "data/addgene"
PLASMID_METADATA = ADDGENE_DIR / "mammalian_plasmids.gbk"
CHECKPOINT_FILE = ADDGENE_DIR / "citations_addgene.jsonl"
PARQUET_OUT = ADDGENE_DIR / "citations_addgene.parquet"

CONCURRENT_REQUESTS = 15  # Do not exceed 30
MAX_RETRIES = 3  # How many times to retry a failed fetch


async def fetch_citations(session: aiohttp.ClientSession, plasmid_id: int, semaphore: asyncio.Semaphore) -> dict:
    """
    Asynchronously fetches and parses the citation page for a single plasmid.
    """
    url = f"https://www.addgene.org/{plasmid_id}/citations/"
    
    # The semaphore ensures only N requests happen at exactly the same time
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=15) as response:
                    if response.status == 429:
                        # Rate limited: Exponential backoff
                        await asyncio.sleep(2 ** attempt)
                        continue
 
                    if response.status == 404:
                        # Page doesn't exist (valid empty result)
                        return {"plasmid_id": plasmid_id, "citing_pmids": [], "n_citations": 0, "status": "not_found"}
                        
                    response.raise_for_status()
                    html = await response.text()
                    
                    # Parse HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    pmid_links = soup.find_all('a', href=re.compile(r'(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/)(\d+)'))
                    
                    pmids = set()
                    for link in pmid_links:
                        match = re.search(r'/(\d+)/?$', link['href'])
                        if match:
                            pmids.add(match.group(1))
                            
                    return {
                        "plasmid_id": plasmid_id,
                        "citing_pmids": list(pmids),
                        "n_citations": len(pmids),
                        "status": "success"
                    }
                    
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"plasmid_id": plasmid_id, "citing_pmids": [], "n_citations": 0, "status": f"error: {str(e)}"}
                await asyncio.sleep(1)


def load_processed_ids() -> set[int]:
    """Reads the JSONL file to figure out which IDs are already done."""
    processed = set()
    if os.path.exists(CHECKPOINT_FILE):
        print(f"[*] Found existing checkpoint file '{CHECKPOINT_FILE}'.")
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed.add(int(record["plasmid_id"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"[*] Resuming pipeline. Skipping {len(processed)} existing entries.")
    return processed


async def main(all_plasmid_ids: list[int]):
    """
    Sets up the async session, handles the task queue, and writes to disk.
    """
    processed_ids = load_processed_ids()
    ids_to_process = [pid for pid in all_plasmid_ids if pid not in processed_ids]
    total_to_process = len(ids_to_process)

    if not ids_to_process:
        print("[*] All plasmids have already been processed.")
        return

    print(f"[*] Starting async collection for {total_to_process} plasmids...")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    # Use a single connection pool (session) for all requests
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:

        # Create a list of pending tasks
        tasks = [fetch_citations(session, pid, semaphore) for pid in ids_to_process]

        # Open file in append mode. As each task finishes, write it immediately.
        counter = 0
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
            for future in asyncio.as_completed(tasks):
                result = await future

                # Write to disk the moment it completes
                f.write(json.dumps(result) + "\n")
                f.flush()  # Force OS to write to disk immediately

                counter += 1
                if counter % 500 == 0 or counter == total_to_process:
                    print(f"[ Status ] Processed {counter} / {total_to_process} pending plasmids.")

    print("[*] Pipeline complete!")


if __name__ == "__main__":
    # Load Mammalian Expression dataset metadata
    download_logs = pl.read_csv(ADDGENE_DIR / "mammalian_plasmids.gbk", separator="\t")
    plasmid_ids = download_logs["plasmid_id"].to_list()

    asyncio.run(main(plasmid_ids))
    if os.path.exists(CHECKPOINT_FILE) and os.path.getsize(CHECKPOINT_FILE) > 0:
        addgene_citations_df = pl.read_ndjson(CHECKPOINT_FILE)
        addgene_citations_df.write_parquet(PARQUET_OUT)

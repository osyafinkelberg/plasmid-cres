import sys
from pathlib import Path
import h5py
import numpy as np
from Bio import SeqIO
import torch
from tqdm import tqdm

sys.path.insert(0, "/projectnb/vtrs/joseff/plasmids/data/puffin")
import puffin


# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
COMBINED_FA = ADDGENE_DIR / "mammalian_plasmids_wrapped.fasta"
CIRCULAR_OVERLAP = 2000  # overlap of "wrapped" plasmid sequences in fasta file (defined in `01_addgene_gbk_download.py`)
PUFFIN_FLANK = 325
PUFFIN_RAW_KEYS = [
    'Prediciton FANTOM_CAGE', 'Prediciton ENCODE_CAGE', 'Prediciton ENCODE_RAMPAGE', 'Prediciton GRO_CAP', 'Prediciton PRO_CAP',
    'Prediciton rev strand FANTOM_CAGE', 'Prediciton rev strand ENCODE_CAGE', 'Prediciton rev strand ENCODE_RAMPAGE', 'Prediciton rev strand GRO_CAP', 'Prediciton rev strand PRO_CAP'
]
PUFFIN_KEYS = [
    'FANTOM_CAGE fwd', 'ENCODE_CAGE fwd', 'ENCODE_RAMPAGE fwd', 'GRO_CAP fwd', 'PRO_CAP fwd',
    'FANTOM_CAGE rev', 'ENCODE_CAGE rev', 'ENCODE_RAMPAGE rev', 'GRO_CAP rev', 'PRO_CAP rev'
]
PREDS_OUT = ADDGENE_DIR / "mammalian_plasmids_puffin_preds.h5"
USE_CUDA = torch.cuda.is_available()


if __name__ == "__main__":  # 90 minutes on L40S GPU
    puffin_model = puffin.Puffin(use_cuda=USE_CUDA)
    fasta_index = SeqIO.index(COMBINED_FA, "fasta")
    N_PLASMIDS = len(fasta_index)

    update_every = 1000
    pbar = tqdm(total=N_PLASMIDS, desc="Obtain Puffin TSS predictions")

    with h5py.File(PREDS_OUT, "w") as h5f:
        h5f.attrs["features"] = PUFFIN_KEYS  # TODO

        for record_idx, (gbk_name, record) in enumerate(fasta_index.items()):
            seq = str(record.seq)[:-CIRCULAR_OVERLAP]
            L = len(seq)

            # multiply the sequence to safely cover the flank length
            repeats = (PUFFIN_FLANK // L) + 1
            extended_seq = seq * repeats
            flanked_seq = extended_seq[-PUFFIN_FLANK:] + seq + extended_seq[:PUFFIN_FLANK]

            # Returns a dense matrix of shape (10, L)
            puffin_preds = puffin_model.predict(flanked_seq).T[PUFFIN_RAW_KEYS].T.to_numpy(dtype=np.float32)  # returns a dense matrix of shape (10, L)

            h5f.create_dataset(
                name=gbk_name, 
                data=puffin_preds, 
                compression="gzip",
                compression_opts=4  # balances fast write speed with good file compression
            )

            if (record_idx + 1) % update_every == 0 or record_idx == N_PLASMIDS - 1:
                pbar.update(update_every)

    pbar.close()

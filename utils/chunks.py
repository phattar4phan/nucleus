import numpy as np

from Bio import SeqIO
from colorama import Fore, init

init(autoreset=True)

# match tokenizes.py
K = 8
SEQ_LENGTH = 128

base_to_id = {
    "A": 0, # adenine
    "C": 1, # cytosine
    "G": 2, # guanine
    "T": 3, # thymine
    "N": 4 # unknown base
}

record_labels = np.load("labels.npy")

chunk_labels = []
chunk_written = 0

for record_idx, record in enumerate(SeqIO.parse("genomes.fna", "fasta")):
    sequence = str(record.seq).upper().strip()
    label_id = int(record_labels[record_idx])
    token_count = 0
    for i in range(len(sequence)-K+1):
        encoded = sequence[i:i+K]
        
        # ignore unexpected characters
        if any(base not in base_to_id for base in encoded):
            continue
        token_count += 1
    
    n_full_chunks = token_count // SEQ_LENGTH
    chunk_labels.extend([label_id] * n_full_chunks)
    chunk_written += n_full_chunks
    print(f'\r[{Fore.LIGHTCYAN_EX}{record.id}{Fore.RESET}]: {token_count} tokens, {n_full_chunks} chunks ({label_id})', end='', flush=True)
    
np.save("chunks.npy", np.array(chunk_labels, dtype=np.int64))
print(f'\n[done]: saved to chunks.npy, total chunks: {chunk_written}')
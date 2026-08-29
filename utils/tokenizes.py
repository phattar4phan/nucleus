import numpy as np

from Bio import SeqIO
from pathlib import Path
from colorama import Fore, init

init(autoreset=True)

# 8-mers tokenization
K = 8
SEQ_LENGTH = 128

base_to_id = {
    "A": 0, # adenine
    "C": 1, # cytosine
    "G": 2, # guanine
    "T": 3, # thymine
    "N": 4 # unknown base
}

def to_id(encoded):
    token_id = 0
    for base in encoded:
        token_id = token_id * 5 + base_to_id[base]
    return token_id

# count exact total bases
bases = 0
for sequence in SeqIO.parse("genomes.fna", "fasta"):
    bases += len(sequence)

total_chunks = 0
chunk_written = 0
record_chunk_counts = []

with open("tokens.bin", "wb") as file:
    total = 0
    for record in SeqIO.parse("genomes.fna", "fasta"):
        sequence = str(record.seq).upper().strip()
        total += len(sequence)
        print(f'\r[{Fore.LIGHTCYAN_EX}{record.id}{Fore.RESET}/{Fore.LIGHTYELLOW_EX}{total}{Fore.RESET}/{Fore.LIGHTWHITE_EX}{bases}{Fore.RESET}]: {len(sequence):,} bp', end='', flush=True)
        
        record_tokens = []
        for i in range(len(sequence)-K+1):
            encoded = sequence[i:i+K]
            
            # ignore unexpected characters
            if any(base not in base_to_id for base in encoded):
                continue
            record_tokens.append(to_id(encoded))
            
        n_full_chunks = len(record_tokens) // SEQ_LENGTH
        usable = n_full_chunks * SEQ_LENGTH
        if usable > 0:
            np.asarray(record_tokens[:usable], dtype=np.uint32).tofile(file)
            
        total_chunks += usable
        chunk_written += n_full_chunks
        record_chunk_counts.append(n_full_chunks)
        
np.save("records.npy", np.array(record_chunk_counts, dtype=np.int64))

tokens = np.memmap(
    "tokens.bin",
    dtype=np.uint32,
    mode="r"
)

seen = set()
duplicate = set()

for item in tokens:
    if item in seen:
        duplicate.add(item)
    else:
        seen.add(item)
        
print(f'\n[done]: tokens: {len(tokens)}, chunks: {chunk_written}, duplicate: {len(duplicate)}')
import numpy as np

from pathlib import Path
from Bio import SeqIO
from colorama import Fore, init

init(autoreset=True)

# k-mers while k=8
K = 8

BUFFER = 1_000_000

base_to_id = {
    "A": 0, # adenine
    "C": 1, # cytosine
    "G": 2, # guanine
    "T": 3, # thymine
    "N": 4  # unknown base (N)
}

def kmer_to_id(encoded):
    token_id = 0
    
    for base in encoded:
        token_id = token_id * 5 + base_to_id[base]
        
    return token_id

bases = 0
for sequence in SeqIO.parse("genomes.fna", "fasta"):
    bases += len(sequence)

with open("tokens.bin", "wb") as file:
    buffer = []
    
    total = 0
    for record in SeqIO.parse("genomes.fna", "fasta"):
        sequence = str(record.seq).upper().strip()
        total += len(sequence)
        
        print(f'\r[{Fore.LIGHTCYAN_EX}{record.id}{Fore.RESET}/{Fore.LIGHTYELLOW_EX}{total}{Fore.RESET}/{Fore.LIGHTWHITE_EX}{bases}{Fore.RESET}]: {len(sequence):,} bp', end='', flush=True)
        
        for i in range(len(sequence)-K+1): # total bases (seq - K + 1)
            encoded = sequence[i:i+K]
            
            # ignore unexpected character but keeping base N
            if any(base not in base_to_id for base in encoded):
                continue
            buffer.append(kmer_to_id(encoded))
            
            # flush to disk if it hit threshold
            if len(buffer) >= BUFFER:
                np.asarray(buffer, dtype=np.uint32).tofile(file)
                buffer.clear()
    
    # write remaing buffer (tokens) 
    if buffer:
        np.asarray(buffer, dtype=np.uint32).tofile(file)
        
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

print(f'\nDone! tokens: {len(tokens)}, example: {tokens[:20]}, duplicate: {duplicate}')
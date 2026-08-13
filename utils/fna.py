from pathlib import Path
from Bio import SeqIO
from colorama import Fore, init

init(autoreset=True)

root = Path("dataset/")
output_path = Path("genomes.fna")

files = sorted(root.rglob("*.fna"))

records = 0
bases = 0

with output_path.open("w", encoding="utf-8") as output:
    for fasta, path in enumerate(files, 1):
        print(f'\r[{Fore.LIGHTCYAN_EX}{records}{Fore.RESET}/{Fore.LIGHTGREEN_EX}{fasta}{Fore.RESET}]: {path}\r', end='', flush=True)
        
        accession = path.parent.name
        
        for record in SeqIO.parse(path, "fasta"):
            description = record.description
            record.description = f"{description} ({accession})"
            
            records += 1
            bases += len(record.seq)
            
            SeqIO.write(record, output, "fasta")

print(f'\nDone! Saved to genomes.fna (total base: {bases})')
import json
import numpy as np

from pathlib import Path
from Bio import SeqIO
from colorama import Fore, init

init(autoreset=True)

root = Path("dataset/")
output_path = Path("genomes.fna")
labels_path = Path("labels.npy")
map_path = Path("mapped.json")

files = sorted(root.rglob("*.fna"))

unique_accession = sorted(list({p.parent.name for p in files}))
accession_to_label = {acc: idx for idx, acc in enumerate(unique_accession)}
num_classes = len(unique_accession)

with open(map_path, "w") as file:
    json.dump(accession_to_label, file, indent=4)

records = 0
bases = 0
label_list = []

with output_path.open("w", encoding="utf-8") as output:
    for fasta, path in enumerate(files, 1):
        print(f'\r[{Fore.LIGHTCYAN_EX}{records}{Fore.RESET}/{Fore.LIGHTGREEN_EX}{fasta}{Fore.RESET}]: {path}\r', end='', flush=True)
        accession = path.parent.name
        label_id = accession_to_label[accession]
        
        for record in SeqIO.parse(path, "fasta"):
            description = record.description
            record.description = f"{description} ({accession})"
            records += 1
            bases += len(record.seq)
            label_list.append(label_id)
            SeqIO.write(record, output, "fasta")
            
np.save(labels_path, np.array(label_list, dtype=np.int64))
print(f'\n[done]: saved to genomes.fna, total bases: {bases}, num_classes: {num_classes}')
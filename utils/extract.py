import json
import zipfile

from pathlib import Path
from colorama import Fore, init

init(autoreset=True)

# load organism, flatten it
with open("refseq.json", "r", encoding="utf-8") as refseq:
    data = json.load(refseq)

# build reverse lookup (accession to its organism)
accession_to_organism = {}
for organism, accessions in data.items():
    for accession in accessions:
        accession_to_organism[accession] = organism
        
output_root = Path("dataset")
extracted = 0
skipped = 0

with zipfile.ZipFile("dataset.zip", "r") as archive:
    for name in archive.namelist():
        if name.endswith("/") or not name.startswith("ncbi_dataset/data/"):
            continue
        if not name.endswith(".fna"):
            continue
        
        # ncbi_dataset/data/{accession}/{accession}_genomic.fna
        # folder name = accession
        parts = name.split("/")
        if len(parts) < 3:
            continue
        accession = parts[2]
        
        organism = accession_to_organism.get(accession)
        if organism is None:
            print(f'[warning]: no organism mapping for {accession}, skipping...', end='', flush=True)
            skipped += 1
            continue
        
        organism_dir = output_root / organism.replace(" ", "_").replace("/", "_")
        organism_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = organism_dir / f"{accession}.fna"
        with archive.open(name) as source:
            with open(dest_path, "wb") as destination:
                while True:
                    chunk = source.read(1024*1024)
                    
                    if not chunk:
                        break
                    
                    destination.write(chunk)
        
        extracted += 1
        print(f'\r[{Fore.LIGHTCYAN_EX}{extracted}{Fore.RESET}]: {accession} -> {organism}', end='', flush=True)
        
print(f'\n[done]: extracted {extracted} files.')
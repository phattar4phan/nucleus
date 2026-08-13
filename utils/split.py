import random

from Bio import SeqIO
from colorama import Fore, init

init(autoreset=True)

# load the records from .fna file
input_path = "genomes.fna"

records = []
for i, record in enumerate(SeqIO.parse(input_path, "fasta"), 1):
    print(f'\r[parsing]: {record.id}', end="", flush=True)
    records.append(record)

# the for...loop above is for progression
# the code below is faster
# records = list(SeqIO.parse(input_path, "fasta"))

# shuffle records randomly with fixed seed
random.seed(42)
random.shuffle(records)

# split ratio
total = len(records)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_records = records[:train_end]
val_records = records[train_end:val_end]
test_records = records[val_end:]
print(f'total: {total}, train: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}')

# write the split to new .fna files
SeqIO.write(train_records, "./data/train.fna", "fasta")
SeqIO.write(val_records, "./data/val.fna", "fasta")
SeqIO.write(test_records, "./data/test.fna", "fasta")
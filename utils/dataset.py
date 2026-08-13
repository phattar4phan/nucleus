import os
import json
import requests

import zipfile
import tempfile

from dotenv import load_dotenv
from colorama import Fore, init

init(autoreset=True)

# load environment variables
load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# load accessions into refseqs variable
with open("refseq.json", "r", encoding="utf-8") as refseq:
    data = json.load(refseq)
    
refseqs = data["accessions"]

# NCBI REST API endpoint
# https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/rest-api/#post-/genome/download
endpoint = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/download"

# headers
headers = {
    "api-key": NCBI_API_KEY,
    "Content-Type": "application/json"
}

# tempfile for merge assembly_data_report.jsonl
assembly = tempfile.NamedTemporaryFile(mode="wb", delete=False)
assembly_report_path = assembly.name
assembly.close()

# set to false as no catalog written yet
# (this is for dataset_catalog.json)
# as I would only keep the first dataset_catalog.json
catalog_written = False

try:
    with zipfile.ZipFile("dataset.zip", "w", zipfile.ZIP_DEFLATED) as output:
        for i in range(0, len(refseqs), 100):
            batch = refseqs[i:i+100]
            print(f'[requested]: {i+1}-{i+len(batch)}/{len(refseqs)}')
            
            payload = {
                "accessions": batch,
                "include_annotation_type": [
                    "GENOME_FASTA",
                    "SEQUENCE_REPORT"
                ]
            }
            
            max_retries = 5
            for attempt in range(1, max_retries+1):
                try:
                    print(f'[attempt]: {attempt}/{max_retries}')
                    
                    # send POST req
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=(30, 1800)
                    )
                    
                    # if error (!= 200) print the error statement
                    if response.status_code != 200:
                        print(f'[error]: {response.status_code}, retrying...')
                        continue
                        
                    # save streamed response to temp file
                    with tempfile.NamedTemporaryFile(suffix=".zip") as temp:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                temp.write(chunk)
                                
                        temp.flush()
                        
                        # read the temp zio
                        with zipfile.ZipFile(temp.name, "r") as input:
                            for name in input.namelist():
                                if name.endswith("/"):
                                    continue
                                
                                # only copy files inside ncbi_dataset/data/
                                if not name.startswith("ncbi_dataset/data/"):
                                    continue
                                
                                # merge assembly_data_report.jsonl
                                if name == "ncbi_dataset/data/assembly_data_report.jsonl":
                                    with input.open(name) as source:
                                        with open(assembly_report_path, "ab") as destination:
                                            while True:
                                                chunk = source.read(1024*1024)
                                                
                                                if not chunk:
                                                    break
                                                
                                                destination.write(chunk)
                                            destination.write(b"\n")
                                    continue
                                
                                # keep the first dataset_catalog.json
                                if name == "ncbi_dataset/data/dataset_catalog.json":
                                    if not catalog_written:
                                        with input.open(name) as source:
                                            with output.open(name, "w") as destination:
                                                while True:
                                                    chunk = source.read(1024*1024)
                                                    
                                                    if not chunk:
                                                        break
                                                    
                                                    destination.write(chunk)
                                        catalog_written = True
                                    continue
                                
                                # copy everything else
                                with input.open(name) as source:
                                    with output.open(name, "w") as destination:
                                        while True:
                                            chunk = source.read(1024*1024)
                                            
                                            if not chunk:
                                                break
                                            
                                            destination.write(chunk)
                    print(f'{Fore.LIGHTCYAN_EX}[completed]{Fore.RESET}: {i+1}-{i+len(batch)}/{len(refseqs)}')
                    break

                except (
                    requests.exceptions.RequestException,
                    zipfile.BadZipFile,
                    OSError,
                    EOFError
                ) as e:
                    print(f'[failed]: {e}\nretrying...')
                    
                    if attempt == max_retries:
                        print(f'[failed]: permanently.')

        # merge assembly report
        with open(assembly_report_path, "rb") as source:
            with output.open("ncbi_dataset/data/assembly_data_report.jsonl", "w") as destination:
                while True:
                    chunk = source.read(1024*1024)
                    
                    if not chunk:
                        break
                    
                    destination.write(chunk)
                
finally:
    # remove temp merged report file
    if os.path.exists(assembly_report_path):
        os.remove(assembly_report_path)
        
print('Done! Saved to dataset.zip')
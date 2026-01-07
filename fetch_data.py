#!/usr/bin/env python3
"""
fetch_data.py

Script to download dataset files from provided URLs and save them to disk.
"""
import os
import requests
from tqdm import tqdm  # progress bar for downloads
import zipfile  # for unpacking zip archives
import gzip    # for decompressing gzip files
import shutil  # for file operations

# Directory where all dataset files will be downloaded
DOWNLOAD_DIR = "data_tmp"

# Mapping of dataset URLs to destination filenames
URLS = {
    "https://data.mendeley.com/public-files/datasets/46htwnp833/files/747b2613-4d0a-4628-8aaf-8fc5547d286e/file_downloaded":
        "mango_data.csv",
    "https://raw.githubusercontent.com/RNL1/Melamine-Dataset/master/Melamine_Dataset.pkl":
        "melamine_data.pkl",
    "https://eigenvector.com/wp-content/uploads/2021/04/CGL_nir.mat_.zip":
        "CGL_nir.mat_.zip",
    "https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip":
        "corn.mat_.zip",
    "https://eigenvector.com/wp-content/uploads/2019/06/SWRI_Diesel_NIR_CSV.zip":
        "SWRI_Diesel_NIR_CSV.zip",
    "https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip":
        "nir_shootout_2002.mat_.zip",
    "https://data.mendeley.com/public-files/datasets/6hn67h2trb/files/0604423d-785c-4076-badb-b3fab8ec8367/file_downloaded":
        "eggs.csv",
    "https://figshare.com/ndownloader/files/6932732":
        "wheat_kernel.xlsx",
    "https://storage.googleapis.com/soilspec4gg-public/ossl_all_L0_v1.2.csv.gz":
        "ossl_all_L0_v1.2.csv.gz",
    "https://storage.googleapis.com/soilspec4gg-public/ossl_all_L1_v1.2.csv.gz":
        "ossl_all_L1_v1.2.csv.gz"
}

def download_file(url, output_path):
    """
    Downloads a file from the specified URL to the given output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Configure headers and allow redirects for better compatibility with Figshare and similar services
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    with requests.get(url, stream=True, headers=headers, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        # download with progress bar
        with open(output_path, 'wb') as f, tqdm(
            total=total, unit='iB', unit_scale=True, unit_divisor=1024,
            desc=os.path.basename(output_path)
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def unpack_archive(file_path, extract_to):
    """
    Unpack archive files (zip) into the extract_to directory.
    """
    if file_path.lower().endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as archive:
            archive.extractall(extract_to)

def decompress_gzip(file_path):
    """
    Decompress a .gz file to its original format and remove the .gz file.
    """
    output_file = file_path[:-3]
    with gzip.open(file_path, 'rb') as f_in, open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return output_file

def main():
    # Ensure the download directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading files to {DOWNLOAD_DIR}...")
    for url, filename in URLS.items():
        # use provided filename for download
        output_path = os.path.join(DOWNLOAD_DIR, filename)
        print(f"  - Downloading {url} to {output_path}")
        download_file(url, output_path)
        # unpack if it's a zip archive
        if output_path.lower().endswith('.zip'):
            print(f"    Unpacking {filename}...")
            unpack_archive(output_path, DOWNLOAD_DIR)
            os.remove(output_path)
            print(f"    Removed archive {filename}")
        # decompress if it's a gzip file
        elif output_path.lower().endswith('.gz'):
            print(f"    Decompressing {filename}...")
            decompressed = decompress_gzip(output_path)
            os.remove(output_path)
            print(f"    Produced {os.path.basename(decompressed)} and removed archive {filename}")
    print("All downloads completed.")

if __name__ == "__main__":
    main()

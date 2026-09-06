#!/usr/bin/env python3
"""
fetch_data.py

Script to download dataset files from provided URLs and save them to disk.
"""
import os
import argparse
import stat
import tempfile
from pathlib import Path
import requests
from tqdm import tqdm  # progress bar for downloads
import zipfile  # for unpacking zip archives
import gzip    # for decompressing gzip files
import shutil  # for file operations
from dataset_catalog import DATASETS, DOWNLOAD_FALLBACKS, select_datasets

# Directory where all dataset files will be downloaded
DOWNLOAD_DIR = str(Path(__file__).resolve().parent / "data_tmp")


def destination_mode(path):
    """Preserve an existing mode or apply the process umask for a new file."""
    if os.path.exists(path):
        return stat.S_IMODE(os.stat(path).st_mode)
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask

def download_file(url, output_path, fallback_urls=()):
    """
    Downloads a file from the specified URL to the given output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Configure headers and allow redirects for better compatibility with Figshare and similar services
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    candidates = (url, *fallback_urls)
    for position, candidate in enumerate(candidates):
        temporary = None
        try:
            with requests.get(candidate, stream=True, headers=headers,
                              allow_redirects=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                if r.status_code != 200 or 'html' in r.headers.get('content-type', '').lower():
                    raise requests.HTTPError(f"Expected a data file, received HTTP {r.status_code} from {candidate}")
                total = int(r.headers.get('content-length', 0))
                received = 0
                with tempfile.NamedTemporaryFile(mode='wb', dir=os.path.dirname(output_path),
                                                 prefix=Path(output_path).name + '.', suffix='.part',
                                                 delete=False) as f, tqdm(
                    total=total, unit='iB', unit_scale=True, unit_divisor=1024,
                    desc=os.path.basename(output_path)
                ) as bar:
                    temporary = f.name
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            received += len(chunk)
                            bar.update(len(chunk))
                if not received or (total and not r.headers.get('content-encoding') and received != total):
                    raise requests.RequestException(f"Empty or incomplete download from {candidate}")
                os.chmod(temporary, destination_mode(output_path))
                os.replace(temporary, output_path)
                temporary = None
                return
        except requests.RequestException as exc:
            if position == len(candidates) - 1:
                raise
            print(f"    Download failed: {exc}; trying an alternate publisher URL")
        finally:
            if temporary is not None:
                os.remove(temporary)

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

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-only', action='store_true',
                        help='Only datasets with reviewed open licenses allowing commercial reuse')
    parser.add_argument('--dry-run', action='store_true', help='List selection without network or writes')
    args = parser.parse_args(argv)
    selected = select_datasets(args.open_only)
    for dataset in DATASETS:
        action = 'INCLUDE' if dataset in selected else 'SKIP'
        print(f"{action} {dataset.id}: {dataset.license} ({dataset.license_source})")
        if dataset in selected:
            print(f"  Attribution: {dataset.attribution}")
            for url, filename in dataset.files:
                print(f"  {url} -> {filename}")
                for fallback in DOWNLOAD_FALLBACKS.get(url, ()):
                    print(f"    Public storage fallback: {fallback}")
    if args.dry_run:
        return 0
    # Ensure the download directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading files to {DOWNLOAD_DIR}...")
    for url, filename in (file for dataset in selected for file in dataset.files):
        # use provided filename for download
        output_path = os.path.join(DOWNLOAD_DIR, filename)
        print(f"  - Downloading {url} to {output_path}")
        download_file(url, output_path, fallback_urls=DOWNLOAD_FALLBACKS.get(url, ()))
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
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

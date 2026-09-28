"""
download_data.py
Downloads real-world benchmark datasets from the web for:
1. Multi-Touch Marketing Attribution Journey Dataset (campaign_data.csv)
2. E-Commerce Customer Transactions Dataset (online_retail.csv - UCI Machine Learning Repository)
"""

import os
import sys
import time
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    "campaign_data.csv": {
        "url": "https://raw.githubusercontent.com/MatCyt/Markov-Chain/master/sample_datasets/campaign_data.csv",
        "description": "Multi-Touch Marketing Attribution Journeys (Cookie, Time, Channel, Interaction, Conversion)"
    },
    "online_retail.csv": {
        "url": "https://github.com/dbdmg/data-science-lab/raw/master/datasets/online_retail.csv",
        "description": "UCI Online Retail Transaction Logs (InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)"
    }
}

def download_file(url, dest_path, desc):
    print(f"\n==================================================================")
    print(f"Downloading: {desc}")
    print(f"Source URL : {url}")
    print(f"Target Path: {dest_path}")
    print(f"==================================================================")
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"File already exists ({size_mb:.2f} MB). Skipping re-download.")
        return
    
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = resp.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 1024 * 256 # 256 KB
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {mb:.1f} MB / {total_size/(1024*1024):.1f} MB ({pct:.1f}%)")
            else:
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Downloaded: {mb:.1f} MB")
            sys.stdout.flush()
            
    elapsed = time.time() - t0
    final_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"\nDownload completed in {elapsed:.1f}s ({final_mb:.2f} MB)")

def verify_datasets():
    print("\n--- Verifying Downloaded Datasets ---")
    for fname, info in DATASETS.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing expected file: {fpath}")
        
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"\nVerifying {fname} ({size_mb:.2f} MB)...")
        df_preview = pd.read_csv(fpath, nrows=5)
        print(f"Columns: {list(df_preview.columns)}")
        print("Sample rows:")
        print(df_preview.head(2))

if __name__ == "__main__":
    for fname, info in DATASETS.items():
        dest = os.path.join(DATA_DIR, fname)
        download_file(info["url"], dest, info["description"])
    
    verify_datasets()
    print("\nAll datasets successfully downloaded and verified!")

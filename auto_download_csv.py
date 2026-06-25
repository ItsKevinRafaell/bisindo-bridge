#!/usr/bin/env python3
"""
Auto-download BISINDO CSV from VPS.
Save to dataset/landmarks_captured_v2.csv (with timestamped backups).
Run manually or via cron every hour.
"""
import os
import sys
import shutil
import urllib.request
from datetime import datetime

# Update this URL when the Cloudflare tunnel changes
TUNNEL_URL = "https://vids-absent-functional-played.trycloudflare.com"
DOWNLOAD_URL = f"{TUNNEL_URL}/api/download"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
CSV_PATH = os.path.join(DATA_DIR, "landmarks_captured_v2.csv")
BACKUP_DIR = os.path.join(DATA_DIR, "local_backups")


def backup_current():
    if not os.path.exists(CSV_PATH):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(BACKUP_DIR, f"landmarks_captured_v2_{timestamp}.csv")
    shutil.copy2(CSV_PATH, dst)
    # Keep only last 10 backups
    backups = sorted([
        os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)
        if f.startswith("landmarks_captured_v2_") and f.endswith(".csv")
    ])
    for old in backups[:-10]:
        os.remove(old)
    return dst


def main():
    print(f"[{datetime.now()}] Downloading {DOWNLOAD_URL}")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Backup current CSV before overwriting
    old_backup = backup_current()
    if old_backup:
        print(f"  Backed up old CSV to {old_backup}")

    # Download new CSV
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, CSV_PATH)
        size = os.path.getsize(CSV_PATH)
        with open(CSV_PATH) as f:
            rows = sum(1 for _ in f) - 1
        print(f"  ✅ Downloaded {size/1024/1024:.1f} MB, {rows} rows")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

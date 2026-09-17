#!/usr/bin/env python3
"""
Raw Dataset Downloader for iPinYou RTB Benchmark.
-----------------------------------------------------------------------------
Downloads the raw iPinYou contest dataset archive or extracts sample data
for local benchmarking and AlphaEvolve offline simulation.

Usage:
  # 1. Download dataset with default options / instructions:
  python3 -m src.data.download_data

  # 2. Download from custom URL:
  python3 -m src.data.download_data --url "https://example.com/ipinyou.contest.dataset.7z"

  # 3. Download via Kaggle CLI (if configured):
  python3 -m src.data.download_data --source kaggle

  # 4. Generate a synthetic / sampled test dataset without downloading 6.3 GB:
  python3 -m src.data.download_data --generate-sample
-----------------------------------------------------------------------------
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


DEFAULT_ARCHIVE_PATH = "data/raw/ipinyou.contest.dataset.7z"
SEASON2_DIR = "data/raw/season2"
KAGGLE_DATASET = "lastsummer/ipinyou"
# Verified official dataset mirror from wnzhang/make-ipinyou-data (PR #11)
DEFAULT_DOWNLOAD_URL = "https://www.dropbox.com/s/txz0ms0axqf7jrl/ipinyou.contest.dataset.7z?dl=1"


def format_bytes(size: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def check_existing_data(dest_path: Path, raw_dir: Path) -> bool:
    """Checks if raw dataset or extracted files already exist."""
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"✅ Found existing archive: {dest_path} ({format_bytes(dest_path.stat().st_size)})")
        return True
    if raw_dir.exists() and any(raw_dir.rglob("*.bz2")):
        count = len(list(raw_dir.rglob("*.bz2")))
        print(f"✅ Found {count} extracted Season 2 bz2 logs in {raw_dir}")
        return True
    return False


def download_with_progress(url: str, dest: Path) -> bool:
    """Downloads a file from URL with a live console progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")
    print(f"🌐 Connecting to: {url}")
    print(f"💾 Saving to:     {dest}")

    start_time = time.time()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (AlphaEvolve-Data-Downloader)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_dest, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size and total_size.isdigit() else 0
            downloaded = 0
            block_size = 65536

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)

                elapsed = max(0.1, time.time() - start_time)
                speed = downloaded / elapsed

                if total_bytes > 0:
                    percent = min(100.0, (downloaded / total_bytes) * 100.0)
                    progress_bar = ("#" * int(percent // 2)).ljust(50, "-")
                    sys.stdout.write(
                        f"\r[{progress_bar}] {percent:5.1f}% | "
                        f"{format_bytes(downloaded)} / {format_bytes(total_bytes)} | "
                        f"{format_bytes(speed)}/s"
                    )
                else:
                    sys.stdout.write(
                        f"\rDownloaded: {format_bytes(downloaded)} | "
                        f"Speed: {format_bytes(speed)}/s"
                    )
                sys.stdout.flush()

        print()
        temp_dest.rename(dest)
        print(f"✅ Successfully downloaded {format_bytes(dest.stat().st_size)} in {time.time() - start_time:.1f}s")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if temp_dest.exists():
            temp_dest.unlink()
        return False


def download_from_kaggle(dest_dir: Path) -> bool:
    """Attempts to download dataset using Kaggle CLI."""
    if not shutil.which("kaggle"):
        print("⚠️ Kaggle CLI ('kaggle') is not installed or not in PATH.")
        print("   Install via: pip install kaggle, and configure ~/.kaggle/kaggle.json")
        return False

    print(f"🚀 Invoking Kaggle CLI to download '{KAGGLE_DATASET}'...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest_dir), "--unzip"]
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle CLI command failed with code {e.returncode}")
        return False


def print_manual_download_guide(dest_path: Path):
    """Prints friendly download instructions with working mirrors."""
    print("=" * 75)
    print("ℹ️  HOW TO OBTAIN THE IPINYOU RTB BENCHMARK DATASET")
    print("=" * 75)
    print("Target Archive: ipinyou.contest.dataset.7z (6.30 GB, uncompressed ~14 GB)")
    print()
    print("Option 1: One-click Direct Download via verified official mirror (PR #11):")
    print("   Run:")
    print("   $ make download-data SOURCE=dropbox")
    print("   Or download with curl:")
    print(f'   $ curl -L -C - --retry 5 -o {dest_path} "{DEFAULT_DOWNLOAD_URL}"')
    print()
    print("Option 2: Kaggle Dataset Mirror (Web UI / Kaggle CLI):")
    print("   👉 https://www.kaggle.com/datasets/lastsummer/ipinyou")
    print("   Or CLI: kaggle datasets download -d lastsummer/ipinyou -p data/raw/ --unzip")
    print()
    print("Option 3: Quick Testing with Synthetic Sample Dataset (Zero download needed):")
    print("   Run:")
    print("   $ .venv/bin/python -m src.data.download_data --generate-sample")
    print()
    print(f"Once downloaded, ensure the archive is located at: {dest_path}")
    print("Then run: make prepare-data")
    print("=" * 75)


def generate_sample_dataset(
    campaign_id: str = "1458",
    num_records: int = 50000,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Generates a realistic synthetic iPinYou sample dataset for quick testing
    without requiring the full 6.3 GB download.
    """
    import numpy as np
    import pandas as pd

    out_dir = Path(output_dir or f"data/processed/{campaign_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"campaign_{campaign_id}_full.parquet"

    print(f"🎲 Generating synthetic RTB dataset for Campaign {campaign_id} ({num_records:,} auctions)...")
    np.random.seed(42)

    days = np.random.choice(range(1, 8), size=num_records, p=[0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15])
    hours = np.random.randint(0, 24, size=num_records)
    ad_exchanges = np.random.choice(["1", "2", "3"], size=num_records, p=[0.4, 0.35, 0.25])
    visibilities = np.random.choice(["0", "1", "2", "255"], size=num_records)
    formats = np.random.choice(["1", "2"], size=num_records)

    # Lognormal market clearing price Z with realistic mode around 50-80
    market_prices = np.clip(np.random.lognormal(mean=4.1, sigma=0.6, size=num_records), 5.0, 300.0)

    # Base pCTR with rare clicks (~0.08% CTR)
    base_pctr = np.random.beta(a=0.8, b=1000.0, size=num_records)
    clicks = np.random.binomial(n=1, p=np.clip(base_pctr, 0.0, 0.05))

    records = {
        "bid_id": [f"synth_bid_{i:08d}" for i in range(num_records)],
        "timestamp": [f"201306{d+5:02d}{h:02d}0000000" for d, h in zip(days, hours)],
        "raw_day": [d + 5 for d in days],
        "day": days,
        "hour": hours,
        "time_bucket": hours // 6,
        "ad_exchange": ad_exchanges,
        "ad_slot_visibility": visibilities,
        "ad_slot_format": formats,
        "ad_slot_floor_price": np.random.choice([0.0, 5.0, 10.0, 20.0], size=num_records),
        "user_tag": ["" for _ in range(num_records)],
        "bid_price": market_prices * 1.1,
        "paying_price": market_prices,
        "market_price": market_prices,
        "win": np.ones(num_records, dtype=int),
        "cost": market_prices,
        "click": clicks,
    }

    df = pd.DataFrame(records)
    df.to_parquet(full_path, index=False)
    print(f"✅ Generated and saved sample dataset to: {full_path}")
    print(f"   Total rows: {len(df):,}, Clicks: {df['click'].sum():,}, CTR: {df['click'].mean()*100:.3f}%")
    return full_path


def main():
    parser = argparse.ArgumentParser(
        description="Download or acquire raw iPinYou RTB benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to download the ipinyou.contest.dataset.7z archive.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_ARCHIVE_PATH,
        help=f"Target path for downloaded archive (default: {DEFAULT_ARCHIVE_PATH}).",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "dropbox", "direct", "kaggle", "sample", "guide"],
        default="auto",
        help="Preferred download source mechanism.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if archive or extracted data already exists.",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a local synthetic sample dataset without downloading the 6.3 GB archive.",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default="1458",
        help="Campaign ID for sample generation (default: 1458).",
    )

    args = parser.parse_args()
    dest_path = Path(args.dest)
    raw_dir = Path(SEASON2_DIR)

    if args.generate_sample or args.source == "sample":
        generate_sample_dataset(campaign_id=args.campaign_id)
        return

    # User explicitly requested guide
    if args.source == "guide":
        print_manual_download_guide(dest_path)
        return

    # Check if already present
    if not args.force and check_existing_data(dest_path, raw_dir):
        print("Data is already available. Use '--force' to re-download or 'make prepare-data' to process.")
        return

    # User provided direct URL
    if args.url:
        success = download_with_progress(args.url, dest_path)
        if success:
            print("Download complete! Next step: run 'make prepare-data'.")
            return
        sys.exit(1)

    # User requested verified Dropbox mirror
    if args.source in ("dropbox", "direct"):
        print(f"🚀 Downloading official iPinYou benchmark from verified Dropbox mirror...")
        success = download_with_progress(DEFAULT_DOWNLOAD_URL, dest_path)
        if success:
            print("Download complete! Next step: run 'make prepare-data'.")
            return
        sys.exit(1)

    # Try Kaggle CLI if requested
    if args.source == "kaggle" or (args.source == "auto" and shutil.which("kaggle")):
        print(f"Attempting to download from Kaggle dataset: {KAGGLE_DATASET} ...")
        if download_from_kaggle(dest_path.parent):
            print("Kaggle download complete! Next step: run 'make prepare-data'.")
            return

    # If no URL and no automatic download succeeded, print guide
    print_manual_download_guide(dest_path)


if __name__ == "__main__":
    main()

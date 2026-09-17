"""
iPinYou Dataset Downloader and Parser
Utility to download official iPinYou benchmark datasets (e.g. Campaign 1458, Season 2)
or parse locally stored raw text files into the standard schema.
"""

import os
import urllib.request
import tarfile
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict


IPINYOU_COLUMN_NAMES = [
    "bid_id", "timestamp", "log_type", "ipinyou_id", "user_agent", "ip",
    "region", "city", "ad_exchange", "domain", "url", "anonymous_url",
    "ad_slot_id", "ad_slot_width", "ad_slot_height", "ad_slot_visibility",
    "ad_slot_format", "ad_slot_floor_price", "creative_id", "bid_price",
    "paying_price", "key_page_url", "advertiser_id", "user_tags"
]


def parse_ipinyou_log_file(file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Parses a standard tab-delimited iPinYou bidding log file.

    Args:
        file_path: Path to the raw .txt log file.
        max_rows: Optional limit on rows to read.

    Returns:
        Structured pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"iPinYou log file not found at: {file_path}")

    # Read tab-separated log
    df = pd.read_csv(
        file_path,
        sep="\t",
        names=IPINYOU_COLUMN_NAMES,
        nrows=max_rows,
        low_memory=False,
    )

    # Clean & format fields
    df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce").fillna(0.0)
    df["paying_price"] = pd.to_numeric(df["paying_price"], errors="coerce").fillna(0.0)
    df["ad_slot_floor_price"] = pd.to_numeric(df["ad_slot_floor_price"], errors="coerce").fillna(0.0)

    # In iPinYou: log_type == 1 is bid, 2 is impression, 3 is click
    # For RTB bidding benchmarks, impressions won have paying_price > 0
    df["win"] = (df["paying_price"] > 0).astype(int)
    # Market price Z in second-price auction is the paying_price
    df["market_price"] = df["paying_price"]
    df["click"] = 0  # To be merged with clk logs if available

    # Temporal feature derivation
    # Timestamp in iPinYou is often YYYYMMDDHHMMSS or epoch ms
    ts_str = df["timestamp"].astype(str)
    if ts_str.iloc[0].startswith("2013") and len(ts_str.iloc[0]) >= 10:
        # e.g. 20130606120000 -> extract day and hour
        df["day"] = ts_str.str[6:8].astype(int)
        # Normalize day to 1..N
        min_day = df["day"].min()
        df["day"] = df["day"] - min_day + 1
        df["hour"] = ts_str.str[8:10].astype(int)
    else:
        # Generic sequential day/hour
        df["hour"] = (df.index // 3600) % 24
        df["day"] = (df.index // 86400) + 1

    df["time_bucket"] = df["hour"] // 6
    return df


def download_ipinyou_campaign(campaign_id: str = "1458", target_dir: str = "data/raw") -> Path:
    """
    Placeholder/Helper to download and extract official iPinYou benchmark files.
    If external link is blocked or unavailable, returns instructional message.
    """
    dest_path = Path(target_dir) / f"campaign_{campaign_id}"
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Official mirrors often require manual credential acceptance or are hosted on research drives
    readme_path = dest_path / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            f"iPinYou Campaign {campaign_id} Raw Data Directory.\n"
            f"If downloading from academic mirrors, place train.log.txt and test.log.txt here.\n"
        )
    return dest_path

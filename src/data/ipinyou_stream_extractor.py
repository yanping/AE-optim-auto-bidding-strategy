"""
Optimized Stream Extractor and Parser for iPinYou Dataset
Extracts Season 2 (Campaign 1458) files directly from the raw 7z archive,
processes clicks and impressions, and converts into standard DataFrame format.
"""

import os
import bz2
import time
from pathlib import Path
from typing import Dict, Set, List, Optional
import pandas as pd
import numpy as np
import py7zr

IPINYOU_SCHEMA = [
    "bidid", "timestamp", "logtype", "ipinyouid", "useragent", "IP",
    "region", "city", "adexchange", "domain", "url", "urlid",
    "slotid", "slotwidth", "slotheight", "slotvisibility", "slotformat",
    "slotprice", "creative", "bidprice", "payprice", "keypage",
    "advertiser", "usertag"
]


def extract_season2_files(
    archive_path: str = "data/raw/ipinyou.contest.dataset.7z",
    extract_dir: str = "data/raw/season2",
) -> Path:
    """
    Selectively extracts only Season 2 files (training2nd and testing2nd) from the 7z archive.
    Avoids extracting Season 1 and Season 3 to save IO and disk space.
    """
    dest = Path(extract_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Opening 7z archive: {archive_path} ...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        all_names = z.getnames()
        # Find Season 2 files
        target_names = [
            n for n in all_names
            if ("training2nd" in n or "testing2nd" in n or "user.profile.tags" in n)
            and not n.endswith("/")
        ]
        print(f"Found {len(target_names)} Season 2 target files in archive. Extracting...")
        z.extract(path=dest, targets=target_names)
        print("Extraction of Season 2 files complete.")
    return dest


def process_campaign_data(
    extracted_dir: str = "data/raw/season2",
    campaign_id: str = "1458",
    output_dir: str = "data/processed/1458",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parses extracted impression and click bz2 logs for a specific campaign,
    correlates clicks, and constructs the standardized DataFrame.
    
    Args:
        extracted_dir: Directory containing extracted Season 2 files.
        campaign_id: Campaign ID to filter (e.g. '1458', '3358').
        output_dir: Directory to save the parquet dataset.
        start_date: Optional start date in YYYYMMDD format (e.g. '20130606').
        end_date: Optional end date in YYYYMMDD format (e.g. '20130612').
        max_records: Optional max number of impression records to load.
    """
    root = Path(extracted_dir)
    # Recursively find training2nd and testing2nd
    train_dir = None
    test_dir = None
    for p in root.rglob("training2nd"):
        if p.is_dir():
            train_dir = p
            break
    for p in root.rglob("testing2nd"):
        if p.is_dir():
            test_dir = p
            break

    if not train_dir:
        raise FileNotFoundError(f"training2nd directory not found in {extracted_dir}")

    print(f"Found training directory: {train_dir}")
    clk_files = sorted(list(train_dir.glob("clk.*.txt.bz2")) + list(train_dir.glob("clk.*.txt")))
    imp_files = sorted(list(train_dir.glob("imp.*.txt.bz2")) + list(train_dir.glob("imp.*.txt")))

    # Pre-filter files by date if specified
    if start_date or end_date:
        import re
        def date_in_range(fname: str) -> bool:
            m = re.search(r'\.(\d{8})\.', fname)
            if not m:
                return True
            d = m.group(1)
            if start_date and d < str(start_date):
                return False
            if end_date and d > str(end_date):
                return False
            return True
        clk_files = [f for f in clk_files if date_in_range(f.name)]
        imp_files = [f for f in imp_files if date_in_range(f.name)]
        print(f"Filtered to {len(imp_files)} impression files within date range [{start_date or 'min'}, {end_date or 'max'}]")

    # 1. Load clicked bid IDs into memory
    print(f"Reading click files ({len(clk_files)} files)...")
    clicked_bids: Set[str] = set()
    for clk_f in clk_files:
        opener = bz2.open if str(clk_f).endswith(".bz2") else open
        with opener(clk_f, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    clicked_bids.add(parts[0])
    print(f"Total unique clicked bid IDs indexed: {len(clicked_bids)}")

    # 2. Stream impressions and filter for campaign_id
    records: List[Dict] = []
    date_filter_msg = f" (Date range: {start_date or 'start'} to {end_date or 'end'})" if (start_date or end_date) else ""
    print(f"Streaming impression files ({len(imp_files)} files) for Campaign {campaign_id}{date_filter_msg}...")
    
    count = 0
    ad_idx = IPINYOU_SCHEMA.index("advertiser")
    bid_idx = IPINYOU_SCHEMA.index("bidid")
    time_idx = IPINYOU_SCHEMA.index("timestamp")
    pay_idx = IPINYOU_SCHEMA.index("payprice")
    bid_price_idx = IPINYOU_SCHEMA.index("bidprice")
    floor_idx = IPINYOU_SCHEMA.index("slotprice")
    ex_idx = IPINYOU_SCHEMA.index("adexchange")
    vis_idx = IPINYOU_SCHEMA.index("slotvisibility")
    fmt_idx = IPINYOU_SCHEMA.index("slotformat")
    tag_idx = IPINYOU_SCHEMA.index("usertag")

    for imp_f in imp_files:
        opener = bz2.open if str(imp_f).endswith(".bz2") else open
        with opener(imp_f, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) <= max(ad_idx, pay_idx):
                    continue
                if parts[ad_idx] != str(campaign_id):
                    continue

                ts_str = parts[time_idx]
                if start_date and len(ts_str) >= 8 and ts_str[:8] < str(start_date):
                    continue
                if end_date and len(ts_str) >= 8 and ts_str[:8] > str(end_date):
                    continue

                bid_id = parts[bid_idx]
                pay_price = float(parts[pay_idx]) if parts[pay_idx].replace(".", "", 1).isdigit() else 0.0
                bid_price = float(parts[bid_price_idx]) if parts[bid_price_idx].replace(".", "", 1).isdigit() else pay_price * 1.1
                floor_price = float(parts[floor_idx]) if parts[floor_idx].replace(".", "", 1).isdigit() else 0.0

                # Extract date and hour from YYYYMMDDHHMMSS
                if len(ts_str) >= 10:
                    day_int = int(ts_str[6:8])
                    hour_int = int(ts_str[8:10])
                else:
                    day_int = 1
                    hour_int = 0

                is_click = 1 if bid_id in clicked_bids else 0

                records.append({
                    "bid_id": bid_id,
                    "timestamp": ts_str,
                    "raw_day": day_int,
                    "hour": hour_int,
                    "time_bucket": hour_int // 6,
                    "ad_exchange": parts[ex_idx] if ex_idx < len(parts) else "unknown",
                    "ad_slot_visibility": parts[vis_idx] if vis_idx < len(parts) else "unknown",
                    "ad_slot_format": parts[fmt_idx] if fmt_idx < len(parts) else "unknown",
                    "ad_slot_floor_price": floor_price,
                    "user_tag": parts[tag_idx] if tag_idx < len(parts) else "",
                    "bid_price": bid_price,
                    "paying_price": pay_price,
                    "market_price": pay_price,  # Ground truth clearing price Z
                    "win": 1,                   # Impression logs in RTB are won impressions
                    "cost": pay_price,
                    "click": is_click,
                })
                count += 1
                if max_records and count >= max_records:
                    break
        if max_records and count >= max_records:
            break

    df = pd.DataFrame(records)
    print(f"Total campaign {campaign_id} impressions collected: {len(df)}")
    if len(df) == 0:
        raise ValueError(f"No impressions found for campaign {campaign_id}")

    # Normalize days from 1 to N
    min_day = df["raw_day"].min()
    df["day"] = df["raw_day"] - min_day + 1

    # Save processed dataset
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / f"campaign_{campaign_id}_full.parquet"
    df.to_parquet(csv_file, index=False)
    print(f"Saved processed dataset to: {csv_file}")
    return df

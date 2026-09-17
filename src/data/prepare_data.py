#!/usr/bin/env python3
"""
Data Preparation & Processing Pipeline for iPinYou RTB Benchmark.
-----------------------------------------------------------------------------
Extracts raw logs, standardizes auction features for a specific Campaign ID
and optional Date Range, performs air-gapped chronological train/val/test splits,
and fits the initial Kaplan-Meier Market Response and CTR baseline models.

Usage:
  # 1. Prepare default Campaign 1458 with all available dates:
  python3 -m src.data.prepare_data

  # 2. Prepare specific campaign:
  python3 -m src.data.prepare_data --campaign-id 3358

  # 3. Filter by date range (e.g. 2013-06-06 to 2013-06-10):
  python3 -m src.data.prepare_data --campaign-id 1458 --start-date 20130606 --end-date 20130610

  # 4. Skip model fitting (extract and split only):
  python3 -m src.data.prepare_data --skip-model-fit
-----------------------------------------------------------------------------
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .ipinyou_stream_extractor import extract_season2_files, process_campaign_data
from .splitter import split_chronological
from ..models.ctr_model import CTRModel
from ..models.market_model import KaplanMeierMarketModel


def update_config_campaign(config_path: Path, campaign_id: str, val_path: str, test_oracle_path: str):
    """Updates config.yaml with the newly prepared campaign paths if needed."""
    if not config_path.exists():
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "bidding" not in cfg:
            cfg["bidding"] = {}
        cfg["bidding"]["campaign_id"] = str(campaign_id)
        cfg["bidding"]["val_data_path"] = str(val_path)
        cfg["bidding"]["test_oracle_path"] = str(test_oracle_path)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"📝 Updated config.yaml bidding target to Campaign {campaign_id}")
    except Exception as e:
        print(f"⚠️ Could not update config.yaml: {e}")


def run_preparation_pipeline(
    archive_path: str = "data/raw/ipinyou.contest.dataset.7z",
    raw_dir: str = "data/raw/season2",
    campaign_id: str = "1458",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
    plot_output: Optional[str] = None,
    skip_model_fit: bool = False,
    update_config: bool = False,
) -> Dict:
    start_time = time.time()
    out_dir_path = Path(output_dir or f"data/processed/{campaign_id}")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    plot_path = plot_output or f"data/calibration_curve_ipinyou_{campaign_id}.png"

    print("=" * 75)
    print(f"🚀 Starting iPinYou Data Preparation Pipeline")
    print(f"   • Campaign ID:    {campaign_id}")
    print(f"   • Date Range:     {start_date or 'Earliest'} to {end_date or 'Latest'}")
    print(f"   • Output Folder:  {out_dir_path}")
    print("=" * 75)

    # 1. Check Raw Data or Pre-extracted Data
    season2_path = Path(raw_dir)
    archive = Path(archive_path)

    has_raw_logs = season2_path.exists() and any(season2_path.rglob("*.bz2"))
    parquet_full = out_dir_path / f"campaign_{campaign_id}_full.parquet"

    if not has_raw_logs:
        if archive.exists() and archive.stat().st_size > 1024 * 1024:
            print(f"📦 Unpacking Season 2 logs from {archive} ...")
            extract_season2_files(archive_path=str(archive), extract_dir=str(season2_path))
        elif parquet_full.exists():
            print(f"ℹ️ Raw bz2 logs not found, but processed parquet already exists at {parquet_full}")
        else:
            print(f"❌ Error: Neither raw logs in '{season2_path}' nor archive '{archive}' was found.")
            print(f"   Please run 'make download-data' or see README for instructions to place raw data.")
            sys.exit(1)

    # 2. Extract and Process Campaign Data
    if (start_date or end_date) or not parquet_full.exists():
        print(f"🔄 Parsing and filtering Campaign {campaign_id} impression and click streams...")
        df = process_campaign_data(
            extracted_dir=str(season2_path),
            campaign_id=str(campaign_id),
            output_dir=str(out_dir_path),
            start_date=start_date,
            end_date=end_date,
        )
    else:
        print(f"📂 Loading existing standardized dataset from {parquet_full} ...")
        df = pd.read_parquet(parquet_full)

    total_records = len(df)
    total_clicks = int(df["click"].sum())
    ctr = (total_clicks / total_records) * 100.0 if total_records > 0 else 0.0

    print(f"\n📊 Campaign {campaign_id} Summary:")
    print(f"   • Total Impressions: {total_records:,}")
    print(f"   • Total Clicks:       {total_clicks:,}")
    print(f"   • Empirical CTR:      {ctr:.4f}%")
    print(f"   • Market Price Range: Min={df['market_price'].min()}, Median={df['market_price'].median()}, Max={df['market_price'].max()}")

    # 3. Chronological Train / Val / Test Partitioning
    print("\n✂️ Partitioning chronologically into Train, Validation, and Test Oracle splits...")
    max_day = int(df["day"].max())
    min_day = int(df["day"].min())
    total_days = max_day - min_day + 1
    print(f"   Available day span: Day {min_day} to Day {max_day} ({total_days} days)")

    if total_days >= 3:
        train_days = (min_day, max_day - 2)
        val_days = (max_day - 1, max_day - 1)
        test_days = (max_day, max_day)
    elif total_days == 2:
        train_days = (min_day, min_day)
        val_days = (max_day, max_day)
        test_days = (max_day, max_day)
    else:
        train_days = (min_day, min_day)
        val_days = (min_day, min_day)
        test_days = (min_day, min_day)

    split = split_chronological(df, train_days=train_days, val_days=val_days, test_days=test_days)

    train_path = out_dir_path / "train_advertiser.parquet"
    val_path = out_dir_path / "val_advertiser.parquet"
    test_adv_path = out_dir_path / "test_advertiser.parquet"
    test_oracle_path = out_dir_path / "test_oracle.parquet"

    split.train_advertiser.to_parquet(train_path, index=False)
    split.val_advertiser.to_parquet(val_path, index=False)
    split.test_advertiser.to_parquet(test_adv_path, index=False)
    split.test_oracle.to_parquet(test_oracle_path, index=False)

    print(f"   ✔ Train Advertiser Set (Days {train_days[0]}-{train_days[1]}): {len(split.train_advertiser):,} rows -> {train_path.name}")
    print(f"   ✔ Val Advertiser Set   (Day {val_days[0]}):       {len(split.val_advertiser):,} rows -> {val_path.name}")
    print(f"   ✔ Test Oracle Set      (Day {test_days[0]}):       {len(split.test_oracle):,} rows -> {test_oracle_path.name}")

    metrics = {
        "campaign_id": campaign_id,
        "total_records": total_records,
        "total_clicks": total_clicks,
        "ctr": ctr,
        "train_size": len(split.train_advertiser),
        "val_size": len(split.val_advertiser),
        "test_size": len(split.test_oracle),
    }

    if skip_model_fit:
        print("\n⚡ Model fitting skipped (--skip-model-fit).")
        print("=" * 75)
        return metrics

    # 4. Train CTR Model
    print("\n🧠 Fitting CTR Model on Advertiser-View Training Set...")
    ctr_model = CTRModel()
    ctr_model.fit(split.train_advertiser, split.train_advertiser["click"].values)
    ctr_eval = ctr_model.evaluate(split.test_oracle, split.test_oracle["click"].values)
    print(f"   ✔ CTR Model Test AUC: {ctr_eval['auc']:.4f}, LogLoss: {ctr_eval['log_loss']:.4f}")
    metrics["ctr_auc"] = ctr_eval["auc"]
    metrics["ctr_logloss"] = ctr_eval["log_loss"]

    # 5. Fit Kaplan-Meier Survival Market Model
    print("\n📈 Fitting Segmented Kaplan-Meier Market Response Model...")
    market_model = KaplanMeierMarketModel(segment_columns=["ad_exchange", "time_bucket"])
    market_model.fit(split.train_advertiser)
    print(f"   ✔ Fitted {len(market_model.segment_models)} segmented sub-models + 1 global fallback.")

    # 6. Oracle Validation & Calibration Curve
    print("\n🎯 Verifying Kaplan-Meier Predictions against Held-out Ground-Truth Clearing Prices...")
    test_sample = split.test_oracle.sample(min(15000, len(split.test_oracle)), random_state=42)
    true_z = test_sample["market_price"].values
    bid_levels = np.percentile(true_z, np.linspace(5, 95, 20))

    learned_win_rates = []
    oracle_win_rates = []
    lower_bounds = []
    upper_bounds = []

    for b in bid_levels:
        bids_array = np.full(len(test_sample), b)
        p_win, std_err, _ = market_model.predict_win_prob(test_sample, bids_array)
        learned_win_rates.append(np.mean(p_win))
        oracle_win_rates.append(np.mean(b > true_z))
        mean_se = np.mean(std_err)
        lower_bounds.append(max(0.0, np.mean(p_win) - 1.96 * mean_se))
        upper_bounds.append(min(1.0, np.mean(p_win) + 1.96 * mean_se))

    learned_win_rates = np.array(learned_win_rates)
    oracle_win_rates = np.array(oracle_win_rates)
    mae = float(np.mean(np.abs(learned_win_rates - oracle_win_rates)))
    r2 = float(1.0 - (np.sum((oracle_win_rates - learned_win_rates) ** 2) / np.sum((oracle_win_rates - np.mean(oracle_win_rates)) ** 2)))

    print(f"   ✔ Calibration MAE: {mae:.4f}")
    print(f"   ✔ Calibration R²:  {r2:.4f}")
    metrics["calibration_mae"] = mae
    metrics["calibration_r2"] = r2

    # Save Calibration Curve Plot
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    plt.plot(bid_levels, oracle_win_rates, "r-o", linewidth=2.5, label="Oracle Ground Truth Win Rate (Unmasked Z)")
    plt.plot(bid_levels, learned_win_rates, "b--s", linewidth=2, label="Kaplan-Meier Survival Estimate (Censored Logs)")
    plt.fill_between(bid_levels, lower_bounds, upper_bounds, color="blue", alpha=0.15, label="95% Confidence Interval")
    plt.title(f"iPinYou Campaign {campaign_id}: Market Response Calibration (R²={r2:.4f})", fontsize=12)
    plt.xlabel("Bid Price (RMB CPM)", fontsize=11)
    plt.ylabel("Expected Win Rate P(win)", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"   ✔ Saved Calibration Curve to {plot_path}")

    # 7. Optionally Update config.yaml
    if update_config:
        config_file = Path("config.yaml")
        update_config_campaign(
            config_path=config_file,
            campaign_id=campaign_id,
            val_path=str(val_path),
            test_oracle_path=str(test_oracle_path),
        )

    elapsed = time.time() - start_time
    print(f"\n✨ Data Preparation Complete in {elapsed:.1f}s!")
    print("=" * 75)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, clean, split, and calibrate iPinYou RTB dataset for AlphaEvolve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default="1458",
        help="Target Campaign ID to filter and process (default: 1458).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date in YYYYMMDD format (e.g. 20130606).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date in YYYYMMDD format (e.g. 20130612).",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default="data/raw/ipinyou.contest.dataset.7z",
        help="Path to 7z dataset archive.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/season2",
        help="Path to extracted Season 2 bz2 logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed parquet files (default: data/processed/{campaign_id}).",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Path to save the market calibration curve image.",
    )
    parser.add_argument(
        "--skip-model-fit",
        action="store_true",
        help="Skip CTR and Kaplan-Meier model training.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Automatically update config.yaml with newly prepared campaign paths.",
    )

    args = parser.parse_args()
    run_preparation_pipeline(
        archive_path=args.archive_path,
        raw_dir=args.raw_dir,
        campaign_id=args.campaign_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        plot_output=args.plot_output,
        skip_model_fit=args.skip_model_fit,
        update_config=args.update_config,
    )


if __name__ == "__main__":
    main()

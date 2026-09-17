"""
Calibration and Goodness-of-Fit Evaluation Script
Plots the learned Kaplan-Meier win probability curve against the oracle ground-truth empirical win rate.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ..data.synthetic import generate_synthetic_rtb_data
from ..data.splitter import split_chronological
from ..models.market_model import KaplanMeierMarketModel
from ..models.ctr_model import CTRModel


def run_calibration_evaluation(output_path: str = "data/calibration_curve.png", n_samples: int = 40000):
    print("=" * 60)
    print("Phase 1: Running Model Calibration & Goodness-of-Fit Validation")
    print("=" * 60)

    # 1. Generate RTB dataset
    print(f"1. Generating {n_samples} synthetic RTB auction logs across 7 days...")
    raw_df = generate_synthetic_rtb_data(n_samples=n_samples, n_days=7, random_seed=42)

    # 2. Chronological Split & Censoring Mask
    print("2. Splitting chronologically and applying Advertiser-View censoring mask...")
    split = split_chronological(raw_df, train_days=(1, 5), val_days=(6, 6), test_days=(7, 7))
    train_adv = split.train_advertiser
    test_oracle = split.test_oracle

    print(f"   Train set size (Advertiser View): {len(train_adv)}")
    print(f"   Test set size (Oracle View):       {len(test_oracle)}")
    
    # Verify censoring in train set
    censored_losses = train_adv[train_adv["win"] == 0]["paying_price"].isna().sum()
    total_losses = (train_adv["win"] == 0).sum()
    print(f"   Censoring check: {censored_losses}/{total_losses} lost auctions have masked paying price (NaN).")
    assert censored_losses == total_losses, "Leakage detected! Losing auctions must be censored."

    # 3. Train CTR Model
    print("3. Fitting CTR Model on Advertiser View...")
    ctr_model = CTRModel()
    ctr_model.fit(train_adv, train_adv["click"].values)
    ctr_metrics = ctr_model.evaluate(test_oracle, test_oracle["click"].values)
    print(f"   CTR Test AUC: {ctr_metrics['auc']:.4f}, LogLoss: {ctr_metrics['log_loss']:.4f}")

    # 4. Train Kaplan-Meier Market Model
    print("4. Fitting Kaplan-Meier Market Model on Censored Logs...")
    market_model = KaplanMeierMarketModel(segment_columns=["ad_exchange", "time_bucket"])
    market_model.fit(train_adv)
    print(f"   Fitted {len(market_model.segment_models)} traffic segments + 1 global fallback model.")

    # 5. Evaluate on Held-out Oracle Test Set across Bid Grid
    print("5. Evaluating estimated win rate vs. unmasked oracle market prices Z...")
    bid_multipliers = np.linspace(0.4, 1.8, 25)
    
    # Base valuation = pctr * target_cpc * 150
    test_pctr = ctr_model.predict_proba(test_oracle)
    base_bids = test_pctr * 2.0 * 150.0

    learned_win_rates = []
    oracle_win_rates = []
    lower_bounds = []
    upper_bounds = []

    true_z = test_oracle["market_price"].values

    for m in bid_multipliers:
        cand_bids = base_bids * m
        # Learned estimate
        p_win, std_err, _ = market_model.predict_win_prob(test_oracle, cand_bids)
        mean_p_win = np.mean(p_win)
        mean_std_err = np.mean(std_err)

        # Oracle truth (actual win if cand_bid > true_z)
        actual_wins = (cand_bids > true_z).astype(float)
        mean_actual_win = np.mean(actual_wins)

        learned_win_rates.append(mean_p_win)
        oracle_win_rates.append(mean_actual_win)
        lower_bounds.append(max(0.0, mean_p_win - 1.96 * mean_std_err))
        upper_bounds.append(min(1.0, mean_p_win + 1.96 * mean_std_err))

    learned_win_rates = np.array(learned_win_rates)
    oracle_win_rates = np.array(oracle_win_rates)
    mae = np.mean(np.abs(learned_win_rates - oracle_win_rates))
    r2 = 1.0 - (np.sum((oracle_win_rates - learned_win_rates) ** 2) / np.sum((oracle_win_rates - np.mean(oracle_win_rates)) ** 2))

    print(f"   Goodness-of-Fit: MAE = {mae:.4f}, R² Score = {r2:.4f}")

    # 6. Plot Calibration Curve
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    plt.plot(bid_multipliers, oracle_win_rates, "r-o", linewidth=2.5, label="Oracle True Win Rate (Unmasked Z)")
    plt.plot(bid_multipliers, learned_win_rates, "b--s", linewidth=2, label="Kaplan-Meier Learned Response (Censored Data)")
    plt.fill_between(bid_multipliers, lower_bounds, upper_bounds, color="blue", alpha=0.15, label="95% Confidence Interval")

    plt.axvline(x=1.0, color="gray", linestyle=":", label="Baseline Multiplier (1.0)")
    plt.title("Advertiser Market Response Curve: Kaplan-Meier Survival Analysis vs. Ground Truth", fontsize=12)
    plt.xlabel("Bid Multiplier (Relative to Base Bid)", fontsize=11)
    plt.ylabel("Expected Win Rate P(win)", fontsize=11)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"6. Calibration plot successfully saved to: {output_path}")
    print("=" * 60)
    return {"mae": mae, "r2": r2, "auc": ctr_metrics["auc"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/calibration_curve.png")
    parser.add_argument("--n_samples", type=int, default=40000)
    args = parser.parse_args()
    run_calibration_evaluation(output_path=args.output, n_samples=args.n_samples)

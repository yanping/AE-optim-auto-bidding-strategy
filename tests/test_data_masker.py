"""
Unit Tests for Data Splitting and Censoring Masker
"""

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import generate_synthetic_rtb_data
from src.data.splitter import create_advertiser_view, create_oracle_view, split_chronological


def test_synthetic_data_generation():
    df = generate_synthetic_rtb_data(n_samples=500, n_days=7, random_seed=42)
    assert len(df) == 500
    assert "market_price" in df.columns
    assert "bid_price" in df.columns
    assert "win" in df.columns
    assert "click" in df.columns
    assert set(df["win"].unique()).issubset({0, 1})
    assert set(df["click"].unique()).issubset({0, 1})
    # Win rate should be reasonable
    assert 0.1 <= df["win"].mean() <= 0.8
    # Click rate on impressions should be reasonable
    assert 0.005 <= df[df["win"] == 1]["click"].mean() <= 0.15


def test_advertiser_view_censoring():
    df = generate_synthetic_rtb_data(n_samples=500, n_days=7, random_seed=42)
    adv_df = create_advertiser_view(df, drop_oracle_columns=True)

    # 1. Oracle ground truth columns must be dropped
    assert "market_price" not in adv_df.columns
    assert "true_pctr" not in adv_df.columns

    # 2. Strict Censoring Test: Every single losing auction MUST have NaN paying_price
    lost_mask = adv_df["win"] == 0
    assert lost_mask.sum() > 0, "Must have some lost auctions to test"
    assert adv_df.loc[lost_mask, "paying_price"].isna().all(), "Data leakage! Lost auction has paying price."

    # 3. Winning auctions must have finite paying_price
    won_mask = adv_df["win"] == 1
    assert adv_df.loc[won_mask, "paying_price"].notna().all()


def test_oracle_view_integrity():
    df = generate_synthetic_rtb_data(n_samples=500, n_days=7, random_seed=42)
    oracle_df = create_oracle_view(df)
    assert "market_price" in oracle_df.columns
    assert (oracle_df["market_price"] > 0).all()


def test_chronological_splitting():
    df = generate_synthetic_rtb_data(n_samples=1000, n_days=7, random_seed=42)
    split = split_chronological(df, train_days=(1, 5), val_days=(6, 6), test_days=(7, 7))

    # Test day boundaries
    assert split.train_advertiser["day"].between(1, 5).all()
    assert split.val_advertiser["day"].eq(6).all()
    assert split.test_advertiser["day"].eq(7).all()

    # Total row counts must match
    assert len(split.train_advertiser) + len(split.val_advertiser) + len(split.test_advertiser) == len(df)

    # Check advertiser vs oracle sizes
    assert len(split.train_advertiser) == len(split.train_oracle)
    assert len(split.test_advertiser) == len(split.test_oracle)

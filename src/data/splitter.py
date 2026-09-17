"""
Data Splitter and Censoring Masker
Implements chronological train/val/test partitioning and air-gapped
Advertiser-view vs. Oracle-view data transformation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from .schema import DatasetSplit


def create_advertiser_view(df: pd.DataFrame, drop_oracle_columns: bool = True) -> pd.DataFrame:
    """
    Transforms raw auction logs into the restricted Advertiser-Side View.
    Enforces the right-censoring rule:
    - If win == 1: paying_price is observed.
    - If win == 0: paying_price is censored (set to NaN).
    - If drop_oracle_columns is True: drops ground-truth market_price and true_pctr.
    """
    adv_df = df.copy()

    # Enforce right-censoring on paying_price
    adv_df.loc[adv_df["win"] == 0, "paying_price"] = np.nan
    
    # Drop oracle ground-truth columns to prevent data leakage during training/evolution
    if drop_oracle_columns:
        cols_to_drop = [c for c in ["market_price", "true_pctr"] if c in adv_df.columns]
        adv_df = adv_df.drop(columns=cols_to_drop)
        
    return adv_df


def create_oracle_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the unrestricted Oracle View containing ground-truth market_price Z.
    Used exclusively for counterfactual replay in held-out evaluations.
    """
    oracle_df = df.copy()
    if "market_price" not in oracle_df.columns:
        raise ValueError("Cannot create OracleView: 'market_price' column is missing from dataset.")
    return oracle_df


def split_chronological(
    df: pd.DataFrame,
    train_days: Tuple[int, int] = (1, 5),
    val_days: Tuple[int, int] = (6, 6),
    test_days: Tuple[int, int] = (7, 7),
) -> DatasetSplit:
    """
    Splits DataFrame chronologically by day into Train, Validation, and Test sets,
    generating both Advertiser-view and Oracle-view for each split.

    Args:
        df: Input DataFrame containing 'day' column.
        train_days: (start_day, end_day) inclusive for training.
        val_days: (start_day, end_day) inclusive for validation (AlphaEvolve search batch).
        test_days: (start_day, end_day) inclusive for final held-out test (Oracle A/B).

    Returns:
        DatasetSplit object containing all partitions.
    """
    if "day" not in df.columns:
        raise ValueError("Dataset must contain 'day' column for chronological splitting.")

    # Slice by days
    train_mask = (df["day"] >= train_days[0]) & (df["day"] <= train_days[1])
    val_mask = (df["day"] >= val_days[0]) & (df["day"] <= val_days[1])
    test_mask = (df["day"] >= test_days[0]) & (df["day"] <= test_days[1])

    raw_train = df[train_mask].reset_index(drop=True)
    raw_val = df[val_mask].reset_index(drop=True)
    raw_test = df[test_mask].reset_index(drop=True)

    split = DatasetSplit(
        train_advertiser=create_advertiser_view(raw_train),
        val_advertiser=create_advertiser_view(raw_val),
        test_advertiser=create_advertiser_view(raw_test),
        train_oracle=create_oracle_view(raw_train),
        val_oracle=create_oracle_view(raw_val),
        test_oracle=create_oracle_view(raw_test),
        metadata={
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "train_size": len(raw_train),
            "val_size": len(raw_val),
            "test_size": len(raw_test),
        }
    )
    return split

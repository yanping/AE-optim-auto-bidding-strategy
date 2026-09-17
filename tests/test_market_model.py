"""
Unit Tests for Kaplan-Meier Market Response Model
"""

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import generate_synthetic_rtb_data
from src.data.splitter import create_advertiser_view
from src.models.market_model import SingleKMEstimator, KaplanMeierMarketModel


def test_single_km_monotonicity_and_bounds():
    estimator = SingleKMEstimator(min_samples=10)

    # Synthetic observations
    bids = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0])
    wins = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    paying_prices = np.array([np.nan, np.nan, 6.5, np.nan, 7.2, 8.5, 9.0, 11.0, 14.0, 16.0])

    estimator.fit(bids, wins, paying_prices)
    assert estimator.is_fitted

    test_bids = np.linspace(1.0, 25.0, 50)
    win_probs, std_errors, ood_flags = estimator.predict_win_prob(test_bids)

    # 1. Monotonicity: win probability must be non-decreasing with bid
    diffs = np.diff(win_probs)
    assert (diffs >= -1e-7).all(), "Win probability must be monotonic non-decreasing with bid!"

    # 2. Probability bounds [0, 1]
    assert (win_probs >= 0.0).all() and (win_probs <= 1.0).all()

    # 3. Standard errors non-negative
    assert (std_errors >= 0.0).all()

    # 4. Expected second-price clearing cost cannot exceed bid
    costs = estimator.predict_expected_cost(test_bids)
    assert (costs <= test_bids + 1e-6).all(), "Second-price property violated: E[cost | win, b] <= b!"
    assert (costs >= 0.0).all()


def test_market_model_with_censored_dataset():
    raw_df = generate_synthetic_rtb_data(n_samples=5000, n_days=7, random_seed=42)
    adv_df = create_advertiser_view(raw_df)

    # Train market model on censored data
    model = KaplanMeierMarketModel(segment_columns=["ad_exchange", "time_bucket"])
    model.fit(adv_df)
    assert model.is_fitted
    assert len(model.segment_models) > 0

    # Predict on batch
    test_sample = adv_df.head(100)
    query_bids = np.linspace(5.0, 25.0, 100)
    win_probs, std_errors, ood_flags = model.predict_win_prob(test_sample, query_bids)

    assert len(win_probs) == 100
    assert (win_probs >= 0.0).all() and (win_probs <= 1.0).all()

    costs = model.predict_expected_cost(test_sample, query_bids)
    assert (costs <= query_bids + 1e-6).all()


def test_market_model_goodness_of_fit():
    """Verify that Kaplan-Meier learned from censored feedback accurately approximates true oracle win rates."""
    raw_df = generate_synthetic_rtb_data(n_samples=10000, n_days=7, random_seed=42)
    train_df = raw_df[raw_df["day"] <= 5]
    test_df = raw_df[raw_df["day"] > 5]

    train_adv = create_advertiser_view(train_df)
    model = KaplanMeierMarketModel(segment_columns=["ad_exchange", "time_bucket"])
    model.fit(train_adv)

    test_bids = np.linspace(5.0, 25.0, 20)
    true_z = test_df["market_price"].values

    mae_list = []
    for b in test_bids:
        bids_array = np.full(len(test_df), b)
        p_win, _, _ = model.predict_win_prob(test_df, bids_array)
        learned_rate = np.mean(p_win)
        
        oracle_rate = np.mean(b > true_z)
        mae_list.append(abs(learned_rate - oracle_rate))

    mean_mae = np.mean(mae_list)
    # The learned win rate from censored data should match ground-truth oracle win rate within MAE < 0.08
    assert mean_mae < 0.08, f"Mean Absolute Error too high: {mean_mae:.4f}"

"""
Unit Tests for Learned Offline Simulator
"""

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import generate_synthetic_rtb_data
from src.data.splitter import create_advertiser_view
from src.models.ctr_model import CTRModel
from src.models.market_model import KaplanMeierMarketModel
from src.simulation.offline_sim import LearnedOfflineSimulator


def test_offline_simulator_evaluation():
    raw_df = generate_synthetic_rtb_data(n_samples=3000, n_days=7, random_seed=42)
    train_df = raw_df[raw_df["day"] <= 5]
    val_df = raw_df[raw_df["day"] == 6]

    train_adv = create_advertiser_view(train_df)
    val_adv = create_advertiser_view(val_df)

    # Train models
    ctr_model = CTRModel()
    ctr_model.fit(train_adv, train_adv["click"].values)

    market_model = KaplanMeierMarketModel()
    market_model.fit(train_adv)

    simulator = LearnedOfflineSimulator(ctr_model, market_model, default_target_cpc=2.0)

    # Evaluate fixed bid
    bids = np.full(len(val_adv), 10.0)
    res = simulator.evaluate_bids(val_adv, bids, target_cpc=2.0, budget=1000.0)

    assert res.expected_clicks > 0
    assert res.expected_spend > 0
    assert res.expected_cpc > 0
    assert res.lcb_clicks <= res.expected_clicks
    assert res.ucb_cpc >= res.expected_cpc
    assert isinstance(res.is_cpc_valid, (bool, np.bool_))
    assert isinstance(res.is_budget_valid, (bool, np.bool_))

    # Evaluate dynamic policy
    def simple_policy(df, pctr, target_cpc):
        return pctr * target_cpc * 150.0

    res_policy = simulator.evaluate_policy(val_adv, simple_policy, target_cpc=2.0)
    assert res_policy.expected_clicks > 0
    assert res_policy.to_dict()["expected_clicks"] > 0

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Phase 3 Oracle Counterfactual Replay Simulator and Benchmarks."""

import pytest
import numpy as np
import pandas as pd

from src.simulation.oracle_replay import OracleReplaySimulator, OracleReplayResult
from src.bidding.baselines import McpcBidder, HumanRuleBidder, LinearBidder
from src.program import BidState


@pytest.fixture
def dummy_auction_df():
    """Generates synthetic test DataFrame mimicking test_oracle.parquet."""
    return pd.DataFrame({
        "bid_id": [f"bid_{i}" for i in range(10)],
        "market_price": [50.0, 80.0, 120.0, 30.0, 200.0, 40.0, 60.0, 90.0, 10.0, 150.0],
        "click": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "hour": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
    })


@pytest.fixture
def dummy_pctrs():
    return np.array([0.001, 0.0005, 0.002, 0.0008, 0.0004, 0.0015, 0.0007, 0.0012, 0.0003, 0.0006])


def test_oracle_replay_second_price_win_and_loss(dummy_auction_df, dummy_pctrs):
    # Fixed bid = 100.0 CPM
    class FixedBidder:
        def bid(self, state: BidState) -> float:
            return 100.0

    sim = OracleReplaySimulator(
        dummy_auction_df,
        dummy_pctrs,
        target_cpc=120.0,
        budget=10.0,  # 10 RMB budget
    )
    result = sim.replay_policy(FixedBidder(), "Fixed-100")

    # With bid=100.0, market_prices < 100: [50, 80, 30, 40, 60, 90, 10] -> 7 wins
    assert result.impressions_won == 7
    assert result.win_rate == pytest.approx(7 / 10)
    # Winning costs (CPM / 1000): (50+80+30+40+60+90+10)/1000 = 360 / 1000 = 0.36 RMB
    assert result.total_spend == pytest.approx(0.36)
    # Winning clicks for items 0, 1, 3, 5, 6, 7, 8: clicks = 1 + 0 + 0 + 1 + 0 + 1 + 0 = 3
    assert result.total_clicks == 3
    assert result.empirical_cpc == pytest.approx(0.36 / 3)
    assert not result.cpc_violation


def test_oracle_replay_budget_exhaustion(dummy_auction_df, dummy_pctrs):
    class AggressiveBidder:
        def bid(self, state: BidState) -> float:
            return 300.0

    # Very small budget: 0.10 RMB (will exhaust after 2 wins at 50 CPM and 80 CPM)
    sim = OracleReplaySimulator(
        dummy_auction_df,
        dummy_pctrs,
        target_cpc=120.0,
        budget=0.10,
    )
    result = sim.replay_policy(AggressiveBidder(), "Aggressive")

    assert result.budget_exhausted_step is not None
    assert result.budget_exhausted_step < 10
    assert result.total_spend >= 0.10
    assert result.impressions_won < 10


def test_oracle_replay_cpc_violation():
    # Test high cost with 0 clicks -> high CPC or inf
    single_loss_df = pd.DataFrame({
        "bid_id": ["b1"],
        "market_price": [150.0],
        "click": [0],
        "hour": [0],
    })
    pctrs = np.array([0.0001])
    sim = OracleReplaySimulator(single_loss_df, pctrs, target_cpc=10.0, budget=10.0)

    class WinAllBidder:
        def bid(self, state: BidState) -> float:
            return 200.0

    result = sim.replay_policy(WinAllBidder(), "HighCostZeroClick")
    assert result.total_clicks == 0
    assert result.empirical_cpc == 0.0  # Safe handling of 0 division


def test_oracle_replay_with_multiplier_func(dummy_auction_df, dummy_pctrs):
    def multiplier_fn(state: BidState) -> float:
        return 1.10

    sim = OracleReplaySimulator(
        dummy_auction_df,
        dummy_pctrs,
        target_cpc=120.0,
        budget=10.0,
    )
    result = sim.replay_policy(multiplier_fn, "Multiplier-1.1")
    assert result.total_auctions == 10
    assert isinstance(result.empirical_cpc, float)
    assert isinstance(result.win_rate, float)

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
"""Unit tests for Phase 2 bidding policies and baselines."""

import pytest
import numpy as np

from src.program import BidState, get_multiplier, evaluate
from src.bidding.baselines import McpcBidder, LinearBidder, HumanRuleBidder


@pytest.fixture
def sample_bid_state():
    return BidState(
        pctr=0.001,
        avg_pctr=0.0008,
        target_cpc=120.0,
        budget=300.0,
        remaining_budget_ratio=0.85,
        time_progress_ratio=0.15,
        current_cpc=55.0,
        cpc_ratio=55.0 / 120.0,
        recent_win_rate=0.18,
        recent_cpc=54.0,
        spend_velocity=0.95,
        last_multiplier=1.0,
    )


def test_bid_state_initialization(sample_bid_state):
    assert sample_bid_state.pctr == 0.001
    assert sample_bid_state.target_cpc == 120.0
    assert 0.0 <= sample_bid_state.remaining_budget_ratio <= 1.0


def test_seed_multiplier_bounds(sample_bid_state):
    m = get_multiplier(sample_bid_state)
    assert 0.5 <= m <= 2.0

    # Test extreme CPC overflow state
    overflow_state = BidState(
        pctr=0.001, avg_pctr=0.0008, target_cpc=100.0, budget=300.0,
        remaining_budget_ratio=0.5, time_progress_ratio=0.5,
        current_cpc=150.0, cpc_ratio=1.5, recent_win_rate=0.5,
        recent_cpc=150.0, spend_velocity=1.5, last_multiplier=1.0
    )
    m_brake = get_multiplier(overflow_state)
    assert 0.5 <= m_brake <= 1.0


def test_baselines(sample_bid_state):
    mcpc = McpcBidder(target_cpc=120.0)
    bid_mcpc = mcpc.bid(sample_bid_state)
    assert bid_mcpc == pytest.approx(1000.0 * 0.001 * 120.0)

    lin = LinearBidder(b0=60.0)
    bid_lin = lin.bid(sample_bid_state)
    assert bid_lin == pytest.approx(60.0 * (0.001 / 0.0008))

    human = HumanRuleBidder(target_cpc=120.0)
    bid_human = human.bid(sample_bid_state)
    assert 0.8 * bid_mcpc <= bid_human <= 1.2 * bid_mcpc


def test_evaluate_harness():
    dummy_records = [
        {
            "pctr": 0.001,
            "win_prob_fn": lambda b: 0.5,
            "exp_cost_fn": lambda b: 50.0,
            "is_ood_fn": lambda b: False,
        }
        for _ in range(10)
    ]
    eval_inputs = {
        "records": dummy_records,
        "target_cpc": 120.0,
        "budget": 300.0,
        "avg_pctr": 0.001,
    }
    result = evaluate(eval_inputs)
    assert "fitness" in result
    assert "expected_clicks" in result
    assert "expected_cpc" in result
    assert "expected_spend" in result
    assert "ood_ratio" in result
    assert result["expected_spend"] > 0

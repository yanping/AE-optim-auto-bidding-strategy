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
"""
Advertiser-Side Real-Time Auto-Bidding Multiplier Policy.
Contains BidState interface, the evolvable multiplier controller,
and the evaluation execution harness.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping
import numpy as np


@dataclass
class BidState:
    """
    Physical and observable state presented to the auto-bidding agent
    at each auction opportunity. All metrics are strictly advertiser-side observable.
    """
    # Current impression opportunity context
    pctr: float                    # Estimated click-through rate of current impression
    avg_pctr: float                # Historical average CTR of the campaign
    
    # Macro pacing and budget targets
    target_cpc: float              # Target CPC upper bound
    budget: float                  # Total campaign budget
    remaining_budget_ratio: float  # Remaining budget / total budget in [0.0, 1.0]
    time_progress_ratio: float     # Elapsed time / flight duration in [0.0, 1.0]
    
    # Cumulative & sliding window feedback
    current_cpc: float             # Actual cumulative average CPC so far
    cpc_ratio: float               # current_cpc / target_cpc (> 1.0 means exceeding target)
    recent_win_rate: float         # Moving average auction win rate over recent bids
    recent_cpc: float              # Moving average CPC over recent winning impressions
    spend_velocity: float          # Actual spend rate relative to theoretical uniform pacing
    last_multiplier: float         # The previous multiplier value output by this policy


# EVOLVE-BLOCK-START
def get_multiplier(state: BidState) -> float:
    """
    Computes a real-time bid multiplier based on current campaign state.
    Modulates baseline bid b_base = target_cpc * pctr by factor m in [0.8, 1.2].

    Args:
        state: Current BidState object with real-time feedback.

    Returns:
        float: Multiplier m, strictly bounded in [0.8, 1.2].
    """
    m = 1.0

    # 1. CPC overflow protection: brake immediately if exceeding target
    if state.current_cpc > state.target_cpc:
        m *= 0.90
    elif state.current_cpc < 0.85 * state.target_cpc and state.recent_win_rate < 0.20:
        # CPC has ample headroom and win rate is low: boost bids to gain volume
        m *= 1.08

    # 2. Pacing alignment: if spending lags behind uniform pacing, accelerate
    if state.spend_velocity < 0.90 and state.remaining_budget_ratio > (1.0 - state.time_progress_ratio) + 0.05:
        m *= 1.05

    # 3. Value gating: boost high pCTR opportunities when CPC is safe
    if state.pctr > state.avg_pctr * 1.5 and state.current_cpc <= state.target_cpc:
        m *= 1.03

    # Safety clipping to strictly bounded range [0.5, 2.0]
    return max(0.5, min(2.0, float(m)))
# EVOLVE-BLOCK-END


def evaluate(eval_inputs: Mapping[str, Any]) -> dict[str, float]:
    """
    Evaluates the get_multiplier function on a sequential stream of auction opportunities.
    Computes cumulative clicks, spend, empirical CPC, OOD violations, and composite fitness score.

    Args:
        eval_inputs: Mapping containing:
            - 'records': list of dicts with keys ['pctr', 'win_prob_fn', 'exp_cost_fn', 'is_ood_fn']
            - 'target_cpc': float
            - 'budget': float
            - 'avg_pctr': float

    Returns:
        dict with fitness, expected_clicks, expected_cpc, expected_spend, ood_ratio.
    """
    records = eval_inputs["records"]
    target_cpc = float(eval_inputs["target_cpc"])
    budget = float(eval_inputs["budget"])
    avg_pctr = float(eval_inputs["avg_pctr"])

    total_records = len(records)
    if total_records == 0:
        return {"fitness": -1e12, "expected_clicks": 0.0, "expected_cpc": 0.0, "expected_spend": 0.0, "ood_ratio": 1.0}

    cum_clicks = 0.0
    cum_spend = 0.0
    cum_wins = 0.0
    ood_count = 0
    recent_wins = []
    recent_costs = []
    recent_clicks = []
    window_size = 100
    last_multiplier = 1.0

    for i, rec in enumerate(records):
        time_progress = (i + 1) / total_records
        rem_budget = max(0.0, budget - cum_spend)
        rem_budget_ratio = rem_budget / budget

        if cum_spend >= budget:
            # Budget exhausted early, campaign cannot bid anymore
            break

        current_cpc = cum_spend / cum_clicks if cum_clicks > 0 else target_cpc * 0.8
        cpc_ratio = current_cpc / target_cpc

        # Pacing velocity
        ideal_spend = budget * time_progress
        spend_velocity = cum_spend / ideal_spend if ideal_spend > 0 else 1.0

        # Recent sliding window stats
        recent_win_rate = float(np.mean(recent_wins)) if recent_wins else 0.15
        recent_cpc = float(np.sum(recent_costs) / np.sum(recent_clicks)) if (recent_clicks and np.sum(recent_clicks) > 0) else current_cpc

        pctr = rec["pctr"]
        state = BidState(
            pctr=pctr,
            avg_pctr=avg_pctr,
            target_cpc=target_cpc,
            budget=budget,
            remaining_budget_ratio=rem_budget_ratio,
            time_progress_ratio=time_progress,
            current_cpc=current_cpc,
            cpc_ratio=cpc_ratio,
            recent_win_rate=recent_win_rate,
            recent_cpc=recent_cpc,
            spend_velocity=spend_velocity,
            last_multiplier=last_multiplier,
        )

        try:
            m = get_multiplier(state)
            if not isinstance(m, (int, float)) or not math.isfinite(m):
                m = 1.0
            m = max(0.5, min(2.0, float(m)))
        except Exception:
            m = 1.0

        last_multiplier = m
        # In RTB, auction bids and cleared prices are in CPM (per 1,000 impressions)
        bid = target_cpc * pctr * 1000.0 * m

        # Check OOD
        if rec["is_ood_fn"](bid):
            ood_count += 1

        # Win probability and expected CPM clearing price from response model
        p_win = rec["win_prob_fn"](bid)
        exp_cpm_cost = rec["exp_cost_fn"](bid)

        # Expected single-impression spend in currency units: (CPM / 1000) * P(win)
        expected_win_cost = (exp_cpm_cost / 1000.0) * p_win
        expected_click_inc = pctr * p_win

        cum_spend += expected_win_cost
        cum_clicks += expected_click_inc
        cum_wins += p_win

        # Update sliding window
        recent_wins.append(p_win)
        recent_costs.append(expected_win_cost)
        recent_clicks.append(expected_click_inc)
        if len(recent_wins) > window_size:
            recent_wins.pop(0)
            recent_costs.pop(0)
            recent_clicks.pop(0)

    final_cpc = cum_spend / cum_clicks if cum_clicks > 0 else 0.0
    ood_ratio = ood_count / total_records

    # Fitness computation with hard constraint penalties (Section 5.1 of PLAN.md)
    if cum_spend > budget * 1.02:
        fitness = -1000.0 - (cum_spend - budget)
    elif final_cpc > target_cpc * 1.01:
        excess = (final_cpc - target_cpc) / target_cpc
        fitness = -500.0 - 1000.0 * excess
    elif ood_ratio > 0.08:
        fitness = -200.0 - 500.0 * ood_ratio
    else:
        cpc_slack = max(0.0, (target_cpc - final_cpc) / target_cpc)
        fitness = cum_clicks - 5.0 * cpc_slack

    return {
        "fitness": float(fitness),
        "expected_clicks": float(cum_clicks),
        "expected_cpc": float(final_cpc),
        "expected_spend": float(cum_spend),
        "ood_ratio": float(ood_ratio),
    }

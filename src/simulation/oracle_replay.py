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
Oracle Offline A/B Counterfactual Replay Simulator.
Evaluates bidding policies on true, unmasked market clearing prices and click labels
from held-out test data (Day 7) under second-price auction mechanics.
"""

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.program import BidState
from src.bidding.baselines import BaseBidder

logger = logging.getLogger("alpha_evolve.oracle_replay")


@dataclass
class OracleReplayResult:
    """Summary metrics of an Oracle replay run."""
    policy_name: str
    total_auctions: int
    impressions_won: int
    win_rate: float
    total_clicks: int
    total_spend: float
    budget: float
    budget_utilization: float
    empirical_cpc: float
    target_cpc: float
    cpc_violation: bool
    cpc_headroom: float
    effective_cpm: float
    ctr_achieved: float
    budget_exhausted_step: Optional[int] = None
    hourly_spend: Optional[Dict[int, float]] = None
    hourly_clicks: Optional[Dict[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_auctions": self.total_auctions,
            "impressions_won": self.impressions_won,
            "win_rate": self.win_rate,
            "total_clicks": self.total_clicks,
            "total_spend": self.total_spend,
            "budget": self.budget,
            "budget_utilization": self.budget_utilization,
            "empirical_cpc": self.empirical_cpc,
            "target_cpc": self.target_cpc,
            "cpc_violation": self.cpc_violation,
            "cpc_headroom": self.cpc_headroom,
            "effective_cpm": self.effective_cpm,
            "ctr_achieved": self.ctr_achieved,
            "budget_exhausted_step": self.budget_exhausted_step,
            "hourly_spend": self.hourly_spend,
            "hourly_clicks": self.hourly_clicks,
        }


class OracleReplaySimulator:
    """
    Simulates second-price RTB auctions over chronological unmasked test impressions.
    
    Auction Rules:
    - If bid b_t > market_price z_t:
      - Advertiser wins the impression;
      - Paying cost = z_t / 1000.0 (converted from CPM to currency unit RMB);
      - Advertiser receives click if ground-truth click == 1.
    - If bid b_t <= market_price z_t:
      - Advertiser loses, cost = 0, click = 0.
    - Pacing / Budget:
      - If cumulative spend >= budget, bidder is out of budget; subsequent bids are 0.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        predicted_pctrs: np.ndarray,
        target_cpc: float = 120.0,
        budget: float = 20000.0,
        window_size: int = 200,
    ):
        """
        Args:
            test_df: DataFrame containing 'market_price' (or 'paying_price'), 'click', and optionally 'hour'.
            predicted_pctrs: 1D array of pCTR predictions aligned with test_df rows.
            target_cpc: Target maximum CPC in currency unit (RMB).
            budget: Campaign total budget in currency unit (RMB).
            window_size: Sliding window length for recent moving average statistics.
        """
        self.num_records = len(test_df)
        self.target_cpc = float(target_cpc)
        self.budget = float(budget)
        self.window_size = int(window_size)

        # Extract contiguous numpy arrays for maximum performance
        self.pctrs = np.asarray(predicted_pctrs, dtype=float)
        self.avg_pctr = float(np.mean(self.pctrs))

        if "market_price" in test_df.columns:
            self.market_prices = test_df["market_price"].to_numpy(dtype=float)
        elif "paying_price" in test_df.columns:
            self.market_prices = test_df["paying_price"].to_numpy(dtype=float)
        else:
            raise KeyError("test_df must contain 'market_price' or 'paying_price' column.")

        self.clicks = test_df["click"].to_numpy(dtype=int)
        self.hours = test_df["hour"].to_numpy(dtype=int) if "hour" in test_df.columns else np.zeros(self.num_records, dtype=int)

    def replay_policy(
        self,
        policy: Union[BaseBidder, Callable[[BidState], float]],
        policy_name: str = "Policy",
    ) -> OracleReplayResult:
        """
        Executes counterfactual second-price auction replay for a given bidding policy.

        Args:
            policy: BaseBidder instance (calling policy.bid(state))
                    or a multiplier function (get_multiplier(state) -> m).
            policy_name: Display name for the policy.
        """
        has_bid_method = hasattr(policy, "bid") and callable(getattr(policy, "bid"))
        is_multiplier_callable = callable(policy) and not has_bid_method

        # Pre-allocate tracking structures
        total_auctions = self.num_records
        cum_spend = 0.0
        cum_clicks = 0
        cum_wins = 0
        budget_exhausted_step: Optional[int] = None

        # Sliding window circular buffers
        w = self.window_size
        win_buf = np.zeros(w, dtype=float)
        spend_buf = np.zeros(w, dtype=float)
        click_buf = np.zeros(w, dtype=float)

        # Hourly breakdown
        hourly_spend: Dict[int, float] = {h: 0.0 for h in range(24)}
        hourly_clicks: Dict[int, int] = {h: 0 for h in range(24)}

        # Stateful BidState object reused across loop to minimize Python object creation
        state = BidState(
            pctr=0.0,
            avg_pctr=self.avg_pctr,
            target_cpc=self.target_cpc,
            budget=self.budget,
            remaining_budget_ratio=1.0,
            time_progress_ratio=0.0,
            current_cpc=self.target_cpc * 0.8,
            cpc_ratio=0.8,
            recent_win_rate=0.15,
            recent_cpc=self.target_cpc * 0.8,
            spend_velocity=1.0,
            last_multiplier=1.0,
        )

        for i in range(total_auctions):
            # 1. Budget Exhaustion Guard
            if cum_spend >= self.budget:
                if budget_exhausted_step is None:
                    budget_exhausted_step = i
                break

            pctr = self.pctrs[i]
            z_cpm = self.market_prices[i]
            hour = self.hours[i]

            # 2. Update BidState
            time_progress = (i + 1) / total_auctions
            rem_budget_ratio = max(0.0, self.budget - cum_spend) / self.budget
            expected_pacing_spend = self.budget * time_progress
            spend_velocity = (cum_spend / expected_pacing_spend) if expected_pacing_spend > 0 else 1.0

            current_cpc = (cum_spend / cum_clicks) if cum_clicks > 0 else (self.target_cpc * 0.8)
            cpc_ratio = current_cpc / self.target_cpc

            # Moving window calculations
            active_w = min(i, w)
            if active_w > 0:
                recent_win_rate = float(win_buf[:active_w].mean())
                w_clicks = click_buf[:active_w].sum()
                w_spend = spend_buf[:active_w].sum()
                recent_cpc = (w_spend / w_clicks) if w_clicks > 0 else current_cpc
            else:
                recent_win_rate = 0.15
                recent_cpc = current_cpc

            state.pctr = pctr
            state.remaining_budget_ratio = rem_budget_ratio
            state.time_progress_ratio = time_progress
            state.current_cpc = current_cpc
            state.cpc_ratio = cpc_ratio
            state.recent_win_rate = recent_win_rate
            state.recent_cpc = recent_cpc
            state.spend_velocity = spend_velocity

            # 3. Determine Bid in CPM
            if has_bid_method:
                bid_cpm = policy.bid(state)
            elif is_multiplier_callable:
                try:
                    m = policy(state)
                    m = max(0.5, min(2.0, float(m)))
                except Exception:
                    m = 1.0
                state.last_multiplier = m
                bid_cpm = self.target_cpc * pctr * 1000.0 * m
            else:
                bid_cpm = self.target_cpc * pctr * 1000.0

            # 4. Auction Resolution (Second-Price Rule)
            buf_idx = i % w
            if bid_cpm > z_cpm:
                # Won impression
                impr_cost = z_cpm / 1000.0
                impr_click = self.clicks[i]

                cum_spend += impr_cost
                cum_clicks += impr_click
                cum_wins += 1

                win_buf[buf_idx] = 1.0
                spend_buf[buf_idx] = impr_cost
                click_buf[buf_idx] = float(impr_click)

                hourly_spend[hour] = hourly_spend.get(hour, 0.0) + impr_cost
                hourly_clicks[hour] = hourly_clicks.get(hour, 0) + impr_click
            else:
                # Lost impression
                win_buf[buf_idx] = 0.0
                spend_buf[buf_idx] = 0.0
                click_buf[buf_idx] = 0.0

        # Final Metrics Computation
        win_rate = (cum_wins / total_auctions) if total_auctions > 0 else 0.0
        budget_utilization = (cum_spend / self.budget) if self.budget > 0 else 0.0
        empirical_cpc = (cum_spend / cum_clicks) if cum_clicks > 0 else 0.0
        cpc_violation = (empirical_cpc > self.target_cpc * 1.01) if cum_clicks > 0 else False
        cpc_headroom = (self.target_cpc - empirical_cpc) if cum_clicks > 0 else self.target_cpc
        effective_cpm = (cum_spend / cum_wins * 1000.0) if cum_wins > 0 else 0.0
        ctr_achieved = (cum_clicks / cum_wins) if cum_wins > 0 else 0.0

        return OracleReplayResult(
            policy_name=policy_name,
            total_auctions=total_auctions,
            impressions_won=cum_wins,
            win_rate=win_rate,
            total_clicks=cum_clicks,
            total_spend=cum_spend,
            budget=self.budget,
            budget_utilization=budget_utilization,
            empirical_cpc=empirical_cpc,
            target_cpc=self.target_cpc,
            cpc_violation=cpc_violation,
            cpc_headroom=cpc_headroom,
            effective_cpm=effective_cpm,
            ctr_achieved=ctr_achieved,
            budget_exhausted_step=budget_exhausted_step,
            hourly_spend=hourly_spend,
            hourly_clicks=hourly_clicks,
        )

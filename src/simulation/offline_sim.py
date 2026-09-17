"""
Learned Offline Simulator
Vectorized evaluation engine estimating campaign clicks, spend, and CPC
under censored market response and CTR models.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union
import numpy as np
import pandas as pd

from ..models.ctr_model import CTRModel
from ..models.market_model import KaplanMeierMarketModel


@dataclass
class SimulationResult:
    """Detailed outcomes of an offline simulation pass."""
    expected_clicks: float
    expected_spend: float
    expected_cpc: float
    lcb_clicks: float            # Conservative Lower Confidence Bound on clicks
    ucb_cpc: float               # Conservative Upper Confidence Bound on CPC
    expected_win_rate: float
    ood_ratio: float             # Ratio of bids falling outside historical support
    target_cpc: float
    budget: float
    is_cpc_valid: bool
    is_budget_valid: bool
    is_ood_valid: bool
    overall_valid: bool
    per_auction_clicks: np.ndarray
    per_auction_costs: np.ndarray
    bids: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_clicks": round(self.expected_clicks, 2),
            "expected_spend": round(self.expected_spend, 2),
            "expected_cpc": round(self.expected_cpc, 4),
            "lcb_clicks": round(self.lcb_clicks, 2),
            "ucb_cpc": round(self.ucb_cpc, 4),
            "expected_win_rate": round(self.expected_win_rate, 4),
            "ood_ratio": round(self.ood_ratio, 4),
            "target_cpc": self.target_cpc,
            "budget": self.budget,
            "is_cpc_valid": self.is_cpc_valid,
            "is_budget_valid": self.is_budget_valid,
            "is_ood_valid": self.is_ood_valid,
            "overall_valid": self.overall_valid,
        }


class LearnedOfflineSimulator:
    """
    Simulates advertiser campaign outcomes using learned CTR and market response models.
    Operates without access to hidden competitor bids or unobserved clearing prices.
    """

    def __init__(
        self,
        ctr_model: CTRModel,
        market_model: KaplanMeierMarketModel,
        default_target_cpc: float = 2.0,
        cpc_slack: float = 0.02,  # 2% allowed tolerance before hard failure
        max_allowed_ood_ratio: float = 0.10,
    ):
        self.ctr_model = ctr_model
        self.market_model = market_model
        self.default_target_cpc = default_target_cpc
        self.cpc_slack = cpc_slack
        self.max_allowed_ood_ratio = max_allowed_ood_ratio

    def evaluate_bids(
        self,
        contexts_df: pd.DataFrame,
        bids: np.ndarray,
        target_cpc: Optional[float] = None,
        budget: Optional[float] = None,
    ) -> SimulationResult:
        """
        Evaluate a candidate array of bids on a batch of auction contexts.

        Args:
            contexts_df: Context DataFrame (features).
            bids: 1D array of proposed bids for each auction.
            target_cpc: Target CPC constraint (default: self.default_target_cpc).
            budget: Total campaign budget (default: sum of empirical historical costs).
        """
        if target_cpc is None:
            target_cpc = self.default_target_cpc

        n_auctions = len(contexts_df)
        bids = np.asarray(bids, dtype=float)
        if len(bids) != n_auctions:
            raise ValueError(f"Length of bids ({len(bids)}) does not match contexts ({n_auctions}).")

        # 1. Predict CTR
        pctr = self.ctr_model.predict_proba(contexts_df)

        # 2. Predict Win Probabilities & Clearing Costs from Market Model
        win_probs, std_errors, ood_flags = self.market_model.predict_win_prob(contexts_df, bids)
        cond_costs = self.market_model.predict_expected_cost(contexts_df, bids)

        # 3. Expected per-auction clicks and spend
        # E[click_i] = P(win_i) * pCTR_i
        per_click = win_probs * pctr
        # E[spend_i] = P(win_i) * E[cost_i | win_i]
        per_spend = win_probs * cond_costs

        # Conservative confidence bounds (LCB on clicks, UCB on cost/CPC)
        lcb_win_probs = np.clip(win_probs - 1.96 * std_errors, 0.0, 1.0)
        ucb_win_probs = np.clip(win_probs + 1.96 * std_errors, 0.0, 1.0)

        lcb_per_click = lcb_win_probs * pctr
        ucb_per_spend = ucb_win_probs * cond_costs

        # 4. Aggregates
        total_clicks = float(np.sum(per_click))
        total_spend = float(np.sum(per_spend))
        expected_cpc = total_spend / (total_clicks + 1e-9)

        lcb_clicks = float(np.sum(lcb_per_click))
        ucb_spend = float(np.sum(ucb_per_spend))
        ucb_cpc = ucb_spend / (lcb_clicks + 1e-9)

        expected_win_rate = float(np.mean(win_probs))
        ood_ratio = float(np.mean(ood_flags))

        # Default budget if not specified: nominal budget assuming target CPC and target volume
        if budget is None:
            budget = total_spend * 1.05

        # 5. Validation checks
        is_cpc_valid = expected_cpc <= target_cpc * (1.0 + self.cpc_slack)
        is_budget_valid = total_spend <= budget * 1.01
        is_ood_valid = ood_ratio <= self.max_allowed_ood_ratio
        overall_valid = is_cpc_valid and is_budget_valid and is_ood_valid

        return SimulationResult(
            expected_clicks=total_clicks,
            expected_spend=total_spend,
            expected_cpc=expected_cpc,
            lcb_clicks=lcb_clicks,
            ucb_cpc=ucb_cpc,
            expected_win_rate=expected_win_rate,
            ood_ratio=ood_ratio,
            target_cpc=target_cpc,
            budget=budget,
            is_cpc_valid=is_cpc_valid,
            is_budget_valid=is_budget_valid,
            is_ood_valid=is_ood_valid,
            overall_valid=overall_valid,
            per_auction_clicks=per_click,
            per_auction_costs=per_spend,
            bids=bids,
        )

    def evaluate_policy(
        self,
        contexts_df: pd.DataFrame,
        policy_fn: Callable[[pd.DataFrame, np.ndarray, float], np.ndarray],
        target_cpc: Optional[float] = None,
        budget: Optional[float] = None,
    ) -> SimulationResult:
        """
        Evaluate a dynamic policy function.
        policy_fn signature: (contexts_df, pctr_array, target_cpc) -> bids_array
        """
        if target_cpc is None:
            target_cpc = self.default_target_cpc

        pctr = self.ctr_model.predict_proba(contexts_df)
        bids = policy_fn(contexts_df, pctr, target_cpc)
        return self.evaluate_bids(contexts_df, bids, target_cpc=target_cpc, budget=budget)

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
Standard RTB Auto-Bidding Baseline Strategies.
Implements:
1. McpcBidder: Maximum CPC heuristic based on expected impression value.
2. LinearBidder: Relative CTR linear bidding (academic standard).
3. HumanRuleBidder: Intuitive feedback controller (AlphaEvolve seed).
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.program import BidState


class BaseBidder(ABC):
    """Abstract base class for auto-bidding agents."""

    @abstractmethod
    def bid(self, state: BidState) -> float:
        """Computes bid in CPM for the given impression state."""
        pass


class McpcBidder(BaseBidder):
    """
    Maximum Cost-Per-Click Heuristic.
    Bids directly proportional to expected value:
    b_CPM = 1000 * pCTR * target_cpc
    """

    def __init__(self, target_cpc: float):
        self.target_cpc = float(target_cpc)

    def bid(self, state: BidState) -> float:
        return 1000.0 * state.pctr * self.target_cpc


class LinearBidder(BaseBidder):
    """
    Linear Bidding (LIN).
    Academic and industry benchmark:
    b_CPM = b_0 * (pCTR / avg_pctr)
    """

    def __init__(self, b0: float):
        self.b0 = float(b0)

    def bid(self, state: BidState) -> float:
        ratio = state.pctr / max(1e-6, state.avg_pctr)
        return self.b0 * ratio


class HumanRuleBidder(BaseBidder):
    """
    Human Heuristic Feedback Controller.
    Modulates expected value bid with rule-based multiplier.
    Serves as the AlphaEvolve evolutionary seed algorithm.
    """

    def __init__(self, target_cpc: float):
        self.target_cpc = float(target_cpc)

    def get_multiplier(self, state: BidState) -> float:
        m = 1.0
        if state.current_cpc > state.target_cpc:
            m *= 0.90
        elif state.current_cpc < 0.85 * state.target_cpc and state.recent_win_rate < 0.20:
            m *= 1.08
        if state.spend_velocity < 0.90 and state.remaining_budget_ratio > (1.0 - state.time_progress_ratio) + 0.05:
            m *= 1.05
        if state.pctr > state.avg_pctr * 1.5 and state.current_cpc <= state.target_cpc:
            m *= 1.03
        return max(0.8, min(1.2, float(m)))

    def bid(self, state: BidState) -> float:
        m = self.get_multiplier(state)
        return 1000.0 * state.pctr * self.target_cpc * m

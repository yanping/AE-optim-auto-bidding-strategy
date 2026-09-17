"""
Kaplan-Meier Survival Analysis Market Response Model
Learns market price distribution and win probability from right-censored bidding logs.
Does NOT require competitor bid data or global auction replay.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union


class SingleKMEstimator:
    """
    Kaplan-Meier product-limit estimator for a single traffic segment.
    Models the distribution of the latent market clearing price Z from censored auction observations.
    """

    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
        self.is_fitted = False
        
        # Fitted curves
        self.price_points: np.ndarray = np.array([])
        self.survival_probs: np.ndarray = np.array([])     # S(z) = P(Z > z)
        self.win_probs: np.ndarray = np.array([])          # F(b) = P(Z <= b) = 1 - S(b)
        self.cum_cost_weights: np.ndarray = np.array([])   # Sum_{z_k <= b} z_k * dF(z_k)
        self.greenwood_vars: np.ndarray = np.array([])     # Var(S(z)) via Greenwood's formula
        
        # Support range
        self.min_observed_bid: float = 0.0
        self.max_observed_bid: float = 0.0
        self.n_samples: int = 0
        self.n_events: int = 0  # Number of uncensored wins

    def fit(self, bids: np.ndarray, wins: np.ndarray, paying_prices: np.ndarray) -> "SingleKMEstimator":
        """
        Fit Kaplan-Meier estimator from advertiser observations.

        Args:
            bids: Array of submitted bids b_j.
            wins: Array of win indicators (1 if won, 0 if lost).
            paying_prices: Array of clearing prices if won, NaN if lost.
        """
        self.n_samples = len(bids)
        if self.n_samples < self.min_samples:
            self.is_fitted = False
            return self

        self.min_observed_bid = float(np.min(bids))
        self.max_observed_bid = float(np.max(bids))

        # Event time T_j: if won, T_j = paying_price; if lost, T_j = bid_price (censored)
        event_times = np.where(wins == 1, paying_prices, bids)
        
        # Clean any invalid values
        valid_mask = np.isfinite(event_times) & (event_times >= 0)
        times = event_times[valid_mask]
        events = wins[valid_mask]

        if len(times) == 0:
            self.is_fitted = False
            return self

        # Sort all observations by time
        # To avoid floating point discretization issues, round to 2 decimals
        times = np.round(times, 2)
        unique_times = np.unique(times)

        # Count events d_k and at-risk n_k
        d_k = np.zeros(len(unique_times))
        n_k = np.zeros(len(unique_times))

        # Calculate counts
        for k, t in enumerate(unique_times):
            d_k[k] = np.sum((times == t) & (events == 1))
            n_k[k] = np.sum(times >= t)

        # Filter out price points with zero risk set
        valid_k = n_k > 0
        unique_times = unique_times[valid_k]
        d_k = d_k[valid_k]
        n_k = n_k[valid_k]

        self.n_events = int(np.sum(d_k))

        # Kaplan-Meier product: S(t_k) = S(t_{k-1}) * (1 - d_k / n_k)
        hazard = np.where(n_k > 0, d_k / n_k, 0.0)
        # Numerical protection: hazard cannot exceed 1.0
        hazard = np.clip(hazard, 0.0, 1.0)
        
        survival = np.cumprod(1.0 - hazard)
        win_prob = 1.0 - survival

        # Greenwood's formula: Var(S(t)) = S(t)^2 * sum( d_i / (n_i * (n_i - d_i)) )
        denom = n_k * (n_k - d_k)
        var_terms = np.divide(d_k, denom, out=np.zeros_like(d_k, dtype=float), where=denom > 0)
        cum_var_sum = np.cumsum(var_terms)
        greenwood_var = (survival ** 2) * cum_var_sum

        # Expected clearing price under second-price auction:
        # E[Z | Z <= b] = ( sum_{z_i <= b} z_i * delta_F(z_i) ) / F(b)
        delta_f = np.diff(np.insert(win_prob, 0, 0.0))
        delta_f = np.maximum(0.0, delta_f)  # Ensure non-negative
        cum_cost_weights = np.cumsum(unique_times * delta_f)

        self.price_points = unique_times
        self.survival_probs = survival
        self.win_probs = win_prob
        self.cum_cost_weights = cum_cost_weights
        self.greenwood_vars = greenwood_var
        self.is_fitted = True
        return self

    def predict_win_prob(self, bids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate P(win | b) along with lower and upper confidence bounds.
        Returns:
            win_probs: 1D array of win probabilities F(b)
            std_errors: 1D array of standard errors
            ood_flags: 1D boolean array (True if bid is out-of-distribution)
        """
        if not self.is_fitted or len(self.price_points) == 0:
            return np.zeros_like(bids, dtype=float), np.zeros_like(bids, dtype=float), np.ones_like(bids, dtype=bool)

        bids = np.asarray(bids, dtype=float)
        
        # Out-of-distribution detection: negative bids or bids exceeding observed market support
        max_support = max(self.max_observed_bid, float(self.price_points[-1]) if len(self.price_points) > 0 else 300.0)
        ood_flags = (bids < 0.0) | (bids > max_support * 1.25)

        # Interpolate step-wise using searchsorted
        idx = np.searchsorted(self.price_points, bids, side="right") - 1
        
        # For bids below lowest price point: win_prob = 0
        win_probs = np.zeros(len(bids), dtype=float)
        std_errors = np.zeros(len(bids), dtype=float)

        valid_idx = idx >= 0
        valid_pos = np.clip(idx[valid_idx], 0, len(self.win_probs) - 1)
        win_probs[valid_idx] = self.win_probs[valid_pos]
        std_errors[valid_idx] = np.sqrt(np.maximum(0.0, self.greenwood_vars[valid_pos]))

        # For bids above highest price point: win_prob = max_win_prob
        high_idx = idx >= len(self.price_points)
        if np.any(high_idx):
            win_probs[high_idx] = self.win_probs[-1]
            std_errors[high_idx] = np.sqrt(np.maximum(0.0, self.greenwood_vars[-1]))

        win_probs = np.clip(win_probs, 0.0, 1.0)
        return win_probs, std_errors, ood_flags

    def predict_expected_cost(self, bids: np.ndarray) -> np.ndarray:
        """
        Estimate conditional clearing cost E[cost | win, b].
        Under second-price auction, clearing price cannot exceed bid b.
        """
        if not self.is_fitted or len(self.price_points) == 0:
            return np.zeros_like(bids, dtype=float)

        bids = np.asarray(bids, dtype=float)
        idx = np.searchsorted(self.price_points, bids, side="right") - 1

        expected_costs = np.zeros(len(bids), dtype=float)
        valid_idx = idx >= 0
        valid_pos = np.clip(idx[valid_idx], 0, len(self.cum_cost_weights) - 1)
        
        cum_weights = self.cum_cost_weights[valid_pos]
        f_b = self.win_probs[valid_pos]
        
        # Safe division
        safe_mask = f_b > 1e-6
        cond_costs = np.zeros_like(f_b)
        cond_costs[safe_mask] = cum_weights[safe_mask] / f_b[safe_mask]

        expected_costs[valid_idx] = cond_costs
        
        # Enforce second-price physical property: E[Z | Z <= b] <= b
        expected_costs = np.minimum(expected_costs, bids)
        expected_costs = np.maximum(0.0, expected_costs)
        return expected_costs


class KaplanMeierMarketModel:
    """
    Segment-Aware Kaplan-Meier Market Response Model.
    Maintains independent estimators per traffic segment (e.g. ad_exchange + time_bucket)
    plus a global fallback estimator.
    """

    def __init__(self, segment_columns: Optional[List[str]] = None, min_segment_samples: int = 50):
        if segment_columns is None:
            self.segment_columns = ["ad_exchange", "time_bucket"]
        else:
            self.segment_columns = segment_columns
            
        self.min_segment_samples = min_segment_samples
        self.segment_models: Dict[Tuple, SingleKMEstimator] = {}
        self.global_model = SingleKMEstimator(min_samples=20)
        self.is_fitted = False

    def _get_segment_key(self, row: pd.Series) -> Tuple:
        return tuple(row[col] for col in self.segment_columns)

    def fit(self, df: pd.DataFrame) -> "KaplanMeierMarketModel":
        """
        Fit Kaplan-Meier survival curves on advertiser-side data.
        Requires 'bid_price', 'win', and 'paying_price' columns.
        """
        bids = df["bid_price"].values
        wins = df["win"].values
        paying_prices = df["paying_price"].values

        # 1. Fit global fallback model
        self.global_model.fit(bids, wins, paying_prices)

        # 2. Fit per-segment models
        grouped = df.groupby(self.segment_columns)
        for seg_key, group in grouped:
            estimator = SingleKMEstimator(min_samples=self.min_segment_samples)
            estimator.fit(
                group["bid_price"].values,
                group["win"].values,
                group["paying_price"].values,
            )
            if estimator.is_fitted:
                # Key as tuple
                key = (seg_key,) if not isinstance(seg_key, tuple) else seg_key
                self.segment_models[key] = estimator

        self.is_fitted = True
        return self

    def predict_win_prob(
        self, df: pd.DataFrame, bids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict P(win | bid, segment) for each row in df.
        Returns:
            win_probs: 1D array of estimated win probabilities
            std_errors: 1D array of Greenwood standard errors
            ood_flags: 1D boolean array indicating out-of-distribution bids
        """
        if not self.is_fitted:
            raise RuntimeError("KaplanMeierMarketModel must be fitted before predict_win_prob.")

        n = len(df)
        bids = np.asarray(bids, dtype=float)
        win_probs = np.zeros(n, dtype=float)
        std_errors = np.zeros(n, dtype=float)
        ood_flags = np.zeros(n, dtype=bool)

        # Group by segment for fast batch prediction
        seg_keys = df[self.segment_columns].itertuples(index=False, name=None)
        
        # Unique segment keys in this batch
        df_temp = pd.DataFrame({"seg_key": list(seg_keys), "bid": bids, "orig_idx": np.arange(n)})
        
        for key, group in df_temp.groupby("seg_key"):
            estimator = self.segment_models.get(key, self.global_model)
            group_bids = group["bid"].values
            orig_indices = group["orig_idx"].values
            
            p, se, ood = estimator.predict_win_prob(group_bids)
            win_probs[orig_indices] = p
            std_errors[orig_indices] = se
            ood_flags[orig_indices] = ood

        return win_probs, std_errors, ood_flags

    def predict_expected_cost(self, df: pd.DataFrame, bids: np.ndarray) -> np.ndarray:
        """
        Predict expected clearing cost E[cost | win, bid, segment].
        """
        if not self.is_fitted:
            raise RuntimeError("KaplanMeierMarketModel must be fitted before predict_expected_cost.")

        n = len(df)
        bids = np.asarray(bids, dtype=float)
        costs = np.zeros(n, dtype=float)

        seg_keys = df[self.segment_columns].itertuples(index=False, name=None)
        df_temp = pd.DataFrame({"seg_key": list(seg_keys), "bid": bids, "orig_idx": np.arange(n)})

        for key, group in df_temp.groupby("seg_key"):
            estimator = self.segment_models.get(key, self.global_model)
            group_bids = group["bid"].values
            orig_indices = group["orig_idx"].values
            
            c = estimator.predict_expected_cost(group_bids)
            costs[orig_indices] = c

        return costs

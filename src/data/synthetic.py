"""
Synthetic RTB Data Generator
Generates realistic display advertising bidding logs with censored feedback,
simulating iPinYou statistical properties (lognormal market prices, power-law CTR, second-price auctions).
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_rtb_data(
    n_samples: int = 50000,
    n_days: int = 7,
    target_cpc: float = 2.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic RTB auction logs.

    Args:
        n_samples: Total number of auction opportunities.
        n_days: Duration in days (e.g. 7 days).
        target_cpc: Base target CPC in currency units (e.g. 2.0).
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame containing full auction records with ground truth market prices.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Temporal features
    days = rng.integers(1, n_days + 1, size=n_samples)
    hours = rng.integers(0, 24, size=n_samples)
    # Time buckets: 0: Night (0-5), 1: Morning (6-11), 2: Afternoon (12-17), 3: Evening (18-23)
    time_buckets = hours // 6
    # Millisecond timestamps across the days
    timestamps = (days - 1) * 86400 + hours * 3600 + rng.integers(0, 3600, size=n_samples)

    # 2. Context features
    exchanges = rng.choice(["exchange_1", "exchange_2", "exchange_3"], size=n_samples, p=[0.5, 0.3, 0.2])
    visibilities = rng.choice(["FirstView", "OtherView"], size=n_samples, p=[0.4, 0.6])
    formats = rng.choice(["Banner", "Native", "Video"], size=n_samples, p=[0.6, 0.3, 0.1])
    user_tags = rng.choice(["gamer", "shopper", "tech", "finance", "general"], size=n_samples, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    
    # Floor prices (in CPM or arbitrary ad unit, e.g. 1.0 to 15.0)
    floor_base = {"exchange_1": 2.0, "exchange_2": 3.0, "exchange_3": 5.0}
    floor_prices = np.array([floor_base[ex] + rng.uniform(0.0, 3.0) for ex in exchanges])

    # 3. True CTR generation (Logit model)
    # Base log-odds around -4.0 (mean CTR ~ 1.8%)
    logit_ctr = -4.0 + (
        (formats == "Video") * 0.8 +
        (formats == "Native") * 0.4 +
        (visibilities == "FirstView") * 0.5 +
        (user_tags == "gamer") * 0.3 +
        (user_tags == "shopper") * 0.25 +
        (time_buckets == 3) * 0.2 +  # Higher CTR in evening
        rng.normal(0, 0.2, size=n_samples)
    )
    true_pctr = 1.0 / (1.0 + np.exp(-logit_ctr))
    # Clip to realistic range [0.005, 0.08]
    true_pctr = np.clip(true_pctr, 0.005, 0.08)

    # 4. True Market Price Z (Lognormal distribution conditioned on context)
    # Higher for Video, FirstView, and Evening
    mu_logz = 1.8 + (
        (formats == "Video") * 0.7 +
        (formats == "Native") * 0.3 +
        (visibilities == "FirstView") * 0.4 +
        (exchanges == "exchange_3") * 0.3 +
        (time_buckets == 3) * 0.2
    )
    sigma_logz = 0.5
    raw_market_price = rng.lognormal(mean=mu_logz, sigma=sigma_logz, size=n_samples)
    # Market price must respect floor price
    market_price = np.maximum(raw_market_price, floor_prices)
    market_price = np.round(market_price, 2)

    # 5. Baseline Bidder behavior (produces historical logs)
    # Baseline bid: b = pCTR * target_CPC * multiplier * scaling_factor
    # In RTB, CPM pricing is often in 1000s; let's scale so bid price is in range [2, 30]
    # base value per impression = pCTR * target_cpc * 100
    base_val = true_pctr * target_cpc * 150.0
    # Natural exploration in historical policy: +/- 15% noise
    bid_multipliers = rng.normal(1.0, 0.12, size=n_samples)
    bid_multipliers = np.clip(bid_multipliers, 0.75, 1.25)
    bid_prices = np.round(np.maximum(floor_prices, base_val * bid_multipliers), 2)

    # 6. Second-Price Auction Outcome
    # Advertiser wins if bid_price > market_price (standard strict inequality)
    win = (bid_prices > market_price).astype(int)
    
    # In second-price auction, winner pays market price (clearing price Z)
    paying_prices = np.where(win == 1, market_price, np.nan)
    costs = np.where(win == 1, paying_prices, 0.0)

    # Click realization: only won impressions have chance to click
    click_draw = rng.random(size=n_samples)
    clicks = ((win == 1) & (click_draw < true_pctr)).astype(int)

    # 7. Assemble DataFrame
    bid_ids = [f"bid_{i:08d}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        "bid_id": bid_ids,
        "timestamp": timestamps,
        "day": days,
        "hour": hours,
        "time_bucket": time_buckets,
        "ad_exchange": exchanges,
        "ad_slot_visibility": visibilities,
        "ad_slot_format": formats,
        "ad_slot_floor_price": np.round(floor_prices, 2),
        "user_tag": user_tags,
        "true_pctr": true_pctr,
        "market_price": market_price,  # Ground truth Z
        "bid_price": bid_prices,        # Advertiser's historical bid b
        "win": win,                    # Win feedback
        "paying_price": paying_prices, # Visible on win, censored on loss
        "cost": costs,                 # Actual spend
        "click": clicks,               # Click outcome
    })

    # Sort chronologically
    df = df.sort_values(by=["timestamp", "bid_id"]).reset_index(drop=True)
    return df

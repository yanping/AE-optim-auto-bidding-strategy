"""
Data Schema and Dataclass Definitions for RTB Auctions
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class AuctionRecord:
    """Represents a single ad auction request context from the advertiser's perspective."""
    bid_id: str
    timestamp: int
    day: int
    hour: int
    time_bucket: int  # 0: Night (0-5), 1: Morning (6-11), 2: Afternoon (12-17), 3: Evening (18-23)
    ad_exchange: str
    ad_slot_visibility: str
    ad_slot_format: str
    ad_slot_floor_price: float
    user_tag: str
    true_pctr: Optional[float] = None
    market_price: Optional[float] = None  # Ground truth market price Z (only available in oracle data)


@dataclass
class BidLog:
    """Represents the advertiser's bid decision and resulting auction feedback."""
    bid_id: str
    bid_price: float
    win: int                       # 1 if won, 0 if lost
    paying_price: Optional[float]  # Actual clearing cost (None if lost in advertiser view)
    click: int                     # 1 if clicked, 0 otherwise
    cost: float                    # Actual charge to advertiser (0.0 if lost)


@dataclass
class DatasetSplit:
    """Container for chronological dataset splits."""
    train_advertiser: pd.DataFrame
    val_advertiser: pd.DataFrame
    test_advertiser: pd.DataFrame
    train_oracle: pd.DataFrame
    val_oracle: pd.DataFrame
    test_oracle: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

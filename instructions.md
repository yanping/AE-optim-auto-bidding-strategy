Evolve an advertiser-side real-time auto-bidding multiplier algorithm to maximize
total clicks under a Target CPC constraint and budget constraint in second-price RTB auctions.

## Problem Description

In online advertising real-time bidding (RTB), advertisers compete in second-price auctions.
The advertiser's goal is to maximize total clicks subject to two critical business constraints:
1. **Target CPC Constraint**: The empirical average Cost-Per-Click must satisfy $\text{CPC} \le \text{CPC}_{\text{target}}$.
2. **Budget Constraint**: Total spend must not exceed the campaign budget $B$ over the flight.

Each incoming auction opportunity has an estimated click-through rate ($p\text{CTR}$).
The baseline expected-value bid is:
$$b_{\text{base}} = \text{target\_cpc} \times p\text{CTR}$$

Your task is to evolve a feedback-control multiplier function:
```python
def get_multiplier(state: BidState) -> float
```
which modulates the base bid as $b = b_{\text{base}} \times m$ where $m \in [0.8, 1.2]$.

"Better" means achieving significantly more total clicks while strictly respecting the Target CPC constraint and budget pacing.

## Function Signature & State Definition

```python
from dataclasses import dataclass

@dataclass
class BidState:
    # Impression value context
    pctr: float                    # Estimated CTR of current opportunity (e.g. 0.0008)
    avg_pctr: float                # Historical average CTR of the campaign (e.g. 0.0008)
    
    # Macro pacing & budget targets
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
```

The function signature to optimize is:
```python
def get_multiplier(state: BidState) -> float:
    """
    Computes a real-time bid multiplier based on current campaign state.
    
    Args:
        state: Current BidState object.
        
    Returns:
        float: Multiplier m, clipped to [0.8, 1.2].
    """
```

## Constraints & Requirements

- **Output Bounds**: The returned multiplier must be bounded within $[0.5, 2.0]$. Always return `max(0.5, min(2.0, m))`.
- **Numerical Stability**: Must handle edge cases (e.g. zero division, extreme values). All outputs must be finite numbers (no `nan`, `inf`).
- **Target CPC Discipline**: If `current_cpc > target_cpc` or `recent_cpc > target_cpc`, the policy MUST aggressively throttle bids to prevent violation.
- **Budget Pacing**: If `remaining_budget_ratio < 1.0 - time_progress_ratio` (over-spending), reduce multiplier. If under-spending with safe CPC, raise multiplier to acquire volume.
- **Allowed Libraries**: `math`, `numpy` (as `np`), standard Python builtins.

## Evaluation & Scoring

Candidates are evaluated by an offline simulator on real RTB auction streams (iPinYou dataset):
- **Hard Disqualification (Large Negative Penalties)**:
  - Budget exceeded by >2%: Score $\le -1000$
  - Cumulative CPC exceeded Target CPC by >1%: Score $\le -500$
  - Decision out-of-distribution ratio > 8%: Score $\le -200$
- **Qualified Strategies**:
  $$\text{Fitness} = \text{Expected Clicks} - 5.0 \times \max\left(0, \frac{\text{target\_cpc} - \text{current\_cpc}}{\text{target\_cpc}}\right)$$
  The primary objective is maximizing clicks, while penalizing excessive under-spending.

## Strategies to Explore

1. **Non-linear Smooth Feedback**: Replace hard `if/else` jumps with smooth mathematical functions like `math.tanh` or sigmoid to avoid high-frequency market oscillation.
2. **Asymmetric Risk Response**: Brake sharply when `current_cpc` approaches `target_cpc`, but accelerate smoothly when CPC headroom exists.
3. **Value-Dependent Scaling**: High-quality traffic ($p\text{CTR} > \text{avg\_pctr}$) warrants higher win priority when budget is healthy; low-quality traffic should be discounted.
4. **Temporal Momentum / Damping**: Incorporate `state.last_multiplier` with exponential smoothing: $m = \alpha \cdot m_{\text{target}} + (1 - \alpha) \cdot m_{\text{prev}}$.
5. **Phase-Aware Pacing**: In early flight ($t < 0.2$), explore conservatively; in mid-late flight, tightly couple spend velocity to remaining time.

## Baselines

The seed program is a simple heuristic feedback rule with linear adjustments. It achieves baseline clicks but suffers from abrupt step-wise reactions and sub-optimal budget pacing. Advanced evolved algorithms can discover non-linear controllers that yield 15%+ more clicks while safely staying under the CPC cap.

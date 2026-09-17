# AlphaEvolve RTB Auto-Bidding: Objective Function Design & Fitness Calculation Methodology

**English** | [简体中文](fitness_and_objective_design_CN.md)

> **Document Scope**: This document details the **end-to-end calculation pipeline of the Fitness Score** used during evolutionary search. Drawing from operations research, computational advertising, game theory, and symbolic LLM code evolution, it provides a rigorous theoretical rationale for **why raw click volume cannot serve as the direct optimization objective**. Intended for technical retrospectives, academic discussions, and executive presentations.

---

## Table of Contents
- [1. Business Formulation & Mathematical Framing](#1-business-formulation--mathematical-framing)
- [2. End-to-End Fitness Score Calculation Pipeline](#2-end-to-end-fitness-score-calculation-pipeline)
  - [2.1 Simulation Horizon & Stream Processing](#21-simulation-horizon--stream-processing)
  - [2.2 Market Response & Second-Price Clearing Settlement](#22-market-response--second-price-clearing-settlement)
  - [2.3 Piecewise Fitness Function with Hard Penalty Cliffs](#23-piecewise-fitness-function-with-hard-penalty-cliffs)
- [3. Deep Analysis: Why Raw Click Volume Fails as an Objective](#3-deep-analysis-why-raw-click-volume-fails-as-an-objective)
  - [Dimension 1: Constrained Optimization Reality vs. Unconstrained Trap (Rogue Bidding Defense)](#dimension-1-constrained-optimization-reality-vs-unconstrained-trap-rogue-bidding-defense)
  - [Dimension 2: Goodhart's Law & Simulator Exploitation (OOD Score Inflation Defense)](#dimension-2-goodharts-law--simulator-exploitation-ood-score-inflation-defense)
  - [Dimension 3: Marginal Capital Efficiency — Balancing Over-Spend vs. Under-Delivery](#dimension-3-marginal-capital-efficiency--balancing-over-spend-vs-under-delivery)
  - [Dimension 4: Operations Research Penalty Method Guiding LLM Code Search](#dimension-4-operations-research-penalty-method-guiding-llm-code-search)
- [4. Executive Reference: Comparison Matrix & Decision Rationale](#4-executive-reference-comparison-matrix--decision-rationale)
- [5. Source Code Mapping & Traceability](#5-source-code-mapping--traceability)

---

## 1. Business Formulation & Mathematical Framing

In real-time bidding (RTB) digital advertising, advertisers participate in millisecond-level generalized second-price auctions via Demand-Side Platforms (DSPs). Advertisers specify a flight budget (e.g., daily budget $B$) and a target cost per acquisition or click (e.g., Target CPC ceiling $\mathrm{CPC}_{\mathrm{target}}$).

Consequently, the core auto-bidding task constitutes a **Constrained Stochastic Optimization Problem**:

$$
\begin{aligned}
\max_{\pi} \quad & \sum_{t=1}^{T} \mathrm{Click}_t(b_t) \\
\text{s.t.} \quad & \sum_{t=1}^{T} \mathrm{Cost}_t(b_t) \le B \quad \text{(Global Budget Constraint)} \\
& \frac{\sum_{t=1}^{T} \mathrm{Cost}_t(b_t)}{\sum_{t=1}^{T} \mathrm{Click}_t(b_t)} \le \mathrm{CPC}_{\mathrm{target}} \quad \text{(Target CPC Constraint)} \\
& b_t \in \mathrm{Support}(\mathcal{D}_{\mathrm{history}}) \quad \text{(Distribution Support Constraint)}
\end{aligned}
$$

Where:
* $\pi$ is the adaptive multiplier policy function `get_multiplier(state: BidState) -> float`;
* $b_t = \mathrm{CPC}_{\mathrm{target}} \times p\mathrm{CTR}_t \times 1000 \times m_t$ represents the effective bid price submitted to the exchange (CPM);
* $\mathrm{Click}_t \in \{0, 1\}$, and $\mathrm{Cost}_t$ is the second-highest bid clearing price.

---

## 2. End-to-End Fitness Score Calculation Pipeline

Evaluation runs within an AST-isolated sandbox executed by [`src/evaluate.py`](../src/evaluate.py) and [`src/program.py`](../src/program.py) through three distinct stages:

### 2.1 Simulation Horizon & Stream Processing
* **Evaluation Dataset**: Day 6 validation split (sampled stream of $N = 5,000$ independent auctions), processed strictly chronologically;
* **Physical State Construction (`BidState`)**: For each auction $t$, the system constructs observable features:
  * **Request-Level Signals**: Current estimated click-through rate $p\mathrm{CTR}_t$, historical average click-through rate $\overline{p\mathrm{CTR}}$;
  * **Pacing & Horizon Signals**: Remaining budget ratio `rem_budget_ratio` $\in [0, 1]$, time progress ratio `time_progress_ratio` $\in [0, 1]$;
  * **Spend Velocity Feedback**: Ratio of actual spend to ideal linear pacing $\mathrm{spend\_velocity} = \frac{\mathrm{Spend}_{\mathrm{actual}}}{\mathrm{Spend}_{\mathrm{ideal}}}$;
  * **Cost Control Signals**: Cumulative CPC `current_cpc`, cost ratio $\mathrm{cpc\_ratio} = \frac{\mathrm{CPC}_{\mathrm{current}}}{\mathrm{CPC}_{\mathrm{target}}}$;
  * **Rolling Window Statistics ($W=100$)**: Short-term win rate `recent_win_rate`, short-term clearing price `recent_cpc`;
  * **Temporal Action Memory**: Previous bidding multiplier `last_multiplier`.

### 2.2 Market Response & Second-Price Clearing Settlement
Candidate code `get_multiplier(state)` outputs multiplier $m_t \in [0.5, 2.0]$, converted to CPM bid $b_t$. The learned offline market response model evaluates outcomes:
* **Win Probability**: Predicted via non-parametric Kaplan-Meier survival modeling (incorporating right-censoring): $P(\mathrm{win} \mid b_t)$;
* **Clearing Cost**: Expected second-price payment conditional on winning: $\mathbb{E}[\mathrm{Cost} \mid \mathrm{win}, b_t]$;
* **Incremental Expected Spend**: $\Delta \mathrm{Spend} = \frac{\mathbb{E}[\mathrm{Cost}]}{1000} \times P(\mathrm{win} \mid b_t)$;
* **Incremental Expected Clicks**: $\Delta \mathrm{Clicks} = p\mathrm{CTR}_t \times P(\mathrm{win} \mid b_t)$;
* **Out-of-Distribution (OOD) Tracking**: Bids exceeding empirical support boundaries increment `ood_count`.

Execution loops until the auction horizon finishes or remaining budget reaches zero (triggering early termination).

### 2.3 Piecewise Fitness Function with Hard Penalty Cliffs
At the conclusion of the simulation stream, summary metrics are computed: `cum_spend`, `cum_clicks`, `final_cpc = cum_spend / cum_clicks`, and `ood_ratio = ood_count / N`. The fitness score evaluates according to:

$$
\mathrm{Fitness} = 
\begin{cases} 
-1000.0 - (\mathrm{Spend}_{\mathrm{cum}} - B) & \text{if } \mathrm{Spend}_{\mathrm{cum}} > 1.02 \cdot B \quad \text{[Budget Overrun Disqualification]} \\
-500.0 - 1000.0 \times \frac{\mathrm{CPC}_{\mathrm{final}} - \mathrm{CPC}_{\mathrm{target}}}{\mathrm{CPC}_{\mathrm{target}}} & \text{if } \mathrm{CPC}_{\mathrm{final}} > 1.01 \cdot \mathrm{CPC}_{\mathrm{target}} \quad \text{[CPC Violation Disqualification]} \\
-200.0 - 500.0 \times \mathrm{Ratio}_{\mathrm{ood}} & \text{if } \mathrm{Ratio}_{\mathrm{ood}} > 0.08 \quad \text{[OOD Outlier Disqualification]} \\
\mathrm{Clicks}_{\mathrm{cum}} - 5.0 \times \frac{\mathrm{CPC}_{\mathrm{target}} - \mathrm{CPC}_{\mathrm{final}}}{\mathrm{CPC}_{\mathrm{target}}} & \text{otherwise} \quad \text{[Qualified Feasible Policy]}
\end{cases}
$$

#### Source Implementation (`src/program.py` Lines 195-206):
```python
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
```

---

## 3. Deep Analysis: Why Raw Click Volume Fails as an Objective

A common intuition when framing bidding optimization is: *"If the ultimate goal is acquiring conversions, why not set the fitness objective directly to $\mathrm{Fitness} = \mathrm{Clicks}_{\mathrm{cum}}$?"*

Optimizing raw clicks without structural penalties triggers four severe failure modes:

### Dimension 1: Constrained Optimization Reality vs. Unconstrained Trap (Rogue Bidding Defense)
* **The Rogue Strategy Shortcut**: Without hard penalty boundaries, LLMs quickly identify an unconstrained exploit:
  * Lock multipliers to the ceiling $m = 2.0$ or higher;
  * Outbid all competitors on the first 100-200 requests, winning impressions with ~100% win rate and securing 3-5 quick clicks;
  * Exhaust the entire campaign budget in minutes and exit the auction prematurely.
* **Commercial Fallout**:
  * While securing initial clicks, the effective CPC explodes to **300-500 RMB** (against a 120 RMB target ceiling);
  * Deploying such an unconstrained policy in production results in complete budget drain within hours and catastrophic ROI collapse.

### Dimension 2: Goodhart's Law & Simulator Exploitation (OOD Score Inflation Defense)
> **Goodhart's Law**: *"When a measure becomes a target, it ceases to be a good measure."*

* **Extrapolation Limits**: The Kaplan-Meier market response model is fitted on historical delivery logs subject to right-censoring. Its statistical validity is strictly confined to the empirical support of observed bids;
* **Adversarial Code Exploitation**: If optimizing raw clicks, LLM code mutators act like security fuzzers, synthesizing expressions that trigger edge-case mathematical artifacts in the simulator (such as probability extrapolation in high-bid domains);
* **Simulator Illusion vs. Live Collapse**: Such candidates appear to achieve massive "expected clicks" in simulation, but collapse catastrophically when evaluated against Day 7 Oracle Ground Truth;
* **The OOD Defense**: Incorporating a steep cliff penalty (`ood_ratio > 0.08` triggering a -200 score deduction) eliminates speculative gaming and forces the LLM to search within credible price supports.

### Dimension 3: Marginal Capital Efficiency — Balancing Over-Spend vs. Under-Delivery
In commercial DSP operations, **severe under-delivery is just as damaging as a budget overrun**:
* Without continuous guidance, evolutionary search easily falls into excessive conservatism: lowering all multipliers to the floor ($m = 0.5$) to avoid risk;
* The policy spends only 5% of its budget, achieving a misleadingly low CPC (e.g. 10 RMB), but capturing virtually no clicks, sabotaging customer acquisition targets;
* **The Slack Incentive Mechanism**:

$$
\mathrm{Slack}_{\mathrm{cpc}} = \frac{\mathrm{CPC}_{\mathrm{target}} - \mathrm{CPC}_{\mathrm{final}}}{\mathrm{CPC}_{\mathrm{target}}}
$$

  By incorporating $-5.0 \times \mathrm{Slack}_{\mathrm{cpc}}$, the formulation penalizes idle capital opportunity cost:
  * When a policy operates safely below the Target CPC ceiling, the slack penalty incentivizes proactive volume expansion;
  * Pushing the effective CPC closer to the target threshold to capture maximum scale.

### Dimension 4: Operations Research Penalty Method Guiding LLM Code Search
AlphaEvolve's code mutation loop is fundamentally an **unconstrained global search over a non-convex discrete code space**.
* **Mathematical Foundation**: In non-differentiable program synthesis, the **Exterior Penalty Method** is the gold standard for enforcing hard constraints;
* **Shaping the Penalty Landscape**:
  * High-magnitude negative scores (`-1000`, `-500`, `-200`) establish steep repulsive walls around the feasible region;
  * **Stage 1 (Generations 1-20)**: Diagnostic insights (`Budget Overrun`, `Target CPC Exceeded`) steer the population toward the feasible region;
  * **Stage 2 (Generations 20-200)**: Within the feasible region, penalties drop to zero, seamlessly transitioning the objective into smooth multi-objective optimization along the Pareto frontier.

---

## 4. Executive Reference: Comparison Matrix & Decision Rationale

### 4.1 Objective Function Design Comparison Matrix

| Candidate Approach | Mathematical Formulation | Convergence Behavior | Day 7 Oracle Ground-Truth Outcome | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: Naive Clicks** | $`\mathrm{Fitness} = \mathrm{Clicks}_{\mathrm{cum}}`$ | Exploits simulator edge cases with ceiling bids | Budget drained in hours; CPC >300 RMB; disastrous failure | ❌ **Non-Viable** |
| **Option B: Pure Cost Target** | $`\mathrm{Fitness} = - \lvert \mathrm{CPC}_{\mathrm{final}} - \mathrm{CPC}_{\mathrm{target}} \rvert`$ | Excessively conservative; purchases long-tail traffic | Budget delivery <10%; fails business acquisition volume | ❌ **Commercially Unviable** |
| **Option C: Piecewise Penalty Fitness (Ours)** | **Penalty Cliffs + Click Maximization + Slack Pacing** | Feasible convergence within 20 gens; smooth cybernetic discovery | **+16.6% Real Clicks; 78.76 RMB CPC honoring 120 RMB cap; 100% compliant** |  **Optimal Production Choice** |

### 4.2 Key Performance Highlights
* **Fitness Trajectory**: Advanced from **0.2088** (Human Heuristic Seed) to **1.3520** (**+547% gain**);
* **Oracle Test Verification**:
  * Real Won Clicks expanded from 251 to **288** (+16.6% vs. Mcpc baseline, +14.7% vs. Human Heuristic);
  * Real clearing CPC held firmly at **78.76 RMB**, well below the **120.00 RMB** ceiling;
  * Verified multi-objective alignment under strict physical and commercial constraints.

---

## 5. Source Code Mapping & Traceability

For code inspection and reproducibility, refer to the following relative repository paths:

1. **Fitness Evaluation Implementation**:
   * Simulation & Penalty Logic: [`src/program.py` (Lines 85-214)](../src/program.py#L85-L214)
   * Evaluation Harness & Diagnostic Logging: [`src/evaluate.py` (Lines 134-258)](../src/evaluate.py#L134-L258)
2. **Automated Unit Tests & Assertion Suites**:
   * Evaluation Harness & Penalty Asserts: [`tests/test_evolution.py`](../tests/test_evolution.py)
   * State Bounds & Multiplier Invariants: [`tests/test_bidding.py`](../tests/test_bidding.py)
3. **Oracle Verification & Reporting**:
   * Second-Price Clearing Replay: [`src/bidding/benchmark.py`](../src/bidding/benchmark.py)
   * Automated Report Generation: [`src/report.py`](../src/report.py)

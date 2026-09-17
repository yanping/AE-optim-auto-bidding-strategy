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
Comprehensive Benchmark & Oracle Ground Truth Evaluation Suite.
Compares Mcpc, Linear, Human Rule, and AlphaEvolve Champion policies
across both Simulated Validation (Day 6) and Real Oracle A/B Replay (Day 7).
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.bidding.baselines import BaseBidder, McpcBidder, LinearBidder, HumanRuleBidder
from src.models.ctr_model import CTRModel
from src.models.market_model import KaplanMeierMarketModel
from src.program import BidState
from src.simulation.oracle_replay import OracleReplaySimulator, OracleReplayResult
from src.utils.task_manager import resolve_task_artifacts_dir, update_latest_pointer

logger = logging.getLogger("alpha_evolve.benchmark")


def load_champion_function(champion_path: Path) -> Callable[[BidState], float]:
    """Loads the evolved get_multiplier function from champion code file."""
    if not champion_path.exists():
        logger.warning("Champion code file %s not found. Falling back to src.program.", champion_path)
        from src.program import get_multiplier
        return get_multiplier

    with open(champion_path, "r", encoding="utf-8") as f:
        code = f.read()

    exec_ns = {
        "np": np,
        "math": math,
        "BidState": BidState,
    }
    exec(code, exec_ns)
    if "get_multiplier" in exec_ns and callable(exec_ns["get_multiplier"]):
        return exec_ns["get_multiplier"]
    else:
        from src.program import get_multiplier
        return get_multiplier


def tune_linear_b0(
    val_df: pd.DataFrame,
    val_pctrs: np.ndarray,
    target_cpc: float,
    budget: float,
) -> float:
    """Finds best base bid b0 for Linear Bidding on validation dataset."""
    val_df_copy = val_df.copy()
    if "market_price" not in val_df_copy.columns and "paying_price" in val_df_copy.columns:
        val_df_copy["market_price"] = val_df_copy["paying_price"]

    sim = OracleReplaySimulator(val_df_copy, val_pctrs, target_cpc=target_cpc, budget=budget)
    best_b0 = 80.0
    best_clicks = -1

    candidates = [40.0, 60.0, 80.0, 95.0, 110.0, 130.0]
    for b0 in candidates:
        bidder = LinearBidder(b0=b0)
        res = sim.replay_policy(bidder, f"Linear-{b0}")
        if not res.cpc_violation and res.total_clicks > best_clicks:
            best_clicks = res.total_clicks
            best_b0 = b0

    logger.info("Tuned LinearBidder b0 = %.1f (Val Clicks: %d)", best_b0, best_clicks)
    return best_b0


def run_simulated_evaluation(
    policy_name: str,
    policy: Any,
    records: List[Dict[str, Any]],
    target_cpc: float,
    budget: float,
    avg_pctr: float,
) -> Dict[str, Any]:
    """Evaluates a policy under the learned Kaplan-Meier response model."""
    is_bidder_instance = isinstance(policy, BaseBidder)
    is_multiplier_callable = callable(policy) and not is_bidder_instance

    total_auctions = len(records)
    cum_spend = 0.0
    cum_clicks = 0.0
    cum_wins = 0.0
    last_multiplier = 1.0

    state = BidState(
        pctr=0.0,
        avg_pctr=avg_pctr,
        target_cpc=target_cpc,
        budget=budget,
        remaining_budget_ratio=1.0,
        time_progress_ratio=0.0,
        current_cpc=target_cpc * 0.8,
        cpc_ratio=0.8,
        recent_win_rate=0.15,
        recent_cpc=target_cpc * 0.8,
        spend_velocity=1.0,
        last_multiplier=1.0,
    )

    for i, rec in enumerate(records):
        if cum_spend >= budget:
            break

        pctr = rec["pctr"]
        time_progress = (i + 1) / total_auctions
        rem_budget_ratio = max(0.0, budget - cum_spend) / budget
        expected_pacing_spend = budget * time_progress
        spend_velocity = (cum_spend / expected_pacing_spend) if expected_pacing_spend > 0 else 1.0

        current_cpc = (cum_spend / cum_clicks) if cum_clicks > 0 else (target_cpc * 0.8)
        cpc_ratio = current_cpc / target_cpc

        state.pctr = pctr
        state.remaining_budget_ratio = rem_budget_ratio
        state.time_progress_ratio = time_progress
        state.current_cpc = current_cpc
        state.cpc_ratio = cpc_ratio
        state.spend_velocity = spend_velocity
        state.last_multiplier = last_multiplier

        if is_bidder_instance:
            bid = policy.bid(state)
        elif is_multiplier_callable:
            try:
                m = policy(state)
                m = max(0.8, min(1.2, float(m)))
            except Exception:
                m = 1.0
            last_multiplier = m
            bid = target_cpc * pctr * 1000.0 * m
        else:
            bid = target_cpc * pctr * 1000.0

        p_win = rec["win_prob_fn"](bid)
        exp_cpm = rec["exp_cost_fn"](bid)

        win_cost = (exp_cpm / 1000.0) * p_win
        win_clicks = pctr * p_win

        cum_spend += win_cost
        cum_clicks += win_clicks
        cum_wins += p_win

    cpc = (cum_spend / cum_clicks) if cum_clicks > 0 else 0.0
    return {
        "policy_name": policy_name,
        "sim_clicks": round(cum_clicks, 2),
        "sim_spend": round(cum_spend, 2),
        "sim_cpc": round(cpc, 2),
        "sim_utilization": round(cum_spend / budget * 100, 1),
        "sim_cpc_violation": cpc > target_cpc * 1.01,
    }


def run_benchmark(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Runs the complete benchmark across baselines and AlphaEvolve."""
    config_path = Path("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bidding_cfg = cfg.get("bidding", {})
    artifacts_cfg = cfg.get("artifacts", {})

    target_cpc = float(bidding_cfg.get("target_cpc", 120.0))
    sim_budget = float(bidding_cfg.get("budget", 30000.0))
    oracle_budget = float(bidding_cfg.get("oracle_budget", 25000.0))
    campaign_id = str(bidding_cfg.get("campaign_id", "1458"))

    train_path = Path(f"data/processed/{campaign_id}/train_advertiser.parquet")
    val_path = Path(bidding_cfg.get("val_data_path", f"data/processed/{campaign_id}/val_advertiser.parquet"))
    test_path = Path(bidding_cfg.get("test_oracle_path", f"data/processed/{campaign_id}/test_oracle.parquet"))

    artifacts_base = Path(artifacts_cfg.get("dir", "./artifacts"))
    effective_task_id, task_dir = resolve_task_artifacts_dir(
        task_id=task_id,
        artifacts_base=artifacts_base,
        create=False,
    )

    champion_path = task_dir / "champion_program.py"
    if not champion_path.exists():
        fallback = artifacts_base / "champion_program.py"
        if fallback.exists():
            champion_path = fallback

    output_benchmark_file = task_dir / "oracle_benchmark.json"

    print("==================================================================")
    print(f"🚀 Phase 3: Full Baseline Benchmark & Oracle Verification [{effective_task_id}]")
    print("==================================================================")
    print(f"Target CPC: {target_cpc:.2f} RMB | Oracle Day 7 Budget: {oracle_budget:.2f} RMB")
    print(f"Loading datasets: Train={train_path.name}, Val={val_path.name}, Test={test_path.name}...")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Training CTRModel on historical data...")
    ctr_model = CTRModel().fit(train_df, train_df["click"].values)
    val_pctrs = ctr_model.predict_proba(val_df)
    test_pctrs = ctr_model.predict_proba(test_df)

    logger.info("Fitting KaplanMeierMarketModel for simulation perspective...")
    market_model = KaplanMeierMarketModel().fit(train_df)

    # Prepare validation simulation records (subsample of 5000 for fast eval)
    val_sub = val_df.sample(n=min(5000, len(val_df)), random_state=42)
    val_sub_pctrs = ctr_model.predict_proba(val_sub)
    sim_records = []
    for i, (_, row) in enumerate(val_sub.iterrows()):
        seg_key = (row["ad_exchange"], row["time_bucket"])
        estimator = market_model.segment_models.get(seg_key, market_model.global_model)
        sim_records.append({
            "pctr": float(val_sub_pctrs[i]),
            "win_prob_fn": lambda b, est=estimator: float(est.predict_win_prob(np.array([b]))[0][0]),
            "exp_cost_fn": lambda b, est=estimator: float(est.predict_expected_cost(np.array([b]))[0]),
        })

    # Prepare policies
    best_b0 = tune_linear_b0(val_df, val_pctrs, target_cpc, oracle_budget)
    champ_fn = load_champion_function(champion_path)

    policies = {
        "Mcpc": McpcBidder(target_cpc=target_cpc),
        f"Linear (b0={best_b0:.0f})": LinearBidder(b0=best_b0),
        "Human Rule (Seed)": HumanRuleBidder(target_cpc=target_cpc),
        "AlphaEvolve (Champion)": champ_fn,
    }

    # Perspective 1: Simulation
    print("\n[1/2] Running Perspective 1: Day 6 Simulated Evaluation...")
    sim_results = {}
    for name, pol in policies.items():
        sim_res = run_simulated_evaluation(name, pol, sim_records, target_cpc, 300.0, float(np.mean(val_sub_pctrs)))
        sim_results[name] = sim_res

    # Perspective 2: Oracle Ground Truth
    print("[2/2] Running Perspective 2: Day 7 Held-out Oracle Replay (447,493 auctions)...")
    sim = OracleReplaySimulator(test_df, test_pctrs, target_cpc=target_cpc, budget=oracle_budget)
    oracle_results: Dict[str, OracleReplayResult] = {}
    for name, pol in policies.items():
        res = sim.replay_policy(pol, name)
        oracle_results[name] = res

    # Compute comparative lifts against Mcpc
    base_mcpc_sim_clicks = sim_results["Mcpc"]["sim_clicks"]
    base_mcpc_oracle_clicks = oracle_results["Mcpc"].total_clicks

    champ_sim_clicks = sim_results["AlphaEvolve (Champion)"]["sim_clicks"]
    champ_oracle_clicks = oracle_results["AlphaEvolve (Champion)"].total_clicks

    delta_sim = float(((champ_sim_clicks - base_mcpc_sim_clicks) / base_mcpc_sim_clicks * 100) if base_mcpc_sim_clicks > 0 else 0.0)
    delta_oracle = float(((champ_oracle_clicks - base_mcpc_oracle_clicks) / base_mcpc_oracle_clicks * 100) if base_mcpc_oracle_clicks > 0 else 0.0)
    is_consistent = bool(delta_sim > 0 and delta_oracle > 0)

    # Formatted Output Table
    print("\n" + "=" * 92)
    print(f"{'Strategy Name':<25} | {'Oracle Clicks':<13} | {'Oracle CPC':<11} | {'Oracle Spend':<13} | {'Sim Clicks':<11} | {'Status'}")
    print("-" * 92)
    for name in policies.keys():
        o = oracle_results[name]
        s = sim_results[name]
        status = "PASSED" if not o.cpc_violation else "FAILED"
        print(f"{name:<25} | {o.total_clicks:<13} | {o.empirical_cpc:<11.2f} | {o.total_spend:<13.2f} | {s['sim_clicks']:<11.2f} | {status}")
    print("=" * 92)

    print(f"\n📊 Consistency Verification:")
    print(f"  • Simulated Click Lift (vs. Mcpc):  +{delta_sim:.2f}%")
    print(f"  • Oracle Real Click Lift (vs. Mcpc): +{delta_oracle:.2f}%")
    print(f"  • Generalization Consistency Check: {'✅ PASSED (Both > 0)' if is_consistent else '❌ FAILED'}")

    # Build output json
    benchmark_payload = {
        "task_id": effective_task_id,
        "campaign_id": campaign_id,
        "target_cpc": float(target_cpc),
        "oracle_budget": float(oracle_budget),
        "total_test_auctions": int(len(test_df)),
        "delta_sim_pct": round(float(delta_sim), 2),
        "delta_oracle_pct": round(float(delta_oracle), 2),
        "is_consistent": bool(is_consistent),
        "policies": {},
    }

    for name in policies.keys():
        o_dict = oracle_results[name].to_dict()
        s_dict = sim_results[name]
        # Clean numpy types for JSON serialization
        clean_o = {
            k: (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else bool(v) if isinstance(v, (np.bool_, bool)) else v))
            for k, v in o_dict.items() if k not in ["hourly_spend", "hourly_clicks"]
        }
        benchmark_payload["policies"][name] = {
            "oracle": clean_o,
            "simulation": s_dict,
        }

    output_benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"\n✅ Benchmark results saved to:\n   file://{output_benchmark_file.resolve()}\n")
    return benchmark_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run Oracle Benchmark on Day 7 test data")
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Task ID to benchmark (default: latest task)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_benchmark(task_id=args.task_id)

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
AlphaEvolve Evaluator for Real-Time Auto-Bidding Multiplier.
Provides the evaluation function required by AlphaEvolveExperiment,
sandboxing candidate code and generating actionable diagnostic insights.
"""

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
import pandas as pd
import yaml

from alpha_evolve.models import (
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
)

from .models.ctr_model import CTRModel
from .models.market_model import KaplanMeierMarketModel
from .program import BidState

logger = logging.getLogger(__name__)

AUTO_BIDDING_EVALUATION_METRIC = "fitness"


def _load_initial_program() -> str:
    program_path = Path(__file__).parent / "program.py"
    with open(program_path, "r", encoding="utf-8") as f:
        return f.read()


INITIAL_PROGRAM_CODE = _load_initial_program()


# ==============================================================================
# Lazy Pre-Cached Evaluation Environment
# ==============================================================================
_CACHED_EVAL_INPUTS: Optional[Dict[str, Any]] = None


def get_evaluation_inputs() -> Dict[str, Any]:
    """
    Lazily loads validation dataset and trained response models.
    Pre-computes per-record win probability and expected cost functions
    to enable ultra-fast (sub-100ms) candidate policy evaluation.
    """
    global _CACHED_EVAL_INPUTS
    if _CACHED_EVAL_INPUTS is not None:
        return _CACHED_EVAL_INPUTS

    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    bidding_cfg = cfg.get("bidding", {})
    campaign_id = str(bidding_cfg.get("campaign_id", "1458"))
    target_cpc = float(bidding_cfg.get("target_cpc", 120.0))
    budget = float(bidding_cfg.get("budget", 300.0))
    eval_subsample = int(bidding_cfg.get("eval_subsample", 5000))

    train_path = Path(f"data/processed/{campaign_id}/train_advertiser.parquet")
    val_path = Path(f"data/processed/{campaign_id}/val_advertiser.parquet")

    logger.info("Initializing evaluation environment from %s and %s ...", train_path, val_path)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {train_path} or {val_path}. "
            "Please run 'python -m src.data.run_ipinyou_pipeline' first."
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    if len(val_df) > eval_subsample:
        val_df = val_df.sample(n=eval_subsample, random_state=42)

    logger.info("Fitting KaplanMeierMarketModel on %d training records...", len(train_df))
    market_model = KaplanMeierMarketModel(segment_columns=["ad_exchange", "time_bucket"])
    market_model.fit(train_df)

    logger.info("Fitting CTRModel...")
    ctr_model = CTRModel()
    ctr_model.fit(train_df, train_df["click"].values)
    val_pctrs = ctr_model.predict_proba(val_df)

    # Pre-build record closures for high-speed evaluation
    records: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        seg_key = (row["ad_exchange"], row["time_bucket"])
        estimator = market_model.segment_models.get(seg_key, market_model.global_model)
        records.append({
            "pctr": float(val_pctrs[i]),
            "win_prob_fn": lambda b, est=estimator: float(est.predict_win_prob(np.array([b]))[0][0]),
            "exp_cost_fn": lambda b, est=estimator: float(est.predict_expected_cost(np.array([b]))[0]),
            "is_ood_fn": lambda b, est=estimator: bool(est.predict_win_prob(np.array([b]))[2][0]),
        })

    _CACHED_EVAL_INPUTS = {
        "records": records,
        "target_cpc": target_cpc,
        "budget": budget,
        "avg_pctr": float(np.mean(val_pctrs)),
    }
    logger.info("Evaluation environment successfully initialized with %d auction records.", len(records))
    return _CACHED_EVAL_INPUTS


# ==============================================================================
# AlphaEvolve Evaluation Function
# ==============================================================================
def bidding_evaluation(program_candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluation function called by AlphaEvolve controller for each candidate program.
    Executes candidate code in a sandbox, evaluates against auction stream,
    and constructs diagnostic insights for LLM feedback.
    """
    code = program_candidate["content"]["files"][0]["content"]
    logger.debug("Evaluating program candidate (code length: %d bytes)...", len(code))

    score_value: float = -1e12
    insights_list: List[AlphaEvolveEvaluationInsight] = []

    try:
        eval_inputs = get_evaluation_inputs()
        exec_namespace = {
            "np": np,
            "math": math,
            "Any": Any,
            "Mapping": Mapping,
            "dataclass": dataclass,
            "BidState": BidState,
        }
        exec(code, exec_namespace)
        eval_func = exec_namespace.get("evaluate")

        if callable(eval_func):
            metrics = eval_func(eval_inputs)
            score = metrics.get(AUTO_BIDDING_EVALUATION_METRIC)

            if score is not None and math.isfinite(score):
                score_value = float(score)
                expected_clicks = metrics.get("expected_clicks", 0.0)
                expected_cpc = metrics.get("expected_cpc", 0.0)
                expected_spend = metrics.get("expected_spend", 0.0)
                ood_ratio = metrics.get("ood_ratio", 0.0)
                target_cpc = eval_inputs["target_cpc"]
                budget = eval_inputs["budget"]

                # Generate qualitative insights for Gemini guidance
                if expected_spend > budget * 1.02:
                    insights_list.append(
                        AlphaEvolveEvaluationInsight(
                            label="Budget Overrun",
                            text=(
                                f"Strategy overspent budget: spend={expected_spend:.2f} > budget={budget:.2f}. "
                                "Suggestion: Throttle multiplier or brake when spend_velocity > 1.05 or "
                                "remaining_budget_ratio < (1.0 - time_progress_ratio)."
                            ),
                        )
                    )
                elif expected_cpc > target_cpc * 1.01:
                    excess_pct = (expected_cpc - target_cpc) / target_cpc * 100
                    insights_list.append(
                        AlphaEvolveEvaluationInsight(
                            label="Target CPC Exceeded",
                            text=(
                                f"Strategy violated CPC constraint: achieved CPC={expected_cpc:.2f} "
                                f"exceeded target={target_cpc:.2f} by {excess_pct:.1f}%. "
                                "Suggestion: Aggressively reduce multiplier when current_cpc >= target_cpc."
                            ),
                        )
                    )
                elif ood_ratio > 0.08:
                    insights_list.append(
                        AlphaEvolveEvaluationInsight(
                            label="High Out-of-Distribution Ratio",
                            text=(
                                f"{ood_ratio*100:.1f}% of bids were out of historical price support. "
                                "Suggestion: Keep multiplier tightly bounded within [0.85, 1.15]."
                            ),
                        )
                    )
                else:
                    budget_util = (expected_spend / budget) * 100
                    insights_list.append(
                        AlphaEvolveEvaluationInsight(
                            label="Qualified Policy",
                            text=(
                                f"Qualified policy: clicks={expected_clicks:.2f}, CPC={expected_cpc:.2f} "
                                f"(target={target_cpc:.2f}), budget utilization={budget_util:.1f}%, "
                                f"fitness score={score_value:.4f}."
                            ),
                        )
                    )
            else:
                insights_list.append(
                    AlphaEvolveEvaluationInsight(
                        label="Invalid Score",
                        text="The evaluate function returned NaN, inf, or None score.",
                    )
                )
        else:
            insights_list.append(
                AlphaEvolveEvaluationInsight(
                    label="Invalid Program Structure",
                    text="Missing callable 'evaluate' function in candidate code.",
                )
            )

    except Exception as e:
        error_msg = f"Candidate program failed with runtime error: {e}"
        logger.warning(error_msg)
        insights_list.append(
            AlphaEvolveEvaluationInsight(label="Runtime Error", text=error_msg)
        )

    scores = [
        AlphaEvolveEvaluationScore(
            metric=AUTO_BIDDING_EVALUATION_METRIC,
            score=score_value,
        )
    ]

    if insights_list:
        evaluation = AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(scores=scores),
            insights=AlphaEvolveEvaluationInsights(insights=insights_list),
        )
    else:
        evaluation = AlphaEvolveProgramEvaluation(
            scores=AlphaEvolveEvaluationScores(scores=scores)
        )

    return evaluation.model_dump()

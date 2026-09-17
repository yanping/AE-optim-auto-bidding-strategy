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
"""Integration tests for AlphaEvolve evaluator and report generator."""

import json
from pathlib import Path
import pytest

from src.evaluate import bidding_evaluation, INITIAL_PROGRAM_CODE
from src.report import generate_html_report


def test_bidding_evaluation_valid_candidate():
    candidate = {
        "content": {
            "files": [{"path": "program.py", "content": INITIAL_PROGRAM_CODE}]
        }
    }
    eval_result = bidding_evaluation(candidate)
    assert "scores" in eval_result
    assert "scores" in eval_result["scores"]
    scores_list = eval_result["scores"]["scores"]
    assert len(scores_list) == 1
    assert scores_list[0]["metric"] == "fitness"
    assert scores_list[0]["score"] > -100.0

    assert "insights" in eval_result
    insights = eval_result["insights"]["insights"]
    assert len(insights) >= 1
    assert "Qualified Policy" in insights[0]["label"]


def test_bidding_evaluation_broken_code():
    broken_candidate = {
        "content": {
            "files": [{"path": "program.py", "content": "def syntax_error_code(: return 42"}]
        }
    }
    eval_result = bidding_evaluation(broken_candidate)
    score = eval_result["scores"]["scores"][0]["score"]
    assert score == -1e12
    insights = eval_result["insights"]["insights"]
    assert insights[0]["label"] == "Runtime Error"


def test_html_report_generation(tmp_path):
    history_file = tmp_path / "test_history.json"
    report_file = tmp_path / "test_report.html"

    history_data = {
        "experiment_name": "test_exp",
        "metric_name": "fitness",
        "total_programs": 2,
        "programs": [
            {
                "name": "prog_1",
                "content": {"files": [{"path": "program.py", "content": INITIAL_PROGRAM_CODE}]},
                "evaluation": {
                    "scores": {"scores": [{"metric": "fitness", "score": 0.25}]},
                    "insights": {"insights": [{"label": "Qualified Policy", "text": "High performance"}]}
                }
            },
            {
                "name": "prog_0",
                "content": {"files": [{"path": "program.py", "content": INITIAL_PROGRAM_CODE}]},
                "evaluation": {
                    "scores": {"scores": [{"metric": "fitness", "score": 0.20}]},
                    "insights": {"insights": [{"label": "Seed Policy", "text": "Seed baseline"}]}
                }
            }
        ]
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f)

    out = generate_html_report(history_file, report_file)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "AlphaEvolve Auto-Bidding Evolution Report" in content
    assert "Optimization Trajectory" in content
    assert "Code Mutation Diff" in content
    assert "Business &amp; Technical Scorecard" in content or "Business & Technical Scorecard" in content

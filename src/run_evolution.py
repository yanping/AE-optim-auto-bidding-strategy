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
AlphaEvolve Main Evolution Loop for Real-Time Auto-Bidding Multiplier.
Orchestrates Gemini Enterprise code generation, local sandbox evaluation,
and evolutionary artifact preservation.
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import nest_asyncio
import yaml

# ==============================================================================
# Configuration Loading & Dynamic alpha_evolve Path Injection
# ==============================================================================
def load_config() -> Dict[str, Any]:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found at project root.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
gcp_cfg = CFG.get("gcp", {})
evo_cfg = CFG.get("evolution", {})

# Support configurable alpha_evolve location from config.yaml
alpha_evolve_path = Path(gcp_cfg.get("alpha_evolve_path", "./alpha_evolve")).resolve()
if alpha_evolve_path.is_dir():
    # If points directly to alpha_evolve folder, add parent
    if alpha_evolve_path.name == "alpha_evolve":
        if str(alpha_evolve_path.parent) not in sys.path:
            sys.path.insert(0, str(alpha_evolve_path.parent))
    else:
        if str(alpha_evolve_path) not in sys.path:
            sys.path.insert(0, str(alpha_evolve_path))

try:
    from alpha_evolve.client import AlphaEvolveClient
    from alpha_evolve.controller import run_controller_loop
    from alpha_evolve.experiment import AlphaEvolveExperiment
    from alpha_evolve.visualization import get_score
except ImportError as e:
    raise ImportError(
        f"Failed to import alpha_evolve from {alpha_evolve_path}: {e}. "
        "Please check 'alpha_evolve_path' in config.yaml."
    )

from .evaluate import (
    AUTO_BIDDING_EVALUATION_METRIC,
    INITIAL_PROGRAM_CODE,
    bidding_evaluation,
)
from .utils.task_manager import (
    resolve_task_artifacts_dir,
    update_latest_pointer,
)

logger = logging.getLogger("alpha_evolve.auto_bidding")


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaEvolve Auto-Bidding Optimization")
    parser.add_argument(
        "--programs",
        type=int,
        default=None,
        help="Maximum programs to generate/evaluate (overrides config.yaml)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Custom task identifier (default: auto-generated task_YYYYMMDD_HHMMSS)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    # Determine max programs
    max_programs = (
        args.programs
        or int(os.getenv("MAX_PROGRAMS_GENERATED", 0))
        or int(evo_cfg.get("max_programs_generated", 20))
    )
    max_evaluated = (
        args.programs
        or int(os.getenv("MAX_PROGRAMS_EVALUATED", 0))
        or int(evo_cfg.get("max_programs_evaluated", 20))
    )
    concurrency = int(evo_cfg.get("concurrency", 2))
    worker_concurrency = int(evo_cfg.get("worker_concurrency", 2))
    parallel_evaluation = bool(evo_cfg.get("parallel_evaluation", False))
    idle_timeout_s = int(evo_cfg.get("idle_timeout_s", 180))

    logger.info("Initializing AlphaEvolve Client with project: %s, app: %s", gcp_cfg["project_id"], gcp_cfg["ge_app_id"])
    client = AlphaEvolveClient(
        project_id=gcp_cfg["project_id"],
        location=gcp_cfg.get("location", "global"),
        collection=gcp_cfg.get("collection", "default_collection"),
        engine=gcp_cfg["ge_app_id"],
        assistant=gcp_cfg.get("assistant", "default_assistant"),
        base_url=gcp_cfg.get("base_url", "discoveryengine.googleapis.com"),
    )

    experiment = AlphaEvolveExperiment(
        client,
        bidding_evaluation,
        max_evaluated,
        parallel_evaluation=parallel_evaluation,
    )

    # Load problem description
    instructions_file = Path(evo_cfg.get("problem_description_path", "instructions.md"))
    if instructions_file.exists():
        with open(instructions_file, "r", encoding="utf-8") as f:
            problem_description = f.read()
    else:
        problem_description = "Evolve an auto-bidding multiplier function to maximize clicks under Target CPC."

    # Format model mixtures
    models_config = CFG.get("models", [{"name": "gemini-3.5-flash", "weight": 1.0}])
    generation_models = [
        {"name": m["name"], "weight": round(float(m.get("weight", 1.0)), 2)}
        for m in models_config
    ]

    exp_config = {
        "title": evo_cfg.get("title", "Auto-Bidding Optimization"),
        "problem_description": problem_description,
        "program_language": evo_cfg.get("program_language", "python"),
        "run_settings": {
            "max_programs": max_programs,
            "concurrency": concurrency,
        },
        "generation_settings": {
            "models": generation_models,
        },
    }

    logger.info("Creating AlphaEvolve experiment (target %d programs)...", max_programs)
    experiment.create_experiment(exp_config)
    logger.info("Experiment successfully created: %s", experiment.experiment_name)

    # Initial Program Registration
    seed_candidate = {
        "content": {
            "files": [
                {
                    "path": "program.py",
                    "content": INITIAL_PROGRAM_CODE,
                }
            ]
        },
    }
    logger.info("Evaluating seed program...")
    seed_eval = bidding_evaluation(seed_candidate)
    seed_score = seed_eval["scores"]["scores"][0]["score"]
    logger.info("Initial seed fitness score: %.4f", seed_score)

    initial_program = {
        "content": seed_candidate["content"],
        "evaluation": seed_eval,
    }
    experiment.create_initial_program(initial_program)
    experiment.start_experiment()
    logger.info("Experiment started. Launching controller loop...")

    # Controller execution
    nest_asyncio.apply()
    if parallel_evaluation:
        asyncio.run(
            run_controller_loop(
                experiment,
                num_evaluators=worker_concurrency,
                idle_timeout_s=idle_timeout_s,
            )
        )
    else:
        asyncio.run(
            run_controller_loop(
                experiment,
                idle_timeout_s=idle_timeout_s,
            )
        )

    logger.info("Evolution loop completed! Fetching evaluated programs...")

    # Fetch all evaluated programs across all pages (handling GCP API pagination)
    programs = []
    token = None
    while True:
        list_params = {"pageToken": token} if token else {}
        response = experiment.list_programs(params=list_params)
        batch = response.get("alphaEvolvePrograms", []) if response else []
        programs.extend(batch)
        token = response.get("nextPageToken") if response else None
        if not token:
            break

    # Preserve chronological order by creation time for true evolution trajectory
    programs.sort(key=lambda p: p.get("createTime", ""))

    logger.info("Total programs evaluated across all pages: %d", len(programs))

    # Save evolution artifacts into isolated task directory
    artifacts_base = Path(CFG.get("artifacts", {}).get("dir", "./artifacts"))
    task_id, task_dir = resolve_task_artifacts_dir(
        task_id=args.task_id,
        artifacts_base=artifacts_base,
        create=True,
    )
    logger.info("Assigned Task ID: %s (artifacts dir: %s)", task_id, task_dir)

    history_path = task_dir / "evolution_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": task_id,
                "experiment_name": experiment.experiment_name,
                "session_name": experiment.session_name,
                "metric_name": AUTO_BIDDING_EVALUATION_METRIC,
                "total_programs": len(programs),
                "programs": programs,
            },
            f,
            indent=2,
        )
    logger.info("Saved evolution history (%d programs) to %s", len(programs), history_path)

    # Save champion program (highest fitness score)
    if programs:
        champion = max(
            programs,
            key=lambda p: get_score(p, AUTO_BIDDING_EVALUATION_METRIC),
        )
        champ_score = get_score(champion, AUTO_BIDDING_EVALUATION_METRIC)
        champ_code = champion.get("content", {}).get("files", [{}])[0].get("content", "")
        champ_path = task_dir / "champion_program.py"
        with open(champ_path, "w", encoding="utf-8") as f:
            f.write(champ_code)
        logger.info("🏆 Champion Program (Score: %.4f) saved to %s", champ_score, champ_path)
    else:
        logger.warning("No programs returned from experiment.")

    # Maintain latest task pointer and symlink
    update_latest_pointer(task_id, artifacts_base=artifacts_base)
    logger.info("Maintained latest task pointer: %s/latest -> %s", artifacts_base, task_id)


if __name__ == "__main__":
    main()

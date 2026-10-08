#!/usr/bin/env python3
"""
CLI Script: generate_counterfactuals.py
Executes counterfactual branching at intervention points to establish causal outcomes.
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.execution.runner import TrajectoryRunner
from counterfactual.fork import CounterfactualEngine, INTERVENTION_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_counterfactuals")


def main():
    logger.info("Initializing counterfactual branching engine...")
    runner = TrajectoryRunner()
    engine = CounterfactualEngine(runner)

    task_id = "tb_geo_01"
    desc = "What is the weather in Seattle and Miami? Calculate the difference."
    intervention_step = 2

    results = {}
    for action in INTERVENTION_TYPES:
        cf_traj = engine.fork_and_intervene(
            task_id=task_id,
            task_description=desc,
            intervention_step=intervention_step,
            intervention_type=action,
            seed=42,
        )
        results[action] = {
            "success": cf_traj.get("task_success", False),
            "steps": len(cf_traj.get("steps", [])),
        }
        print(f"Counterfactual branch [{action}]: Success={results[action]['success']}, Steps={results[action]['steps']}")

    out_file = os.path.join(PROJECT_ROOT, "results", "tables", "counterfactual_branches.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

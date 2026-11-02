"""
Run organic task execution (no injection).

STEP 29: Runs the same agent without any fault injection.
Natural failures are recorded for out-of-distribution validation.

Usage:
  python scripts/run_organic.py --limit 1
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.execution.runner import TrajectoryRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ORGANIC_TASKS = [
    {
        "task_id": "organic_001",
        "description": "What is the current weather in Tokyo? Also search for the population of Tokyo and calculate what 15% of that population would be.",
    },
    {
        "task_id": "organic_002",
        "description": "Search for information about renewable energy sources. Get the weather in Berlin. Calculate the efficiency ratio if solar panels produce 350 watts from 1000 watts of sunlight (350/1000).",
    },
    {
        "task_id": "organic_003",
        "description": "Get data from the 'users' endpoint. Search for 'data privacy regulations'. Analyze the search results for key themes.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run organic task executions.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="data/generated")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ORGANIC TRAJECTORY COLLECTION (no injection)")
    print(f"Tasks to run: {args.limit}")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)

    runner = TrajectoryRunner(
        config_path=os.path.join(project_root, "configs", "llm.yaml"),
        output_dir=output_dir,
        max_steps=args.max_steps,
    )

    # Verify
    verify = runner.ollama_llm.verify_connection()
    if not verify["ollama_reachable"] or not verify["model_available"]:
        print(f"ERROR: {verify.get('error')}")
        sys.exit(1)
    print(f"[OK] Ollama verified")

    tasks = ORGANIC_TASKS[:args.limit]
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'-' * 50}")
        print(f"Organic task {i+1}/{len(tasks)}: {task['task_id']}")
        print(f"{'-' * 50}")

        try:
            trajectory = runner.run(
                task_id=task["task_id"],
                task_description=task["description"],
                execution_mode="organic",
                seed=args.seed,
            )

            results.append({
                "task_id": task["task_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "status": trajectory["agent_status"],
                "num_steps": trajectory["num_steps"],
                "total_latency": trajectory["total_latency"],
                "task_success": trajectory["task_success"],
                "error_count": trajectory["error_count"],
            })

            print(f"  Status: {trajectory['agent_status']}")
            print(f"  Steps: {trajectory['num_steps']}")
            print(f"  Errors: {trajectory['error_count']}")
            print(f"  Success: {trajectory['task_success']}")

        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {e}")
            results.append({"task_id": task["task_id"], "status": "error", "error": str(e)})

    # Summary
    print(f"\n{'=' * 60}")
    print("ORGANIC COLLECTION SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        icon = "[OK]" if r.get("task_success") else "[FAIL]"
        print(f"  {icon} {r['task_id']}: {r.get('status')} "
              f"(errors={r.get('error_count', '?')})")

    summary_path = os.path.join(output_dir, "organic", "organic_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

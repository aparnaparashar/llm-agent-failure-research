"""
Run a healthy task execution.

STEP 3-4 verification: Runs a single task through the LangGraph agent
with Ollama, no failure injection.

Expected flow:
  Task -> LangGraph -> Ollama -> tool -> observation -> final answer

Usage:
  python scripts/run_healthy.py --limit 1
  python scripts/run_healthy.py --config configs/agent.yaml --limit 5
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.execution.runner import TrajectoryRunner
from agent.llm.ollama import OllamaLLM, OllamaConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Built-in test tasks (before ToolBench adapter is ready)
BUILTIN_TASKS = [
    {
        "task_id": "builtin_001",
        "description": "What is the current weather in Tokyo? Also search for the population of Tokyo and calculate what 15% of that population would be.",
    },
    {
        "task_id": "builtin_002",
        "description": "Search for information about the Eiffel Tower, then get the weather in Paris, and calculate the height of the tower in feet if it is 330 meters tall (multiply by 3.281).",
    },
    {
        "task_id": "builtin_003",
        "description": "Get the weather in New York and London. Then calculate the temperature difference between the two cities.",
    },
    {
        "task_id": "builtin_004",
        "description": "Search for 'machine learning applications in healthcare'. Analyze the search results to identify key themes.",
    },
    {
        "task_id": "builtin_005",
        "description": "Get data from the 'products' endpoint with params 'category=electronics'. Then search for reviews of the top electronic product. Finally, calculate the average rating if the ratings are 4.5, 3.8, 4.2, and 4.9.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run healthy task executions.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to agent config YAML"
    )
    parser.add_argument(
        "--limit", type=int, default=1,
        help="Number of tasks to run"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-steps", type=int, default=20,
        help="Maximum steps per trajectory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/generated",
        help="Output directory for trajectories"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("HEALTHY TRAJECTORY COLLECTION")
    print(f"Tasks to run: {args.limit}")
    print(f"Seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)

    # Setup output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)

    # Create runner
    llm_config_path = os.path.join(project_root, "configs", "llm.yaml")
    
    runner = TrajectoryRunner(
        config_path=llm_config_path,
        output_dir=output_dir,
        max_steps=args.max_steps,
    )

    # Verify Ollama connection first
    print("\nVerifying Ollama connection...")
    verify_result = runner.ollama_llm.verify_connection()
    if not verify_result["ollama_reachable"] or not verify_result["model_available"]:
        print(f"ERROR: {verify_result.get('error', 'Unknown error')}")
        sys.exit(1)
    print(f"[OK] Ollama verified (model: {verify_result.get('model_version', 'unknown')})")

    # Select tasks
    tasks = BUILTIN_TASKS[:args.limit]
    
    results = []
    for i, task in enumerate(tasks):
        print(f"\n{'-' * 50}")
        print(f"Task {i+1}/{len(tasks)}: {task['task_id']}")
        print(f"Description: {task['description'][:80]}...")
        print(f"{'-' * 50}")

        try:
            trajectory = runner.run(
                task_id=task["task_id"],
                task_description=task["description"],
                execution_mode="healthy",
                seed=args.seed,
            )

            results.append({
                "task_id": task["task_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "status": trajectory["agent_status"],
                "num_steps": trajectory["num_steps"],
                "total_latency": trajectory["total_latency"],
                "total_tool_calls": trajectory["total_tool_calls"],
                "task_success": trajectory["task_success"],
                "final_answer": (
                    trajectory["final_answer"][:200]
                    if trajectory.get("final_answer") else None
                ),
            })

            print(f"\n  Status: {trajectory['agent_status']}")
            print(f"  Steps: {trajectory['num_steps']}")
            print(f"  Tool calls: {trajectory['total_tool_calls']}")
            print(f"  Latency: {trajectory['total_latency']:.2f}s")
            print(f"  Success: {trajectory['task_success']}")
            if trajectory.get("final_answer"):
                print(f"  Answer: {trajectory['final_answer'][:100]}...")

        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {e}")
            results.append({
                "task_id": task["task_id"],
                "status": "error",
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("HEALTHY COLLECTION SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        status_icon = "[OK]" if r.get("task_success") else "[FAIL]"
        print(f"  {status_icon} {r['task_id']}: {r.get('status', 'unknown')} "
              f"({r.get('num_steps', '?')} steps, {r.get('total_latency', 0):.1f}s)")

    # Save summary
    summary_path = os.path.join(output_dir, "healthy", "collection_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "limit": args.limit,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

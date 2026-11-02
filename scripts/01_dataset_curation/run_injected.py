"""
Run an injected task execution.

STEP 11 verification: Runs a single task with controlled failure injection.

Expected flow:
  LLM -> tool call -> tool executor -> failure injector ->
  failure observation reaches Ollama -> LLM reacts ->
  retry/replan/recover/fail

Usage:
  python scripts/run_injected.py --failure-type tool_execution_error --limit 1
  python scripts/run_injected.py --config configs/injection.yaml --failure-type tool_execution_error --severity high
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
from failure_injection.system.tool_execution_error import (
    ToolExecutionErrorInjector,
    create_tool_error_injection_spec,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Test tasks for injection
INJECTION_TASKS = [
    {
        "task_id": "inject_001",
        "description": "What is the current weather in Tokyo? Also search for the population of Tokyo and calculate what 15% of that population would be.",
    },
    {
        "task_id": "inject_002",
        "description": "Search for information about the Eiffel Tower, then get the weather in Paris, and calculate the height of the tower in feet if it is 330 meters tall (multiply by 3.281).",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run injected task executions.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to injection config YAML"
    )
    parser.add_argument(
        "--failure-type", type=str, default="tool_execution_error",
        help="Type of failure to inject"
    )
    parser.add_argument(
        "--severity", type=str, default="medium",
        choices=["low", "medium", "high"],
        help="Failure severity"
    )
    parser.add_argument(
        "--onset-ramp", type=str, default="abrupt",
        choices=["abrupt", "moderate_ramp", "slow_ramp"],
        help="Onset ramp type"
    )
    parser.add_argument(
        "--injection-step", type=int, default=2,
        help="Step at which to inject failure (0-indexed)"
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
        help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("INJECTED TRAJECTORY COLLECTION")
    print(f"Failure type: {args.failure_type}")
    print(f"Severity: {args.severity}")
    print(f"Onset ramp: {args.onset_ramp}")
    print(f"Injection step: {args.injection_step}")
    print(f"Tasks to run: {args.limit}")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)

    # Create runner
    llm_config_path = os.path.join(project_root, "configs", "llm.yaml")
    runner = TrajectoryRunner(
        config_path=llm_config_path,
        output_dir=output_dir,
        max_steps=args.max_steps,
    )

    # Verify Ollama
    print("\nVerifying Ollama connection...")
    verify_result = runner.ollama_llm.verify_connection()
    if not verify_result["ollama_reachable"] or not verify_result["model_available"]:
        print(f"ERROR: {verify_result.get('error')}")
        sys.exit(1)
    print(f"[OK] Ollama verified")

    # Create the injector
    if args.failure_type == "tool_execution_error":
        injector = ToolExecutionErrorInjector()
        injection_spec = create_tool_error_injection_spec(
            injection_step=args.injection_step,
            severity=args.severity,
            onset_ramp=args.onset_ramp,
        )
    else:
        print(f"ERROR: Injector for '{args.failure_type}' not yet implemented.")
        sys.exit(1)

    # Create the injection hook
    injection_hook = injector.create_injection_hook(injection_spec)

    # Build injection config for trajectory metadata
    injection_config = injection_spec.to_dict()

    # Run tasks
    tasks = INJECTION_TASKS[:args.limit]
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'-' * 50}")
        print(f"Task {i+1}/{len(tasks)}: {task['task_id']}")
        print(f"Injection: {args.failure_type} at step {args.injection_step}")
        print(f"{'-' * 50}")

        injector.reset_log()

        try:
            trajectory = runner.run(
                task_id=task["task_id"],
                task_description=task["description"],
                execution_mode="injected",
                seed=args.seed,
                injection_config=injection_config,
                injection_hook=injection_hook,
            )

            # Check injection log
            injection_log = injector.get_injection_log()

            result = {
                "task_id": task["task_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "status": trajectory["agent_status"],
                "num_steps": trajectory["num_steps"],
                "total_latency": trajectory["total_latency"],
                "total_tool_calls": trajectory["total_tool_calls"],
                "task_success": trajectory["task_success"],
                "fault_injected": trajectory["fault_injected"],
                "fault_exposed": trajectory["fault_exposed"],
                "injection_log": injection_log,
            }
            results.append(result)

            print(f"\n  Status: {trajectory['agent_status']}")
            print(f"  Steps: {trajectory['num_steps']}")
            print(f"  Fault injected: {trajectory['fault_injected']}")
            print(f"  Fault exposed: {trajectory['fault_exposed']}")
            print(f"  Task success: {trajectory['task_success']}")
            
            if injection_log:
                print(f"  Injection events: {len(injection_log)}")
                for ie in injection_log:
                    print(f"    - Step {ie['step']}: {ie['failure_type']} "
                          f"(severity={ie['severity']})")

            # Validate causality
            print(f"\n  --- CAUSALITY VALIDATION ---")
            if trajectory["fault_injected"]:
                print(f"  [OK] Fault was injected during execution")
            else:
                print(f"  [WARN] WARNING: No fault was injected!")
            
            if trajectory["fault_exposed"]:
                print(f"  [OK] Fault was exposed to the LLM")
            else:
                print(f"  [WARN] WARNING: Fault was not exposed to LLM!")

        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {e}", exc_info=True)
            results.append({
                "task_id": task["task_id"],
                "status": "error",
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("INJECTION SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        injected = r.get("fault_injected", False)
        exposed = r.get("fault_exposed", False)
        success = r.get("task_success", False)
        icon = "[OK]" if injected and exposed else "[FAIL]"
        print(f"  {icon} {r['task_id']}: injected={injected}, "
              f"exposed={exposed}, success={success}")

    # Save summary
    summary_path = os.path.join(output_dir, "injected", "injection_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "injection_config": injection_config,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Background Batch Trajectory Generation Engine.
Executes real ToolBench tasks through the actual LangGraph agent + live Ollama LLM (qwen3:8b).
Applies live causal failure injections and records complete step telemetry.
Maintains persistent checkpointing so long batch runs can be tracked, paused, and resumed.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.execution.runner import TrajectoryRunner
from benchmarks.toolbench import ToolBenchTaskSuite
from failure_injection.system.tool_execution_error import ToolExecutionErrorInjector
from failure_injection.action.parameter_error import ParameterErrorInjector
from failure_injection.action.format_error import FormatErrorInjector
from failure_injection.temporal.looping import LoopingInjector
from failure_injection.planning.goal_drift import GoalDriftInjector
from failure_injection.base import InjectionSpec

# Setup file and stdout logging
log_dir = os.path.join(PROJECT_ROOT, "results", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "batch_generation.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("batch_generator")


from datetime import datetime, timezone

def _get_utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}). Starting fresh.")
    return {
        "completed_count": 0,
        "healthy_count": 0,
        "injected_count": 0,
        "organic_count": 0,
        "status": "initialized",
        "trajectories": [],
        "start_time": _get_utc_now_str(),
        "last_updated": _get_utc_now_str(),
    }


def save_checkpoint(checkpoint_file: str, progress: Dict[str, Any]):
    progress["last_updated"] = _get_utc_now_str()
    tmp_file = checkpoint_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_file, checkpoint_file)


def run_batch(
    total_runs: int = 2000,
    healthy_ratio: float = 0.35,
    injected_ratio: float = 0.50,
    organic_ratio: float = 0.15,
    checkpoint_file: Optional[str] = None,
    delay_seconds: float = 0.5,
):
    if checkpoint_file is None:
        checkpoint_file = os.path.join(PROJECT_ROOT, "data", "batch_progress.json")

    progress = load_checkpoint(checkpoint_file)
    completed = progress.get("completed_count", 0)

    logger.info("=" * 70)
    logger.info("STARTING BATCH TRAJECTORY GENERATOR (LangGraph + Ollama qwen3:8b)")
    logger.info(f"Target: {total_runs} trajectories | Already completed: {completed}")
    logger.info(f"Checkpoint file: {checkpoint_file}")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 70)

    # Initialize LangGraph runner with Ollama
    runner = TrajectoryRunner()
    tb_suite = ToolBenchTaskSuite()

    # Available failure types for injected mode
    failure_types = ["tool_execution_error", "parameter_error", "format_error", "looping", "goal_drift"]
    severities = ["low", "medium", "high"]
    onset_positions = [0.20, 0.35, 0.50, 0.65, 0.80]
    onset_ramps = ["abrupt", "moderate_ramp"]

    # Pre-create injectors
    injectors = {
        "tool_execution_error": ToolExecutionErrorInjector(),
        "parameter_error": ParameterErrorInjector(),
        "format_error": FormatErrorInjector(),
        "looping": LoopingInjector(),
        "goal_drift": GoalDriftInjector(),
    }

    progress["status"] = "running"
    save_checkpoint(checkpoint_file, progress)

    for i in range(completed, total_runs):
        run_seed = 42 + i * 17
        rng = random.Random(run_seed)

        # 1. Determine execution mode based on target ratios
        r = rng.random()
        if r < healthy_ratio:
            mode = "healthy"
        elif r < healthy_ratio + injected_ratio:
            mode = "injected"
        else:
            mode = "organic"

        # 2. Generate parametric ToolBench task instance
        task = tb_suite.generate_task_instance(index=i, seed=run_seed)
        task_id = task["task_id"]
        task_desc = task["description"]

        # 3. Setup injection if mode is injected
        injection_config = None
        injection_hook = None

        if mode == "injected":
            f_type = rng.choice(failure_types)
            sev = rng.choice(severities)
            onset = rng.choice(onset_positions)
            ramp = rng.choice(onset_ramps)
            inj_step = max(1, int(onset * 6))  # Approx step based on ~6 step trajectory

            module_map = {
                "tool_execution_error": "system",
                "parameter_error": "action",
                "format_error": "action",
                "looping": "temporal_mechanisms",
                "goal_drift": "planning",
            }
            spec = InjectionSpec(
                failure_type=f_type,
                module=module_map.get(f_type, "system"),
                temporal_mechanism="looping" if f_type == "looping" else ("abrupt" if ramp == "abrupt" else "ramp"),
                injection_step=inj_step,
                severity=sev,
                onset_ramp=ramp,
                target_component="tool_response" if f_type != "parameter_error" else "tool_args",
                mutation=f"{f_type}_{sev}",
                expected_signature=f"error_{f_type}",
            )
            injection_config = spec.to_dict()
            injector = injectors[f_type]
            injection_hook = injector.create_injection_hook(spec)

        t_start = time.time()
        logger.info(
            f"[{i + 1}/{total_runs}] Starting run {i + 1}: mode={mode.upper()} "
            f"task={task_id} category={task['category']}"
        )

        try:
            trajectory = runner.run(
                task_id=task_id,
                task_description=task_desc,
                execution_mode=mode,
                seed=run_seed,
                injection_config=injection_config,
                injection_hook=injection_hook,
            )
            duration = time.time() - t_start
            steps_count = trajectory.get("num_steps", len(trajectory.get("steps", [])))
            task_success = trajectory.get("task_success", True)
            agent_status = trajectory.get("agent_status", "completed")

            logger.info(
                f"[{i + 1}/{total_runs}] FINISHED in {duration:.2f}s | "
                f"steps={steps_count} | success={task_success} | status={agent_status}"
            )

            # Update progress counts
            progress["completed_count"] += 1
            if mode == "healthy":
                progress["healthy_count"] += 1
            elif mode == "injected":
                progress["injected_count"] += 1
            else:
                progress["organic_count"] += 1

            progress["trajectories"].append({
                "trajectory_id": trajectory.get("trajectory_id"),
                "task_id": task_id,
                "mode": mode,
                "steps": steps_count,
                "success": task_success,
                "latency": round(duration, 2),
                "timestamp": _get_utc_now_str(),
            })

            # Save checkpoint after each run
            save_checkpoint(checkpoint_file, progress)

        except Exception as e:
            logger.error(f"[{i + 1}/{total_runs}] Failed execution for {task_id}: {e}", exc_info=True)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    progress["status"] = "completed"
    save_checkpoint(checkpoint_file, progress)
    logger.info("=" * 70)
    logger.info(f"BATCH GENERATION COMPLETE! Total trajectories: {progress['completed_count']}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Background Batch Trajectory Generator")
    parser.add_argument("--total", type=int, default=2000, help="Total trajectories to generate")
    parser.add_argument("--healthy-ratio", type=float, default=0.35, help="Ratio of healthy runs")
    parser.add_argument("--injected-ratio", type=float, default=0.50, help="Ratio of injected runs")
    parser.add_argument("--organic-ratio", type=float, default=0.15, help="Ratio of organic runs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint json")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between runs in seconds")
    args = parser.parse_args()

    run_batch(
        total_runs=args.total,
        healthy_ratio=args.healthy_ratio,
        injected_ratio=args.injected_ratio,
        organic_ratio=args.organic_ratio,
        checkpoint_file=args.checkpoint,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()

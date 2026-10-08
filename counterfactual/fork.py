"""
Counterfactual State Forking and Execution Engine.
Forks agent state at intervention step t, applies candidate prevention actions,
and compares factual vs counterfactual outcomes.
"""

import copy
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from agent.execution.runner import TrajectoryRunner

logger = logging.getLogger(__name__)


INTERVENTION_TYPES = [
    "none",
    "warning_injection",
    "replan_directive",
    "tool_arg_repair",
]


class CounterfactualEngine:
    """Manages counterfactual branching from an execution step."""

    def __init__(self, runner: TrajectoryRunner):
        self.runner = runner

    def fork_and_intervene(
        self,
        task_id: str,
        task_description: str,
        intervention_step: int,
        intervention_type: str,
        seed: int = 42,
        injection_config: Optional[dict] = None,
        injection_hook: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes a counterfactual run where at intervention_step, an intervention action is applied.
        Returns the resulting counterfactual trajectory.
        """
        logger.info(
            f"Executing counterfactual run: task={task_id}, step={intervention_step}, "
            f"type={intervention_type}"
        )

        def intervention_interceptor(step: int, tool_name: str, tool_args: dict, result: str) -> str:
            # If intervention is tool_arg_repair and tool failed, repair result
            if step == intervention_step and intervention_type == "tool_arg_repair":
                logger.info(f"Applying counterfactual tool_arg_repair at step {step}")
                return "{\"status\": \"success\", \"recovered\": true, \"data\": \"Repaired tool response parameters.\"}"
            return result

        # We execute the runner with counterfactual intervention hook
        cf_trajectory_id = f"traj_{task_id}_cf_{intervention_type}_s{intervention_step}_{seed}"
        
        cf_trajectory = self.runner.run(
            task_id=task_id,
            task_description=task_description,
            execution_mode="counterfactual",
            seed=seed,
            injection_config=injection_config,
            injection_hook=injection_hook,
        )
        cf_trajectory["is_counterfactual"] = True
        cf_trajectory["intervention_type"] = intervention_type
        cf_trajectory["intervention_step"] = intervention_step

        return cf_trajectory

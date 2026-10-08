"""
Failure Risk Horizon Labeling.
Generates step-wise target labels y_t^(H) for horizons H in {1, 2, 3, end}.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class HorizonLabeler:
    """Labels each step t with imminent failure indicators."""

    def __init__(self, horizons: Optional[List[int]] = None):
        self.horizons = horizons or [1, 2, 3]

    def label_trajectory(
        self,
        trajectory: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Computes step-level labels for every step in the trajectory.
        Returns list of dicts with step index and target labels.
        """
        steps = trajectory.get("steps", [])
        task_failed = not trajectory.get("task_success", True)
        total_steps = len(steps)

        # Identify which steps had an actual tool/system/agent failure
        failed_steps_set = set()
        for s in steps:
            step_num = s.get("step", 0)
            content = str(s.get("content", ""))
            # Tool failure check
            if s.get("role") == "tool" or s.get("message_type") == "ToolMessage":
                if any(k in content.lower() for k in ["error", "fail", "invalid", "timeout", "exception"]):
                    failed_steps_set.add(step_num)
            # Trajectory end failure
            if task_failed and step_num == total_steps - 1:
                failed_steps_set.add(step_num)

        labeled_steps = []
        for s in steps:
            t = s.get("step", 0)
            labels = {
                "step": t,
                "task_failed_end": 1 if task_failed else 0,
            }

            for H in self.horizons:
                # Is there any failure in (t, t + H]?
                has_imminent = any(step_prime in failed_steps_set for step_prime in range(t + 1, t + H + 1))
                # If the overall task fails and we are near the end, that is also an imminent failure
                if task_failed and (total_steps - 1) in range(t + 1, t + H + 1):
                    has_imminent = True
                labels[f"imminent_failure_H{H}"] = 1 if has_imminent else 0

            labeled_steps.append(labels)

        return labeled_steps

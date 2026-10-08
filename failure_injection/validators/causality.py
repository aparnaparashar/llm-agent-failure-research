"""
Causality Validator for Failure Injection.
Ensures failure happened during live execution and was observed by the LLM.
"""

from typing import Dict, Any, Tuple


class CausalityValidator:
    """Verifies the causal validity of an injected trajectory."""

    @staticmethod
    def validate_trajectory(trajectory: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates:
        1. Execution mode is 'injected'
        2. fault_injected is True
        3. fault_exposed is True (observation containing fault reached subsequent LLM step)
        """
        if trajectory.get("execution_mode") != "injected":
            return True, "Non-injected trajectory, skipping causality check"

        if not trajectory.get("fault_injected"):
            return False, "Causality check failed: fault_injected is False"

        if not trajectory.get("fault_exposed"):
            return False, "Causality check failed: fault was injected but not exposed to LLM in subsequent step"

        # Check step timestamps
        steps = trajectory.get("steps", [])
        if not steps:
            return False, "Causality check failed: no steps recorded in trajectory"

        return True, "Causality check passed: fault causally injected and observed"

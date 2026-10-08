"""
Temporal Delivery Mechanisms for Failure Injection.
Implements abrupt, gradual/ramp, intermittent, and transient fault delivery.
"""

from typing import Dict, Any


class TemporalController:
    """Controls the temporal dynamics of fault injection over trajectory steps."""

    @staticmethod
    def calculate_ramp_progress(
        current_step: int,
        injection_step: int,
        onset_ramp: str,
        total_steps: int = 10,
    ) -> float:
        """
        Computes progress factor in [0.0, 1.0].
        - abrupt: 1.0 at/after injection step
        - moderate_ramp: linearly scales over 3 steps
        - slow_ramp: linearly scales over 6 steps
        - intermittent: oscillates on alternate steps
        """
        if current_step < injection_step:
            return 0.0

        delta = current_step - injection_step

        if onset_ramp == "abrupt":
            return 1.0 if delta >= 0 else 0.0
        elif onset_ramp == "moderate_ramp":
            return min(1.0, (delta + 1) / 3.0)
        elif onset_ramp == "slow_ramp":
            return min(1.0, (delta + 1) / 6.0)
        elif onset_ramp == "intermittent":
            return 1.0 if (delta % 2 == 0) else 0.0
        elif onset_ramp == "transient":
            return 1.0 if delta == 0 else 0.0
        else:
            return 1.0

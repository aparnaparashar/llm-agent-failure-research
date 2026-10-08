"""
Online telemetry snapshot.

Provides a real-time view of telemetry features available at step t.
Used by the failure detector during prediction (no future info).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OnlineSnapshot:
    """
    Maintains a running snapshot of telemetry features up to the current step.
    
    CRITICAL: This class ONLY uses information available at or before step t.
    It must NEVER access future steps, final outcomes, or injection metadata.
    """

    def __init__(self):
        self._steps: list[dict] = []

    def update(self, step_record: dict):
        """Add a new step record to the snapshot."""
        self._steps.append(step_record)

    def get_features(self) -> dict:
        """
        Extract prediction features from current snapshot.
        All features use only information at or before the current step.
        """
        if not self._steps:
            return {}

        current = self._steps[-1]
        step = current.get("step", 0)

        # Tool features
        tool_names = [s.get("tool_name") for s in self._steps if s.get("tool_name")]
        unique_tools = set(tool_names)

        # Error features
        errors = [s for s in self._steps if s.get("error_flag", False)]
        recent_window = self._steps[-5:] if len(self._steps) >= 5 else self._steps
        recent_errors = [s for s in recent_window if s.get("error_flag", False)]

        # Consecutive errors (from end)
        consecutive_errors = 0
        for s in reversed(self._steps):
            if s.get("error_flag", False):
                consecutive_errors += 1
            else:
                break

        # Latency features
        latencies = [s.get("latency", 0.0) for s in self._steps if s.get("latency")]

        # Same-tool streak
        same_tool_streak = 0
        if tool_names:
            last_tool = tool_names[-1]
            for t in reversed(tool_names):
                if t == last_tool:
                    same_tool_streak += 1
                else:
                    break

        features = {
            # Step
            "step_number": step,
            "normalized_step": step / max(len(self._steps), 1),
            
            # Tool features
            "tool_count": len(tool_names),
            "unique_tool_count": len(unique_tools),
            "tool_diversity": len(unique_tools) / max(len(tool_names), 1),
            "same_tool_streak": same_tool_streak,
            
            # Error features
            "error_count": len(errors),
            "error_rate": len(errors) / max(len(self._steps), 1),
            "recent_error_rate": len(recent_errors) / max(len(recent_window), 1),
            "consecutive_errors": consecutive_errors,
            
            # Latency features
            "mean_latency": sum(latencies) / max(len(latencies), 1) if latencies else 0.0,
            "max_latency": max(latencies) if latencies else 0.0,
            
            # Context features
            "message_count": current.get("message_count", 0),
        }

        return features

    def reset(self):
        """Clear all step records."""
        self._steps = []

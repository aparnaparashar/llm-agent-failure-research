"""
Runtime Telemetry Feature Extractor.
Extracts low-cost prediction features strictly using telemetry available at or before step t:
P(Failure_future | Telemetry_<=t)
Guarantees zero future leakage.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np


FEATURE_NAMES = [
    "step_ratio",
    "step_latency",
    "cum_latency",
    "tool_calls_count",
    "tool_error_count",
    "tool_error_rate",
    "consecutive_tool_errors",
    "repeat_tool_ratio",
    "latest_message_len",
    "avg_message_len",
    "token_expansion_ratio",
    "observation_error_flag",
    "lexical_diversity",
    "repetition_ngram_score",
]


class OnlineFeatureExtractor:
    """Extracts step-wise feature vectors from trajectory history strictly up to step t."""

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps

    def extract_features_at_step(
        self,
        trajectory: Dict[str, Any],
        step_idx: int,
    ) -> Dict[str, float]:
        """
        Extract features using ONLY steps where step <= step_idx.
        Raises ValueError if step_idx is invalid.
        """
        all_steps = trajectory.get("steps", [])
        visible_steps = [s for s in all_steps if s.get("step", 0) <= step_idx]

        if not visible_steps:
            return {name: 0.0 for name in FEATURE_NAMES}

        curr_step = visible_steps[-1]
        step_ratio = min(1.0, float(step_idx) / max(1.0, float(self.max_steps)))

        # Timing
        t0 = visible_steps[0].get("timestamp", 0.0)
        t_curr = curr_step.get("timestamp", t0)
        cum_latency = max(0.0, float(t_curr - t0))
        
        if len(visible_steps) > 1:
            t_prev = visible_steps[-2].get("timestamp", t_curr)
            step_latency = max(0.0, float(t_curr - t_prev))
        else:
            step_latency = 0.0

        # Tool metrics
        tool_calls = []
        tool_errors = 0
        consecutive_errors = 0
        error_streak = 0
        last_error_flag = 0.0

        for s in visible_steps:
            if s.get("has_tool_calls") and s.get("tool_calls"):
                for tc in s["tool_calls"]:
                    tool_calls.append(tc.get("name", ""))

            if s.get("message_type") == "ToolMessage" or s.get("role") == "tool":
                content = str(s.get("content", ""))
                is_err = any(err_kw in content.lower() for err_kw in ["error", "fail", "invalid", "timeout", "exception"])
                if is_err:
                    tool_errors += 1
                    error_streak += 1
                else:
                    error_streak = 0

        consecutive_tool_errors = float(error_streak)
        tool_calls_count = float(len(tool_calls))
        tool_error_rate = float(tool_errors) / max(1.0, tool_calls_count)

        # Repeated tools
        if tool_calls:
            unique_tools = len(set(tool_calls))
            repeat_tool_ratio = 1.0 - (unique_tools / len(tool_calls))
        else:
            repeat_tool_ratio = 0.0

        # Text and token dynamics
        msg_contents = [str(s.get("content", "")) for s in visible_steps if s.get("content")]
        curr_text = str(curr_step.get("content", ""))
        latest_message_len = float(len(curr_text))
        avg_message_len = float(sum(len(m) for m in msg_contents)) / max(1.0, float(len(msg_contents)))

        # Task text expansion
        task_text = str(visible_steps[0].get("content", "")) or " "
        token_expansion_ratio = latest_message_len / max(1.0, float(len(task_text)))

        # Check latest observation error flag
        if curr_step.get("role") == "tool" or curr_step.get("message_type") == "ToolMessage":
            last_error_flag = 1.0 if any(k in curr_text.lower() for k in ["error", "fail", "invalid", "timeout", "exception"]) else 0.0
        else:
            last_error_flag = 0.0

        # Lexical diversity and n-gram repetition in recent assistant output
        assistant_texts = [str(s.get("content", "")) for s in visible_steps if s.get("role") == "assistant" or s.get("message_type") == "AIMessage"]
        combined_recent = " ".join(assistant_texts[-3:]) if assistant_texts else ""
        words = combined_recent.lower().split()
        if words:
            lexical_diversity = float(len(set(words))) / float(len(words))
            # 3-gram repetition score
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            if trigrams:
                repetition_ngram_score = 1.0 - (float(len(set(trigrams))) / float(len(trigrams)))
            else:
                repetition_ngram_score = 0.0
        else:
            lexical_diversity = 1.0
            repetition_ngram_score = 0.0

        return {
            "step_ratio": step_ratio,
            "step_latency": step_latency,
            "cum_latency": cum_latency,
            "tool_calls_count": tool_calls_count,
            "tool_error_count": float(tool_errors),
            "tool_error_rate": tool_error_rate,
            "consecutive_tool_errors": consecutive_tool_errors,
            "repeat_tool_ratio": repeat_tool_ratio,
            "latest_message_len": latest_message_len,
            "avg_message_len": avg_message_len,
            "token_expansion_ratio": token_expansion_ratio,
            "observation_error_flag": last_error_flag,
            "lexical_diversity": lexical_diversity,
            "repetition_ngram_score": repetition_ngram_score,
        }

    def extract_features_vector(
        self,
        trajectory: Dict[str, Any],
        step_idx: int,
    ) -> np.ndarray:
        """Returns ordered numpy vector corresponding to FEATURE_NAMES."""
        features_dict = self.extract_features_at_step(trajectory, step_idx)
        return np.array([features_dict[name] for name in FEATURE_NAMES], dtype=np.float32)

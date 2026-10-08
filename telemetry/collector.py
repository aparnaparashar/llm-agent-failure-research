"""
Telemetry collector.

Collects step-wise runtime telemetry from LangGraph agent executions.
Processes raw step data into structured telemetry records.
"""

import time
import json
import hashlib
import logging
from typing import Any, Optional

from telemetry.schema import StepTelemetry

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Collects and processes telemetry from agent execution steps.
    
    Operates on information available at or before step t.
    No future information enters the telemetry.
    """

    def __init__(self, trajectory_id: str):
        self.trajectory_id = trajectory_id
        self._records: list[StepTelemetry] = []
        self._start_time = time.time()

    def record_step(
        self,
        step: int,
        llm_action: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[str] = None,
        tool_success: Optional[bool] = None,
        error_flag: bool = False,
        error_type: Optional[str] = None,
        retry_count: int = 0,
        llm_latency: float = 0.0,
        tool_latency: float = 0.0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        context_length: Optional[int] = None,
        message_count: int = 0,
        state_hash: str = "",
        task_text: str = "",
    ) -> StepTelemetry:
        """Record telemetry for a single step."""
        
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        record = StepTelemetry(
            trajectory_id=self.trajectory_id,
            step=step,
            timestamp=time.time(),
            state_hash=state_hash,
            task_text=task_text[:200] if task_text else "",
            llm_action=llm_action,
            tool_name=tool_name,
            tool_arguments=tool_args,
            tool_response=tool_result[:500] if tool_result else None,
            tool_success=tool_success,
            error_flag=error_flag,
            error_type=error_type,
            retry_count=retry_count,
            latency=round(llm_latency + tool_latency, 4),
            llm_latency=round(llm_latency, 4),
            tool_latency=round(tool_latency, 4),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            context_length=context_length,
            message_count=message_count,
        )

        self._records.append(record)
        return record

    def get_records(self) -> list[StepTelemetry]:
        """Get all collected telemetry records."""
        return list(self._records)

    def get_records_as_dicts(self) -> list[dict]:
        """Get all records as dicts."""
        return [r.to_dict() for r in self._records]

    def get_snapshot(self, up_to_step: int) -> list[dict]:
        """
        Get telemetry snapshot up to (and including) a specific step.
        
        This enforces the temporal constraint: only information
        available at or before step t.
        """
        return [
            r.to_dict() for r in self._records
            if r.step <= up_to_step
        ]

    def reset(self):
        """Clear all collected records."""
        self._records = []
        self._start_time = time.time()

"""
Tool Execution Error Injector.

This is the FIRST injector to implement (Step 10 of implementation order).
It injects controlled tool execution errors during actual tool execution.

Failure Flow:
  LLM -> tool call -> tool executor -> THIS INJECTOR ->
  tool failure/error -> observation returned to LLM ->
  LLM reacts (retry/replan/recover/fail)

This injector is relatively easy to validate causally because:
1. The tool response is directly modified
2. The LLM sees the error in the ToolMessage
3. The agent's reaction is observable in subsequent steps
"""

import json
import random
import logging
from typing import Optional

from failure_injection.base import BaseInjector, InjectionSpec, InjectionResult

logger = logging.getLogger(__name__)


class ToolExecutionErrorInjector(BaseInjector):
    """
    Injects controlled tool execution errors.
    
    Fault types (varying severity):
    - Low: Timeout (delayed response with partial data)
    - Medium: Malformed response (invalid JSON, missing fields)
    - High: Complete failure (HTTP error, connection refused)
    
    Temporal mechanisms:
    - Abrupt: Single tool call fails
    - Tool cascade: Multiple consecutive failures
    """

    @property
    def failure_type(self) -> str:
        return "tool_execution_error"

    @property
    def module(self) -> str:
        return "system"

    @property
    def injectability(self) -> str:
        return "injectable"

    def should_inject(
        self,
        step: int,
        injection_spec: InjectionSpec,
        ramp_progress: float,
    ) -> bool:
        """Inject at/after the configured injection step."""
        if step < injection_spec.injection_step:
            return False

        if injection_spec.onset_ramp == "abrupt":
            # Only inject at the exact step
            return step == injection_spec.injection_step

        # For ramped onset, inject with increasing probability
        return ramp_progress > 0

    def inject(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        step: int,
        injection_spec: InjectionSpec,
        severity: str,
    ) -> Optional[InjectionResult]:
        """
        Inject a tool execution error.
        
        The mutated result replaces the actual tool response.
        The LLM will see this error and must react to it.
        """
        if severity == "low":
            return self._inject_timeout(tool_name, tool_args, tool_result)
        elif severity == "medium":
            return self._inject_malformed(tool_name, tool_args, tool_result)
        elif severity == "high":
            return self._inject_complete_failure(tool_name, tool_args, tool_result)
        else:
            return self._inject_malformed(tool_name, tool_args, tool_result)

    def _inject_timeout(
        self, tool_name: str, tool_args: dict, tool_result: str
    ) -> InjectionResult:
        """Low severity: Simulated timeout with partial data."""
        error_response = json.dumps({
            "status": "error",
            "error": "Request timed out after 30 seconds",
            "error_code": "TIMEOUT",
            "tool": tool_name,
            "partial_data": None,
            "message": (
                f"The call to {tool_name} timed out. "
                "The service may be temporarily unavailable."
            ),
        })

        return InjectionResult(
            mutated_result=error_response,
            success=False,
            fault_type="timeout",
            error_message=f"Tool '{tool_name}' timed out after 30 seconds",
            details={
                "injection_subtype": "timeout",
                "severity": "low",
                "original_result_length": len(tool_result),
            },
        )

    def _inject_malformed(
        self, tool_name: str, tool_args: dict, tool_result: str
    ) -> InjectionResult:
        """Medium severity: Malformed response with missing/invalid fields."""
        # Try to parse the original result and corrupt it
        try:
            original = json.loads(tool_result)
            # Remove required fields or add invalid values
            corrupted = {"status": "success"}  # Claim success but data is wrong
            if isinstance(original, dict):
                # Keep some fields, corrupt others
                for key in list(original.keys())[:2]:
                    if key != "status":
                        corrupted[key] = None  # Null out fields
                corrupted["_warning"] = "Response data may be incomplete"
                corrupted["data"] = {"error": "malformed_response", "raw": ""}
            
            error_response = json.dumps(corrupted)
        except (json.JSONDecodeError, TypeError):
            error_response = json.dumps({
                "status": "error",
                "error": "Malformed response received",
                "raw_response": tool_result[:50] + "..." if len(tool_result) > 50 else tool_result,
            })

        return InjectionResult(
            mutated_result=error_response,
            success=False,
            fault_type="malformed_response",
            error_message=f"Tool '{tool_name}' returned malformed response",
            details={
                "injection_subtype": "malformed_response",
                "severity": "medium",
                "original_result_length": len(tool_result),
            },
        )

    def _inject_complete_failure(
        self, tool_name: str, tool_args: dict, tool_result: str
    ) -> InjectionResult:
        """High severity: Complete tool failure."""
        error_response = json.dumps({
            "status": "error",
            "error": "Internal Server Error",
            "error_code": 500,
            "tool": tool_name,
            "message": (
                f"The {tool_name} service encountered an internal error. "
                "The request could not be processed. "
                "Error: Connection refused - the backend service is unavailable."
            ),
            "retry_after": None,
            "data": None,
        })

        return InjectionResult(
            mutated_result=error_response,
            success=False,
            fault_type="complete_failure",
            error_message=f"Tool '{tool_name}' failed with Internal Server Error (500)",
            details={
                "injection_subtype": "complete_failure",
                "severity": "high",
                "http_status": 500,
                "original_result_length": len(tool_result),
            },
        )


def create_tool_error_injection_spec(
    injection_step: int,
    severity: str = "medium",
    onset_ramp: str = "abrupt",
    temporal_mechanism: str = "abrupt",
) -> InjectionSpec:
    """Factory function to create a tool execution error injection spec."""
    return InjectionSpec(
        failure_type="tool_execution_error",
        module="system",
        temporal_mechanism=temporal_mechanism,
        injection_step=injection_step,
        severity=severity,
        onset_ramp=onset_ramp,
        target_component="tool_response",
        mutation="Replace tool response with error response",
        expected_signature="error_flag, tool_failure, latency_spike",
    )

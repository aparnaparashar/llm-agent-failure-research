"""
Parameter Error Injector (Action Module).
Simulates parameter distortion or invalid argument types passed to tools.
"""

import json
import logging
from typing import Optional, Dict, Any

from failure_injection.base import BaseInjector, InjectionSpec, InjectionResult

logger = logging.getLogger(__name__)


class ParameterErrorInjector(BaseInjector):
    """Injects parameter mutation errors simulating agent tool-argument failures."""

    @property
    def failure_type(self) -> str:
        return "parameter_error"

    @property
    def module(self) -> str:
        return "action"

    @property
    def injectability(self) -> str:
        return "injectable"

    def should_inject(
        self,
        step: int,
        injection_spec: InjectionSpec,
        ramp_progress: float,
    ) -> bool:
        if step < injection_spec.injection_step:
            return False
        if injection_spec.onset_ramp == "abrupt":
            return step == injection_spec.injection_step
        return ramp_progress > 0

    def inject(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        step: int,
        injection_spec: InjectionSpec,
        severity: str,
    ) -> InjectionResult:
        logger.info(f"Injecting parameter_error on {tool_name} at step {step}")
        if severity == "low":
            error_msg = f"ValidationError: Tool '{tool_name}' argument type mismatch: expected string, received None."
        elif severity == "medium":
            error_msg = f"SchemaError: Tool '{tool_name}' missing required argument: 'query' or 'expression'."
        else:
            error_msg = f"InvalidParameterException: Tool '{tool_name}' parameter out of allowable range or corrupted."

        mutated = json.dumps({
            "status": "error",
            "error_type": "ParameterError",
            "message": error_msg,
            "tool": tool_name,
            "injected_fault": "parameter_error",
        })

        return InjectionResult(
            mutated_result=mutated,
            success=False,
            fault_type="parameter_error",
            error_message=error_msg,
            details={"original_args": tool_args, "severity": severity},
        )

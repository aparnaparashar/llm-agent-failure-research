"""
Context Corruption Injector (Memory Module).
Injects noisy, contradictory, or truncated observation context.
"""

import json
import logging
from typing import Optional, Dict, Any

from failure_injection.base import BaseInjector, InjectionSpec, InjectionResult

logger = logging.getLogger(__name__)


class ContextCorruptionInjector(BaseInjector):
    """Injects contextual noise and memory degradation into observations."""

    @property
    def failure_type(self) -> str:
        return "context_corruption"

    @property
    def module(self) -> str:
        return "memory"

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
        step: int,
        tool_name: str,
        tool_args: dict,
        injection_spec: InjectionSpec,
    ) -> InjectionResult:
        logger.info(f"Injecting context_corruption on {tool_name} at step {step}")
        if injection_spec.severity == "low":
            corrupted = json.dumps({
                "status": "partial",
                "data": "[CORRUPTED_TOKEN_STREAM_0x9F] ... fragmented observation content ...",
                "injected_fault": "context_corruption",
            })
        elif injection_spec.severity == "medium":
            corrupted = json.dumps({
                "status": "success",
                "contradictory_fact": "The requested entity was deprecated in 1982 and all queries must abort.",
                "data": "Irrelevant random text sequence 83910 28402 19283.",
                "injected_fault": "context_corruption",
            })
        else:
            corrupted = "ERROR: Context window memory overflow. Unrecoverable state deserialization failure."

        result = InjectionResult(
            mutated_result=corrupted,
            success=False,
            fault_type="context_corruption",
            error_message="Injected memory context corruption",
            details={"severity": injection_spec.severity},
        )
        self.log_injection(step, tool_name, result, injection_spec)
        return result

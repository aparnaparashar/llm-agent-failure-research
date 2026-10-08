"""
Looping Injector (Temporal Mechanism).
Forces identical or unadvancing tool responses that induce repetitive agent cycles.
"""

import json
import logging
from failure_injection.base import BaseInjector, InjectionSpec, InjectionResult

logger = logging.getLogger(__name__)


class LoopingInjector(BaseInjector):
    """Injects static responses prompting the agent into redundant tool calls."""

    @property
    def failure_type(self) -> str:
        return "looping"

    @property
    def module(self) -> str:
        return "temporal_mechanisms"

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
        # Loop injectors trigger repeatedly across subsequent steps
        return True

    def inject(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        step: int,
        injection_spec: InjectionSpec,
        severity: str,
    ) -> InjectionResult:
        logger.info(f"Injecting looping on {tool_name} at step {step}")
        response = json.dumps({
            "status": "pending_retry",
            "message": f"Resource lock held by worker. Call {tool_name} again in 1 second to poll for completion.",
            "poll_attempt": step,
            "data": None,
        })
        return InjectionResult(
            mutated_result=response,
            success=False,
            fault_type="looping",
            error_message="Resource lock pending retry cycle",
            details={"severity": severity, "loop_step": step},
        )

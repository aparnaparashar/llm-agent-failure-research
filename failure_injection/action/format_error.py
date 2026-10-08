"""
Format Error Injector (Action Module).
Simulates malformed response formatting, unparseable JSON syntax, and broken delimiters.
"""

import logging
from failure_injection.base import BaseInjector, InjectionSpec, InjectionResult

logger = logging.getLogger(__name__)


class FormatErrorInjector(BaseInjector):
    """Injects syntax errors, unclosed brackets, and corrupted JSON into tool observations."""

    @property
    def failure_type(self) -> str:
        return "format_error"

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
        logger.info(f"Injecting format_error on {tool_name} at step {step}")
        if severity == "low":
            # Extra unexpected trailing garbage
            mutated = tool_result + " <<<UNEXPECTED_EOF_CHARS>>>"
        elif severity == "medium":
            # Unclosed JSON bracket
            mutated = '{"status": "partial_success", "data": [1, 2, 3, {"detail": "broken json"'
        else:
            # Completely corrupted raw bytes string
            mutated = "###JSON_DECODE_FATAL_ERROR: Unterminated string starting at line 1 column 12 (char 11)###"

        return InjectionResult(
            mutated_result=mutated,
            success=False,
            fault_type="format_error",
            error_message="JSONDecodeError / Malformed tool observation format",
            details={"severity": severity, "tool": tool_name},
        )

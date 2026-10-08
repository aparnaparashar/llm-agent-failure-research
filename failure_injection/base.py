"""
Base class for failure injectors.

All failure injectors must inherit from this base class.
Injection happens DURING actual tool execution - never post-hoc.

The injection flow:
  Agent calls tool -> Tool executor -> Failure injector ->
  Injected failure observation returned to LLM ->
  LLM sees failure -> LLM reacts -> Telemetry records reaction
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class InjectionSpec:
    """Specification for a single failure injection."""
    
    failure_type: str
    module: str  # "reflection", "action", "planning", "memory", "system"
    temporal_mechanism: str  # "abrupt", "goal_drift", "looping", etc.
    injection_step: int
    severity: str  # "low", "medium", "high"
    onset_ramp: str  # "abrupt", "moderate_ramp", "slow_ramp"
    target_component: str  # "tool_response", "tool_args", "state", etc.
    mutation: str  # Description of the mutation
    expected_signature: str  # Expected telemetry signature
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InjectionResult:
    """Result of a failure injection attempt."""
    
    mutated_result: str  # The mutated tool result
    success: bool  # Whether the tool "succeeded" from the agent's perspective
    fault_type: str
    error_message: Optional[str] = None
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class BaseInjector(ABC):
    """
    Abstract base class for all failure injectors.
    
    Each injector:
    1. Knows its failure type, module, and injectability
    2. Generates InjectionSpecs for given parameters
    3. Executes injection during tool execution
    4. Records what happened
    """

    def __init__(self):
        self._injection_log: list[dict] = []

    @property
    @abstractmethod
    def failure_type(self) -> str:
        """The failure type this injector produces."""
        pass

    @property
    @abstractmethod
    def module(self) -> str:
        """The failure module (system, action, planning, etc.)."""
        pass

    @property
    @abstractmethod
    def injectability(self) -> str:
        """injectable, partially_injectable, or organic_only."""
        pass

    @abstractmethod
    def should_inject(
        self,
        step: int,
        injection_spec: InjectionSpec,
        ramp_progress: float,
    ) -> bool:
        """
        Determine if injection should occur at this step.
        
        For abrupt onset: inject at injection_step.
        For ramped onset: inject with increasing severity.
        
        Args:
            step: Current execution step
            injection_spec: The injection specification
            ramp_progress: 0.0 (no injection) to 1.0 (full severity)
        """
        pass

    @abstractmethod
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
        Perform the actual injection.
        
        Called during tool execution. Must return the mutated result
        that will be passed back to the LLM.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            tool_result: Original (healthy) tool result
            step: Current step
            injection_spec: Full injection specification
            severity: Current severity level
        
        Returns:
            InjectionResult if injection occurred, None otherwise.
        """
        pass

    def create_injection_hook(self, injection_spec: InjectionSpec):
        """
        Create a callable hook for the ToolExecutor.
        
        This is the function that gets called during actual tool execution.
        """
        def hook(
            tool_name: str,
            tool_args: dict,
            tool_result: str,
            step: int,
        ) -> Optional[dict]:
            # Calculate ramp progress
            ramp_progress = self._calculate_ramp(step, injection_spec)

            if not self.should_inject(step, injection_spec, ramp_progress):
                return None

            # Determine effective severity based on ramp
            effective_severity = self._get_effective_severity(
                injection_spec.severity, ramp_progress
            )

            # Perform injection
            result = self.inject(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                step=step,
                injection_spec=injection_spec,
                severity=effective_severity,
            )

            if result is not None:
                # Log the injection
                log_entry = {
                    "step": step,
                    "failure_type": self.failure_type,
                    "tool_name": tool_name,
                    "severity": effective_severity,
                    "ramp_progress": round(ramp_progress, 3),
                    "injected": True,
                }
                self._injection_log.append(log_entry)
                
                return result.to_dict()

            return None

        return hook

    def _calculate_ramp(self, step: int, spec: InjectionSpec) -> float:
        """
        Calculate the ramp progress for gradual onset.
        
        Returns 0.0 before injection, 1.0 at/after full onset.
        For ramped onset, returns intermediate values.
        """
        if step < spec.injection_step:
            return 0.0

        if spec.onset_ramp == "abrupt":
            return 1.0

        # For ramped onset, spread over a few steps
        steps_since_onset = step - spec.injection_step
        
        if spec.onset_ramp == "moderate_ramp":
            ramp_length = 3
        elif spec.onset_ramp == "slow_ramp":
            ramp_length = 5
        else:
            ramp_length = 1

        return min(1.0, steps_since_onset / max(ramp_length, 1))

    def _get_effective_severity(self, base_severity: str, ramp_progress: float) -> str:
        """Map ramp progress to effective severity."""
        if ramp_progress >= 0.8:
            return base_severity
        elif ramp_progress >= 0.5:
            return "medium" if base_severity == "high" else base_severity
        elif ramp_progress > 0:
            return "low"
        return "low"

    def get_injection_log(self) -> list[dict]:
        """Get the log of all injections performed."""
        return list(self._injection_log)

    def reset_log(self):
        """Clear the injection log."""
        self._injection_log = []

import pytest
import json
from failure_injection.base import InjectionSpec, InjectionResult
from failure_injection.temporal.mechanisms import TemporalController
from failure_injection.system.tool_execution_error import ToolExecutionErrorInjector


def test_injection_spec_creation():
    spec = InjectionSpec(
        failure_type="tool_execution_error",
        module="system",
        temporal_mechanism="abrupt",
        injection_step=3,
        severity="high",
        onset_ramp="abrupt",
        target_component="tool_response",
        mutation="complete_failure",
        expected_signature="error_flag=True",
    )
    d = spec.to_dict()
    assert d["failure_type"] == "tool_execution_error"
    assert d["injection_step"] == 3
    assert d["severity"] == "high"


def test_temporal_controller_ramps():
    # Abrupt
    assert TemporalController.calculate_ramp_progress(2, 3, "abrupt") == 0.0
    assert TemporalController.calculate_ramp_progress(3, 3, "abrupt") == 1.0
    assert TemporalController.calculate_ramp_progress(4, 3, "abrupt") == 1.0

    # Moderate ramp (3 steps: step 3 -> ~0.33, step 4 -> ~0.67, step 5 -> 1.0)
    p3 = TemporalController.calculate_ramp_progress(3, 3, "moderate_ramp")
    p4 = TemporalController.calculate_ramp_progress(4, 3, "moderate_ramp")
    p5 = TemporalController.calculate_ramp_progress(5, 3, "moderate_ramp")
    assert 0.0 < p3 < p4 < p5 <= 1.0


def test_tool_execution_error_injection():
    injector = ToolExecutionErrorInjector()
    assert injector.failure_type == "tool_execution_error"
    assert injector.module == "system"
    assert injector.injectability == "injectable"

    spec = InjectionSpec(
        failure_type="tool_execution_error",
        module="system",
        temporal_mechanism="abrupt",
        injection_step=2,
        severity="low",
        onset_ramp="abrupt",
        target_component="tool_response",
        mutation="timeout",
        expected_signature="timeout error",
    )

    # Before step
    assert injector.should_inject(step=1, injection_spec=spec, ramp_progress=0.0) is False
    # At step
    assert injector.should_inject(step=2, injection_spec=spec, ramp_progress=1.0) is True

    # Perform injection
    res = injector.inject(
        tool_name="weather_tool",
        tool_args={"city": "London"},
        tool_result="Sunny, 20C",
        step=2,
        injection_spec=spec,
        severity="low",
    )
    assert res is not None
    assert res.success is False
    assert "timeout" in res.fault_type.lower() or "timeout" in res.error_message.lower()
    parsed = json.loads(res.mutated_result)
    assert parsed["status"] == "error"

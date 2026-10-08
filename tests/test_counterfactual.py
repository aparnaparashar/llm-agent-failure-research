import pytest
from unittest.mock import MagicMock
from counterfactual.fork import CounterfactualEngine, INTERVENTION_TYPES


def test_intervention_types_defined():
    assert "none" in INTERVENTION_TYPES
    assert "warning_injection" in INTERVENTION_TYPES
    assert "replan_directive" in INTERVENTION_TYPES
    assert "tool_arg_repair" in INTERVENTION_TYPES


def test_counterfactual_engine_fork():
    mock_runner = MagicMock()
    mock_runner.run.return_value = {
        "trajectory_id": "traj_test_base",
        "task_id": "test_1",
        "steps": [],
        "task_success": True,
    }

    engine = CounterfactualEngine(runner=mock_runner)
    cf_traj = engine.fork_and_intervene(
        task_id="test_1",
        task_description="Solve problem",
        intervention_step=2,
        intervention_type="tool_arg_repair",
        seed=42,
    )

    assert cf_traj["is_counterfactual"] is True
    assert cf_traj["intervention_type"] == "tool_arg_repair"
    assert cf_traj["intervention_step"] == 2
    assert mock_runner.run.called

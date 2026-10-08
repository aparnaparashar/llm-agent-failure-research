import pytest
from telemetry.schema import StepTelemetry
from telemetry.collector import TelemetryCollector
from telemetry.online_snapshot import OnlineSnapshot


def test_step_telemetry_schema():
    record = StepTelemetry(
        trajectory_id="traj_001",
        step=1,
        timestamp=100.0,
        llm_action="tool_call",
        tool_name="test_tool",
        error_flag=False,
    )
    d = record.to_dict()
    assert d["trajectory_id"] == "traj_001"
    assert d["step"] == 1
    assert d["llm_action"] == "tool_call"
    assert d["tool_name"] == "test_tool"
    assert d["error_flag"] is False


def test_telemetry_collector_lifecycle():
    collector = TelemetryCollector(trajectory_id="traj_col_1")
    rec1 = collector.record_step(
        step=0,
        llm_action="tool_call",
        tool_name="search",
        error_flag=False,
    )
    assert rec1.step == 0
    assert len(collector.get_records()) == 1

    rec2 = collector.record_step(
        step=1,
        llm_action="tool_call",
        tool_name="search",
        error_flag=True,
        error_type="tool_execution_error",
    )
    assert rec2.step == 1
    assert rec2.error_flag is True
    assert len(collector.get_records()) == 2

    snapshot_0 = collector.get_snapshot(up_to_step=0)
    assert len(snapshot_0) == 1
    assert snapshot_0[0]["step"] == 0

    snapshot_1 = collector.get_snapshot(up_to_step=1)
    assert len(snapshot_1) == 2


def test_online_snapshot_features():
    snapshot = OnlineSnapshot()
    assert snapshot.get_features() == {}

    snapshot.update({
        "step": 0,
        "tool_name": "search",
        "error_flag": False,
        "latency": 1.5,
    })
    feat1 = snapshot.get_features()
    assert feat1["tool_count"] == 1
    assert feat1["error_count"] == 0
    assert feat1["consecutive_errors"] == 0

    # Add error step
    snapshot.update({
        "step": 1,
        "tool_name": "search",
        "error_flag": True,
        "latency": 2.0,
    })
    feat2 = snapshot.get_features()
    assert feat2["tool_count"] == 2
    assert feat2["error_count"] == 1
    assert feat2["consecutive_errors"] == 1

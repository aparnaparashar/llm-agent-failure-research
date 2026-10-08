import pytest
import numpy as np
from features.extractor import OnlineFeatureExtractor, FEATURE_NAMES


@pytest.fixture
def sample_trajectory():
    return {
        "trajectory_id": "test_traj_01",
        "task_id": "task_1",
        "steps": [
            {
                "step": 0,
                "timestamp": 100.0,
                "role": "user",
                "content": "Please search for hotels in Paris and book the cheapest one.",
            },
            {
                "step": 1,
                "timestamp": 101.5,
                "role": "assistant",
                "content": "I will search for hotels in Paris now.",
                "has_tool_calls": True,
                "tool_calls": [{"name": "search_hotels", "args": {"city": "Paris"}}],
            },
            {
                "step": 2,
                "timestamp": 102.0,
                "role": "tool",
                "message_type": "ToolMessage",
                "content": "Error: Timeout connecting to hotel API server.",
            },
            {
                "step": 3,
                "timestamp": 103.5,
                "role": "assistant",
                "content": "The tool returned an error. I will retry searching.",
                "has_tool_calls": True,
                "tool_calls": [{"name": "search_hotels", "args": {"city": "Paris"}}],
            },
            {
                "step": 4,
                "timestamp": 104.2,
                "role": "tool",
                "message_type": "ToolMessage",
                "content": "Found 3 hotels: Hotel A ($100), Hotel B ($120), Hotel C ($80).",
            },
        ],
    }


def test_feature_extractor_causality(sample_trajectory):
    extractor = OnlineFeatureExtractor(max_steps=10)

    # Features at step 1: step 2, 3, 4 must NOT be visible!
    feat_step1 = extractor.extract_features_at_step(sample_trajectory, step_idx=1)
    assert feat_step1["tool_error_count"] == 0.0
    assert feat_step1["consecutive_tool_errors"] == 0.0
    assert feat_step1["observation_error_flag"] == 0.0

    # Features at step 2: tool error should now be detected!
    feat_step2 = extractor.extract_features_at_step(sample_trajectory, step_idx=2)
    assert feat_step2["tool_error_count"] == 1.0
    assert feat_step2["consecutive_tool_errors"] == 1.0
    assert feat_step2["observation_error_flag"] == 1.0

    # Features at step 4: latest tool observation is success, observation_error_flag becomes 0.0
    feat_step4 = extractor.extract_features_at_step(sample_trajectory, step_idx=4)
    assert feat_step4["observation_error_flag"] == 0.0
    assert feat_step4["consecutive_tool_errors"] == 0.0
    assert feat_step4["tool_error_count"] == 1.0


def test_feature_vector_shape_and_names(sample_trajectory):
    extractor = OnlineFeatureExtractor()
    vec = extractor.extract_features_vector(sample_trajectory, step_idx=3)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()

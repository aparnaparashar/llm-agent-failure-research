import pytest
from policy.policies import (
    PassivePolicy,
    StaticThresholdPolicy,
    CalibratedThresholdPolicy,
    AdaptiveRegretPolicy,
    RegretCalculator,
)


def test_passive_policy():
    policy = PassivePolicy()
    dec = policy.decide_intervention(step=1, predicted_risk=0.99)
    assert dec["should_intervene"] is False
    assert dec["intervention_type"] == "none"
    assert dec["policy_name"] == "passive"


def test_static_threshold_policy():
    policy = StaticThresholdPolicy(threshold=0.6, intervention_type="warning_injection")
    # Below threshold
    dec_low = policy.decide_intervention(step=1, predicted_risk=0.4)
    assert dec_low["should_intervene"] is False
    assert dec_low["intervention_type"] == "none"

    # Above threshold
    dec_high = policy.decide_intervention(step=1, predicted_risk=0.75)
    assert dec_high["should_intervene"] is True
    assert dec_high["intervention_type"] == "warning_injection"


def test_calibrated_threshold_policy():
    policy = CalibratedThresholdPolicy(threshold=0.5, intervention_type="replan_directive")
    # Calibrated risk is below threshold even if uncalibrated is higher
    dec = policy.decide_intervention(step=1, predicted_risk=0.8, calibrated_risk=0.3)
    assert dec["should_intervene"] is False

    dec2 = policy.decide_intervention(step=1, predicted_risk=0.3, calibrated_risk=0.6)
    assert dec2["should_intervene"] is True
    assert dec2["intervention_type"] == "replan_directive"


def test_adaptive_regret_policy():
    policy = AdaptiveRegretPolicy(cost_fp=1.0, cost_fn=5.0, cost_fix=0.5)
    # Optimal threshold: 1.0 / (5.0 - 0.5 + 1.0) = 1.0 / 5.5 = ~0.1818
    assert 0.15 < policy.optimal_threshold < 0.25

    dec = policy.decide_intervention(step=1, calibrated_risk=0.1)
    assert dec["should_intervene"] is False

    dec2 = policy.decide_intervention(step=1, calibrated_risk=0.85)
    assert dec2["should_intervene"] is True
    assert dec2["intervention_type"] == "tool_arg_repair"


def test_regret_calculator():
    calc = RegretCalculator(cost_fp=1.5, cost_fn=6.0)

    # True Negative: No intervention on successful run
    r1 = calc.calculate_regret(intervened=False, factual_failed=False, counterfactual_succeeded=True)
    assert r1["regret"] == 0.0

    # False Positive: Intervened on run that would have succeeded
    r2 = calc.calculate_regret(intervened=True, factual_failed=False, counterfactual_succeeded=True)
    assert r2["regret"] == 1.5
    assert r2["decision_case"] == "false_positive_intervention"

    # False Negative: Did not intervene and task failed
    r3 = calc.calculate_regret(intervened=False, factual_failed=True, counterfactual_succeeded=True)
    assert r3["regret"] == 6.0
    assert r3["decision_case"] == "false_negative_missed_failure"

    # Successful intervention: Intervened and prevented failure
    r4 = calc.calculate_regret(intervened=True, factual_failed=True, counterfactual_succeeded=True)
    assert r4["regret"] == 0.0

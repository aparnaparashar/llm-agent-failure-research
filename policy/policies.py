"""
Adaptive Prevention Policies and Regret Minimization.
Implements Policy 0 (Passive), Policy 1 (Static Threshold), Policy 2 (Calibrated),
and Policy 3 (Adaptive Cost-Sensitive Regret Minimizing).
"""

from typing import Dict, Any, List, Optional
import numpy as np


class PreventionPolicy:
    """Base class for intervention policies."""

    def decide_intervention(
        self,
        step: int,
        predicted_risk: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class PassivePolicy(PreventionPolicy):
    """Policy 0: Never intervene."""

    def decide_intervention(
        self,
        step: int,
        predicted_risk: float = 0.0,
        calibrated_risk: Optional[float] = None,
        risk: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        r = predicted_risk if predicted_risk != 0.0 else (calibrated_risk if calibrated_risk is not None else risk)
        return {
            "should_intervene": False,
            "intervention_type": "none",
            "policy_name": "passive",
            "risk": r,
        }


class StaticThresholdPolicy(PreventionPolicy):
    """Policy 1: Intervene if uncalibrated risk > threshold."""

    def __init__(self, threshold: float = 0.5, intervention_type: str = "warning_injection"):
        self.threshold = threshold
        self.intervention_type = intervention_type

    def decide_intervention(
        self,
        step: int,
        predicted_risk: float = None,
        calibrated_risk: float = None,
        risk: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r = predicted_risk if predicted_risk is not None else (calibrated_risk if calibrated_risk is not None else risk)
        should_intervene = bool(r >= self.threshold)
        return {
            "should_intervene": should_intervene,
            "intervention_type": self.intervention_type if should_intervene else "none",
            "policy_name": "static_threshold",
            "risk": r,
            "threshold": self.threshold,
        }


class CalibratedThresholdPolicy(PreventionPolicy):
    """Policy 2: Intervene using calibrated probability."""

    def __init__(self, threshold: float = 0.45, intervention_type: str = "replan_directive"):
        self.threshold = threshold
        self.intervention_type = intervention_type

    def decide_intervention(
        self,
        step: int,
        predicted_risk: float = None,
        calibrated_risk: float = None,
        risk: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r = calibrated_risk if calibrated_risk is not None else (predicted_risk if predicted_risk is not None else risk)
        should_intervene = bool(r >= self.threshold)
        return {
            "should_intervene": should_intervene,
            "intervention_type": self.intervention_type if should_intervene else "none",
            "policy_name": "calibrated_threshold",
            "risk": r,
            "threshold": self.threshold,
        }


class AdaptiveRegretPolicy(PreventionPolicy):
    """
    Policy 3: Cost-sensitive regret-minimizing adaptive policy.
    Decides intervention based on false-positive cost (C_FP) vs false-negative cost (C_FN).
    Intervene if risk >= C_FP / (C_FN + C_FP).
    """

    def __init__(
        self,
        cost_fp: float = 1.0,    # Cost of unnecessary intervention (extra tokens, delay)
        cost_fn: float = 5.0,    # Cost of unmitigated task failure
        cost_fix: float = 0.5,   # Cost of executing repair
    ):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_fix = cost_fix
        # Optimal Bayes decision threshold
        denom = max(0.01, (self.cost_fn - self.cost_fix + self.cost_fp))
        self.optimal_threshold = float(self.cost_fp / denom)

    def decide_intervention(
        self,
        step: int,
        predicted_risk: float = None,
        calibrated_risk: float = None,
        risk: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r = calibrated_risk if calibrated_risk is not None else (predicted_risk if predicted_risk is not None else risk)
        should_intervene = bool(r >= self.optimal_threshold)
        
        # Select intervention type adaptively based on risk severity
        if not should_intervene:
            action = "none"
        elif r >= 0.8:
            action = "tool_arg_repair"
        elif r >= 0.5:
            action = "replan_directive"
        else:
            action = "warning_injection"

        return {
            "should_intervene": should_intervene,
            "intervention_type": action,
            "policy_name": "adaptive_regret",
            "risk": r,
            "optimal_threshold": round(self.optimal_threshold, 3),
        }


class RegretCalculator:
    """Calculates empirical intervention regret."""

    def __init__(self, cost_fp: float = 1.0, cost_fn: float = 5.0):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn

    def calculate_regret(
        self,
        intervened: bool,
        factual_failed: bool,
        counterfactual_succeeded: bool,
    ) -> Dict[str, float]:
        """
        Calculates regret for a single trajectory decision:
        - If intervened and factual would have succeeded anyway: False Positive Regret = C_FP
        - If did not intervene and factual failed: False Negative Regret = C_FN
        - If intervened and successfully prevented failure: Net Regret = 0.0 (Reward = C_FN - C_FP)
        - If did not intervene and factual succeeded: Net Regret = 0.0
        """
        if intervened:
            if not factual_failed:
                # Unnecessary intervention on healthy run
                regret = self.cost_fp
                case = "false_positive_intervention"
            else:
                # Needed intervention
                regret = 0.0
                case = "successful_prevention" if counterfactual_succeeded else "attempted_prevention"
        else:
            if factual_failed:
                # Missed failure that could have been intervened
                regret = self.cost_fn
                case = "false_negative_missed_failure"
            else:
                # Correct non-intervention
                regret = 0.0
                case = "true_negative_no_intervention"

        return {
            "regret": float(regret),
            "decision_case": case,
        }

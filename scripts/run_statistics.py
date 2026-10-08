#!/usr/bin/env python3
"""
Statistical Rigor & Significance Testing (PROMPT §36, §59).
Calculates:
1. Non-parametric Bootstrap 95% Confidence Intervals for:
   - ROC-AUC, PR-AUC, F1, Precision, Recall, Brier Score, ECE
2. Paired Hypothesis Tests (XGBoost vs. Each Baseline):
   - McNemar's Test for paired binary classifications
   - Wilcoxon Signed-Rank Test across bootstrap resamples
   - Paired Permutation Test
3. Multiple Testing Correction:
   - Bonferroni correction
   - Benjamini-Hochberg False Discovery Rate (FDR)
4. Saves structured outputs to results/statistical_tests/.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.extractor import FEATURE_NAMES
from models.failure_detector.classifier import FailureDetector
from evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_statistics")


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Computes mean, median, std, and 95% percentile bootstrap CIs for each metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "brier_score", "ece"]
    boot_records = {k: [] for k in metric_keys}

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        
        # Guard against single-class resample
        if len(np.unique(sample_true)) < 2:
            continue

        res = EvaluationMetrics.compute_detection_metrics(sample_true, sample_prob)
        for k in metric_keys:
            boot_records[k].append(res[k])

    summary = {}
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    for k in metric_keys:
        vals = np.array(boot_records[k])
        summary[k] = {
            "N": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "ci_lower": float(np.percentile(vals, lower_pct)),
            "ci_upper": float(np.percentile(vals, upper_pct)),
        }
    return summary


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Dict[str, Any]:
    """
    McNemar test comparing model A vs model B.
    Contingency table:
      b: A correct, B incorrect
      c: A incorrect, B correct
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    n_00 = int(np.sum((~correct_a) & (~correct_b)))  # both wrong
    n_01 = int(np.sum((~correct_a) & (correct_b)))   # A wrong, B right
    n_10 = int(np.sum((correct_a) & (~correct_b)))   # A right, B wrong
    n_11 = int(np.sum((correct_a) & (correct_b)))    # both right

    b = n_10
    c = n_01
    total_discordant = b + c

    if total_discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Continuity-corrected chi-square: (|b - c| - 1)^2 / (b + c)
        stat = float((abs(b - c) - 1.0) ** 2 / total_discordant) if abs(b - c) > 0 else 0.0
        p_val = float(stats.chi2.sf(stat, df=1))

    return {
        "contingency_table": {"both_correct": n_11, "a_only": n_10, "b_only": n_01, "both_wrong": n_00},
        "chi2_stat": round(stat, 4),
        "p_value": float(p_val),
        "significant_at_05": bool(p_val < 0.05),
    }


def permutation_test_auc(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-sided paired permutation test for difference in ROC-AUC between A and B."""
    rng = np.random.RandomState(seed)
    from sklearn.metrics import roc_auc_score

    obs_auc_a = roc_auc_score(y_true, probs_a)
    obs_auc_b = roc_auc_score(y_true, probs_b)
    obs_diff = obs_auc_a - obs_auc_b

    diffs = []
    n = len(y_true)
    for _ in range(n_permutations):
        # Randomly swap predictions between A and B for each sample
        swap = rng.rand(n) > 0.5
        perm_a = np.where(swap, probs_b, probs_a)
        perm_b = np.where(swap, probs_a, probs_b)
        diffs.append(roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b))

    diffs = np.array(diffs)
    p_val = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))

    return {
        "observed_diff": round(float(obs_diff), 4),
        "p_value": float(p_val),
        "significant_at_05": bool(p_val < 0.05),
    }


def wilcoxon_test_bootstrap(
    diff_scores: np.ndarray,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on paired differences."""
    nonzero_diffs = diff_scores[diff_scores != 0]
    if len(nonzero_diffs) < 10:
        return {"stat": 0.0, "p_value": 1.0, "significant_at_05": False}
    
    stat, p_val = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
    return {
        "stat": round(float(stat), 4),
        "p_value": float(p_val),
        "significant_at_05": bool(p_val < 0.05),
    }


def apply_fdr_and_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[float], List[bool], List[bool]]:
    """Applies Bonferroni and Benjamini-Hochberg (FDR) corrections."""
    m = len(p_values)
    if m == 0:
        return [], [], [], []

    # Bonferroni
    bonf_p = [min(1.0, p * m) for p in p_values]
    bonf_sig = [p_adj < alpha for p_adj in bonf_p]

    # Benjamini-Hochberg
    sorted_indices = np.argsort(p_values)
    fdr_p = np.zeros(m)
    sorted_p = np.array(p_values)[sorted_indices]
    
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = (sorted_p[i] * m) / rank
        running_min = min(running_min, adj)
        fdr_p[sorted_indices[i]] = min(1.0, running_min)

    fdr_sig = [p_adj < alpha for p_adj in fdr_p]
    return bonf_p, list(fdr_p), bonf_sig, fdr_sig


def main():
    logger.info("Starting statistical significance evaluation...")

    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    available_features = [col for col in FEATURE_NAMES if col in df.columns]
    X = df[available_features].values.astype(np.float32)
    y = df["imminent_failure_H2"].values.astype(int)

    # Trajectory-disjoint split
    trajectories = df["trajectory_id"].unique()
    rng = np.random.RandomState(42)
    shuffled_trajs = rng.permutation(trajectories)
    n_train = int(len(shuffled_trajs) * 0.70)
    train_trajs = set(shuffled_trajs[:n_train])
    test_trajs = set(shuffled_trajs[n_train:])

    train_mask = df["trajectory_id"].isin(train_trajs).values
    test_mask = df["trajectory_id"].isin(test_trajs).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    models = {
        "XGBoost": FailureDetector("xgboost"),
        "Gradient Boosting": FailureDetector("gradient_boosting"),
        "Random Forest": FailureDetector("random_forest"),
        "Logistic Regression": FailureDetector("logistic_regression"),
        "MLP Classifier": FailureDetector("mlp"),
    }

    # Train all models and gather test probabilities
    logger.info("Fitting models on training split...")
    probs = {}
    preds = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)
        probs[name] = p
        preds[name] = (p >= 0.5).astype(int)

    # 1. Bootstrap 95% Confidence Intervals for each model
    logger.info("Computing Bootstrap 95% Confidence Intervals (B=1000)...")
    boot_cis = {}
    for name, p in probs.items():
        logger.info(f"  Bootstrapping for {name}...")
        boot_cis[name] = bootstrap_metric_ci(y_test, p, n_bootstraps=1000, seed=42)

    # 2. Paired Hypothesis Tests (XGBoost vs Baselines)
    logger.info("Computing paired tests comparing XGBoost vs each baseline...")
    comparisons = {}
    mcnemar_p_vals = []
    comparison_names = []

    for name in models.keys():
        if name == "XGBoost":
            continue

        comparison_names.append(f"XGBoost vs {name}")
        mcnemar_res = mcnemar_test(y_test, preds["XGBoost"], preds[name])
        mcnemar_p_vals.append(mcnemar_res["p_value"])

        # Paired permutation test for AUC difference
        perm_res = permutation_test_auc(y_test, probs["XGBoost"], probs[name], n_permutations=1000, seed=42)

        # Wilcoxon on bootstrap sample AUC differences
        rng_boot = np.random.RandomState(42)
        n_test = len(y_test)
        auc_diffs = []
        from sklearn.metrics import roc_auc_score
        for _ in range(500):
            idx = rng_boot.randint(0, n_test, size=n_test)
            if len(np.unique(y_test[idx])) > 1:
                auc_a = roc_auc_score(y_test[idx], probs["XGBoost"][idx])
                auc_b = roc_auc_score(y_test[idx], probs[name][idx])
                auc_diffs.append(auc_a - auc_b)
        
        wilcox_res = wilcoxon_test_bootstrap(np.array(auc_diffs))

        comp_key = f"XGBoost_vs_{name.replace(' ', '_')}"
        comparisons[comp_key] = {
            "baseline": name,
            "mcnemar": mcnemar_res,
            "permutation_auc": perm_res,
            "wilcoxon_bootstrap_auc": wilcox_res,
        }

    # 3. Multiple comparisons correction
    bonf_p, fdr_p, bonf_sig, fdr_sig = apply_fdr_and_bonferroni(mcnemar_p_vals)
    for idx, comp_name in enumerate(comparison_names):
        key = comp_name.replace(" ", "_")
        comparisons[key]["mcnemar"]["bonferroni_p"] = round(float(bonf_p[idx]), 6)
        comparisons[key]["mcnemar"]["bonferroni_sig"] = bool(bonf_sig[idx])
        comparisons[key]["mcnemar"]["fdr_bh_p"] = round(float(fdr_p[idx]), 6)
        comparisons[key]["mcnemar"]["fdr_bh_sig"] = bool(fdr_sig[idx])

    # Save outputs
    out_dir = os.path.join(PROJECT_ROOT, "results", "statistical_tests")
    os.makedirs(out_dir, exist_ok=True)

    ci_json = os.path.join(out_dir, "bootstrap_confidence_intervals.json")
    with open(ci_json, "w", encoding="utf-8") as f:
        json.dump(boot_cis, f, indent=2)

    tests_json = os.path.join(out_dir, "significance_results.json")
    with open(tests_json, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    # Generate Markdown summary tables
    ci_md = os.path.join(out_dir, "bootstrap_confidence_intervals.md")
    with open(ci_md, "w", encoding="utf-8") as f:
        f.write("# Bootstrap 95% Confidence Intervals (B=1000, Test Split)\n\n")
        f.write("| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | F1 (95% CI) | Brier Score (95% CI) | ECE (95% CI) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, metrics in boot_cis.items():
            f.write(
                f"| **{name}** | "
                f"{metrics['roc_auc']['mean']:.3f} [{metrics['roc_auc']['ci_lower']:.3f}, {metrics['roc_auc']['ci_upper']:.3f}] | "
                f"{metrics['pr_auc']['mean']:.3f} [{metrics['pr_auc']['ci_lower']:.3f}, {metrics['pr_auc']['ci_upper']:.3f}] | "
                f"{metrics['f1']['mean']:.3f} [{metrics['f1']['ci_lower']:.3f}, {metrics['f1']['ci_upper']:.3f}] | "
                f"{metrics['brier_score']['mean']:.3f} [{metrics['brier_score']['ci_lower']:.3f}, {metrics['brier_score']['ci_upper']:.3f}] | "
                f"{metrics['ece']['mean']:.3f} [{metrics['ece']['ci_lower']:.3f}, {metrics['ece']['ci_upper']:.3f}] |\n"
            )

    tests_md = os.path.join(out_dir, "significance_table.md")
    with open(tests_md, "w", encoding="utf-8") as f:
        f.write("# Statistical Significance Tests (XGBoost vs. Baselines)\n\n")
        f.write("| Baseline | McNemar Chi2 | Raw p-value | Bonferroni p | FDR (B-H) p | Permutation AUC Diff | Wilcoxon p |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for key, res in comparisons.items():
            base = res["baseline"]
            mcn = res["mcnemar"]
            perm = res["permutation_auc"]
            wilc = res["wilcoxon_bootstrap_auc"]
            f.write(
                f"| **{base}** | {mcn['chi2_stat']} | {mcn['p_value']:.4e} | {mcn['bonferroni_p']:.4e} | "
                f"{mcn['fdr_bh_p']:.4e} | +{perm['observed_diff']:.3f} (p={perm['p_value']:.3f}) | {wilc['p_value']:.4e} |\n"
            )

    logger.info(f"Statistical rigor artifacts successfully written to {out_dir}")
    print("\n" + "=" * 70)
    print("STATISTICAL RIGOR SUMMARY")
    print("=" * 70)
    with open(tests_md, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()

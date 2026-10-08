"""
Publication-Quality Research Figures Generator.
Generates all 8 static figures specified in the project requirements:
1. risk_probability_over_time.png
2. calibration_curve.png
3. lead_time_distribution.png
4. regret_distribution.png
5. failure_type_comparison.png
6. ablation_results.png
7. generalization_results.png
8. injected_vs_organic.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from sklearn.calibration import calibration_curve

# Publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


class ResearchFigureGenerator:
    """Generates publication-grade figures for paper and thesis."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_risk_over_time(self, step_series_healthy: List[float], step_series_failing: List[float], save_name: str = "risk_probability_over_time.png"):
        """Figure 1: Risk Probability Over Time (Healthy vs Failing trajectories)."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        steps_h = range(len(step_series_healthy))
        steps_f = range(len(step_series_failing))

        ax.plot(steps_h, step_series_healthy, label="Healthy Trajectory (Normal Run)", color="#2b5c8f", marker="o", linewidth=2.2)
        ax.plot(steps_f, step_series_failing, label="Failing Trajectory (Injected/Organic Fault)", color="#d9534f", marker="s", linewidth=2.2)
        ax.axhline(0.45, color="#888888", linestyle="--", linewidth=1.5, label="Intervention Threshold (tau=0.45)")

        ax.set_title("Runtime Failure Risk Probability Over Execution Steps")
        ax.set_xlabel("Agent Execution Step (t)")
        ax.set_ylabel("P(Failure_future | Telemetry_<=t)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_calibration_curve(self, y_true: np.ndarray, y_prob_uncal: np.ndarray, y_prob_cal: np.ndarray, save_name: str = "calibration_curve.png"):
        """Figure 2: Reliability Diagram (Calibration Curve)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        prob_true_u, prob_pred_u = calibration_curve(y_true, y_prob_uncal, n_bins=8)
        prob_true_c, prob_pred_c = calibration_curve(y_true, y_prob_cal, n_bins=8)

        ax.plot(prob_pred_u, prob_true_u, "s-", color="#e67e22", label="Uncalibrated Random Forest")
        ax.plot(prob_pred_c, prob_true_c, "o-", color="#27ae60", linewidth=2, label="Platt Calibrated Sigmoid")

        ax.set_title("Reliability Diagram (Probability Calibration)")
        ax.set_xlabel("Mean Predicted Risk Probability")
        ax.set_ylabel("Fraction of True Positive Failures")
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_lead_time_distribution(self, lead_times: List[int], save_name: str = "lead_time_distribution.png"):
        """Figure 3: Early Warning Lead Time Distribution."""
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bins = np.arange(min(lead_times or [0]) - 0.5, max(lead_times or [3]) + 1.5, 1)
        ax.hist(lead_times, bins=bins, color="#3498db", edgecolor="#1d6fa5", alpha=0.85, rwidth=0.7)
        ax.axvline(np.mean(lead_times or [1]), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean Lead Time: {np.mean(lead_times or [1]):.2f} steps")

        ax.set_title("Early Failure Detection Lead Time Distribution")
        ax.set_xlabel("Lead Time Steps (Delta t before Actual Failure)")
        ax.set_ylabel("Trajectory Frequency")
        ax.legend(frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_regret_distribution(self, regrets_by_policy: Dict[str, List[float]], save_name: str = "regret_distribution.png"):
        """Figure 4: Regret Distribution across Policies."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = list(regrets_by_policy.keys())
        data = [regrets_by_policy[k] for k in labels]

        bplot = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title("Policy Regret Distribution Comparison")
        ax.set_ylabel("Empirical Regret (Loss Units)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_failure_type_comparison(self, f1_scores: Dict[str, float], save_name: str = "failure_type_comparison.png"):
        """Figure 5: Detection F1 by Failure Taxonomy Type."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        types = list(f1_scores.keys())
        scores = [f1_scores[k] for k in types]

        bars = ax.barh(types, scores, color="#4a90e2", edgecolor="#205081", height=0.55)
        ax.set_xlim(0, 1.05)
        ax.set_title("Failure Detection F1-Score across Unified Taxonomy")
        ax.set_xlabel("F1-Score")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_ablation_results(self, ablation_results: Dict[str, float], save_name: str = "ablation_results.png"):
        """Figure 6: Telemetry Feature Group Ablation Study."""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        configs = list(ablation_results.keys())
        rocs = [ablation_results[k] for k in configs]

        bars = ax.bar(configs, rocs, color="#16a085", edgecolor="#0e6655", width=0.5)
        ax.set_ylim(0.4, 1.0)
        ax.set_title("Feature Group Ablation: Detection ROC-AUC")
        ax.set_ylabel("ROC-AUC")
        plt.xticks(rotation=15, ha="right")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_generalization_results(self, gen_scores: Dict[str, Dict[str, float]], save_name: str = "generalization_results.png"):
        """Figure 7: In-Distribution vs Out-of-Distribution Generalization."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tasks = list(gen_scores.keys())
        in_dist = [gen_scores[t]["in_dist"] for t in tasks]
        ood = [gen_scores[t]["ood"] for t in tasks]

        x = np.arange(len(tasks))
        width = 0.35

        ax.bar(x - width/2, in_dist, width, label="In-Distribution (ToolBench)", color="#2980b9")
        ax.bar(x + width/2, ood, width, label="Out-of-Distribution (AgentErrorBench)", color="#e67e22")

        ax.set_title("Cross-Domain Generalization Evaluation")
        ax.set_ylabel("Detection F1-Score")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_injected_vs_organic(self, dist_injected: List[float], dist_organic: List[float], save_name: str = "injected_vs_organic.png"):
        """Figure 8: Injected vs Organic Telemetry Feature Distributions."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(dist_injected, bins=15, alpha=0.6, label="Injected Fault Telemetry", color="#e74c3c", density=True)
        ax.hist(dist_organic, bins=15, alpha=0.6, label="Organic Fault Telemetry", color="#3498db", density=True)

        ax.set_title("Telemetry Distribution Alignment: Injected vs Organic")
        ax.set_xlabel("Normalized Telemetry Anomaly Score")
        ax.set_ylabel("Probability Density")
        ax.legend(frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path

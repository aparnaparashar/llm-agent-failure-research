# Lightweight Telemetry-Based Failure Prediction and Degradation Cascading Analysis for LLM Agents

## 1. Project Overview & Research Question

Autonomous LLM agents deployed on multi-step reasoning and tool-use workflows frequently suffer from unrecoverable failures—such as infinite tool loops, parameter corruption, cascading errors, and goal drift. Existing guardrails operate either retrospectively (after a trajectory has already failed) or incur substantial latency and financial overhead by invoking expensive LLM judges at every step.

**Core Research Question:**
> *Can low-cost runtime telemetry collected during execution of an LLM agent predict an imminent failure within a forward horizon ($H=2$) before the next LLM/tool call, and how does telemetry perform in detecting ongoing failure cascades versus cold-start initial tool faults?*

The prediction formulation strictly enforces causal ordering:
$$\mathcal{P}(\text{Failure}_{\text{future}} \mid \text{Telemetry}_{\le t})$$
**Zero Future Leakage:** Only runtime telemetry observable at or before step $t$ is included in the feature representation. No future outcomes, trajectory lengths, or injection metadata are ever leaked to the predictor.

---

## 2. Experimental Architecture

```
ToolBench Grounded Tasks
        │
        ▼
LangGraph Agent Execution (Planning -> Tool Selection -> Observation)
        │
        ├── LLM Provider: Ollama (qwen3:8b)
        │
        ├── Ingestion & Injection Engine
        │     ├── Mode A: Healthy Run (No fault)
        │     ├── Mode B: Controlled Injected Run (Causal injection during tool call)
        │     └── Mode C: Organic Run (Natural model failures)
        │
        ▼
Runtime Telemetry Logger (Collector & Online Snapshot)
        │
        ▼
Online Feature Extractor (14 Lightweight Causal Metrics)
        │
        ├── Failure Detectors (XGBoost, HistGradientBoosting, Random Forest, Logistic Reg, MLP)
        ├── Probability Calibrator (Platt Sigmoid / Isotonic Scaling)
        │
        ▼
Evaluations & Analysis Pipeline
        ├── 1. Multi-Model Benchmark (ROC-AUC 0.908, PR-AUC 0.831)
        ├── 2. Probability Calibration (ECE = 0.0231)
        ├── 3. Runtime Failure Persistence & Degradation Cascading Analysis
        ├── 4. 21-Category Failure Taxonomy Attribution
        ├── 5. Feature Family Ablation Study
        ├── 6. Execution Mode Robustness (Organic vs Injected)
        └── 7. Latency Profiling (0.004 ms/step inference)
```

---

## 3. Core Experimental Results Summary

* **Discriminative Power:** Primary XGBoost detector achieves **0.9076 ROC-AUC**, **0.8311 PR-AUC**, and **0.7328 F1** on held-out trajectory-disjoint test set (2,753 steps).
* **Probability Calibration:** Natural uncalibrated ECE of **0.0231** (2.3% error margin), ensuring risk outputs behave as honest probabilities.
* **Failure Persistence & Degradation Cascading:**
  * **Step-1 Cold-Start Baseline:** In 240 of 305 failing test trajectories (78.69%), failures occur at the very first tool call ($t=1$) where historical telemetry is absent ($L=0$).
  * **Multi-Step Dynamic Cascades:** On subsequent degradation and runaway loops ($t \ge 2$), the detector flags failing runs with an average lead time of **2.66 steps** (up to 56 steps).
  * **Failure Persistence:** Accurately flags 94.75% of failing trajectories, serving as an ultra-fast filter separating transient recoveries from fatal cascades.
* **Feature Drivers:** Error signals are the primary marginal contributor (-0.029 ROC-AUC when removed). Context dynamics contribute -0.017 ROC-AUC.
* **Execution Efficiency:** Scalar telemetry inference executes in **0.004 ms per step** with **zero token cost**.

---

## 4. Repository Structure

```
├── agent/                       # LangGraph ReAct agent scaffold
├── configs/                     # System and experiment configurations
├── data/                        # Processed, raw, and split datasets (70/15/15)
├── experiments/                 # Experiment manifests and audit records
├── features/                    # 14-dimensional causal telemetry extraction
├── models/                      # XGBoost, tree baselines, and calibration modules
├── results/                     # High-res publication figures and structured tables
├── scripts/                     # Automated data generation, training, and evaluation
│   ├── 01_dataset_curation/
│   ├── 02_feature_extraction_and_splits/
│   ├── 03_training_and_calibration/
│   ├── 04_policy_and_regret/
│   ├── 05_evaluation_and_benchmarks/
│   │   ├── evaluate_failure_persistence.py  # Runtime Persistence & Cascade Analysis
│   │   ├── evaluate_organic.py
│   │   ├── run_ablations.py
│   │   └── run_statistics.py
│   └── 06_reporting_and_figures/
└── README.md                    # Primary documentation
```

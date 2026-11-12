# Complete Step-by-Step Experiments Deep Dive
## LLM Agent Failure Research: Technical Execution Guide

This document details the exact input, mathematical logic, code execution flow, and metric generation for every experiment in the research project.

---

## 1. Why Did We NOT Re-Run the 4,000 Queries for Every Experiment?

1. **The Principle of Scientific Reproducibility & Benchmark Stability:**
   LLMs have non-zero temperature and inherent stochasticity. If queries were re-executed for every downstream experiment, the benchmark dataset would mutate between runs, making fair, rigorous comparisons across models (e.g. XGBoost vs. Random Forest) scientifically invalid.
2. **Standard Machine Learning Protocol (Collect Once, Evaluate Rigorously):**
   Analogous to autonomous driving (where vehicle sensor logs are collected once and used to benchmark hundreds of safety algorithms without re-crashing vehicles), agent trajectories are captured with full step-by-step fidelity into structured JSON files.
3. **Compute & Environmental Feasibility:**
   Generating 4,000 trajectories on local open-weight models (`qwen3:8b`) required approximately 20 GPU hours. Re-running queries across 9 separate evaluation experiments would require over 180 hours of redundant computation.

---

## 2. Deep Dive: Every Experiment Explained

### Experiment 1: Multi-Model Benchmark (`scripts/03_training_and_calibration/train_detector.py`)
- **Input Data:** `data/splits/train/train_features.csv` (13,463 rows, 70%), `data/splits/test/test_features.csv` (2,885 rows, 15%).
- **Features Used:** 14 causal features ($X$); **Target:** `imminent_failure_H2` ($y \in \{0, 1\}$).
- **Execution Mechanism:**
  1. Initializes five classifier architectures: XGBoost, Gradient Boosting, Random Forest, Logistic Regression, and Multi-Layer Perceptron (MLP).
  2. Fits each model on `X_train, y_train`.
  3. Predicts continuous risk probabilities on unseen held-out test data: `y_prob = model.predict_proba(X_test)[:, 1]`.
  4. Evaluates `roc_auc_score(y_test, y_prob)`, `average_precision_score`, `f1_score`, and Brier score.
- **Results:** XGBoost achieved ROC-AUC 0.9076, PR-AUC 0.8311, and F1 0.7328, demonstrating optimal non-linear boundary separation on telemetry spikes.

---

### Experiment 2: Probability Calibration (`scripts/03_training_and_calibration/calibrate.py`)
- **Goal:** Assess whether a predicted risk score of $p = 0.8$ corresponds to an empirical 80% failure rate.
- **Input Data:** Trained XGBoost model + `data/splits/validation/val_features.csv` (2,885 rows, 15%).
- **Execution Mechanism:**
  1. Partitions predicted validation probabilities into 10 bins ($B_1, \dots, B_{10}$).
  2. Computes Expected Calibration Error:
     $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  3. Fits Platt Scaling ($P(y=1|p) = \sigma(Ap + B)$) and non-parametric Isotonic Regression.
- **Results:** Uncalibrated XGBoost naturally achieved ECE = 0.0231 (2.3% error), confirming that tree-based gradient boosting produces honest probabilities without post-hoc distortions.

---

### Experiment 3: Early Detection & Lead Time (`scripts/05_evaluation_and_benchmarks/evaluate_early_detection.py`)
- **Goal:** Measure how many steps in advance failure is flagged before physical trajectory termination.
- **Input Data:** 600 held-out test trajectories from `test_features.csv`.
- **Execution Mechanism:**
  1. Iterates over each trajectory group: `for traj_id, traj_data in test_df.groupby("trajectory_id"):`
  2. Obtains step-by-step risk predictions: `probs = model.predict_proba(X_traj)[:, 1]`.
  3. Identifies the first step where risk exceeds 0.5: `t_alarm = steps[np.where(probs >= 0.5)[0][0]]`.
  4. Identifies the ground-truth failure step: `t_fail = steps[np.where(y_traj == 1)[0][0]]`.
  5. Computes lead time: $\text{Lead Time } L = t_{\text{fail}} - t_{\text{alarm}}$.
- **Results:** 94.75% of failing trajectories were detected prior to or at failure, with an average lead time of 2.66 steps.

---

### Experiment 4: Decision Regret & Policy Benchmark (`scripts/04_policy_and_regret/compute_regret.py`)
- **Goal:** Benchmark the decision-theoretic trade-off between unnecessary interruptions (False Alarms) and unmitigated failures (False Negatives).
- **Input Data:** 600 test trajectories + metadata.
- **Execution Mechanism:**
  1. Formulates cost parameters: $C_{\text{FP}} = 1.0$ (cost of interrupting a healthy agent), $C_{\text{FN}} = 5.0$ (penalty for letting a failure execute to completion).
  2. Derives optimal Bayes threshold: $\theta^* = \frac{C_{\text{FP}}}{C_{\text{FN}} + C_{\text{FP}}} \approx 0.167$.
  3. Computes trajectory peak risk: $r_{\max} = \max_t(p_t)$.
  4. Determines intervention: `intervened = bool(r_max >= theta_star)`.
  5. Calculates trajectory regret:
     - If intervened on healthy run: $\text{Regret} = 1.0$.
     - If missed a failing run: $\text{Regret} = 5.0$.
     - If correct intervention or correct non-intervention: $\text{Regret} = 0.0$.
- **Results:** Adaptive Regret Policy achieved a mean regret of 0.2801 (an 83.4% reduction relative to the Passive baseline of 1.6891).

---

### Experiment 5: Injected-to-Organic Generalization (`scripts/05_evaluation_and_benchmarks/evaluate_organic.py`)
- **Goal (Hypothesis H6):** Evaluate whether models trained strictly on synthetic fault injections generalize zero-shot to natural organic LLM reasoning failures.
- **Input Data:** Test split segmented by `execution_mode`: Injected ($n=1,356$) vs. Organic ($n=430$).
- **Execution Mechanism:**
  1. Evaluates XGBoost (trained on synthetic injections) strictly on organic test steps.
  2. Computes ROC-AUC, PR-AUC, Precision, and Recall on organic instances.
- **Results:** Model achieved ROC-AUC 0.915 on organic failures (higher than 0.899 on in-distribution injected failures), validating that controlled fault injection serves as an effective training proxy for real-world monitoring.

---

### Experiment 6: 21-Category Failure Taxonomy Breakdown (`scripts/05_evaluation_and_benchmarks/evaluate_failure_taxonomy.py`)
- **Goal:** Characterize predictive sensitivity and feature drivers across 16 cognitive and 5 temporal failure modes.
- **Input Data:** `data/processed/telemetry_dataset.csv` annotated with failure categories.
- **Execution Mechanism:**
  1. Filters instances by failure mode (e.g. `looping`, `tool_execution_error`, `hallucination`).
  2. Evaluates category-specific AUC and feature attribution weights.
- **Results:** Tool execution errors are driven by `tool_error_rate` (35%), while cognitive hallucinations and loops are signaled by drops in `lexical_diversity` (35%), making `lexical_diversity` the top global feature (importance weight 0.3069).

---

### Experiment 7: Feature Ablation Study (`scripts/05_evaluation_and_benchmarks/run_ablations.py`)
- **Goal:** Determine the marginal predictive utility of the five telemetry feature groups (temporal, tool, error, context, semantic).
- **Execution Mechanism:**
  1. Masks feature groups from the training and test matrices.
  2. Re-trains XGBoost from scratch on the reduced feature space.
  3. Computes degradation: $\Delta \text{ROC-AUC} = \text{AUC}_{\text{full}} - \text{AUC}_{\text{ablated}}$.
- **Results:** Removing context features (message length, token expansion) caused the steepest drop ($\Delta \text{AUC} = -0.0284$). No individual feature group in isolation achieved above 0.77 ROC-AUC, proving that multi-modal telemetry signals are complementary.

---

### Experiment 8: Statistical Significance & Bootstrap Analysis (`scripts/05_evaluation_and_benchmarks/run_statistics.py`)
- **Goal:** Prove that performance gains are statistically significant rather than artifacts of random variance.
- **Execution Mechanism:**
  1. Performs $B = 1000$ bootstrap resamples to establish 95% Confidence Intervals for all metrics.
  2. Constructs $2 \times 2$ contingency tables comparing prediction agreements between XGBoost and baselines.
  3. Executes McNemar's test:
     $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
- **Results:** XGBoost vs. Logistic Regression produced $\chi^2 = 105.52$ ($p = 9.39 \times 10^{-25}$), and vs. Random Forest produced $\chi^2 = 18.41$ ($p = 1.78 \times 10^{-5}$), confirming statistical significance ($p < 0.05$).

---

### Experiment 9: Telemetry vs. LLM-as-a-Judge Baseline (`scripts/05_evaluation_and_benchmarks/run_llm_judge_baseline.py`)
- **Goal:** Quantify the computational and economic trade-off between lightweight telemetry scoring and prompting an LLM to evaluate agent failure at each step.
- **Execution Mechanism:**
  1. Runs telemetry feature extraction and XGBoost `predict_proba()` timed with `time.perf_counter()`.
  2. In parallel, formats the agent step into an evaluation prompt and queries `qwen3:8b` via Ollama API, measuring round-trip latency and token consumption.
- **Results:** Telemetry requires **0.004 ms per step** at **0 tokens ($0 cost)**, operating **156,219× faster** than the LLM judge (643.9 ms, 171 tokens, $342/1M steps).

# Reproducibility Guide

This guide describes how to reproduce all experimental findings, dataset splits, model training, probability calibration, counterfactual branching, policy regret metrics, and publication figures reported in the research.

---

## 1. Prerequisites and Environment Setup

- **Python Version:** 3.10+ (tested on Python 3.13)
- **Local LLM Engine:** Ollama with `qwen3:8b` (or another model specified in `configs/llm.yaml`)
- **Required Packages:** Listed in `requirements.txt`

```bash
# Clone the repository and navigate to root directory
cd llm-agent-failure-research

# Install dependencies
pip install -r requirements.txt
```

Verify Ollama is running locally:
```bash
ollama serve
# Verify qwen3:8b is available
ollama list
```

---

## 2. Step-by-Step Reproduction Workflow

### Step 1: Benchmark Acquisition and Audit
Download and verify ToolBench and AgentErrorBench grounding datasets:
```bash
python scripts/download_datasets.py
python scripts/audit_datasets.py
```
*Output Artifact:* `data/audit_report.json`

### Step 2: Agent Execution (Live LangGraph + Ollama)
Run healthy, controlled-injected, and organic agent executions:
```bash
# Healthy normal baseline runs
python scripts/run_healthy.py --config configs/agent.yaml --limit 1

# Controlled failure injection (causal execution)
python scripts/run_injected.py --config configs/injection.yaml --failure-type tool_execution_error --limit 1

# Organic failure runs
python scripts/run_organic.py --config configs/agent.yaml --limit 1
```
*Output Artifacts:* Trajectory JSON files stored under `data/generated/healthy/`, `data/generated/injected/`, and `data/generated/organic/`.

### Step 3: Feature Extraction (Zero Future Leakage)
Extract the 14 causal telemetry signals and horizon labels ($H=2$):
```bash
python scripts/build_features.py
```
*Output Artifacts:* `data/processed/features.npz` and `data/processed/features_summary.json`.

### Step 4: Model Training and Baseline Benchmarking
Train failure detection classifiers (Random Forest, Logistic Regression, Gradient Boosting, MLP) alongside heuristic baselines:
```bash
python scripts/train_detector.py
```
*Output Artifact:* `results/tables/detector_training_results.json`

### Step 5: Probability Calibration
Calibrate predicted risk using Platt scaling and calculate Brier Score and Expected Calibration Error (ECE):
```bash
python scripts/calibrate.py
```
*Output Artifact:* `results/tables/calibration_results.json`

### Step 6: Counterfactual Branching & Policy Evaluation
Fork trajectories at failure points, execute counterfactual interventions (Continue, Verify, Replan, Abort), and calculate empirical intervention regret:
```bash
python scripts/generate_counterfactuals.py
python scripts/train_policy.py
```
*Output Artifacts:* `results/tables/counterfactual_branches.json` and `results/tables/policy_summary.json`.

### Step 7: Static Research Figures & Tables
Generate the 8 publication-grade static figures and summary evaluation markdown tables:
```bash
python scripts/evaluate.py
python scripts/generate_figures.py
```
*Output Figures:*
1. `results/figures/risk_probability_over_time.png`
2. `results/figures/calibration_curve.png`
3. `results/figures/lead_time_distribution.png`
4. `results/figures/regret_distribution.png`
5. `results/figures/failure_type_comparison.png`
6. `results/figures/ablation_results.png`
7. `results/figures/generalization_results.png`
8. `results/figures/injected_vs_organic.png`

### Step 8: Full End-to-End Pipeline
To run the complete automated research pipeline from start to finish:
```bash
python scripts/run_research_pipeline.py
```

### Step 9: Automated Test Suite Verification
Run pytest to verify all unit, causal, and integration tests:
```bash
python -m pytest -q
```
All 23 tests should pass.

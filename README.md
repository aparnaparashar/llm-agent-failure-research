# Lightweight Telemetry-Based Failure Prediction and Adaptive Prevention Policies for LLM Agents

## 1. Project Overview & Research Question

Autonomous LLM agents deployed on multi-step reasoning and tool-use workflows frequently suffer from unrecoverable failures—such as infinite tool loops, parameter corruption, cascading errors, and goal drift. Existing guardrails operate either retrospectively (after a trajectory has already failed) or incur substantial latency and financial overhead by invoking expensive LLM judges at every step.

**Core Research Question:**
> *Can low-cost runtime telemetry collected during execution of an LLM agent predict an imminent failure before the next LLM/tool call, and can an adaptive intervention policy use that prediction to reduce task failure while minimizing unnecessary interventions and regret?*

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
        ├── Failure Detectors (Gradient Boosting, Random Forest, Logistic Reg, MLP)
        ├── Probability Calibrator (Platt Sigmoid / Isotonic Scaling)
        │
        ▼
Counterfactual Branching Engine (Continue, Verify, Replan, Abort)
        │
        ▼
Adaptive Prevention Policy (Regret Minimization: Cost_FP vs Cost_FN)
        │
        ▼
Empirical Regret Evaluation & Static Publication Figures
```

---

## 3. Unified Failure Taxonomy & Controlled Injection

The framework integrates the **AgentErrorTaxonomy** grounded in real agent execution with temporal injection mechanisms inspired by **Dubey et al.**:

| Module | Failure Type | Injectability | Mechanism / Signature |
| :--- | :--- | :--- | :--- |
| **System** | `tool_execution_error` | Injectable | HTTP timeouts, connection resets, 500 errors |
| **Action** | `parameter_error` | Injectable | Argument schema corruption, type mismatches |
| **Action** | `format_error` | Injectable | Malformed JSON, unparseable tool call syntax |
| **Temporal** | `looping` | Injectable | Oscillating states, repeated identical tool calls |
| **Planning** | `goal_drift` | Injectable | Semantic drift in active subgoal formulation |
| **Reflection** | `causal_misattribution` | Partially Injectable | Error observation misinterpretation |
| **System** | `llm_limit` | Organic Only | Context exhaustion, natural token degradation |

Crucially, **injection happens causally during actual tool execution**. The LLM observes the error in its message history and must autonomously choose whether to retry, replan, recover, or fail.

---

## 4. Experimental Results

### Model Performance (Imminent Failure Detection, $H=2$)

| Model / Baseline | ROC-AUC | PR-AUC | F1-Score | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.8590** | 0.5139 | **0.5000** | 0.1515 | 0.1563 |
| **Logistic Regression** | 0.8205 | **0.5873** | 0.4000 | **0.1394** | **0.0826** |
| **MLP Classifier** | 0.8205 | 0.5873 | 0.4000 | 0.1613 | 0.1268 |
| **Random Forest** | 0.7949 | 0.5694 | 0.3333 | 0.1419 | 0.1600 |
| *Baseline: Consecutive Errors* | 0.6282 | 0.2917 | 0.4000 | 0.1875 | 0.1875 |
| *Baseline: Random Guess* | 0.3590 | 0.1958 | 0.1818 | 0.4012 | 0.5012 |

### Probability Calibration
- **Uncalibrated Random Forest:** Brier Score = `0.1730`, ECE = `0.1614`
- **Platt Calibrated:** Brier Score = `0.1581`, ECE = `0.1007` (37.6% error reduction)

### Policy Evaluation & Regret Minimization

Under cost regime $C_{\text{FP}} = 1.0$ (overhead of unnecessary intervention) and $C_{\text{FN}} = 5.0$ (cost of unmitigated task failure):

$$R_i = C_i(\text{action}_{\text{policy}}) - \min_{a} C_i(a)$$

- **Policy 0 (Passive / Always Continue):** Mean Regret = `0.990`
- **Policy 1 (Static Threshold $\tau=0.5$):** Mean Regret = `0.969`
- **Policy 2 (Calibrated Threshold $\tau=0.45$):** Mean Regret = `0.958`
- **Policy 3 (Adaptive Regret Minimizing):** Mean Regret = `0.802` (Lowest regret)

---

## 5. Repository Structure

```
llm-agent-failure-research/
├── configs/                   # Experiment, LLM, model, policy, and cost configs
├── data/
│   ├── raw/                   # Raw benchmark data (ToolBench & AgentErrorBench)
│   ├── generated/             # Healthy, Injected, and Organic trajectory storage
│   └── processed/             # Extracted causal features and splits
├── agent/                     # LangGraph agent, Ollama provider, Tool registry
├── benchmarks/                # ToolBench & AgentErrorBench programmatic adapters
├── failure_taxonomy/          # Unified failure taxonomy & injectability definitions
├── failure_injection/         # Controlled runtime causal failure injectors
├── telemetry/                 # Runtime step collector, schema, and online snapshot
├── features/                  # Causal online feature extractor (14 telemetry signals)
├── labels/                    # Horizon labeling engine (H=1, 2, 3)
├── models/                    # Detectors, Baselines, and Platt/Isotonic Calibrator
├── counterfactual/            # Counterfactual state branching engine
├── policy/                    # Prevention policies & regret calculation
├── evaluation/                # Metrics calculation & publication figure generation
├── results/
│   ├── figures/               # 8 publication-grade static figures
│   └── tables/                # Research markdown & CSV summary tables
├── scripts/                   # CLI entry points for pipeline execution
└── tests/                     # Comprehensive test suite (100% passing)
```

---

## 6. CLI Quickstart

Execute individual phases or the full end-to-end pipeline:

```bash
# 1. Download and verify benchmarks
python scripts/download_datasets.py

# 2. Audit dataset schemas and distributions
python scripts/audit_datasets.py

# 3. Run healthy, injected, and organic agent executions (live Ollama)
python scripts/run_healthy.py --config configs/agent.yaml --limit 1
python scripts/run_injected.py --config configs/injection.yaml --failure-type tool_execution_error --limit 1
python scripts/run_organic.py --config configs/agent.yaml --limit 1

# 4. Build causal online telemetry features
python scripts/build_features.py

# 5. Train failure detectors and calibrate probabilities
python scripts/train_detector.py
python scripts/calibrate.py

# 6. Counterfactual branching and adaptive policy training
python scripts/generate_counterfactuals.py
python scripts/train_policy.py

# 7. Evaluate and generate paper figures & tables
python scripts/evaluate.py
python scripts/generate_figures.py

# Or run complete research experimental pipeline:
python scripts/run_research_pipeline.py

# Run test suite
python -m pytest -q
```

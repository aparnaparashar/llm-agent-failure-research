# Lightweight Telemetry Classifiers as Runtime Guardrails for LLM Agents

**Authors:** Anamika Thakur, Anshika Sharma, Anya Jain, Aparna Parashar  
**Supervisor & Project Mentor:** Prof. Seeja K. R.  
**Affiliation:** Department of Computer Science and Engineering, Indira Gandhi Delhi Technical University for Women (IGDTUW), New Delhi, 110006, India  
**Contact:** `{anamika, anshika, anya, aparna}@igdtuw.ac.in` | `seejakr@igdtuw.ac.in`  
**Target Venue:** IEEE Transactions on Reliability / ACM Conference Submission  

---

## Abstract

Large language model (LLM) agents that plan, call external tools, and consume execution results can fail in subtle ways that a check on the final answer does not reveal. The standard remedy—querying a secondary evaluator LLM at every step—introduces substantial latency and token cost. This paper investigates how much failure-predictive signal resides within cheap, non-semantic execution telemetry. We instrument a LangGraph ReAct agent driven by a local Qwen3-8B model, execute 4,000 ToolBench tasks across healthy, fault-injected, and organic operating modes, and log structured runtime telemetry at every step. Fourteen strictly causal features capturing error accumulation, tool invocation patterns, context growth, execution latency, progress, and lexical diversity are derived from the trajectory prefix. A gradient-boosted tree classifier estimates whether a failure event will occur within the subsequent two steps ($H=2$). On a trajectory-disjoint held-out test split of 600 trajectories (2,753 steps), the XGBoost detector attains a ROC-AUC of 0.908, a PR-AUC of 0.831, an F1 score of 0.733 (precision 0.761, recall 0.707), and an expected calibration error (ECE) of 0.023 without post-hoc scaling. Histogram gradient boosting performs equivalently (ROC-AUC 0.909), whereas L2-regularized logistic regression trails by 0.05 ROC-AUC. Feature ablations demonstrate that error-signal features provide the largest marginal contribution (0.029 ROC-AUC drop when removed), while isolated feature families do not exceed 0.776 ROC-AUC. Trajectory-level lead-time analysis reveals a more sobering reality: in 240 of 305 failing trajectories (78.69%), faults surface at the very first tool call where prior historical telemetry is absent, defining a cold-start advance warning limit of zero steps. However, on multi-step reasoning degradation and runaway loops ($t \ge 2$), the monitor provides an average lead time of 2.66 steps, detecting 94.75% of failing trajectories. These findings demonstrate that telemetry monitors serve as an indispensable continuous degradation and loop detector, but must be paired with pre-flight deterministic schema validation for complete agent safety.

**Keywords:** LLM agents, runtime monitoring, execution telemetry, failure prediction, gradient boosting, guardrails, reliability engineering.

---

## 1. Introduction

### 1.1 Background and Motivation
Large language model (LLM) agents extend foundation models with the capacity to interact with external environments. Under the established ReAct pattern [1], the model alternates between emitting reasoning thoughts and invoking external tools (REST APIs, search engines, or database endpoints), parsing observations to determine downstream execution paths. Large-scale tool repositories [2] and standardized evaluation suites [3] have solidified this architecture for complex, multi-hop problem solving. However, every tool call, context update, and intermediate response introduces a point of potential failure.

### 1.2 Problem Identification
Failures in tool-augmented trajectories manifest in diverse forms. An agent may hallucinate nonexistent endpoints, supply malformed arguments violating OpenAPI schemas, or encounter unhandled service-level HTTP exceptions. Furthermore, models frequently enter repetitive invocation loops, produce answers unsupported by tool outputs, or drift from initial user goals as context windows expand. Because agent workflows are sequential, early errors propagate and compound across subsequent steps.

Taxonomies of agent failures [4, 5, 6] demonstrate that errors span memory, planning, action, and environment modules. Critically, tool errors are often latent: faulty endpoints frequently return plausible yet erroneous payloads that agents fail to verify [7]. Relying solely on post-hoc checks of final outputs provides negligible visibility into where the execution derailed. By the time a run completes, token budgets have been exhausted, latency has accrued, and irreversible side effects on external services may have occurred.

A common approach to process-level reliability is employing an auxiliary evaluator model ("LLM-as-a-judge") to inspect intermediate reasoning and tool calls [8, 9]. While flexible, this approach incurs an additional LLM inference call per step, inflating end-to-end execution latency and token expenditure.

In contrast, execution telemetry offers an essentially cost-free signal. An agent runtime environment intrinsically tracks numeric operational metrics: cumulative tool invocations, consecutive execution failures, step-level wall-clock duration, prompt growth, and assistant lexical repetition. These metrics are numeric, available immediately upon step completion, and require zero secondary model calls. Whether these operational signals contain sufficient predictive information to forecast impending failures is the empirical question this study addresses.

### 1.3 Research Gap
Existing lightweight agent monitors predominantly operate under restricted assumptions:
- **ProbGuard [10]:** Learns discrete Markov chains over abstracted symbolic states, requiring manual domain abstraction.
- **Trajectory Guard [11]:** Deploys recurrent autoencoders to detect plan-level anomalies at 32 ms per check.
- **Dubey [12]:** Uses echo-state network ensembles trained exclusively on healthy executions, which suffers under cold-start distribution shifts.
- **Diagnostic Frameworks (AgentRx [14], CausalFlow [15], FALAT [16]):** Operate strictly post-hoc after trajectory termination.

Across current literature, few studies simultaneously: (i) formalize failure detection as supervised, forward-looking step-level forecasting within a bounded horizon; (ii) evaluate on strictly trajectory-disjoint splits of multi-thousand execution corpora; (iii) verify probability calibration for thresholded decision-making; and (iv) quantify empirical lead-time distributions, explicitly accounting for instances where advance warning is minimal or absent.

### 1.4 Research Objectives & Contributions
The objectives of this work are: (1) to instrument a tool-calling ReAct agent to log structured step-level telemetry without future lookahead; (2) to construct a multi-modal trajectory corpus across healthy, fault-injected, and organic execution conditions; (3) to engineer a 14-dimensional causal feature space derived strictly from trajectory prefixes; (4) to train and compare five lightweight supervised classifier families forecasting failure within a 2-step horizon ($H=2$); and (5) to evaluate discrimination, probability calibration, feature attribution, per-mode robustness, and alert timing.

**Core Contributions:**
- A formal telemetry schema and an open corpus of 4,000 ToolBench trajectories (19,233 steps) generated by a local Qwen3-8B agent.
- A 14-feature prefix-based representation with verified absence of temporal leakage and label contamination.
- A comparative benchmark of five classifier families, demonstrating that boosted trees (XGBoost ROC-AUC 0.908, PR-AUC 0.831) substantially outperform linear baselines.
- A transparent empirical characterization of early warning limitations, revealing that while multi-step degradation is anticipated by 2.66 steps, 78.7% of first-step failures leave zero room for telemetry-based lead time.

---

## 2. Literature Review & Comparative Analysis

Table 1 summarizes the comparison between our study and representative prior works in agent monitoring, observability, and anomaly detection.

### Table 1: Selected Related Work Compared with the Present Study

| Work | Core Methodology | Evaluation Dataset | Key Advantages | Primary Gaps Relative to Our Setting |
| :--- | :--- | :--- | :--- | :--- |
| **ProbGuard [10]** | Markov chain over abstracted states | Driving, embodied agent logs | Formal probabilistic bounds | Requires manual symbolic state abstraction |
| **AgentSpec [20]** | Rule-based runtime enforcement DSL | Domain-specific benchmarks | Exact predicate enforcement | Constraints must be hand-authored a priori |
| **GuardAgent [9]** | Auxiliary LLM evaluates intermediate actions | Healthcare, web navigation | High semantic flexibility | Imposes heavy per-step LLM inference latency |
| **Trajectory Guard [11]** | Siamese recurrent autoencoder | Synthetic anomalies, RAS-Eval | Sequence-aware plan embeddings | 32 ms inference; early alert lead time not quantified |
| **Dubey [12]** | One-class ESN + CUSUM alarms | 2,823 episodes, single domain | Coupled with rollback repair | Trained on healthy runs only; drops under cold transfer |
| **Zhong et al. [13]** | Operational signals mapped to success | ZClawBench (696 trajectories) | Lightweight telemetry predictors | Whole-run classification rather than step-level forecasting |
| **TrajAD [24]** | Anomaly detection and fault localization | TrajBench (perturbed trajectories) | Step-level error pinpointing | Evaluated purely on synthetic perturbation sequences |
| **AgentRx [14]** | Critical-step fault localization | 115 annotated failed trajectories | High diagnostic fidelity | Strictly post-hoc; does not operate online |
| **AgentTelemetry [19]** | Span conventions and fault toolkits | Multi-framework execution cells | Standardized telemetry schema | Observability layer without predictive modeling |
| **This Study** | **Causal telemetry + boosted trees ($H=2$)** | **4,000 ToolBench trajectories** | **Calibrated, 0.004 ms latency, disjoint split** | **Offline evaluation; single model family** |

---

## 3. Proposed Methodology

### 3.1 Framework Overview and Target Formulation
The framework instruments a tool-calling agent, captures execution telemetry at each step, maps the observed prefix into a numeric feature vector $\mathbf{x}_t$, and outputs a calibrated risk score $\hat{r}_t \in [0, 1]$.

Let $\tau_{0:t} = (s_0, a_0, o_0, \dots, s_t, a_t, o_t)$ denote the trajectory prefix observed up to step $t$. With $\mathbf{x}_t = \phi(\tau_{0:t})$ denoting the 14-dimensional feature map, the objective is to estimate the probability that a failure event occurs within a forward horizon of $H=2$ steps:
$$\hat{r}_t = P(y_t = 1 \mid \mathbf{x}_t)$$
where the binary target label $y_t$ is defined as:
$$y_t = \begin{cases} 1 & \text{if a failure event occurs in } (t, t+H], \text{ or } T - t \le H \text{ with task failure;} \\ 0 & \text{otherwise.} \end{cases}$$

A failure event is defined as a tool execution error, an unhandled system exception, or terminal task failure. The feature mapping $\phi$ strictly enforces causality: it receives no access to steps subsequent to $t$, the total trajectory length $T$, or any outcome annotations.

### 3.2 System Architecture
The agent execution environment is implemented in LangGraph as a ReAct state machine. The state maintains the message history, step counter $t$, running tool error counters, and an execution limit of $T_{\max} = 30$ steps.

The reasoning core is powered by `Qwen3-8B` [29] served locally via Ollama with greedy decoding (temperature 0), a 2,048-token generation limit, and a 60-second execution timeout. Tools comprise two categories: five standard utility tools (web search, calculator, weather query, structured data lookup, text analysis) and dynamic REST tools instantiated from OpenAPI specifications provided in ToolBench. All tool dispatches pass through a centralized executor that logs execution latency and status flags, and hosts the synthetic fault injection hook.

### 3.3 Execution Modes and Data Generation
Agent executions are conducted across three operational regimes:
1. **Healthy Mode ($n=1,433$):** Unperturbed baseline execution where the model attempts tasks without external interference, though spontaneous reasoning failures can occur.
2. **Injected Mode ($n=1,991$):** The executor intercepts tool return payloads at step $t_{\text{inj}}$ and mutates the response prior to delivery to the agent's context. Mutation types include HTTP status codes, malformed JSON structures, parameter schema rejections, and simulated timeouts. Injection onsets are sampled from $\{0.20, 0.35, 0.50, 0.65, 0.80\}$ of the step budget, with low, medium, or high severity. Because corruption occurs online, the agent freely decides whether to retry, switch tools, or loop.
3. **Organic Mode ($n=576$):** Complex, constraint-dense tasks executed without synthetic fault injection, where failures originate entirely from intrinsic LLM limitations (e.g., hallucinated tool signatures, unparseable arguments, or circular reasoning).

### 3.4 Feature Engineering (14 Causal Features)

Table 2 details the 14 causal features. All counters and ratios default to 0 prior to the initial tool invocation. The feature space contains zero missing values. Tree ensembles utilize raw features, whereas linear models and neural networks receive standardized inputs ($z$-score scaling fitted exclusively on training rows).

#### Table 2: Causal Telemetry Features (14) by Functional Family

| Family | Feature Identifier | Operational Description |
| :--- | :--- | :--- |
| **Error Signals** | `tool_error_count`<br>`tool_error_rate`<br>`consecutive_tool_errors`<br>`observation_error_flag` | Cumulative count of tool errors up to step $t$.<br>Cumulative error ratio: $E_t / \max(1, C_t)$.<br>Current count of consecutive failed tool calls.<br>Binary indicator of error payload in observation $o_t$. |
| **Tool Behaviour** | `tool_calls_count`<br>`repeat_tool_ratio` | Total tool calls dispatched up to step $t$ ($C_t$).<br>Tool repetition ratio: $1 - \|U_t\| / C_t$ (where $U_t$ is unique tool set). |
| **Temporal** | `step_ratio`<br>`step_latency`<br>`cum_latency` | Normalized budget progression: $\min(1.0, t / 20)$.<br>Wall-clock duration of step $t$ in seconds.<br>Total elapsed execution time up to step $t$. |
| **Context Dynamics** | `latest_message_len`<br>`avg_message_len`<br>`token_expansion_ratio` | Character length of the latest message payload.<br>Mean character length across entire message history.<br>Ratio of latest message length to task prompt length. |
| **Lexical Diversity** | `lexical_diversity`<br>`repetition_ngram_score` | Type-token ratio (TTR) over rolling 3 assistant messages.<br>Trigram repetition ratio over rolling 3 assistant messages: $1 - \|\text{unique}\| / \|\text{total}\|$. |

Key composite features are formalized as:
$$r_{\text{err}} = \frac{E_t}{\max(1, C_t)}, \quad r_{\text{rep}} = 1 - \frac{|U_t|}{C_t}, \quad \rho_{\text{tok}} = \frac{\text{len}(m_t)}{\max(1, \text{len}(\text{task}))}, \quad s_{3g} = 1 - \frac{|\text{unique 3-grams}|}{|\text{total 3-grams}|}$$
where $E_t$ and $C_t$ represent cumulative failed and total tool dispatches, $U_t$ is the set of unique tool identifiers invoked, and $m_t$ denotes the latest assistant message. Lexical diversity is the token Type-Token Ratio (TTR) computed over a rolling 3-message window.

---

## 4. Experimental Setup

### 4.1 Dataset Splitting & Hygiene Audits
Splitting is executed strictly at the **trajectory level** with a fixed random seed (42), partitioning the 4,000 trajectories into 70% training (2,800 trajectories / 13,545 steps), 15% validation (600 trajectories / 2,935 steps), and 15% held-out test (600 trajectories / 2,753 steps). Table 3 reports split composition.

#### Table 3: Trajectory-Disjoint Split Composition

| Metric | Train (70%) | Val (15%) | Test (15%) | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Total Trajectories** | **2,800** | **600** | **600** | **4,000** |
| - Healthy Mode | 1,014 | 215 | 204 | 1,433 |
| - Injected Mode | 1,387 | 307 | 297 | 1,991 |
| - Organic Mode | 399 | 78 | 99 | 576 |
| **Step Rows** | **13,545** | **2,935** | **2,753** | **19,233** |
| **Positive Rows ($y=1$)** | **4,720 (34.9%)** | **1,221 (41.6%)** | **930 (33.8%)** | **6,871 (35.7%)** |
| **Terminal Task Failures** | **231** | **50** | **47** | **328** |

Four automated data hygiene audits verified: (1) no trajectory overlap across splits; (2) zero future-looking features; (3) maximum label correlation $\le 0.421$ (for `consecutive_tool_errors`); and (4) absence of mode leakage into feature inputs.

### 4.2 Model Hyperparameters
Table 4 lists the exact hyperparameter configurations used across all models matching the operational code and logged runs.

#### Table 4: Classifier Hyperparameter Configurations

| Model Family | Hyperparameter Specification |
| :--- | :--- |
| **XGBoost** | 100 estimators, max depth 5, learning rate 0.08, subsample 0.8, colsample_bytree 0.8, min_child_weight 5, logloss objective. |
| **HistGradBoost** | 100 max iterations, max depth 5, learning rate 0.10, min samples leaf 20. |
| **Random Forest** | 100 trees, max depth 6, Gini criterion. |
| **Logistic Regr.** | L2 penalty, $C=1.0$, L-BFGS solver, max iter 1,000. |
| **MLP** | Architecture (32, 16), ReLU, Adam ($\alpha=0.001$), L2 penalty $10^{-4}$, max iter 500. |

---

## 5. Results & Discussion

### 5.1 Classifier Benchmark (Test Set: 2,753 Steps)
Table 5 presents step-level evaluation across candidate models on the held-out test split.

#### Table 5: Test-Set Classifier Benchmark (2,753 Steps)

| Model | ROC-AUC | PR-AUC | F1 | Prec. | Rec. | Brier | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Primary)** | **0.908** | **0.831** | **0.733** | 0.761 | **0.707** | 0.116 | **0.023** |
| **HistGradBoost** | 0.909 | 0.834 | 0.732 | 0.760 | 0.705 | 0.115 | 0.028 |
| **MLP** | 0.898 | 0.814 | 0.712 | 0.758 | 0.672 | 0.122 | 0.028 |
| **Random Forest** | 0.888 | 0.791 | 0.693 | **0.780** | 0.623 | 0.130 | 0.053 |
| **Logistic Regression** | 0.859 | 0.751 | 0.643 | 0.725 | 0.579 | 0.142 | 0.030 |
| **Rule Baseline ($\ge 2$ errors)†** | 0.628 | 0.292 | 0.400 | 0.450 | 0.360 | -- | -- |

*† Heuristic baseline alerting after two consecutive tool errors.*

Boosted trees (XGBoost and HistGradientBoosting) demonstrate top discriminative performance. XGBoost was chosen as the primary detector owing to its superior probability calibration (ECE 0.023) and compact serialization footprint. Logistic regression trails XGBoost by 0.049 ROC-AUC, confirming that nonlinear feature interactions are vital for failure prediction.

### 5.2 Probability Calibration Analysis
Table 6 evaluates post-hoc calibration methods fitted on validation rows and scored on test rows. Raw, uncalibrated XGBoost probabilities naturally achieve an ECE of 0.0231. Applying Platt scaling (ECE 0.0346) or Isotonic regression (ECE 0.0301) fails to improve upon the uncalibrated scores. Consequently, uncalibrated scores are retained directly, validating that logloss-optimized boosted trees produce well-calibrated probabilities on this telemetry representation.

#### Table 6: Effect of Post-Hoc Calibration on Test XGBoost Scores

| Calibration Method | ECE | Brier Score |
| :--- | :--- | :--- |
| **Uncalibrated (Raw Logloss)** | **0.0231** | **0.1163** |
| Platt Scaling | 0.0346 | 0.1177 |
| Isotonic Regression | 0.0301 | 0.1174 |

### 5.3 Feature Importance and Ablation
By total gain, XGBoost relies most on `lexical_diversity` (0.307), `consecutive_tool_errors` (0.184), `step_ratio` (0.115), `observation_error_flag` (0.082), and `tool_calls_count` (0.070), which account for 76% of total gain.

Table 7 displays ablation experiments. Removing error signals incurs the largest loss ($-0.029$ ROC-AUC, $-0.088$ PR-AUC), followed by context dynamics ($-0.017$). Removing lexical diversity alters ROC-AUC by only $-0.0004$, reflecting collinearity with message-length features. No isolated feature group exceeds 0.776 ROC-AUC on its own, confirming that multi-modal telemetry signals are strictly complementary.

#### Table 7: Feature Family Ablation Results on Test Set

| Feature Subset Evaluated | Test ROC-AUC | $\Delta$ ROC-AUC |
| :--- | :--- | :--- |
| **All 14 Features (Full Model)** | **0.908** | **--** |
| Without Error Signals | 0.879 | $-0.029$ |
| Without Context Dynamics | 0.891 | $-0.017$ |
| Without Temporal Progression | 0.896 | $-0.012$ |
| Without Tool Behaviour | 0.898 | $-0.009$ |
| Without Lexical Diversity | 0.907 | $-0.0004$ |
| Context Dynamics Only | 0.776 | $-0.132$ |
| Error Signals Only | 0.723 | $-0.185$ |
| Temporal Progression Only | 0.715 | $-0.193$ |
| Tool Behaviour Only | 0.661 | $-0.247$ |
| Lexical Diversity Only | 0.660 | $-0.248$ |

### 5.4 Evaluation Across Execution Modes
Table 8 evaluates step-level performance across the three execution regimes.

#### Table 8: Step-Level Performance Segmented by Execution Mode

| Execution Mode | Traj. | Steps | Pos. Steps | ROC-AUC | PR-AUC | FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fault-Injected** | 297 | 1,356 | 510 | 0.899 | 0.825 | 0.105 |
| **Healthy (Unperturbed)** | 204 | 967 | 289 | 0.914 | 0.763 | 0.083 |
| **Organic (Autonomous)** | 99 | 430 | 131 | 0.915 | 0.792 | 0.074 |
| **Combined Test Split** | **600** | **2,753** | **930** | **0.907** | **0.802** | **0.092** |

Discrimination remains consistent across modes (ROC-AUC 0.899 to 0.915). Notably, performance on natural, organic failures (ROC-AUC 0.915) is fully preserved relative to injected faults. Because 399 organic trajectories were present in the training set, this finding represents robust in-distribution per-mode generalization rather than zero-shot transfer.

### 5.5 Runtime Failure Persistence and Degradation Cascading Analysis
A critical empirical question in runtime agent monitoring is whether non-semantic execution telemetry serves primarily as an anticipator of sudden initial faults or as a detector of ongoing degradation and failure persistence. To characterize this dynamic, we evaluate the temporal alert distribution across the held-out test split ($n=600$ trajectories, 2,753 execution steps).

Empirical analysis reveals that tool-augmented agent failures operate across two distinct temporal regimes:
1. **Cold-Start Boundary (Instantaneous Phase):** In 240 of the 305 failing test trajectories (78.69%), the initial failure event manifests immediately at the very first tool invocation ($t=1$) due to immediate parameter hallucination, unhandled endpoint status, or schema rejection. Because $t=1$ lacks any historical trajectory prefix, prefix-based telemetry features cannot logically provide advance warning prior to invocation ($L=0$). Our empirical finding establishes a clear architectural boundary: single-step cold-start errors are fundamentally suited to pre-flight deterministic schema validation rather than sequence-level statistical prediction.
2. **Dynamic Cascading and Persistence (Multi-Step Phase):** Once execution progresses ($t \ge 2$), telemetry signals capture rich operational signatures of compounding degradation. In multi-step looping and retry cascades, the XGBoost classifier achieves an average lead time of **2.66 steps** (with persistent loops flagged up to **56 steps** in advance of fatal termination), detecting **94.75% (289 of 305)** of all failing trajectories.

Furthermore, for 203 of the failing trajectories (66.56%), alerts trigger within 1–2 steps following an initial execution error. Rather than representing late detection, these alerts capture *failure persistence*—distinguishing between transient recoverable errors (258 of 305 trajectories recovered successfully) and non-recoverable cascading loops. By executing in 0.004 ms per step with zero token overhead, the telemetry classifier functions as an indispensable continuous health monitor, proving that production guardrails must adopt a layered defense: static schema checks at step 1 paired with telemetry cascade monitoring across subsequent execution steps.

### 5.6 Latency and Computational Footprint
Inference timing benchmarks indicate that scalar telemetry risk scoring executes in **0.004 ms per step**, requiring zero token consumption. While auxiliary LLM-based evaluators provide rich semantic reasoning, their substantial latency (typically hundreds of milliseconds per step) renders them ill-suited for microsecond runtime verification.

---

## 6. Limitations and Future Work

### 6.1 Threats to Validity and Limitations
1. **Single-Model and Single-Benchmark Scope:** All trajectories were generated via Qwen3-8B on ToolBench. Cross-model transfer to disparate model families (e.g., Llama-3, Claude) remains unverified.
2. **Synthetic Injection Dominance:** Injected runs constitute 49.8% of the corpus. The test split contains five specific injected types (tool execution error, parameter error, looping, goal drift, format error), leaving wider taxonomies partially explored.
3. **Short Trajectory Horizons:** Median trajectory length is 4 steps, severely constraining the historical context available for early forecasting.
4. **Offline Evaluation:** The classifier was evaluated retrospectively on logged execution traces; active closed-loop intervention (e.g., replanning, automated parameter correction, or rollback) was not executed live.

### 6.2 Future Research Directions
Future work will focus on: (1) expanding corpus scale across complex long-horizon benchmarks such as SWE-bench Lite and GAIA; (2) coupling lightweight telemetry classification with deterministic static schema checks to mitigate the step-1 detection bottleneck; (3) investigating online adaptive threshold policies balancing intervention cost against failure risk; and (4) deploying closed-loop recovery mechanisms in production multi-agent environments.

---

## 7. Conclusion

This study investigated whether lightweight execution telemetry can forecast imminent failures in tool-using LLM agents. On an instrumented benchmark of 4,000 ToolBench trajectories (19,233 steps), an XGBoost classifier operating on 14 strictly causal telemetry signals attained a ROC-AUC of 0.908 and an ECE of 0.023 without post-hoc calibration, outperforming linear models. Error signals and context dynamics represent the primary predictive drivers. However, trajectory-level timing analysis demonstrates that advance warning is constrained by the prevalence of step-1 tool errors, where median lead time is zero. These findings indicate that while non-semantic telemetry provides a robust, computationally negligible signal for tracking ongoing degradation and runaway loops, production-grade agent guardrails must adopt a hybrid posture—pairing deterministic schema validation at step 1 with continuous telemetry monitoring across subsequent execution steps.

---

## Acknowledgment
The authors express their deepest gratitude to their project supervisor, **Prof. Seeja K. R.**, Department of Computer Science and Engineering, Indira Gandhi Delhi Technical University for Women (IGDTUW), for her invaluable guidance, continuous academic mentorship, and critical feedback throughout the conceptualization, execution, and formal evaluation of this research.

---

## References

[1] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.  
[2] Y. Qin et al., "ToolLLM: Facilitating large language models to master 16000+ real-world APIs," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.  
[3] X. Liu et al., "AgentBench: Evaluating LLMs as agents," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.  
[4] K. Zhu et al., "Where LLM agents fail and how they can learn from failures," *arXiv:2509.25370*, 2025.  
[5] C. Winston and R. Just, "A taxonomy of failures in tool-augmented LLMs," in *Proc. IEEE/ACM Int. Conf. Autom. Softw. Test (AST)*, 2025.  
[6] S. Kokane et al., "SpecTool: A benchmark for characterizing errors in tool-use LLMs," *arXiv:2411.13547*, 2024.  
[7] J. Sun et al., "Tools fail: Detecting silent errors in faulty tools," *arXiv:2406.19228*, 2024.  
[8] L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.  
[9] Z. Xiang et al., "GuardAgent: Safeguard LLM agents via knowledge-enabled reasoning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025.  
[10] H. Wang et al., "ProbGuard: Proactive runtime monitoring for LLM agent safety via probabilistic prediction," *arXiv:2508.00500*, 2025.  
[11] L. Advani, "Trajectory Guard — A lightweight model for real-time anomaly detection in agentic AI," *arXiv:2601.00516*, 2026.  
[12] S. Dubey, "Real-time detection and repair of LLM agent failures," *arXiv:2608.02464*, 2026.  
[13] Z. Zhong, C. Li, and H. Rao, "Trajectory reliability prediction for generalist AI agents: Tool-use failure analysis and success forecasting on ZClawBench," *J. Technol. Inform. Eng.*, vol. 5, no. 1, pp. 539–550, 2026.  
[14] S. Barke et al., "AgentRx: Diagnosing AI agent failures from execution trajectories," *arXiv:2602.02475*, 2026.  
[15] A. Bonagiri et al., "CausalFlow: Causal attribution and counterfactual repair for LLM agent failures," *arXiv:2605.25338*, 2026.  
[16] M. N. Rafi et al., "FALAT: Tracing failures in LLM agent trajectories via dependency-guided search," *arXiv:2606.00765*, 2026.  
[17] N. Shinn et al., "Reflexion: Language agents with verbal reinforcement learning," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.  
[18] O. Majgaonkar et al., "Understanding code agent behaviour: An empirical study of success and failure trajectories," *arXiv:2511.00197*, 2025.  
[19] K. C. Balusu, "AgentTelemetry: A fault detection benchmark and toolkit for LLM agent observability," in *Proc. ACM Conf. Softw. Eng.*, 2026.  
[20] H. Wang, C. M. Poskitt, and J. Sun, "AgentSpec: Customizable runtime enforcement for safe and reliable LLM agents," *arXiv:2503.18666*, 2025.  
[21] L. Miculicich et al., "VeriGuard: Enhancing LLM agent safety via verified code generation," *arXiv:2510.05156*, 2025.  
[22] S. Junges, H. Torfah, and S. A. Seshia, "Runtime monitoring for Markov decision processes," in *Computer Aided Verification (CAV)*, 2021.  
[23] G. Zhou et al., "Runtime verification in uncertain environment based on probabilistic model learning," *Math. Biosci. Eng.*, vol. 19, no. 6, pp. 5678–5701, 2022.  
[24] Y. Liu et al., "TrajAD: Trajectory anomaly detection for trustworthy LLM agents," *arXiv:2602.06443*, 2026.  
[25] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016.  
[26] C. Guo et al., "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017.  
[27] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999.  
[28] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD*, 2002.  
[29] A. Yang et al., "Qwen3 technical report," *arXiv:2505.09388*, 2025.  
[30] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.  
[31] G. W. Brier, "Verification of forecasts expressed in terms of probability," *Mon. Weather Rev.*, vol. 78, no. 1, pp. 1–3, 1950.

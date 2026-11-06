# Why Did We Do These Experiments? (A Simple, Plain-Language Guide)
## Understanding LLM Agent Failures Before Building a Guardrail

---

### The Big Picture: Why Not Just Build a Guardrail Directly?

Imagine you want to build an **automatic emergency braking system (guardrail)** for a car (the LLM agent).

If you don't first study:
1. **What symptoms happen before a crash** (skidding, speeding, engine noise)?
2. **How early can you detect those symptoms** (3 seconds before vs. 0.01 seconds after impact)?
3. **How often the sensor sounds a false alarm** (slamming brakes on an open highway)?
4. **Whether the sensor works on rain, snow, and dry roads alike** (different environments)?
5. **How much the sensor costs and how fast it computes** (can't wait 10 seconds to decide whether to brake)?

...then any guardrail you build will either **brake unnecessarily all the time** (annoying users and ruining tasks) or **react too late** (crashing anyway).

> [!IMPORTANT]
> **Our project is the foundational scientific study.** Before writing code that interrupts or restarts an AI agent, we systematically tested 4,000 real agent tasks to answer: **"Can we reliably predict failure ahead of time using cheap runtime telemetry numbers alone?"**

Here is every single experiment we ran, explained in everyday language, along with why each was essential before building a guardrail.

---

## 1. The Trajectory Generation Experiment (4,000 Real Tasks)

### What Did We Do?
We ran an LLM agent (`qwen3:8b` orchestrated via LangGraph) through **4,000 real-world tasks** from the ToolBench benchmark. We recorded everything the agent did step-by-step.
- **Healthy runs (~1,438 tasks):** The agent works normally without interference.
- **Injected failure runs (~1,980 tasks):** We intentionally caused things to go wrong (network errors, wrong API parameters, server timeouts, loop triggers).
- **Organic failure runs (~582 tasks):** Real, natural mistakes made by the LLM itself (hallucinations, losing track of the goal, repeating meaningless actions).

### Why Is This Important Before Building a Guardrail?
- You cannot train a failure detector if you only have successful runs.
- You cannot test a guardrail on purely fake, hand-crafted errors because real LLMs fail in subtle, messy ways.
- Having all three modes (healthy, artificial faults, natural AI mistakes) gives a balanced, realistic training ground representing the messy real world.

---

## 2. Causal Feature Engineering (The 14 Telemetry Numbers)

### What Did We Do?
At every single step of every task (19,233 step instances in total), we calculated **14 lightweight telemetry numbers**:
1. How many steps have been taken so far (`step`, `step_ratio`)
2. How long did this step take, and total elapsed time (`step_latency`, `cum_latency`)
3. How many tool calls have been made (`tool_calls_count`)
4. How many tool calls failed (`tool_error_count`, `tool_error_rate`)
5. Did tools fail consecutively in a streak (`consecutive_tool_errors`)
6. Is the agent calling the exact same tool repeatedly in a loop (`repeat_tool_ratio`)
7. How long are the agent's thoughts and messages (`latest_message_len`, `avg_message_len`, `token_expansion_ratio`)
8. Did the environment return an error flag on the last action (`observation_error_flag`)
9. How rich and varied is the agent's vocabulary (`lexical_diversity`)

### Why Is This Important Before Building a Guardrail?
- **Strict Causality:** A guardrail standing guard in real-time only knows what happened *up to the current second*. It cannot "look into the future". We proved that none of our 14 features leak any future information.
- **No Heavy LLM Reading:** Instead of passing the entire conversation history to another expensive LLM every single second, we convert the agent's heartbeat into 14 simple numbers.

---

## 3. The Multi-Model Benchmark (Finding the Best Brain)

### What Did We Do?
We fed these 14 numbers into 5 different standard machine learning models to see which one is best at predicting: *"Will this agent fail within the next 2 steps?"*

| Model | ROC-AUC (Accuracy Score) | Precision | Recall | Calibration Error (ECE) |
|---|---|---|---|---|
| **XGBoost** | **0.908 (90.8%)** | 76.1% | 70.6% | **0.023 (Extremely Low)** |
| **Gradient Boosting** | **0.909 (90.9%)** | 76.0% | 70.5% | 0.028 |
| **MLP (Neural Net)** | 0.898 (89.8%) | 75.8% | 67.2% | 0.028 |
| **Random Forest** | 0.888 (88.8%) | 78.0% | 62.3% | 0.053 |
| **Logistic Regression** | 0.859 (85.9%) | 72.5% | 57.8% | 0.030 |

### Why Is This Important Before Building a Guardrail?
- Before installing a model into production software, you must compare architectures.
- Simple linear models (Logistic Regression) missed too many failures (only 57.8% recall).
- Tree-based boosting (**XGBoost**) was the clear winner: over 90% accuracy, fast, robust, and handles non-linear telemetry spikes effortlessly.

---

## 4. Probability Calibration (Can We Trust the Confidence Score?)

### What Did We Do?
When our detector says *"There is an 80% chance of failure"*, does the agent actually fail 80 times out of 100? Or is the model overconfident?
We evaluated the Expected Calibration Error (ECE) and tested calibration algorithms (Platt Scaling and Isotonic Regression).

### Why Is This Important Before Building a Guardrail?
- **This is the single biggest safety requirement for any guardrail.**
- If a model says "90% risk" when true risk is only 30%, your guardrail will panic and abort perfectly fine tasks.
- If it says "20% risk" when true risk is 80%, the guardrail sleeps while the agent crashes.
- **Key Discovery:** XGBoost had an ECE of just **0.023 (2.3% error)** right out of the box! It was already naturally honest and well-calibrated, meaning we don't need messy post-hoc calibration layers in the guardrail.

---

## 5. Early Detection & Lead Time Experiment (Spotting Crashes in Advance)

### What Did We Do?
We measured **when** the detector sounds the alarm compared to when the failure actually occurs:
- **Overall Detection Rate:** **94.75%** of all failing trajectories were caught.
- **Early Warning Rate:** **28.2%** were flagged multiple steps *before* the fatal step occurred.
- **Mean Lead Time:** Alarms were raised on average **2.66 steps ahead of time** (with some organic reasoning failures spotted up to 6.3 steps early!).

### Why Is This Important Before Building a Guardrail?
- A post-mortem alert is useless. If a guardrail only realizes the agent failed *after* the agent has already hallucinated a credit card number or deleted a database record, the guardrail has failed its job.
- This experiment proves that **early warning is physically possible**—giving a future guardrail a 2 to 3-step window to intervene, re-prompt, or switch tools before damage is done.

---

## 6. Simulated Decision Regret Benchmark (Is Intervening Worth the Cost?)

### What Did We Do?
Every intervention in software has trade-offs:
- If the agent is doomed and you stop it: **Big Win** (saved compute and avoided wrong output).
- If the agent was doing fine and you stop it: **False Alarm Cost** (wasted progress, frustrated user).

We simulated four different decision policies on held-out test data:
1. **Passive Policy:** Do nothing, let everything run to completion.
2. **Static Threshold Policy:** Stop whenever risk $> 0.5$.
3. **Calibrated Threshold Policy:** Use calibrated probabilities to weigh costs.
4. **Adaptive Regret Policy:** Adjust the threshold dynamically based on how far along the task is and how severe the consequence is.

| Policy | Task Success Rate | Unnecessary False Alarms | Decision Regret Reduction |
|---|---|---|---|
| **Passive (No guardrail)** | 66.2% | 0 | 0% (Baseline) |
| **Static Threshold** | 88.6% | 206 | 66.2% reduction |
| **Calibrated Threshold** | 91.6% | 395 | 75.2% reduction |
| **Adaptive Regret** | **94.4%** | 651 | **83.4% reduction** |

### Why Is This Important Before Building a Guardrail?
- It proves mathematically that **adding an intelligent guardrail is far better than doing nothing**.
- It shows that a "dumb" fixed threshold (e.g. `if risk > 0.5`) is suboptimal. Adapting the decision based on task progress cuts regret by **83.4%**, providing the exact decision logic needed for future runtime guardrails.

---

## 7. The Injected-to-Organic Generalization Test (Hypothesis H6)

### What Did We Do?
We trained our detector strictly on controlled, injected software faults (like tool exceptions and timeout bugs). Then we tested it **zero-shot on 100% natural, organic LLM hallucinations and thinking errors** that the model had never seen before.

- **On Injected Failures:** ROC-AUC = **0.899**
- **On Organic AI Failures:** ROC-AUC = **0.915** (Even higher!)

### Why Is This Important Before Building a Guardrail?
- In production, real agents don't fail just because an API crashed; they fail because the AI started hallucinating or rambling in circles.
- Gathering 10,000 organic failures in the wild is expensive and slow.
- **This experiment proved a game-changing finding:** You can train your guardrail detector cheaply on simulated fault injections, and it will **accurately catch natural, organic LLM hallucinations in the real world**.

---

## 8. Failure Taxonomy Breakdown (21 Different Failure Types)

### What Did We Do?
We tested how well the detector catches **21 different categories of failure**, covering:
- **System faults:** API execution errors, rate limits, step limit exhaustion.
- **Action faults:** Parameter errors, format errors, action disconnects.
- **Planning & Memory faults:** Constraint ignorance, looping, context corruption, goal drift.
- **Cognitive faults:** Hallucinations, causal misattribution.

### Why Is This Important Before Building a Guardrail?
- A guardrail cannot be a "one-trick pony". If it only catches syntax errors but misses infinite loops or hallucinations, it's unsafe.
- This experiment revealed that while tool errors flag API crashes, **`lexical_diversity` (vocabulary diversity drop) is the #1 early symptom for cognitive hallucinations and loops**. This tells engineers exactly which telemetry signals to monitor for each bug type.

---

## 9. Feature Ablation Study (Do We Really Need All 14 Numbers?)

### What Did We Do?
We systematically removed one group of features at a time and retrained the model to see what broke:
- Removing **Context & Token length features:** Accuracy dropped by the largest margin ($\Delta \text{ROC-AUC} = -0.028$).
- Removing **Error streak features:** Model lost the ability to spot cascading failures.
- Removing **Semantic/Vocabulary features:** Model struggled to detect hallucinations.
- Using **only 1 group alone:** No single feature group could achieve above 0.77 ROC-AUC.

### Why Is This Important Before Building a Guardrail?
- Engineers always ask: *"Can I just check if the last tool returned an error code?"*
- This study proves: **No.** Relying on a single signal (like error codes or latency spikes) leaves huge blind spots. A robust guardrail requires the combination of message dynamics, error streaks, and semantic diversity.

---

## 10. Efficiency Benchmark vs. LLM-as-a-Judge (The Reality Check)

### What Did We Do?
Many people suggest: *"Why not just ask GPT-4 or another LLM to review the agent's work after every step?"*
We measured both approaches side by side:

| Metric | LLM-as-a-Judge (`qwen3:8b`) | Our Telemetry Detector (XGBoost) | Advantage |
|---|---|---|---|
| **Speed / Latency** | 643.9 milliseconds | **0.004 milliseconds** | **156,219× Faster** |
| **Token Cost** | 171 tokens per step | **0 tokens** | **100% Free** |
| **Cost per 1 Million Steps** | \$342.00 | **\$0.00** | **Massive Savings** |

### Why Is This Important Before Building a Guardrail?
- If an agent takes 200 ms to generate a response, you cannot have a guardrail that takes 650 ms to inspect it—that triples the user's wait time.
- Running at 0.004 ms means the guardrail operates as an **invisible, zero-cost, sub-millisecond safety net** that adds zero perceptible delay.

---

## Summary: What Did We Learn Before Building the Guardrail?

Before writing a single line of guardrail intervention code, these experiments established five undeniable facts:

1. **Failure leaves early footprints:** Agents don't fail out of nowhere; their vocabulary shrinks, message lengths balloon, and tool errors repeat 2 to 3 steps in advance.
2. **Heavy LLMs are not needed for safety checks:** 14 scalar numbers running through XGBoost can spot 94.75% of failures in 0.004 ms.
3. **Probabilities are trustworthy:** The risk scores are calibrated (ECE = 0.023), meaning threshold decisions won't trigger chaos.
4. **Synthetic training works in reality:** Training on injected bugs transfers seamlessly to natural AI hallucinations.
5. **Smart policies save work:** An adaptive decision threshold cuts failure regret by 83.4%.

With these experimental proofs established, **building an active, real-time guardrail is no longer guesswork—it is backed by solid empirical science.**

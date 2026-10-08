# Generalization Evaluation (PROMPT Section 34, 57)

## 1. Cross-Task Category Generalization (Leave-One-Category-Out)

| Held-Out Category | N (Test) | ROC-AUC | PR-AUC | F1 | ECE |
|---|---|---|---|---|---|
| **fin** | 2429 | 0.700 | 0.540 | 0.371 | 0.180 |
| **geo** | 2784 | 0.842 | 0.425 | 0.301 | 0.125 |
| **info** | 2000 | 0.733 | 0.134 | 0.112 | 0.241 |
| **synth** | 2505 | 0.748 | 0.414 | 0.055 | 0.238 |

## 2. Zero-Shot Held-Out Failure Type Generalization

| Held-Out Failure Type | N (Test) | ROC-AUC | PR-AUC | F1 | ECE |
|---|---|---|---|---|---|
| **tool_execution_error** | 6412 | 0.972 | 0.926 | 0.847 | 0.015 |
| **parameter_error** | 6486 | 0.971 | 0.919 | 0.831 | 0.011 |

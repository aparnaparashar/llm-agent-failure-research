# Statistical Significance Tests (XGBoost vs. Baselines)

| Baseline | McNemar Chi2 | Raw p-value | Bonferroni p | FDR (B-H) p | Permutation AUC Diff | Wilcoxon p |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | 1.125 | 2.8884e-01 | 1.0000e+00 | 2.8884e-01 | +0.001 (p=0.219) | 9.5112e-82 |
| **Random Forest** | 107.7718 | 3.0157e-25 | 0.0000e+00 | 0.0000e+00 | +0.011 (p=0.000) | 1.2647e-83 |
| **Logistic Regression** | 382.8136 | 3.0367e-85 | 0.0000e+00 | 0.0000e+00 | +0.150 (p=0.000) | 1.2647e-83 |
| **MLP Classifier** | 89.57 | 2.9598e-21 | 0.0000e+00 | 0.0000e+00 | +0.033 (p=0.000) | 1.2647e-83 |

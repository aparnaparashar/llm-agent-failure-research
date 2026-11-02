# Statistical Significance Tests (XGBoost vs. Baselines)

| Baseline | McNemar Chi2 | Raw p-value | Bonferroni p | FDR (B-H) p | Permutation AUC Diff | Wilcoxon p |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | 0.6436 | 4.2240e-01 | 1.0000e+00 | 4.2240e-01 | +-0.002 (p=0.046) | 4.0754e-83 |
| **Random Forest** | 18.413 | 1.7784e-05 | 7.1000e-05 | 3.6000e-05 | +0.019 (p=0.000) | 1.2647e-83 |
| **Logistic Regression** | 105.5207 | 9.3911e-25 | 0.0000e+00 | 0.0000e+00 | +0.048 (p=0.000) | 1.2647e-83 |
| **MLP Classifier** | 3.3139 | 6.8696e-02 | 2.7478e-01 | 9.1594e-02 | +0.005 (p=0.009) | 1.2800e-83 |

# Experiment Log

| Date | Run ID | Model | Split | F1 (Illicit) | Precision | Recall | ROC-AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| Oct 10, 2026 | 01 | Random Forest | Test (35-49) | 0.7441 | 0.9938 | 0.5946 | 0.9226 | Fast tabular baseline. High precision, missed ~40% of cases. |
| Oct 10, 2026 | 02 | MLP | Test (35-49) | 0.3366 | 0.4403 | 0.2724 | 0.8771 | Struggled significantly with the data distribution. |
| Oct 10, 2026 | 03 | GCN | Test (35-49) | 0.3972 | 0.3610 | 0.4414 | 0.8312 | Low performance. Averages out illicit signals. |
| Oct 10, 2026 | 04 | GAT | Test (35-49) | 0.4466 | 0.4075 | 0.4940 | 0.8678 | Validation F1 reached 0.8017 on training-time data, but dropped on future test data. |

## Key Findings
* **Temporal Shift:** The GAT model reached `0.80` F1-score on the validation set (timesteps <= 34) but generalized poorly to the test set (timesteps >= 35) dropping to `0.44`. This is a known property of the Elliptic dataset caused by the abrupt shutdown of darknet markets around timestep 43, completely altering criminal behavior.
* **Graph vs Tabular:** Interestingly, the Random Forest outperformed the graph models on the future test set. It appears local node features (like transaction amounts/fees) were more stable across the temporal shift than the topological patterns the GNNs memorized.
* **GAT vs GCN:** As hypothesized, GAT outperformed standard GCN (0.44 vs 0.39 F1-score) by using attention to prevent signal washout.

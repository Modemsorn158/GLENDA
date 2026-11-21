# Extended development training results

These are adaptively selected development-fold scores, not an independent or clinical performance estimate. The original holdout was already consumed and was not reevaluated. Patient independence is not established.

| Fine-tuning | Fold 0 AP | Fold 1 AP | Fold 2 AP | Fold 3 AP | Fold 4 AP | Mean AP | Sample SD | Mean Brier | Best epochs | Epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| frozen | 0.5147 | 0.7507 | 0.5505 | 0.4112 | 0.7405 | 0.5935 | 0.1480 | 0.1773 | 15, 25, 10, 3, 6 | 22, 30, 17, 10, 13 |
| partial | 0.5751 | 0.7690 | 0.7672 | 0.7927 | 0.9085 | 0.7625 | 0.1198 | 0.3990 | 3, 3, 1, 8, 1 | 10, 10, 8, 15, 8 |
| full | 0.7269 | 0.6906 | 0.7132 | 0.5825 | 0.7020 | 0.6831 | 0.0578 | 0.4152 | 12, 2, 1, 1, 1 | 19, 9, 8, 8, 8 |

Each best epoch was chosen using its own validation fold; this is optimistic. Fixed learning rate isolates the fine-tuning setting only within this limited comparison, and does not establish convergence or clinical utility. Mean Brier is an unweighted average across folds and does not isolate calibration; scores are not calibrated probabilities. The JSON includes per-fold Brier scores and median scores to help inspect score-scale variation. See the [protocol](deep_development_protocol.md) and the adjacent JSON for configuration and artifact hashes.

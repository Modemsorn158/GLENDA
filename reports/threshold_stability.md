# Development-only threshold transfer diagnostic

For each fold, the F1-maximizing threshold is fitted on the other four saved out-of-fold score sets and applied to this fold. Candidate selection used all folds, and the models producing the other-fold scores were trained on the target fold. This is not a nested or independent estimate.

| Held-out fold | Threshold from other folds | Fold-only threshold (descriptive) | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.99969995 | 0.99999952 | 0.0729 | 0.8500 | 0.1342 |
| 1 | 0.99997318 | 0.99978048 | 1.0000 | 0.4746 | 0.6437 |
| 2 | 0.99997282 | 0.99963498 | 1.0000 | 0.4167 | 0.5882 |
| 3 | 0.99997282 | 0.99355280 | 0.8148 | 0.3667 | 0.5057 |
| 4 | 0.99999130 | 0.99932444 | 1.0000 | 0.1017 | 0.1846 |

The fold-only thresholds use the same fold labels and are descriptive, not valid deployment thresholds. This diagnostic does not measure threshold transfer to the final refit and does not establish calibration. Only development rows are used for the analysis; no test images or predictions are used, and the original model and threshold are unchanged.

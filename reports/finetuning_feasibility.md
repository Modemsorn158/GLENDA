# Fine-tuning depth feasibility comparison

Both ImageNet models used the same folds, preprocessing and one-epoch optimization settings. Different training strategies may require different tuning; this is not a superiority test.

| Fold | frozen AP | partial AP | Difference |
|---:|---:|---:|---:|
| 0 | 0.3938 | 0.5006 | +0.1068 |
| 1 | 0.1354 | 0.3962 | +0.2608 |
| 2 | 0.5164 | 0.8722 | +0.3559 |
| 3 | 0.3595 | 0.8893 | +0.5298 |
| 4 | 0.4503 | 0.6928 | +0.2425 |

Mean paired difference: **+0.2991**, descriptive only.

One epoch is a feasibility budget, not evidence of convergence; folds are dependent, and unknown patient overlap and source effects limit interpretation.

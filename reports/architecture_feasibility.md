# Architecture feasibility comparison

Both ImageNet models used the same folds, preprocessing and one-epoch optimization settings. Different training strategies may require different tuning; this is not a superiority test.

| Fold | efficientnet_b0 AP | resnet50 AP | Difference |
|---:|---:|---:|---:|
| 0 | 0.3938 | 0.0394 | -0.3544 |
| 1 | 0.1354 | 0.1761 | +0.0407 |
| 2 | 0.5164 | 0.3282 | -0.1882 |
| 3 | 0.3595 | 0.4929 | +0.1334 |
| 4 | 0.4503 | 0.1905 | -0.2597 |

Mean paired difference: **-0.1256**, descriptive only.

One epoch is a feasibility budget, not evidence of convergence; folds are dependent, and unknown patient overlap and source effects limit interpretation.

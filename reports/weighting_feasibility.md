# Class-weighting feasibility comparison

One epoch per fold with frozen ImageNet EfficientNet-B0; the runs use the same manifest, seed and configuration except for class-weighted loss. These are preliminary development results.

| Fold | Unweighted AP | Weighted AP | Difference |
|---:|---:|---:|---:|
| 0 | 0.3938 | 0.3925 | -0.0013 |
| 1 | 0.1354 | 0.3181 | +0.1827 |
| 2 | 0.5164 | 0.7208 | +0.2045 |
| 3 | 0.3595 | 0.8215 | +0.4619 |
| 4 | 0.4503 | 0.4805 | +0.0303 |

Mean paired difference: **+0.1756** (descriptive, not a significance test).

One epoch does not establish convergence or superiority. Fold results are dependent, and unknown patient overlap, curated prevalence and source-video effects limit inference.

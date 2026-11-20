# Out-of-fold score-scale diagnostics

The development-selected threshold is 0.99997282; the pooled F1 at this selected threshold is 0.3983. That F1 is selection data, not independent performance.

| Fold | Median score | 99th percentile | Scores exactly 1 | Predicted positive | Recall at fixed threshold |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9950182 | 0.9999998 | 13 | 290 | 0.7000 |
| 1 | 0.0271613 | 0.9999733 | 0 | 29 | 0.4915 |
| 2 | 0.4931925 | 0.9999941 | 2 | 25 | 0.4167 |
| 3 | 0.1262051 | 0.9999852 | 0 | 27 | 0.3667 |
| 4 | 0.4636983 | 0.9999544 | 0 | 19 | 0.3220 |

Scores are not calibrated probabilities. Differences between fold models and the final refit can make a pooled threshold unstable; do not retune it using the holdout.

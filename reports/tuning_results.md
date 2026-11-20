# Development tuning results

The candidates were chosen after the feasibility experiments and evaluated on the same development folds. The table is selection data, not an unbiased performance estimate. Both learning rate and fine-tuning depth differ.

| Trial | Fine-tuning | Learning rate | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AP | SD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | frozen | 0.0003 | 0.4908 | 0.7138 | 0.5375 | 0.4051 | 0.7460 | 0.5787 | 0.1465 |
| 1 | partial | 0.0001 | 0.5348 | 0.8102 | 0.7488 | 0.6829 | 0.8881 | 0.7330 | 0.1342 |

Selected trial: **1**, using the prespecified highest-mean-AP rule.

Variation is descriptive. A short training budget, source imbalance and unknown patient overlap limit interpretation.

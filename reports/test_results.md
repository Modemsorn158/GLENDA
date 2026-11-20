# Locked holdout evaluation

Frames: 2763; positive-frame fraction: 0.0271; fixed threshold: 0.99997282.
Known linked groups: 22; groups containing negative frames: 3.

Confusion counts: TP 20, FP 0, TN 2688, FN 55.

| Metric | Estimate | Cluster bootstrap 95% interval | Valid replicates |
|---|---:|---:|---:|
| Accuracy | 0.9801 | 0.2716–0.9940 | 1000 |
| Balanced accuracy | 0.6333 | 0.5656–0.7000 | 960 |
| Precision | 1.0000 | 1.0000–1.0000 | 1000 |
| Recall / sensitivity | 0.2667 | 0.1316–0.4000 | 1000 |
| Specificity | 1.0000 | 1.0000–1.0000 | 960 |
| F1 | 0.4211 | 0.2325–0.5714 | 1000 |
| ROC-AUC | 0.9506 | 0.9238–1.0000 | 960 |
| Average precision | 0.6573 | 0.4485–1.0000 | 960 |
| Brier score | 0.4793 | 0.0047–0.5142 | 1000 |

| Fixed operating point | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|
| Development-selected (0.99997282) | 1.0000 | 0.2667 | 1.0000 | 0.4211 |
| 0.5 reference | 0.0422 | 1.0000 | 0.3668 | 0.0810 |

At the curated release-frame prevalence of 0.0270, applying these held-out rates gives an illustrative 0.0 false positives and 19.8 false negatives per 1,000 frames, and precision 1.0000. This is not a clinical forecast.

| Excluded negative-containing group | Remaining specificity | Remaining sensitivity |
|---|---:|---:|
| Group 1 | 1.0000 | 0.2667 |
| Group 2 | 1.0000 | 0.2639 |
| Group 3 | 1.0000 | 0.2740 |

The leave-one-group-out results are descriptive sensitivity analyses at the fixed threshold, not independent tests or a basis for retuning. 3 known groups contribute negative frames, with 84.4% of those frames in the largest group; unmapped patient relationships remain possible. Intervals are conditional on the available groups and may be unstable. With zero observed false positives, the empirical bootstrap produces degenerate intervals for some metrics; those intervals do not establish perfect specificity or precision on unseen data. This dataset has been used previously; external validation is absent.

# Five-fold feasibility results

Generated from `outputs/search/feasibility/` and the recorded split protocol. This is a single one-epoch configuration, not a comparison or held-out test.

The constant-score baseline has average precision equal to the fold's positive-frame fraction.

| Development fold | Positive frames | Negative frames | Constant-score AP | Model AP |
|---:|---:|---:|---:|---:|
| 0 | 60 | 2090 | 0.0279 | 0.3938 |
| 1 | 59 | 2090 | 0.0275 | 0.1354 |
| 2 | 60 | 2060 | 0.0283 | 0.5164 |
| 3 | 59 | 2477 | 0.0233 | 0.3595 |
| 4 | 60 | 2033 | 0.0287 | 0.4503 |

Mean average precision: **0.3711**; sample standard deviation: **0.1445**.

Fold variation is descriptive and does not establish statistical significance. Unknown patient links, curated prevalence and model selection limit interpretation.

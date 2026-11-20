# Split composition

Group counts reflect known linked source components, not confirmed independent patients.

| Partition | Positive frames | Negative frames | Positive-bearing groups | Negative-bearing groups |
|---|---:|---:|---:|---:|
| development | 298 | 10750 | 80 | 14 |
| test | 75 | 2688 | 21 | 3 |

| Validation fold | Positive frames | Negative frames | Positive-bearing groups | Negative-bearing groups |
|---:|---:|---:|---:|---:|
| 0 | 60 | 2341 | 17 | 3 |
| 1 | 59 | 2730 | 15 | 2 |
| 2 | 60 | 1484 | 15 | 3 |
| 3 | 60 | 1821 | 17 | 3 |
| 4 | 59 | 2374 | 16 | 3 |

Groups can contain frames from both labels, so the two group columns need not sum to the total. The holdout is unchanged from v1; it contains 3 negative-bearing groups, and the largest contributes 84.4% of its negative frames.

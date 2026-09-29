# Fresh MXFP4 projection-guard result

Integrity: **PASS** (126/126).

| Comparison | Mean | Median | Saving ms | Wins | 95% CI | Gate |
|:--|--:|--:|--:|--:|--:|:--:|
| A → B | 1.007188x | 1.006809x | 37.004 | 33/40 | [1.002001x, 1.012199x] | PASS |
| B → C | 0.995823x | 1.000533x | -21.595 | 19/40 | [0.991713x, 0.999893x] | FAIL |
| A → C | 1.002980x | 1.007345x | 15.409 | 32/40 | [0.997648x, 1.008260x] | FAIL |

## Validation NLL

- A: 7.88865
- B: 8.05890; delta vs A +0.17025; FAIL
- C: 7.89820; delta vs A +0.00955; PASS
- Non-gating C minus B delta: -0.16070

## Decision

retain whole-layer tail1 A; projection guards did not pass.
Selected policy: A.

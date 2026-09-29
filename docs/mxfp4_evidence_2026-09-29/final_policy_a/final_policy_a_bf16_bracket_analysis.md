# Fresh BF16 / MXFP4 Policy A / BF16 bracket

Integrity: **PASS** (108/108); final confirmation: **FAIL**.

Only fresh steps 11--50 enter the paired result. The BF16 control is the same-step midpoint of A1 and A2.

| Metric | Result | Gate |
|:---|---:|:---:|
| Mean speedup | 1.558675x | >= 1.6x |
| Median speedup | 1.576439x | >= 1.6x |
| Paired wins | 40/40 | >= 28/40 |
| Block-4 95% speedup CI | [1.542335x, 1.573744x] | lower >= 1.6x |
| BF16 mean drift | -0.109% | abs < 3% |
| BF16 median drift | -0.122% | abs < 3% |
| Delta NLL | +0.00790 | <= +0.01 |

## Decision

MXFP4 Policy A did not pass every preregistered bracket gate.

confirmation_passed=false.

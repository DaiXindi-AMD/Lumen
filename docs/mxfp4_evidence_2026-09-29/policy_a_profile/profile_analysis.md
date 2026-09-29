# Fresh BF16 / MXFP4 Policy A profiling analysis

Integrity result: **FAIL**

This report uses only artifacts under the campaign root. Profile timing is diagnostic; the 1.6x target still requires a fresh unprofiled symmetric confirmation.

## Failed integrity checks

- `profile_meta`
- `source_audit`
- `paired_run_evidence`

## Paired profile outcome

| Metric | BF16 | MXFP4 policy_a | Delta / ratio |
|---|---:|---:|---:|
| Profiler span per step (ms) | 8144.621 | 5203.277 | speedup 1.565x |
| Validation NLL | 12.383200 | 12.776500 | 0.393300 |
| 1.6x target step time (ms) | n/a | 5090.388 | gap 112.888 ms |

The validation delta is a paired eight-step diagnostic, not a long-horizon convergence claim.

## Overlap-safe category comparison

| Category | BF16 raw ms/step | PolicyA raw ms/step | BF16 union ms/step | PolicyA union ms/step |
|---|---:|---:|---:|---:|
| `a4w4_gemm` | 0.000 | 1244.885 | 0.000 | 1239.704 |
| `activation_residual` | 667.253 | 531.381 | 654.827 | 522.441 |
| `attention` | 1721.483 | 1684.812 | 1719.095 | 1682.531 |
| `bf16_gemm` | 5138.239 | 604.013 | 5131.509 | 603.747 |
| `collectives` | 289.820 | 184.365 | 289.782 | 184.346 |
| `copy_memset` | 24.019 | 71.651 | 24.019 | 71.639 |
| `cross_entropy` | 55.268 | 54.022 | 55.264 | 54.017 |
| `mxfp4_quant_layout` | 0.000 | 363.705 | 0.000 | 363.387 |
| `norm` | 227.038 | 229.059 | 223.860 | 225.115 |
| `optimizer` | 54.943 | 55.217 | 54.569 | 54.886 |
| `other` | 11.225 | 10.942 | 11.125 | 10.833 |
| `rope` | 78.158 | 75.750 | 76.282 | 74.670 |

## Amdahl ceiling

A4W4 plus quant/layout occupies 1597.108 ms/step by interval union (30.69% of the policy_a profile span). Even removing that union entirely would yield an idealized BF16-over-policy_a ratio of 2.259x. This is an impossible-zero-cost ceiling, not a predicted saving.

## Integrity checks

| Check | Result |
|---|---|
| `required_artifacts` | PASS |
| `profile_meta` | FAIL |
| `source_audit` | FAIL |
| `imports_before` | PASS |
| `campaign_stage_status` | PASS |
| `campaign_progress` | PASS |
| `campaign_kfd_before` | PASS |
| `campaign_kfd_after` | PASS |
| `cache_schema_identity_and_copies` | PASS |
| `smoke_mxfp4_policy_a.run_meta` | PASS |
| `smoke_mxfp4_policy_a.train_status` | PASS |
| `smoke_mxfp4_policy_a.postflight_status` | PASS |
| `smoke_mxfp4_policy_a.kfd_before` | PASS |
| `smoke_mxfp4_policy_a.kfd_prelaunch` | PASS |
| `smoke_mxfp4_policy_a.kfd_after` | PASS |
| `smoke_mxfp4_policy_a.kfd_after_attempts` | PASS |
| `smoke_mxfp4_policy_a.source_freeze` | PASS |
| `smoke_mxfp4_policy_a.training_log` | PASS |
| `smoke_mxfp4_policy_a.profile_disabled_sentinel` | PASS |
| `smoke_mxfp4_policy_a.cache_before_after` | PASS |
| `smoke_mxfp4_policy_a.aiter_config_cache` | PASS |
| `replay_mxfp4_policy_a.run_meta` | PASS |
| `replay_mxfp4_policy_a.train_status` | PASS |
| `replay_mxfp4_policy_a.postflight_status` | PASS |
| `replay_mxfp4_policy_a.kfd_before` | PASS |
| `replay_mxfp4_policy_a.kfd_prelaunch` | PASS |
| `replay_mxfp4_policy_a.kfd_after` | PASS |
| `replay_mxfp4_policy_a.kfd_after_attempts` | PASS |
| `replay_mxfp4_policy_a.source_freeze` | PASS |
| `replay_mxfp4_policy_a.training_log` | PASS |
| `replay_mxfp4_policy_a.profile_disabled_sentinel` | PASS |
| `replay_mxfp4_policy_a.cache_before_after` | PASS |
| `replay_mxfp4_policy_a.aiter_config_cache` | PASS |
| `profile_bf16.run_meta` | PASS |
| `profile_bf16.train_status` | PASS |
| `profile_bf16.postflight_status` | PASS |
| `profile_bf16.kfd_before` | PASS |
| `profile_bf16.kfd_prelaunch` | PASS |
| `profile_bf16.kfd_after` | PASS |
| `profile_bf16.kfd_after_attempts` | PASS |
| `profile_bf16.source_freeze` | PASS |
| `profile_bf16.training_log` | PASS |
| `profile_bf16.cache_before_after` | PASS |
| `profile_bf16.aiter_config_cache` | PASS |
| `profile_mxfp4_policy_a.run_meta` | PASS |
| `profile_mxfp4_policy_a.train_status` | PASS |
| `profile_mxfp4_policy_a.postflight_status` | PASS |
| `profile_mxfp4_policy_a.kfd_before` | PASS |
| `profile_mxfp4_policy_a.kfd_prelaunch` | PASS |
| `profile_mxfp4_policy_a.kfd_after` | PASS |
| `profile_mxfp4_policy_a.kfd_after_attempts` | PASS |
| `profile_mxfp4_policy_a.source_freeze` | PASS |
| `profile_mxfp4_policy_a.training_log` | PASS |
| `profile_mxfp4_policy_a.cache_before_after` | PASS |
| `profile_mxfp4_policy_a.aiter_config_cache` | PASS |
| `route_smoke_replay_identity` | PASS |
| `paired_run_evidence` | FAIL |
| `formal_command_pairing` | PASS |
| `profile_bf16.trace` | PASS |
| `profile_mxfp4_policy_a.trace` | PASS |
| `finalization` | PASS |

## Finalization state

`post-analysis-pending`

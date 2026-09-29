# Supplemental BF16 / MXFP4 Policy A profile audit

Supplemental integrity result: **PASS**

The original analyzer result remains **FAIL** and no original campaign artifact was rewritten. This supplemental audit corrects only the two known analyzer contracts and independently streams both raw traces on CPU.

## Corrected integrity contracts

- Source audit after path normalization: **PASS**. Relative key: `run_policy_a_profile.sh`; it resolves against the campaign root and its recorded digest matches the file and `profile-meta.txt`.
- 640-sample smoke/replay pairing: **PASS**.
- 1,024-sample BF16/Policy A formal pairing: **PASS**.
- Model initialization hash matches across all four arms: **PASS**.
- Validation input/label digests match across all four arms: **PASS**.
- The 640-sample and 1,024-sample first-update digest sets are different, as expected for different formal data extents; they are not compared as one pair.

## Paired formal outcome

| Metric | BF16 | MXFP4 Policy A | Delta / ratio |
|---|---:|---:|---:|
| Profiler span per step (ms) | 8144.621 | 5203.277 | speedup 1.565x |
| Validation NLL | 12.383200 | 12.776500 | 0.393300 |
| 1.6x target step time (ms) | n/a | 5090.388 | gap 112.888 ms |

Profiler timing remains diagnostic evidence only.

## Trace windows and GPU occupancy

| Metric per step (ms) | BF16 | MXFP4 Policy A |
|---|---:|---:|
| Profiler span | 8144.621 | 5203.277 |
| GPU envelope | 8142.013 | 5198.703 |
| GPU busy interval union | 7998.705 | 4922.522 |
| GPU idle inside envelope | 143.308 | 276.181 |
| Profiler non-busy | 145.917 | 280.755 |
| Profiler edge outside GPU envelope | 2.608 | 4.574 |
| Raw GPU duration sum | 8267.447 | 5109.802 |
| Raw-minus-busy overlap | 268.742 | 187.280 |

## Overlap-safe GPU categories

| Category | BF16 raw ms/step | Policy A raw ms/step | BF16 union ms/step | Policy A union ms/step |
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

## Collectives decoded from `Collective name`

| Operation | BF16 calls/step | BF16 raw ms/step | Policy A calls/step | Policy A raw ms/step |
|---|---:|---:|---:|---:|
| `all_gather` | 81.000 | 239.087 | 81.000 | 137.158 |
| `all_reduce` | 1.000 | 0.026 | 1.000 | 0.106 |
| `reduce_scatter` | 37.000 | 50.707 | 37.000 | 47.100 |

All physical collective kernels in both traces were resolved directly from the GPU event argument; none remained in an `other` collective bucket.

## MXFP4 A4W4 exact shapes

| Phase | Shape | Calls/step | Raw ms/step | Union ms/step | Backend | Symbol / tile / split |
|---|---|---:|---:|---:|---|---|
| `a4w4_forward` | `M=16384,N=12288,K=4096` | 560.000 | 236.495 | 236.495 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E` / `128x512` / `0` |
| `a4w4_wgrad` | `M=12288,N=4096,K=16384` | 560.000 | 209.025 | 209.025 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E` / `128x512` / `0` |
| `a4w4_dgrad` | `M=16384,N=4096,K=12288` | 560.000 | 199.396 | 199.396 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_dgrad` | `M=16384,N=12288,K=4096` | 280.000 | 121.556 | 121.556 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E` / `128x512` / `0` |
| `a4w4_forward` | `M=16384,N=4096,K=12288` | 280.000 | 97.933 | 97.933 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_wgrad` | `M=4096,N=12288,K=16384` | 280.000 | 97.140 | 97.140 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_packed_qkv_forward` | `M=16384,N=6144,K=4096` | 280.000 | 62.633 | 62.633 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_packed_qkv_dgrad` | `M=16384,N=4096,K=6144` | 280.000 | 52.319 | 52.319 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_packed_qkv_wgrad` | `M=6144,N=4096,K=16384` | 280.000 | 50.268 | 50.268 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E` / `192x256` / `0` |
| `a4w4_dgrad` | `M=16384,N=4096,K=4096` | 280.000 | 43.860 | 43.860 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_forward` | `M=16384,N=4096,K=4096` | 280.000 | 39.773 | 39.773 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |
| `a4w4_wgrad` | `M=4096,N=4096,K=16384` | 280.000 | 34.487 | 34.487 | `asm` | `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` / `256x256` / `0` |

## Selected host/runtime calls

| Call/category | BF16 calls/step | BF16 raw ms/step | Policy A calls/step | Policy A raw ms/step |
|---|---:|---:|---:|---:|
| `quantized_linear_forward` | 0.000 | 0.000 | 1120.000 | 454.959 |
| `quantized_linear_backward` | 0.000 | 0.000 | 1120.000 | 805.732 |
| `packed_qkv_forward` | 0.000 | 0.000 | 280.000 | 120.337 |
| `packed_qkv_backward` | 0.000 | 0.000 | 280.000 | 215.151 |
| `split_swiglu_forward` | 0.000 | 0.000 | 280.000 | 24.100 |
| `split_swiglu_backward` | 0.000 | 0.000 | 280.000 | 50.748 |
| HIP `kernel_launch` | 21471.000 | 1316.629 | 23186.000 | 1079.907 |
| HIP `pointer_query` | 18640.000 | 11.225 | 42265.000 | 27.928 |
| HIP `synchronization` | 27.500 | 4179.310 | 27.500 | 1319.831 |
| HIP `blocking_copy` | 143.000 | 14.749 | 143.000 | 3.148 |

## Missing External id and Amdahl ceiling

- BF16 missing-External-id GPU events: 0.
- Policy A missing-External-id GPU events: 1540; all are retained and resolved by kernel-name rules plus correlated HIP API grouping.
- A4W4 plus quant/layout union: 1597.108 ms/step (30.69% of the Policy A profiler span).
- Impossible-zero-cost BF16-over-Policy-A ceiling: 2.259x.

## Preservation and hashes

- Original analyzer JSON SHA256: `f0d2735443dd746e0b06520f0a7ae8cc0b7e4781af35b59eaae505bed2b8f9c3`
- Original analyzer Markdown SHA256: `8bce84e421d0815c1e8337799ac66208eb55e9ac4b79e7e99566140ca73e0a49`
- Original analyzer source SHA256: `6e5b8c7cfa30755706a626d2ac326e0d6f666f1c46d640da3fedb953c36095bc`
- Supplemental script SHA256: `44063f946a183afda972c38edca8e666025fe8755bb251ced461728fde33cbef`
- Original exit sentinel remains `1`; campaign stage remains `failed_stage=analysis`, `failed_status=1`.

## Limits and risks

- The traces cover optimizer steps 7 and 8 only. No `mxfp4_weight_quant` GPU event appears in that window, so this audit does not estimate one-time initialization, cache-build, or pre-window weight-quantization cost.
- The GPU kernel-name `split_swiglu` bucket has 288 calls/step while the Policy A custom split-SwiGLU host scope has 280 calls/step; the extra eight calls/step come from work outside the 35 quantized-layer custom scope (including the protected BF16 tail), so the GPU bucket must not be read as a quantized-layer count.
- The 1,540 Policy A events without `External id` are name-resolved helper kernels. Their categories are reliable at kernel-family level, but they have no producer shape/scope attribution.
- These supplemental files are deliberately outside the original failed campaign's completion manifest and are not a replacement for a signed zero-exit campaign result.

Raw duration sums and per-category unions can overlap across streams/categories. They are diagnostic and must not be added to predict step-time savings.

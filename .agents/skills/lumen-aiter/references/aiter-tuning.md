# AITER tuning and config files

Read before adding, tuning, moving, or renaming Triton/Gluon config files or changing their loaders.

## Source and branch compatibility

The authoritative source is [configs/CLAUDE.md](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/aiter/ops/triton/configs/CLAUDE.md), verified 2026-09-18. The [maintainer guide](https://github.com/ROCm/aiter/blob/1834cd14b33ff7fd366c8ba6e6486e41df027b70/aiter/ops/triton/README.md) adds autotuning conventions. Read the actual target revision's rulebook and loaders; they can evolve.

At this upstream revision there is one nested layout. Older Lumen AITER forks/checkouts may still use `configs/gemm/{arch}-{CONFIG_NAME}.json`. For a backport, follow the loader actually present and clearly identify the compatibility target. For an upstream patch, use the current nested layout; do not reintroduce old flat paths or quietly undertake an unrelated config migration.

## Layout and resolution

```text
aiter/ops/triton/configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
aiter/ops/triton/configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

- `<arch>` is a `gfx*` identifier; `<backend>` is `triton` or `gluon`, declared by the caller.
- `<op>` families include `gemm`, `moe`, `conv`, `mhc`, `attention`, `gmm`, `fusions`.
- `<d_type> = config_name.lower().replace("-", "_")`; check for collisions between names after folding.
- No architecture prefix inside filenames; required defaults are named exactly `DEFAULT.json`. No new `.gitkeep` placeholders.
- Every read uses a shared family loader, or the core `resolve_config_dir()` plus `load_config_json()`. No raw `json.load(open(...))`, hand-built path strings, or per-function caches.
- `resolve_config_dir(op, config_name, backend="triton", arch=None)` builds one validated path. It does not probe a list or search across backends/architectures. MHC has an explicitly documented gfx942 fallback; do not generalize it.
- `load_config_json(required=True)` raises with the missing path; optional reads return `None`. Its cached dict is shared: copy before mutating. Optional misses are cached too, so tooling that writes configs must clear the cache or restart the process.
- Tuning values live in JSON, not inline Python config dicts, `setdefault` blocks, or architecture-specific tuning constants.

## GEMM family

Use `utils/gemm_config_utils.py::get_gemm_config()` and preserve `(config, is_tuned)` through the thin `_get_config()` wrapper. Normalize a public wrapper's optional backend before passing it to loaders; kernel-level backend defaults are `"triton"`, never `None`.

```python
def _get_config(M, N, K, backend="triton"):
    return get_gemm_config("GEMM-A16W16", M, N, K, backend=backend)
```

For split-K, use `compute_splitk_params(config, K)` and preserve the flag. The config returned by this family loader is a fresh deep copy. `is_tuned` distinguishes specialized resolution from default resolution; inspect the current loader for bucket details rather than treating an existing file as proof the intended tuning was selected.

- Required default: `gemm_a16w16/DEFAULT.json` under the correct arch/backend/op directory.
- Specialized: `GEMM-A16W16-N=256-K=7168.json`; batched filenames can include `-B={B}-N={N}-K={K}`. Custom fused suffixes use the loader's `specialized_filename` API.
- Use `M_LEQ_<x>` ascending, then `M_GEQ_<x>` descending, then `any`. Include `any` unless all reachable M values are explicitly covered. Do not use old `small`/`large` buckets. Custom bounds are strictly increasing positive integers.
- GEMM entries include `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `cache_modifier`, `NUM_KSPLIT`; loader backfill is not permission to omit them.
- In AFP4WFP4 filenames, K is logical K (`2 * K_bytes`), not packed storage width.
- Do not add `kpack` to gfx950 or new RDNA configs. Legacy gfx942 settings may still use it; inspect the target backend and family.

## Other families and autotuning

| Family | Loader and invariant |
| --- | --- |
| MoE | `get_moe_dispatch(config_name, arch, backend)`; returned shared table is read-only. Triton uses `bm<block_m>_n<N>_k<K>`, Gluon adds M buckets and `bm<block_m>_any`. Do not mix schemas. |
| Convolution | `get_conv_config()`; use shared shape-key formatters and the documented exact-shape/bucket/default rules. |
| MHC | `get_mhc_config()` / `get_mhc_post_config()`; follow the documented C/M thresholds and deliberate gfx942 fallback. |
| Attention/GMM simple tables | Core resolver + JSON loader, without inventing a new family module for a single file read. |
| Pinned autotune tile | `get_tuned_kernel_config()`; the fallback supplied to this helper must be launchable across supported architectures, not merely fast on one. |

For newly tuned Gluon MoE shapes, include all six bucket suffixes (`tiny`, `small`, `medium`, `medium2`, `large`, `xlarge`) and the `bm<block_m>_any` tier. `block_m` is part of the dispatch key, not a tuning value inside an entry. Use backend-specific parameter names.

Every Triton autotune config list goes through `utils/tuned_config_utils.py::autotune_configs()`. Normal execution pins one config; `<FAMILY>_TRITON_AUTOTUNE=1` enables the search. Preserve existing documented family-specific environment/default exceptions via the helper's arguments. A JSON list is still a search space, not permission to benchmark every candidate at launch. Pin the default through `get_tuned_kernel_config`; correctness tests must not autotune.

The helper's explicitly supported portable fallback/search-space API is distinct from adding new per-op hardcoded tuning fallback dictionaries. Prefer measured JSON settings and preserve the documented helper contract.

## Changes and verification

- Keep config moves/renames in a pure `git mv` commit with 100% similarity; put content edits in a following commit.
- At the verified revision, the allowed cross-arch seed is a byte-identical **gfx950 → gfx1250, Triton-only** copy, disclosed in the commit message. Never seed Gluon from another architecture or seed backwards into gfx950. Seeded data is not measured tuning.
- Verify the actual config path and backend, specialized selection on an intended shape, fallback selection for uncovered shapes, and unchanged numerics. Run the matching benchmark and retain measurements.
- Runtime `configs/gemm/aot/` and `configs/paged_mqa_logits/aot/` files are compiled metadata caches, not tuning files: do not commit or migrate them.

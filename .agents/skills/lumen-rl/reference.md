# Lumen RL Reference

The baseline and roadmap below are historical. Verify the target adapters, pinned dependencies, and test results before identifying unfinished work; user-selected scope overrides the default v1 choices.


## Objective

Evaluate realistic framework options for distributed RL training on top of Lumen, prioritize the integration order, and keep the landing path aligned with Lumen's current AMD-first backend reality.

## Current Baseline

Lumen currently exposes two real training backends:

- `Megatron` via `lumen/models/megatron.py`
- `HF + FSDP/FSDP2` via `lumen/models/fsdp.py`

Implications for RL:

- Lumen already patches HuggingFace causal LMs with FP8, LoRA, warmup, and checkpoint behavior.
- Lumen already builds Megatron-backed training flows with TP/PP/CP/SP and Lumen-specific communication hooks.
- The original roadmap predated `lumen/rl/`. The migration checkout already contains TRL and VERL adapters; inspect them and their tests to determine what remains.
- Lumen does not currently expose a native `DeepSpeed` integration path.
- Several distributed efficiency items are still partial or unwired for end-to-end RL, especially `fp8_param_gather`, full TP overlap, and full `delay_wgrad` wiring.
- The existing FSDP path is trainer-centric. It works well for Lumen-owned training loops, but it is not yet packaged as a reusable backend adapter for third-party RL frameworks.

## Framework Comparison

| Framework | Backend shape | Lumen fit | Estimated difficulty | Recommendation |
|-----------|---------------|-----------|----------------------|----------------|
| `TRL` | `Transformers` + `Trainer` / `Accelerate`, optional `DeepSpeed`, optional `vLLM` | Best fit with `HF + FSDP/FSDP2` | Medium-low | First implementation target |
| `VERL` | Native `FSDP/FSDP2/Megatron` training plus `HF/vLLM/SGLang` rollout | Strong long-term fit, but more moving pieces | Medium-high | Second target after TRL |
| `OpenRLHF` | `Ray + vLLM + DeepSpeed` | Weak fit because training side is `DeepSpeed`-centric | High | Not recommended as first Lumen target |
| `DeepSpeedChat` | End-to-end `DeepSpeed` RLHF pipeline | Weak fit and older architecture | High to very high | Reference only, not first-class target |

## Why `TRL` Comes First

`TRL` is the easiest path because it is built around HuggingFace `transformers` models and trainer abstractions. Lumen already has a viable `HF + FSDP/FSDP2` path, so the first integration can be reduced to:

1. Build a HuggingFace causal LM
2. Apply Lumen FP8 / LoRA / warmup / checkpoint semantics
3. Let `TRL` own the RL algorithm loop
4. Use `Accelerate` FSDP configs for distributed execution

Why this fits:

- Lumen's existing FSDP path already targets HuggingFace LLaMA models.
- `TRL` supports `GRPO`, `RLOO`, `DPO`, reward modeling, and optional `vLLM`.
- The initial rollout path can stay on plain HuggingFace generation, which avoids early `vLLM` or `DeepSpeed` complexity on AMD.
- A `GRPO`-first milestone avoids the extra value-head and actor-critic complexity of a PPO-first path.

Remaining work:

- Lumen needs a reusable adapter layer instead of only task-specific trainers.
- Lumen needs explicit lifecycle handling for FP8 warmup, `reset_fp8_state()`, checkpoint save/load, and logging.
- `FSDP2` support must be expressed through `Accelerate` / `Trainer` configs rather than only the existing Lumen-owned trainer loop.

Bottom line: `TRL` is the shortest path to a useful distributed RL proof of concept.

## Why `VERL` Comes Second

`VERL` is the most attractive long-term framework because it already understands both `FSDP/FSDP2` and `Megatron-LM`, while separating rollout and training responsibilities more cleanly than `TRL`.

Why this fits long-term:

- It can host a Lumen-aware FSDP path first.
- It has a native place for a future Lumen Megatron backend.
- It separates actor, reference, critic, and rollout responsibilities more cleanly than `TRL`.
- It is a better home for large-scale multi-node on-policy training once the basics are stable.

Why it is not the first target:

- The orchestration layer is substantially larger: workers, placement, config trees, rollout engines, and checkpoint synchronization all have to line up.
- The initial integration surface is wider than `TRL`, even before Megatron enters the picture.
- `VERL` can use `vLLM`, `SGLang`, or `HF Transformers` for rollout. On AMD, the simplest path is not the default fastest path.

Strategic rule:

- `VERL + Lumen FSDP` should be the first `VERL` milestone.
- `VERL + Lumen Megatron` should be a second milestone, not part of the initial bring-up.
- For the Megatron phase, actor and reference model support should come first. The critic can remain `HF + FSDP` initially if that reduces scope and checkpoint-conversion risk.

Bottom line: `VERL` is the right second target once Lumen has already proven RL value with `TRL`.

## Why Not Start with `OpenRLHF` or `DeepSpeedChat`

### `OpenRLHF`

`OpenRLHF` is impressive, but it is a poor first fit for Lumen because its architecture is intentionally centered on `Ray + vLLM + DeepSpeed`, and the training path is explicitly tied to `DeepSpeed ZeRO-3`.

Problems for Lumen:

- Lumen does not currently offer a `DeepSpeed` training backend.
- A lot of OpenRLHF's value comes from hybrid-engine assumptions, not just the model implementation.
- Even if Lumen is injected into the HuggingFace model, training-system ownership still belongs to `DeepSpeed`.
- End-to-end speedups are harder to realize if rollout remains dominated by `vLLM`.

### `DeepSpeedChat`

`DeepSpeedChat` is even more tightly coupled to `DeepSpeed` than OpenRLHF and is best treated as a reference architecture, not a first integration target.

Problems for Lumen:

- It assumes `DeepSpeed` ownership of both training system behavior and model execution strategy.
- The project is more monolithic and less attractive than `VERL` as a long-term extensible integration target.
- The work to make `Lumen + DeepSpeedChat` useful is almost the same as the work to build `Lumen + DeepSpeed`.

## Direct `DeepSpeed` Support

Direct `DeepSpeed` support is a separate decision from RL framework selection.

### Level 1: Run Lumen inside ZeRO

Goal:

- Let a Lumen-patched HuggingFace model run under `DeepSpeed ZeRO-2/3`
- Do not try to preserve Lumen-specific communication optimizations yet

Difficulty: medium

Needed:

- parameter lifecycle validation under ZeRO
- optimizer and checkpoint compatibility checks
- BF16 and FP8 correctness validation
- basic smoke coverage for save/resume

### Level 2: Usable DeepSpeed backend

Goal:

- Stable actor / reference / reward / critic training under `DeepSpeed`
- LoRA, checkpointing, and RL-friendly save/load semantics work reliably

Difficulty: medium-high

Needed:

- model-state materialization correctness under ZeRO-3
- reward model and RL step coverage
- stronger checkpoint compatibility
- practical launcher and config surface

### Level 3: DeepSpeed that preserves Lumen's differentiated value

Goal:

- Not merely "Lumen modules happen to run inside DeepSpeed"
- But "Lumen-specific FP8 and communication features still matter inside the DeepSpeed system"

Difficulty: very high

Why it is hard:

- Lumen's strongest distributed value today is tied to `Megatron` and `MORI/SDMA`.
- `DeepSpeed` owns its own partitioning, communication, and hybrid-engine assumptions.
- Bridging those systems cleanly is a project of its own.

Recommendation:

- Do not make first-class `DeepSpeed` support a prerequisite for initial RL bring-up.

## Recommended Landing Path

### Phase 0: Shared Prerequisites

Before integrating any RL framework:

1. Create a `lumen/rl/` package boundary for framework-specific adapters.
2. Freeze the first supported model family to LLaMA-class HuggingFace checkpoints.
3. Freeze the first RL algorithm to `GRPO`, so framework comparisons are apples-to-apples.
4. Use plain HuggingFace rollout for the first milestone unless a framework strictly requires something else.
5. Define a shared smoke-test matrix:
   - CPU unit tests for config / model lifecycle helpers
   - 2-GPU `FSDP1` smoke
   - 2-GPU `FSDP2` smoke
   - save / resume smoke
   - loss decreases over a tiny fixed dataset
6. Define shared stop / go criteria:
   - no NaN / Inf
   - stable loss curve on a tiny run
   - checkpoint resume works
   - FP8 warmup / reset semantics survive distributed startup

### Phase 1: `TRL + Lumen + FSDP/FSDP2`

Scope:

- `GRPO` first, not PPO first
- HuggingFace rollout first
- `FSDP1` and `FSDP2`
- LLaMA-family models only
- no `Megatron`
- no `DeepSpeed`

### Phase 2: `VERL + Lumen FSDP`

Scope:

- `VERL` workers with a Lumen-aware HuggingFace / FSDP backend
- keep rollout simple first: HuggingFace before `vLLM` or `SGLang`
- preserve Lumen FP8 / LoRA / warmup / checkpoint behavior

### Phase 3: `VERL + Lumen Megatron`

Scope:

- actor and reference model first
- Megatron model construction via existing `lumen/models/megatron.py`
- initial critic can remain `HF + FSDP`
- do not attempt full parity with all Megatron distributed features on day one

### Phase 4: Optional `DeepSpeed`

Start only if:

- a target RL framework forces it
- a customer or internal workload specifically requires `ZeRO-3`
- both `TRL` and `VERL` paths hit a real memory or scaling ceiling that `DeepSpeed` would solve

## Default Recommendation Summary

Recommended order:

1. `TRL + Lumen + FSDP/FSDP2`
2. `VERL + Lumen FSDP`
3. `VERL + Lumen Megatron`
4. Optional `DeepSpeed` support if a real workload justifies it

Avoid:

- starting with `OpenRLHF`
- starting with `DeepSpeedChat`
- making direct `DeepSpeed` support a prerequisite for initial RL validation
- trying to bring up `VERL` FSDP and `VERL` Megatron in the same first patch series

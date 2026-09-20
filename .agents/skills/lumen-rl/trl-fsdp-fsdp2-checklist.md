# TRL + Lumen FSDP/FSDP2 Execution Checklist

## Goal

Bring up `GRPO`-first distributed RL training in `TRL` using Lumen-patched HuggingFace models under `FSDP1` and `FSDP2`, while keeping rollout generation on plain HuggingFace Transformers in the first milestone.

The paths and milestones below describe the original rollout plan. Reuse implementations already present; create missing pieces only for the requested milestone. The v1 defaults yield to an explicit user-selected framework, algorithm, rollout engine, or model family.

## Fixed Decisions

- [ ] Lock the first online RL algorithm to `GRPO`
- [ ] Lock rollout generation to plain HuggingFace Transformers for v1
- [ ] Lock the training backend to `FSDP1` and `FSDP2`
- [ ] Lock the first supported model family to LLaMA-class HuggingFace checkpoints
- [ ] Keep `DeepSpeed`, `vLLM`, `PPO` value-head support, Megatron, and non-LLaMA models out of scope for v1

## Pre-Flight

- [ ] Confirm the target machine has 2 usable GPUs for distributed smoke validation
- [ ] Confirm the implementation will reuse `lumen.models.fsdp` helpers rather than cloning LoRA / FP8 / checkpoint logic
- [ ] Confirm `examples/rl/trl/requirements.txt` will be the source of truth for the pinned `trl` / `transformers` / `accelerate` / `datasets` stack
- [ ] Confirm the team understands the ownership risk: the pinned `TRL + Accelerate + FSDP` stack may or may not accept a pre-built model

## Task 1: Package Boundary and Argument Contract

Files:

- `lumen/rl/__init__.py`
- `lumen/rl/trl/__init__.py`
- `lumen/rl/trl/args.py`
- `tests/rl/trl/test_args.py`

Checklist:

- [ ] Write the failing contract tests in `tests/rl/trl/test_args.py`
- [ ] Add `TrlLumenArgs`
- [ ] Add `build_accelerate_config_path(...)`
- [ ] Add `build_grpo_config_kwargs(...)`
- [ ] Keep `lumen/rl/trl/args.py` pure: do not import `torch`, `transformers`, or `trl`
- [ ] Re-export the stable entrypoints from `lumen/rl/trl/__init__.py`
- [ ] Run `pytest tests/rl/trl/test_args.py -v`
- [ ] Mark this task done only after the contract tests pass

## Task 2: HuggingFace Model Builders with Lumen Patching

Files:

- `lumen/rl/trl/modeling.py`
- `tests/rl/trl/test_modeling.py`

Checklist:

- [ ] Write the failing lifecycle tests for actor construction
- [ ] Add `build_actor_model(args)`
- [ ] Add `build_reference_model(args)`
- [ ] Add `build_reward_model(args)`
- [ ] Ensure actor ordering is `gradient_checkpointing -> LoRA -> FP8`
- [ ] Ensure only the actor gets LoRA in v1
- [ ] Ensure reference and reward builders can apply FP8 without LoRA
- [ ] Run `pytest tests/rl/trl/test_modeling.py -v`
- [ ] Mark this task done only after all model lifecycle tests pass

## Task 3: Synthetic Warmup and FP8 Reset

Files:

- `lumen/rl/trl/warmup.py`
- `tests/rl/trl/test_warmup.py`

Checklist:

- [ ] Write the failing warmup tests
- [ ] Add `maybe_run_synthetic_warmup(model, args, *, device)`
- [ ] Use a zero-learning-rate optimizer so warmup exercises backward and optimizer state paths without meaningful weight drift
- [ ] Reset FP8 state after the warmup loop completes
- [ ] Keep distributed logic out of this helper; device selection belongs in the runner
- [ ] Run `pytest tests/rl/trl/test_warmup.py -v`
- [ ] Mark this task done only after warmup tests pass

## Task 4: GRPO Runner and Launch Entry Points

Files:

- `lumen/rl/trl/runner.py`
- `examples/rl/trl/run_grpo_fsdp.py`
- `examples/rl/trl/run_grpo_fsdp.sh`
- `examples/rl/trl/requirements.txt`
- `examples/rl/trl/accelerate/fsdp1.yaml`
- `examples/rl/trl/accelerate/fsdp2.yaml`
- `tests/rl/trl/test_runner_smoke.py`

Checklist:

- [ ] Write the failing runner wiring test
- [ ] Add a pinned dependency stack to `examples/rl/trl/requirements.txt`
- [ ] Add `fsdp1.yaml` and `fsdp2.yaml`
- [ ] Add `run_grpo(args, *, reward_fn)` to the runner
- [ ] Add the `accelerate launch` entrypoint in `examples/rl/trl/run_grpo_fsdp.py`
- [ ] Add the canonical shell launcher in `examples/rl/trl/run_grpo_fsdp.sh`
- [ ] Verify whether the pinned `TRL + Accelerate + FSDP` stack accepts a pre-built model
- [ ] If the pinned stack rejects a pre-built model, switch to a trainer-owned lazy-init path before proceeding
- [ ] Keep the ownership choice explicit in runner comments so future upgrades know which mode is in use
- [ ] Run `pytest tests/rl/trl/test_runner_smoke.py -v`
- [ ] Mark this task done only after the runner wiring test passes

## Task 5: Distributed Smoke and Output Validation

Files:

- `tests/rl/trl/test_runner_smoke.py`
- `examples/rl/trl/run_grpo_fsdp.sh`

Checklist:

- [ ] Add an opt-in slow subprocess-based distributed smoke test
- [ ] Document the FSDP1 launcher command: `bash examples/rl/trl/run_grpo_fsdp.sh 1`
- [ ] Document the FSDP2 launcher command: `bash examples/rl/trl/run_grpo_fsdp.sh 2`
- [ ] Run the fast suite: `pytest tests/rl/trl/test_args.py tests/rl/trl/test_modeling.py tests/rl/trl/test_warmup.py tests/rl/trl/test_runner_smoke.py -v`
- [ ] Run the opt-in slow suite on a 2-GPU machine: `LUMEN_RUN_SLOW_RL_TESTS=1 pytest tests/rl/trl/test_runner_smoke.py -v`
- [ ] Run the example launcher directly: `bash examples/rl/trl/run_grpo_fsdp.sh 1`
- [ ] Confirm trainer startup succeeds under the pinned stack
- [ ] Confirm the run reaches the expected small-step smoke target
- [ ] Confirm the output directory contains trainer state and checkpoint artifacts
- [ ] If FSDP wrapping or trainer init fails because of the model ownership mode, go back to Task 4 and change the runner before retrying this task

## Exit Gate

- [ ] `lumen/rl/trl/` exists and imports cleanly
- [ ] The `GRPO` runner builds a Lumen-patched HuggingFace actor model
- [ ] Synthetic warmup plus `reset_fp8_state()` runs before real RL training
- [ ] `FSDP1` and `FSDP2` launch configs both exist
- [ ] Fast unit tests pass
- [ ] 2-GPU smoke validation passes
- [ ] No NaN / Inf is observed during the smoke run
- [ ] Checkpoint and resume behavior is understood for the pinned stack

## Deferred Until Post-v1

- `vLLM` generation support
- `PPO` value-head support
- multi-turn environment integration
- non-LLaMA model families
- DeepSpeed support

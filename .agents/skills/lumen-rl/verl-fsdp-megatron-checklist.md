# VERL + Lumen FSDP/Megatron Execution Checklist

## Goal

Integrate Lumen with `VERL` in two staged milestones: first a Lumen-aware `FSDP/FSDP2` backend for actor/reference/critic, then a Lumen-aware `Megatron` backend for actor/reference with the critic allowed to remain on `FSDP` in the first Megatron milestone.

The paths and milestones below describe the original rollout plan. Reuse implementations already present; create missing pieces only for the requested milestone. The v1 defaults yield to an explicit user-selected framework, algorithm, rollout engine, or model family.

## Fixed Decisions

- [ ] Milestone A must be `VERL + Lumen FSDP`
- [ ] Milestone B must start only after Milestone A is stable
- [ ] Keep HuggingFace rollout in v1
- [ ] Keep `DeepSpeed`, `vLLM`, `SGLang`, async RL, and full Megatron critic support out of scope for v1
- [ ] Keep the first Megatron milestone limited to actor/reference on Megatron with critic still on `FSDP`

## Pre-Flight

- [ ] Confirm the target machine has 2 usable GPUs for smoke validation
- [ ] Confirm `examples/rl/verl/requirements.txt` will pin the exact `verl` / `omegaconf` / `transformers` stack used by the bridge code
- [ ] Confirm the first implementation will keep Lumen-owned HuggingFace builders free of direct `VERL` imports
- [ ] Confirm the team understands the staged delivery boundary: no Megatron critic in v1

## Milestone A: VERL + Lumen FSDP

### Task 1: Package Boundary and Backend Matrix Contract

Files:

- `lumen/rl/__init__.py`
- `lumen/rl/verl/__init__.py`
- `lumen/rl/verl/config.py`
- `tests/rl/verl/test_config.py`

Checklist:

- [ ] Write the failing backend validation tests
- [ ] Add `VerlLumenArgs`
- [ ] Add `validate_backend_matrix(args)`
- [ ] Reject `train_backend=megatron` plus `critic_backend=megatron` in v1
- [ ] Reject non-HF rollout for v1, including the later Megatron milestone
- [ ] Keep the contract layer pure and import-light
- [ ] Run `pytest tests/rl/verl/test_config.py -v`
- [ ] Mark this task done only after the backend validation tests pass

### Task 2: FSDP Role Builders

Files:

- `lumen/rl/verl/fsdp_backend.py`
- `tests/rl/verl/test_fsdp_backend.py`

Checklist:

- [ ] Write the failing FSDP role-builder tests
- [ ] Add `build_fsdp_actor(args)`
- [ ] Add `build_fsdp_reference(args)`
- [ ] Add `build_fsdp_critic(args)`
- [ ] Keep this file Lumen-owned and free of `VERL` imports
- [ ] Ensure actor ordering is `gradient_checkpointing -> LoRA -> FP8`
- [ ] Ensure only the actor gets LoRA in v1
- [ ] Run `pytest tests/rl/verl/test_fsdp_backend.py -v`
- [ ] Mark this task done only after the FSDP builder tests pass

### Task 3: Checkpoint Bridge and FSDP Assembly Helper

Files:

- `lumen/rl/verl/checkpoint_bridge.py`
- `lumen/rl/verl/runner.py`
- `lumen/rl/verl/verl_entry.py`
- `tests/rl/verl/test_checkpoint_bridge.py`
- `tests/rl/verl/test_runner_smoke.py`

Checklist:

- [ ] Write the failing checkpoint normalization tests
- [ ] Write the failing runner wiring tests
- [ ] Add `normalize_state_dict_keys(...)`
- [ ] Add `build_verl_fsdp_components(args)`
- [ ] Add `load_lumen_args(config_path)`
- [ ] Add `build_components_from_config(config_path)`
- [ ] Keep all VERL-version-specific API knowledge isolated to `launch_with_verified_verl_entrypoint(...)`
- [ ] Run `pytest tests/rl/verl/test_checkpoint_bridge.py tests/rl/verl/test_runner_smoke.py -v`
- [ ] Mark this task done only after bridge and runner tests pass

### Task 4: FSDP Config, Launcher, and Smoke Path

Files:

- `examples/rl/verl/configs/grpo_fsdp_lumen.yaml`
- `examples/rl/verl/run_grpo_fsdp.sh`
- `examples/rl/verl/requirements.txt`
- `tests/rl/verl/test_runner_smoke.py`

Checklist:

- [ ] Add the pinned `VERL` dependency stack
- [ ] Add `grpo_fsdp_lumen.yaml`
- [ ] Keep the shell script pointed at `python -m lumen.rl.verl.verl_entry`
- [ ] Replace the launcher stub with the verified API for the pinned `VERL` release
- [ ] Run `pytest tests/rl/verl/test_config.py tests/rl/verl/test_fsdp_backend.py tests/rl/verl/test_checkpoint_bridge.py tests/rl/verl/test_runner_smoke.py -v`
- [ ] Run the manual smoke: `bash examples/rl/verl/run_grpo_fsdp.sh`
- [ ] Confirm config parsing succeeds
- [ ] Confirm the FSDP-first smoke run finishes without NaN / Inf

## Promotion Gate

Do not start Megatron until all items below are true:

- [ ] Milestone A fast tests pass
- [ ] Milestone A smoke run passes
- [ ] The pinned `VERL` launcher path is validated for the selected release
- [ ] The team agrees that actor/reference/critic construction is stable enough to extend

## Milestone B: VERL + Lumen Megatron

### Task 5: Megatron Actor/Reference Adapter

Files:

- `lumen/rl/verl/megatron_backend.py`
- `tests/rl/verl/test_megatron_backend.py`

Checklist:

- [ ] Write the failing Megatron provider tests
- [ ] Add `build_megatron_argv(args)`
- [ ] Add `prime_megatron_global_args(args)`
- [ ] Add `build_megatron_actor_reference_provider(args)`
- [ ] Reuse `make_lumen_model_provider(...)`
- [ ] Keep dataset-provider logic out of this adapter
- [ ] Keep critic support off Megatron in v1
- [ ] Replace the Megatron arg-init stub with the verified path used by the installed Lumen stack
- [ ] Run `pytest tests/rl/verl/test_megatron_backend.py -v`
- [ ] Mark this task done only after Megatron adapter tests pass

### Task 6: Megatron Config and Smoke Path

Files:

- `examples/rl/verl/configs/grpo_megatron_actor_lumen.yaml`
- `examples/rl/verl/run_grpo_megatron.sh`
- `tests/rl/verl/test_megatron_backend.py`

Checklist:

- [ ] Add `grpo_megatron_actor_lumen.yaml`
- [ ] Keep `critic_backend: fsdp` in the Megatron-stage config
- [ ] Keep the launcher pointed at `python -m lumen.rl.verl.verl_entry`
- [ ] Run `pytest tests/rl/verl/test_megatron_backend.py -v`
- [ ] Run the manual smoke: `bash examples/rl/verl/run_grpo_megatron.sh`
- [ ] Confirm config parsing succeeds
- [ ] Confirm the Lumen-owned entrypoint primes Megatron args before provider creation
- [ ] Confirm actor/reference enter the Lumen Megatron path
- [ ] Confirm the critic remains on the HuggingFace / FSDP path
- [ ] Confirm the tiny smoke run completes without crashes

## Final Exit Gate

- [ ] `lumen/rl/verl/` exists and imports cleanly
- [ ] Backend matrix validation is present and tested
- [ ] FSDP actor/reference/critic builders are Lumen-aware
- [ ] Checkpoint normalization exists and is tested
- [ ] FSDP config and launcher exist and have been exercised
- [ ] Megatron actor/reference adapter exists and is tested
- [ ] Megatron-stage config keeps the critic on `FSDP`
- [ ] Small smoke validation has passed for both milestones

## Deferred Until Post-v1

- shared RL HuggingFace builders extracted across `TRL` and `VERL`
- `vLLM` rollout
- `SGLang` rollout
- full Megatron critic support
- async RL and larger-scale placement tuning
- `DeepSpeed` unless a real workload justifies it

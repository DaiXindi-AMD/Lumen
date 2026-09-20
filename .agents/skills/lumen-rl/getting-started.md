# Lumen RL Getting Started

The baseline and roadmap below are historical. Verify the target adapters, pinned dependencies, and test results before identifying unfinished work; user-selected scope overrides the default v1 choices.


## Start Here

If the goal is to get Lumen RL work moving quickly on AMD, do this in order:

1. Restate the current Lumen baseline:
   - real backends today are `Megatron` and `HF + FSDP/FSDP2`
   - there is no first-class `DeepSpeed` path yet
   - some distributed items are still partial, especially `fp8_param_gather`, TP overlap, and `delay_wgrad`
2. Freeze the v1 scope before discussing code:
   - algorithm: `GRPO`
   - rollout: plain HuggingFace
   - model family: LLaMA-class HuggingFace checkpoints
   - validation target: 2-GPU smoke
3. Pick the lane:
   - fastest useful PoC: `TRL + Lumen + FSDP/FSDP2`
   - long-term architecture: still start with `VERL + Lumen FSDP`, then move to `VERL + Lumen Megatron`
4. Explicitly avoid starting with:
   - `OpenRLHF`
   - `DeepSpeedChat`
   - direct `DeepSpeed` integration as a prerequisite
5. Open the matching checklist:
   - `trl-fsdp-fsdp2-checklist.md`
   - `verl-fsdp-megatron-checklist.md`
6. Do not expand scope until the base path proves:
   - fast tests pass
   - 2-GPU smoke passes
   - no NaN / Inf
   - checkpoint resume works

## Default Rule

If there is no strong workload-specific reason to do otherwise, start with `TRL`, not `VERL`, and treat `DeepSpeed` as optional follow-up work.

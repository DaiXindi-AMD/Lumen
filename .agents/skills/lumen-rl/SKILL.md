---
name: lumen-rl
description: "Use when choosing or executing Lumen RL integrations on AMD GPUs. Covers TRL-first then VERL landing path, explains why OpenRLHF and DeepSpeedChat are not first targets, provides execution checklists for TRL + Lumen + FSDP/FSDP2 and VERL + Lumen FSDP/Megatron."
---

# Lumen RL Guide

## Codex use and source

Ported from `Lumen/.claude/skills/lumen-RL/` at Lumen commit `03102ffb954b`; all companion guides are included. Repository paths below are relative to the target Lumen checkout, and commands run from its root. Read its current `CLAUDE.md` for project conventions.

Feature counts, coverage inventories, benchmark measurements, and implementation plans in this skill and its references are historical snapshots. Check the target code and hardware before treating them as current support or running a command. Apply only the requested workflow; roadmap checklists are not permission to implement unrelated milestones.


## Mission

Help choose the right RL framework for Lumen on AMD GPUs and execute the integration in the right order without drifting outside Lumen's current backend constraints.

For the most concrete day-0 startup flow, read [getting-started.md](getting-started.md) first.

## Use When

- Choosing the recommended RL landing path for Lumen on AMD GPUs
- Comparing `TRL` and `VERL`, or explaining why `OpenRLHF` and `DeepSpeedChat` are not first targets
- Estimating the difficulty of direct `DeepSpeed` support for Lumen
- Planning or implementing `lumen/rl/trl/` or `lumen/rl/verl/`
- Reviewing RL roadmap sequencing, scope boundaries, or exit criteria

When code changes are involved, pair this skill with `lumen-coding` and `lumen-test`.

## Current Lumen RL Baseline

Lumen currently has two real training backends:

- `Megatron` via `lumen/models/megatron.py`
- `HF + FSDP/FSDP2` via `lumen/models/fsdp.py`

That implies several hard constraints:

- Lumen already knows how to patch HuggingFace causal LMs with FP8, LoRA, warmup, and checkpoint behavior.
- Lumen already knows how to build Megatron-backed training flows with TP/PP/CP/SP and Lumen-specific communication hooks.
- The original roadmap predated `lumen/rl/`. The migration checkout already contains TRL and VERL adapters; inspect them and their tests to determine what remains.
- Lumen does not currently expose a native `DeepSpeed` integration path.
- Several distributed-training efficiency items are still partial or unwired for end-to-end RL, especially `fp8_param_gather`, full TP overlap, and full `delay_wgrad` wiring.

## Default Recommendation

Use this order unless the user gives a strong workload-specific reason to do otherwise:

1. `TRL + Lumen + FSDP/FSDP2`
2. `VERL + Lumen FSDP`
3. `VERL + Lumen Megatron`
4. Optional `DeepSpeed` support only if a real workload forces it

## Decision Rules

### If the user wants the fastest useful RL PoC

- Recommend `TRL`
- Freeze v1 to `GRPO`, HuggingFace rollout, LLaMA-family models, and `FSDP1/FSDP2`
- Read [trl-fsdp-fsdp2-checklist.md](trl-fsdp-fsdp2-checklist.md)

### If the user wants the best long-term architectural fit

- Recommend `VERL`, but do not start with the Megatron stage
- Land `VERL + Lumen FSDP` before `VERL + Lumen Megatron`
- Read [verl-fsdp-megatron-checklist.md](verl-fsdp-megatron-checklist.md)

### If the user asks about `OpenRLHF` or `DeepSpeedChat`

- Explain that both are poor first targets because the training system is `DeepSpeed`-centric
- Treat them as reference architectures, not first implementation targets
- Read [reference.md](reference.md)

### If the user asks about direct `DeepSpeed` support

- Explain the three ambition levels:
  - run Lumen inside ZeRO
  - usable DeepSpeed backend
  - DeepSpeed that preserves Lumen's differentiated distributed value
- Treat `DeepSpeed` as optional work, not a prerequisite for initial RL bring-up
- Read [reference.md](reference.md)

## Execution Workflow

1. Classify the request:
   - framework selection
   - TRL implementation
   - VERL implementation
   - DeepSpeed feasibility
2. Restate the current Lumen backend baseline and scope boundaries.
3. Freeze the first supported algorithm, rollout engine, and model family before proposing implementation work.
4. Use the matching checklist reference rather than inventing a fresh plan.
5. Keep verification gates explicit:
   - 2-GPU smoke
   - no NaN / Inf
   - checkpoint resume works
   - FP8 warmup / reset survives distributed startup

The v1 exclusions below are defaults for a fresh bring-up, not prohibitions on an explicitly requested workload or on maintaining an integration already present.

## What To Avoid

- Starting with `OpenRLHF`
- Starting with `DeepSpeedChat`
- Making direct `DeepSpeed` support a prerequisite for initial RL validation
- Bringing up `VERL` FSDP and `VERL` Megatron in the same first patch series
- Adding `vLLM` or `SGLang` before the basic HF rollout path is stable
- Expanding to non-LLaMA models in v1

## References

- [getting-started.md](getting-started.md) -- concrete day-0 startup flow
- [reference.md](reference.md) -- framework comparison, landing path, DeepSpeed difficulty
- [trl-fsdp-fsdp2-checklist.md](trl-fsdp-fsdp2-checklist.md) -- execution checklist for `TRL + Lumen + FSDP/FSDP2`
- [verl-fsdp-megatron-checklist.md](verl-fsdp-megatron-checklist.md) -- execution checklist for `VERL + Lumen FSDP/Megatron`

## Examples

- "Which RL framework should Lumen integrate first on AMD GPUs?"
- "Plan the first `TRL + Lumen + FSDP2` milestone."
- "What should we verify before starting `VERL + Lumen Megatron`?"
- "How hard would DeepSpeed support be for Lumen?"

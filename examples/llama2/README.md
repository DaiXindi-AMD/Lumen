# LLaMA2 SFT

Full fine-tuning or LoRA on LLaMA2 (7B / 13B / 70B) with FP8 attention, packed sequences, and early stopping.

## Quick Start

```bash
# 1. Prepare data and model checkpoint
bash examples/llama2/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama2/run_finetune.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama2/run_finetune.sh
```

The training script (`finetune_llama2.py`) selects the backend via `--backend megatron|fsdp`.

## MLPerf-Aligned Training: Llama2-70B LoRA SFT (8x MI300X)

Llama2-70B LoRA SFT with FP8 quantization, TP=1 DP=8 parallelism, aligned with the
AMD MLPerf v5.1 `MI300X_EPYC_9575F_pytorch_llama2_70b` reference submission.

**Lumen passes the MLPerf target (val_loss < 0.925)** with best val_loss = 0.9223
and pre-eval step time of **4,730 ms** (1.19x vs local MLPerf reference at 3,967 ms).

### Prerequisites

- **Docker image**: `lumen_unit_test:latest` (contains Megatron-LM-AMD, ROCm, RCCL)
- **AITER**: `lumen/triton_kernels` branch from `ZhangDanyang-AMD/aiter.git` (commit `cfaeaad3b`)
- **GPUs**: 8x AMD MI300X (192 GB HBM each)
- **Host RAM**: >= 256 GB (checkpoint loading uses mmap but still needs headroom)
- **Disk**: ~300 GB free on `/data1/` (model ~140 GB, dataset ~30 GB, checkpoints/results)
- **Model**: `NousResearch/Llama-2-70b-hf` converted to Megatron TP=1 format
- **Dataset**: `regisss/scrolls_gov_report_preprocessed_mlperf_2` preprocessed to `.npy`

### Step 1 — Download and convert model + dataset

```bash
# Download HuggingFace model
python examples/llama2/scripts/download_model.py

# Convert HF checkpoint to Megatron TP=1 format
python examples/llama2/scripts/convert_to_megatron.py \
    --hf-dir /data1/lumen/nous_llama2_70b_hf \
    --out-dir /data1/lumen/megatron_ckpt_nous_tp1 \
    --tp 1

# Download and preprocess dataset
python examples/llama2/scripts/download_dataset.py
python examples/llama2/scripts/convert_dataset.py \
    --tokenizer examples/llama2/tokenizer \
    --out-dir /data1/lumen/data \
    --seq-len 8192
```

### Step 2 — Launch training

```bash
bash examples/llama2/run_tp1_dp8.sh
```

The script applies system tunables (`runtime_tunables.sh`), launches a Docker container
with all required environment variables, applies Megatron patches, and starts training.

All speed and convergence optimizations are enabled by default (v47):
- hipBLASLt for all GEMMs with `.t()` view (`LUMEN_PREFER_HIPBLASLT=1`)
- Fused quant+amax, quant+scale, norm+quant, cast+transpose kernels
- Fused SwiGLU fwd/bwd with fused amax (`LUMEN_FUSED_SWIGLU=1`)
- Epoch-level data shuffling (`LUMEN_SHUFFLE_TRAIN=1`)
- Aligned eval schedule every 192 steps (`LUMEN_EVAL_ALIGNED=1`)
- Post-eval allocator fixes (eval recompute, warmup GC, cache clear)
- Backend caching + sync elimination (`LUMEN_SKIP_BACKEND_SYNC=1`)
- FP8 weight gradients via hipBLASLt (`FP8_WGRAD=1`)
- ACL=21 activation checkpointing (`RECOMPUTE_NUM_LAYERS=21`)

### Step 3 — Monitor training

Logs stream to stdout and are tee'd to `~/mlperf_llama2_70b.log`:

```bash
# Watch loss and grad norms
tail -f ~/mlperf_llama2_70b.log | grep -E "iteration|lm loss|grad_norm"

# Watch validation eval
tail -f ~/mlperf_llama2_70b.log | grep "validation loss"

# Quick GPU memory check
rocm-smi --showmeminfo vram

# Check GPU thermals and power (step time is sensitive to thermal throttling)
rocm-smi --showtemp && rocm-smi --showpower
```

### Script chain

```
[Host]  bash run_tp1_dp8.sh
  ├── runtime_tunables.sh (CPU perf governor, THP, cache drop, NUMA/ASLR)
  └── docker run lumen_unit_test:latest bash -c '...'
        │
        ├── pip install runtime deps (huggingface-hub, sentencepiece, peft, ...)
        ├── Fix numpy.product → numpy.prod (Megatron compat)
        │
        ├── python scripts/patch_gpt_layer_specs.py   # RMSNorm / FusedRMSNorm compat
        ├── python scripts/patch_checkpointing.py     # LoRA base_layer key remap + mmap
        ├── python scripts/patch_requires_grad.py     # Grad flow fix for LoRA + recompute
        ├── python scripts/patch_lora_scaling.py      # LoRA alpha/rank scaling fix
        ├── python scripts/patch_sft_loss_norm.py     # SFT loss normalization alignment
        │
        └── CONFIG=config_MI300X_tp1_dp8.sh bash run_finetune.sh
              └── torchrun --nproc_per_node=8 finetune_llama2.py \
                    --linear-fp8 --fp8-param-storage --lora-rank 16 ...
```

The five Megatron patches are applied at runtime because they modify the container's
Megatron-LM-AMD installation (not part of the Lumen repo). The patches are idempotent
and skip themselves if already applied.

### Key training parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Tensor Parallel | 1 | Single-GPU model, same as MLPerf reference |
| Data Parallel | 8 | One rank per GPU |
| Global Batch Size | 8 | MBS=1 x DP=8 |
| Sequence Length | 8192 | |
| Learning Rate | 4e-4 | Matches MLPerf reference |
| LR Schedule | Cosine decay | Over full 1024 steps, min\_lr=0 |
| LR Warmup Steps | 0 | Matches MLPerf reference |
| Synthetic Warmup | 5 steps | Zero loss\_mask — calibrates FP8 scales without updating LoRA weights |
| Weight Decay | 1e-4 | Matches MLPerf |
| Gradient Clip | 0.3 | Matches MLPerf |
| Adam Betas | (0.9, 0.999) | Matches MLPerf |
| LoRA | rank=16, alpha=32, dropout=0.1 | Attention-only, matches MLPerf |
| FP8 Format | E4M3 hybrid | Delayed scaling, amax history=4, most\_recent |
| FP8 Param Storage | Enabled | Weights stored in FP8 to save memory |
| FP8 Wgrad | Enabled | Weight gradients computed in FP8 via hipBLASLt |
| Activation Recompute | 21 layers (full/block) | Matches MLPerf reference |
| Distributed Optimizer | Disabled | Attention-only LoRA has small optimizer states (~540 MB/GPU) |
| Data Shuffling | Enabled | Epoch-level shuffle matching NeMo reference |
| Eval Interval | Every 192 steps | Aligned with MLPerf eval cadence |
| Seed | 1234 | Fixed for reproducibility |

### Speed optimizations

| Optimization | Env Var | Measured Savings |
|--------------|---------|-----------------|
| hipBLASLt for all GEMMs | `LUMEN_PREFER_HIPBLASLT=1` | **-790 ms/step (14.2%)** — fwd+bwd; `.t()` view eliminates weight transpose |
| Fused quant+amax | `LUMEN_FUSED_QUANT_AMAX=1` | -377 ms/step (6.1%) |
| Fused quant+scale | `LUMEN_FUSED_QUANT_SCALE=1` | -206 ms/step (2.8%) |
| Post-eval allocator fixes | `LUMEN_EVAL_RECOMPUTE=1`, `LUMEN_POST_EVAL_CACHE_CLEAR=1`, etc. | -11.1% total training time |
| Backend caching + sync elimination | `LUMEN_SKIP_BACKEND_SYNC=1` | ~1-2% step time |
| Fused SwiGLU fwd+bwd + fused amax | `LUMEN_FUSED_SWIGLU=1` | -696 ms/step from baseline; fused amax saves ~30-80 ms (v47) |
| Fused RMSNorm + FP8 quant | `LUMEN_FUSED_NORM_QUANT=1` | ~0.2% step time |
| Fused cast+transpose in backward | `LUMEN_FUSED_CAST_TRANSPOSE=1` | ~11 ms/step (consumed by hipBLASLt wgrad) |
| Wgrad `.t()` view (v47) | Code-level | ~50 ms/step — eliminates grad transpose copy |
| SwiGLU FP8 cache | `LUMEN_FUSED_SWIGLU_QUANT=1` | Saves redundant quantization |

### Data shuffling

Data shuffling is the single most important factor for reaching the MLPerf target.

The training dataset (`train.npy`) contains 3901 pre-packed samples of 8192 tokens each.
The AMD MLPerf reference (NeMo) shuffles all sample indices into a random permutation at
epoch start (seeded by the training seed). Without shuffling, consecutive mini-batches
contain adjacent packed sequences that are highly correlated, degrading early convergence.

| Setting | Best val\_loss | Passes MLPerf? |
|---------|---------------|----------------|
| `LUMEN_SHUFFLE_TRAIN=0` | 0.9371 | No |
| `LUMEN_SHUFFLE_TRAIN=1` | **0.9223** | **Yes** |

Implementation: `lumen/models/llama2/dataset.py` — `LLaMA2SFTDataset._build_samples_mapping()`
creates a permuted index array and remaps `__getitem__` through it.

### Expected results

With the default configuration (all v47 optimizations enabled):

| Metric | Lumen v47 | MLPerf Ref (local) |
|--------|-----------|-------------------|
| Initial loss (step 6, after warmup) | ~4.1 | — |
| Loss at step 100 | ~1.3 | — |
| Best validation loss | **0.9223** (step 576) | 0.9243 (step 384) |
| MLPerf target | 0.925 | 0.925 |
| Pre-eval step time | **4,730 ms** | **3,967 ms** |
| Post-eval step time | ~5,550 ms | ~4,000 ms |
| Speed ratio (Lumen / MLPerf) | **1.19x** | 1.0x |
| Peak GPU memory per device | ~189 GB / 192 GB (98.7%) | ~157 GB (~82%) |
| Stability | 0 NaN / 0 skipped | 0 NaN / 0 skipped |

**Local MLPerf reference**: `rocm/amd-mlperf:llama2_70b_training_5.1` Docker, `SEED=1234`,
same MI300X machine.

Step times are sensitive to GPU thermal state and power throttling. On MI300X at
750W TDP, sustained training reaches thermal equilibrium at 88-98C junction temperature,
which may increase step times by 5-7% compared to cold-start measurements.

See [`results/mlperf_llama2_70b_lora/`](results/mlperf_llama2_70b_lora/) for the full
comparison against the AMD MLPerf reference.

### MLPerf alignment status

| Parameter | Lumen v47 | MLPerf Reference (local) | Status |
|-----------|----------|--------------------------|--------|
| Learning Rate | 4e-4 | 4e-4 | Matched |
| LR Warmup | 0 | 0 | Matched |
| LR Schedule | Cosine, 1024 steps | Cosine, 1024 steps | Matched |
| LoRA rank/alpha | 16/32 | 16/32 | Matched |
| FP8 Format | E4M3 hybrid (delayed) | E4M3 hybrid (delayed) | Matched |
| Data Shuffling | Epoch-level | Epoch-level | Matched |
| Activation Recompute | 21 layers (full/block) | 21 layers | Matched |
| Seed | 1234 | 1234 | Matched |
| FP8 Engine | AITER (hipBLASLt + Triton) | TransformerEngine | Different (kernel impl) |
| Attention | AITER CK FMHA v3 | TE CK fused attn v3 | Different (same CK kernel) |
| RMSNorm | AITER Triton | TE Triton / apex | Different |
| Step time | **4,730 ms** | **3,967 ms** | 1.19x gap |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `SIGKILL` during checkpoint load | CPU OOM — 8 ranks each loading 128 GB | Ensure `patch_checkpointing.py` ran (adds `mmap=True`) |
| `HIP out of memory` in forward pass | Activation memory overflow | Verify `RECOMPUTE_NUM_LAYERS=21` in config |
| `grad_norm: 0.000` every step | Broken autograd chain with LoRA + recompute | Ensure `patch_requires_grad.py` ran |
| NCCL timeout on step 1 | AITER kernel tuning takes > default timeout | Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` (already in `run_tp1_dp8.sh`) |
| Loss spikes / divergence | Missing patches or incomplete env var set | Use `run_tp1_dp8.sh` as-is — all patches and env vars are required |
| `numpy.product` error on save | Deprecated numpy API in Megatron | Already patched in `run_tp1_dp8.sh` |
| val\_loss stuck at ~0.937 | Data not shuffled | Set `LUMEN_SHUFFLE_TRAIN=1` (default in `run_tp1_dp8.sh`) |
| Step time ~400ms above target | GPU thermal throttling at 750W TDP | Normal at thermal equilibrium; check `rocm-smi --showtemp` |
| AITER JIT compile hangs on first run | 8 ranks wait for rank 0 to finish JIT build | Expected — first launch takes ~5 min extra for kernel compilation |

## CLI Flags

| Feature | CLI Flag |
|---------|----------|
| Attention backend | `--lumen-attn-backend {aiter_csrc,aiter_triton,aiter_triton_fp8,aiter_csrc_fp8}` |
| FP8 quantised training | `--linear-fp8 --fp8-format e4m3` |
| MXFP8 block sizes | `--mxfp8-block-m-fwd 128 ...` (6 independent dims) |
| LoRA | `--lora-rank 16 --lora-alpha 32` |
| LoRA A2A comm opt | `--lora-a2a` |
| Synthetic warmup | `--warmup-steps 5` |
| Early stopping | `--val-loss-target 1.5` |
| Context Parallelism | `--context-parallel-size 2` |

See `run_finetune.sh` for the full list of environment variables and defaults.

## Megatron Patches

When running inside the Docker container, five patches are applied to the container's
Megatron-LM-AMD installation. These live in `scripts/` and are applied at launch:

| Patch | Purpose |
|-------|---------|
| `patch_gpt_layer_specs.py` | Creates `MegatronFusedRMSNorm` wrapper; patches `gpt_layer_specs.py` and `transformer_block.py` to use it when RMSNorm is detected |
| `patch_checkpointing.py` | Remaps checkpoint keys for LoRA `base_layer` wrapping; injects `mmap=True` into `torch.load` to prevent CPU OOM with 8 ranks loading a 128 GB checkpoint |
| `patch_requires_grad.py` | Forces `hidden_states.requires_grad_(True)` before `_checkpointed_forward` so LoRA gradients flow through activation checkpointing |
| `patch_lora_scaling.py` | Fixes LoRA alpha/rank scaling to match the MLPerf reference implementation |
| `patch_sft_loss_norm.py` | Aligns SFT loss normalization with the MLPerf reference (per-sample vs per-token) |

All patches are idempotent and skip themselves if already applied.

## Reference Logs

See [`results/`](results/) for full training logs from LLaMA2-70B SFT runs on MI300X/MI355X GPUs across different quantization configurations and MLPerf comparisons.

## Megatron Pretraining (Llama2-7B, 8×MI325X)

A self-contained pretraining launcher that runs Llama2-7B in either BF16 or FP8
delayed/hybrid, using the Megatron backend with all validated Lumen fusion
optimizations enabled. It generates mock data internally, so no dataset
download is required for a quick perf/functional run.

### Launch

```bash
# FP8 delayed/hybrid (default) — 50 steps on 8 GPUs
bash examples/llama2/run_pretrain_llama2_7b.sh

# BF16
PRECISION=bf16 bash examples/llama2/run_pretrain_llama2_7b.sh

# Override common knobs (defaults shown)
PRECISION=fp8 MBS=4 GBS=256 SEQ_LEN=4096 TRAIN_STEPS=50 \
    bash examples/llama2/run_pretrain_llama2_7b.sh
```

### Environment overrides

| Variable | Default | Notes |
|----------|---------|-------|
| `PRECISION` | `fp8` | `fp8` (delayed/hybrid) or `bf16` |
| `IMAGE` | `lumen:dev` | Docker image |
| `MBS` / `GBS` | `4` / `256` | micro / global batch size |
| `SEQ_LEN` | `4096` | sequence length |
| `TRAIN_STEPS` | `50` | training iterations |
| `TOKENIZER_DIR` | `examples/llama2/tokenizer` | HuggingFace tokenizer dir |
| `RESULTS_DIR` | `examples/llama2/results` | logs + mock data output |

FP8 mode enables the validated forward optimizations (fused quant+amax, fused
cast-transpose, fused SwiGLU/norm quant, transpose cache, weight-quant-once,
etc.) plus `--linear-fp8 --fp8-format hybrid --linear-fp8-scaling delayed`.
BF16 keeps only the precision-agnostic fusions (SwiGLU, residual-norm).

### Model config

32 layers · hidden 4096 · FFN 11008 · 32 heads (no GQA) · RoPE base 1e4 · RMSNorm · SwiGLU.

### Reference results (50 steps, steady-state step time)

| Precision | Log | step time |
|-----------|-----|-----------|
| BF16 | [`results/llama2_7b_pretrain_bf16.log`](results/llama2_7b_pretrain_bf16.log) | ~11.6 s |
| FP8 delayed | [`results/llama2_7b_pretrain_fp8_delayed.log`](results/llama2_7b_pretrain_fp8_delayed.log) | ~8.0 s |

### Lumen vs TransformerEngine (8×MI325X)

Same config (MBS=4, GBS=256, seq=4096, TP=1, PP=1), steady-state step time and
peak allocated memory:

| Precision | Framework | step time | peak memory |
|-----------|-----------|-----------|-------------|
| BF16 | TransformerEngine | ~11.24 s | 110.9 GB |
| BF16 | Lumen | ~11.52 s | 112.6 GB |
| BF16 | Δ | +2.5% | +1.5% |
| FP8 delayed | TransformerEngine | ~8.14 s | 111.6 GB |
| FP8 delayed | Lumen | ~8.19 s | 113.5 GB |
| FP8 delayed | Δ | +0.5% | +1.7% |

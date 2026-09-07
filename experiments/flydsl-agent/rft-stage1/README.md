# RFT Stage A — Diversity-Preserving Rejection Fine-Tuning

Based on the SFT v5f model: large-scale candidate kernel generation → FlyDSL-Gym sandbox verification (compile + run + correctness) → retain all compilation-passing implementations → 1 epoch training.

## Results

| Metric | v5e | v5f SFT | **RFT v5f** | Target |
|------|-----|---------|-------------|--------|
| Overall API Score | 74% | 74% | **75%** | >60% |
| L5 (Expert) | 50% | 50% | **57%** | >20% |
| Format Compliance | — | 96% | **96%** | >90% |
| Sandbox Compile (122 specs) | 22% | 38% | **53%** | — |

**Three-level verification** (1952 candidates from RFT model):

| Level | Passed | Rate |
|-------|--------|------|
| Compilation | 944 | 48% |
| Runtime | 211 | 11% |
| Correctness | 0 | 0% |

Correctness 0% indicates kernel internal compute logic is wrong — this is the target of RL Stage B/C.

HuggingFace: [Zhangdanyang/Qwen2.5-Coder-RFT-v5f](https://huggingface.co/Zhangdanyang/Qwen2.5-Coder-RFT-v5f)

See [REPORT.md](REPORT.md) for detailed analysis.

## Running

### Prerequisites

- SFT v5f model at `/home/danyzhan/sft-results/Qwen2.5-Coder-SFT-v5f`
- FlyDSL-Gym sandbox image `flydsl-gym:latest`
- Docker image `lumen/flydsl-cpt:latest`
- RL specs at `/home/danyzhan/flydsl-agent-dataset/data/rl/train-00000-of-00001.jsonl`
- 8x AMD MI350X GPUs

### Fully Automated Pipeline

One command runs all 7 steps (generate → verify → build data → train → export → evaluate):

```bash
cd /home/danyzhan/Lumen/experiments/flydsl-agent/rft-stage1
bash run_rft.sh
```

Total time ~26h (generate 23h + verify 0.5h + train 2h + export+eval 1h).

Customizable via environment variables:

```bash
SFT_MODEL=/path/to/model \
MAX_SPECS=50 \
N_CANDIDATES=8 \
HARDWARE=gfx942 \
bash run_rft.sh
```

### Step-by-Step Execution

#### Step 1: Candidate Generation (~23h)

```bash
docker run --rm --init \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    -v experiments:/workspace/experiments \
    -v /path/to/sft-model:/model:ro \
    -v /path/to/dataset:/data:ro \
    -v /path/to/rft-results:/results \
    lumen/flydsl-cpt:latest \
    python3 /workspace/experiments/flydsl-agent/rft-stage1/generate_candidates.py \
        --model /model \
        --specs /data/data/rl/train-00000-of-00001.jsonl \
        --output /results/candidates_v5f_gfx950.jsonl \
        --n-candidates 16 --max-specs 213 --hardware gfx950 --device cuda:0
```

| Parameter | Value | Description |
|------|-----|------|
| `--max-specs` | 213 | All gfx950 specs (122 actually sampled) |
| `--n-candidates` | 16 | 16 candidates per spec |
| `--hardware` | gfx950 | Target hardware |
| temperature | 0.8 | Encourage diversity |
| 3 prompt styles | precise / natural / optimization | Rotated |

#### Step 2-3: Sandbox Verification (~30min)

Three-level verification: static check → Docker sandbox compile → runtime execution + correctness check

```bash
python3 verify_candidates.py \
    --input /path/to/candidates_v5f_gfx950.jsonl \
    --output /path/to/verified_v5f_gfx950.jsonl \
    --metadata /path/to/verify_stats_v5f_gfx950.json \
    --use-sandbox
```

Verification stats output (`verify_stats_*.json`) includes:
- `passed_static` — syntax + FlyDSL pattern + ≥15 lines
- `passed_sandbox` — FlyDSL JIT compile pass
- `passed_runtime` — entry point invocation success
- `passed_correctness` — output matches PyTorch reference
- `by_operator` — per-operator breakdown

#### Step 4: Build RFT Dataset

```bash
python3 build_rft_dataset.py \
    --verified /path/to/verified_v5f_gfx950.jsonl \
    --sft-data /path/to/format_aligned/train.jsonl \
    --output /path/to/rft_v5f_train.jsonl \
    --rft-repeat 2
```

Key design: **Diversity-preserving** — retain all compilation-passing implementations, no top-K selection.

#### Step 5: RFT Training (~2h)

1 epoch training on v5f merged model. Uses `config_rft.sh` hyperparameters.

| Parameter | Value |
|------|-----|
| Base model | Qwen2.5-Coder-SFT-v5f (merged) |
| LR | 5e-6 |
| Epochs | 1 |
| LoRA | r=64, alpha=128, dropout=0.05 |

#### Step 6-7: Export + Evaluation

Automatically export HF model and run benchmark (API Score + format compliance + sandbox compilation).

## File Inventory

```
rft-stage1/
├── README.md                  # This file
├── REPORT.md                  # Detailed experiment report
├── generate_candidates.py     # Candidate generator (N=16, 3 prompt styles)
├── verify_candidates.py       # Static + sandbox compile + runtime + correctness verification
├── build_rft_dataset.py       # verified → ChatML + merge
├── config_rft.sh              # Training hyperparameters (lr=5e-6, 1 epoch)
└── run_rft.sh                 # Fully automated 7-step pipeline
```

## Key Data Paths

| Data | Path |
|------|------|
| v5f candidates (122 specs) | `rft-results/candidates_v5f_gfx950.jsonl` |
| RFT candidates (122 specs, RFT model) | `rft-results/candidates_rft_v5f_gfx950.jsonl` |
| Sandbox verification stats (compile level) | `rft-results/verify_stats_rft_v5f_gfx950.json` |
| Three-level verification stats (incl. runtime + correctness) | `rft-results/verify_stats_rft_v5f_runtime.json` |
| RFT training data | `rft-results/rft_v5f_train.jsonl` (8,275 samples) |
| RFT model (merged) | `rft-results/Qwen2.5-Coder-RFT-v5f` |
| Benchmark | `rft-results/benchmark_rft_v5f.json` |

## Sandbox Verifier (`sandbox/verify.py`)

Three-level verification:

1. **Compile** — `importlib.util.spec_from_file_location` + `exec_module` triggers FlyDSL JIT
2. **Run** — detect `@flyc.jit` / `launch_*` / `forward` and other entry points, construct operator-specific inputs, invoke kernel
3. **Correctness** — compare output to PyTorch reference (`torch.allclose`)

Supports input construction and reference computation for 12 operator types: gemm, softmax, rmsnorm, layernorm, rope, topk, quant, flash_attn, moe, mla, paged_attn, custom.

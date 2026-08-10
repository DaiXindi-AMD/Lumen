#!/bin/bash
###############################################################################
# Qwen3-8B pretrain body, shared by the docker and native launchers.
#
# Everything is read from the environment so the two launchers differ only in
# how they set up paths. Required:
#
#   LUMEN_ROOT      Lumen checkout
#   MEGATRON_ROOT   Megatron-LM checkout (needed on PYTHONPATH and for patching)
#   RESULTS_ROOT    writable dir for logs, mock data and the autotune cache
#   TOKENIZER_PATH  HuggingFace tokenizer directory
#
# Optional: PRECISION NUM_LAYERS TAIL_BF16 MBS GBS SEQ_LEN TRAIN_STEPS SEED NPROC
#           TRAIN_JSONL VALID_JSONL LOG_TAG EVAL_INTERVAL EVAL_ITERS SPLIT
#           WANDB_PROJECT WANDB_NAME WANDB_ENTITY
###############################################################################
set -euo pipefail

: "${LUMEN_ROOT:?}"
: "${MEGATRON_ROOT:?}"
: "${RESULTS_ROOT:?}"
: "${TOKENIZER_PATH:?}"

PRECISION="${PRECISION:-mxfp4}"
NUM_LAYERS="${NUM_LAYERS:-36}"
TAIL_BF16="${TAIL_BF16:-5}"
MBS="${MBS:-2}"
GBS="${GBS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
NPROC="${NPROC:-8}"

# Evaluate ~10 times over the run instead of once at the end: a single end-of-run
# eval gives wandb one data point per val metric, which it can only render as a bar,
# and one point cannot show where the curves separate.
EVAL_INTERVAL="${EVAL_INTERVAL:-$(( TRAIN_STEPS / 10 > 0 ? TRAIN_STEPS / 10 : 1 ))}"
# Each eval pass consumes EVAL_ITERS × GBS sequences from VALID_JSONL, capped by
# however many that file holds. Keep it small: eval time is pure overhead, and the
# eval batch must stay identical across runs for the comparison to be paired.
EVAL_ITERS="${EVAL_ITERS:-2}"
# Kept only because Megatron requires it; PretrainTextDataset ignores it.
SPLIT="${SPLIT:-98,1,1}"

DATA_DIR="${RESULTS_ROOT}/mock_data"
# Point TRAIN_JSONL at a real corpus (one {"text": ...} object per line) to skip
# the mock generator below. LOG_TAG then keeps the two logs apart.
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/mock_train_s${SEQ_LEN}_g${GBS}_t${TRAIN_STEPS}.jsonl}"
# PretrainTextDataset builds each of train/valid/test from its own path starting at
# chunk 0 and ignores --split entirely, so leaving VALID_JSONL at the train file
# makes "validation loss" a measurement on data the model also trains on. Point it
# at a disjoint file to get a real held-out number.
VALID_JSONL="${VALID_JSONL:-${TRAIN_JSONL}}"
LOG_FILE="${RESULTS_ROOT}/lumen_qwen3_8b${LOG_TAG:+_${LOG_TAG}}_${PRECISION}.log"
mkdir -p "${DATA_DIR}"

# Megatron builds its wandb writer only when --wandb-project is non-empty, and it
# writes from the last rank rather than rank 0. Without WANDB_PROJECT the metrics
# exist only in LOG_FILE; examples/qwen3/scripts/wandb_backfill_megatron_log.py
# can replay one of those logs after the fact.
#
# --tensorboard-dir is not optional here. Every wandb_writer.log() call in
# Megatron's training_log() sits inside `if writer and ...`, where `writer` is
# the *TensorBoard* writer, which exists only when --tensorboard-dir is set. Ask
# for wandb without it and the run appears in the project, connects, and uploads
# nothing but a runtime -- an empty workspace with no way to tell it apart from a
# run that never started.
WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME:-megatron-${PRECISION}-8b${LOG_TAG:+-${LOG_TAG}}-${TRAIN_STEPS}}"
        --wandb-save-dir "${RESULTS_ROOT}/wandb"
        --tensorboard-dir "${RESULTS_ROOT}/tensorboard/${WANDB_NAME:-${PRECISION}${LOG_TAG:+-${LOG_TAG}}}"
    )
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

export PYTHONPATH="${MEGATRON_ROOT}:${LUMEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import megatron" || { echo "ERROR: megatron not importable"; exit 1; }

# Lumen vendors Triton kernels under third_party/aiter that the installed aiter
# does not ship. They have to be copied over the installed tree rather than put
# on PYTHONPATH: the submodule has no compiled extensions for hipbsolgemm (every
# hipBLASLt GEMM) or the a4w4 GEMMs (all of MXFP4), so importing aiter from there
# trades a missing Triton file for a missing .so.
# `|| true` so a failed import reaches the message below instead of tripping
# `set -e` on the command substitution's exit status.
AITER_DIR="$(python -c 'import aiter,os;print(os.path.dirname(aiter.__file__))' 2>/dev/null || true)"
[ -n "${AITER_DIR}" ] || { echo "ERROR: aiter not importable"; exit 1; }

# Missing these surfaces as a ModuleNotFoundError from deep inside the
# lumen.models.megatron import chain.
AITER_REQUIRED=(
    ops/triton/cross_entropy.py
    ops/triton/_triton_kernels/cross_entropy.py
)
# Missing these still trains, which is exactly the problem. fast_transpose_fp8
# falls back to .t().contiguous() and warns per call: the first FP8 baseline ran
# that way and paid 166 ms/step plus 709k warnings into a 78 MB log, with nothing
# in the run to say so. The quant kernels are reached only by the blockwise and
# mxfp8 scaling types.
AITER_OPTIONAL=(
    ops/triton/quant/fast_transpose.py
    ops/triton/_triton_kernels/quant/fast_transpose.py
    ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py
    ops/triton/_triton_kernels/quant/quant_mxfp8.py
)

aiter_missing() {
    local rel
    for rel in "$@"; do
        [ -f "${AITER_DIR}/${rel}" ] || echo "${rel}"
    done
}
aiter_copy_hint() {
    local rel
    for rel in "$@"; do
        echo "  cp ${LUMEN_ROOT}/third_party/aiter/aiter/${rel} ${AITER_DIR}/${rel}"
    done
}

mapfile -t MISSING_REQUIRED < <(aiter_missing "${AITER_REQUIRED[@]}")
if [ ${#MISSING_REQUIRED[@]} -gt 0 ]; then
    echo "ERROR: the installed aiter (${AITER_DIR}) is missing kernels Lumen vendors:"
    aiter_copy_hint "${MISSING_REQUIRED[@]}"
    exit 1
fi

mapfile -t MISSING_OPTIONAL < <(aiter_missing "${AITER_OPTIONAL[@]}")
if [ ${#MISSING_OPTIONAL[@]} -gt 0 ]; then
    echo "[setup] WARNING: the installed aiter (${AITER_DIR}) is missing vendored kernels."
    echo "[setup] Training will run, but FP8 transposes fall back to .t().contiguous()"
    echo "[setup] and any step time measured here is not comparable to a complete run:"
    aiter_copy_hint "${MISSING_OPTIONAL[@]}"
fi

# The RMSNorm layer-spec patch rewrites Megatron to use apex's FusedRMSNorm, so
# it only applies where apex exists. Without apex Megatron already falls back to
# WrappedTorchNorm, and applying the patch would inject an unsatisfiable
# `import apex` into the RMSNorm path.
if python -c "import apex" 2>/dev/null; then
    python "${LUMEN_ROOT}/examples/llama2/scripts/patch_gpt_layer_specs.py" "${MEGATRON_ROOT}"
    ROPE_ARGS=()
else
    echo "[setup] apex not installed: skipping FusedRMSNorm patch, disabling rope fusion"
    # Megatron defaults apply_rope_fusion=True and asserts on a fused kernel that
    # only TE or Lumen's apex bridge can supply. Note this flag does not reach the
    # Lumen backend, which overwrites apply_rope_fusion from --lumen-fused-rope.
    ROPE_ARGS=(--no-rope-fusion)
fi

# The Lumen backend ignores both branches above -- it sets apply_rope_fusion from
# its own flag, which routes to AITER's fused kernel instead of apex's or TE's.
# Without it the rotate-half runs as separate neg/cat/mul kernels: 137 ms/step at
# this shape, measured (§5.7).
ROPE_ARGS+=(--lumen-fused-rope)

# Mock corpus of random token ids. Repetitive enough that the model memorises it
# within ~50 steps, which is what makes the loss curve usable as a regression
# signal. Generation is the same as the BF16/FP8 reference runs so the curves
# stay comparable; it is skipped when the file is already there.
if [ ! -s "${TRAIN_JSONL}" ]; then
    SEQ_LEN="${SEQ_LEN}" GBS="${GBS}" TRAIN_STEPS="${TRAIN_STEPS}" SEED="${SEED}" \
    TRAIN_JSONL="${TRAIN_JSONL}" python - <<'PYEOF'
import json
import os
import random

seq = int(os.environ["SEQ_LEN"])
gbs = int(os.environ["GBS"])
steps = int(os.environ["TRAIN_STEPS"])
need_chunks = gbs * (steps + 5)
need_tokens = int(need_chunks * (seq + 1) * 1.2)
random.seed(int(os.environ["SEED"]))
path = os.environ["TRAIN_JSONL"]
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, 151999)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] wrote {docs} docs to {path}")
PYEOF
else
    echo "[data] using existing ${TRAIN_JSONL}"
fi

# ---- MXFP4-only flags -------------------------------------------------------
# MXFP4 is selected with --linear-fp8-format because Megatron's own --fp8-format
# has no MX formats among its choices. The last TAIL_BF16 layers stay BF16: both
# FP4 papers find the tail layers are the sensitive ones, and 8B diverged around
# step 1300 without this (docs/mxfp4_training_report.md §1.5, §6.3).
#
# Do NOT add --lumen-linear. It swaps in LumenColumnParallelLinear, which
# enable_fp8_for_parallel_linear configures from the --linear-fp8-scaling string
# ("blockwise") rather than the resolved recipe ("mxfp4"), so the run would
# silently execute FP8 blockwise instead. Without the flag Megatron's
# ColumnParallelLinear / RowParallelLinear are patched by quant.enable(), which
# does route MXFP4 correctly.
QUANT_ARGS=()
if [ "${PRECISION}" = "mxfp4" ]; then
    QUANT_ARGS=(
        --linear-fp8
        --linear-fp8-format mxfp4
        --linear-fp8-scaling blockwise
        --linear-fp8-block-size 32
        --first-last-layers-bf16
        --num-layers-at-start-in-bf16 0
        --num-layers-at-end-in-bf16 "${TAIL_BF16}"
    )
fi

# AITER can only reach its prebuilt A4W4 asm kernels for shapes listed in the
# tuned table, and Qwen3-8B's fused qkv/gate_up shapes at this token count are
# absent from the stock one: 8 of the 11 MXFP4 GEMMs fell back to Triton, worth
# ~132 ms/step (docs/mxfp4_training_report.md §5.5). The extra rows were produced
# by scripts/mxfp4_tune_shapes.py and each is bit-exact against Triton. They key
# on the exact M/N/K, so they only fire at MBS x SEQ_LEN = 16384 with TP=1.
# Setting the variable turns off AITER's own config discovery, hence relisting
# the stock tables it would otherwise have merged.
if [ "${PRECISION}" = "mxfp4" ] && [ -z "${AITER_CONFIG_GEMM_A4W4:-}" ]; then
    AITER_CFG_DIR="$(python -c 'import aiter, os; print(os.path.join(os.path.dirname(aiter.__file__), "configs"))')"
    A4W4_PATHS="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv")"
    A4W4_PATHS="${A4W4_PATHS}:${AITER_CFG_DIR}/a4w4_blockscale_tuned_gemm.csv"
    for f in "${AITER_CFG_DIR}"/model_configs/*a4w4_blockscale_tuned_gemm.csv; do
        if [ -e "$f" ]; then A4W4_PATHS="${A4W4_PATHS}:${f}"; fi
    done
    export AITER_CONFIG_GEMM_A4W4="${A4W4_PATHS}"
    echo "[setup] AITER_CONFIG_GEMM_A4W4=${AITER_CONFIG_GEMM_A4W4}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/qwen3_8b_common_args.sh"

# Word-split on purpose: EXTRA_ARGS carries whole flags, e.g. profiler options.
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

cd "${LUMEN_ROOT}/examples/llama31"
set -x
torchrun --nproc_per_node="${NPROC}" --nnodes=1 pretrain_llama31.py \
    --backend megatron \
    "${COMMON_ARGS[@]}" \
    --lumen-attn-backend csrc \
    "${ROPE_ARGS[@]}" \
    "${QUANT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EXTRA_ARGS_ARR[@]}" \
    2>&1 | tee "${LOG_FILE}"

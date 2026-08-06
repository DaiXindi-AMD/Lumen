#!/bin/bash
###############################################################################
# Qwen3-8B pretrain on stock Megatron + TransformerEngine — the control arm for
# the Lumen MXFP4 comparison. No Lumen model code runs here; see
# scripts/pretrain_qwen3_te.py for why Lumen's entry point cannot serve as the
# control and how the data pipeline is kept identical anyway.
#
# Every hyperparameter that is not under test comes from
# qwen3_8b_common_args.sh, shared with train_qwen3_8b.sh.
#
# Required:
#   LUMEN_ROOT      Lumen checkout (only for the dataset class + this script)
#   MEGATRON_ROOT   Megatron checkout on a branch that has megatron/core/fp4_utils.py
#                   (ROCm rocm_dev; core_r0.15.0_rocm does not)
#   RESULTS_ROOT    writable dir for logs
#   TOKENIZER_PATH  HuggingFace tokenizer directory
#
# Optional: PRECISION(mxfp4|bf16) NUM_LAYERS MBS GBS SEQ_LEN TRAIN_STEPS SEED
#           NPROC TRAIN_JSONL VALID_JSONL LOG_TAG EVAL_INTERVAL EVAL_ITERS SPLIT
#           WANDB_PROJECT WANDB_NAME WANDB_ENTITY TAIL_BF16 HADAMARD
###############################################################################
set -euo pipefail

: "${LUMEN_ROOT:?}"
: "${MEGATRON_ROOT:?}"
: "${RESULTS_ROOT:?}"
: "${TOKENIZER_PATH:?}"

PRECISION="${PRECISION:-mxfp4}"
NUM_LAYERS="${NUM_LAYERS:-36}"
MBS="${MBS:-2}"
GBS="${GBS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
NPROC="${NPROC:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-$(( TRAIN_STEPS / 10 > 0 ? TRAIN_STEPS / 10 : 1 ))}"
EVAL_ITERS="${EVAL_ITERS:-2}"
SPLIT="${SPLIT:-98,1,1}"
# Both default to TE's stock recipe, so an unset environment reproduces the
# original control arm. Raise them to align with Lumen's recipe instead.
TAIL_BF16="${TAIL_BF16:-0}"
HADAMARD="${HADAMARD:-0}"

: "${TRAIN_JSONL:?TRAIN_JSONL is required: this launcher has no mock-data generator}"
VALID_JSONL="${VALID_JSONL:-${TRAIN_JSONL}}"
LOG_FILE="${RESULTS_ROOT}/te_qwen3_8b${LOG_TAG:+_${LOG_TAG}}_${PRECISION}.log"

WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME:-te-${PRECISION}-8b${LOG_TAG:+-${LOG_TAG}}-${TRAIN_STEPS}}"
        --wandb-save-dir "${RESULTS_ROOT}/wandb"
        # Required, not decorative: Megatron gates every wandb write on the
        # TensorBoard writer existing. Without this the run uploads nothing.
        --tensorboard-dir "${RESULTS_ROOT}/tensorboard/${WANDB_NAME:-te-${PRECISION}${LOG_TAG:+-${LOG_TAG}}}"
    )
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

# Deliberately NOT putting LUMEN_ROOT on PYTHONPATH: pretrain_qwen3_te.py reads
# LUMEN_ROOT and loads the one dataset file by path, so nothing else of Lumen can
# be imported by accident.
export PYTHONPATH="${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LUMEN_ROOT

# Importing TE succeeds but tearing it down does not: on torch 2.13 the exit-time
# torch.library cleanup raises, and the native libraries abort on heap corruption
# while unloading. That happens after everything has been flushed, so it only
# costs us the exit code — which is why these checks look for a sentinel on stdout
# instead of trusting the status.
te_check="$(python -c "import transformer_engine, transformer_engine.pytorch; print('TE_OK')" 2>/dev/null || true)"
case "${te_check}" in
    *TE_OK*) ;;
    *) echo "ERROR: TransformerEngine not importable — this launcher is the TE arm"; exit 1 ;;
esac
fp4_check="$(python -c "import megatron.core.fp4_utils; print('FP4_OK')" 2>/dev/null || true)"
case "${fp4_check}" in
    *FP4_OK*) ;;
    *)
        echo "ERROR: ${MEGATRON_ROOT} has no usable megatron/core/fp4_utils.py."
        echo "The FP4 recipe plumbing landed on ROCm's rocm_dev branch; core_r0.15.0_rocm predates it."
        exit 1
        ;;
esac

QUANT_ARGS=()
if [ "${PRECISION}" = "mxfp4" ]; then
    QUANT_ARGS=(--fp4-format e2m1 --fp4-recipe mxfp4)
    # Megatron defaults --num-layers-at-start-in-bf16 to 1, so a tail-only
    # request has to pass the start count explicitly or layer 0 also drops out
    # of FP4. get_fp4_context() honours these for both init and forward.
    if [ "${TAIL_BF16}" -gt 0 ]; then
        QUANT_ARGS+=(
            --first-last-layers-bf16
            --num-layers-at-start-in-bf16 0
            --num-layers-at-end-in-bf16 "${TAIL_BF16}"
        )
    fi
    # MXFP4BlockScaling reads this at construction (recipe/__init__.py), and it
    # is a fixed 16-point Hadamard fused into the cast kernel. Note the scope
    # differs from Lumen's: TE rotates every quantizer's tensor (fprop
    # activation and weight, and the backward operands), whereas Lumen rotates
    # only the two WGrad operands. Turning this on does not reproduce Lumen's
    # recipe, it applies a strictly wider rotation.
    export NVTE_MXFP4_USE_HADAMARD="${HADAMARD}"
    # The Triton cast path silently ignores use_hadamard.
    export NVTE_USE_CAST_TRANSPOSE_TRITON=0
fi

source "$(dirname "${BASH_SOURCE[0]}")/qwen3_8b_common_args.sh"

# Word-split on purpose: EXTRA_ARGS carries whole flags, e.g. profiler options.
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

cd "${MEGATRON_ROOT}"
set -x
torchrun --nproc_per_node="${NPROC}" --nnodes=1 \
    "${LUMEN_ROOT}/examples/qwen3/scripts/pretrain_qwen3_te.py" \
    --transformer-impl transformer_engine \
    "${COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EXTRA_ARGS_ARR[@]}" \
    2>&1 | tee "${LOG_FILE}"

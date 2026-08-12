#!/bin/bash
###############################################################################
# Run the BF16 / FP8-delayed / MXFP4 precision matrix over the benchmark models.
#
# Usage:
#   bash examples/scripts/run_precision_matrix.sh                  # full matrix
#   MODELS=qwen3_8b PRECISIONS="bf16 mxfp4" bash ...               # a slice
#   TRAIN_STEPS=3 LOG_TAG=smoke bash ...                           # smoke pass
#   REPEAT_TAG=rep2 MODELS=... PRECISIONS=mxfp4 bash ...           # noise floor
#
# Runs are strictly sequential: each one wants all 8 GPUs, and two at once would
# make every number in the matrix wrong rather than just slow. Cells whose log
# already exists are skipped, so the script is restartable after a failure — pass
# FORCE=1 to re-measure one anyway.
#
# Every cell goes through examples/scripts/train_pretrain.sh with the same
# TRAIN_STEPS, SEED and mock corpus; only the quantization recipe differs.
###############################################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EXAMPLES_DIR="${LUMEN_DIR}/examples"
MEGATRON_ROOT="${MEGATRON_ROOT:-${HOME}/Megatron-LM}"

MODELS="${MODELS:-llama2_7b llama31_8b qwen3_8b}"
PRECISIONS="${PRECISIONS:-bf16 fp8 mxfp4}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
NPROC="${NPROC:-8}"
# Named separately from LOG_TAG so a repeat keeps the tag that says what it is
# while still landing beside the arm it repeats.
REPEAT_TAG="${REPEAT_TAG:-}"
SUMMARY="${SUMMARY:-${EXAMPLES_DIR}/precision_matrix_runs.tsv}"

[ -d "${MEGATRON_ROOT}/megatron" ] || {
    echo "ERROR: no Megatron checkout at MEGATRON_ROOT=${MEGATRON_ROOT}"
    exit 1
}

model_dir() {
    case "$1" in
        llama2_7b) echo llama2 ;;
        llama31_8b) echo llama31 ;;
        qwen3_8b) echo qwen3 ;;
        *) echo "ERROR: unknown model $1" >&2; exit 1 ;;
    esac
}

[ -f "${SUMMARY}" ] || printf 'model\tprecision\ttag\tstatus\tseconds\tlog\n' > "${SUMMARY}"

for model in ${MODELS}; do
    mdir="$(model_dir "${model}")"
    results="${EXAMPLES_DIR}/${mdir}/results"
    tokenizer="${EXAMPLES_DIR}/${mdir}/tokenizer"
    mkdir -p "${results}"

    for precision in ${PRECISIONS}; do
        tag="${REPEAT_TAG:-${LOG_TAG:-}}"
        log="${results}/lumen_${model}${tag:+_${tag}}_${precision}.log"

        if [ -s "${log}" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "[skip] ${model} ${precision}${tag:+ (${tag})}: ${log} exists"
            continue
        fi

        echo ""
        echo "==============================================================="
        echo "[run] ${model} ${precision}${tag:+ (${tag})} — ${TRAIN_STEPS} steps, $(date '+%H:%M:%S')"
        echo "==============================================================="
        start=$(date +%s)

        # Shared runtime env. Every entry is a ROCm/RCCL tuning knob or a Lumen
        # fusion switch that applies to all three precisions; the FP8-only and
        # MXFP4-only switches live in train_pretrain.sh so a precision cannot be
        # measured with the wrong set.
        #
        # The MXFP4 autotune cache defaults to one file per model, which is what a
        # repeat run wants: the per-shape backend choice is pinned, so the repeat
        # measures noise and not a different set of kernels. Any run that changes
        # which kernels are *reachable* -- adding or removing a tuned GEMM table --
        # must not reuse it, or it inherits the choices made when the other table
        # was in place and the change appears to do nothing.
        #
        # CACHE_TAG suffixes the per-model name rather than replacing the path, so
        # one invocation can re-probe every model in MODELS. Setting
        # LUMEN_MXFP4_AUTOTUNE_CACHE directly still works but names a single file,
        # which silently makes several models share one cache. Set CACHE_TAG to the
        # run's tag and collect_precision_matrix.py finds the cache on its own.
        env \
            HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            TOKENIZERS_PARALLELISM=false \
            HSA_NO_SCRATCH_RECLAIM=1 \
            HIP_FORCE_DEV_KERNARG=1 \
            GPU_MAX_HW_QUEUES=8 \
            NCCL_IB_DISABLE=1 \
            NCCL_SOCKET_IFNAME=lo \
            NCCL_DEBUG=WARN \
            CUDA_DEVICE_MAX_CONNECTIONS=8 \
            OMP_NUM_THREADS=1 \
            TORCHDYNAMO_DISABLE=1 \
            USE_HIPBLASLT=1 \
            TORCH_BLAS_PREFER_HIPBLASLT=1 \
            PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
            LUMEN_FUSED_SWIGLU=1 \
            LUMEN_FUSED_RESIDUAL_NORM=1 \
            LUMEN_FUSED_RES_BWD=1 \
            LUMEN_SKIP_BACKEND_SYNC=1 \
            LUMEN_MXFP4_AUTOTUNE_CACHE="${LUMEN_MXFP4_AUTOTUNE_CACHE:-${results}/mxfp4_autotune_${model}${CACHE_TAG:+_${CACHE_TAG}}.json}" \
            LUMEN_ROOT="${LUMEN_DIR}" \
            MEGATRON_ROOT="${MEGATRON_ROOT}" \
            RESULTS_ROOT="${results}" \
            TOKENIZER_PATH="${tokenizer}" \
            MODEL="${model}" \
            PRECISION="${precision}" \
            TRAIN_STEPS="${TRAIN_STEPS}" \
            SEED="${SEED}" \
            NPROC="${NPROC}" \
            ${tag:+LOG_TAG="${tag}"} \
            ${FP8_FORMAT:+FP8_FORMAT="${FP8_FORMAT}"} \
            ${FP8_LUMEN_LINEAR:+FP8_LUMEN_LINEAR="${FP8_LUMEN_LINEAR}"} \
            ${TRAIN_JSONL:+TRAIN_JSONL="${TRAIN_JSONL}"} \
            ${VALID_JSONL:+VALID_JSONL="${VALID_JSONL}"} \
            ${WANDB_PROJECT:+WANDB_PROJECT="${WANDB_PROJECT}"} \
            ${WANDB_ENTITY:+WANDB_ENTITY="${WANDB_ENTITY}"} \
            ${EVAL_INTERVAL:+EVAL_INTERVAL="${EVAL_INTERVAL}"} \
            ${EVAL_ITERS:+EVAL_ITERS="${EVAL_ITERS}"} \
            ${MBS:+MBS="${MBS}"} \
            ${GBS:+GBS="${GBS}"} \
            ${SEQ_LEN:+SEQ_LEN="${SEQ_LEN}"} \
            ${TAIL_BF16:+TAIL_BF16="${TAIL_BF16}"} \
            ${DRY_RUN:+DRY_RUN="${DRY_RUN}"} \
            bash "${SCRIPT_DIR}/train_pretrain.sh"
        rc=$?
        elapsed=$(( $(date +%s) - start ))

        # A cell that dies must not stop the matrix: one broken recipe is a result,
        # and the remaining arms are still worth having.
        status=ok
        [ ${rc} -eq 0 ] || status="FAILED(rc=${rc})"
        # A dry run prints the command and exits, so recording it would put a row
        # in the index that looks like a completed cell but has no log behind it.
        if [ -n "${DRY_RUN:-}" ]; then
            echo "[dry-run] ${model} ${precision}: not recorded in ${SUMMARY}"
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${model}" "${precision}" "${tag:--}" "${status}" "${elapsed}" "${log}" \
            >> "${SUMMARY}"
        echo "[done] ${model} ${precision} ${status} in ${elapsed}s"
    done
done

echo ""
echo "[matrix] run index: ${SUMMARY}"

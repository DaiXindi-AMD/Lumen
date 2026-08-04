#!/bin/bash
###############################################################################
# Lumen — Qwen3-8B MXFP4 pretrain on the Megatron backend
#
# Usage:
#   bash run_pretrain_qwen3_8b_mxfp4.sh                    # in the lumen:dev image
#   LAUNCH=native bash run_pretrain_qwen3_8b_mxfp4.sh      # on the host
#   PRECISION=bf16 LAUNCH=native bash ...                  # reference run
#
# The training command itself lives in scripts/train_qwen3_8b.sh; this file only
# sets up paths and environment for one of the two launch modes.
#
# Native mode needs a Megatron-LM checkout (MEGATRON_ROOT) and works without
# TransformerEngine or apex — Megatron falls back to torch norms and optimizers.
# Rope fusion still runs: the body asks for AITER's kernel, which needs neither.
#
# Differences from run_pretrain_qwen3_8b.sh (BF16 / FP8 delayed):
#   * MXFP4 is selected with --linear-fp8-format, not Megatron's --fp8-format.
#   * Last 5 of 36 layers stay BF16 (docs/mxfp4_training_report.md §1.5, §6.3).
#   * --qk-layernorm is on, matching HF Qwen3's per-head q_norm/k_norm. The
#     BF16/FP8 script leaves it off, so its loss curve is not comparable here.
#
# Override any of MBS / GBS / SEQ_LEN / TRAIN_STEPS / SEED / NPROC via env.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

LAUNCH="${LAUNCH:-docker}"             # docker | native
PRECISION="${PRECISION:-mxfp4}"        # mxfp4 | bf16
IMAGE="${IMAGE:-lumen:dev}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${SCRIPT_DIR}/tokenizer}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_qwen3_8b_${PRECISION}}"

CFG_ENV=(
    PRECISION="${PRECISION}"
    NUM_LAYERS="${NUM_LAYERS:-36}"
    TAIL_BF16="${TAIL_BF16:-5}"
    MBS="${MBS:-2}"
    GBS="${GBS:-128}"
    SEQ_LEN="${SEQ_LEN:-8192}"
    TRAIN_STEPS="${TRAIN_STEPS:-50}"
    SEED="${SEED:-1234}"
    NPROC="${NPROC:-8}"
)

# Shared by both modes. Everything here is either a ROCm/RCCL tuning knob or a
# Lumen fusion switch; none of them change numerics.
RUNTIME_ENV=(
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    HSA_NO_SCRATCH_RECLAIM=1
    HIP_FORCE_DEV_KERNARG=1
    GPU_MAX_HW_QUEUES=8
    NCCL_IB_DISABLE=1
    NCCL_SOCKET_IFNAME=lo
    NCCL_DEBUG=WARN
    CUDA_DEVICE_MAX_CONNECTIONS=8
    OMP_NUM_THREADS=1
    TORCHDYNAMO_DISABLE=1
    USE_HIPBLASLT=1
    TORCH_BLAS_PREFER_HIPBLASLT=1
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    LUMEN_FUSED_SWIGLU=1
    LUMEN_FUSED_RESIDUAL_NORM=1
    LUMEN_FUSED_RES_BWD=1
    LUMEN_SKIP_BACKEND_SYNC=1
)

mkdir -p "${RESULTS_DIR}"

if [ "${LAUNCH}" = "native" ]; then
    MEGATRON_ROOT="${MEGATRON_ROOT:-${HOME}/Megatron-LM}"
    [ -d "${MEGATRON_ROOT}/megatron" ] || {
        echo "ERROR: no Megatron checkout at MEGATRON_ROOT=${MEGATRON_ROOT}"
        echo "  git clone --depth 1 https://github.com/ROCm/Megatron-LM.git ${MEGATRON_ROOT}"
        exit 1
    }

    # MXFP4-only: the tuned A4W4 table widens which shapes reach AITER's prebuilt
    # ASM kernels and the autotune cache makes the per-shape backend choice
    # reproducible across processes. Every MXFP4 backend is bit-identical, so both
    # affect speed only (docs/mxfp4_training_report.md §2.2).
    if [ "${PRECISION}" = "mxfp4" ]; then
        RUNTIME_ENV+=(
            AITER_CONFIG_GEMM_A4W4="${SCRIPT_DIR}/configs/a4w4_blockscale_tuned_gemm.csv"
            LUMEN_MXFP4_AUTOTUNE_CACHE="${RESULTS_DIR}/mxfp4_autotune_qwen3_8b.json"
            LUMEN_FAST_QUANT_DISPATCH=1
        )
    fi

    exec env "${RUNTIME_ENV[@]}" "${CFG_ENV[@]}" \
        LUMEN_ROOT="${LUMEN_DIR}" \
        MEGATRON_ROOT="${MEGATRON_ROOT}" \
        RESULTS_ROOT="${RESULTS_DIR}" \
        TOKENIZER_PATH="${TOKENIZER_DIR}" \
        bash "${SCRIPT_DIR}/scripts/train_qwen3_8b.sh"
fi

# ---- docker mode ------------------------------------------------------------
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

if [ "${PRECISION}" = "mxfp4" ]; then
    RUNTIME_ENV+=(
        AITER_CONFIG_GEMM_A4W4=/workspace/Lumen/examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv
        LUMEN_MXFP4_AUTOTUNE_CACHE=/results/mxfp4_autotune_qwen3_8b.json
        LUMEN_FAST_QUANT_DISPATCH=1
    )
fi

DOCKER_ENV=()
for kv in "${RUNTIME_ENV[@]}" "${CFG_ENV[@]}"; do
    DOCKER_ENV+=(-e "${kv}")
done

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size 16G \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    -v "${TOKENIZER_DIR}:/tokenizer:ro" \
    -v "${RESULTS_DIR}:/results" \
    "${DOCKER_ENV[@]}" \
    -e LUMEN_ROOT=/workspace/Lumen \
    -e MEGATRON_ROOT="${MEGATRON_ROOT:-/workspace/megatron_lm}" \
    -e RESULTS_ROOT=/results \
    -e TOKENIZER_PATH=/tokenizer \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/qwen3/scripts/train_qwen3_8b.sh

echo ""
echo "[DONE] log: ${RESULTS_DIR}/lumen_qwen3_8b_${PRECISION}.log"

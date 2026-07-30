#!/usr/bin/env bash
# run_dsv4_pretrain.sh — DSV4 4-layer native Megatron pretrain smoke (no Miles train.py / Ray).
#
# Usage:
#   bash examples/dsv4/run_dsv4_pretrain.sh
#   TRAIN_ITERS=10 IMAGE=lumen/dsv4-lumen:mi308x bash examples/dsv4/run_dsv4_pretrain.sh
#
# Optional checkpoint prep uses Miles convert scripts only (prepare_dsv4_checkpoint.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MILES_DIR="${MILES_DIR:-/home/leiwu/miles}"
TILEKERNELS_DIR="${TILEKERNELS_DIR:-/home/leiwu/TileKernels}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/mnt/data/leiwu/lumen-dsv4-bootstrap}"
IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
MODEL_DIR="${MODEL_DIR:-/mnt/data/leiwu/models}"
LOG_DIR="${LOG_DIR:-/mnt/data/leiwu/logs}"
TVM_CACHE_DIR="${TVM_CACHE_DIR:-/mnt/data/leiwu/tvm-cache}"
LOGFILE="${LOG_DIR}/lumen_dsv4_pretrain_$(date +%Y%m%d_%H%M%S).log"

MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-FP8-4layer}"
DSV4_HC_MULT="${DSV4_HC_MULT:-2}"
# shellcheck source=examples/dsv4/dsv4_4layer_megatron_args.sh
source "${SCRIPT_DIR}/dsv4_4layer_megatron_args.sh"

TORCH_DIST="${MODEL_DIR}/${MODEL_NAME}_torch_dist_hc${DSV4_HC_MULT}"

V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND:-triton}"
MHC_BACKEND="${MHC_BACKEND:-triton}"
V4_INDEXER_IMPL="${V4_INDEXER_IMPL:-tilelang}"
V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N:-64}"
V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES:-1}"
LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8:-0}"
LUMEN_DSV4_FP8_SCALING="${LUMEN_DSV4_FP8_SCALING:-blockwise}"
LUMEN_DSV4_MOE_MORI="${LUMEN_DSV4_MOE_MORI:-0}"
MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
LOAD_CKPT="${LOAD_CKPT:-0}"
EVAL_ITERS="${EVAL_ITERS:-1}"

USE_BOOTSTRAP=0
BOOTSTRAP_MOUNT="${BOOTSTRAP_DIR}"
if [[ "${IMAGE}" == "lumen/tests:latest" || "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
    USE_BOOTSTRAP=1
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        BOOTSTRAP_MOUNT=""
    elif [[ ! -f "${BOOTSTRAP_DIR}/.ready" ]]; then
        echo "[prepare] bootstrap missing — running prepare_bootstrap.sh"
        bash "${SCRIPT_DIR}/prepare_bootstrap.sh"
    fi
fi

if ! docker image inspect "${IMAGE}" &>/dev/null; then
    if [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
        bash "${SCRIPT_DIR}/build_dsv4_lumen_image.sh"
    else
        echo "[ERROR] Image not found: ${IMAGE}"
        exit 1
    fi
fi

if ! ls /dev/kfd &>/dev/null; then
    echo "[ERROR] /dev/kfd not found — ROCm device not accessible."
    exit 1
fi

mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${MODEL_DIR}/miopen-cache" "${TVM_CACHE_DIR}"

echo "════════════════════════════════════════════════"
echo "  Lumen DSV4 4-layer Megatron pretrain smoke"
echo "  Image     : ${IMAGE}"
echo "  Bootstrap : $([[ ${USE_BOOTSTRAP} -eq 1 ]] && echo yes || echo no)"
echo "  Steps     : ${TRAIN_ITERS}"
echo "  Batch     : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
echo "  HC mult   : ${DSV4_HC_MULT} (MHC_BACKEND=${MHC_BACKEND})"
echo "  FP8       : $([[ ${LUMEN_DSV4_LINEAR_FP8} -eq 1 ]] && echo "Lumen (${LUMEN_DSV4_FP8_SCALING})" || echo BF16)"
echo "  MoE EP    : $([[ ${LUMEN_DSV4_MOE_MORI} -eq 1 ]] && echo "MORI" || echo NCCL alltoall)"
echo "  Ckpt      : ${TORCH_DIST}"
echo "  Log       : ${LOGFILE}"
echo "════════════════════════════════════════════════"

DOCKER_MOUNTS=(
    -v "${LUMEN_DIR}:/workspace/Lumen"
    -v "${MODEL_DIR}:/root/models"
    -v "${MODEL_DIR}/miopen-cache:/root/.config/miopen"
    -v "${TVM_CACHE_DIR}:/root/.cache/tvm-ffi"
)
if [[ -d "${MILES_DIR}" ]]; then
    DOCKER_MOUNTS+=(-v "${MILES_DIR}:/workspace/miles")
fi
if [[ -d "${TILEKERNELS_DIR}" ]]; then
    DOCKER_MOUNTS+=(-v "${TILEKERNELS_DIR}:/workspace/TileKernels")
fi
if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
    DOCKER_MOUNTS+=(-v "${BOOTSTRAP_MOUNT}:/bootstrap:ro")
fi

DOCKER_ENV=(
    -e LUMEN_DIR=/workspace/Lumen
    -e LUMEN_DSV4_PRETRAIN=1
    -e MODEL_DIR=/root/models
    -e MODEL_NAME="${MODEL_NAME}"
    -e TRAIN_ITERS="${TRAIN_ITERS}"
    -e GBS="${GBS}"
    -e MBS="${MBS}"
    -e SEQ_LEN="${SEQ_LEN}"
    -e SKIP_PREPARE="${SKIP_PREPARE}"
    -e LOAD_CKPT="${LOAD_CKPT}"
    -e EVAL_ITERS="${EVAL_ITERS}"
    -e DSV4_HC_MULT="${DSV4_HC_MULT}"
    -e V4_SPARSE_MLA_BACKEND="${V4_SPARSE_MLA_BACKEND}"
    -e MHC_BACKEND="${MHC_BACKEND}"
    -e V4_INDEXER_IMPL="${V4_INDEXER_IMPL}"
    -e V4_INDEXER_BLOCK_N="${V4_INDEXER_BLOCK_N}"
    -e V4_INDEXER_NUM_STAGES="${V4_INDEXER_NUM_STAGES}"
    -e LUMEN_DSV4_LINEAR_FP8="${LUMEN_DSV4_LINEAR_FP8}"
    -e LUMEN_DSV4_FP8_SCALING="${LUMEN_DSV4_FP8_SCALING}"
    -e LUMEN_DSV4_MOE_MORI="${LUMEN_DSV4_MOE_MORI}"
    -e DSV4_ENABLE_RECOMPUTE="${DSV4_ENABLE_RECOMPUTE:-1}"
)
if [[ -d "${TILEKERNELS_DIR}" ]]; then
    DOCKER_ENV+=(-e TILEKERNELS_DIR=/workspace/TileKernels)
fi
if [[ -d "${MILES_DIR}" ]]; then
    DOCKER_ENV+=(-e MILES_DIR=/workspace/miles)
fi
if [[ "${MORI_ENABLE_SDMA}" == "1" ]]; then
    DOCKER_ENV+=(-e MORI_ENABLE_SDMA=1)
fi
DOCKER_ENV+=(
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
    -e NCCL_NVLS_ENABLE=0
    -e RCCL_MSCCL_ENABLE=0
    -e HSA_FORCE_FINE_GRAIN_PCIE=1
    -e TORCHDYNAMO_DISABLE=1
)
if [[ "${USE_BOOTSTRAP}" -eq 1 && -n "${BOOTSTRAP_MOUNT}" ]]; then
    DOCKER_ENV+=(-e BOOTSTRAP_DIR=/bootstrap)
elif [[ "${IMAGE}" == "lumen/dsv4-lumen:mi308x" ]]; then
    DOCKER_ENV+=(-e BOOTSTRAP_DIR=/opt/dsv4-bootstrap -e WRITABLE_ROOT=/opt/dsv4-runtime)
fi

docker rm -f lumen-dsv4-pretrain 2>/dev/null || true

docker run --rm \
    --name lumen-dsv4-pretrain \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --network=host \
    --shm-size=128g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    "${DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    "${IMAGE}" \
    bash /workspace/Lumen/examples/dsv4/run_dsv4_pretrain_inner.sh \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Log saved to: ${LOGFILE}"

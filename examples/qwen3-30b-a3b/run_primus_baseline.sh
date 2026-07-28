#!/bin/bash
# Qwen3-30B-A3B Primus baseline — full-param BF16 training.
# Hyperparams aligned with Miles RL framework defaults:
#   lr=1e-6, constant lr, weight_decay=0.1, adam_beta2=0.98
# Produces loss logs for comparison with Lumen FSDP EP=8 runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PRIMUS_IMAGE="${PRIMUS_IMAGE:-rocm/primus:v26.4}"
CONTAINER_NAME="primus_qwen3_30b_baseline"

# ---- Miles-aligned training hyperparams ----
LR="${LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
ADAM_EPS="${ADAM_EPS:-1e-5}"

mkdir -p "${RESULTS_DIR}"

echo "=== Primus Baseline: Full-Param BF16 ==="
echo "  Image:        ${PRIMUS_IMAGE}"
echo "  lr:           ${LR}"
echo "  lr_decay:     ${LR_DECAY_STYLE}"
echo "  weight_decay: ${WEIGHT_DECAY}"
echo "  adam_beta2:   ${ADAM_BETA2}"
echo ""

# Pull image if not present
if ! docker image inspect "${PRIMUS_IMAGE}" &>/dev/null; then
    echo "> Pulling ${PRIMUS_IMAGE} ..."
    docker pull "${PRIMUS_IMAGE}"
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# --- BF16 Full-Param Baseline ---
echo "> Running BF16 full-param baseline ..."
docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --network host --ipc host \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v /mnt/raid0:/mnt/raid0 \
    -v "${RESULTS_DIR}:/results" \
    --shm-size 64G \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1 \
    -e NVTE_CK_IS_V3_ATOMIC_FP32=1 \
    "${PRIMUS_IMAGE}" \
    bash -c '
set -euo pipefail
echo "--- Checking available Qwen3-30B configs ---"
ls examples/megatron/configs/MI300X/ 2>/dev/null | grep -i qwen3_30B || \
    echo "WARNING: No qwen3_30B config found in MI300X dir, listing all:"
ls examples/megatron/configs/MI300X/ 2>/dev/null

CONFIG="examples/megatron/configs/MI300X/qwen3_30B_A3B-BF16-pretrain.yaml"
if [[ -f "${CONFIG}" ]]; then
    echo "--- Running BF16 full-param training ---"
    bash runner/primus-cli direct \
        --log_file /results/primus_bf16_baseline.log \
        -- train pretrain --config "${CONFIG}" \
        --override lr='"${LR}"' \
        --override lr_decay_style='"${LR_DECAY_STYLE}"' \
        --override weight_decay='"${WEIGHT_DECAY}"' \
        --override adam_beta1='"${ADAM_BETA1}"' \
        --override adam_beta2='"${ADAM_BETA2}"' \
        --override adam_eps='"${ADAM_EPS}"'
    echo "--- BF16 full-param baseline done ---"
    grep -E "step.*loss|iteration" /results/primus_bf16_baseline.log | tail -20 \
        > /results/primus_bf16_baseline_loss.txt
else
    echo "ERROR: Config not found: ${CONFIG}"
    echo "Available configs:"
    find examples/megatron/configs -name "*qwen3*" -o -name "*30B*" 2>/dev/null
    exit 1
fi
'

echo ""
echo "=== Primus BF16 full-param baseline complete ==="
echo "  Log:  ${RESULTS_DIR}/primus_bf16_baseline.log"
echo "  Loss: ${RESULTS_DIR}/primus_bf16_baseline_loss.txt"

#!/usr/bin/env bash
# Run llama2-7b / llama31-8b / qwen3-8b Megatron pretrain smoke (FP8 blockwise, 2 steps).
#
# Usage:
#   MODEL=llama2 bash examples/run_fp8_blockwise_pretrain_2step.sh
#   MODEL=all  bash examples/run_fp8_blockwise_pretrain_2step.sh

set -euo pipefail

LUMEN_DIR="${LUMEN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-zhangdanyangamd/lumen:dev-20260818}"
MODEL="${MODEL:-all}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
MBS="${MBS:-1}"
GBS="${GBS:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
LOG_DIR="${LOG_DIR:-/data/leiwu/logs/pretrain_smoke_fp8_blockwise}"
mkdir -p "${LOG_DIR}"

# Patch FP8_ARGS in official scripts: delayed -> blockwise
fp8_blockwise_sed='s/--linear-fp8-scaling delayed/--linear-fp8-scaling blockwise --linear-fp8-block-size 128/; s/--linear-fp8-amax-history 1024/--linear-fp8-amax-history 4/; s/--lr-warmup-iters 2/--lr-warmup-iters 1/'

run_one() {
    local name="$1" script="$2"
    local log="${LOG_DIR}/${name}_fp8_blockwise_${TRAIN_STEPS}step_$(date +%Y%m%d_%H%M%S).log"
    local tmp_script
    tmp_script="$(mktemp /tmp/lumen_pretrain_XXXXXX.sh)"
    sed "${fp8_blockwise_sed}" "${script}" > "${tmp_script}"
    chmod +x "${tmp_script}"
    echo "=== ${name} → ${log}"
    IMAGE="${IMAGE}" LUMEN_DIR="${LUMEN_DIR}" TRAIN_STEPS="${TRAIN_STEPS}" MBS="${MBS}" GBS="${GBS}" SEQ_LEN="${SEQ_LEN}" PRECISION=fp8 \
        bash "${tmp_script}" 2>&1 | tee "${log}"
    rm -f "${tmp_script}"
    if grep -qE 'iteration\s+2\s|train iteration 2|iteration:.*2/' "${log}" 2>/dev/null || \
       grep -q 'training ... iteration.*2' "${log}" 2>/dev/null || \
       grep -q '\[done\]' "${log}" 2>/dev/null; then
        echo "[OK] ${name} completed"
    elif grep -q 'ChildFailedError\|Traceback\|ERROR' "${log}"; then
        echo "[FAIL] ${name} — see ${log}"
        return 1
    else
        echo "[?] ${name} finished — check ${log}"
    fi
}

case "${MODEL}" in
    llama2)  run_one llama2-7b  "${LUMEN_DIR}/examples/llama2/run_pretrain_llama2_7b.sh" ;;
    llama31) run_one llama31-8b "${LUMEN_DIR}/examples/llama31/run_pretrain_llama31_8b.sh" ;;
    qwen3)   run_one qwen3-8b   "${LUMEN_DIR}/examples/qwen3/run_pretrain_qwen3_8b.sh" ;;
    all)
        run_one llama2-7b  "${LUMEN_DIR}/examples/llama2/run_pretrain_llama2_7b.sh"
        run_one llama31-8b "${LUMEN_DIR}/examples/llama31/run_pretrain_llama31_8b.sh"
        run_one qwen3-8b   "${LUMEN_DIR}/examples/qwen3/run_pretrain_qwen3_8b.sh"
        ;;
    *) echo "Unknown MODEL=${MODEL}"; exit 1 ;;
esac

echo "[DONE] logs in ${LOG_DIR}"

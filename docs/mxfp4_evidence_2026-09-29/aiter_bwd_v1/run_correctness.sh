#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/xdai/profile-results/lumen-mxfp4-aiter-bwd-fusion-fresh-20260928-v1
readonly AITER=/home/xdai/aiter
readonly LOCK=/home/xdai/profile-results/.lumen-gpu-exclusive.lock
readonly GPUAGENT_SHA=5bc2a7d45f2fd992fcf3f53e12b5437a9b3fc7878c653a37ac647c380fb0e913

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

record_idle() {
    local output=$1 proc_dir pid comm exe cmdline owner ppid cgroup busy=0
    : > "${output}"
    date --iso-8601=seconds >> "${output}"
    for proc_dir in /sys/class/kfd/kfd/proc/[0-9]*; do
        [[ -e "${proc_dir}" ]] || continue
        pid=${proc_dir##*/}
        [[ -d "/proc/${pid}" ]] || { echo "unreadable pid=${pid}" >> "${output}"; busy=1; continue; }
        comm=$(cat "/proc/${pid}/comm" 2>/dev/null || true)
        exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
        cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
        owner=$(stat -c %U "/proc/${pid}" 2>/dev/null || true)
        ppid=$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)
        cgroup=$(cat "/proc/${pid}/cgroup" 2>/dev/null || true)
        if [[ "${comm}" == gpuagent && "${cmdline% }" == /usr/local/bin/gpuagent \
              && "${owner}" == root && "${ppid}" == 1 \
              && "${cgroup}" == "0::/system.slice/gpuagent.service" \
              && "$(sha256sum /usr/local/bin/gpuagent | cut -d' ' -f1)" == "${GPUAGENT_SHA}" \
              && ( -z "${exe}" || "${exe}" == /usr/local/bin/gpuagent ) ]]; then
            echo "allowed_service pid=${pid}" >> "${output}"
        else
            printf 'busy pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}" >> "${output}"
            busy=1
        fi
    done
    if pgrep -af '[t]orchrun|[t]rain_qwen3_fsdp\.py|[r]ocprof(v3)?|[n]sys profile' >> "${output}"; then
        busy=1
    fi
    (( busy == 0 )) || return 1
    echo status=idle >> "${output}"
}

[[ -r /dev/kfd && -w /dev/kfd ]] || fail "/dev/kfd is not readable and writable"
[[ -r /sys/class/kfd/kfd/proc ]] || fail "KFD process table is unavailable"
[[ "$(< /proc/sys/kernel/numa_balancing)" == 0 ]] || fail "NUMA balancing must be disabled"

exec {gpu_lock_fd}>"${LOCK}"
flock -n "${gpu_lock_fd}" || fail "GPU lock is held"
record_idle "${ROOT}/kfd-before.txt" || fail "GPU is busy before correctness"

mkdir -p "${ROOT}/cache/triton" "${ROOT}/cache/xdg" "${ROOT}/cache/inductor" "${ROOT}/tmp"
git -C "${AITER}" branch --show-current > "${ROOT}/aiter-branch.txt"
git -C "${AITER}" rev-parse HEAD > "${ROOT}/aiter-head.txt"
git -C "${AITER}" status --porcelain=v1 > "${ROOT}/aiter-status-before.txt"
git -C "${AITER}" diff --binary | sha256sum > "${ROOT}/aiter-tracked-diff-before.sha256"
sha256sum \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/quant.py" \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-DUAL-LAYOUT-MXFP4.json" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-FUSED-SWIGLU-BWD-DUAL-LAYOUT-MXFP4.json" \
    "${AITER}/op_tests/triton_tests/quant/test_fused_swiglu_bwd_dual_layout_mxfp4.py" \
    > "${ROOT}/source-manifest-before.sha256"

set +e
env \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${AITER}" \
    HIP_VISIBLE_DEVICES=0 \
    ROCR_VISIBLE_DEVICES=0 \
    CUDA_VISIBLE_DEVICES=0 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    AITER_REBUILD=0 \
    AITER_TUNE_GEMM=0 \
    AITER_ONLINE_TUNE=0 \
    TRITON_CACHE_DIR="${ROOT}/cache/triton" \
    XDG_CACHE_HOME="${ROOT}/cache/xdg" \
    TORCHINDUCTOR_CACHE_DIR="${ROOT}/cache/inductor" \
    TMPDIR="${ROOT}/tmp" \
    numactl --cpunodebind=0 --membind=0 \
    /usr/bin/python3 -m pytest -q \
    op_tests/triton_tests/quant/test_fused_swiglu_bwd_dual_layout_mxfp4.py \
    -vv 2>&1 | tee "${ROOT}/correctness.log"
status=${PIPESTATUS[0]}
set -e
echo "${status}" > "${ROOT}/correctness-exit-status.txt"

git -C "${AITER}" status --porcelain=v1 > "${ROOT}/aiter-status-after.txt"
git -C "${AITER}" diff --binary | sha256sum > "${ROOT}/aiter-tracked-diff-after.sha256"
sha256sum \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/quant.py" \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-DUAL-LAYOUT-MXFP4.json" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-FUSED-SWIGLU-BWD-DUAL-LAYOUT-MXFP4.json" \
    "${AITER}/op_tests/triton_tests/quant/test_fused_swiglu_bwd_dual_layout_mxfp4.py" \
    > "${ROOT}/source-manifest-after.sha256"
cmp "${ROOT}/aiter-tracked-diff-before.sha256" "${ROOT}/aiter-tracked-diff-after.sha256" || fail "tracked diff changed"
cmp "${ROOT}/source-manifest-before.sha256" "${ROOT}/source-manifest-after.sha256" || fail "source manifest changed"
record_idle "${ROOT}/kfd-after.txt" || fail "GPU is busy after correctness"
exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/xdai/profile-results/lumen-mxfp4-aiter-bwd-diagnostic-fresh-20260928-v2-CwA4L
readonly AITER=/home/xdai/aiter
readonly LUMEN=/home/xdai/Lumen
readonly LOCK=/home/xdai/profile-results/.lumen-gpu-exclusive.lock
readonly GPUAGENT_SHA=5bc2a7d45f2fd992fcf3f53e12b5437a9b3fc7878c653a37ac647c380fb0e913

scan_idle_once() {
    local output=$1 proc_dir pid comm exe cmdline owner ppid cgroup busy=0 transient=0
    : > "${output}"
    for proc_dir in /sys/class/kfd/kfd/proc/[0-9]*; do
        [[ -e "${proc_dir}" ]] || continue
        pid=${proc_dir##*/}
        if [[ ! -d "/proc/${pid}" ]]; then
            echo "transient_disappeared pid=${pid}" >> "${output}"
            transient=1
            continue
        fi
        comm=$(cat "/proc/${pid}/comm" 2>/dev/null || true)
        exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
        cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
        owner=$(stat -c %U "/proc/${pid}" 2>/dev/null || true)
        ppid=$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)
        cgroup=$(cat "/proc/${pid}/cgroup" 2>/dev/null || true)
        if [[ ! -d "/proc/${pid}" || -z "${comm}" || -z "${owner}" ]]; then
            echo "transient_unreadable pid=${pid}" >> "${output}"
            transient=1
        elif [[ "${comm}" == gpuagent && "${cmdline% }" == /usr/local/bin/gpuagent \
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
    (( transient == 0 )) || return 2
    return 0
}

record_idle() {
    local output=$1 attempt tmp status
    for attempt in $(seq 1 10); do
        tmp="${output}.attempt-${attempt}.tmp"
        set +e
        scan_idle_once "${tmp}"
        status=$?
        set -e
        if (( status == 0 )); then
            {
                date --iso-8601=seconds
                echo "attempt=${attempt}"
                cat "${tmp}"
                echo status=idle
            } > "${output}"
            rm -f "${tmp}"
            return 0
        fi
        if (( status == 1 )); then
            {
                date --iso-8601=seconds
                echo "attempt=${attempt}"
                cat "${tmp}"
                echo status=busy
            } > "${output}"
            rm -f "${tmp}"
            return 1
        fi
        rm -f "${tmp}"
        sleep 0.2
    done
    {
        date --iso-8601=seconds
        echo "attempts=10"
        echo status=unstable_kfd_table
    } > "${output}"
    return 1
}

[[ -r /dev/kfd && -w /dev/kfd ]] || exit 2
[[ -r /sys/class/kfd/kfd/proc ]] || exit 3
[[ "$(< /proc/sys/kernel/numa_balancing)" == 0 ]] || exit 4
exec {gpu_lock_fd}>"${LOCK}"
flock -n "${gpu_lock_fd}" || exit 5
record_idle "${ROOT}/kfd-before.txt" || exit 6

mkdir -p "${ROOT}/cache/triton" "${ROOT}/cache/xdg" "${ROOT}/cache/inductor" "${ROOT}/tmp"
git -C "${AITER}" branch --show-current > "${ROOT}/aiter-branch.txt"
git -C "${AITER}" rev-parse HEAD > "${ROOT}/aiter-head.txt"
git -C "${AITER}" status --porcelain=v1 > "${ROOT}/aiter-status-before.txt"
git -C "${AITER}" diff --binary | sha256sum > "${ROOT}/aiter-tracked-diff-before.sha256"
git -C "${LUMEN}" branch --show-current > "${ROOT}/lumen-branch.txt"
git -C "${LUMEN}" rev-parse HEAD > "${ROOT}/lumen-head.txt"
git -C "${LUMEN}" status --porcelain=v1 > "${ROOT}/lumen-status-before.txt"
git -C "${LUMEN}" diff --binary | sha256sum > "${ROOT}/lumen-tracked-diff-before.sha256"
sha256sum \
    "${ROOT}/diagnose_outputs.py" \
    "${ROOT}/run_diagnostic.sh" \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/quant.py" \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-DUAL-LAYOUT-MXFP4.json" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-FUSED-SWIGLU-BWD-DUAL-LAYOUT-MXFP4.json" \
    "${LUMEN}/lumen/ops/quantize/ops.py" \
    "${LUMEN}/lumen/kernels/mxfp4.py" \
    > "${ROOT}/source-manifest-before.sha256"
/usr/bin/rocm-smi --showuse --showmemuse --showpids > "${ROOT}/rocm-smi-before.txt"

set +e
env \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${LUMEN}:${AITER}" \
    HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
    HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1 \
    AITER_REBUILD=0 AITER_TUNE_GEMM=0 AITER_ONLINE_TUNE=0 \
    TRITON_CACHE_DIR="${ROOT}/cache/triton" \
    XDG_CACHE_HOME="${ROOT}/cache/xdg" \
    TORCHINDUCTOR_CACHE_DIR="${ROOT}/cache/inductor" \
    TMPDIR="${ROOT}/tmp" \
    numactl --cpunodebind=0 --membind=0 \
    /usr/bin/python3 "${ROOT}/diagnose_outputs.py" 2>&1 | tee "${ROOT}/diagnostic.log"
status=${PIPESTATUS[0]}
set -e
echo "${status}" > "${ROOT}/diagnostic-exit-status.txt"

/usr/bin/rocm-smi --showuse --showmemuse --showpids > "${ROOT}/rocm-smi-after.txt"
git -C "${AITER}" status --porcelain=v1 > "${ROOT}/aiter-status-after.txt"
git -C "${AITER}" diff --binary | sha256sum > "${ROOT}/aiter-tracked-diff-after.sha256"
git -C "${LUMEN}" status --porcelain=v1 > "${ROOT}/lumen-status-after.txt"
git -C "${LUMEN}" diff --binary | sha256sum > "${ROOT}/lumen-tracked-diff-after.sha256"
sha256sum \
    "${ROOT}/diagnose_outputs.py" \
    "${ROOT}/run_diagnostic.sh" \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/quant.py" \
    "${AITER}/aiter/ops/triton/_triton_kernels/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-DUAL-LAYOUT-MXFP4.json" \
    "${AITER}/aiter/ops/triton/configs/quant/gfx950-FUSED-SWIGLU-BWD-DUAL-LAYOUT-MXFP4.json" \
    "${LUMEN}/lumen/ops/quantize/ops.py" \
    "${LUMEN}/lumen/kernels/mxfp4.py" \
    > "${ROOT}/source-manifest-after.sha256"
cmp "${ROOT}/aiter-tracked-diff-before.sha256" "${ROOT}/aiter-tracked-diff-after.sha256"
cmp "${ROOT}/lumen-tracked-diff-before.sha256" "${ROOT}/lumen-tracked-diff-after.sha256"
cmp "${ROOT}/source-manifest-before.sha256" "${ROOT}/source-manifest-after.sha256"
record_idle "${ROOT}/kfd-after.txt" || exit 7
exit "${status}"

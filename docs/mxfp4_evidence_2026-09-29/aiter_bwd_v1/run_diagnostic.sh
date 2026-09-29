#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/xdai/profile-results/lumen-mxfp4-aiter-bwd-fusion-fresh-20260928-v1
readonly LOCK=/home/xdai/profile-results/.lumen-gpu-exclusive.lock
readonly GPUAGENT_SHA=5bc2a7d45f2fd992fcf3f53e12b5437a9b3fc7878c653a37ac647c380fb0e913

idle() {
    local output=$1 proc_dir pid comm cmdline owner ppid cgroup busy=0
    : > "${output}"
    date --iso-8601=seconds >> "${output}"
    for proc_dir in /sys/class/kfd/kfd/proc/[0-9]*; do
        [[ -e "${proc_dir}" ]] || continue
        pid=${proc_dir##*/}
        comm=$(cat "/proc/${pid}/comm" 2>/dev/null || true)
        cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
        owner=$(stat -c %U "/proc/${pid}" 2>/dev/null || true)
        ppid=$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)
        cgroup=$(cat "/proc/${pid}/cgroup" 2>/dev/null || true)
        if [[ "${comm}" == gpuagent && "${cmdline% }" == /usr/local/bin/gpuagent \
              && "${owner}" == root && "${ppid}" == 1 \
              && "${cgroup}" == "0::/system.slice/gpuagent.service" \
              && "$(sha256sum /usr/local/bin/gpuagent | cut -d' ' -f1)" == "${GPUAGENT_SHA}" ]]; then
            echo "allowed_service pid=${pid}" >> "${output}"
        else
            echo "busy pid=${pid} comm=${comm} cmdline=${cmdline}" >> "${output}"
            busy=1
        fi
    done
    if pgrep -af '[t]orchrun|[t]rain_qwen3_fsdp\.py|[r]ocprof(v3)?|[n]sys profile' >> "${output}"; then
        busy=1
    fi
    (( busy == 0 )) || return 1
    echo status=idle >> "${output}"
}

[[ -r /dev/kfd && -w /dev/kfd ]] || exit 2
[[ "$(< /proc/sys/kernel/numa_balancing)" == 0 ]] || exit 3
exec {gpu_lock_fd}>"${LOCK}"
flock -n "${gpu_lock_fd}" || exit 4
idle "${ROOT}/kfd-diagnostic-before.txt" || exit 5

set +e
env \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/home/xdai/Lumen:/home/xdai/aiter \
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
idle "${ROOT}/kfd-diagnostic-after.txt" || exit 6
exit "${status}"

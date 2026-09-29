#!/usr/bin/env bash
set -euo pipefail

case_label="${1:?usage: run_direct_case.sh CASE_LABEL PRECISION [steps]}"
precision="${2:?usage: run_direct_case.sh CASE_LABEL PRECISION [steps]}"
steps="${3:-24}"
if [[ "${precision}" != "bf16" && "${precision}" != "mxfp4" ]]; then
    echo "precision must be bf16 or mxfp4" >&2
    exit 2
fi

root="${EXPERIMENT_ROOT:-/home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-numa0-kfdv4-E9aLIK}"
repo=/home/xdai/Lumen
gpuagent_sha=5bc2a7d45f2fd992fcf3f53e12b5437a9b3fc7878c653a37ac647c380fb0e913
runner_path="$(readlink -f "${BASH_SOURCE[0]}")"
train_entry="$(readlink -f "${TRAIN_ENTRYPOINT:-${repo}/examples/qwen3/train_qwen3_fsdp.py}")"
run_dir="${root}/${case_label}"
cache_namespace="${CACHE_NAMESPACE:-${case_label}}"
cache_dir="${root}/cache/${cache_namespace}"
aiter_cache_dir="${AITER_CONFIG_CACHE_DIR:-${root}/cache/aiter-configs}"
gpu_lock_path="$(readlink -f "${GPU_LOCK_PATH:-/home/xdai/profile-results/.lumen-gpu-exclusive.lock}")"
if [[ "${REQUIRE_GPU_LOCK:-0}" == "1" ]]; then
    if [[ ! "${GPU_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
        echo "REQUIRE_GPU_LOCK=1 requires a numeric inherited GPU_LOCK_FD" >&2
        exit 9
    fi
    inherited_lock="$(readlink -f "/proc/$$/fd/${GPU_LOCK_FD}" 2>/dev/null || true)"
    if [[ "${inherited_lock}" != "${gpu_lock_path}" ]]; then
        echo "inherited GPU lock does not reference ${gpu_lock_path}" >&2
        exit 9
    fi
    flock -n "${GPU_LOCK_FD}" || {
        echo "inherited GPU lock is not exclusively held" >&2
        exit 9
    }
fi
if [[ -e "${run_dir}" ]]; then
    echo "case directory already exists: ${run_dir}" >&2
    exit 5
fi
if [[ "${REQUIRE_FRESH_CACHE:-1}" == "1" && -e "${cache_dir}" ]]; then
    echo "fresh cache namespace already exists: ${cache_dir}" >&2
    exit 6
fi
mkdir -p "${run_dir}" "${cache_dir}/xdg" "${cache_dir}/triton" \
    "${cache_dir}/inductor" "${cache_dir}/tmp" "${aiter_cache_dir}"

record_kfd_idle() {
    local output=$1
    local proc_dir pid comm exe cmdline owner ppid cgroup workloads
    local busy=0
    {
        date --iso-8601=seconds
        for proc_dir in /sys/class/kfd/kfd/proc/[0-9]*; do
            [[ -e "${proc_dir}" ]] || continue
            pid=${proc_dir##*/}
            if [[ ! -d "/proc/${pid}" ]]; then
                printf 'unreadable_kfd_client pid=%s phase=before_read\n' "${pid}"
                busy=1
                continue
            fi
            comm=$(cat "/proc/${pid}/comm" 2>/dev/null || true)
            exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
            cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
            owner=$(stat -c %U "/proc/${pid}" 2>/dev/null || true)
            ppid=$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)
            cgroup=$(cat "/proc/${pid}/cgroup" 2>/dev/null || true)
            if [[ ! -d "/proc/${pid}" ]]; then
                printf 'unreadable_kfd_client pid=%s phase=after_read\n' "${pid}"
                busy=1
                continue
            fi
            if [[ -z "${comm}" || -z "${cmdline}" || -z "${owner}" \
                  || -z "${ppid}" || -z "${cgroup}" ]]; then
                printf 'unreadable_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                    "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}"
                busy=1
            elif [[ "${comm}" == "gpuagent" \
                  && "${cmdline% }" == "/usr/local/bin/gpuagent" \
                  && "${owner}" == "root" && "${ppid}" == "1" \
                  && "${cgroup}" == "0::/system.slice/gpuagent.service" \
                  && -x /usr/local/bin/gpuagent \
                  && "$(sha256sum /usr/local/bin/gpuagent 2>/dev/null | cut -d' ' -f1)" == "${gpuagent_sha}" \
                  && ( -z "${exe}" || "${exe}" == "/usr/local/bin/gpuagent" ) ]]; then
                printf 'allowed_service pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                    "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}"
            else
                printf 'non_service_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                    "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}"
                busy=1
            fi
        done
        workloads=$(pgrep -af '[t]orchrun|[t]rain_qwen3_fsdp\.py|[r]ank_entry\.py|[r]ocprof(v3)?|[n]sys profile' || true)
        if [[ -n "${workloads}" ]]; then
            printf 'known_gpu_workloads=%q\n' "${workloads}"
            busy=1
        fi
        if (( busy == 0 )); then
            echo 'status=idle'
        else
            echo 'status=busy'
        fi
    } > "${output}"
    (( busy == 0 ))
}

[[ -r /dev/kfd && -w /dev/kfd ]] || {
    echo "/dev/kfd must be readable and writable" >&2
    exit 10
}
[[ -r /sys/class/kfd/kfd/proc ]] || {
    echo "/sys/class/kfd/kfd/proc must be readable" >&2
    exit 10
}
record_kfd_idle "${run_dir}/kfd-before.txt"

repo_state_digest() {
    local state_repo=$1
    {
        git -C "${state_repo}" rev-parse HEAD
        git -C "${state_repo}" diff --binary HEAD --
        while IFS= read -r -d '' relative; do
            local file_digest
            file_digest=$(sha256sum "${state_repo}/${relative}")
            printf '%s  %s\n' "${file_digest%% *}" "${relative}"
        done < <(git -C "${state_repo}" ls-files --others --exclude-standard -z | sort -z)
    } | sha256sum | cut -d' ' -f1
}

directory_digest() {
    local directory=$1
    local -a files=()
    while IFS= read -r -d '' path; do
        files+=("${path}")
    done < <(find "${directory}" -type f -print0 | sort -z)
    if (( ${#files[@]} == 0 )); then
        echo empty
    else
        sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
    fi
}

runtime_modules_digest() {
    local -a files=()
    while IFS= read -r -d '' path; do
        files+=("${path}")
    done < <(find /home/xdai/aiter/aiter/jit -maxdepth 1 -type f \
        -name '*.so' -print0 | sort -z)
    (( ${#files[@]} > 0 )) || {
        echo missing
        return
    }
    sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
}

source_files=(
    "${runner_path}"
    "${train_entry}"
    "${repo}/examples/qwen3/train_qwen3_fsdp.py"
    "${repo}/lumen/config.py"
    "${repo}/lumen/models/fsdp.py"
    "${repo}/lumen/quantize/__init__.py"
    "${repo}/lumen/quantize/config.py"
    "${repo}/lumen/ops/dispatch.py"
    "${repo}/lumen/ops/quantize/linear.py"
    "${repo}/lumen/ops/quantize/ops.py"
    "${repo}/lumen/kernels/mxfp4.py"
    "${repo}/lumen/ops/quantize/mxfp4_autotune.py"
    "${repo}/lumen/ops/quantize/mxfp4_asm.py"
    "${repo}/examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv"
    "${repo}/examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv"
    "/home/xdai/aiter/aiter/jit/core.py"
    "/home/xdai/aiter/aiter/ops/gemm_op_a4w4.py"
    "/home/xdai/aiter/aiter/ops/shuffle.py"
    "/home/xdai/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py"
    "/home/xdai/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py"
    "/home/xdai/aiter/aiter/ops/triton/utils/shuffle.py"
    "/home/xdai/aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv"
    "/home/xdai/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
    "/home/xdai/aiter/aiter/jit/module_aiter_core.so"
    "/home/xdai/aiter/aiter/jit/module_gemm_a4w4_asm.so"
    "/home/xdai/aiter/aiter/jit/module_fmha_v3_fwd.so"
    "/home/xdai/aiter/aiter/jit/module_fmha_v3_bwd.so"
    "/home/xdai/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co"
    "/home/xdai/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.co"
    "/home/xdai/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co"
    "/home/xdai/models/Qwen3-8B/config.json"
    "/home/xdai/models/Qwen3-8B/generation_config.json"
    "/home/xdai/models/Qwen3-8B/tokenizer.json"
    "/home/xdai/models/Qwen3-8B/tokenizer_config.json"
    "/home/xdai/models/Qwen3-8B/merges.txt"
    "/home/xdai/models/Qwen3-8B/vocab.json"
    "/home/xdai/fp8-coworker-repro/data/c4_train_1k_repeat4.jsonl"
    "/home/xdai/fp8-coworker-repro/data/c4_valid_heldout.jsonl"
)
while IFS= read -r runtime_module; do
    source_files+=("${runtime_module}")
done < <(find /home/xdai/aiter/aiter/jit -maxdepth 1 -type f -name '*.so' | sort)
if [[ -n "${TUNED_CONFIG_OVERRIDE:-}" ]]; then
    IFS=':' read -r -a tuned_config_paths <<< "${TUNED_CONFIG_OVERRIDE}"
    for tuned_path in "${tuned_config_paths[@]}"; do
        [[ -f "${tuned_path}" ]] || {
            echo "missing tuned config override: ${tuned_path}" >&2
            exit 6
        }
        source_files+=("$(readlink -f "${tuned_path}")")
    done
fi
if [[ -n "${EXTRA_SOURCE_FILES:-}" ]]; then
    IFS=':' read -r -a extra_source_paths <<< "${EXTRA_SOURCE_FILES}"
    for extra_path in "${extra_source_paths[@]}"; do
        [[ -f "${extra_path}" ]] || {
            echo "missing extra source file: ${extra_path}" >&2
            exit 6
        }
        source_files+=("$(readlink -f "${extra_path}")")
    done
fi
# A file can be both a tuned override and an explicit campaign input. Hash it
# once so every arm has the same source bundle when both candidate tables are
# frozen and only the selected runtime override changes.
declare -A seen_source_files=()
unique_source_files=()
for source_path in "${source_files[@]}"; do
    source_path=$(readlink -f "${source_path}")
    if [[ -z "${seen_source_files[${source_path}]+x}" ]]; then
        seen_source_files["${source_path}"]=1
        unique_source_files+=("${source_path}")
    fi
done
mapfile -t source_files < <(printf '%s\n' "${unique_source_files[@]}" | sort)
source_bundle_sha256=$(sha256sum "${source_files[@]}" | sha256sum | awk '{print $1}')
if [[ -n "${EXPECTED_SOURCE_BUNDLE_SHA256:-}" \
      && "${source_bundle_sha256}" != "${EXPECTED_SOURCE_BUNDLE_SHA256}" ]]; then
    echo "source bundle changed: expected ${EXPECTED_SOURCE_BUNDLE_SHA256}, got ${source_bundle_sha256}" >&2
    exit 3
fi
echo "${source_bundle_sha256}" > "${run_dir}/source-bundle-before.sha256"

lumen_diff_sha256=$(git -C "${repo}" diff --binary | sha256sum | awk '{print $1}')
lumen_status_sha256=$(git -C "${repo}" status --porcelain=v1 | sha256sum | awk '{print $1}')
aiter_diff_sha256=$(git -C /home/xdai/aiter diff --binary | sha256sum | awk '{print $1}')
aiter_status_sha256=$(git -C /home/xdai/aiter status --porcelain=v1 | sha256sum | awk '{print $1}')
lumen_tree_sha256=$(repo_state_digest "${repo}")
aiter_tree_sha256=$(repo_state_digest /home/xdai/aiter)
aiter_cache_sha256=$(directory_digest "${aiter_cache_dir}")
runtime_modules_sha256=$(runtime_modules_digest)
f4gemm_directory_sha256=$(directory_digest /home/xdai/aiter/hsa/gfx950/f4gemm)

cache_sha256_before=absent
if [[ -f "${cache_dir}/mxfp4-autotune.json" ]]; then
    cache_sha256_before=$(sha256sum "${cache_dir}/mxfp4-autotune.json" | awk '{print $1}')
    echo "${cache_sha256_before}  ${cache_dir}/mxfp4-autotune.json" \
        > "${run_dir}/cache-before.sha256"
fi
if [[ -n "${EXPECTED_CACHE_SHA256:-}" \
      && "${cache_sha256_before}" != "${EXPECTED_CACHE_SHA256}" ]]; then
    echo "cache changed: expected ${EXPECTED_CACHE_SHA256}, got ${cache_sha256_before}" >&2
    exit 6
fi

nproc="${NPROC:-8}"
mbs="${MBS:-2}"
gbs="${GBS:-128}"
seq_len="${SEQ_LEN:-8192}"
if (( gbs % (nproc * mbs) != 0 )); then
    echo "GBS must be divisible by NPROC*MBS" >&2
    exit 2
fi
grad_accum=$((gbs / (nproc * mbs)))
train_samples="${TRAIN_SAMPLES:-$((gbs * (steps + 2)))}"

args=(
    --model-name-or-path /home/xdai/models/Qwen3-8B
    --tokenizer-name-or-path /home/xdai/models/Qwen3-8B
    --task pretrain
    --init-from-scratch
    --train-data-path /home/xdai/fp8-coworker-repro/data/c4_train_1k_repeat4.jsonl
    --val-data-path /home/xdai/fp8-coworker-repro/data/c4_valid_heldout.jsonl
    --seq-length "${seq_len}"
    --micro-batch-size "${mbs}"
    --gradient-accumulation-steps "${grad_accum}"
    --max-steps "${steps}"
    --lr "${LR:-1.0e-4}"
    --min-lr 0
    --lr-warmup-steps "${LR_WARMUP_STEPS:-50}"
    --weight-decay 0.1
    --max-grad-norm 1.0
    --lora-rank 0
    --fsdp-version 2
    --sharding "${SHARDING:-full_shard}"
    --fsdp-reduce-dtype "${REDUCE_DTYPE:-bf16}"
    --aiter-attn
    --lumen-norm
    --fuse-rope
    --fused-cross-entropy
    --first-last-layers-bf16
    --num-layers-at-start-in-bf16 0
    --num-layers-at-end-in-bf16 "${TAIL_BF16:-5}"
    --train-samples "${train_samples}"
    --num-workers 0
    --log-interval 1
    --eval-interval "${EVAL_INTERVAL:-${steps}}"
    --val-samples "${VAL_SAMPLES:-64}"
    --eval-batches "${EVAL_BATCHES:-2}"
    --seed "${SEED:-1234}"
    --mode "${precision}"
)
if [[ "${RETAIN_ACCUM_PARAMS:-1}" == "1" ]]; then
    args+=(--fsdp-retain-accumulated-params)
fi
if [[ "${GRAD_CHECKPOINTING:-0}" != "1" ]]; then
    args+=(--no-grad-checkpointing)
fi
if [[ "${MXFP4_COMM:-0}" == "1" ]]; then
    args+=(--fsdp-mxfp4-comm)
fi
if [[ -n "${EXPERIMENT_EXTRA_ARGS:-}" ]]; then
    read -r -a experiment_args <<< "${EXPERIMENT_EXTRA_ARGS}"
    args+=("${experiment_args[@]}")
fi

profile_env=()
if [[ -n "${PROFILE_START:-}" ]]; then
    profile_env+=(
        LUMEN_PROF_START="${PROFILE_START}"
        LUMEN_PROF_END="${PROFILE_END:-${PROFILE_START}}"
        LUMEN_PROF_OUTPUT="${run_dir}/profile.txt"
        LUMEN_PROF_TRACE="${run_dir}/trace.json"
        LUMEN_PROF_SHAPES="${PROFILE_SHAPES:-1}"
        LUMEN_COPY_TRACE="${COPY_TRACE:-0}"
    )
fi
shape_log_env=()
if [[ "${ENABLE_SHAPE_LOG:-0}" == "1" ]]; then
    if [[ -z "${TRAIN_ENTRYPOINT:-}" ]]; then
        echo "ENABLE_SHAPE_LOG=1 requires the rank-aware TRAIN_ENTRYPOINT" >&2
        exit 2
    fi
    shape_log_env+=(
        LUMEN_MXFP4_GEMM_SHAPE_LOG_TEMPLATE="${run_dir}/mxfp4-shapes-rank{rank}.csv"
    )
fi
tuned_config_env=(-u AITER_CONFIG_GEMM_A4W4)
if [[ -n "${TUNED_CONFIG_OVERRIDE:-}" ]]; then
    tuned_config_env=(AITER_CONFIG_GEMM_A4W4="${TUNED_CONFIG_OVERRIDE}")
fi
printf -v command_string '%q ' /usr/local/bin/torchrun --standalone --nnodes=1 \
    --nproc-per-node="${nproc}" "${train_entry}" \
    "${args[@]}"

started_epoch_ns=$(date +%s%N)
{
    date --iso-8601=ns
    echo "case=${case_label}"
    echo "started_epoch_ns=${started_epoch_ns}"
    echo "precision=${precision}"
    echo "command=${command_string}"
    echo "global_batch=${gbs}"
    echo "gradient_accumulation=${grad_accum}"
    echo "tokens_per_update=$((gbs * seq_len))"
    echo "cache_namespace=${cache_namespace}"
    echo "cache_file=${cache_dir}/mxfp4-autotune.json"
    echo "cache_sha256_before=${cache_sha256_before}"
    echo "aiter_config_cache_dir=${aiter_cache_dir}"
    echo "gpu_lock_path=${gpu_lock_path}"
    echo "gpu_lock_verified=${REQUIRE_GPU_LOCK:-0}"
    echo "profile_start=${PROFILE_START:-none}"
    echo "profile_end=${PROFILE_END:-none}"
    echo "profile_shapes=${PROFILE_SHAPES:-0}"
    echo "copy_trace=${COPY_TRACE:-0}"
    echo "shape_log_enabled=${ENABLE_SHAPE_LOG:-0}"
    echo "numa_balancing=$(< /proc/sys/kernel/numa_balancing)"
    echo "tail_bf16=${TAIL_BF16:-5}"
    echo "eval_batches=${EVAL_BATCHES:-2}"
    echo "val_samples=${VAL_SAMPLES:-64}"
    echo "registry_freeze_removed=1"
    echo "weight_cache_fast_hit_removed=1"
    echo "mxfp4_activation_descriptor_cache=${LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE:-0}"
    echo "tuned_config_override=${TUNED_CONFIG_OVERRIDE:-trainer-default}"
    echo "source_bundle_sha256=${source_bundle_sha256}"
    echo "lumen_commit=$(git -C "${repo}" rev-parse HEAD)"
    echo "lumen_diff_sha256=${lumen_diff_sha256}"
    echo "lumen_status_sha256=${lumen_status_sha256}"
    echo "aiter_commit=$(git -C /home/xdai/aiter rev-parse HEAD)"
    echo "aiter_diff_sha256=${aiter_diff_sha256}"
    echo "aiter_status_sha256=${aiter_status_sha256}"
    echo "lumen_tree_sha256=${lumen_tree_sha256}"
    echo "aiter_tree_sha256=${aiter_tree_sha256}"
    echo "aiter_config_cache_sha256=${aiter_cache_sha256}"
    echo "runtime_modules_sha256=${runtime_modules_sha256}"
    echo "f4gemm_directory_sha256=${f4gemm_directory_sha256}"
    sha256sum "${source_files[@]}"
    PYTHONPATH="${repo}:/home/xdai/aiter" /usr/bin/python3 -c 'import aiter, lumen, torch; print(f"lumen_import={lumen.__file__}"); print(f"aiter_import={aiter.__file__}"); print(f"torch={torch.__version__} hip={torch.version.hip} devices={torch.cuda.device_count()}")'
    uname -a
    stat -c 'kfd=%A:%U:%G' /dev/kfd
} | tee "${run_dir}/run-meta.txt"

rocm-smi --showproductname --showuse --showmemuse --showpids \
    > "${run_dir}/rocm-smi-before.txt"
record_kfd_idle "${run_dir}/kfd-prelaunch.txt"

cd "${repo}"
sanitize_env_args=()
while IFS='=' read -r env_name _; do
    case "${env_name}" in
        LUMEN_*|AITER_*|CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|NCCL_*|TORCH_*|PYTORCH_*|TRITON_*|XDG_CACHE_HOME|TMPDIR|LD_PRELOAD|HSA_TOOLS_LIB|ROCR_TOOL_LIB|ROCPROFILER_*|CUDA_LAUNCH_BLOCKING|AMD_SERIALIZE_*|HSA_NO_SCRATCH_RECLAIM|HIP_FORCE_DEV_KERNARG|GPU_MAX_HW_QUEUES|CUDA_DEVICE_MAX_CONNECTIONS|OMP_NUM_THREADS|USE_HIPBLASLT|ENABLE_CK|FP4_DMA_INTRINSIC)
            sanitize_env_args+=(-u "${env_name}")
            ;;
    esac
done < <(env)
set +e
env \
    "${sanitize_env_args[@]}" \
    "${tuned_config_env[@]}" \
    "${profile_env[@]}" \
    "${shape_log_env[@]}" \
    PYTHONPATH="${repo}:/home/xdai/aiter" \
    PYTHONHASHSEED="${SEED:-1234}" \
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1 \
    GPU_MAX_HW_QUEUES=8 CUDA_DEVICE_MAX_CONNECTIONS=8 OMP_NUM_THREADS=1 \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1 \
    USE_HIPBLASLT=1 TORCH_BLAS_PREFER_HIPBLASLT=1 \
    NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=WARN \
    AITER_REBUILD=0 AITER_TUNE_GEMM=0 AITER_ONLINE_TUNE=0 \
    AITER_TRITON_ONLY=0 AITER_AOT_IMPORT=0 AITER_ENABLE_FMHA_OPUS=0 ENABLE_CK=1 \
    LUMEN_FAST_QUANT_DISPATCH=1 LUMEN_SKIP_BACKEND_SYNC=0 \
    LUMEN_USE_APEX_RMSNORM=0 LUMEN_WEIGHT_QUANT_ONCE=0 \
    LUMEN_MXFP4_ALLOW_UNVALIDATED_ARCH=0 \
    LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=0 LUMEN_MXFP4_DGRAD_HADAMARD=0 \
    LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE="${LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE:-0}" \
    LUMEN_SR_PHILOX_ROUNDS=7 LUMEN_MXFP4_AUTOTUNE=1 \
    LUMEN_MXFP4_FLYDSL=0 LUMEN_MXFP4_PROFILE_WARMUP=3 \
    LUMEN_MXFP4_PROFILE_ITERS=11 LUMEN_MXFP4_REQUIRE_CONSENSUS=1 \
    LUMEN_MXFP4_AUTOTUNE_CACHE="${cache_dir}/mxfp4-autotune.json" \
    LUMEN_PAIRED_RUN_EVIDENCE="${PAIRED_EVIDENCE:-1}" \
    AITER_CONFIG_CACHE_DIR="${aiter_cache_dir}" \
    XDG_CACHE_HOME="${cache_dir}/xdg" TRITON_CACHE_DIR="${cache_dir}/triton" \
    TORCHINDUCTOR_CACHE_DIR="${cache_dir}/inductor" TMPDIR="${cache_dir}/tmp" \
    /usr/local/bin/torchrun --standalone --nnodes=1 --nproc-per-node="${nproc}" \
    "${train_entry}" "${args[@]}" \
    2>&1 | tee "${run_dir}/train.log"
pipeline_status=("${PIPESTATUS[@]}")
train_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
{
    echo "torchrun=${train_status}"
    echo "tee=${tee_status}"
} > "${run_dir}/train-exit-status.txt"
rocm-smi --showproductname --showuse --showmemuse --showpids \
    > "${run_dir}/rocm-smi-after.txt"
rocm_status=$?
post_idle_status=1
post_idle_attempt=0
for post_idle_attempt in $(seq 1 10); do
    if record_kfd_idle "${run_dir}/kfd-after.txt"; then
        post_idle_status=0
        break
    fi
    if (( post_idle_attempt < 10 )); then
        sleep 1
    fi
done
echo "${post_idle_attempt}" > "${run_dir}/kfd-after-attempts.txt"
provenance_status=0

if [[ -f "${cache_dir}/mxfp4-autotune.json" ]]; then
    sha256sum "${cache_dir}/mxfp4-autotune.json" > "${run_dir}/cache-after.sha256"
fi
source_bundle_after=$(sha256sum "${source_files[@]}" | sha256sum | awk '{print $1}')
echo "${source_bundle_after}" > "${run_dir}/source-bundle-after.sha256"
if [[ "${source_bundle_after}" != "${source_bundle_sha256}" ]]; then
    echo "source bundle changed during run: before ${source_bundle_sha256}, after ${source_bundle_after}" >&2
    provenance_status=4
fi

lumen_diff_after=$(git -C "${repo}" diff --binary | sha256sum | awk '{print $1}')
lumen_status_after=$(git -C "${repo}" status --porcelain=v1 | sha256sum | awk '{print $1}')
aiter_diff_after=$(git -C /home/xdai/aiter diff --binary | sha256sum | awk '{print $1}')
aiter_status_after=$(git -C /home/xdai/aiter status --porcelain=v1 | sha256sum | awk '{print $1}')
lumen_tree_after=$(repo_state_digest "${repo}")
aiter_tree_after=$(repo_state_digest /home/xdai/aiter)
aiter_cache_after=$(directory_digest "${aiter_cache_dir}")
runtime_modules_after=$(runtime_modules_digest)
f4gemm_directory_after=$(directory_digest /home/xdai/aiter/hsa/gfx950/f4gemm)
{
    echo "lumen_diff_sha256=${lumen_diff_after}"
    echo "lumen_status_sha256=${lumen_status_after}"
    echo "aiter_diff_sha256=${aiter_diff_after}"
    echo "aiter_status_sha256=${aiter_status_after}"
    echo "lumen_tree_sha256=${lumen_tree_after}"
    echo "aiter_tree_sha256=${aiter_tree_after}"
    echo "aiter_config_cache_sha256=${aiter_cache_after}"
    echo "runtime_modules_sha256=${runtime_modules_after}"
    echo "f4gemm_directory_sha256=${f4gemm_directory_after}"
} > "${run_dir}/tree-state-after.txt"
if [[ "${lumen_diff_after}" != "${lumen_diff_sha256}" \
      || "${lumen_status_after}" != "${lumen_status_sha256}" \
      || "${aiter_diff_after}" != "${aiter_diff_sha256}" \
      || "${aiter_status_after}" != "${aiter_status_sha256}" \
      || "${lumen_tree_after}" != "${lumen_tree_sha256}" \
      || "${aiter_tree_after}" != "${aiter_tree_sha256}" \
      || "${runtime_modules_after}" != "${runtime_modules_sha256}" \
      || "${f4gemm_directory_after}" != "${f4gemm_directory_sha256}" \
      || "${aiter_cache_sha256}" != "empty" \
      || "${aiter_cache_after}" != "empty" ]]; then
    echo "repository state changed during run" >&2
    [[ "${provenance_status}" != "0" ]] || provenance_status=7
fi
if [[ -n "${EXPECTED_CACHE_SHA256:-}" ]]; then
    cache_sha256_after=absent
    if [[ -f "${cache_dir}/mxfp4-autotune.json" ]]; then
        cache_sha256_after=$(sha256sum "${cache_dir}/mxfp4-autotune.json" | awk '{print $1}')
    fi
    if [[ "${cache_sha256_after}" != "${EXPECTED_CACHE_SHA256}" ]]; then
        echo "cache changed during run: expected ${EXPECTED_CACHE_SHA256}, got ${cache_sha256_after}" >&2
        [[ "${provenance_status}" != "0" ]] || provenance_status=8
    fi
fi
{
    echo "torchrun=${train_status}"
    echo "tee=${tee_status}"
    echo "rocm_smi=${rocm_status}"
    echo "post_idle=${post_idle_status}"
    echo "provenance=${provenance_status}"
} > "${run_dir}/postflight-status.txt"
if [[ "${train_status}" != "0" ]]; then
    exit "${train_status}"
fi
if [[ "${tee_status}" != "0" ]]; then
    exit 21
fi
if [[ "${rocm_status}" != "0" ]]; then
    exit 23
fi
if [[ "${post_idle_status}" != "0" ]]; then
    exit 22
fi
if [[ "${provenance_status}" != "0" ]]; then
    exit "${provenance_status}"
fi

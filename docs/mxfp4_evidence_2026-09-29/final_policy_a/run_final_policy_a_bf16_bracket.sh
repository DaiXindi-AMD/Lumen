#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/xdai/profile-results/lumen-mxfp4-final-policy-a-bf16-bracket-fresh-20260928-fOB5j0
readonly REPO=/home/xdai/Lumen
readonly AITER_REPO=/home/xdai/aiter
readonly RUNNER=${ROOT}/run_case.sh
readonly BASE_ANALYZER=${ROOT}/analyze_tail210_base.py
readonly ENTRY=${ROOT}/inventory_entry.py
readonly GIT_EXCLUDES=${ROOT}/provenance-git-excludes
readonly DRIVER="$(readlink -f "${BASH_SOURCE[0]}")"
readonly ANALYZER=${ROOT}/analyze_final_policy_a_bf16_bracket.py
readonly SELF_TEST=${ROOT}/test_final_policy_a_bf16_bracket.py
readonly PROTOCOL=${ROOT}/protocol.md
readonly LOCK=/home/xdai/profile-results/.lumen-gpu-exclusive.lock
readonly GPUAGENT_SHA=5bc2a7d45f2fd992fcf3f53e12b5437a9b3fc7878c653a37ac647c380fb0e913
readonly MODEL_TABLE=${REPO}/examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv
readonly GENERIC_TABLE=${REPO}/examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv
readonly STOCK_TABLE=${AITER_REPO}/aiter/configs/a4w4_blockscale_tuned_gemm.csv
readonly TABLES=${MODEL_TABLE}:${GENERIC_TABLE}:${STOCK_TABLE}
readonly CACHE_NAMESPACE=final_policy_a_bf16_bracket
readonly CACHE=${ROOT}/cache/${CACHE_NAMESPACE}/mxfp4-autotune.json
readonly AITER_CACHE_DIR=${ROOT}/cache/aiter-configs
readonly META=${ROOT}/campaign-meta.txt
readonly PHASE1_COMPLETE=${ROOT}/phase1-complete.txt
readonly PHASE1_VALIDATION=${ROOT}/phase1-validation.json
readonly PHASE1_MANIFEST=${ROOT}/phase1-artifacts.sha256
readonly FORMAL_COMPLETE=${ROOT}/formal-arms-complete.txt
readonly CAMPAIGN_COMPLETE=${ROOT}/campaign-complete.txt
readonly LUMEN_COMMIT=6b9aee1569247eca20937c14319ba6adcd69e0cb
readonly AITER_COMMIT=e35bb17f4f815903bf73598facedbb321e15af28
readonly RUNNER_SHA=5ec1ca5b1a24585ce0634b44e1c4569b05dedb25274058951e2e4eb06c18aa9a
readonly ENTRY_SHA=0a25a6579ff2838ad64935115ce407f23a563b5a67f37d7000a7de4d08145fdc
readonly BASE_ANALYZER_SHA=ab6973743dabd1042a431304662e5d2805cff7d98aabd671df2c418b967009d6
readonly GIT_EXCLUDES_SHA=669f8c1235e39e040bc27437a69a98c578f86ee2659c546c594a8b78a924ed8e
readonly TRAIN_ENTRY_SHA=ffd60a612d522c2e125fb2a622da8bd9fcd0c4c5bb478ad6a26b544c9e38a74f
readonly FORMAL_STEPS=50
readonly TRAIN_SAMPLES=6400
readonly SMOKE=smoke_tail1
readonly FORMAL_ORDER=bf16_a1,mxfp4_policy_a,bf16_a2
readonly PHASE1_ORDER=smoke_tail1,bf16_a1,mxfp4_policy_a

# Ignore only mutable Codex logs in complete dirty-tree digests. Every explicit
# source/input remains in the runner's source bundle.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=core.excludesFile
export GIT_CONFIG_VALUE_0=${GIT_EXCLUDES}

mode=${1:-}
if [[ "${mode}" != phase1 && "${mode}" != phase2 && "${mode}" != dry-run ]]; then
    echo "usage: /usr/bin/bash ${DRIVER} phase1|phase2|dry-run" >&2
    exit 2
fi

expected_cache_sha=
expected_source_sha=

fail() { echo "$*" >&2; exit 6; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

meta_value() {
    local key=$1 path=$2
    awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); print; found=1} END {if (!found) exit 1}' "${path}"
}

write_atomic() {
    local target=$1 temporary=${1}.tmp.$$
    shift
    printf '%s\n' "$@" > "${temporary}"
    mv "${temporary}" "${target}"
}

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
    [[ -d "${directory}" && ! -L "${directory}" ]] \
        || fail "required directory missing or not a real directory: ${directory}"
    while IFS= read -r -d '' path; do files+=("${path}"); done \
        < <(find "${directory}" -type f -print0 | sort -z)
    if (( ${#files[@]} == 0 )); then
        echo empty
    else
        sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
    fi
}

runtime_modules_digest() {
    local -a files=()
    while IFS= read -r -d '' path; do files+=("${path}"); done \
        < <(find "${AITER_REPO}/aiter/jit" -maxdepth 1 -type f -name '*.so' -print0 | sort -z)
    (( ${#files[@]} > 0 )) || { echo missing; return; }
    sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
}

workload_digest() {
    sha256sum \
        /home/xdai/models/Qwen3-8B/config.json \
        /home/xdai/models/Qwen3-8B/generation_config.json \
        /home/xdai/models/Qwen3-8B/tokenizer.json \
        /home/xdai/models/Qwen3-8B/tokenizer_config.json \
        /home/xdai/models/Qwen3-8B/merges.txt \
        /home/xdai/models/Qwen3-8B/vocab.json \
        /home/xdai/fp8-coworker-repro/data/c4_train_1k_repeat4.jsonl \
        /home/xdai/fp8-coworker-repro/data/c4_valid_heldout.jsonl \
        | sha256sum | cut -d' ' -f1
}

source_manifest_digest() {
    awk '$1 ~ /^[0-9a-f]{64}$/ && $2 ~ /^\// {print $1 "  " $2}' "$1" \
        | sha256sum | cut -d' ' -f1
}

record_idle_recheck() {
    local output=$1 attempt proc_dir pid comm exe cmdline owner ppid cgroup workloads busy
    : > "${output}"
    for attempt in {1..10}; do
        busy=0
        {
            date --iso-8601=seconds
            echo "attempt=${attempt}"
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
                if [[ ! -d "/proc/${pid}" || -z "${comm}" || -z "${cmdline}" \
                      || -z "${owner}" || -z "${ppid}" || -z "${cgroup}" ]]; then
                    printf 'unreadable_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                        "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}"
                    busy=1
                elif [[ "${comm}" == gpuagent \
                      && "${cmdline% }" == /usr/local/bin/gpuagent \
                      && "${owner}" == root && "${ppid}" == 1 \
                      && "${cgroup}" == "0::/system.slice/gpuagent.service" \
                      && -x /usr/local/bin/gpuagent \
                      && "$(sha256sum /usr/local/bin/gpuagent 2>/dev/null | cut -d' ' -f1)" == "${GPUAGENT_SHA}" \
                      && ( -z "${exe}" || "${exe}" == /usr/local/bin/gpuagent ) ]]; then
                    printf 'allowed_service pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                        "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}"
                else
                    printf 'non_service_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q cgroup=%q\n' \
                        "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}" "${cgroup}"
                    busy=1
                fi
            done
            workloads=$(pgrep -af '[t]orchrun|[t]rain_qwen3_fsdp\.py|[i]nventory_entry\.py|[r]ocprof(v3)?|[n]sys profile' || true)
            [[ -z "${workloads}" ]] || { printf 'known_gpu_workloads=%q\n' "${workloads}"; busy=1; }
            (( busy == 0 )) && echo status=idle || echo status=busy
        } >> "${output}"
        (( busy == 0 )) && return 0
        (( attempt == 10 )) || sleep 1
    done
    return 1
}

verify_common_inputs() {
    local path
    for path in "${RUNNER}" "${BASE_ANALYZER}" "${ENTRY}" "${GIT_EXCLUDES}" \
        "${DRIVER}" "${ANALYZER}" "${SELF_TEST}" "${PROTOCOL}" \
        "${MODEL_TABLE}" "${GENERIC_TABLE}" "${STOCK_TABLE}"; do
        [[ -f "${path}" ]] || fail "required input missing: ${path}"
    done
    [[ "$(sha "${RUNNER}")" == "${RUNNER_SHA}" ]] || fail "runner changed"
    [[ "$(sha "${ENTRY}")" == "${ENTRY_SHA}" ]] || fail "inventory entry changed"
    [[ "$(sha "${BASE_ANALYZER}")" == "${BASE_ANALYZER_SHA}" ]] || fail "base analyzer changed"
    [[ "$(sha "${GIT_EXCLUDES}")" == "${GIT_EXCLUDES_SHA}" ]] || fail "Git excludes changed"
    [[ "$(sha "${REPO}/examples/qwen3/train_qwen3_fsdp.py")" == "${TRAIN_ENTRY_SHA}" ]] || fail "trainer changed"
    [[ "$(< "${GIT_EXCLUDES}")" == .codex/ ]] || fail "Git excludes must contain only .codex/"
    [[ "$(git -C "${REPO}" branch --show-current)" == dev/mxfp4 ]] || fail "Lumen branch changed"
    [[ "$(git -C "${REPO}" rev-parse HEAD)" == "${LUMEN_COMMIT}" ]] || fail "Lumen commit changed"
    [[ "$(git -C "${AITER_REPO}" rev-parse HEAD)" == "${AITER_COMMIT}" ]] || fail "AITER commit changed"
    [[ "$(< /proc/sys/kernel/numa_balancing)" == 0 ]] || fail "NUMA balancing must be zero"
    command -v numactl >/dev/null || fail "numactl is missing"
    (( TRAIN_SAMPLES == FORMAL_STEPS * 128 )) || fail "TRAIN_SAMPLES must equal steps*GBS"
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}" --self-test
    PYTHONPATH="${ROOT}" PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -m unittest -q "${SELF_TEST}"
}

verify_runtime_environment() {
    [[ -r /dev/kfd && -w /dev/kfd ]] || fail "/dev/kfd must be readable and writable"
    [[ -r /sys/class/kfd/kfd/proc ]] || fail "/sys/class/kfd/kfd/proc must be readable"
}

extra_source_files() {
    printf '%s' "${DRIVER}:${ANALYZER}:${SELF_TEST}:${PROTOCOL}:${ENTRY}:${GIT_EXCLUDES}:${BASE_ANALYZER}"
    printf ':%s' \
        "${REPO}/lumen/models/qwen3.py" \
        "${REPO}/lumen/ops/fused_swiglu.py" \
        "${REPO}/lumen/ops/quantize/__init__.py" \
        "${AITER_REPO}/aiter/ops/triton/activation.py" \
        "${AITER_REPO}/aiter/ops/triton/_triton_kernels/activation.py"
}

run_arm() {
    local label=$1 precision=$2 steps=$3 fresh_cache=$4 route_probe=$5
    local experiment_extra_args=
    local -a env_args
    [[ "${precision}" != mxfp4 ]] || experiment_extra_args='--mxfp4-pack-qkv --mxfp4-fuse-swiglu'
    env_args=(
        EXPERIMENT_ROOT="${ROOT}"
        CACHE_NAMESPACE="${CACHE_NAMESPACE}"
        AITER_CONFIG_CACHE_DIR="${AITER_CACHE_DIR}"
        TUNED_CONFIG_OVERRIDE="${TABLES}"
        EXTRA_SOURCE_FILES="$(extra_source_files)"
        REQUIRE_GPU_LOCK=1 GPU_LOCK_FD="${gpu_lock_fd}" GPU_LOCK_PATH="${LOCK}"
        REQUIRE_FRESH_CACHE="${fresh_cache}"
        ENABLE_SHAPE_LOG="${route_probe}"
        PROFILE_START= PROFILE_END=
        EXPERIMENT_EXTRA_ARGS="${experiment_extra_args}"
        NPROC=8 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_SAMPLES="${TRAIN_SAMPLES}"
        LR=1.0e-4 LR_WARMUP_STEPS=50 SHARDING=full_shard REDUCE_DTYPE=bf16
        RETAIN_ACCUM_PARAMS=1 GRAD_CHECKPOINTING=0 MXFP4_COMM=0
        TAIL_BF16=1 EVAL_INTERVAL="${steps}" EVAL_BATCHES=16 VAL_SAMPLES=256
        PAIRED_EVIDENCE=1 SEED=1234 LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE=0
    )
    if [[ "${route_probe}" == 1 ]]; then
        env_args+=(TRAIN_ENTRYPOINT="${ENTRY}" ROUTE_REPORT_TEMPLATE="${ROOT}/${label}/route-rank{rank}.json")
    else
        env_args+=(TRAIN_ENTRYPOINT= ROUTE_REPORT_TEMPLATE=)
    fi
    [[ -z "${expected_cache_sha}" ]] || env_args+=(EXPECTED_CACHE_SHA256="${expected_cache_sha}")
    [[ -z "${expected_source_sha}" ]] || env_args+=(EXPECTED_SOURCE_BUNDLE_SHA256="${expected_source_sha}")
    set +e
    numactl --cpunodebind=0 --membind=0 env "${env_args[@]}" \
        "${RUNNER}" "${label}" "${precision}" "${steps}"
    local status=$?
    set -e
    if (( status != 0 )); then
        if (( status == 22 )) \
            && grep -qx torchrun=0 "${ROOT}/${label}/train-exit-status.txt" \
            && grep -qx tee=0 "${ROOT}/${label}/train-exit-status.txt" \
            && grep -qx rocm_smi=0 "${ROOT}/${label}/postflight-status.txt" \
            && grep -qx provenance=0 "${ROOT}/${label}/postflight-status.txt" \
            && record_idle_recheck "${ROOT}/${label}/kfd-after-recheck.txt"; then
            write_atomic "${ROOT}/${label}/postflight-recheck-status.txt" accepted_transient_postflight_kfd_race=1
        else
            return "${status}"
        fi
    fi
}

write_meta() {
    local smoke_meta=${ROOT}/${SMOKE}/run-meta.txt temporary=${META}.tmp.$$
    {
        date --iso-8601=ns
        echo schema=1
        echo branch=dev/mxfp4
        echo lumen_commit="${LUMEN_COMMIT}"
        echo aiter_commit="${AITER_COMMIT}"
        echo source_bundle_sha256="${expected_source_sha}"
        echo source_manifest_sha256="$(source_manifest_digest "${smoke_meta}")"
        echo fresh_autotune_cache_sha256="${expected_cache_sha}"
        echo workload_sha256="$(workload_digest)"
        for key in lumen_tree_sha256 aiter_tree_sha256 runtime_modules_sha256 f4gemm_directory_sha256 aiter_config_cache_sha256; do
            echo "${key}=$(meta_value "${key}" "${smoke_meta}")"
        done
        echo formal_order="${FORMAL_ORDER}"
        echo phase1_order="${PHASE1_ORDER}"
        echo formal_steps="${FORMAL_STEPS}"
        echo train_samples="${TRAIN_SAMPLES}"
        echo timing_window=11-50
        echo speed_gate_min_ratio=1.6
        echo speed_gate_min_wins=28
        echo bootstrap_ci_lower_minimum=1.6
        echo bootstrap_block_length=4
        echo bootstrap_resamples=100000
        echo bf16_replicate_drift_gate_pct=3.0
        echo precision_gate_delta_nll=0.01
        echo candidate_stack=policy_a_tail1+packed_qkv+split_swiglu
        echo lm_head_precision=bf16
        echo lm_head_evidence=fresh_policy_a_smoke_plus_frozen_source
        echo mxfp4_communication=disabled
        echo git_exclude_scope=.codex/
        echo source_integrity=exact_manifest+full_tree_without_codex_runtime_logs
        echo shape_count_contract=policy_a_whole_smoke_totals
        echo train_pairing_scope=all_arms_exact_first_update
        echo kfd_identity_policy=fail_closed_complete_identity
        echo gpuagent_sha256="${GPUAGENT_SHA}"
        echo numa_balancing="$(< /proc/sys/kernel/numa_balancing)"
        echo driver_sha256="$(sha "${DRIVER}")"
        echo analyzer_sha256="$(sha "${ANALYZER}")"
        echo self_test_sha256="$(sha "${SELF_TEST}")"
        echo protocol_sha256="$(sha "${PROTOCOL}")"
        echo runner_sha256="$(sha "${RUNNER}")"
        echo entry_sha256="$(sha "${ENTRY}")"
        echo base_analyzer_sha256="$(sha "${BASE_ANALYZER}")"
        echo git_excludes_sha256="$(sha "${GIT_EXCLUDES}")"
        echo train_entry_sha256="$(sha "${REPO}/examples/qwen3/train_qwen3_fsdp.py")"
        stat -c 'kfd=%A:%U:%G' /dev/kfd
    } > "${temporary}"
    mv "${temporary}" "${META}"
}

verify_harness_hashes() {
    local key path
    while read -r key path; do
        [[ "$(meta_value "${key}" "${META}")" == "$(sha "${path}")" ]] \
            || fail "harness/source changed: ${path}"
    done <<EOF
driver_sha256 ${DRIVER}
analyzer_sha256 ${ANALYZER}
self_test_sha256 ${SELF_TEST}
protocol_sha256 ${PROTOCOL}
runner_sha256 ${RUNNER}
entry_sha256 ${ENTRY}
base_analyzer_sha256 ${BASE_ANALYZER}
git_excludes_sha256 ${GIT_EXCLUDES}
train_entry_sha256 ${REPO}/examples/qwen3/train_qwen3_fsdp.py
EOF
}

verify_current_state() {
    local key expected actual
    [[ "$(workload_digest)" == "$(meta_value workload_sha256 "${META}")" ]] || fail "workload changed"
    for key in lumen_tree_sha256 aiter_tree_sha256 aiter_config_cache_sha256 runtime_modules_sha256 f4gemm_directory_sha256; do
        expected=$(meta_value "${key}" "${META}")
        case "${key}" in
            lumen_tree_sha256) actual=$(repo_state_digest "${REPO}") ;;
            aiter_tree_sha256) actual=$(repo_state_digest "${AITER_REPO}") ;;
            aiter_config_cache_sha256) actual=$(directory_digest "${AITER_CACHE_DIR}") ;;
            runtime_modules_sha256) actual=$(runtime_modules_digest) ;;
            f4gemm_directory_sha256) actual=$(directory_digest "${AITER_REPO}/hsa/gfx950/f4gemm") ;;
        esac
        [[ "${actual}" == "${expected}" ]] || fail "current ${key} changed"
    done
}

phase1_artifact_paths() {
    local directory
    printf '%s\0' \
        "${META}" "${CACHE}" \
        "${ROOT}/kfd-final-policy-a-phase1-before.txt" \
        "${ROOT}/kfd-final-policy-a-phase1-after.txt" \
        "${PHASE1_VALIDATION}" \
        "${ROOT}/final-policy-a-bf16-bracket-phase1-exit-status.txt"
    for directory in "${ROOT}/${SMOKE}" "${ROOT}/bf16_a1" "${ROOT}/mxfp4_policy_a"; do
        find "${directory}" -maxdepth 1 -type f -print0
    done
}

write_phase1_manifest() {
    local temporary=${PHASE1_MANIFEST}.tmp.$$
    local -a files=()
    while IFS= read -r -d '' path; do files+=("${path}"); done \
        < <(phase1_artifact_paths | sort -zu)
    (( ${#files[@]} > 0 )) || fail "phase1 artifact set is empty"
    sha256sum "${files[@]}" > "${temporary}"
    mv "${temporary}" "${PHASE1_MANIFEST}"
}

verify_phase1_manifest() {
    PYTHONPATH="${ROOT}" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - <<'PY'
import analyze_final_policy_a_bf16_bracket as analysis
if not analysis.checksum_manifest_ok(
    analysis.PHASE1_MANIFEST, analysis.phase1_artifact_paths()
):
    raise SystemExit("phase1 artifact manifest mismatch")
PY
}

verify_phase1() {
    [[ -f "${META}" && -f "${CACHE}" && -f "${PHASE1_COMPLETE}" ]] || fail "phase1 artifacts missing"
    [[ "$(< "${ROOT}/final-policy-a-bf16-bracket-phase1-exit-status.txt")" == 0 ]] || fail "phase1 exit was nonzero"
    expected_source_sha=$(meta_value source_bundle_sha256 "${META}")
    expected_cache_sha=$(meta_value fresh_autotune_cache_sha256 "${META}")
    [[ "$(sha "${CACHE}")" == "${expected_cache_sha}" ]] || fail "cache changed before phase2"
    [[ "$(meta_value completed "${PHASE1_COMPLETE}")" == "${PHASE1_ORDER}" ]] || fail "phase1 order mismatch"
    [[ "$(meta_value source_bundle_sha256 "${PHASE1_COMPLETE}")" == "${expected_source_sha}" ]] || fail "phase1 source mismatch"
    [[ "$(meta_value cache_sha256 "${PHASE1_COMPLETE}")" == "${expected_cache_sha}" ]] || fail "phase1 cache mismatch"
    [[ "$(meta_value meta_sha256 "${PHASE1_COMPLETE}")" == "$(sha "${META}")" ]] || fail "phase1 metadata changed"
    [[ "$(meta_value phase1_artifacts_manifest_sha256 "${PHASE1_COMPLETE}")" == "$(sha "${PHASE1_MANIFEST}")" ]] || fail "phase1 manifest changed"
    verify_harness_hashes
    verify_current_state
    verify_phase1_manifest
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
        "${ANALYZER}" --validate-phase1-boundary >/dev/null
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}" --validate-phase1 >/dev/null
}

run_phase1() {
    local path
    [[ ! -e "${ROOT}/cache" ]] || fail "fresh cache root already exists"
    for path in "${SMOKE}" bf16_a1 mxfp4_policy_a bf16_a2; do
        [[ ! -e "${ROOT}/${path}" ]] || fail "output exists: ${path}"
    done
    for path in "${META}" "${PHASE1_COMPLETE}" "${PHASE1_VALIDATION}" \
        "${PHASE1_MANIFEST}" "${FORMAL_COMPLETE}" "${CAMPAIGN_COMPLETE}" \
        "${ROOT}/final-policy-a-bf16-bracket-phase1-exit-status.txt" \
        "${ROOT}/final-policy-a-bf16-bracket-phase2-exit-status.txt" \
        "${ROOT}/final_policy_a_bf16_bracket_analysis.json" \
        "${ROOT}/final_policy_a_bf16_bracket_analysis.md"; do
        [[ ! -e "${path}" ]] || fail "campaign artifact exists: ${path}"
    done
    exec {gpu_lock_fd}>"${LOCK}"
    flock -n "${gpu_lock_fd}" || fail "another GPU campaign owns ${LOCK}"
    record_idle_recheck "${ROOT}/kfd-final-policy-a-phase1-before.txt" || fail "KFD busy before phase1"
    run_arm "${SMOKE}" mxfp4 3 1 1
    [[ -f "${CACHE}" ]] || fail "fresh smoke did not create cache"
    expected_cache_sha=$(sha "${CACHE}")
    expected_source_sha=$(< "${ROOT}/${SMOKE}/source-bundle-before.sha256")
    [[ "${expected_source_sha}" == "$(< "${ROOT}/${SMOKE}/source-bundle-after.sha256")" ]] || fail "smoke source changed"
    write_meta
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}" --validate-smoke >/dev/null
    run_arm bf16_a1 bf16 "${FORMAL_STEPS}" 0 0
    run_arm mxfp4_policy_a mxfp4 "${FORMAL_STEPS}" 0 0
    record_idle_recheck "${ROOT}/kfd-final-policy-a-phase1-after.txt" || fail "KFD busy after phase1"
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}" --validate-phase1 > "${PHASE1_VALIDATION}"
    write_atomic "${PHASE_STATUS}" 0
    write_phase1_manifest
    write_atomic "${PHASE1_COMPLETE}" schema=1 completed="${PHASE1_ORDER}" \
        source_bundle_sha256="${expected_source_sha}" cache_sha256="${expected_cache_sha}" \
        meta_sha256="$(sha "${META}")" \
        phase1_artifacts_manifest_sha256="$(sha "${PHASE1_MANIFEST}")" \
        completed_epoch_ns="$(date +%s%N)"
}

run_phase2() {
    verify_phase1
    [[ ! -e "${ROOT}/bf16_a2" ]] || fail "bf16_a2 output already exists"
    [[ ! -e "${ROOT}/final-policy-a-bf16-bracket-phase2-exit-status.txt" ]] || fail "phase2 status already exists"
    for path in "${FORMAL_COMPLETE}" "${CAMPAIGN_COMPLETE}" \
        "${ROOT}/final_policy_a_bf16_bracket_analysis.json" \
        "${ROOT}/final_policy_a_bf16_bracket_analysis.md"; do
        [[ ! -e "${path}" ]] || fail "final artifact exists: ${path}"
    done
    exec {gpu_lock_fd}>"${LOCK}"
    flock -n "${gpu_lock_fd}" || fail "another GPU campaign owns ${LOCK}"
    record_idle_recheck "${ROOT}/kfd-final-policy-a-phase2-before.txt" || fail "KFD busy before phase2"
    run_arm bf16_a2 bf16 "${FORMAL_STEPS}" 0 0
    record_idle_recheck "${ROOT}/kfd-final-policy-a-phase2-after.txt" || fail "KFD busy after phase2"
    write_atomic "${FORMAL_COMPLETE}" schema=1 completed="${FORMAL_ORDER}" \
        source_bundle_sha256="${expected_source_sha}" cache_sha256="${expected_cache_sha}" \
        completed_epoch_ns="$(date +%s%N)"
    write_atomic "${PHASE_STATUS}" 0
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}"
    write_atomic "${CAMPAIGN_COMPLETE}" schema=1 completed="${FORMAL_ORDER}" \
        source_bundle_sha256="${expected_source_sha}" cache_sha256="${expected_cache_sha}" \
        analysis_json_sha256="$(sha "${ROOT}/final_policy_a_bf16_bracket_analysis.json")" \
        analysis_markdown_sha256="$(sha "${ROOT}/final_policy_a_bf16_bracket_analysis.md")" \
        confirmation_passed="$(PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["confirmation_passed"]).lower())' "${ROOT}/final_policy_a_bf16_bracket_analysis.json")" \
        completed_epoch_ns="$(date +%s%N)"
}

if [[ "${mode}" == dry-run ]]; then
    verify_common_inputs
    echo "dry-run PASS: no GPU process launched"
    echo "formal_order=${FORMAL_ORDER}"
    echo "phase1_order=${PHASE1_ORDER}"
    echo "formal_steps=${FORMAL_STEPS} timing_window=11-50 train_samples=${TRAIN_SAMPLES}"
    echo "policy_a=tail1+packed_qkv+split_swiglu"
    exit 0
fi

readonly PHASE_STATUS=${ROOT}/final-policy-a-bf16-bracket-${mode}-exit-status.txt
finish() {
    local status=$? temporary=${PHASE_STATUS}.tmp.$$
    printf '%s\n' "${status}" > "${temporary}"
    mv "${temporary}" "${PHASE_STATUS}"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

verify_common_inputs
verify_runtime_environment
if [[ "${mode}" == phase1 ]]; then run_phase1; else run_phase2; fi

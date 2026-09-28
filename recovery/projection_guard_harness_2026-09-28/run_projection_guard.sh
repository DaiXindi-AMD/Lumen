#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/xdai/profile-results/lumen-mxfp4-projection-guard-formal-fresh-20260928-XWXD4L
readonly REPO=/home/xdai/Lumen
readonly AITER_REPO=/home/xdai/aiter
readonly RUNNER=${ROOT}/run_case.sh
readonly BASE_ANALYZER=${ROOT}/analyze_tail210_base.py
readonly ENTRY=${ROOT}/inventory_entry.py
readonly DRIVER="$(readlink -f "${BASH_SOURCE[0]}")"
readonly ANALYZER=${ROOT}/analyze_projection_guard.py
readonly SELF_TEST=${ROOT}/test_analyze_projection_guard.py
readonly PROTOCOL=${ROOT}/protocol.md
readonly GIT_EXCLUDES=${ROOT}/provenance-git-excludes
readonly SHAPE_FIXTURE_DIR=${ROOT}/fixtures/policy_b_smoke
readonly -a SHAPE_FIXTURES=(
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank0.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank1.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank2.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank3.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank4.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank5.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank6.csv"
    "${SHAPE_FIXTURE_DIR}/mxfp4-shapes-rank7.csv"
)
readonly LOCK=/home/xdai/profile-results/.lumen-gpu-exclusive.lock
readonly MODEL_TABLE=${REPO}/examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv
readonly GENERIC_TABLE=${REPO}/examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv
readonly STOCK_TABLE=${AITER_REPO}/aiter/configs/a4w4_blockscale_tuned_gemm.csv
readonly TABLES=${MODEL_TABLE}:${GENERIC_TABLE}:${STOCK_TABLE}
readonly CACHE_NAMESPACE=projection_guard
readonly CACHE=${ROOT}/cache/${CACHE_NAMESPACE}/mxfp4-autotune.json
readonly AITER_CACHE_DIR=${ROOT}/cache/aiter-configs
readonly META=${ROOT}/campaign-meta.txt
readonly PHASE1_COMPLETE=${ROOT}/phase1-complete.txt
readonly FORMAL_COMPLETE=${ROOT}/formal-arms-complete.txt
readonly CAMPAIGN_COMPLETE=${ROOT}/campaign-complete.txt
readonly LUMEN_COMMIT=6b9aee1569247eca20937c14319ba6adcd69e0cb
readonly AITER_COMMIT=e35bb17f4f815903bf73598facedbb321e15af28
readonly RUNNER_SHA=3740d3308545e8c5f394525caac48e26b3fdbfe85d272f31b6ee4b2d1c7817a7
readonly BASE_ANALYZER_SHA=ab6973743dabd1042a431304662e5d2805cff7d98aabd671df2c418b967009d6
readonly TRAIN_ENTRY_SHA=ffd60a612d522c2e125fb2a622da8bd9fcd0c4c5bb478ad6a26b544c9e38a74f
readonly SHAPE_FIXTURE_SHA=6d68e8566b6ac24603cece0b20f0c811fb4b07119e8b53767e5404c35775b271
readonly PRIOR_SMOKE_ROOT=/home/xdai/profile-results/lumen-mxfp4-projection-guard-smoke-fresh-20260922-f68RA4
readonly PRIOR_ROUTE=${PRIOR_SMOKE_ROOT}/guard_o_down_smoke/route-rank0.json
readonly PRIOR_META=${PRIOR_SMOKE_ROOT}/guard_o_down_smoke/run-meta.txt
readonly PRIOR_ROUTE_SHA=985881f03d7b7d5106c2de46d52a787975d2a3b4d97323cc7fd775516416f336
readonly PRIOR_META_SHA=8359c3c324ae5e7217b05822a74f0099abf9a9fee75cad04927881290985d43b
readonly FORMAL_STEPS=50
readonly TRAIN_SAMPLES=6400
readonly SMOKE=route_guard_o_down_fresh
readonly -a FORMAL_CASES=(tail1_a1 guard_o_down_b1 guard_down_c guard_o_down_b2 tail1_a2)
readonly FORMAL_ORDER=tail1_a1,guard_o_down_b1,guard_down_c,guard_o_down_b2,tail1_a2
readonly PHASE1_ORDER=route_guard_o_down_fresh,tail1_a1,guard_o_down_b1,guard_down_c

# Ignore only mutable, non-source Codex runtime logs in Git dirty-tree scans.
# The shared runner still hashes every explicit source/input and the complete
# remaining Lumen/AITER dirty trees.
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

fixture_digest() {
    local path
    {
        for path in "${SHAPE_FIXTURES[@]}"; do
            printf '%s  %s\n' "$(sha "${path}")" "${path##*/}"
        done
    } | sha256sum | cut -d' ' -f1
}

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
    while IFS= read -r -d '' path; do files+=("${path}"); done < <(find "${directory}" -type f -print0 | sort -z)
    if (( ${#files[@]} == 0 )); then echo empty; else sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1; fi
}

runtime_modules_digest() {
    local -a files=()
    while IFS= read -r -d '' path; do files+=("${path}"); done < <(find "${AITER_REPO}/aiter/jit" -maxdepth 1 -type f -name '*.so' -print0 | sort -z)
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
    awk '$1 ~ /^[0-9a-f]{64}$/ && $2 ~ /^\// {print $1 "  " $2}' "$1" | sha256sum | cut -d' ' -f1
}

record_idle_recheck() {
    local output=$1 attempt proc_dir pid comm exe cmdline owner ppid workloads busy
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
                if [[ ! -d "/proc/${pid}" ]]; then
                    printf 'unreadable_kfd_client pid=%s phase=after_read\n' "${pid}"
                    busy=1
                    continue
                fi
                if [[ -z "${comm}" || -z "${exe}" || -z "${cmdline}" \
                      || -z "${owner}" || -z "${ppid}" ]]; then
                    printf 'unreadable_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q\n' \
                        "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}"
                    busy=1
                elif [[ "${comm}" == gpuagent \
                      && "${exe}" == /usr/local/bin/gpuagent \
                      && "${cmdline% }" == /usr/local/bin/gpuagent \
                      && "${owner}" == root && "${ppid}" == 1 ]]; then
                    printf 'allowed_service pid=%s comm=%q owner=%q ppid=%q cmdline=%q\n' \
                        "${pid}" "${comm}" "${owner}" "${ppid}" "${cmdline}"
                else
                    printf 'non_service_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q\n' \
                        "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}"
                    busy=1
                fi
            done
            workloads=$(pgrep -af '[t]orchrun|[t]rain_qwen3_fsdp\.py|[i]nventory_entry\.py|[r]ocprof(v3)?|[n]sys profile' || true)
            [[ -z "${workloads}" ]] || { printf 'known_gpu_workloads=%q\n' "${workloads}"; busy=1; }
            (( busy == 0 )) && echo status=idle || echo status=busy
        } >> "${output}"
        (( busy == 0 )) && return 0
        sleep 1
    done
    return 1
}

verify_common_inputs() {
    local path
    for path in "${RUNNER}" "${BASE_ANALYZER}" "${ENTRY}" "${DRIVER}" "${ANALYZER}" "${SELF_TEST}" "${PROTOCOL}" "${GIT_EXCLUDES}" "${MODEL_TABLE}" "${GENERIC_TABLE}" "${STOCK_TABLE}" "${PRIOR_ROUTE}" "${PRIOR_META}"; do
        [[ -f "${path}" ]] || fail "required input missing: ${path}"
    done
    for path in "${SHAPE_FIXTURES[@]}"; do
        [[ -f "${path}" ]] || fail "required shape fixture missing: ${path}"
    done
    [[ "$(sha "${RUNNER}")" == "${RUNNER_SHA}" ]] || fail "shared runner changed"
    [[ "$(sha "${BASE_ANALYZER}")" == "${BASE_ANALYZER_SHA}" ]] || fail "base analyzer changed"
    [[ "$(sha "${REPO}/examples/qwen3/train_qwen3_fsdp.py")" == "${TRAIN_ENTRY_SHA}" ]] || fail "hardened trainer changed"
    [[ "$(fixture_digest)" == "${SHAPE_FIXTURE_SHA}" ]] || fail "shape fixture changed"
    [[ "$(sha "${PRIOR_ROUTE}")" == "${PRIOR_ROUTE_SHA}" ]] || fail "prior route metadata changed"
    [[ "$(sha "${PRIOR_META}")" == "${PRIOR_META_SHA}" ]] || fail "prior smoke metadata changed"
    [[ "$(< "${GIT_EXCLUDES}")" == .codex/ ]] || fail "Git excludes must contain only .codex/"
    [[ "$(git -C "${REPO}" branch --show-current)" == dev/mxfp4 ]] || fail "Lumen branch is not dev/mxfp4"
    [[ "$(git -C "${REPO}" rev-parse HEAD)" == "${LUMEN_COMMIT}" ]] || fail "unexpected Lumen commit"
    [[ "$(git -C "${AITER_REPO}" rev-parse HEAD)" == "${AITER_COMMIT}" ]] || fail "unexpected AITER commit"
    (( TRAIN_SAMPLES == FORMAL_STEPS * 128 )) || fail "TRAIN_SAMPLES must equal steps*GBS"
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "${ANALYZER}" --self-test
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover \
        -s "${ROOT}" -p 'test_analyze_projection_guard.py' -q
}

verify_runtime_environment() {
    [[ -r /dev/kfd && -w /dev/kfd ]] || fail "/dev/kfd must be readable and writable"
    [[ -r /sys/class/kfd/kfd/proc ]] || fail "/sys/class/kfd/kfd/proc must be readable"
    [[ "$(< /proc/sys/kernel/numa_balancing)" == 1 ]] || fail "NUMA balancing changed"
}

policy_args() {
    case "$1" in
        a) printf '%s' '--mxfp4-pack-qkv --mxfp4-fuse-swiglu' ;;
        b) printf '%s' '--mxfp4-pack-qkv --mxfp4-fuse-swiglu --mxfp4-last-layer-bf16-projections o_proj down_proj' ;;
        c) printf '%s' '--mxfp4-pack-qkv --mxfp4-fuse-swiglu --mxfp4-last-layer-bf16-projections down_proj' ;;
        *) fail "unknown policy $1" ;;
    esac
}

policy_tail() { [[ "$1" == a ]] && echo 1 || echo 0; }

run_arm() {
    local label=$1 policy=$2 steps=$3 fresh_cache=$4 shape_log=$5
    local extra_sources fixture
    extra_sources="${DRIVER}:${ANALYZER}:${SELF_TEST}:${PROTOCOL}:${ENTRY}:${GIT_EXCLUDES}:${BASE_ANALYZER}:${REPO}/lumen/models/qwen3.py:${REPO}/lumen/ops/fused_swiglu.py:${REPO}/lumen/ops/quantize/__init__.py:${AITER_REPO}/aiter/ops/triton/activation.py:${AITER_REPO}/aiter/ops/triton/_triton_kernels/activation.py"
    for fixture in "${SHAPE_FIXTURES[@]}"; do
        extra_sources+=":${fixture}"
    done
    local -a env_args=(
        EXPERIMENT_ROOT="${ROOT}"
        CACHE_NAMESPACE="${CACHE_NAMESPACE}"
        AITER_CONFIG_CACHE_DIR="${AITER_CACHE_DIR}"
        TUNED_CONFIG_OVERRIDE="${TABLES}"
        EXTRA_SOURCE_FILES="${extra_sources}"
        REQUIRE_GPU_LOCK=1 GPU_LOCK_FD="${gpu_lock_fd}" GPU_LOCK_PATH="${LOCK}"
        REQUIRE_FRESH_CACHE="${fresh_cache}"
        ENABLE_SHAPE_LOG="${shape_log}"
        TRAIN_ENTRYPOINT="${ENTRY}"
        ROUTE_REPORT_TEMPLATE="${ROOT}/${label}/route-rank{rank}.json"
        PROFILE_START= PROFILE_END=
        EXPERIMENT_EXTRA_ARGS="$(policy_args "${policy}")"
        NPROC=8 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_SAMPLES="${TRAIN_SAMPLES}"
        LR=1.0e-4 LR_WARMUP_STEPS=50 SHARDING=full_shard REDUCE_DTYPE=bf16
        RETAIN_ACCUM_PARAMS=1 GRAD_CHECKPOINTING=0 MXFP4_COMM=0
        TAIL_BF16="$(policy_tail "${policy}")" EVAL_INTERVAL="${steps}" EVAL_BATCHES=16 VAL_SAMPLES=256
        PAIRED_EVIDENCE=1 SEED=1234 LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE=0
    )
    [[ -z "${expected_cache_sha}" ]] || env_args+=(EXPECTED_CACHE_SHA256="${expected_cache_sha}")
    [[ -z "${expected_source_sha}" ]] || env_args+=(EXPECTED_SOURCE_BUNDLE_SHA256="${expected_source_sha}")

    set +e
    numactl --cpunodebind=0 --membind=0 env "${env_args[@]}" "${RUNNER}" "${label}" mxfp4 "${steps}"
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
    [[ "$(git -C "${REPO}" rev-parse HEAD)" == "${LUMEN_COMMIT}" ]]
    [[ "$(git -C "${AITER_REPO}" rev-parse HEAD)" == "${AITER_COMMIT}" ]]
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
        echo lumen_tree_sha256="$(meta_value lumen_tree_sha256 "${smoke_meta}")"
        echo aiter_tree_sha256="$(meta_value aiter_tree_sha256 "${smoke_meta}")"
        echo runtime_modules_sha256="$(meta_value runtime_modules_sha256 "${smoke_meta}")"
        echo f4gemm_directory_sha256="$(meta_value f4gemm_directory_sha256 "${smoke_meta}")"
        echo aiter_config_cache_sha256="$(meta_value aiter_config_cache_sha256 "${smoke_meta}")"
        echo formal_order="${FORMAL_ORDER}"
        echo phase1_order="${PHASE1_ORDER}"
        echo formal_steps="${FORMAL_STEPS}"
        echo train_samples="${TRAIN_SAMPLES}"
        echo timing_window=11-50
        echo precision_gate_delta_nll=0.01
        echo speed_gate_min_ratio=1.003
        echo speed_gate_min_wins=28
        echo bootstrap_block_length=4
        echo bootstrap_resamples=100000
        echo replicate_drift_gate_pct=3.0
        echo candidate_stack=packed_qkv+split_swiglu
        echo lm_head_precision=bf16
        echo git_exclude_scope=.codex/
        echo source_integrity=exact_manifest+full_tree_without_codex_runtime_logs
        echo shape_count_contract=whole_smoke_totals
        echo train_pairing_scope=formal_arms_equal_train_samples
        echo kfd_identity_policy=fail_closed_complete_identity
        echo prior_artifacts_role=design_provenance_only
        echo prior_projection_route_sha256="${PRIOR_ROUTE_SHA}"
        echo prior_projection_meta_sha256="${PRIOR_META_SHA}"
        echo numa_balancing="$(< /proc/sys/kernel/numa_balancing)"
        echo driver_sha256="$(sha "${DRIVER}")"
        echo analyzer_sha256="$(sha "${ANALYZER}")"
        echo self_test_sha256="$(sha "${SELF_TEST}")"
        echo protocol_sha256="$(sha "${PROTOCOL}")"
        echo entry_sha256="$(sha "${ENTRY}")"
        echo git_excludes_sha256="$(sha "${GIT_EXCLUDES}")"
        echo shared_runner_sha256="$(sha "${RUNNER}")"
        echo base_analyzer_sha256="$(sha "${BASE_ANALYZER}")"
        echo train_entry_sha256="$(sha "${REPO}/examples/qwen3/train_qwen3_fsdp.py")"
        echo shape_fixture_sha256="$(fixture_digest)"
        stat -c 'kfd=%A:%U:%G' /dev/kfd
    } > "${temporary}"
    mv "${temporary}" "${META}"
}

verify_harness_hashes() {
    local key path expected actual
    while read -r key path; do
        expected=$(meta_value "${key}" "${META}")
        actual=$(sha "${path}")
        [[ "${actual}" == "${expected}" ]] || fail "harness/source changed: ${path}"
    done <<EOF
driver_sha256 ${DRIVER}
analyzer_sha256 ${ANALYZER}
self_test_sha256 ${SELF_TEST}
protocol_sha256 ${PROTOCOL}
entry_sha256 ${ENTRY}
git_excludes_sha256 ${GIT_EXCLUDES}
shared_runner_sha256 ${RUNNER}
base_analyzer_sha256 ${BASE_ANALYZER}
train_entry_sha256 ${REPO}/examples/qwen3/train_qwen3_fsdp.py
EOF
    [[ "$(fixture_digest)" == "$(meta_value shape_fixture_sha256 "${META}")" ]] \
        || fail "shape fixture changed"
}

verify_current_state() {
    local reference=${ROOT}/guard_down_c/run-meta.txt key expected actual
    [[ "$(workload_digest)" == "$(meta_value workload_sha256 "${META}")" ]] || fail "workload changed"
    for key in lumen_tree_sha256 aiter_tree_sha256 aiter_config_cache_sha256 runtime_modules_sha256 f4gemm_directory_sha256; do
        expected=$(meta_value "${key}" "${reference}")
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

verify_phase1() {
    [[ -f "${PHASE1_COMPLETE}" && -f "${ROOT}/projection-guard-phase1-exit-status.txt" ]] || fail "phase1 sentinel/status missing"
    [[ "$(< "${ROOT}/projection-guard-phase1-exit-status.txt")" == 0 ]] || fail "phase1 did not exit zero"
    [[ "$(meta_value completed "${PHASE1_COMPLETE}")" == "${PHASE1_ORDER}" ]] || fail "phase1 order mismatch"
    [[ "$(meta_value source_bundle_sha256 "${PHASE1_COMPLETE}")" == "${expected_source_sha}" ]] || fail "phase1 source mismatch"
    [[ "$(meta_value cache_sha256 "${PHASE1_COMPLETE}")" == "${expected_cache_sha}" ]] || fail "phase1 cache mismatch"
    [[ "$(meta_value meta_sha256 "${PHASE1_COMPLETE}")" == "$(sha "${META}")" ]] || fail "phase1 metadata changed"
}

run_phase1() {
    local path
    [[ ! -e "${ROOT}/cache" ]] || fail "fresh cache root already exists"
    for path in "${SMOKE}" "${FORMAL_CASES[@]}"; do [[ ! -e "${ROOT}/${path}" ]] || fail "output exists: ${path}"; done
    for path in "${META}" "${PHASE1_COMPLETE}" "${FORMAL_COMPLETE}" "${CAMPAIGN_COMPLETE}" "${ROOT}/projection_guard_analysis.json" "${ROOT}/projection_guard_analysis.md"; do
        [[ ! -e "${path}" ]] || fail "campaign artifact exists: ${path}"
    done
    exec {gpu_lock_fd}>"${LOCK}"
    flock -n "${gpu_lock_fd}" || fail "another GPU campaign owns ${LOCK}"
    record_idle_recheck "${ROOT}/kfd-projection-guard-phase1-before.txt" || fail "KFD busy before phase1"
    run_arm "${SMOKE}" b 3 1 1
    [[ -f "${CACHE}" ]] || fail "fresh smoke did not create cache"
    expected_cache_sha=$(sha "${CACHE}")
    expected_source_sha=$(< "${ROOT}/${SMOKE}/source-bundle-before.sha256")
    [[ "${expected_source_sha}" == "$(< "${ROOT}/${SMOKE}/source-bundle-after.sha256")" ]] || fail "smoke source changed"
    write_meta
    /usr/bin/python3 "${ANALYZER}" --validate-smoke
    run_arm tail1_a1 a "${FORMAL_STEPS}" 0 0
    run_arm guard_o_down_b1 b "${FORMAL_STEPS}" 0 0
    run_arm guard_down_c c "${FORMAL_STEPS}" 0 0
    /usr/bin/python3 "${ANALYZER}" --validate-phase1
    record_idle_recheck "${ROOT}/kfd-projection-guard-phase1-after.txt" || fail "KFD busy after phase1"
    write_atomic "${PHASE1_COMPLETE}" schema=1 completed="${PHASE1_ORDER}" source_bundle_sha256="${expected_source_sha}" cache_sha256="${expected_cache_sha}" meta_sha256="$(sha "${META}")" completed_epoch_ns="$(date +%s%N)"
}

run_phase2() {
    [[ -f "${META}" && -f "${CACHE}" ]] || fail "phase1 metadata/cache missing"
    expected_source_sha=$(meta_value source_bundle_sha256 "${META}")
    expected_cache_sha=$(meta_value fresh_autotune_cache_sha256 "${META}")
    [[ "$(sha "${CACHE}")" == "${expected_cache_sha}" ]] || fail "cache changed before phase2"
    verify_harness_hashes
    verify_phase1
    verify_current_state
    /usr/bin/python3 "${ANALYZER}" --validate-phase1
    [[ ! -e "${ROOT}/guard_o_down_b2" && ! -e "${ROOT}/tail1_a2" ]] || fail "phase2 output exists"
    [[ ! -e "${FORMAL_COMPLETE}" && ! -e "${CAMPAIGN_COMPLETE}" ]] || fail "completion artifact exists"
    exec {gpu_lock_fd}>"${LOCK}"
    flock -n "${gpu_lock_fd}" || fail "another GPU campaign owns ${LOCK}"
    record_idle_recheck "${ROOT}/kfd-projection-guard-phase2-before.txt" || fail "KFD busy before phase2"
    run_arm guard_o_down_b2 b "${FORMAL_STEPS}" 0 0
    run_arm tail1_a2 a "${FORMAL_STEPS}" 0 0
    record_idle_recheck "${ROOT}/kfd-projection-guard-phase2-after.txt" || fail "KFD busy after phase2"
    write_atomic "${FORMAL_COMPLETE}" schema=1 completed="${FORMAL_ORDER}" source_bundle_sha256="${expected_source_sha}" cache_sha256="${expected_cache_sha}" completed_epoch_ns="$(date +%s%N)"
    /usr/bin/python3 "${ANALYZER}"
    write_atomic "${CAMPAIGN_COMPLETE}" schema=1 completed="${FORMAL_ORDER}" source_bundle_sha256="${expected_source_sha}" cache_sha256="${expected_cache_sha}" analysis_json_sha256="$(sha "${ROOT}/projection_guard_analysis.json")" analysis_markdown_sha256="$(sha "${ROOT}/projection_guard_analysis.md")" selected_policy="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_policy"])' "${ROOT}/projection_guard_analysis.json")" completed_epoch_ns="$(date +%s%N)"
}

if [[ "${mode}" == dry-run ]]; then
    verify_common_inputs
    echo "dry-run PASS: no GPU process launched"
    echo "formal_order=${FORMAL_ORDER}"
    echo "phase1_order=${PHASE1_ORDER}"
    echo "formal_steps=${FORMAL_STEPS} timing_window=11-50 train_samples=${TRAIN_SAMPLES}"
    exit 0
fi

readonly PHASE_STATUS=${ROOT}/projection-guard-${mode}-exit-status.txt
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

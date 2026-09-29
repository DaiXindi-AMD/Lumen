#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO=/home/xdai/Lumen
readonly AITER_REPO=/home/xdai/aiter
readonly TRAIN_ENTRY=${REPO}/examples/qwen3/train_qwen3_fsdp.py
readonly ROUTE_ENTRY=${ROOT}/route_entry.py
readonly ANALYZER=${ROOT}/analyze_policy_a_profile.py
readonly ANALYZER_TEST=${ROOT}/test_analyze_policy_a_profile.py
readonly PROTOCOL=${ROOT}/protocol.md
readonly TEST_PATTERN='test_*.py'
readonly LOCK=/home/xdai/profile-results/.lumen-gpu-exclusive.lock
readonly LUMEN_COMMIT_EXPECTED=6b9aee1569247eca20937c14319ba6adcd69e0cb
readonly AITER_COMMIT_EXPECTED=e35bb17f4f815903bf73598facedbb321e15af28
readonly MODEL_TABLE=${REPO}/examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv
readonly GENERIC_TABLE=${REPO}/examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv
readonly STOCK_TABLE=${AITER_REPO}/aiter/configs/a4w4_blockscale_tuned_gemm.csv
readonly ASM_MANIFEST=${AITER_REPO}/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv
readonly TUNED_TABLES=${MODEL_TABLE}:${GENERIC_TABLE}:${STOCK_TABLE}
readonly MODEL_DIR=/home/xdai/models/Qwen3-8B
readonly TRAIN_DATA=/home/xdai/fp8-coworker-repro/data/c4_train_1k_repeat4.jsonl
readonly VAL_DATA=/home/xdai/fp8-coworker-repro/data/c4_valid_heldout.jsonl
readonly CACHE_BUILD_DIR=${ROOT}/cache/cache_build_policy_a
readonly CACHE_FILE=${CACHE_BUILD_DIR}/mxfp4-autotune.json
readonly SMOKE_AITER_CACHE=${ROOT}/cache/aiter-configs-smoke-policy_a
readonly BF16_CACHE_DIR=${ROOT}/cache/profile_bf16
readonly MXFP4_CACHE_DIR=${ROOT}/cache/profile_mxfp4_policy_a
readonly BF16_AITER_CACHE=${ROOT}/cache/aiter-configs-profile-bf16
readonly MXFP4_AITER_CACHE=${ROOT}/cache/aiter-configs-profile-mxfp4-policy_a
readonly SOURCE_AUDIT=${ROOT}/source-audit.json
readonly META=${ROOT}/profile-meta.txt
readonly COMPLETE=${ROOT}/profile-complete.txt
readonly EXIT_FILE=${ROOT}/policy_a-profile-exit-status.txt
readonly STATUS_FILE=${ROOT}/campaign-stage-status.txt
readonly PROGRESS=${ROOT}/campaign-progress.log
readonly MANIFEST=${ROOT}/campaign-artifacts.sha256
readonly FORMAL_STEPS=8
readonly SMOKE_STEPS=3
readonly TRAIN_SAMPLES_FORMAL=1024
readonly TRAIN_SAMPLES_SMOKE=640

usage() {
    echo "usage: /usr/bin/bash ${ROOT}/run_policy_a_profile.sh dry-run|run" >&2
}

mode=${1:-}
if [[ "${mode}" != dry-run && "${mode}" != run ]]; then
    usage
    exit 2
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

sha_file() {
    sha256sum "$1" | cut -d' ' -f1
}

write_atomic() {
    local path=$1
    shift
    local temporary=${path}.tmp.$$
    printf '%s\n' "$@" > "${temporary}"
    mv -- "${temporary}" "${path}"
}

directory_digest() {
    local directory=$1
    local -a files=()
    if [[ ! -d "${directory}" ]]; then
        echo absent
        return
    fi
    while IFS= read -r -d '' path; do
        files+=("${path}")
    done < <(find "${directory}" -type f -print0 | sort -z)
    if (( ${#files[@]} == 0 )); then
        echo empty
    else
        sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
    fi
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

runtime_modules_digest() {
    local -a files=()
    while IFS= read -r -d '' path; do
        files+=("${path}")
    done < <(find "${AITER_REPO}/aiter/jit" -maxdepth 1 -type f -name '*.so' -print0 | sort -z)
    (( ${#files[@]} > 0 )) || {
        echo missing
        return
    }
    sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
}

workload_digest() {
    sha256sum \
        "${MODEL_DIR}/config.json" \
        "${MODEL_DIR}/generation_config.json" \
        "${MODEL_DIR}/tokenizer.json" \
        "${MODEL_DIR}/tokenizer_config.json" \
        "${MODEL_DIR}/merges.txt" \
        "${MODEL_DIR}/vocab.json" \
        "${TRAIN_DATA}" \
        "${VAL_DATA}" \
        | sha256sum | cut -d' ' -f1
}

tuned_tables_digest() {
    sha256sum "${MODEL_TABLE}" "${GENERIC_TABLE}" "${STOCK_TABLE}" \
        | sha256sum | cut -d' ' -f1
}

declare -a SOURCE_FILES=()

collect_source_files() {
    SOURCE_FILES=(
        "${BASH_SOURCE[0]}"
        "${ROUTE_ENTRY}"
        "${ANALYZER}"
        "${ANALYZER_TEST}"
        "${PROTOCOL}"
        "${ROOT}/test_runner.py"
        "${TRAIN_ENTRY}"
        "${REPO}/examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh"
        "${REPO}/examples/scripts/train_pretrain.sh"
        "${REPO}/lumen/config.py"
        "${REPO}/lumen/models/fsdp.py"
        "${REPO}/lumen/models/qwen3.py"
        "${REPO}/lumen/quantize/__init__.py"
        "${REPO}/lumen/quantize/config.py"
        "${REPO}/lumen/ops/dispatch.py"
        "${REPO}/lumen/ops/fused_swiglu.py"
        "${REPO}/lumen/ops/quantize/__init__.py"
        "${REPO}/lumen/ops/quantize/linear.py"
        "${REPO}/lumen/ops/quantize/mxfp4_asm.py"
        "${REPO}/lumen/ops/quantize/mxfp4_autotune.py"
        "${AITER_REPO}/aiter/ops/gemm_op_a4w4.py"
        "${AITER_REPO}/aiter/ops/triton/activation.py"
        "${AITER_REPO}/aiter/ops/triton/quant/fused_swiglu_dual_layout_mxfp4.py"
        "${MODEL_TABLE}"
        "${GENERIC_TABLE}"
        "${STOCK_TABLE}"
        "${AITER_REPO}/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
    )
    local path
    while IFS= read -r -d '' path; do
        SOURCE_FILES+=("${path}")
    done < <(find "${AITER_REPO}/aiter/jit" -maxdepth 1 -type f -name '*.so' -print0 | sort -z)
    while IFS= read -r -d '' path; do
        SOURCE_FILES+=("${path}")
    done < <(find "${AITER_REPO}/hsa/gfx950/f4gemm" -maxdepth 1 -type f -print0 | sort -z)

    declare -A seen=()
    local -a unique=()
    for path in "${SOURCE_FILES[@]}"; do
        [[ -f "${path}" ]] || fail "required source input is missing: ${path}"
        path=$(readlink -f "${path}")
        if [[ -z "${seen[${path}]+x}" ]]; then
            seen["${path}"]=1
            unique+=("${path}")
        fi
    done
    mapfile -t SOURCE_FILES < <(printf '%s\n' "${unique[@]}" | sort)
}

source_bundle_digest() {
    sha256sum "${SOURCE_FILES[@]}" | sha256sum | cut -d' ' -f1
}

record_kfd_idle() {
    local output=$1
    local proc_root=/sys/class/kfd/kfd/proc
    local proc_dir pid comm exe cmdline owner ppid workloads
    local busy=0
    {
        date --iso-8601=ns
        if [[ ! -d "${proc_root}" || ! -r "${proc_root}" || ! -x "${proc_root}" ]]; then
            printf 'status=error reason=kfd_proc_unreadable path=%q\n' "${proc_root}"
            return 2
        fi
        while IFS= read -r -d '' proc_dir; do
            pid=${proc_dir##*/}
            if [[ ! -d "/proc/${pid}" ]]; then
                printf 'stale_kfd_entry pid=%s phase=before_read\n' "${pid}"
                continue
            fi
            comm=$(cat -- "/proc/${pid}/comm" 2>/dev/null || true)
            exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
            cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
            owner=$(stat -c %U "/proc/${pid}" 2>/dev/null || true)
            ppid=$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)
            if [[ ! -d "/proc/${pid}" ]]; then
                printf 'stale_kfd_entry pid=%s phase=after_read\n' "${pid}"
                continue
            fi
            if [[ "${comm}" == gpuagent \
                  && ( "${exe}" == /usr/local/bin/gpuagent \
                       || ( -z "${exe}" && "${cmdline% }" == /usr/local/bin/gpuagent ) ) \
                  && "${owner}" == root && "${ppid}" == 1 ]]; then
                printf 'allowed_service pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q\n' \
                    "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}"
            else
                printf 'non_service_kfd_client pid=%s comm=%q owner=%q ppid=%q exe=%q cmdline=%q\n' \
                    "${pid}" "${comm}" "${owner}" "${ppid}" "${exe}" "${cmdline}"
                busy=1
            fi
        done < <(find "${proc_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
        workloads=$(pgrep -af '[t]orchrun|[t]rain_qwen3_fsdp\.py|[r]oute_entry\.py|[r]ocprof(v3)?|[n]sys profile' || true)
        if [[ -n "${workloads}" ]]; then
            printf 'supplemental_known_workloads=%q\n' "${workloads}"
            busy=1
        fi
        if (( busy == 0 )); then
            echo status=idle
        else
            echo status=busy
        fi
    } > "${output}"
    (( busy == 0 ))
}

record_kfd_idle_with_retry() {
    local output=$1
    local attempt
    for attempt in 1 2 3 4 5; do
        if record_kfd_idle "${output}"; then
            echo "${attempt}"
            return 0
        fi
        if (( attempt < 5 )); then
            sleep 1
        fi
    done
    echo 5
    return 1
}

snapshot_state() {
    local output=$1
    {
        echo "lumen_commit=$(git -C "${REPO}" rev-parse HEAD)"
        echo "lumen_branch=$(git -C "${REPO}" branch --show-current)"
        echo "aiter_commit=$(git -C "${AITER_REPO}" rev-parse HEAD)"
        echo "lumen_tree_sha256=$(repo_state_digest "${REPO}")"
        echo "aiter_tree_sha256=$(repo_state_digest "${AITER_REPO}")"
        echo "source_bundle_sha256=$(source_bundle_digest)"
        echo "runtime_modules_sha256=$(runtime_modules_digest)"
        echo "f4gemm_directory_sha256=$(directory_digest "${AITER_REPO}/hsa/gfx950/f4gemm")"
        echo "tuned_tables_sha256=$(tuned_tables_digest)"
    } > "${output}"
}

audit_value() {
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - "${SOURCE_AUDIT}" "$1" <<'PY'
import json
import sys

value = json.loads(open(sys.argv[1], encoding="utf-8").read())
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

create_source_audit() {
    local temporary=${SOURCE_AUDIT}.tmp.$$
    local imports=${ROOT}/imports-before.txt
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPO}:${AITER_REPO}" \
        /usr/bin/python3 -B - <<'PY' > "${imports}"
import aiter
import lumen
import torch

print(f"lumen={lumen.__file__}")
print(f"aiter={aiter.__file__}")
print(f"torch={torch.__version__}")
print(f"hip={torch.version.hip}")
print(f"devices={torch.cuda.device_count()}")
PY
    local lumen_import aiter_import torch_version hip_version devices
    lumen_import=$(awk -F= '$1=="lumen" {print $2}' "${imports}")
    aiter_import=$(awk -F= '$1=="aiter" {print $2}' "${imports}")
    torch_version=$(awk -F= '$1=="torch" {print $2}' "${imports}")
    hip_version=$(awk -F= '$1=="hip" {print $2}' "${imports}")
    devices=$(awk -F= '$1=="devices" {print $2}' "${imports}")
    [[ "${lumen_import}" == "${REPO}/lumen/__init__.py" ]] || fail "unexpected Lumen import: ${lumen_import}"
    [[ "${aiter_import}" == "${AITER_REPO}/aiter/__init__.py" ]] || fail "unexpected AITER import: ${aiter_import}"
    [[ "${devices}" == 8 ]] || fail "expected 8 visible GPUs, got ${devices}"

    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - \
        "${temporary}" \
        "$(git -C "${REPO}" rev-parse HEAD)" \
        "$(git -C "${REPO}" branch --show-current)" \
        "$(git -C "${AITER_REPO}" rev-parse HEAD)" \
        "$(repo_state_digest "${REPO}")" \
        "$(repo_state_digest "${AITER_REPO}")" \
        "$(source_bundle_digest)" \
        "$(runtime_modules_digest)" \
        "$(directory_digest "${AITER_REPO}/hsa/gfx950/f4gemm")" \
        "$(tuned_tables_digest)" \
        "$(workload_digest)" \
        "${lumen_import}" "${aiter_import}" "${torch_version}" "${hip_version}" "${devices}" \
        "${BASH_SOURCE[0]}" "${ROUTE_ENTRY}" "${ANALYZER}" "${ANALYZER_TEST}" \
        "${PROTOCOL}" "${ROOT}/test_runner.py" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    output,
    lumen_commit,
    lumen_branch,
    aiter_commit,
    lumen_tree,
    aiter_tree,
    source_bundle,
    runtime_modules,
    f4gemm,
    tuned_tables,
    workload,
    lumen_import,
    aiter_import,
    torch_version,
    hip_version,
    devices,
    *harness_paths,
) = sys.argv[1:]

def digest(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

payload = {
    "schema": 1,
    "lumen_commit": lumen_commit,
    "lumen_branch": lumen_branch,
    "aiter_commit": aiter_commit,
    "lumen_tree_sha256": lumen_tree,
    "aiter_tree_sha256": aiter_tree,
    "source_bundle_sha256": source_bundle,
    "runtime_modules_sha256": runtime_modules,
    "f4gemm_directory_sha256": f4gemm,
    "tuned_tables_sha256": tuned_tables,
    "workload_sha256": workload,
    "imports": {"lumen": lumen_import, "aiter": aiter_import},
    "runtime": {"torch": torch_version, "hip": hip_version, "devices": int(devices)},
    "harness_sha256": {path: digest(path) for path in harness_paths},
}
path = pathlib.Path(output)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    mv -- "${temporary}" "${SOURCE_AUDIT}"
}

verify_current_state() {
    local deep=${1:-0}
    [[ "$(git -C "${REPO}" rev-parse HEAD)" == "$(audit_value lumen_commit)" ]] || return 1
    [[ "$(git -C "${REPO}" branch --show-current)" == "$(audit_value lumen_branch)" ]] || return 1
    [[ "$(git -C "${AITER_REPO}" rev-parse HEAD)" == "$(audit_value aiter_commit)" ]] || return 1
    [[ "$(repo_state_digest "${REPO}")" == "$(audit_value lumen_tree_sha256)" ]] || return 1
    [[ "$(repo_state_digest "${AITER_REPO}")" == "$(audit_value aiter_tree_sha256)" ]] || return 1
    [[ "$(source_bundle_digest)" == "$(audit_value source_bundle_sha256)" ]] || return 1
    [[ "$(runtime_modules_digest)" == "$(audit_value runtime_modules_sha256)" ]] || return 1
    [[ "$(directory_digest "${AITER_REPO}/hsa/gfx950/f4gemm")" == "$(audit_value f4gemm_directory_sha256)" ]] || return 1
    [[ "$(tuned_tables_digest)" == "$(audit_value tuned_tables_sha256)" ]] || return 1
    if [[ "${deep}" == 1 ]]; then
        [[ "$(workload_digest)" == "$(audit_value workload_sha256)" ]] || return 1
    fi
    return 0
}

validate_cache_structure() {
    local cache=$1
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - \
        "${cache}" "${ASM_MANIFEST}" <<'PY'
import csv
import hashlib
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != 6:
    raise SystemExit("unexpected MXFP4 cache schema")
if payload.get("arch") != "gfx950":
    raise SystemExit("cache was not produced for gfx950")
choices = payload.get("choices")
profiles = payload.get("profiles")
if not isinstance(choices, dict) or not choices:
    raise SystemExit("cache contains no backend choices")
if not isinstance(profiles, dict) or set(profiles) != set(choices):
    raise SystemExit("cache profile keys do not exactly match choice keys")

manifest_bytes = manifest.read_bytes()
# Lumen's persisted runtime identities intentionally use 16-hex SHA256
# prefixes (see mxfp4_asm._file_sha256), while the campaign separately freezes
# the full f4gemm directory digest. Compare the exact persisted representation
# here, then rely on the full-directory digest to detect any collision/change.
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()[:16]
symbol_re = re.compile(
    r"^_ZN5aiter\d+f4gemm_bf16_per1x32Fp4_BpreShuffle_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)E$"
)
manifest_rows = {}
ambiguous = set()
with manifest.open(newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source):
        symbol = row.get("knl_name", "")
        match = symbol_re.fullmatch(symbol)
        try:
            tile_m = int(row.get("tile_M", ""))
            tile_n = int(row.get("tile_N", ""))
            split_capable = int(row.get("splitK", ""))
            bpreshuffle = int(row.get("bpreshuffle", ""))
        except ValueError:
            continue
        code_object = row.get("co_name", "")
        if (
            match is None
            or tile_m != int(match.group("tile_m"))
            or tile_n != int(match.group("tile_n"))
            or split_capable not in (0, 1)
            or bpreshuffle != 1
            or not code_object
            or pathlib.Path(code_object).name != code_object
            or symbol in ambiguous
        ):
            continue
        if symbol in manifest_rows:
            manifest_rows.pop(symbol, None)
            ambiguous.add(symbol)
            continue
        code_path = manifest.parent / code_object
        if not code_path.is_file():
            continue
        manifest_rows[symbol] = {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "split_k_capable": bool(split_capable),
            "code_object": code_object,
            "code_object_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest()[:16],
        }

for shape, backend in choices.items():
    if backend not in {"asm", "shuffled"}:
        raise SystemExit(f"unexpected backend for {shape}: {backend}")
    profile = profiles[shape]
    identities = profile.get("identities", {})
    identity = identities.get(backend)
    if not isinstance(identity, dict):
        raise SystemExit(f"missing selected identity for {shape}/{backend}")
    if backend == "asm":
        required = (
            "implementation",
            "kernel_name",
            "tile_m",
            "tile_n",
            "log2_k_split",
            "split_k_capable",
            "manifest_sha256",
            "code_object",
            "code_object_sha256",
        )
        if any(key not in identity for key in required):
            raise SystemExit(f"incomplete ASM identity for {shape}")
        if identity["implementation"] != "asm":
            raise SystemExit(f"wrong ASM implementation identity for {shape}")
        symbol = identity["kernel_name"]
        row = manifest_rows.get(symbol)
        if row is None:
            raise SystemExit(f"ASM symbol absent or ambiguous in live manifest for {shape}")
        if identity["manifest_sha256"] != manifest_sha256:
            raise SystemExit(f"ASM manifest hash mismatch for {shape}")
        for key in (
            "tile_m",
            "tile_n",
            "split_k_capable",
            "code_object",
            "code_object_sha256",
        ):
            if identity[key] != row[key]:
                raise SystemExit(f"ASM live artifact identity mismatch for {shape}: {key}")
        split = identity["log2_k_split"]
        if isinstance(split, bool) or not isinstance(split, int) or split < 0:
            raise SystemExit(f"invalid ASM split policy for {shape}")
        if not row["split_k_capable"] and split != 0:
            raise SystemExit(f"unsupported ASM split policy for {shape}")
PY
}

validate_route_arm() {
    local arm=$1
    local replay=${2:-0}
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - "${ROOT}/${arm}" "${replay}" <<'PY'
import csv
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
is_replay = sys.argv[2] == "1"
log = (root / "train.log").read_text(encoding="utf-8", errors="replace")
if log.count("Quantization enabled on 245 nn.Linear layers") != 1:
    raise SystemExit("Policy A quantized-linear count is not exactly 245")
if log.count("bf16_layers_skipped=8") != 1:
    raise SystemExit("Policy A BF16 skip count is not exactly 8")
if log.count("> MXFP4 packed QKV enabled on 35 Qwen3 attention layers") != 1:
    raise SystemExit("packed QKV did not enable on exactly 35 layers")
if log.count("> MXFP4 split SwiGLU enabled on 35 Qwen3 MLPs") != 1:
    raise SystemExit("split SwiGLU did not enable on exactly 35 layers")
qkv_warning = "all Q/K/V projections must already be quantized"
swiglu_warning = "both projections must already be quantized"
if log.count(qkv_warning) != 8 or log.count(swiglu_warning) != 8:
    raise SystemExit("protected-tail warning count is not exactly one per rank and integration")
if is_replay and re.search(r"MXFP4 autotune \d+x\d+x\d+:\s", log):
    raise SystemExit("cache replay performed online MXFP4 autotuning")

bad = (
    r"out of memory|\boom\b|\bnan\b|(?<![a-z])[-+]?inf(?![a-z])|"
    r"skipped update|kernel launch failed|kernel failure"
)
prefix = log.split("Training complete", 1)[0]
if re.search(bad, prefix, flags=re.IGNORECASE):
    raise SystemExit("training failure marker found before completion")

shape_bytes = []
for rank in range(8):
    report_path = root / f"route-rank{rank}.json"
    shape_path = root / f"mxfp4-shapes-rank{rank}.csv"
    if not report_path.is_file() or not shape_path.is_file():
        raise SystemExit(f"rank {rank} route or shape artifact is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report.get("counts", {})
    expected = {
        "rank": rank,
        "world_size": 8,
        "instrumentation_installed": 1,
        "enabled_qkv": 35,
        "enabled_swiglu": 35,
    }
    for key, value in expected.items():
        if counts.get(key) != value:
            raise SystemExit(f"rank {rank} bad route counter {key}: {counts.get(key)}")
    for key in (
        "qkv_linear_failure",
        "swiglu_fwd_failure",
        "swiglu_bwd_failure",
        "eligible_original_qkv_forward",
        "eligible_original_swiglu_forward",
    ):
        if counts.get(key, 0) != 0:
            raise SystemExit(f"rank {rank} unexpected route counter {key}")
    exact_success_counts = {
        "qkv_linear_success": 1400,
        "swiglu_fwd_success": 1400,
        "swiglu_bwd_success": 840,
    }
    for key, value in exact_success_counts.items():
        if counts.get(key) != value:
            raise SystemExit(
                f"rank {rank} bad route counter {key}: "
                f"expected {value}, got {counts.get(key)}"
            )
    names = report.get("unquantized_linear_names", [])
    if len(names) != 8 or names.count("lm_head") != 1:
        raise SystemExit(f"rank {rank} unquantized linear inventory is not Policy A+lm_head")
    tail_names = [name for name in names if name != "lm_head"]
    if len(tail_names) != 7 or any(
        re.search(r"layers\.35\.", name) is None for name in tail_names
    ):
        raise SystemExit(f"rank {rank} unquantized transformer inventory is not layer 35")
    if report.get("lm_head_count") != 1:
        raise SystemExit(f"rank {rank} did not identify exactly one lm_head")
    if report.get("lm_head_weight_dtype") != "bfloat16":
        raise SystemExit(f"rank {rank} lm_head is not BF16")
    if report.get("lm_head_quant_enabled") is not False:
        raise SystemExit(f"rank {rank} lm_head is quantized")
    rows = list(csv.reader(shape_path.open(newline="", encoding="utf-8")))
    if len(rows) < 2:
        raise SystemExit(f"rank {rank} shape CSV is empty")
    shape_bytes.append(shape_path.read_bytes())
if any(content != shape_bytes[0] for content in shape_bytes[1:]):
    raise SystemExit("rank-local shape CSV files differ")
PY
}

validate_replay_identity() {
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - \
        "${ROOT}/smoke_mxfp4_policy_a" "${ROOT}/replay_mxfp4_policy_a" <<'PY'
import hashlib
import json
import pathlib
import sys

left, right = map(pathlib.Path, sys.argv[1:])
for rank in range(8):
    left_route = json.loads((left / f"route-rank{rank}.json").read_text())
    right_route = json.loads((right / f"route-rank{rank}.json").read_text())
    if left_route != right_route:
        raise SystemExit(f"rank {rank} route report changed on replay")
    left_shape = (left / f"mxfp4-shapes-rank{rank}.csv").read_bytes()
    right_shape = (right / f"mxfp4-shapes-rank{rank}.csv").read_bytes()
    if hashlib.sha256(left_shape).digest() != hashlib.sha256(right_shape).digest():
        raise SystemExit(f"rank {rank} shape log changed on replay")
PY
}

normalize_and_compare_formal_commands() {
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B - \
        "${ROOT}/profile_bf16/run-meta.txt" "${ROOT}/profile_mxfp4_policy_a/run-meta.txt" <<'PY'
import shlex
import sys

def meta(path):
    result = {}
    for line in open(path, encoding="utf-8"):
        if "=" in line:
            key, value = line.rstrip("\n").split("=", 1)
            result[key] = value
    return result

def normalized(command):
    tokens = shlex.split(command)
    output = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--mode":
            output.extend((token, "<precision>"))
            index += 2
            continue
        if token in {"--mxfp4-pack-qkv", "--mxfp4-fuse-swiglu"}:
            index += 1
            continue
        output.append(token)
        index += 1
    return output

left = meta(sys.argv[1])
right = meta(sys.argv[2])
if normalized(left["command"]) != normalized(right["command"]):
    raise SystemExit("formal commands differ beyond precision and MXFP4 integration flags")
if left.get("profile_start") != "7" or left.get("profile_end") != "8":
    raise SystemExit("BF16 profiler window is not steps 7-8")
if right.get("profile_start") != "7" or right.get("profile_end") != "8":
    raise SystemExit("MXFP4 profiler window is not steps 7-8")
PY
}

run_case() {
    local label=$1 precision=$2 steps=$3 cache_dir=$4 aiter_cache=$5
    local require_fresh_cache=$6 shape_log=$7 profile=$8 expected_cache=${9:-}
    local expected_aiter=${10:-}
    local run_dir=${ROOT}/${label}
    local train_entry=${TRAIN_ENTRY}
    local -a extra_args=()

    [[ ! -e "${run_dir}" && ! -L "${run_dir}" ]] || fail "arm already exists: ${run_dir}"
    if [[ "${require_fresh_cache}" == 1 ]]; then
        [[ ! -e "${cache_dir}" && ! -L "${cache_dir}" ]] || fail "fresh cache directory already exists: ${cache_dir}"
    fi
    if [[ "${shape_log}" == 1 ]]; then
        train_entry=${ROUTE_ENTRY}
    fi
    if [[ "${precision}" == mxfp4 ]]; then
        extra_args+=(--mxfp4-pack-qkv --mxfp4-fuse-swiglu)
    fi

    verify_current_state 1 || fail "${label}: frozen source/workload state changed before launch"
    mkdir -p "${run_dir}" "${cache_dir}/xdg" "${cache_dir}/triton" \
        "${cache_dir}/inductor" "${cache_dir}/tmp" "${aiter_cache}"

    local cache_before=absent
    if [[ -f "${cache_dir}/mxfp4-autotune.json" ]]; then
        cache_before=$(sha_file "${cache_dir}/mxfp4-autotune.json")
        sha256sum "${cache_dir}/mxfp4-autotune.json" > "${run_dir}/cache-before.sha256"
    else
        echo absent > "${run_dir}/cache-before.sha256"
    fi
    [[ -z "${expected_cache}" || "${cache_before}" == "${expected_cache}" ]] \
        || fail "${label}: cache before mismatch"
    local aiter_before
    aiter_before=$(directory_digest "${aiter_cache}")
    [[ -z "${expected_aiter}" || "${aiter_before}" == "${expected_aiter}" ]] \
        || fail "${label}: AITER config cache before mismatch"

    record_kfd_idle "${run_dir}/kfd-before.txt" || fail "KFD busy before ${label}"
    echo "$(source_bundle_digest)" > "${run_dir}/source-bundle-before.sha256"
    local source_expected
    source_expected=$(audit_value source_bundle_sha256)
    [[ "$(< "${run_dir}/source-bundle-before.sha256")" == "${source_expected}" ]] \
        || fail "${label}: source bundle mismatch before launch"

    local nproc=8 mbs=2 gbs=128 seq_len=8192
    local grad_accum=$((gbs / (nproc * mbs)))
    local train_samples=${TRAIN_SAMPLES_FORMAL}
    if (( steps == SMOKE_STEPS )); then
        train_samples=${TRAIN_SAMPLES_SMOKE}
    fi
    local -a args=(
        --model-name-or-path "${MODEL_DIR}"
        --tokenizer-name-or-path "${MODEL_DIR}"
        --task pretrain
        --init-from-scratch
        --train-data-path "${TRAIN_DATA}"
        --val-data-path "${VAL_DATA}"
        --seq-length "${seq_len}"
        --micro-batch-size "${mbs}"
        --gradient-accumulation-steps "${grad_accum}"
        --max-steps "${steps}"
        --lr 1.0e-4
        --min-lr 0
        --lr-warmup-steps 50
        --weight-decay 0.1
        --max-grad-norm 1.0
        --lora-rank 0
        --fsdp-version 2
        --sharding full_shard
        --fsdp-reduce-dtype bf16
        --fsdp-retain-accumulated-params
        --no-grad-checkpointing
        --aiter-attn
        --lumen-norm
        --fuse-rope
        --fused-cross-entropy
        --first-last-layers-bf16
        --num-layers-at-start-in-bf16 0
        --num-layers-at-end-in-bf16 1
        --train-samples "${train_samples}"
        --num-workers 0
        --log-interval 1
        --eval-interval "${steps}"
        --val-samples 256
        --eval-batches 16
        --seed 1234
        --mode "${precision}"
        "${extra_args[@]}"
    )

    local -a profile_env=()
    if [[ "${profile}" == 1 ]]; then
        profile_env+=(
            LUMEN_PROF_START=7
            LUMEN_PROF_END=8
            LUMEN_PROF_OUTPUT="${run_dir}/profile.txt"
            LUMEN_PROF_TRACE="${run_dir}/trace.json"
            LUMEN_PROF_SHAPES=1
            LUMEN_COPY_TRACE=0
        )
    fi
    local -a shape_env=()
    if [[ "${shape_log}" == 1 ]]; then
        shape_env+=(
            LUMEN_MXFP4_GEMM_SHAPE_LOG_TEMPLATE="${run_dir}/mxfp4-shapes-rank{rank}.csv"
            ROUTE_REPORT_TEMPLATE="${run_dir}/route-rank{rank}.json"
        )
    fi
    printf -v command_string '%q ' /usr/local/bin/torchrun --standalone --nnodes=1 \
        --nproc-per-node=8 "${train_entry}" "${args[@]}"
    {
        date --iso-8601=ns
        echo "case=${label}"
        echo "precision=${precision}"
        echo "command=${command_string}"
        echo "global_batch=128"
        echo "gradient_accumulation=8"
        echo "tokens_per_update=1048576"
        echo "train_samples=${train_samples}"
        echo "tail_bf16=1"
        echo "lm_head_precision=bf16"
        echo "profile_start=$([[ "${profile}" == 1 ]] && echo 7 || echo none)"
        echo "profile_end=$([[ "${profile}" == 1 ]] && echo 8 || echo none)"
        echo "profile_shapes=${profile}"
        echo "shape_log_enabled=${shape_log}"
        echo "cache_file=${cache_dir}/mxfp4-autotune.json"
        echo "cache_sha256_before=${cache_before}"
        echo "aiter_config_cache_dir=${aiter_cache}"
        echo "aiter_config_cache_sha256_before=${aiter_before}"
        echo "source_bundle_sha256=${source_expected}"
        echo "lumen_commit=$(audit_value lumen_commit)"
        echo "aiter_commit=$(audit_value aiter_commit)"
        echo "lumen_tree_sha256=$(audit_value lumen_tree_sha256)"
        echo "aiter_tree_sha256=$(audit_value aiter_tree_sha256)"
        echo "runtime_modules_sha256=$(audit_value runtime_modules_sha256)"
        echo "f4gemm_directory_sha256=$(audit_value f4gemm_directory_sha256)"
        echo "tuned_tables_sha256=$(audit_value tuned_tables_sha256)"
        echo "workload_sha256=$(audit_value workload_sha256)"
        echo "eval_batches=16"
        echo "val_samples=256"
        echo "seed=1234"
        echo "numa_cpu_node=0"
        echo "numa_memory_node=0"
        stat -c 'kfd=%A:%U:%G' /dev/kfd
    } > "${run_dir}/run-meta.txt"

    numactl --cpunodebind=0 --membind=0 numactl --show > "${run_dir}/numa-policy.txt"
    rocm-smi --showproductname --showuse --showmemuse --showpids \
        > "${run_dir}/rocm-smi-before.txt"
    record_kfd_idle "${run_dir}/kfd-prelaunch.txt" || fail "KFD busy immediately before ${label}"

    local -a sanitize_env=()
    local env_name
    while IFS='=' read -r env_name _; do
        case "${env_name}" in
            LUMEN_*|AITER_*|CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|NCCL_*|TORCH_*|PYTORCH_*|TRITON_*|XDG_CACHE_HOME|TMPDIR|LD_PRELOAD|HSA_TOOLS_LIB|ROCR_TOOL_LIB|ROCPROFILER_*|CUDA_LAUNCH_BLOCKING|AMD_SERIALIZE_*|HSA_NO_SCRATCH_RECLAIM|HIP_FORCE_DEV_KERNARG|GPU_MAX_HW_QUEUES|CUDA_DEVICE_MAX_CONNECTIONS|OMP_NUM_THREADS|USE_HIPBLASLT|ENABLE_CK|FP4_DMA_INTRINSIC)
                sanitize_env+=(-u "${env_name}")
                ;;
        esac
    done < <(env)

    set +e
    numactl --cpunodebind=0 --membind=0 env \
        "${sanitize_env[@]}" \
        "${profile_env[@]}" \
        "${shape_env[@]}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${REPO}:${AITER_REPO}" \
        PYTHONHASHSEED=1234 \
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
        AITER_CONFIG_GEMM_A4W4="${TUNED_TABLES}" \
        AITER_CONFIG_CACHE_DIR="${aiter_cache}" \
        LUMEN_FAST_QUANT_DISPATCH=1 LUMEN_SKIP_BACKEND_SYNC=0 \
        LUMEN_USE_APEX_RMSNORM=0 LUMEN_WEIGHT_QUANT_ONCE=0 \
        LUMEN_MXFP4_ALLOW_UNVALIDATED_ARCH=0 \
        LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=0 LUMEN_MXFP4_DGRAD_HADAMARD=0 \
        LUMEN_MXFP4_ACTIVATION_DESCRIPTOR_CACHE=0 \
        LUMEN_SR_PHILOX_ROUNDS=7 LUMEN_MXFP4_AUTOTUNE=1 \
        LUMEN_MXFP4_FLYDSL=0 LUMEN_MXFP4_PROFILE_WARMUP=3 \
        LUMEN_MXFP4_PROFILE_ITERS=11 LUMEN_MXFP4_REQUIRE_CONSENSUS=1 \
        LUMEN_MXFP4_AUTOTUNE_CACHE="${cache_dir}/mxfp4-autotune.json" \
        LUMEN_PAIRED_RUN_EVIDENCE=1 \
        XDG_CACHE_HOME="${cache_dir}/xdg" \
        TRITON_CACHE_DIR="${cache_dir}/triton" \
        TORCHINDUCTOR_CACHE_DIR="${cache_dir}/inductor" \
        TMPDIR="${cache_dir}/tmp" \
        /usr/local/bin/torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        "${train_entry}" "${args[@]}" 2>&1 | tee "${run_dir}/train.log"
    local -a pipeline_status=("${PIPESTATUS[@]}")
    set -e
    local train_status=${pipeline_status[0]}
    local tee_status=${pipeline_status[1]}
    {
        echo "torchrun=${train_status}"
        echo "tee=${tee_status}"
    } > "${run_dir}/train-exit-status.txt"

    set +e
    rocm-smi --showproductname --showuse --showmemuse --showpids \
        > "${run_dir}/rocm-smi-after.txt"
    local rocm_status=$?
    set -e
    local post_idle_status=0 attempts
    if ! attempts=$(record_kfd_idle_with_retry "${run_dir}/kfd-after.txt"); then
        post_idle_status=1
    fi
    echo "${attempts}" > "${run_dir}/kfd-after-attempts.txt"

    local cache_after=absent
    if [[ -f "${cache_dir}/mxfp4-autotune.json" ]]; then
        cache_after=$(sha_file "${cache_dir}/mxfp4-autotune.json")
        sha256sum "${cache_dir}/mxfp4-autotune.json" > "${run_dir}/cache-after.sha256"
    else
        echo absent > "${run_dir}/cache-after.sha256"
    fi
    local source_after
    source_after=$(source_bundle_digest)
    echo "${source_after}" > "${run_dir}/source-bundle-after.sha256"
    snapshot_state "${run_dir}/tree-state-after.txt"
    local aiter_after
    aiter_after=$(directory_digest "${aiter_cache}")

    local provenance_status=0
    [[ "${source_after}" == "${source_expected}" ]] || provenance_status=4
    [[ -z "${expected_cache}" || "${cache_after}" == "${expected_cache}" ]] \
        || provenance_status=8
    [[ -z "${expected_aiter}" || "${aiter_after}" == "${expected_aiter}" ]] \
        || provenance_status=9
    if ! verify_current_state 1; then
        provenance_status=7
    fi
    if [[ "${profile}" == 0 ]]; then
        write_atomic "${run_dir}/trace.json" \
            '{"schema":1,"profiling_enabled":false,"reason":"route-cache arm"}'
        write_atomic "${run_dir}/profile.txt" \
            profiling_enabled=0 \
            reason=route-cache-arm
    fi
    {
        echo "torchrun=${train_status}"
        echo "tee=${tee_status}"
        echo "rocm_smi=${rocm_status}"
        echo "post_idle=${post_idle_status}"
        echo "provenance=${provenance_status}"
        echo "cache_sha256_after=${cache_after}"
        echo "aiter_config_cache_sha256_after=${aiter_after}"
    } > "${run_dir}/postflight-status.txt"

    (( train_status == 0 )) || return "${train_status}"
    (( tee_status == 0 )) || return 21
    (( rocm_status == 0 )) || return 23
    (( post_idle_status == 0 )) || return 22
    (( provenance_status == 0 )) || return "${provenance_status}"
}

run_dry_run() {
    command -v bash >/dev/null
    command -v git >/dev/null
    command -v sha256sum >/dev/null
    command -v flock >/dev/null
    command -v numactl >/dev/null
    command -v rocm-smi >/dev/null
    command -v /usr/local/bin/torchrun >/dev/null
    command -v /usr/bin/python3 >/dev/null
    for path in \
        "${BASH_SOURCE[0]}" "${ROUTE_ENTRY}" "${ANALYZER}" "${ANALYZER_TEST}" "${PROTOCOL}" \
        "${ROOT}/test_runner.py" "${TRAIN_ENTRY}" "${MODEL_TABLE}" \
        "${GENERIC_TABLE}" "${STOCK_TABLE}" "${MODEL_DIR}/config.json" \
        "${MODEL_DIR}/tokenizer.json" "${TRAIN_DATA}" "${VAL_DATA}"; do
        [[ -f "${path}" ]] || fail "dry-run required file missing: ${path}"
    done
    [[ "$(git -C "${REPO}" branch --show-current)" == dev/mxfp4 ]] \
        || fail "Lumen must be on dev/mxfp4"
    [[ "$(git -C "${REPO}" rev-parse HEAD)" == "${LUMEN_COMMIT_EXPECTED}" ]] \
        || fail "unexpected Lumen commit"
    [[ "$(git -C "${AITER_REPO}" rev-parse HEAD)" == "${AITER_COMMIT_EXPECTED}" ]] \
        || fail "unexpected AITER commit"
    /usr/bin/bash -n "${BASH_SOURCE[0]}"
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest discover \
        -s "${ROOT}" -p "${TEST_PATTERN}" -v
    echo "dry-run: PASS"
}

if [[ "${mode}" == dry-run ]]; then
    run_dry_run
    exit 0
fi

while IFS= read -r -d '' existing; do
    case "${existing##*/}" in
        protocol.md|run_policy_a_profile.sh|route_entry.py|analyze_policy_a_profile.py|test_analyze_policy_a_profile.py|test_runner.py)
            ;;
        *)
            fail "unexpected pre-existing campaign-root entry: ${existing}"
            ;;
    esac
done < <(find "${ROOT}" -mindepth 1 -maxdepth 1 -print0 | sort -z)

for path in \
    "${ROOT}/smoke_mxfp4_policy_a" \
    "${ROOT}/replay_mxfp4_policy_a" \
    "${ROOT}/profile_bf16" \
    "${ROOT}/profile_mxfp4_policy_a" \
    "${ROOT}/cache" \
    "${SOURCE_AUDIT}" "${META}" "${COMPLETE}" "${EXIT_FILE}" \
    "${STATUS_FILE}" "${PROGRESS}" "${MANIFEST}" \
    "${ROOT}/imports-before.txt" \
    "${ROOT}/kfd-campaign-before.txt" "${ROOT}/kfd-campaign-after.txt" \
    "${ROOT}/profile_analysis.json" "${ROOT}/profile_analysis.md"; do
    [[ ! -e "${path}" && ! -L "${path}" ]] || fail "fresh output already exists: ${path}"
done

current_stage=preflight
finish() {
    local status=$?
    trap - EXIT
    if [[ -f "${STATUS_FILE}" && "${status}" != 0 ]]; then
        printf 'failed_stage=%s\nfailed_status=%s\n' "${current_stage}" "${status}" \
            >> "${STATUS_FILE}"
    fi
    write_atomic "${EXIT_FILE}" "${status}"
    exit "${status}"
}
trap finish EXIT

run_dry_run
(( TRAIN_SAMPLES_FORMAL == FORMAL_STEPS * 128 )) \
    || fail "formal train samples must exactly cover all optimizer updates"
bytecode_artifact=$(find "${ROOT}" \
    \( -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
    -print -quit)
[[ -z "${bytecode_artifact}" ]] \
    || fail "Python bytecode artifacts must be absent before the campaign: ${bytecode_artifact}"
[[ -r /dev/kfd && -w /dev/kfd ]] || fail "/dev/kfd must be readable and writable"
[[ -r /sys/class/kfd/kfd/proc ]] || fail "KFD process state must be readable"
exec {gpu_lock_fd}>"${LOCK}"
flock -n "${gpu_lock_fd}" || fail "another GPU campaign owns ${LOCK}"

collect_source_files
[[ "$(git -C "${REPO}" branch --show-current)" == dev/mxfp4 ]] || fail "Lumen branch changed"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${LUMEN_COMMIT_EXPECTED}" ]] || fail "Lumen commit changed"
[[ "$(git -C "${AITER_REPO}" rev-parse HEAD)" == "${AITER_COMMIT_EXPECTED}" ]] || fail "AITER commit changed"
record_kfd_idle "${ROOT}/kfd-campaign-before.txt" || fail "KFD busy before campaign"
create_source_audit
verify_current_state 1 || fail "frozen source state changed while creating the audit"
write_atomic "${STATUS_FILE}" preflight=0
write_atomic "${PROGRESS}" "$(date --iso-8601=ns) stage=preflight_complete"

current_stage=smoke_mxfp4_policy_a
run_case smoke_mxfp4_policy_a mxfp4 "${SMOKE_STEPS}" "${CACHE_BUILD_DIR}" \
    "${SMOKE_AITER_CACHE}" 1 1 0 "" ""
validate_route_arm smoke_mxfp4_policy_a 0
[[ -f "${CACHE_FILE}" ]] || fail "cache-build smoke produced no MXFP4 cache"
validate_cache_structure "${CACHE_FILE}"
readonly CACHE_SHA=$(sha_file "${CACHE_FILE}")
readonly SMOKE_AITER_SHA=$(directory_digest "${SMOKE_AITER_CACHE}")
printf 'smoke_mxfp4_policy_a=0\n' >> "${STATUS_FILE}"
printf '%s stage=smoke_mxfp4_policy_a_complete\n' "$(date --iso-8601=ns)" >> "${PROGRESS}"

current_stage=replay_mxfp4_policy_a
run_case replay_mxfp4_policy_a mxfp4 "${SMOKE_STEPS}" "${CACHE_BUILD_DIR}" \
    "${SMOKE_AITER_CACHE}" 0 1 0 "${CACHE_SHA}" "${SMOKE_AITER_SHA}"
validate_route_arm replay_mxfp4_policy_a 1
validate_replay_identity
[[ "$(sha_file "${CACHE_FILE}")" == "${CACHE_SHA}" ]] || fail "cache changed during replay"
[[ "$(directory_digest "${SMOKE_AITER_CACHE}")" == "${SMOKE_AITER_SHA}" ]] \
    || fail "AITER config cache changed during replay"
printf 'replay_mxfp4_policy_a=0\n' >> "${STATUS_FILE}"
printf '%s stage=replay_mxfp4_policy_a_complete\n' "$(date --iso-8601=ns)" >> "${PROGRESS}"

mkdir -p "${BF16_CACHE_DIR}" "${MXFP4_CACHE_DIR}" \
    "${BF16_AITER_CACHE}" "${MXFP4_AITER_CACHE}"
cp --reflink=auto -- "${CACHE_FILE}" "${BF16_CACHE_DIR}/mxfp4-autotune.json"
cp --reflink=auto -- "${CACHE_FILE}" "${MXFP4_CACHE_DIR}/mxfp4-autotune.json"
[[ "$(sha_file "${BF16_CACHE_DIR}/mxfp4-autotune.json")" == "${CACHE_SHA}" ]]
[[ "$(sha_file "${MXFP4_CACHE_DIR}/mxfp4-autotune.json")" == "${CACHE_SHA}" ]]
[[ "$(directory_digest "${BF16_AITER_CACHE}")" == empty ]]
[[ "$(directory_digest "${MXFP4_AITER_CACHE}")" == empty ]]

{
    date --iso-8601=ns
    echo schema=1
    echo branch=dev/mxfp4
    echo arms=smoke_mxfp4_policy_a,replay_mxfp4_policy_a,profile_bf16,profile_mxfp4_policy_a
    echo formal_arms=profile_bf16,profile_mxfp4_policy_a
    echo formal_steps=8
    echo profile_steps=7,8
    echo profile_rank=0
    echo profile_device=0
    echo nproc=8
    echo micro_batch=2
    echo global_batch=128
    echo gradient_accumulation=8
    echo sequence_length=8192
    echo tail_bf16=1
    echo expected_quantized_linears=245
    echo expected_bf16_skipped_linears=8
    echo expected_packed_qkv_layers=35
    echo expected_split_swiglu_layers=35
    echo lm_head_precision=bf16
    echo eval_batches=16
    echo val_samples=256
    echo formal_train_samples="${TRAIN_SAMPLES_FORMAL}"
    echo seed=1234
    echo cache_sha256="${CACHE_SHA}"
    echo smoke_aiter_config_cache_sha256="${SMOKE_AITER_SHA}"
    echo source_audit_sha256="$(sha_file "${SOURCE_AUDIT}")"
    echo runner_sha256="$(sha_file "${BASH_SOURCE[0]}")"
    echo route_entry_sha256="$(sha_file "${ROUTE_ENTRY}")"
    echo analyzer_sha256="$(sha_file "${ANALYZER}")"
    echo analyzer_test_sha256="$(sha_file "${ANALYZER_TEST}")"
    echo protocol_sha256="$(sha_file "${PROTOCOL}")"
    echo test_sha256="$(sha_file "${ROOT}/test_runner.py")"
} > "${META}"

current_stage=profile_bf16
run_case profile_bf16 bf16 "${FORMAL_STEPS}" "${BF16_CACHE_DIR}" \
    "${BF16_AITER_CACHE}" 0 0 1 "${CACHE_SHA}" empty
for required in trace.json profile.txt profile_shapes.txt; do
    [[ -s "${ROOT}/profile_bf16/${required}" ]] || fail "profile_bf16 missing ${required}"
done
printf 'profile_bf16=0\n' >> "${STATUS_FILE}"
printf '%s stage=profile_bf16_complete\n' "$(date --iso-8601=ns)" >> "${PROGRESS}"

current_stage=profile_mxfp4_policy_a
run_case profile_mxfp4_policy_a mxfp4 "${FORMAL_STEPS}" "${MXFP4_CACHE_DIR}" \
    "${MXFP4_AITER_CACHE}" 0 0 1 "${CACHE_SHA}" empty
for required in trace.json profile.txt profile_shapes.txt; do
    [[ -s "${ROOT}/profile_mxfp4_policy_a/${required}" ]] || fail "profile_mxfp4_policy_a missing ${required}"
done
normalize_and_compare_formal_commands
printf 'profile_mxfp4_policy_a=0\n' >> "${STATUS_FILE}"
printf '%s stage=profile_mxfp4_policy_a_complete\n' "$(date --iso-8601=ns)" >> "${PROGRESS}"

current_stage=postflight
verify_current_state 1 || fail "frozen source or workload changed before campaign postflight"
[[ "$(sha_file "${CACHE_FILE}")" == "${CACHE_SHA}" ]] || fail "build cache changed at postflight"
[[ "$(sha_file "${BF16_CACHE_DIR}/mxfp4-autotune.json")" == "${CACHE_SHA}" ]] || fail "BF16 cache copy changed"
[[ "$(sha_file "${MXFP4_CACHE_DIR}/mxfp4-autotune.json")" == "${CACHE_SHA}" ]] || fail "MXFP4 cache copy changed"
[[ "$(directory_digest "${SMOKE_AITER_CACHE}")" == "${SMOKE_AITER_SHA}" ]] || fail "smoke AITER cache changed"
[[ "$(directory_digest "${BF16_AITER_CACHE}")" == empty ]] || fail "BF16 AITER config cache changed"
[[ "$(directory_digest "${MXFP4_AITER_CACHE}")" == empty ]] || fail "MXFP4 AITER config cache changed"
record_kfd_idle "${ROOT}/kfd-campaign-after.txt" || fail "KFD busy after campaign"
printf 'postflight=0\n' >> "${STATUS_FILE}"

current_stage=analysis
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}"
[[ -s "${ROOT}/profile_analysis.json" && -s "${ROOT}/profile_analysis.md" ]] \
    || fail "analyzer did not produce both reports"
printf 'analysis=0\n' >> "${STATUS_FILE}"

current_stage=manifest
printf '%s stage=manifest_ready\n' "$(date --iso-8601=ns)" >> "${PROGRESS}"
manifest_temporary=${MANIFEST}.tmp.$$
find "${ROOT}" -type f \
    ! -path '*/__pycache__/*' \
    ! -name '*.pyc' \
    ! -name '*.pyo' \
    ! -path "${MANIFEST}" \
    ! -path "${manifest_temporary}" \
    ! -path "${COMPLETE}" \
    ! -path "${EXIT_FILE}" \
    -print0 | sort -z | xargs -0 sha256sum > "${manifest_temporary}"
mv -- "${manifest_temporary}" "${MANIFEST}"
write_atomic "${COMPLETE}" \
    schema=1 \
    completed=smoke_mxfp4_policy_a,replay_mxfp4_policy_a,profile_bf16,profile_mxfp4_policy_a \
    "cache_sha256=${CACHE_SHA}" \
    "source_audit_sha256=$(sha_file "${SOURCE_AUDIT}")" \
    "profile_meta_sha256=$(sha_file "${META}")" \
    "campaign_stage_status_sha256=$(sha_file "${STATUS_FILE}")" \
    "analysis_json_sha256=$(sha_file "${ROOT}/profile_analysis.json")" \
    "analysis_markdown_sha256=$(sha_file "${ROOT}/profile_analysis.md")" \
    "artifact_manifest_sha256=$(sha_file "${MANIFEST}")"
write_atomic "${EXIT_FILE}" 0
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "${ANALYZER}" \
    --root "${ROOT}" --verify-completion
current_stage=complete

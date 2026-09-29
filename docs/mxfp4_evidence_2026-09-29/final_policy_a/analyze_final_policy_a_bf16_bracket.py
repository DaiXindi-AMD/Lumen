#!/usr/bin/env python3
"""Analyze the fresh BF16 / MXFP4 Policy A / BF16 bracket campaign."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
import shlex
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "analyze_tail210_base.py"
RUNNER = ROOT / "run_case.sh"
ENTRY = ROOT / "inventory_entry.py"
EXCLUDES = ROOT / "provenance-git-excludes"
DRIVER = ROOT / "run_final_policy_a_bf16_bracket.sh"
SELF_TEST = ROOT / "test_final_policy_a_bf16_bracket.py"
PROTOCOL = ROOT / "protocol.md"
TRAIN_ENTRY = Path("/home/xdai/Lumen/examples/qwen3/train_qwen3_fsdp.py")
META = ROOT / "campaign-meta.txt"
PHASE1_COMPLETE = ROOT / "phase1-complete.txt"
PHASE1_VALIDATION = ROOT / "phase1-validation.json"
PHASE1_MANIFEST = ROOT / "phase1-artifacts.sha256"
PHASE1_STATUS = ROOT / "final-policy-a-bf16-bracket-phase1-exit-status.txt"
PHASE2_STATUS = ROOT / "final-policy-a-bf16-bracket-phase2-exit-status.txt"
AITER_CONFIG_CACHE = ROOT / "cache/aiter-configs"

SMOKE = "smoke_tail1"
ARMS = ("bf16_a1", "mxfp4_policy_a", "bf16_a2")
ORDER = (SMOKE, *ARMS)
PHASE1_ORDER = (SMOKE, ARMS[0], ARMS[1])
PRECISIONS = {
    SMOKE: "mxfp4",
    "bf16_a1": "bf16",
    "mxfp4_policy_a": "mxfp4",
    "bf16_a2": "bf16",
}

SMOKE_STEPS = 3
FORMAL_STEPS = 50
TRAIN_SAMPLES = 6400
WINDOW = tuple(range(11, 51))
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_RESAMPLES = 100_000
BLOCK_LENGTH = 4
MIN_SPEEDUP = 1.6
MIN_WINS = 28
CI_LOWER_MINIMUM = 1.6
DRIFT_LIMIT_PCT = 3.0
NLL_LIMIT = 0.01
CAMPAIGN_NAMESPACE = "final_policy_a_bf16_bracket"

LUMEN_COMMIT = "6b9aee1569247eca20937c14319ba6adcd69e0cb"
AITER_COMMIT = "e35bb17f4f815903bf73598facedbb321e15af28"
EXPECTED_RUNNER_SHA256 = (
    "5ec1ca5b1a24585ce0634b44e1c4569b05dedb25274058951e2e4eb06c18aa9a"
)
EXPECTED_ENTRY_SHA256 = (
    "0a25a6579ff2838ad64935115ce407f23a563b5a67f37d7000a7de4d08145fdc"
)
EXPECTED_BASE_SHA256 = (
    "ab6973743dabd1042a431304662e5d2805cff7d98aabd671df2c418b967009d6"
)
EXPECTED_EXCLUDES_SHA256 = (
    "669f8c1235e39e040bc27437a69a98c578f86ee2659c546c594a8b78a924ed8e"
)
EXPECTED_TRAIN_ENTRY_SHA256 = (
    "ffd60a612d522c2e125fb2a622da8bd9fcd0c4c5bb478ad6a26b544c9e38a74f"
)

EXPECTED_CACHE_CHOICES = {
    "4096,4096,16384": "asm",
    "4096,12288,16384": "asm",
    "6144,4096,16384": "asm",
    "12288,4096,16384": "asm",
    "16384,4096,4096": "asm",
    "16384,4096,6144": "asm",
    "16384,4096,12288": "asm",
    "16384,6144,4096": "asm",
    "16384,12288,4096": "asm",
}
EXPECTED_CACHE_BACKENDS = "dd7fa0173867fffa"
EXPECTED_TUNED_TABLES = "f970d344c7dec601"
EXPECTED_ASM_REASON = (
    "multi-device consensus is required; retaining protected ASM without "
    "per-rank online profiling"
)
EXPECTED_ASM_IDENTITIES = {
    "12288,4096,16384": {
        "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E",
        "log2_k_split": 0,
        "code_object": "f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co",
        "code_object_sha256": "c32d5356e5d92b77",
        "manifest_sha256": "4e0dce9b9642d4fb",
        "tile_m": 128,
        "tile_n": 512,
        "split_k_capable": True,
    },
    "16384,12288,4096": {
        "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E",
        "log2_k_split": 0,
        "code_object": "f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co",
        "code_object_sha256": "c32d5356e5d92b77",
        "manifest_sha256": "4e0dce9b9642d4fb",
        "tile_m": 128,
        "tile_n": 512,
        "split_k_capable": True,
    },
    "6144,4096,16384": {
        "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E",
        "log2_k_split": 0,
        "code_object": "f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256.co",
        "code_object_sha256": "01453b424c0b54ca",
        "manifest_sha256": "4e0dce9b9642d4fb",
        "tile_m": 192,
        "tile_n": 256,
        "split_k_capable": False,
    },
}
for _shape in EXPECTED_CACHE_CHOICES:
    EXPECTED_ASM_IDENTITIES.setdefault(
        _shape,
        {
            "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
            "log2_k_split": 0,
            "code_object": "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co",
            "code_object_sha256": "ff0d6bb10a211068",
            "manifest_sha256": "4e0dce9b9642d4fb",
            "tile_m": 256,
            "tile_n": 256,
            "split_k_capable": True,
        },
    )

LAST = "model.layers.35"
EXPECTED_UNQUANTIZED = [
    f"{LAST}.self_attn.q_proj",
    f"{LAST}.self_attn.k_proj",
    f"{LAST}.self_attn.v_proj",
    f"{LAST}.self_attn.o_proj",
    f"{LAST}.mlp.gate_proj",
    f"{LAST}.mlp.up_proj",
    f"{LAST}.mlp.down_proj",
    "lm_head",
]
EXPECTED_SMOKE_SHAPES = {
    (4096, 4096, 16384): 840,
    (4096, 12288, 16384): 840,
    (6144, 4096, 16384): 840,
    (12288, 4096, 16384): 1680,
    (16384, 4096, 4096): 2240,
    (16384, 4096, 6144): 840,
    (16384, 4096, 12288): 3080,
    (16384, 6144, 4096): 1400,
    (16384, 12288, 4096): 3640,
}
SOURCE_LINE_RE = re.compile(r"^([0-9a-f]{64})  (/.+)$")
PAIRED_PROVENANCE_KEYS = tuple(
    key for key in (
        "global_batch",
        "gradient_accumulation",
        "tokens_per_update",
        "cache_namespace",
        "cache_file",
        "aiter_config_cache_dir",
        "gpu_lock_path",
        "gpu_lock_verified",
        "profile_start",
        "profile_end",
        "profile_shapes",
        "copy_trace",
        "eval_batches",
        "val_samples",
        "numa_balancing",
        "registry_freeze_removed",
        "weight_cache_fast_hit_removed",
        "mxfp4_activation_descriptor_cache",
        "tuned_config_override",
        "source_bundle_sha256",
        "lumen_commit",
        "lumen_diff_sha256",
        "lumen_status_sha256",
        "aiter_commit",
        "aiter_diff_sha256",
        "aiter_status_sha256",
        "lumen_tree_sha256",
        "aiter_tree_sha256",
        "aiter_config_cache_sha256",
        "runtime_modules_sha256",
        "f4gemm_directory_sha256",
        "lumen_import",
        "aiter_import",
        "torch",
    )
)


def _load_base() -> ModuleType:
    spec = importlib.util.spec_from_file_location("final_policy_a_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base analyzer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.ROOT = ROOT
base.SMOKE = SMOKE
base.ARMS = ARMS
base.TAILS = {name: 1 for name in ARMS}
base.EXPECTED_ORDER = ORDER
base.PHASE1_ORDER = PHASE1_ORDER
base.ENTRY_PATH = ENTRY
base.TRAIN_ENTRY_PATH = TRAIN_ENTRY
base.WINDOW = WINDOW
base.TRAIN_SAMPLES = TRAIN_SAMPLES
base.LUMEN_COMMIT = LUMEN_COMMIT
base.RUNNER_PATH = RUNNER


def strict_json_load(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(base.read_text(path), object_pairs_hook=no_duplicates)


def option_absent(tokens: list[str], flag: str) -> bool:
    return flag not in tokens and not any(
        token.startswith(flag + "=") for token in tokens
    )


def torchrun_entrypoint_is(tokens: list[str], expected_entry: Path) -> bool:
    prefix = [
        "/usr/local/bin/torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(expected_entry),
    ]
    return tokens[: len(prefix)] == prefix


def command_checks(
    command: str,
    *,
    precision: str,
    total_steps: int,
    meta: dict[str, str],
    smoke: bool,
) -> dict[str, bool]:
    tokens = shlex.split(command)
    expected_entry = ENTRY if smoke else TRAIN_ENTRY
    expected = {
        "--nproc-per-node": "8",
        "--seq-length": "8192",
        "--micro-batch-size": "2",
        "--gradient-accumulation-steps": "8",
        "--max-steps": str(total_steps),
        "--train-samples": str(TRAIN_SAMPLES),
        "--eval-interval": str(total_steps),
        "--eval-batches": "16",
        "--val-samples": "256",
        "--seed": "1234",
        "--mode": precision,
        "--fsdp-version": "2",
        "--sharding": "full_shard",
        "--fsdp-reduce-dtype": "bf16",
        "--num-layers-at-start-in-bf16": "0",
        "--num-layers-at-end-in-bf16": "1",
    }
    parsed: dict[str, str | None] = {}
    for flag in expected:
        try:
            parsed[flag] = base.option_value(tokens, flag)
        except ValueError:
            parsed[flag] = None
    common_flags = (
        "--init-from-scratch",
        "--first-last-layers-bf16",
        "--fsdp-retain-accumulated-params",
        "--no-grad-checkpointing",
    )
    mxfp4_flags = ("--mxfp4-pack-qkv", "--mxfp4-fuse-swiglu")
    try:
        global_batch = (
            int(parsed["--nproc-per-node"] or "")
            * int(parsed["--micro-batch-size"] or "")
            * int(parsed["--gradient-accumulation-steps"] or "")
        )
    except ValueError:
        global_batch = -1
    return {
        "entrypoint": torchrun_entrypoint_is(tokens, expected_entry),
        "fixed_values": all(parsed[key] == value for key, value in expected.items()),
        "common_flags": all(base.flag_once(tokens, flag) for flag in common_flags),
        "precision_flags": (
            all(base.flag_once(tokens, flag) for flag in mxfp4_flags)
            if precision == "mxfp4"
            else all(option_absent(tokens, flag) for flag in mxfp4_flags)
        ),
        "no_projection_override": option_absent(
            tokens, "--mxfp4-last-layer-bf16-projections"
        ),
        "no_mxfp4_communication": option_absent(tokens, "--fsdp-mxfp4-comm")
        and option_absent(tokens, "--mxfp4-comm"),
        "no_positive_grad_checkpointing": option_absent(
            tokens, "--grad-checkpointing"
        ),
        "global_batch": global_batch == 128
        and meta.get("global_batch") == "128"
        and meta.get("gradient_accumulation") == "8",
    }


def normalized_formal_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    mode = base.option_value(tokens, "--mode")
    for index, token in enumerate(tokens):
        if token == "--mode":
            tokens[index + 1] = "<PRECISION>"
            break
        if token == f"--mode={mode}":
            tokens[index] = "--mode=<PRECISION>"
            break
    return [
        token
        for token in tokens
        if token not in ("--mxfp4-pack-qkv", "--mxfp4-fuse-swiglu")
    ]


def fallback_warning_lines(text: str) -> list[str]:
    phrases = (
        "fallback",
        "falling back",
        "fall back",
        "fell back",
        "emergency bf16",
        "gradients are bf16",
        "dequant_bf16",
        "all backends exhausted",
        "trying next backend",
        "cached backend failure",
        "standard blockscale",
        "quant failure",
        "requant per forward",
        "using original path",
        "using pytorch path",
    )
    matches: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        broad = ("warning" in lowered or "error" in lowered) and (
            "bf16" in lowered or "fallback" in lowered or "fall back" in lowered
        )
        if broad or any(phrase in lowered for phrase in phrases):
            matches.append(line)
    return matches


def strict_post_success_only(
    text: str, marker: str, *, smoke_shape_line: str | None = None
) -> bool:
    if text.count(marker) != 1:
        return False
    before, after = text.split(marker, 1)
    if "Traceback (most recent call last):" in before:
        return False
    lines = [line for line in after.splitlines() if line.strip()]
    if smoke_shape_line is not None:
        if lines[:2] != [smoke_shape_line, smoke_shape_line]:
            return False
        lines = lines[2:]
        if smoke_shape_line in lines:
            return False
    elif any("MXFP4 shape log:" in line for line in lines):
        return False
    fragments = (
        'File "/usr/lib/python3.10/weakref.py", line 667, in _exitfunc',
        "f()",
        'File "/usr/lib/python3.10/weakref.py", line 591, in __call__',
        "return info.func(*info.args, **(info.kwargs or {}))",
        'File "/usr/local/lib/python3.10/dist-packages/torch/library.py", line 667, in _del_library',
        "_clear_torch_ops_cache(op_defs)",
        'File "/usr/local/lib/python3.10/dist-packages/torch/library.py", line 623, in _clear_torch_ops_cache',
        'ns, name_with_overload = qualname.split("::")',
        "ValueError: too many values to unpack (expected 2)",
    )
    ranks: list[int] = []
    index = 0
    while index < len(lines):
        header = re.fullmatch(
            r"\[rank(\d+)\]: Traceback \(most recent call last\):", lines[index]
        )
        if header is None or index + 9 >= len(lines):
            return False
        rank = int(header.group(1))
        prefix = f"[rank{rank}]:"
        block = lines[index + 1 : index + 10]
        if any(not line.startswith(prefix) for line in block):
            return False
        stripped = [line[len(prefix) :].strip() for line in block]
        if any(
            fragment != actual
            for fragment, actual in zip(fragments, stripped, strict=True)
        ):
            return False
        ranks.append(rank)
        index += 10
    return not ranks or (
        len(ranks) == 16
        and collections.Counter(ranks) == {rank: 2 for rank in range(8)}
    )


def listed_source_files(meta_text: str) -> list[tuple[str, Path]]:
    listing = [
        (match.group(1), Path(match.group(2)))
        for line in meta_text.splitlines()
        if (match := SOURCE_LINE_RE.fullmatch(line)) is not None
    ]
    if not listing:
        raise ValueError("run metadata has no source-file hashes")
    return listing


def source_listing_check(case: dict) -> bool:
    listing = case["source_listing"]
    if len({str(path) for _digest, path in listing}) != len(listing):
        return False
    if any(
        path.is_symlink() or not path.is_file() or base.sha256(path) != digest
        for digest, path in listing
    ):
        return False
    encoded = "".join(f"{digest}  {path}\n" for digest, path in listing).encode()
    return hashlib.sha256(encoded).hexdigest() == case["source_before"]


def source_manifest(case: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for digest, path in case["source_listing"]:
        key = str(path)
        if key in result:
            raise ValueError(f"duplicate source path: {key}")
        result[key] = digest
    return result


def source_manifest_digest(case: dict) -> str:
    encoded = "".join(
        f"{digest}  {path}\n" for digest, path in case["source_listing"]
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_mxfp4_comm_disabled(text: str) -> bool:
    return text.count("mxfp4_comm=False") == 1 and "mxfp4_comm=True" not in text


def kfd_terminal_idle(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    attempts = [
        index for index, line in enumerate(lines) if re.fullmatch(r"attempt=\d+", line)
    ]
    terminal = lines[attempts[-1] :] if attempts else lines
    statuses = [line for line in terminal if line.startswith("status=")]
    return statuses == ["status=idle"] and not any(
        line.startswith("non_service_kfd_client")
        or line.startswith("unreadable_kfd_client")
        or line.startswith("known_gpu_workloads")
        or line.startswith("status=error")
        for line in terminal
    )


def parse_case(name: str) -> dict:
    total_steps = SMOKE_STEPS if name == SMOKE else FORMAL_STEPS
    case = base.parse_case(name, tail=1, total_steps=total_steps)
    precision = PRECISIONS[name]
    marker = f"Training complete after {total_steps} steps."
    shape_line = (
        "INFO:lumen.ops.quantize.mxfp4_autotune:MXFP4 shape log: wrote 9 "
        f"distinct shapes to {ROOT / SMOKE / 'mxfp4-shapes-rank0.csv'}"
        if name == SMOKE
        else None
    )
    case.update(
        {
            "precision": precision,
            "source_listing": listed_source_files(
                base.read_text(case["directory"] / "run-meta.txt")
            ),
            "fallback_warning_lines": fallback_warning_lines(case["log"]),
            "strict_post_success": strict_post_success_only(
                case["log"], marker, smoke_shape_line=shape_line
            ),
            "runtime_mxfp4_comm_disabled": runtime_mxfp4_comm_disabled(
                case["log"]
            ),
        }
    )
    return case


def cache_integrity(path: Path) -> tuple[bool, dict]:
    payload = strict_json_load(path)
    profiles = payload.get("profiles", {})
    profiles_ok = set(profiles) == set(EXPECTED_CACHE_CHOICES)
    if profiles_ok:
        for shape, expected_identity in EXPECTED_ASM_IDENTITIES.items():
            profile = profiles[shape]
            asm_identity = profile.get("identities", {}).get("asm", {})
            profiles_ok = profiles_ok and (
                profile.get("winner") == "asm"
                and profile.get("incumbent") == "asm"
                and profile.get("decision_scope") == "single_device"
                and profile.get("profile_device_count") == 1
                and profile.get("selection_policy")
                == "protected_asm_consensus_required"
                and profile.get("reason") == EXPECTED_ASM_REASON
                and profile.get("switch_margin") == 1.05
                and asm_identity.get("implementation") == "asm"
                and all(
                    asm_identity.get(key) == value
                    for key, value in expected_identity.items()
                )
            )
    valid = (
        payload.get("schema") == 6
        and payload.get("arch") == "gfx950"
        and payload.get("backends") == EXPECTED_CACHE_BACKENDS
        and payload.get("tuned_tables") == EXPECTED_TUNED_TABLES
        and payload.get("decision_scope") == "single_device"
        and payload.get("profile_device_count") == 1
        and payload.get("profile_settings")
        == {"timed_iters": 11, "warmup_iters": 3}
        and payload.get("choices") == EXPECTED_CACHE_CHOICES
        and profiles_ok
    )
    return valid, payload


def smoke_route_integrity(directory: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for rank in range(8):
        report_path = directory / f"route-rank{rank}.json"
        shape_path = directory / f"mxfp4-shapes-rank{rank}.csv"
        if not report_path.is_file() or not shape_path.is_file():
            errors.append(f"rank {rank}: missing route/shape artifact")
            continue
        try:
            report = strict_json_load(report_path)
            expected_counts = {
                "rank": rank,
                "world_size": 8,
                "inventory_captured": 1,
                "enabled_qkv": 35,
                "enabled_swiglu": 35,
            }
            if report.get("counts") != expected_counts:
                errors.append(f"rank {rank}: route counts mismatch")
            if report.get("rank") != rank:
                errors.append(f"rank {rank}: top-level rank mismatch")
            if report.get("unquantized_linear_names") != EXPECTED_UNQUANTIZED:
                errors.append(f"rank {rank}: unquantized linear inventory mismatch")
            if report.get("lm_head_count") != 1:
                errors.append(f"rank {rank}: lm_head count mismatch")
            if report.get("lm_head_weight_dtype") != "bfloat16":
                errors.append(f"rank {rank}: lm_head is not BF16")
            if report.get("lm_head_quant_enabled") is not False:
                errors.append(f"rank {rank}: lm_head unexpectedly quantized")
            if report.get("lumen_import") != "/home/xdai/Lumen/lumen/__init__.py":
                errors.append(f"rank {rank}: unexpected Lumen import")
            if report.get("aiter_import") != "/home/xdai/aiter/aiter/__init__.py":
                errors.append(f"rank {rank}: unexpected AITER import")
            with shape_path.open(newline="", encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
            rows = {
                (int(row["M"]), int(row["N"]), int(row["K"])): row
                for row in raw_rows
            }
            if len(raw_rows) != 9 or set(rows) != set(EXPECTED_SMOKE_SHAPES):
                errors.append(f"rank {rank}: unexpected or duplicate shape rows")
                continue
            for shape, calls in EXPECTED_SMOKE_SHAPES.items():
                row = rows[shape]
                if (
                    row.get("backend") != "asm"
                    or row.get("asm_available") != "1"
                    or int(row["calls"]) != calls
                ):
                    errors.append(f"rank {rank}: shape mismatch {shape}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"rank {rank}: malformed route/shape artifact: {error}")
    return not errors, errors


def route_check(case: dict) -> bool:
    if case["precision"] == "bf16":
        return (
            case["quantized_reports"] == 0
            and case["skip_reports"] == 0
            and case["qkv_reports"] == 0
            and case["swiglu_reports"] == 0
            and case["qkv_tail_warnings"] == 0
            and case["swiglu_tail_warnings"] == 0
        )
    route = base.expected_route(1)
    return (
        case["quantized_reports"] == 1
        and case["skip_reports"] == 1
        and case["qkv_reports"] == 1
        and case["swiglu_reports"] == 1
        and case["qkv_tail_warnings"] == route["warnings"]
        and case["swiglu_tail_warnings"] == route["warnings"]
    )


def case_checks(
    case: dict,
    *,
    expected_source: str,
    expected_cache: str,
    expected_model: list[str],
    expected_train: dict[int, str],
    expected_validation: dict[int, str],
    expected_provenance: dict[str, str],
) -> dict[str, bool]:
    name = case["name"]
    smoke = name == SMOKE
    command = command_checks(
        case["meta"]["command"],
        precision=case["precision"],
        total_steps=case["total_steps"],
        meta=case["meta"],
        smoke=smoke,
    )
    metadata_pairing = (
        all(
            case["meta"].get(key) == expected_provenance.get(key)
            and expected_provenance.get(key) not in (None, "")
            for key in PAIRED_PROVENANCE_KEYS
        )
        and case["meta"].get("precision") == case["precision"]
        and case["meta"].get("cache_namespace") == CAMPAIGN_NAMESPACE
        and case["meta"].get("gpu_lock_verified") == "1"
        and case["meta"].get("tail_bf16") == "1"
        and case["meta"].get("eval_batches") == "16"
        and case["meta"].get("val_samples") == "256"
        and case["meta"].get("numa_balancing") == "0"
        and case["meta"].get("registry_freeze_removed") == "1"
        and case["meta"].get("weight_cache_fast_hit_removed") == "1"
        and case["meta"].get("mxfp4_activation_descriptor_cache") == "0"
        and case["meta"].get("lumen_commit") == LUMEN_COMMIT
        and case["meta"].get("aiter_commit") == AITER_COMMIT
        and case["meta"].get("lumen_import")
        == "/home/xdai/Lumen/lumen/__init__.py"
        and case["meta"].get("aiter_import")
        == "/home/xdai/aiter/aiter/__init__.py"
        and all(
            case["tree_after"].get(key) == case["meta"].get(key)
            and case["meta"].get(key) not in (None, "")
            for key in base.TREE_KEYS
        )
    )
    return {
        f"{name}_complete_steps": case["complete"]
        and [row["step"] for row in case["steps"]]
        == list(range(1, case["total_steps"] + 1))
        and all(row["total"] == case["total_steps"] for row in case["steps"])
        and all(row["time_ms"] > 0 for row in case["steps"])
        and (smoke or len(case["window"]) == len(WINDOW)),
        f"{name}_finite": all(
            math.isfinite(row[field])
            for row in case["steps"]
            for field in ("loss", "grad_norm", "lr", "time_ms", "memory_gib")
        )
        and math.isfinite(case["validation_nll"]),
        f"{name}_no_failure_or_fallback": not case["failure_lines"]
        and not case["unexpected_disabled"]
        and not case["fallback_warning_lines"],
        f"{name}_strict_known_post_success_only": case["strict_post_success"],
        f"{name}_statuses": base.status_ok(case),
        f"{name}_source_frozen": case["source_before"]
        == expected_source
        == case["source_after"],
        f"{name}_source_file_hashes": source_listing_check(case),
        f"{name}_cache_final": case["cache_after"] == expected_cache,
        f"{name}_model_pairing": case["model_hashes"] == expected_model
        and len(expected_model) == 1,
        f"{name}_train_pairing": case["train_digests"] == expected_train,
        f"{name}_validation_pairing": case["validation_digests"]
        == expected_validation,
        f"{name}_metadata_pairing": metadata_pairing,
        f"{name}_runtime_mxfp4_comm_disabled": case[
            "runtime_mxfp4_comm_disabled"
        ],
        f"{name}_route": route_check(case),
        f"{name}_kfd_idle": kfd_terminal_idle(case["kfd"][0])
        and kfd_terminal_idle(case["kfd"][1])
        and (
            kfd_terminal_idle(case["kfd"][2])
            or kfd_terminal_idle(case["kfd_recheck"])
        ),
        f"{name}_command": all(command.values()),
    }


def metadata_checks(meta: dict[str, str]) -> dict[str, bool]:
    expected = {
        "schema": "1",
        "branch": "dev/mxfp4",
        "lumen_commit": LUMEN_COMMIT,
        "aiter_commit": AITER_COMMIT,
        "formal_order": ",".join(ARMS),
        "phase1_order": ",".join(PHASE1_ORDER),
        "formal_steps": "50",
        "train_samples": "6400",
        "timing_window": "11-50",
        "speed_gate_min_ratio": "1.6",
        "speed_gate_min_wins": "28",
        "bootstrap_ci_lower_minimum": "1.6",
        "bootstrap_block_length": "4",
        "bootstrap_resamples": "100000",
        "bf16_replicate_drift_gate_pct": "3.0",
        "precision_gate_delta_nll": "0.01",
        "candidate_stack": "policy_a_tail1+packed_qkv+split_swiglu",
        "lm_head_precision": "bf16",
        "lm_head_evidence": "fresh_policy_a_smoke_plus_frozen_source",
        "mxfp4_communication": "disabled",
        "git_exclude_scope": ".codex/",
        "source_integrity": "exact_manifest+full_tree_without_codex_runtime_logs",
        "shape_count_contract": "policy_a_whole_smoke_totals",
        "train_pairing_scope": "all_arms_exact_first_update",
        "kfd_identity_policy": "fail_closed_complete_identity",
        "gpuagent_sha256": "5bc2a7d45f2fd992fcf3f53e12b5437a9b3fc7878c653a37ac647c380fb0e913",
        "numa_balancing": "0",
    }
    return {
        "metadata_protocol": all(meta.get(key) == value for key, value in expected.items()),
        "metadata_workload_hash": re.fullmatch(
            r"[0-9a-f]{64}", meta.get("workload_sha256", "")
        )
        is not None,
        "metadata_train_samples_exact": TRAIN_SAMPLES == FORMAL_STEPS * 128,
    }


def harness_checks(meta: dict[str, str]) -> dict[str, bool]:
    paths = {
        "driver_sha256": DRIVER,
        "analyzer_sha256": Path(__file__).resolve(),
        "self_test_sha256": SELF_TEST,
        "protocol_sha256": PROTOCOL,
        "runner_sha256": RUNNER,
        "entry_sha256": ENTRY,
        "base_analyzer_sha256": BASE_PATH,
        "git_excludes_sha256": EXCLUDES,
        "train_entry_sha256": TRAIN_ENTRY,
    }
    checks = {
        key.removesuffix("_sha256") + "_frozen": path.is_file()
        and meta.get(key) == base.sha256(path)
        for key, path in paths.items()
    }
    checks.update(
        {
            "runner_expected": meta.get("runner_sha256")
            == EXPECTED_RUNNER_SHA256,
            "entry_expected": meta.get("entry_sha256") == EXPECTED_ENTRY_SHA256,
            "base_analyzer_expected": meta.get("base_analyzer_sha256")
            == EXPECTED_BASE_SHA256,
            "git_excludes_expected": meta.get("git_excludes_sha256")
            == EXPECTED_EXCLUDES_SHA256,
            "train_entry_expected": meta.get("train_entry_sha256")
            == EXPECTED_TRAIN_ENTRY_SHA256,
            "git_excludes_exact": EXCLUDES.read_text(encoding="utf-8")
            == ".codex/\n",
        }
    )
    return checks


def source_checks(cases: dict[str, dict], meta: dict[str, str]) -> dict[str, bool]:
    manifests = {name: source_manifest(case) for name, case in cases.items()}
    reference = manifests[SMOKE]
    critical = {
        str(RUNNER),
        str(ENTRY),
        str(BASE_PATH),
        str(EXCLUDES),
        str(DRIVER),
        str(Path(__file__).resolve()),
        str(SELF_TEST),
        str(PROTOCOL),
        str(TRAIN_ENTRY),
        "/home/xdai/Lumen/lumen/models/fsdp.py",
        "/home/xdai/Lumen/lumen/models/qwen3.py",
        "/home/xdai/Lumen/lumen/ops/fused_swiglu.py",
        "/home/xdai/Lumen/lumen/ops/quantize/linear.py",
        "/home/xdai/Lumen/lumen/ops/quantize/mxfp4_autotune.py",
        "/home/xdai/Lumen/lumen/ops/quantize/mxfp4_asm.py",
        "/home/xdai/aiter/aiter/ops/gemm_op_a4w4.py",
        "/home/xdai/aiter/aiter/ops/triton/activation.py",
        "/home/xdai/aiter/aiter/ops/triton/_triton_kernels/activation.py",
    }
    return {
        "source_manifests_identical": all(
            manifest == reference for manifest in manifests.values()
        ),
        "critical_source_inputs_present": critical <= set(reference),
        "source_manifest_excludes_codex_runtime_logs": all(
            "/.codex/" not in path for path in reference
        ),
        "source_manifest_paths_current": all(
            Path(path).is_file() and base.sha256(Path(path)) == digest
            for path, digest in reference.items()
        ),
        "source_manifest_digest_locked": meta.get("source_manifest_sha256")
        == source_manifest_digest(cases[SMOKE]),
    }


def phase1_artifact_paths() -> set[Path]:
    paths = {
        META,
        ROOT / "cache" / CAMPAIGN_NAMESPACE / "mxfp4-autotune.json",
        ROOT / "kfd-final-policy-a-phase1-before.txt",
        ROOT / "kfd-final-policy-a-phase1-after.txt",
        PHASE1_VALIDATION,
        PHASE1_STATUS,
    }
    for name in PHASE1_ORDER:
        directory = ROOT / name
        if directory.is_dir():
            paths.update(path for path in directory.iterdir() if path.is_file())
    return paths


def checksum_manifest_ok(path: Path, expected_paths: set[Path]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    entries: list[tuple[str, Path]] = []
    for line in base.read_text(path).splitlines():
        match = SOURCE_LINE_RE.fullmatch(line)
        if match is None:
            return False
        entries.append((match.group(1), Path(match.group(2))))
    listed = [item for _digest, item in entries]
    return (
        bool(entries)
        and len(set(listed)) == len(listed)
        and set(listed) == expected_paths
        and all(
            item.is_file()
            and not item.is_symlink()
            and base.sha256(item) == digest
            for digest, item in entries
        )
    )


def real_empty_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and next(path.iterdir(), None) is None


def phase1_boundary_checks(
    *,
    phase1_complete: Path = PHASE1_COMPLETE,
    phase1_manifest: Path = PHASE1_MANIFEST,
    candidate_postflight: Path = ROOT / "mxfp4_policy_a/postflight-status.txt",
    aiter_config_cache: Path = AITER_CONFIG_CACHE,
    pre_a2_epoch_ns: int | None = None,
) -> dict[str, bool]:
    phase1 = base.parse_kv(phase1_complete)
    raw_completed_epoch_ns = phase1.get("completed_epoch_ns", "")
    timestamp_valid = re.fullmatch(r"[1-9][0-9]{18}", raw_completed_epoch_ns) is not None
    completed_epoch_ns = int(raw_completed_epoch_ns) if timestamp_valid else None
    if pre_a2_epoch_ns is None:
        pre_a2_epoch_ns = time.time_ns()
    candidate_postflight_mtime_ns = candidate_postflight.stat().st_mtime_ns
    manifest_mtime_ns = phase1_manifest.stat().st_mtime_ns

    return {
        "phase1_sentinel_regular_file": phase1_complete.is_file()
        and not phase1_complete.is_symlink(),
        "phase1_sentinel_schema": phase1.get("schema") == "1",
        "phase1_completed_epoch_ns_format": timestamp_valid,
        "phase1_sentinel_after_candidate_postflight": completed_epoch_ns is not None
        and completed_epoch_ns >= candidate_postflight_mtime_ns,
        "phase1_sentinel_after_manifest": completed_epoch_ns is not None
        and completed_epoch_ns >= manifest_mtime_ns,
        "phase2_preflight_after_phase1_completion": completed_epoch_ns is not None
        and completed_epoch_ns < pre_a2_epoch_ns,
        "aiter_config_cache_real_empty_directory": real_empty_directory(
            aiter_config_cache
        ),
    }


def phase1_boundary_report() -> dict:
    checks = phase1_boundary_checks()
    return {
        "mode": "validate-phase1-boundary",
        "passed": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def sentinel_checks(meta: dict[str, str], cases: dict[str, dict]) -> dict[str, bool]:
    phase1 = base.parse_kv(PHASE1_COMPLETE)
    formal = base.parse_kv(ROOT / "formal-arms-complete.txt")
    phase1_manifest_ok = checksum_manifest_ok(
        PHASE1_MANIFEST, phase1_artifact_paths()
    )
    return {
        "phase1_exit_zero": base.read_text(PHASE1_STATUS).strip() == "0",
        "phase2_exit_zero": base.read_text(PHASE2_STATUS).strip() == "0",
        "phase1_sentinel": phase1.get("schema") == "1"
        and phase1.get("completed") == ",".join(PHASE1_ORDER)
        and phase1.get("source_bundle_sha256") == meta.get("source_bundle_sha256")
        and phase1.get("cache_sha256") == meta.get("fresh_autotune_cache_sha256")
        and phase1.get("meta_sha256") == base.sha256(META)
        and phase1.get("phase1_artifacts_manifest_sha256")
        == base.sha256(PHASE1_MANIFEST)
        and phase1_manifest_ok,
        "formal_completion_sentinel": formal.get("schema") == "1"
        and formal.get("completed") == ",".join(ARMS)
        and formal.get("source_bundle_sha256") == meta.get("source_bundle_sha256")
        and formal.get("cache_sha256") == meta.get("fresh_autotune_cache_sha256"),
        "phase_boundaries_idle": all(
            kfd_terminal_idle(base.read_text(ROOT / filename))
            for filename in (
                "kfd-final-policy-a-phase1-before.txt",
                "kfd-final-policy-a-phase1-after.txt",
                "kfd-final-policy-a-phase2-before.txt",
                "kfd-final-policy-a-phase2-after.txt",
            )
        ),
        "phase1_sentinel_after_candidate_postflight": int(
            phase1["completed_epoch_ns"]
        )
        >= cases["mxfp4_policy_a"]["postflight_mtime_ns"],
        "phase2_started_after_phase1_completion": int(phase1["completed_epoch_ns"])
        < cases["bf16_a2"]["started_epoch_ns"],
        "formal_sentinel_after_bf16_a2_postflight": int(
            formal["completed_epoch_ns"]
        )
        >= cases["bf16_a2"]["postflight_mtime_ns"],
    }


def integrity_for_cases(
    names: tuple[str, ...], *, complete: bool
) -> tuple[dict[str, bool], dict[str, list[str]], dict, dict[str, dict]]:
    meta = base.parse_kv(META)
    cases = {name: parse_case(name) for name in names}
    smoke = cases[SMOKE]
    expected_source = meta["source_bundle_sha256"]
    expected_cache = meta["fresh_autotune_cache_sha256"]
    expected_provenance = {
        key: smoke["meta"].get(key, "") for key in PAIRED_PROVENANCE_KEYS
    }
    checks: dict[str, bool] = {}
    checks.update(metadata_checks(meta))
    checks.update(harness_checks(meta))
    for case in cases.values():
        checks.update(
            case_checks(
                case,
                expected_source=expected_source,
                expected_cache=expected_cache,
                expected_model=smoke["model_hashes"],
                expected_train=smoke["train_digests"],
                expected_validation=smoke["validation_digests"],
                expected_provenance=expected_provenance,
            )
        )
    checks.update(source_checks(cases, meta))
    cache_path = Path(smoke["meta"]["cache_file"])
    cache_ok, cache_payload = cache_integrity(cache_path)
    route_ok, route_errors = smoke_route_integrity(smoke["directory"])
    formal = {name: case for name, case in cases.items() if name != SMOKE}
    candidate = formal.get("mxfp4_policy_a")
    checks.update(
        {
            "campaign_meta_matches_smoke": smoke["meta"].get(
                "source_bundle_sha256"
            )
            == expected_source
            and smoke["cache_after"] == expected_cache
            and base.sha256(cache_path) == expected_cache
            and all(
                meta.get(key) == smoke["meta"].get(key)
                for key in (
                    "lumen_tree_sha256",
                    "aiter_tree_sha256",
                    "runtime_modules_sha256",
                    "f4gemm_directory_sha256",
                    "aiter_config_cache_sha256",
                )
            ),
            "smoke_cache_absent_before": smoke["cache_before"] == "absent"
            and smoke["meta"].get("cache_sha256_before") == "absent",
            "fresh_cache_exact_protected_nine_asm_choices": cache_ok,
            "smoke_exact_policy_a_rank_route_and_shapes": route_ok,
            "shape_logging_only_for_smoke": smoke["meta"].get(
                "shape_log_enabled"
            )
            == "1"
            and all(
                case["meta"].get("shape_log_enabled") == "0"
                for case in formal.values()
            ),
            "available_formal_cache_replays_frozen": all(
                case["cache_before"] == expected_cache
                and case["cache_after"] == expected_cache
                for case in formal.values()
            ),
            "available_bf16_arms_no_mxfp4_autotune": all(
                not formal[name]["autotune_events"]
                for name in ("bf16_a1", "bf16_a2")
                if name in formal
            ),
            "candidate_loaded_nine_without_online_autotune": candidate is None
            or (
                not candidate["autotune_events"]
                and candidate["loaded_cache_counts"]
                and set(candidate["loaded_cache_counts"]) == {9}
            ),
            "candidate_policy_a_route_derived_from_fresh_smoke_and_source": candidate
            is None
            or (
                route_ok
                and route_check(candidate)
                and candidate["source_before"] == smoke["source_before"]
                and candidate["source_after"] == smoke["source_after"]
            ),
            "available_formal_commands_differ_only_by_precision_stack": len(formal)
            < 2
            or all(
                normalized_formal_command(case["meta"]["command"])
                == normalized_formal_command(
                    next(iter(formal.values()))["meta"]["command"]
                )
                for case in formal.values()
            ),
            "available_formal_lr_sequences_match": len(formal) < 2
            or all(
                [row["lr"] for row in case["steps"]]
                == [row["lr"] for row in next(iter(formal.values()))["steps"]]
                for case in formal.values()
            ),
        }
    )
    route_errors_by_case = {name: [] for name in cases}
    route_errors_by_case[SMOKE] = route_errors
    if complete:
        checks.update(sentinel_checks(meta, cases))
        starts = [cases[name]["started_epoch_ns"] for name in ORDER]
        checks["actual_start_order"] = all(
            first < second
            for first, second in zip(starts[:-1], starts[1:], strict=True)
        )
        checks["no_case_overlap"] = all(
            cases[current]["postflight_mtime_ns"]
            <= cases[following]["started_epoch_ns"]
            for current, following in zip(ORDER[:-1], ORDER[1:], strict=True)
        )
    return checks, route_errors_by_case, cache_payload, cases


def performance_pass(
    speedup_mean: float,
    speedup_median: float,
    wins: int,
    speedup_ci95: list[float],
) -> bool:
    return (
        speedup_mean >= MIN_SPEEDUP
        and speedup_median >= MIN_SPEEDUP
        and wins >= MIN_WINS
        and speedup_ci95[0] >= CI_LOWER_MINIMUM
    )


def compare_series(
    control: list[float],
    candidate: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    if len(control) != len(WINDOW) or len(candidate) != len(WINDOW):
        raise ValueError("comparison requires complete 40-point series")
    control_stats = base.describe(control)
    candidate_stats = base.describe(candidate)
    boot = base.bootstrap(
        control, candidate, resamples=resamples, seed=BOOTSTRAP_SEED
    )
    wins = sum(
        candidate_value < control_value
        for control_value, candidate_value in zip(control, candidate, strict=True)
    )
    speedup_mean = control_stats["mean_ms"] / candidate_stats["mean_ms"]
    speedup_median = control_stats["median_ms"] / candidate_stats["median_ms"]
    return {
        "control": control_stats,
        "candidate": candidate_stats,
        "saving_mean_ms": control_stats["mean_ms"] - candidate_stats["mean_ms"],
        "saving_median_ms": control_stats["median_ms"]
        - candidate_stats["median_ms"],
        "speedup_mean": speedup_mean,
        "speedup_median": speedup_median,
        "paired_wins": wins,
        "performance_pass": performance_pass(
            speedup_mean, speedup_median, wins, boot["speedup_ci95"]
        ),
        **boot,
    }


def bf16_drift(cases: dict[str, dict]) -> dict[str, float | bool]:
    first = cases["bf16_a1"]["window"]
    second = cases["bf16_a2"]["window"]
    mean_pct = (statistics.fmean(second) / statistics.fmean(first) - 1.0) * 100.0
    median_pct = (
        statistics.median(second) / statistics.median(first) - 1.0
    ) * 100.0
    return {
        "a1_mean_ms": statistics.fmean(first),
        "a2_mean_ms": statistics.fmean(second),
        "mean_relative_pct": mean_pct,
        "a1_median_ms": statistics.median(first),
        "a2_median_ms": statistics.median(second),
        "median_relative_pct": median_pct,
        "pass": abs(mean_pct) < DRIFT_LIMIT_PCT
        and abs(median_pct) < DRIFT_LIMIT_PCT,
    }


def confirmation_pass(
    *, integrity: bool, performance: bool, drift_pass: bool, nll_pass: bool
) -> bool:
    return integrity and performance and drift_pass and nll_pass


def validate_subset(names: tuple[str, ...], mode: str) -> dict:
    checks, route_errors, cache, _cases = integrity_for_cases(
        names, complete=False
    )
    return {
        "mode": mode,
        "passed": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "route_errors": route_errors,
        "cache_schema": cache.get("schema"),
        "cache_arch": cache.get("arch"),
        "cases_read": list(names),
    }


def build_report(*, bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    checks, route_errors, cache, cases = integrity_for_cases(ORDER, complete=True)
    arms = {name: cases[name] for name in ARMS}
    control = [
        statistics.fmean(pair)
        for pair in zip(
            arms["bf16_a1"]["window"],
            arms["bf16_a2"]["window"],
            strict=True,
        )
    ]
    candidate = arms["mxfp4_policy_a"]["window"]
    comparison = compare_series(control, candidate, resamples=bootstrap_resamples)
    drift = bf16_drift(arms)
    bf16_midpoint_nll = statistics.fmean(
        (arms["bf16_a1"]["validation_nll"], arms["bf16_a2"]["validation_nll"])
    )
    candidate_nll = arms["mxfp4_policy_a"]["validation_nll"]
    delta_nll = candidate_nll - bf16_midpoint_nll
    nll_pass = delta_nll <= NLL_LIMIT
    integrity_pass = all(checks.values())
    confirmed = confirmation_pass(
        integrity=integrity_pass,
        performance=bool(comparison["performance_pass"]),
        drift_pass=bool(drift["pass"]),
        nll_pass=nll_pass,
    )
    if not integrity_pass:
        decision = "analysis integrity failed; no trustworthy confirmation"
    elif confirmed:
        decision = "MXFP4 Policy A passes the preregistered short-run 1.6x bracket"
    else:
        decision = "MXFP4 Policy A did not pass every preregistered bracket gate"
    return {
        "protocol": {
            "formal_order": list(ARMS),
            "formal_steps": FORMAL_STEPS,
            "train_samples": TRAIN_SAMPLES,
            "timing_window": [WINDOW[0], WINDOW[-1]],
            "timing_points": len(WINDOW),
            "minimum_speedup_mean_and_median": MIN_SPEEDUP,
            "minimum_paired_wins": MIN_WINS,
            "bootstrap": {
                "method": "paired circular moving-block",
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "block_length": BLOCK_LENGTH,
                "speedup_ci_lower_minimum_inclusive": CI_LOWER_MINIMUM,
            },
            "bf16_replicate_drift_pct_strict": DRIFT_LIMIT_PCT,
            "delta_nll_limit": NLL_LIMIT,
            "candidate_policy": "tail1+packed_qkv+split_swiglu",
            "lm_head_precision": "bf16",
            "mxfp4_communication": "disabled",
        },
        "integrity": {
            "passed": integrity_pass,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "failed_checks": [key for key, value in checks.items() if not value],
            "route_errors": route_errors,
            "cache_schema": cache.get("schema"),
            "cache_arch": cache.get("arch"),
            "cache_choices": cache.get("choices"),
        },
        "artifacts": {
            name: {
                "log_sha256": case["log_sha256"],
                "validation_nll": case["validation_nll"],
                "peak_memory_gib": max(row["memory_gib"] for row in case["steps"]),
                "timing_window": base.describe(case["window"])
                if name in ARMS
                else None,
                "started_epoch_ns": case["started_epoch_ns"],
                "postflight_mtime_ns": case["postflight_mtime_ns"],
            }
            for name, case in cases.items()
        },
        "bf16_midpoint": base.describe(control),
        "mxfp4_policy_a": base.describe(candidate),
        "comparison": comparison,
        "bf16_replicate_drift": drift,
        "validation": {
            "bf16_a1_nll": arms["bf16_a1"]["validation_nll"],
            "bf16_a2_nll": arms["bf16_a2"]["validation_nll"],
            "bf16_midpoint_nll": bf16_midpoint_nll,
            "mxfp4_policy_a_nll": candidate_nll,
            "delta_nll_mxfp4_policy_a_vs_bf16_midpoint": delta_nll,
            "pass": nll_pass,
        },
        "confirmation_passed": confirmed,
        "decision": decision,
    }


def markdown_report(report: dict) -> str:
    integrity = report["integrity"]
    comparison = report["comparison"]
    drift = report["bf16_replicate_drift"]
    validation = report["validation"]
    ci = comparison["speedup_ci95"]
    lines = [
        "# Fresh BF16 / MXFP4 Policy A / BF16 bracket",
        "",
        f"Integrity: **{'PASS' if integrity['passed'] else 'FAIL'}** "
        f"({integrity['checks_passed']}/{integrity['checks_total']}); "
        f"final confirmation: **{'PASS' if report['confirmation_passed'] else 'FAIL'}**.",
        "",
        "Only fresh steps 11--50 enter the paired result. The BF16 control is "
        "the same-step midpoint of A1 and A2.",
        "",
        "| Metric | Result | Gate |",
        "|:---|---:|:---:|",
        f"| Mean speedup | {comparison['speedup_mean']:.6f}x | >= 1.6x |",
        f"| Median speedup | {comparison['speedup_median']:.6f}x | >= 1.6x |",
        f"| Paired wins | {comparison['paired_wins']}/40 | >= 28/40 |",
        f"| Block-4 95% speedup CI | [{ci[0]:.6f}x, {ci[1]:.6f}x] | lower >= 1.6x |",
        f"| BF16 mean drift | {drift['mean_relative_pct']:+.3f}% | abs < 3% |",
        f"| BF16 median drift | {drift['median_relative_pct']:+.3f}% | abs < 3% |",
        f"| Delta NLL | {validation['delta_nll_mxfp4_policy_a_vs_bf16_midpoint']:+.5f} | <= +0.01 |",
        "",
        "## Decision",
        "",
        report["decision"] + ".",
        "",
        f"confirmation_passed={str(report['confirmation_passed']).lower()}.",
    ]
    if integrity["failed_checks"]:
        lines.extend(["", "## Failed integrity checks", ""])
        lines.extend(f"- {name}" for name in integrity["failed_checks"])
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> None:
    json_target = ROOT / "final_policy_a_bf16_bracket_analysis.json"
    markdown_target = ROOT / "final_policy_a_bf16_bracket_analysis.md"
    json_tmp = json_target.with_suffix(".json.tmp")
    markdown_tmp = markdown_target.with_suffix(".md.tmp")
    json_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_tmp.write_text(markdown_report(report), encoding="utf-8")
    json_tmp.replace(json_target)
    markdown_tmp.replace(markdown_target)


def self_test() -> None:
    control = [160.0] * 40
    candidate = [100.0] * 40
    first = base.bootstrap(control, candidate, resamples=200, seed=BOOTSTRAP_SEED)
    second = base.bootstrap(control, candidate, resamples=200, seed=BOOTSTRAP_SEED)
    assert first == second
    assert first["speedup_ci95"][0] == 1.6
    assert performance_pass(1.6, 1.6, 28, [1.6, 1.7])
    assert not performance_pass(1.6, 1.6, 28, [1.599999, 1.7])
    assert confirmation_pass(
        integrity=True, performance=True, drift_pass=True, nll_pass=True
    )
    assert not confirmation_pass(
        integrity=False, performance=True, drift_pass=True, nll_pass=True
    )
    print("analyze_final_policy_a_bf16_bracket self-test: PASS")


def write_parse_failure(error: Exception) -> None:
    payload = {
        "integrity": {
            "passed": False,
            "checks_passed": 0,
            "checks_total": 1,
            "checks": {"analysis_parse_and_artifact_read": False},
            "failed_checks": ["analysis_parse_and_artifact_read"],
        },
        "confirmation_passed": False,
        "decision": "analysis failed closed before a trustworthy decision",
        "error": f"{type(error).__name__}: {error}",
    }
    (ROOT / "final_policy_a_bf16_bracket_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "final_policy_a_bf16_bracket_analysis.md").write_text(
        "# Fresh BF16 / MXFP4 Policy A / BF16 bracket\n\n"
        f"Analysis failed closed: {type(error).__name__}: {error}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-smoke", action="store_true")
    modes.add_argument("--validate-phase1", action="store_true")
    modes.add_argument("--validate-phase1-boundary", action="store_true")
    parser.add_argument("--test-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.validate_phase1_boundary:
        try:
            report = phase1_boundary_report()
        except Exception as error:
            print(
                json.dumps(
                    {
                        "mode": "validate-phase1-boundary",
                        "passed": False,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise SystemExit(2) from error
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(2)
        return
    if args.validate_smoke:
        names = (SMOKE,)
        mode = "validate-smoke"
    elif args.validate_phase1:
        names = PHASE1_ORDER
        mode = "validate-phase1"
    else:
        try:
            report = build_report(bootstrap_resamples=args.test_resamples)
            write_report(report)
        except Exception as error:
            write_parse_failure(error)
            print(
                f"analysis failed closed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["integrity"]["passed"]:
            raise SystemExit(2)
        return
    try:
        report = validate_subset(names, mode)
    except Exception as error:
        print(
            json.dumps(
                {"mode": mode, "passed": False, "error": f"{type(error).__name__}: {error}"},
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

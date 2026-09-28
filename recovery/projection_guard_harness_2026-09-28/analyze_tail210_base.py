#!/usr/bin/env python3
"""Analyze the fresh 50-step MXFP4 tail-2/tail-1/tail-0 campaign."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import shlex
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = "smoke_tail0"
ARMS = (
    "tail2_a1",
    "tail1_b1",
    "tail0_c",
    "tail1_b2",
    "tail2_a2",
)
TAILS = {
    "tail2_a1": 2,
    "tail1_b1": 1,
    "tail0_c": 0,
    "tail1_b2": 1,
    "tail2_a2": 2,
}
GROUPS = {
    2: ("tail2_a1", "tail2_a2"),
    1: ("tail1_b1", "tail1_b2"),
    0: ("tail0_c",),
}
EXPECTED_ORDER = (SMOKE, *ARMS)
PHASE1_ORDER = (SMOKE, "tail2_a1", "tail1_b1", "tail0_c")

SMOKE_STEPS = 3
FORMAL_STEPS = 50
TRAIN_SAMPLES = 6400
WINDOW = tuple(range(11, 51))
BOOTSTRAP_SEED = 20260922
BOOTSTRAP_RESAMPLES = 100_000
BLOCK_LENGTH = 4
PRECISION_GATE = 0.01
MIN_SPEEDUP = 1.01
MIN_WINS = 28
DRIFT_GATE_PCT = 3.0

LUMEN_COMMIT = "6b9aee1569247eca20937c14319ba6adcd69e0cb"
AITER_COMMIT = "e35bb17f4f815903bf73598facedbb321e15af28"
RUNNER_PATH = ROOT / "run_case.sh"
ENTRY_PATH = ROOT / "inventory_entry.py"
TRAIN_ENTRY_PATH = Path("/home/xdai/Lumen/examples/qwen3/train_qwen3_fsdp.py")
EXPECTED_RUNNER_SHA256 = (
    "3740d3308545e8c5f394525caac48e26b3fdbfe85d272f31b6ee4b2d1c7817a7"
)
EXPECTED_ENTRY_SHA256 = (
    "0a25a6579ff2838ad64935115ce407f23a563b5a67f37d7000a7de4d08145fdc"
)

PRIOR_ROOT = Path(
    "/home/xdai/profile-results/lumen-mxfp4-tail-boundary-fresh-20260921-xIqY6u"
)
PRIOR_DESIGN_ARTIFACTS = {
    "independent_runtime_audit_py": (
        PRIOR_ROOT / "independent_runtime_audit.py",
        "e338dd0bef0a3e41fa0b2928d2fca176bcff509ca95cf0a3df10e99823ff784a",
    ),
    "independent_runtime_audit_md": (
        PRIOR_ROOT / "independent_runtime_audit.md",
        "88a2d685a1d9ba8863afd277bd4ce804b3fb8c94c8c5f7049d94069f57d08ac0",
    ),
    "stage2_supplemental_v2_json": (
        PRIOR_ROOT / "stage2_analysis_supplemental_v2.json",
        "33b8f6003ba94190b009d292990dc8074766ebdbf3f2cbdb981340820e919576",
    ),
}

POSTFLIGHT_KEYS = {"torchrun", "tee", "rocm_smi", "post_idle", "provenance"}
TREE_KEYS = (
    "lumen_diff_sha256",
    "lumen_status_sha256",
    "aiter_diff_sha256",
    "aiter_status_sha256",
    "lumen_tree_sha256",
    "aiter_tree_sha256",
    "aiter_config_cache_sha256",
    "runtime_modules_sha256",
    "f4gemm_directory_sha256",
)
COMMON_PROVENANCE_KEYS = (
    "precision",
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
EXPECTED_SMOKE_COUNTS = {
    "enabled_qkv": 36,
    "enabled_swiglu": 36,
    "eligible_original_qkv_forward": 0,
    "eligible_original_swiglu_forward": 0,
    "instrumentation_installed": 1,
    "qkv_linear_failure": 0,
    "qkv_linear_success": 1440,
    "swiglu_bwd_failure": 0,
    "swiglu_bwd_success": 864,
    "swiglu_fwd_failure": 0,
    "swiglu_fwd_success": 1440,
    "world_size": 8,
}
EXPECTED_SMOKE_SHAPES = {
    (4096, 4096, 16384): 864,
    (4096, 12288, 16384): 864,
    (6144, 4096, 16384): 864,
    (12288, 4096, 16384): 1728,
    (16384, 4096, 4096): 2304,
    (16384, 4096, 6144): 864,
    (16384, 4096, 12288): 3168,
    (16384, 6144, 4096): 1440,
    (16384, 12288, 4096): 3744,
}

SWIGLU_TAIL_WARNING = (
    "MXFP4 split SwiGLU disabled for Qwen3MLP: both projections must already "
    "be quantized; using the original Qwen3MLP forward"
)
QKV_TAIL_WARNING = (
    "MXFP4 packed QKV disabled for Qwen3Attention: all Q/K/V projections must "
    "already be quantized; using the original Qwen3Attention forward"
)
STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+"
    r"loss\s+(?P<loss>\S+)\s+\|\s+grad_norm\s+(?P<grad>\S+)\s+\|\s+"
    r"lr\s+(?P<lr>\S+)\s+\|\s+step_time_ms\s+(?P<time>\S+)\s+\|\s+"
    r"peak_mem_gib\s+(?P<mem>\S+)"
)
MODEL_RE = re.compile(r"model_init_sha256=([0-9a-f]{64})")
TRAIN_DIGEST_RE = re.compile(
    r"PAIRING_EVIDENCE rank=(\d+) step=1 microbatches=8 "
    r"input_ids_labels_sha256=([0-9a-f]{64})"
)
VALIDATION_DIGEST_RE = re.compile(
    r"VALIDATION_EVIDENCE rank=(\d+) batches=16 "
    r"input_ids_labels_sha256=([0-9a-f]{64})"
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def parse_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if key in values:
            raise ValueError(f"duplicate key {key!r} in {path}")
        values[key] = value.strip()
    return values


def digest_artifact(path: Path, expected_target: Path | None = None) -> str:
    lines = read_text(path).splitlines()
    if len(lines) != 1:
        raise ValueError(f"invalid digest artifact: {path}")
    if expected_target is None:
        match = re.fullmatch(r"([0-9a-f]{64})", lines[0])
    else:
        match = re.fullmatch(
            rf"([0-9a-f]{{64}})  {re.escape(str(expected_target))}", lines[0]
        )
    if match is None:
        raise ValueError(f"invalid digest artifact: {path}")
    return match.group(1)


def rank_map(pattern: re.Pattern[str], text: str) -> dict[int, str]:
    matches = pattern.findall(text)
    result = {int(rank): value for rank, value in matches}
    if len(matches) != 8 or set(result) != set(range(8)):
        raise ValueError(f"rank evidence is not exactly ranks 0..7: {matches}")
    return result


def option_value(tokens: list[str], flag: str) -> str:
    separated = [index for index, token in enumerate(tokens) if token == flag]
    attached = [
        token.split("=", 1)[1] for token in tokens if token.startswith(flag + "=")
    ]
    if len(separated) + len(attached) != 1:
        raise ValueError(f"expected exactly one {flag} option")
    if attached:
        return attached[0]
    index = separated[0]
    if index + 1 >= len(tokens):
        raise ValueError(f"missing value for {flag}")
    return tokens[index + 1]


def flag_once(tokens: list[str], flag: str) -> bool:
    return tokens.count(flag) == 1


def normalized_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    value = option_value(tokens, "--num-layers-at-end-in-bf16")
    separated = [
        index
        for index, token in enumerate(tokens)
        if token == "--num-layers-at-end-in-bf16"
    ]
    if separated:
        tokens[separated[0] + 1] = "<TAIL_BF16>"
    else:
        token = f"--num-layers-at-end-in-bf16={value}"
        tokens[tokens.index(token)] = "--num-layers-at-end-in-bf16=<TAIL_BF16>"
    return tokens


def command_checks(
    command: str,
    *,
    tail: int,
    total_steps: int,
    meta: dict[str, str],
    smoke: bool,
) -> dict[str, bool]:
    tokens = shlex.split(command)
    expected_entry = ENTRY_PATH if smoke else TRAIN_ENTRY_PATH
    values = {
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
        "--mode": "mxfp4",
        "--fsdp-version": "2",
        "--sharding": "full_shard",
        "--fsdp-reduce-dtype": "bf16",
        "--num-layers-at-start-in-bf16": "0",
        "--num-layers-at-end-in-bf16": str(tail),
    }
    parsed: dict[str, str | None] = {}
    for flag in values:
        try:
            parsed[flag] = option_value(tokens, flag)
        except ValueError:
            parsed[flag] = None
    required_flags = (
        "--init-from-scratch",
        "--first-last-layers-bf16",
        "--fsdp-retain-accumulated-params",
        "--no-grad-checkpointing",
        "--mxfp4-pack-qkv",
        "--mxfp4-fuse-swiglu",
    )
    try:
        mbs = int(parsed["--micro-batch-size"] or "")
        ga = int(parsed["--gradient-accumulation-steps"] or "")
        world = int(parsed["--nproc-per-node"] or "")
    except ValueError:
        mbs = ga = world = -1
    return {
        "entrypoint": str(expected_entry) in tokens,
        "fixed_values": all(
            parsed[flag] == expected for flag, expected in values.items()
        ),
        "required_flags": all(flag_once(tokens, flag) for flag in required_flags),
        "no_mxfp4_communication": "--mxfp4-comm" not in tokens,
        "no_positive_grad_checkpointing_flag": "--grad-checkpointing" not in tokens,
        "global_batch_inferred": (
            mbs * ga * world == 128
            and meta.get("global_batch") == "128"
            and meta.get("gradient_accumulation") == "8"
        ),
        "train_samples_exact": parsed["--train-samples"] == str(TRAIN_SAMPLES),
    }


def expected_route(tail: int) -> dict[str, int]:
    return {
        "quantized": (36 - tail) * 7,
        "skipped": tail * 7 + 1,
        "enabled": 36 - tail,
        "warnings": tail * 8,
    }


def known_post_success_only(
    text: str,
    marker: str,
    *,
    smoke_shape_line: str | None = None,
) -> bool:
    """Allow the smoke flush pair, then only the known library teardown."""
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
            fragment not in actual
            for fragment, actual in zip(fragments, stripped, strict=True)
        ):
            return False
        ranks.append(rank)
        index += 10
    if not ranks:
        return True
    return len(ranks) == 16 and collections.Counter(ranks) == {
        rank: 2 for rank in range(8)
    }


def status_ok(case: dict) -> bool:
    clean_postflight = set(case["postflight"]) == POSTFLIGHT_KEYS and all(
        value == 0 for value in case["postflight"].values()
    )
    accepted_idle_race = (
        set(case["postflight"]) == POSTFLIGHT_KEYS
        and case["postflight"].get("torchrun") == 0
        and case["postflight"].get("tee") == 0
        and case["postflight"].get("rocm_smi") == 0
        and case["postflight"].get("provenance") == 0
        and case["postflight"].get("post_idle") == 1
        and case["postflight_recheck"].get("accepted_transient_postflight_kfd_race")
        == "1"
        and "status=idle" in case["kfd_recheck"]
    )
    return case["train_status"] == {"torchrun": 0, "tee": 0} and (
        clean_postflight or accepted_idle_race
    )


def parse_case(name: str, *, tail: int, total_steps: int) -> dict:
    directory = ROOT / name
    log_path = directory / "train.log"
    log = read_text(log_path)
    meta = parse_kv(directory / "run-meta.txt")
    cache_path = Path(meta["cache_file"])
    steps = [
        {
            "step": int(match.group("step")),
            "total": int(match.group("total")),
            "loss": float(match.group("loss")),
            "grad_norm": float(match.group("grad")),
            "lr": float(match.group("lr")),
            "time_ms": float(match.group("time")),
            "memory_gib": float(match.group("mem")),
        }
        for match in STEP_RE.finditer(log)
    ]
    validation = re.findall(
        rf"step\s+{total_steps}/{total_steps}\s+\|\s+val_loss\s+(\S+)", log
    )
    marker = f"Training complete after {total_steps} steps."
    before_complete = log.split(marker, 1)[0]
    disabled_lines = [
        line
        for line in before_complete.splitlines()
        if "MXFP4" in line and " disabled " in line
    ]
    unexpected_disabled = [
        line
        for line in disabled_lines
        if SWIGLU_TAIL_WARNING not in line and QKV_TAIL_WARNING not in line
    ]
    failure_lines = [
        line
        for line in before_complete.splitlines()
        if re.search(
            r"CUDA out of memory|HIP out of memory|OutOfMemoryError|RuntimeError:|"
            r"ChildFailedError|ProcessRaisedException|skipped update|emergency BF16|"
            r"illegal memory access|backend[^\n]*(?:failed|trying next)|"
            r"all backends[^\n]*exhausted|kernel[^\n]*fail|falling back|"
            r"unexpected fallback|\b(?:nan|inf)\b",
            line,
            flags=re.IGNORECASE,
        )
    ]
    smoke_shape_line = (
        "INFO:lumen.ops.quantize.mxfp4_autotune:MXFP4 shape log: wrote 9 "
        f"distinct shapes to {ROOT / SMOKE / 'mxfp4-shapes-rank0.csv'}"
        if name == SMOKE
        else None
    )
    route = expected_route(tail)
    return {
        "name": name,
        "tail": tail,
        "total_steps": total_steps,
        "directory": directory,
        "log": log,
        "log_sha256": sha256(log_path),
        "meta": meta,
        "steps": steps,
        "window": [step["time_ms"] for step in steps if step["step"] in WINDOW],
        "validation_nll": float(validation[0]) if len(validation) == 1 else math.nan,
        "model_hashes": MODEL_RE.findall(log),
        "train_digests": rank_map(TRAIN_DIGEST_RE, log),
        "validation_digests": rank_map(VALIDATION_DIGEST_RE, log),
        "complete": log.count(marker) == 1,
        "post_success_known": known_post_success_only(
            log, marker, smoke_shape_line=smoke_shape_line
        ),
        "failure_lines": failure_lines,
        "unexpected_disabled": unexpected_disabled,
        "swiglu_tail_warnings": before_complete.count(SWIGLU_TAIL_WARNING),
        "qkv_tail_warnings": before_complete.count(QKV_TAIL_WARNING),
        "quantized_reports": before_complete.count(
            f"Quantization enabled on {route['quantized']} nn.Linear layers"
        ),
        "skip_reports": before_complete.count(
            f"bf16_layers_skipped={route['skipped']}"
        ),
        "qkv_reports": before_complete.count(
            f"> MXFP4 packed QKV enabled on {route['enabled']} Qwen3 attention layers"
        ),
        "swiglu_reports": before_complete.count(
            f"> MXFP4 split SwiGLU enabled on {route['enabled']} Qwen3 MLPs"
        ),
        "loaded_cache_counts": [
            int(value) for value in re.findall(r"loaded (\d+) cached decisions", log)
        ],
        "autotune_events": re.findall(r"MXFP4 autotune \d+x\d+x\d+:", log),
        "train_status": {
            key: int(value)
            for key, value in parse_kv(directory / "train-exit-status.txt").items()
        },
        "postflight": {
            key: int(value)
            for key, value in parse_kv(directory / "postflight-status.txt").items()
        },
        "postflight_recheck": (
            parse_kv(directory / "postflight-recheck-status.txt")
            if directory.joinpath("postflight-recheck-status.txt").is_file()
            else {}
        ),
        "source_before": digest_artifact(directory / "source-bundle-before.sha256"),
        "source_after": digest_artifact(directory / "source-bundle-after.sha256"),
        "cache_before": (
            digest_artifact(directory / "cache-before.sha256", cache_path)
            if directory.joinpath("cache-before.sha256").is_file()
            else "absent"
        ),
        "cache_after": digest_artifact(directory / "cache-after.sha256", cache_path),
        "tree_after": parse_kv(directory / "tree-state-after.txt"),
        "kfd": [
            read_text(directory / f"kfd-{stage}.txt")
            for stage in ("before", "prelaunch", "after")
        ],
        "kfd_recheck": (
            read_text(directory / "kfd-after-recheck.txt")
            if directory.joinpath("kfd-after-recheck.txt").is_file()
            else ""
        ),
        "started_epoch_ns": int(meta["started_epoch_ns"]),
        "postflight_mtime_ns": (directory / "postflight-status.txt").stat().st_mtime_ns,
    }


def route_check(case: dict) -> bool:
    route = expected_route(case["tail"])
    return (
        case["quantized_reports"] == 1
        and case["skip_reports"] == 1
        and case["qkv_reports"] == 1
        and case["swiglu_reports"] == 1
        and case["swiglu_tail_warnings"] == route["warnings"]
        and case["qkv_tail_warnings"] == route["warnings"]
    )


def cache_integrity(path: Path) -> tuple[bool, dict]:
    payload = json.loads(read_text(path))
    profiles = payload.get("profiles", {})
    profile_ok = set(profiles) == set(EXPECTED_CACHE_CHOICES) and all(
        profile.get("winner") == "asm"
        and profile.get("identities", {}).get("asm", {}).get("implementation") == "asm"
        for profile in profiles.values()
    )
    valid = (
        payload.get("schema") == 6
        and payload.get("arch") == "gfx950"
        and payload.get("choices") == EXPECTED_CACHE_CHOICES
        and profile_ok
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
            report = json.loads(read_text(report_path))
            counts = report.get("counts", {})
            for key, expected in EXPECTED_SMOKE_COUNTS.items():
                if int(counts.get(key, 0)) != expected:
                    errors.append(
                        f"rank {rank}: {key}={counts.get(key, 0)} expected {expected}"
                    )
            if report.get("rank") != rank or int(counts.get("rank", -1)) != rank:
                errors.append(f"rank {rank}: rank identity mismatch")
            if report.get("unquantized_linear_names") != ["lm_head"]:
                errors.append(f"rank {rank}: sole unquantized linear is not lm_head")
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
                (int(row["M"]), int(row["N"]), int(row["K"])): row for row in raw_rows
            }
            if len(raw_rows) != 9 or set(rows) != set(EXPECTED_SMOKE_SHAPES):
                errors.append(f"rank {rank}: unexpected or duplicate shape rows")
                continue
            for shape, expected_calls in EXPECTED_SMOKE_SHAPES.items():
                row = rows[shape]
                if (
                    row.get("backend") != "asm"
                    or row.get("asm_available") != "1"
                    or int(row["calls"]) != expected_calls
                ):
                    errors.append(f"rank {rank}: shape mismatch {shape}: {row}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"rank {rank}: malformed route/shape artifact: {error}")
    return not errors, errors


def case_provenance_check(case: dict, expected_provenance: dict[str, str]) -> bool:
    meta = case["meta"]
    return (
        all(
            meta.get(key) == expected_provenance.get(key)
            and expected_provenance.get(key) not in (None, "")
            for key in COMMON_PROVENANCE_KEYS
        )
        and meta.get("precision") == "mxfp4"
        and meta.get("cache_namespace") == "tail210_confirm"
        and meta.get("gpu_lock_verified") == "1"
        and meta.get("lumen_import") == "/home/xdai/Lumen/lumen/__init__.py"
        and meta.get("aiter_import") == "/home/xdai/aiter/aiter/__init__.py"
        and meta.get("lumen_commit") == LUMEN_COMMIT
        and meta.get("aiter_commit") == AITER_COMMIT
        and meta.get("tail_bf16") == str(case["tail"])
        and meta.get("eval_batches") == "16"
        and meta.get("val_samples") == "256"
        and meta.get("numa_balancing") == "1"
        and meta.get("registry_freeze_removed") == "1"
        and meta.get("weight_cache_fast_hit_removed") == "1"
        and meta.get("mxfp4_activation_descriptor_cache") == "0"
        and all(
            case["tree_after"].get(key) == meta.get(key)
            and meta.get(key) not in (None, "")
            for key in TREE_KEYS
        )
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
    smoke: bool,
) -> dict[str, bool]:
    name = case["name"]
    total_steps = case["total_steps"]
    command = command_checks(
        case["meta"]["command"],
        tail=case["tail"],
        total_steps=total_steps,
        meta=case["meta"],
        smoke=smoke,
    )
    checks = {
        f"{name}_complete_steps": (
            case["complete"]
            and [step["step"] for step in case["steps"]]
            == list(range(1, total_steps + 1))
            and all(step["total"] == total_steps for step in case["steps"])
            and all(step["time_ms"] > 0 for step in case["steps"])
            and (smoke or len(case["window"]) == len(WINDOW))
        ),
        f"{name}_finite": (
            all(
                math.isfinite(step[field])
                for step in case["steps"]
                for field in ("loss", "grad_norm", "lr", "time_ms", "memory_gib")
            )
            and math.isfinite(case["validation_nll"])
        ),
        f"{name}_no_failure_or_unknown_post_success": (
            not case["failure_lines"]
            and not case["unexpected_disabled"]
            and case["post_success_known"]
        ),
        f"{name}_statuses": status_ok(case),
        f"{name}_source_frozen": (
            case["source_before"] == expected_source == case["source_after"]
        ),
        f"{name}_cache_final": case["cache_after"] == expected_cache,
        f"{name}_paired_model": (
            case["model_hashes"] == expected_model and len(expected_model) == 1
        ),
        f"{name}_paired_train_data": case["train_digests"] == expected_train,
        f"{name}_paired_validation_data": (
            case["validation_digests"] == expected_validation
        ),
        f"{name}_provenance": case_provenance_check(case, expected_provenance),
        f"{name}_route": route_check(case),
        f"{name}_kfd_idle": (
            "status=idle" in case["kfd"][0]
            and "status=idle" in case["kfd"][1]
            and (
                "status=idle" in case["kfd"][2] or "status=idle" in case["kfd_recheck"]
            )
        ),
        f"{name}_command": all(command.values()),
    }
    return checks


def metadata_protocol_checks(meta: dict[str, str]) -> dict[str, bool]:
    return {
        "metadata_protocol": (
            meta.get("schema") == "1"
            and meta.get("branch") == "dev/mxfp4"
            and meta.get("lumen_commit") == LUMEN_COMMIT
            and meta.get("aiter_commit") == AITER_COMMIT
            and meta.get("formal_order") == ",".join(ARMS)
            and meta.get("phase1_order") == ",".join(PHASE1_ORDER)
            and meta.get("formal_steps") == str(FORMAL_STEPS)
            and meta.get("train_samples") == str(TRAIN_SAMPLES)
            and meta.get("timing_window") == "11-50"
            and meta.get("precision_gate_delta_nll") == "0.01"
            and meta.get("speed_gate_min_ratio") == "1.01"
            and meta.get("speed_gate_min_wins") == "28"
            and meta.get("bootstrap_block_length") == "4"
            and meta.get("bootstrap_resamples") == "100000"
            and meta.get("replicate_drift_gate_pct") == "3.0"
            and meta.get("candidate_stack") == "packed_qkv+split_swiglu"
            and meta.get("lm_head_precision") == "bf16"
            and meta.get("numa_balancing") == "1"
            and re.fullmatch(r"[0-9a-f]{64}", meta.get("workload_sha256", ""))
            is not None
        ),
        "metadata_train_samples_matches_steps_times_gbs": (
            TRAIN_SAMPLES == FORMAL_STEPS * 128
        ),
    }


def harness_checks(meta: dict[str, str]) -> dict[str, bool]:
    paths = {
        "driver_sha256": ROOT / "run_tail210.sh",
        "analyzer_sha256": ROOT / "analyze_tail210.py",
        "self_test_sha256": ROOT / "test_analyze_tail210.py",
        "protocol_sha256": ROOT / "protocol.md",
        "runner_sha256": RUNNER_PATH,
        "entry_sha256": ENTRY_PATH,
    }
    checks = {
        key.removesuffix("_sha256") + "_frozen": (
            path.is_file() and meta.get(key) == sha256(path)
        )
        for key, path in paths.items()
    }
    checks["shared_runner_expected_sha256"] = (
        meta.get("runner_sha256") == EXPECTED_RUNNER_SHA256
    )
    checks["rank_entry_expected_sha256"] = (
        meta.get("entry_sha256") == EXPECTED_ENTRY_SHA256
    )
    return checks


def prior_design_checks(meta: dict[str, str]) -> tuple[dict[str, bool], dict]:
    checks: dict[str, bool] = {
        "role_recorded_as_design_provenance_only": (
            meta.get("prior_artifacts_role") == "design_provenance_only"
        )
    }
    artifact_hashes: dict[str, dict[str, str]] = {}
    for name, (path, expected) in PRIOR_DESIGN_ARTIFACTS.items():
        actual = sha256(path) if path.is_file() else "missing"
        meta_key = f"design_{name}_sha256"
        checks[f"prior_{name}_hash_locked"] = (
            actual == expected and meta.get(meta_key) == expected
        )
        artifact_hashes[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }

    prior_json_path, _expected = PRIOR_DESIGN_ARTIFACTS["stage2_supplemental_v2_json"]
    payload = json.loads(read_text(prior_json_path))
    facts = {
        "integrity_passed": payload.get("integrity", {}).get("passed") is True,
        "integrity_98_of_98": (
            payload.get("integrity", {}).get("checks_passed") == 98
            and payload.get("integrity", {}).get("checks_total") == 98
        ),
        "tail0_endpoint_passed": (
            payload.get("comparisons", {}).get("0", {}).get("endpoint_pass") is True
        ),
        "tail0_precision_passed": (
            payload.get("comparisons", {}).get("0", {}).get("precision_pass") is True
        ),
        "tail1_to_tail0_marginal_passed": (
            payload.get("marginal_comparisons", {}).get("0", {}).get("marginal_pass")
            is True
        ),
        "tail2_to_tail1_marginal_failed": (
            payload.get("marginal_comparisons", {}).get("1", {}).get("marginal_pass")
            is False
        ),
    }
    checks["prior_locked_design_facts_match"] = all(facts.values())
    details = {
        "role": "design_provenance_only",
        "used_for_current_timing_or_bootstrap": False,
        "used_for_current_unlock": False,
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": artifact_hashes,
        "locked_structure_facts": facts,
    }
    return checks, details


def smoke_validation() -> dict:
    """Validate smoke/cache/route state without reading any formal arm."""
    meta = parse_kv(ROOT / "tail210-meta.txt")
    smoke = parse_case(SMOKE, tail=0, total_steps=SMOKE_STEPS)
    expected_source = meta["source_bundle_sha256"]
    expected_cache = meta["fresh_autotune_cache_sha256"]
    expected_provenance = {
        key: smoke["meta"].get(key, "") for key in COMMON_PROVENANCE_KEYS
    }
    checks: dict[str, bool] = {}
    checks.update(metadata_protocol_checks(meta))
    checks.update(harness_checks(meta))
    checks.update(
        case_checks(
            smoke,
            expected_source=expected_source,
            expected_cache=expected_cache,
            expected_model=smoke["model_hashes"],
            expected_train=smoke["train_digests"],
            expected_validation=smoke["validation_digests"],
            expected_provenance=expected_provenance,
            smoke=True,
        )
    )
    cache_path = Path(smoke["meta"]["cache_file"])
    cache_ok, cache_payload = cache_integrity(cache_path)
    route_ok, route_errors = smoke_route_integrity(smoke["directory"])
    checks.update(
        {
            "smoke_meta_matches_campaign": (
                smoke["meta"].get("source_bundle_sha256") == expected_source
                and smoke["cache_after"] == expected_cache
                and sha256(cache_path) == expected_cache
                and meta.get("lumen_tree_sha256")
                == smoke["meta"].get("lumen_tree_sha256")
                and meta.get("aiter_tree_sha256")
                == smoke["meta"].get("aiter_tree_sha256")
                and meta.get("runtime_modules_sha256")
                == smoke["meta"].get("runtime_modules_sha256")
                and meta.get("f4gemm_directory_sha256")
                == smoke["meta"].get("f4gemm_directory_sha256")
                and meta.get("aiter_config_cache_sha256")
                == smoke["meta"].get("aiter_config_cache_sha256")
            ),
            "smoke_cache_was_absent": (
                smoke["cache_before"] == "absent"
                and smoke["meta"].get("cache_sha256_before") == "absent"
            ),
            "smoke_fresh_exact_nine_choice_cache": cache_ok,
            "smoke_rank_route_shape_lm_head_bf16": route_ok,
            "smoke_shape_logging_enabled": (
                smoke["meta"].get("shape_log_enabled") == "1"
            ),
            "smoke_tail0_lm_head_only": (
                smoke["quantized_reports"] == 1
                and smoke["skip_reports"] == 1
                and smoke["qkv_tail_warnings"] == 0
                and smoke["swiglu_tail_warnings"] == 0
                and route_ok
            ),
        }
    )
    return {
        "mode": "validate-smoke",
        "passed": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "route_errors": route_errors,
        "cache_schema": cache_payload.get("schema"),
        "cache_arch": cache_payload.get("arch"),
        "cache_choices": cache_payload.get("choices"),
        "formal_arms_read": [],
    }


def sentinel_checks(meta: dict[str, str]) -> dict[str, bool]:
    phase1 = parse_kv(ROOT / "phase1-complete.txt")
    formal = parse_kv(ROOT / "formal-arms-complete.txt")
    return {
        "phase1_exit_zero": (
            read_text(ROOT / "tail210-phase1-exit-status.txt").strip() == "0"
        ),
        "phase1_sentinel": (
            phase1.get("schema") == "1"
            and phase1.get("completed") == ",".join(PHASE1_ORDER)
            and phase1.get("source_bundle_sha256") == meta.get("source_bundle_sha256")
            and phase1.get("cache_sha256") == meta.get("fresh_autotune_cache_sha256")
            and phase1.get("meta_sha256") == sha256(ROOT / "tail210-meta.txt")
        ),
        "formal_completion_sentinel": (
            formal.get("schema") == "1"
            and formal.get("completed") == ",".join(ARMS)
            and formal.get("source_bundle_sha256") == meta.get("source_bundle_sha256")
            and formal.get("cache_sha256") == meta.get("fresh_autotune_cache_sha256")
        ),
        "phase_boundaries_idle": all(
            "status=idle" in read_text(ROOT / filename)
            for filename in (
                "kfd-tail210-phase1-before.txt",
                "kfd-tail210-phase1-after.txt",
                "kfd-tail210-phase2-before.txt",
                "kfd-tail210-phase2-after.txt",
            )
        ),
    }


def adjacent_order_checks(cases: dict[str, dict]) -> tuple[bool, bool]:
    ordered = [cases[name] for name in EXPECTED_ORDER]
    starts = [case["started_epoch_ns"] for case in ordered]
    start_order = all(
        first < second for first, second in zip(starts[:-1], starts[1:], strict=True)
    )
    nonoverlap = all(
        current["postflight_mtime_ns"] <= following["started_epoch_ns"]
        for current, following in zip(ordered[:-1], ordered[1:], strict=True)
    )
    return start_order, nonoverlap


def order_checks(cases: dict[str, dict]) -> dict[str, bool]:
    start_order, nonoverlap = adjacent_order_checks(cases)
    phase1 = parse_kv(ROOT / "phase1-complete.txt")
    formal = parse_kv(ROOT / "formal-arms-complete.txt")
    return {
        "actual_start_order": start_order,
        "no_case_overlap": nonoverlap,
        "phase1_sentinel_after_tail0_postflight": (
            int(phase1["completed_epoch_ns"]) >= cases["tail0_c"]["postflight_mtime_ns"]
        ),
        "phase2_started_after_phase1_completion": (
            int(phase1["completed_epoch_ns"]) < cases["tail1_b2"]["started_epoch_ns"]
        ),
        "formal_sentinel_after_last_postflight": (
            int(formal["completed_epoch_ns"])
            >= cases["tail2_a2"]["postflight_mtime_ns"]
        ),
    }


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot describe an empty series")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty series")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap(
    control: list[float],
    candidate: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    if (
        len(control) != len(candidate)
        or len(control) != len(WINDOW)
        or len(control) % BLOCK_LENGTH
        or resamples <= 0
    ):
        raise ValueError("bootstrap requires two complete 40-point paired series")
    rng = random.Random(seed)
    speedups: list[float] = []
    savings: list[float] = []
    block_count = len(control) // BLOCK_LENGTH
    for _ in range(resamples):
        indices: list[int] = []
        for _block in range(block_count):
            start = rng.randrange(len(control))
            indices.extend(
                (start + offset) % len(control) for offset in range(BLOCK_LENGTH)
            )
        control_mean = statistics.fmean(control[index] for index in indices)
        candidate_mean = statistics.fmean(candidate[index] for index in indices)
        speedups.append(control_mean / candidate_mean)
        savings.append(control_mean - candidate_mean)
    return {
        "sample_points": len(control),
        "resamples": resamples,
        "block_length": BLOCK_LENGTH,
        "seed": seed,
        "speedup_ci95": [percentile(speedups, 0.025), percentile(speedups, 0.975)],
        "saving_ms_ci95": [percentile(savings, 0.025), percentile(savings, 0.975)],
    }


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
        and speedup_ci95[0] > 1.0
    )


def compare_series(
    control: list[float],
    candidate: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    if len(control) != len(WINDOW) or len(candidate) != len(WINDOW):
        raise ValueError("comparison requires complete 40-point series")
    control_stats = describe(control)
    candidate_stats = describe(candidate)
    boot = bootstrap(control, candidate, resamples=resamples)
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
        "saving_median_ms": (control_stats["median_ms"] - candidate_stats["median_ms"]),
        "speedup_mean": speedup_mean,
        "speedup_median": speedup_median,
        "paired_wins": wins,
        "performance_pass": performance_pass(
            speedup_mean, speedup_median, wins, boot["speedup_ci95"]
        ),
        **boot,
    }


def midpoint_series(cases: dict[str, dict], names: tuple[str, ...]) -> list[float]:
    series = [cases[name]["window"] for name in names]
    if any(len(values) != len(WINDOW) for values in series):
        raise ValueError(f"incomplete timing window for {names}")
    return [statistics.fmean(values) for values in zip(*series, strict=True)]


def replicate_drift(
    cases: dict[str, dict], names: tuple[str, str]
) -> dict[str, float | bool]:
    first = midpoint_series(cases, (names[0],))
    second = midpoint_series(cases, (names[1],))
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    first_median = statistics.median(first)
    second_median = statistics.median(second)
    mean_pct = (second_mean / first_mean - 1.0) * 100.0
    median_pct = (second_median / first_median - 1.0) * 100.0
    return {
        "first_mean_ms": first_mean,
        "second_mean_ms": second_mean,
        "mean_relative_pct": mean_pct,
        "first_median_ms": first_median,
        "second_median_ms": second_median,
        "median_relative_pct": median_pct,
        "pass": abs(mean_pct) < DRIFT_GATE_PCT and abs(median_pct) < DRIFT_GATE_PCT,
    }


def select_tail(
    *,
    integrity_pass: bool,
    drift_pass: bool,
    performance: dict[str, bool],
    precision_tail1: bool,
    precision_tail0: bool,
) -> tuple[int | None, bool, dict[str, bool]]:
    tail1_chain = (
        integrity_pass and drift_pass and performance["2_to_1"] and precision_tail1
    )
    tail0_chain = (
        tail1_chain
        and performance["1_to_0"]
        and performance["2_to_0"]
        and precision_tail0
    )
    if not integrity_pass:
        selected: int | None = None
    elif tail0_chain:
        selected = 0
    elif tail1_chain:
        selected = 1
    else:
        selected = 2
    return (
        selected,
        tail0_chain,
        {
            "tail1_complete_chain_pass": tail1_chain,
            "tail0_complete_chain_pass": tail0_chain,
        },
    )


def build_report() -> dict:
    meta = parse_kv(ROOT / "tail210-meta.txt")
    smoke = parse_case(SMOKE, tail=0, total_steps=SMOKE_STEPS)
    arms = {
        name: parse_case(name, tail=TAILS[name], total_steps=FORMAL_STEPS)
        for name in ARMS
    }
    all_cases = {SMOKE: smoke, **arms}
    expected_source = meta["source_bundle_sha256"]
    expected_cache = meta["fresh_autotune_cache_sha256"]
    expected_model = smoke["model_hashes"]
    expected_train = smoke["train_digests"]
    expected_validation = smoke["validation_digests"]
    expected_provenance = {
        key: smoke["meta"].get(key, "") for key in COMMON_PROVENANCE_KEYS
    }

    checks: dict[str, bool] = {}
    checks.update(metadata_protocol_checks(meta))
    checks.update(harness_checks(meta))
    prior_checks, prior_details = prior_design_checks(meta)
    for name, case in all_cases.items():
        checks.update(
            case_checks(
                case,
                expected_source=expected_source,
                expected_cache=expected_cache,
                expected_model=expected_model,
                expected_train=expected_train,
                expected_validation=expected_validation,
                expected_provenance=expected_provenance,
                smoke=name == SMOKE,
            )
        )

    cache_path = Path(smoke["meta"]["cache_file"])
    cache_ok, cache_payload = cache_integrity(cache_path)
    route_ok, route_errors = smoke_route_integrity(smoke["directory"])
    checks.update(
        {
            "campaign_meta_matches_smoke": (
                smoke["meta"].get("source_bundle_sha256") == expected_source
                and smoke["cache_after"] == expected_cache
                and sha256(cache_path) == expected_cache
                and all(
                    meta.get(key) == smoke["meta"].get(key)
                    for key in (
                        "lumen_tree_sha256",
                        "aiter_tree_sha256",
                        "runtime_modules_sha256",
                        "f4gemm_directory_sha256",
                        "aiter_config_cache_sha256",
                    )
                )
            ),
            "smoke_cache_was_absent": (
                smoke["cache_before"] == "absent"
                and smoke["meta"].get("cache_sha256_before") == "absent"
            ),
            "smoke_exact_nine_choice_cache": cache_ok,
            "smoke_rank_route_shape_lm_head_bf16": route_ok,
            "smoke_shape_logging_enabled_only_for_smoke": (
                smoke["meta"].get("shape_log_enabled") == "1"
                and all(
                    arm["meta"].get("shape_log_enabled") == "0" for arm in arms.values()
                )
            ),
            "formal_cache_frozen": all(
                arm["cache_before"] == expected_cache
                and arm["cache_after"] == expected_cache
                for arm in arms.values()
            ),
            "formal_no_online_autotune": all(
                not arm["autotune_events"]
                and arm["loaded_cache_counts"]
                and set(arm["loaded_cache_counts"]) == {9}
                for arm in arms.values()
            ),
            "formal_commands_differ_only_by_tail": all(
                normalized_command(arm["meta"]["command"])
                == normalized_command(arms[ARMS[0]]["meta"]["command"])
                for arm in arms.values()
            ),
            "formal_lr_sequences_match": all(
                [step["lr"] for step in arm["steps"]]
                == [step["lr"] for step in arms[ARMS[0]]["steps"]]
                for arm in arms.values()
            ),
        }
    )
    checks.update(sentinel_checks(meta))
    checks.update(order_checks(all_cases))

    tail2_series = midpoint_series(arms, GROUPS[2])
    tail1_series = midpoint_series(arms, GROUPS[1])
    tail0_series = midpoint_series(arms, GROUPS[0])
    comparisons = {
        "2_to_1": compare_series(tail2_series, tail1_series),
        "1_to_0": compare_series(tail1_series, tail0_series),
        "2_to_0": compare_series(tail2_series, tail0_series),
    }

    validation = {
        "tail2_midpoint_nll": statistics.fmean(
            arms[name]["validation_nll"] for name in GROUPS[2]
        ),
        "tail1_midpoint_nll": statistics.fmean(
            arms[name]["validation_nll"] for name in GROUPS[1]
        ),
        "tail0_nll": arms["tail0_c"]["validation_nll"],
    }
    validation["tail1_delta_nll_vs_tail2"] = (
        validation["tail1_midpoint_nll"] - validation["tail2_midpoint_nll"]
    )
    validation["tail0_delta_nll_vs_tail2"] = (
        validation["tail0_nll"] - validation["tail2_midpoint_nll"]
    )
    validation["tail0_delta_nll_vs_tail1"] = (
        validation["tail0_nll"] - validation["tail1_midpoint_nll"]
    )
    validation["tail1_precision_pass"] = (
        validation["tail1_delta_nll_vs_tail2"] <= PRECISION_GATE
    )
    validation["tail0_precision_pass"] = (
        validation["tail0_delta_nll_vs_tail2"] <= PRECISION_GATE
    )

    drift = {
        "tail2": replicate_drift(arms, GROUPS[2]),
        "tail1": replicate_drift(arms, GROUPS[1]),
    }
    drift_gate_pass = all(item["pass"] for item in drift.values())
    integrity_pass = all(checks.values())
    selected, tail0_unlocked, chain = select_tail(
        integrity_pass=integrity_pass,
        drift_pass=drift_gate_pass,
        performance={
            name: item["performance_pass"] for name, item in comparisons.items()
        },
        precision_tail1=bool(validation["tail1_precision_pass"]),
        precision_tail0=bool(validation["tail0_precision_pass"]),
    )

    if not integrity_pass:
        decision = "analysis integrity failed; no trustworthy selection"
    elif tail0_unlocked:
        decision = "tail0 unlocked for a fresh matched BF16 confirmation campaign"
    elif selected == 1:
        decision = (
            "tail0 did not pass its full chain; advance tail1 to BF16 confirmation"
        )
    else:
        decision = "retain tail2; no less-protected tail passed its complete chain"

    return {
        "protocol": {
            "fresh_measurements_only": True,
            "formal_steps": FORMAL_STEPS,
            "train_samples": TRAIN_SAMPLES,
            "timing_window": [WINDOW[0], WINDOW[-1]],
            "timing_points": len(WINDOW),
            "formal_order": list(ARMS),
            "bootstrap": {
                "method": "paired circular moving-block",
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "block_length": BLOCK_LENGTH,
            },
            "minimum_speedup_mean_and_median": MIN_SPEEDUP,
            "minimum_paired_wins": MIN_WINS,
            "speedup_ci_lower_strictly_greater_than": 1.0,
            "replicate_drift_gate_pct_strict": DRIFT_GATE_PCT,
            "precision_gate_delta_nll_vs_tail2": PRECISION_GATE,
            "lm_head_precision": "bf16",
        },
        "integrity": {
            "passed": integrity_pass,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "smoke_route_errors": route_errors,
            "cache_schema": cache_payload.get("schema"),
            "cache_arch": cache_payload.get("arch"),
            "cache_choices": cache_payload.get("choices"),
        },
        "prior_design_provenance": prior_details,
        "artifacts": {
            name: {
                "log_sha256": case["log_sha256"],
                "validation_nll": case["validation_nll"],
                "peak_memory_gib": max(step["memory_gib"] for step in case["steps"]),
                "timing_window": (describe(case["window"]) if name in ARMS else None),
                "started_epoch_ns": case["started_epoch_ns"],
                "postflight_mtime_ns": case["postflight_mtime_ns"],
            }
            for name, case in all_cases.items()
        },
        "series": {
            "tail2_midpoint": describe(tail2_series),
            "tail1_midpoint": describe(tail1_series),
            "tail0": describe(tail0_series),
        },
        "comparisons": comparisons,
        "validation": validation,
        "replicate_drift": drift,
        "drift_gate_pass": drift_gate_pass,
        "selection_chain": chain,
        "selected_tail_for_bf16_confirmation": selected,
        "tail0_unlocked_for_bf16_confirmation": tail0_unlocked,
        "decision": decision,
    }


def markdown_report(report: dict) -> str:
    integrity = report["integrity"]
    lines = [
        "# Fresh MXFP4 tail-2/tail-1/tail-0 confirmation",
        "",
        f"Integrity: **{'PASS' if integrity['passed'] else 'FAIL'}** "
        f"({integrity['checks_passed']}/{integrity['checks_total']}); "
        f"replicate drift: **{'PASS' if report['drift_gate_pass'] else 'FAIL'}**.",
        "",
        "All performance statistics below use only this campaign's steps 11--50. "
        "The locked prior artifacts are design provenance only and contribute "
        "no timing sample, bootstrap draw, NLL gate, or unlock condition.",
        "",
        "| Comparison | Mean speedup | Median speedup | Saving ms | Wins | "
        "95% speedup CI | Gate |",
        "|:---|---:|---:|---:|---:|---:|:---:|",
    ]
    labels = {
        "2_to_1": "tail-2 → tail-1",
        "1_to_0": "tail-1 → tail-0",
        "2_to_0": "tail-2 → tail-0",
    }
    for key in ("2_to_1", "1_to_0", "2_to_0"):
        item = report["comparisons"][key]
        ci = item["speedup_ci95"]
        lines.append(
            f"| {labels[key]} | {item['speedup_mean']:.6f}x | "
            f"{item['speedup_median']:.6f}x | {item['saving_mean_ms']:.3f} | "
            f"{item['paired_wins']}/40 | [{ci[0]:.6f}x, {ci[1]:.6f}x] | "
            f"{'PASS' if item['performance_pass'] else 'FAIL'} |"
        )

    validation = report["validation"]
    lines.extend(
        [
            "",
            "## Validation NLL",
            "",
            "| Candidate | NLL | ΔNLL vs tail-2 | Gate |",
            "|:---|---:|---:|:---:|",
            f"| tail-1 | {validation['tail1_midpoint_nll']:.5f} | "
            f"{validation['tail1_delta_nll_vs_tail2']:+.5f} | "
            f"{'PASS' if validation['tail1_precision_pass'] else 'FAIL'} |",
            f"| tail-0 | {validation['tail0_nll']:.5f} | "
            f"{validation['tail0_delta_nll_vs_tail2']:+.5f} | "
            f"{'PASS' if validation['tail0_precision_pass'] else 'FAIL'} |",
            "",
            f"Non-gating tail-0 minus tail-1 ΔNLL: "
            f"{validation['tail0_delta_nll_vs_tail1']:+.5f}.",
            "",
            "## Replicate drift",
            "",
        ]
    )
    for tail in ("tail2", "tail1"):
        item = report["replicate_drift"][tail]
        lines.append(
            f"- {tail}: mean {item['mean_relative_pct']:+.3f}%, median "
            f"{item['median_relative_pct']:+.3f}% — "
            f"{'PASS' if item['pass'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"] + ".",
            "",
            f"Selected tail: "
            f"{report['selected_tail_for_bf16_confirmation']}; "
            f"tail0_unlocked_for_bf16_confirmation="
            f"{str(report['tail0_unlocked_for_bf16_confirmation']).lower()}.",
        ]
    )
    if integrity["failed_checks"]:
        lines.extend(["", "## Failed integrity checks", ""])
        lines.extend(f"- {name}" for name in integrity["failed_checks"])
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> None:
    json_target = ROOT / "tail210_analysis.json"
    markdown_target = ROOT / "tail210_analysis.md"
    json_temporary = json_target.with_suffix(".json.tmp")
    markdown_temporary = markdown_target.with_suffix(".md.tmp")
    json_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_temporary.write_text(markdown_report(report), encoding="utf-8")
    json_temporary.replace(json_target)
    markdown_temporary.replace(markdown_target)


def self_test() -> None:
    control = [100.0] * 40
    candidate = [98.0] * 40
    first = bootstrap(control, candidate, resamples=200)
    second = bootstrap(control, candidate, resamples=200)
    assert first == second
    assert first["sample_points"] == 40
    assert first["speedup_ci95"][0] > 1.0
    assert performance_pass(1.01, 1.01, 28, [1.0001, 1.02])
    assert not performance_pass(1.01, 1.01, 27, [1.0001, 1.02])
    cases = {
        name: {"started_epoch_ns": index * 10, "postflight_mtime_ns": index * 10 + 5}
        for index, name in enumerate(EXPECTED_ORDER)
    }
    assert adjacent_order_checks(cases) == (True, True)
    selected, unlocked, _chain = select_tail(
        integrity_pass=True,
        drift_pass=True,
        performance={"2_to_1": True, "1_to_0": True, "2_to_0": True},
        precision_tail1=True,
        precision_tail0=True,
    )
    assert selected == 0 and unlocked
    print("analyze_tail210 self-test: PASS")


def write_parse_failure(error: Exception) -> None:
    report = {
        "integrity": {
            "passed": False,
            "checks_passed": 0,
            "checks_total": 1,
            "checks": {"analysis_parse_and_artifact_read": False},
            "failed_checks": ["analysis_parse_and_artifact_read"],
        },
        "tail0_unlocked_for_bf16_confirmation": False,
        "selected_tail_for_bf16_confirmation": None,
        "decision": "analysis failed closed before a trustworthy decision",
        "error": f"{type(error).__name__}: {error}",
    }
    (ROOT / "tail210_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "tail210_analysis.md").write_text(
        "# Fresh MXFP4 tail confirmation\n\n"
        f"Analysis failed closed: {type(error).__name__}: {error}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-smoke", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.validate_smoke:
        try:
            report = smoke_validation()
        except Exception as error:
            print(
                json.dumps(
                    {
                        "mode": "validate-smoke",
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
    try:
        report = build_report()
        write_report(report)
    except Exception as error:
        write_parse_failure(error)
        print(
            f"analysis failed closed: {type(error).__name__}: {error}", file=sys.stderr
        )
        raise SystemExit(2) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["integrity"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

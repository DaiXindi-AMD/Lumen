#!/usr/bin/env python3
"""Analyze the fresh MXFP4 final-layer projection-guard campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shlex
import statistics
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "analyze_tail210_base.py"
SHARED_RUNNER = ROOT / "run_case.sh"
FIXTURE_DIR = ROOT / "fixtures" / "policy_b_smoke"
FIXTURE_PATHS = tuple(FIXTURE_DIR / f"mxfp4-shapes-rank{rank}.csv" for rank in range(8))
SMOKE = "route_guard_o_down_fresh"
ARMS = (
    "tail1_a1",
    "guard_o_down_b1",
    "guard_down_c",
    "guard_o_down_b2",
    "tail1_a2",
)
PHASE1_ARMS = ARMS[:3]
EXPECTED_ORDER = (SMOKE, *ARMS)
GROUPS = {
    "a": ("tail1_a1", "tail1_a2"),
    "b": ("guard_o_down_b1", "guard_o_down_b2"),
    "c": ("guard_down_c",),
}
POLICIES = {
    SMOKE: (0, ("o_proj", "down_proj"), "b"),
    "tail1_a1": (1, (), "a"),
    "guard_o_down_b1": (0, ("o_proj", "down_proj"), "b"),
    "guard_down_c": (0, ("down_proj",), "c"),
    "guard_o_down_b2": (0, ("o_proj", "down_proj"), "b"),
    "tail1_a2": (1, (), "a"),
}
SMOKE_STEPS = 3
FORMAL_STEPS = 50
TRAIN_SAMPLES = 6400
WINDOW = tuple(range(11, 51))
PRECISION_GATE = 0.01
MIN_SPEEDUP = 1.003
MIN_WINS = 28
DRIFT_GATE_PCT = 3.0
BOOTSTRAP_RESAMPLES = 100_000
BLOCK_LENGTH = 4
CAMPAIGN_NAMESPACE = "projection_guard"
LUMEN_COMMIT = "6b9aee1569247eca20937c14319ba6adcd69e0cb"
AITER_COMMIT = "e35bb17f4f815903bf73598facedbb321e15af28"
ENTRY = ROOT / "inventory_entry.py"
TRAIN_ENTRY = Path("/home/xdai/Lumen/examples/qwen3/train_qwen3_fsdp.py")
EXCLUDES = ROOT / "provenance-git-excludes"
EXPECTED_SHARED_RUNNER_SHA = (
    "3740d3308545e8c5f394525caac48e26b3fdbfe85d272f31b6ee4b2d1c7817a7"
)
EXPECTED_BASE_SHA = "ab6973743dabd1042a431304662e5d2805cff7d98aabd671df2c418b967009d6"
EXPECTED_TRAIN_ENTRY_SHA = (
    "ffd60a612d522c2e125fb2a622da8bd9fcd0c4c5bb478ad6a26b544c9e38a74f"
)
EXPECTED_FIXTURE_SHA = (
    "6d68e8566b6ac24603cece0b20f0c811fb4b07119e8b53767e5404c35775b271"
)

LAST = "model.layers.35"
EXPECTED_UNQUANTIZED = {
    "a": [
        f"{LAST}.self_attn.q_proj",
        f"{LAST}.self_attn.k_proj",
        f"{LAST}.self_attn.v_proj",
        f"{LAST}.self_attn.o_proj",
        f"{LAST}.mlp.gate_proj",
        f"{LAST}.mlp.up_proj",
        f"{LAST}.mlp.down_proj",
        "lm_head",
    ],
    "b": [f"{LAST}.self_attn.o_proj", f"{LAST}.mlp.down_proj", "lm_head"],
    "c": [f"{LAST}.mlp.down_proj", "lm_head"],
}
EXPECTED_ENABLED = {"a": 35, "b": 36, "c": 36}
EXPECTED_POLICY_SHAPE_TOTALS = {
    "a": {
        (4096, 4096, 16384): 840,
        (4096, 12288, 16384): 840,
        (6144, 4096, 16384): 840,
        (12288, 4096, 16384): 1680,
        (16384, 4096, 4096): 2240,
        (16384, 4096, 6144): 840,
        (16384, 4096, 12288): 3080,
        (16384, 6144, 4096): 1400,
        (16384, 12288, 4096): 3640,
    },
    "b": {
        (4096, 4096, 16384): 840,
        (4096, 12288, 16384): 840,
        (6144, 4096, 16384): 864,
        (12288, 4096, 16384): 1728,
        (16384, 4096, 4096): 2240,
        (16384, 4096, 6144): 864,
        (16384, 4096, 12288): 3128,
        (16384, 6144, 4096): 1440,
        (16384, 12288, 4096): 3720,
    },
    "c": {
        (4096, 4096, 16384): 864,
        (4096, 12288, 16384): 840,
        (6144, 4096, 16384): 864,
        (12288, 4096, 16384): 1728,
        (16384, 4096, 4096): 2304,
        (16384, 4096, 6144): 864,
        (16384, 4096, 12288): 3128,
        (16384, 6144, 4096): 1440,
        (16384, 12288, 4096): 3720,
    },
}
SOURCE_LINE = re.compile(r"^([0-9a-f]{64})  (/.*)$")
CRITICAL_SOURCES = {
    str(SHARED_RUNNER),
    str(BASE_PATH),
    str(ENTRY),
    str(TRAIN_ENTRY),
    str(ROOT / "run_projection_guard.sh"),
    str(ROOT / "analyze_projection_guard.py"),
    str(ROOT / "test_analyze_projection_guard.py"),
    str(ROOT / "protocol.md"),
    str(EXCLUDES),
    "/home/xdai/Lumen/lumen/models/fsdp.py",
    "/home/xdai/Lumen/lumen/models/qwen3.py",
    "/home/xdai/Lumen/lumen/ops/fused_swiglu.py",
    "/home/xdai/Lumen/lumen/ops/quantize/linear.py",
    "/home/xdai/Lumen/lumen/ops/quantize/mxfp4_autotune.py",
    "/home/xdai/Lumen/lumen/ops/quantize/mxfp4_asm.py",
    "/home/xdai/aiter/aiter/ops/gemm_op_a4w4.py",
    "/home/xdai/aiter/aiter/ops/triton/activation.py",
    "/home/xdai/aiter/aiter/ops/triton/_triton_kernels/activation.py",
    *(str(path) for path in FIXTURE_PATHS),
}


def _load_base() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tail210_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base analyzer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.ROOT = ROOT
base.SMOKE = SMOKE
base.ARMS = ARMS
base.TAILS = {name: POLICIES[name][0] for name in ARMS}
base.GROUPS = {1: GROUPS["a"], 0: (*GROUPS["b"], *GROUPS["c"])}
base.EXPECTED_ORDER = EXPECTED_ORDER
base.PHASE1_ORDER = (SMOKE, *PHASE1_ARMS)
base.ENTRY_PATH = ENTRY
base.TRAIN_ENTRY_PATH = ENTRY
base.WINDOW = WINDOW
base.TRAIN_SAMPLES = TRAIN_SAMPLES
base.MIN_SPEEDUP = MIN_SPEEDUP
base.MIN_WINS = MIN_WINS
base.LUMEN_COMMIT = LUMEN_COMMIT
base.RUNNER_PATH = SHARED_RUNNER


def policy_projection_args(tokens: list[str]) -> tuple[str, ...]:
    flag = "--mxfp4-last-layer-bf16-projections"
    positions = [index for index, token in enumerate(tokens) if token == flag]
    attached = [token for token in tokens if token.startswith(flag + "=")]
    if attached or len(positions) > 1:
        raise ValueError("projection guard must use one separated option")
    if not positions:
        return ()
    values: list[str] = []
    for token in tokens[positions[0] + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return tuple(values)


def policy_command_check(case: dict) -> bool:
    tail, projections, _policy = POLICIES[case["name"]]
    tokens = shlex.split(case["meta"]["command"])
    try:
        actual_tail = int(base.option_value(tokens, "--num-layers-at-end-in-bf16"))
        actual_projections = policy_projection_args(tokens)
    except (TypeError, ValueError):
        return False
    return actual_tail == tail and actual_projections == projections


def normalized_policy_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    tail_flag = "--num-layers-at-end-in-bf16"
    tail_index = tokens.index(tail_flag)
    tokens[tail_index + 1] = "<POLICY_TAIL>"
    projection_flag = "--mxfp4-last-layer-bf16-projections"
    if projection_flag in tokens:
        start = tokens.index(projection_flag)
        end = start + 1
        while end < len(tokens) and not tokens[end].startswith("--"):
            end += 1
        tokens[start:end] = ["<POLICY_PROJECTIONS>"]
    else:
        tokens.append("<POLICY_PROJECTIONS>")
    return tokens


def source_manifest(case: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    text = base.read_text(case["directory"] / "run-meta.txt")
    for line in text.splitlines():
        match = SOURCE_LINE.fullmatch(line)
        if match:
            digest, path = match.groups()
            if path in result:
                raise ValueError(f"duplicate source path in manifest: {path}")
            result[path] = digest
    if not result:
        raise ValueError(f"empty source manifest for {case['name']}")
    return result


def fixture_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in FIXTURE_PATHS:
        digest.update(f"{base.sha256(path)}  {path.name}\n".encode("utf-8"))
    return digest.hexdigest()


def route_inventory(case: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    policy = POLICIES[case["name"]][2]
    expected_names = EXPECTED_UNQUANTIZED[policy]
    expected_enabled = EXPECTED_ENABLED[policy]
    for rank in range(8):
        path = case["directory"] / f"route-rank{rank}.json"
        if not path.is_file():
            errors.append(f"rank {rank}: missing route inventory")
            continue
        try:
            report = json.loads(base.read_text(path))
            counts = report.get("counts", {})
            expected_counts = {
                "rank": rank,
                "world_size": 8,
                "inventory_captured": 1,
                "enabled_qkv": expected_enabled,
                "enabled_swiglu": expected_enabled,
            }
            if counts != expected_counts:
                errors.append(
                    f"rank {rank}: counts={counts}, expected={expected_counts}"
                )
            if report.get("rank") != rank:
                errors.append(f"rank {rank}: top-level rank mismatch")
            if report.get("unquantized_linear_names") != expected_names:
                errors.append(
                    f"rank {rank}: unquantized={report.get('unquantized_linear_names')}"
                )
            if report.get("lm_head_count") != 1:
                errors.append(f"rank {rank}: lm_head count is not one")
            if report.get("lm_head_weight_dtype") != "bfloat16":
                errors.append(f"rank {rank}: lm_head is not BF16")
            if report.get("lm_head_quant_enabled") is not False:
                errors.append(f"rank {rank}: lm_head unexpectedly quantized")
            if report.get("lumen_import") != "/home/xdai/Lumen/lumen/__init__.py":
                errors.append(f"rank {rank}: unexpected Lumen import")
            if report.get("aiter_import") != "/home/xdai/aiter/aiter/__init__.py":
                errors.append(f"rank {rank}: unexpected AITER import")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"rank {rank}: malformed route inventory: {error}")
    return not errors, errors


def shape_inventory(directory: Path, policy: str) -> tuple[bool, list[str]]:
    expected = EXPECTED_POLICY_SHAPE_TOTALS[policy]
    errors: list[str] = []
    for rank in range(8):
        path = directory / f"mxfp4-shapes-rank{rank}.csv"
        if not path.is_file():
            errors.append(f"rank {rank}: missing shape inventory")
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                raw = list(csv.DictReader(handle))
            rows = {(int(row["M"]), int(row["N"]), int(row["K"])): row for row in raw}
            if len(raw) != 9 or set(rows) != set(expected):
                errors.append(f"rank {rank}: unexpected or duplicate shape rows")
                continue
            for shape, calls in expected.items():
                row = rows[shape]
                if (
                    row.get("backend") != "asm"
                    or row.get("asm_available") != "1"
                    or int(row["calls"]) != calls
                ):
                    errors.append(f"rank {rank}: shape mismatch {shape}: {row}")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"rank {rank}: malformed shape inventory: {error}")
    return not errors, errors


def smoke_shapes(case: dict) -> tuple[bool, list[str]]:
    policy = POLICIES[case["name"]][2]
    return shape_inventory(case["directory"], policy)


def campaign_case_provenance(case: dict, expected: dict[str, str]) -> bool:
    meta = case["meta"]
    return (
        all(
            meta.get(key) == expected.get(key) and expected.get(key) not in (None, "")
            for key in base.COMMON_PROVENANCE_KEYS
        )
        and meta.get("precision") == "mxfp4"
        and meta.get("cache_namespace") == CAMPAIGN_NAMESPACE
        and meta.get("gpu_lock_verified") == "1"
        and meta.get("lumen_import") == "/home/xdai/Lumen/lumen/__init__.py"
        and meta.get("aiter_import") == "/home/xdai/aiter/aiter/__init__.py"
        and meta.get("lumen_commit") == LUMEN_COMMIT
        and meta.get("aiter_commit") == AITER_COMMIT
        and meta.get("tail_bf16") == str(POLICIES[case["name"]][0])
        and meta.get("eval_batches") == "16"
        and meta.get("val_samples") == "256"
        and meta.get("numa_balancing") == "1"
        and meta.get("registry_freeze_removed") == "1"
        and meta.get("weight_cache_fast_hit_removed") == "1"
        and meta.get("mxfp4_activation_descriptor_cache") == "0"
        and all(
            case["tree_after"].get(key) == meta.get(key)
            and meta.get(key) not in (None, "")
            for key in base.TREE_KEYS
        )
    )


base.case_provenance_check = campaign_case_provenance


def parse_case(name: str) -> dict:
    steps = SMOKE_STEPS if name == SMOKE else FORMAL_STEPS
    return base.parse_case(name, tail=POLICIES[name][0], total_steps=steps)


def case_train_samples(case: dict) -> int:
    tokens = shlex.split(case["meta"]["command"])
    return int(base.option_value(tokens, "--train-samples"))


def formal_train_digest_references(cases: dict[str, dict]) -> dict[str, dict[int, str]]:
    groups: dict[int, list[dict]] = {}
    for name in ARMS:
        if name in cases:
            groups.setdefault(case_train_samples(cases[name]), []).append(cases[name])
    references: dict[str, dict[int, str]] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        expected = group[0]["train_digests"]
        references.update({case["name"]: expected for case in group})
    return references


def common_case_checks(
    case: dict,
    *,
    expected_source: str,
    expected_cache: str,
    expected_model: list[str],
    expected_train: dict[int, str] | None,
    expected_validation: dict[int, str],
    expected_provenance: dict[str, str],
) -> tuple[dict[str, bool], list[str]]:
    checks = base.case_checks(
        case,
        expected_source=expected_source,
        expected_cache=expected_cache,
        expected_model=expected_model,
        expected_train=(
            expected_train if expected_train is not None else case["train_digests"]
        ),
        expected_validation=expected_validation,
        expected_provenance=expected_provenance,
        smoke=case["name"] == SMOKE,
    )
    route_ok, route_errors = route_inventory(case)
    paired_key = f"{case['name']}_paired_train_data"
    if expected_train is None:
        checks.pop(paired_key)
        checks[f"{case['name']}_train_evidence_complete"] = set(
            case["train_digests"]
        ) == set(range(8))
    else:
        checks[f"{case['name']}_paired_train_data_equal_sampler_cardinality"] = (
            checks.pop(paired_key)
        )
    checks[f"{case['name']}_policy_command"] = policy_command_check(case)
    checks[f"{case['name']}_exact_rank_route_inventory"] = route_ok
    return checks, route_errors


def metadata_checks(meta: dict[str, str]) -> dict[str, bool]:
    expected = {
        "schema": "1",
        "branch": "dev/mxfp4",
        "lumen_commit": LUMEN_COMMIT,
        "aiter_commit": AITER_COMMIT,
        "formal_order": ",".join(ARMS),
        "phase1_order": ",".join((SMOKE, *PHASE1_ARMS)),
        "formal_steps": "50",
        "train_samples": "6400",
        "timing_window": "11-50",
        "precision_gate_delta_nll": "0.01",
        "speed_gate_min_ratio": "1.003",
        "speed_gate_min_wins": "28",
        "bootstrap_block_length": "4",
        "bootstrap_resamples": "100000",
        "replicate_drift_gate_pct": "3.0",
        "candidate_stack": "packed_qkv+split_swiglu",
        "lm_head_precision": "bf16",
        "git_exclude_scope": ".codex/",
        "source_integrity": "exact_manifest+full_tree_without_codex_runtime_logs",
        "shape_count_contract": "whole_smoke_totals",
        "train_pairing_scope": "formal_arms_equal_train_samples",
        "kfd_identity_policy": "fail_closed_complete_identity",
    }
    return {
        "metadata_protocol": all(
            meta.get(key) == value for key, value in expected.items()
        ),
        "metadata_workload_hash": re.fullmatch(
            r"[0-9a-f]{64}", meta.get("workload_sha256", "")
        )
        is not None,
        "metadata_samples_equal_steps_times_gbs": TRAIN_SAMPLES == FORMAL_STEPS * 128,
    }


def harness_checks(meta: dict[str, str]) -> dict[str, bool]:
    paths = {
        "driver_sha256": ROOT / "run_projection_guard.sh",
        "analyzer_sha256": ROOT / "analyze_projection_guard.py",
        "self_test_sha256": ROOT / "test_analyze_projection_guard.py",
        "protocol_sha256": ROOT / "protocol.md",
        "entry_sha256": ENTRY,
        "git_excludes_sha256": EXCLUDES,
        "shared_runner_sha256": SHARED_RUNNER,
        "base_analyzer_sha256": BASE_PATH,
        "train_entry_sha256": TRAIN_ENTRY,
    }
    checks = {
        key.removesuffix("_sha256") + "_frozen": path.is_file()
        and meta.get(key) == base.sha256(path)
        for key, path in paths.items()
    }
    checks.update(
        {
            "shared_runner_expected": meta.get("shared_runner_sha256")
            == EXPECTED_SHARED_RUNNER_SHA,
            "base_analyzer_expected": meta.get("base_analyzer_sha256")
            == EXPECTED_BASE_SHA,
            "hardened_train_entry_expected": meta.get("train_entry_sha256")
            == EXPECTED_TRAIN_ENTRY_SHA,
            "shape_fixture_frozen": meta.get("shape_fixture_sha256")
            == fixture_bundle_sha256(),
            "shape_fixture_expected": meta.get("shape_fixture_sha256")
            == EXPECTED_FIXTURE_SHA,
            "git_excludes_exact": EXCLUDES.read_text(encoding="utf-8") == ".codex/\n",
        }
    )
    return checks


def source_checks(cases: dict[str, dict]) -> dict[str, bool]:
    manifests = {name: source_manifest(case) for name, case in cases.items()}
    reference = manifests[SMOKE]
    return {
        "source_manifests_identical": all(
            value == reference for value in manifests.values()
        ),
        "critical_source_inputs_present": CRITICAL_SOURCES <= set(reference),
        "source_manifest_excludes_codex_runtime_logs": all(
            "/.codex/" not in path for path in reference
        ),
        "source_manifest_paths_current": all(
            Path(path).is_file() and base.sha256(Path(path)) == digest
            for path, digest in reference.items()
        ),
    }


def base_integrity(
    cases: dict[str, dict], *, complete: bool
) -> tuple[dict[str, bool], dict[str, list[str]], dict]:
    meta = base.parse_kv(ROOT / "campaign-meta.txt")
    smoke = cases[SMOKE]
    expected_source = meta["source_bundle_sha256"]
    expected_cache = meta["fresh_autotune_cache_sha256"]
    expected_provenance = {
        key: smoke["meta"].get(key, "") for key in base.COMMON_PROVENANCE_KEYS
    }
    checks: dict[str, bool] = {}
    checks.update(metadata_checks(meta))
    checks.update(harness_checks(meta))
    route_errors: dict[str, list[str]] = {}
    train_references = formal_train_digest_references(cases)
    for name, case in cases.items():
        item, errors = common_case_checks(
            case,
            expected_source=expected_source,
            expected_cache=expected_cache,
            expected_model=smoke["model_hashes"],
            expected_train=train_references.get(name),
            expected_validation=smoke["validation_digests"],
            expected_provenance=expected_provenance,
        )
        checks.update(item)
        route_errors[name] = errors
    checks.update(source_checks(cases))
    cache_path = Path(smoke["meta"]["cache_file"])
    cache_ok, cache_payload = base.cache_integrity(cache_path)
    shape_ok, shape_errors = smoke_shapes(smoke)
    route_errors[SMOKE].extend(shape_errors)
    formal_cases = {name: case for name, case in cases.items() if name != SMOKE}
    formal_command_match = True
    formal_lr_match = True
    if formal_cases:
        formal_reference = next(iter(formal_cases.values()))
        formal_command_match = all(
            normalized_policy_command(case["meta"]["command"])
            == normalized_policy_command(formal_reference["meta"]["command"])
            for case in formal_cases.values()
        )
        formal_lr_match = all(
            [step["lr"] for step in case["steps"]]
            == [step["lr"] for step in formal_reference["steps"]]
            for case in formal_cases.values()
        )
    checks.update(
        {
            "smoke_cache_absent_before": smoke["cache_before"] == "absent"
            and smoke["meta"].get("cache_sha256_before") == "absent",
            "fresh_cache_exact_nine_asm_choices": cache_ok,
            "smoke_exact_rank_shape_inventory": shape_ok,
            "smoke_shape_logging_only": smoke["meta"].get("shape_log_enabled") == "1"
            and all(
                case["meta"].get("shape_log_enabled") == "0"
                for case in formal_cases.values()
            ),
            "cache_frozen_all_replays": all(
                case["cache_before"] == expected_cache
                and case["cache_after"] == expected_cache
                for case in formal_cases.values()
            ),
            "formal_no_online_autotune": all(
                not case["autotune_events"]
                and case["loaded_cache_counts"]
                and set(case["loaded_cache_counts"]) == {9}
                for case in formal_cases.values()
            ),
            "formal_commands_differ_only_by_policy": formal_command_match,
            "formal_lr_sequences_match": formal_lr_match,
            "campaign_meta_matches_smoke": (
                smoke["meta"].get("source_bundle_sha256") == expected_source
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
                )
            ),
        }
    )
    if complete:
        phase1 = base.parse_kv(ROOT / "phase1-complete.txt")
        formal = base.parse_kv(ROOT / "formal-arms-complete.txt")
        checks.update(
            {
                "phase1_exit_zero": (ROOT / "projection-guard-phase1-exit-status.txt")
                .read_text()
                .strip()
                == "0",
                "phase1_sentinel": phase1.get("completed")
                == ",".join((SMOKE, *PHASE1_ARMS))
                and phase1.get("source_bundle_sha256") == expected_source
                and phase1.get("cache_sha256") == expected_cache
                and phase1.get("meta_sha256")
                == base.sha256(ROOT / "campaign-meta.txt"),
                "formal_sentinel": formal.get("completed") == ",".join(ARMS)
                and formal.get("source_bundle_sha256") == expected_source
                and formal.get("cache_sha256") == expected_cache,
            }
        )
        starts = [cases[name]["started_epoch_ns"] for name in EXPECTED_ORDER]
        ends = [cases[name]["postflight_mtime_ns"] for name in EXPECTED_ORDER]
        checks["actual_start_order"] = all(
            a < b for a, b in zip(starts[:-1], starts[1:], strict=True)
        )
        checks["no_case_overlap"] = all(
            a <= b for a, b in zip(ends[:-1], starts[1:], strict=True)
        )
    return checks, route_errors, cache_payload


def select_policy(
    *,
    integrity: bool,
    drift: bool,
    performance: dict[str, bool],
    precision_b: bool,
    precision_c: bool,
) -> tuple[str | None, dict[str, bool]]:
    b_chain = integrity and drift and performance["a_to_b"] and precision_b
    c_chain = (
        b_chain and performance["b_to_c"] and performance["a_to_c"] and precision_c
    )
    selected = None if not integrity else ("C" if c_chain else "B" if b_chain else "A")
    return selected, {
        "b_complete_chain_pass": b_chain,
        "c_complete_chain_pass": c_chain,
    }


def validate_subset(names: tuple[str, ...], mode: str) -> dict:
    cases = {name: parse_case(name) for name in names}
    checks, route_errors, cache = base_integrity(cases, complete=False)
    return {
        "mode": mode,
        "passed": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "route_errors": route_errors,
        "cache_schema": cache.get("schema"),
        "cache_arch": cache.get("arch"),
        "cases_read": list(names),
    }


def build_report() -> dict:
    cases = {name: parse_case(name) for name in EXPECTED_ORDER}
    checks, route_errors, cache = base_integrity(cases, complete=True)
    formal = {name: cases[name] for name in ARMS}
    series = {key: base.midpoint_series(formal, names) for key, names in GROUPS.items()}
    comparisons = {
        "a_to_b": base.compare_series(series["a"], series["b"]),
        "b_to_c": base.compare_series(series["b"], series["c"]),
        "a_to_c": base.compare_series(series["a"], series["c"]),
    }
    validation = {
        "a_nll": statistics.fmean(
            formal[name]["validation_nll"] for name in GROUPS["a"]
        ),
        "b_nll": statistics.fmean(
            formal[name]["validation_nll"] for name in GROUPS["b"]
        ),
        "c_nll": formal[GROUPS["c"][0]]["validation_nll"],
    }
    validation["b_delta_nll_vs_a"] = validation["b_nll"] - validation["a_nll"]
    validation["c_delta_nll_vs_a"] = validation["c_nll"] - validation["a_nll"]
    validation["c_delta_nll_vs_b"] = validation["c_nll"] - validation["b_nll"]
    validation["b_precision_pass"] = validation["b_delta_nll_vs_a"] <= PRECISION_GATE
    validation["c_precision_pass"] = validation["c_delta_nll_vs_a"] <= PRECISION_GATE
    drift = {
        "a": base.replicate_drift(formal, GROUPS["a"]),
        "b": base.replicate_drift(formal, GROUPS["b"]),
    }
    drift_pass = all(item["pass"] for item in drift.values())
    integrity = all(checks.values())
    selected, chain = select_policy(
        integrity=integrity,
        drift=drift_pass,
        performance={
            key: value["performance_pass"] for key, value in comparisons.items()
        },
        precision_b=bool(validation["b_precision_pass"]),
        precision_c=bool(validation["c_precision_pass"]),
    )
    if selected is None:
        decision = "analysis integrity failed; no trustworthy policy selection"
    elif selected == "C":
        decision = "advance down_proj-only guard C to longer quality validation"
    elif selected == "B":
        decision = "retain o_proj+down_proj guard B; C failed its complete chain"
    else:
        decision = "retain whole-layer tail1 A; projection guards did not pass"
    return {
        "protocol": {
            "formal_order": list(ARMS),
            "formal_steps": FORMAL_STEPS,
            "timing_window": [11, 50],
            "speed_gate": {
                "mean": MIN_SPEEDUP,
                "median": MIN_SPEEDUP,
                "wins": MIN_WINS,
                "ci_lower_gt": 1.0,
            },
            "bootstrap": {
                "method": "paired circular moving-block",
                "block_length": BLOCK_LENGTH,
                "resamples": BOOTSTRAP_RESAMPLES,
            },
            "precision_gate_delta_nll_vs_a": PRECISION_GATE,
            "lm_head_precision": "bf16",
        },
        "integrity": {
            "passed": integrity,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "failed_checks": [key for key, value in checks.items() if not value],
            "route_errors": route_errors,
            "cache_schema": cache.get("schema"),
            "cache_arch": cache.get("arch"),
        },
        "series": {key: base.describe(value) for key, value in series.items()},
        "comparisons": comparisons,
        "validation": validation,
        "replicate_drift": drift,
        "drift_gate_pass": drift_pass,
        "selection_chain": chain,
        "selected_policy": selected,
        "decision": decision,
    }


def markdown_report(report: dict) -> str:
    integrity = report["integrity"]
    lines = [
        "# Fresh MXFP4 projection-guard result",
        "",
        f"Integrity: **{'PASS' if integrity['passed'] else 'FAIL'}** "
        f"({integrity['checks_passed']}/{integrity['checks_total']}).",
        "",
        "| Comparison | Mean | Median | Saving ms | Wins | 95% CI | Gate |",
        "|:--|--:|--:|--:|--:|--:|:--:|",
    ]
    labels = {"a_to_b": "A → B", "b_to_c": "B → C", "a_to_c": "A → C"}
    for key in ("a_to_b", "b_to_c", "a_to_c"):
        item = report["comparisons"][key]
        ci = item["speedup_ci95"]
        lines.append(
            f"| {labels[key]} | {item['speedup_mean']:.6f}x | {item['speedup_median']:.6f}x | "
            f"{item['saving_mean_ms']:.3f} | {item['paired_wins']}/40 | "
            f"[{ci[0]:.6f}x, {ci[1]:.6f}x] | {'PASS' if item['performance_pass'] else 'FAIL'} |"
        )
    value = report["validation"]
    lines.extend(
        [
            "",
            "## Validation NLL",
            "",
            f"- A: {value['a_nll']:.5f}",
            f"- B: {value['b_nll']:.5f}; delta vs A {value['b_delta_nll_vs_a']:+.5f}; "
            f"{'PASS' if value['b_precision_pass'] else 'FAIL'}",
            f"- C: {value['c_nll']:.5f}; delta vs A {value['c_delta_nll_vs_a']:+.5f}; "
            f"{'PASS' if value['c_precision_pass'] else 'FAIL'}",
            f"- Non-gating C minus B delta: {value['c_delta_nll_vs_b']:+.5f}",
            "",
            "## Decision",
            "",
            report["decision"] + ".",
            f"Selected policy: {report['selected_policy']}.",
        ]
    )
    if integrity["failed_checks"]:
        lines.extend(["", "## Failed integrity checks", ""])
        lines.extend(f"- {name}" for name in integrity["failed_checks"])
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> None:
    json_path = ROOT / "projection_guard_analysis.json"
    md_path = ROOT / "projection_guard_analysis.md"
    json_tmp = json_path.with_suffix(".json.tmp")
    md_tmp = md_path.with_suffix(".md.tmp")
    json_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_tmp.write_text(markdown_report(report), encoding="utf-8")
    json_tmp.replace(json_path)
    md_tmp.replace(md_path)


def self_test() -> None:
    assert policy_projection_args(["x"]) == ()
    assert policy_projection_args(
        ["x", "--mxfp4-last-layer-bf16-projections", "o_proj", "down_proj"]
    ) == ("o_proj", "down_proj")
    control = [100.0 + index / 100 for index in range(40)]
    candidate = [98.0 + index / 100 for index in range(40)]
    first = base.bootstrap(control, candidate, resamples=200)
    second = base.bootstrap(control, candidate, resamples=200)
    assert first == second and first["speedup_ci95"][0] > 1.0
    fixture_ok, fixture_errors = shape_inventory(FIXTURE_DIR, "b")
    assert fixture_ok, fixture_errors
    assert not shape_inventory(FIXTURE_DIR, "a")[0]
    assert not shape_inventory(FIXTURE_DIR, "c")[0]
    selected, chain = select_policy(
        integrity=True,
        drift=True,
        performance={"a_to_b": True, "b_to_c": True, "a_to_c": True},
        precision_b=True,
        precision_c=True,
    )
    assert selected == "C" and chain["c_complete_chain_pass"]
    selected, _chain = select_policy(
        integrity=True,
        drift=True,
        performance={"a_to_b": True, "b_to_c": False, "a_to_c": True},
        precision_b=True,
        precision_c=True,
    )
    assert selected == "B"
    print("analyze_projection_guard self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-smoke", action="store_true")
    modes.add_argument("--validate-phase1", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.validate_smoke:
        report = validate_subset((SMOKE,), "validate-smoke")
    elif args.validate_phase1:
        report = validate_subset((SMOKE, *PHASE1_ARMS), "validate-phase1")
    else:
        try:
            report = build_report()
            write_report(report)
        except Exception as error:
            report = {
                "integrity": {
                    "passed": False,
                    "failed_checks": ["analysis_parse_and_artifact_read"],
                },
                "selected_policy": None,
                "decision": "analysis failed closed before a trustworthy decision",
                "error": f"{type(error).__name__}: {error}",
            }
            (ROOT / "projection_guard_analysis.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(report["error"], file=sys.stderr)
            raise SystemExit(2) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("passed", report.get("integrity", {}).get("passed", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

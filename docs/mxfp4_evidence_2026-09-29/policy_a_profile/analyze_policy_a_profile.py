#!/usr/bin/env python3
"""Streaming, fail-closed analysis for the fresh BF16/MXFP4 Policy A profile."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import ijson as _ijson
except ImportError:  # pragma: no cover - explicitly tested by forcing fallback
    _ijson = None


DEFAULT_ROOT = Path(__file__).resolve().parent
ARMS = (
    "smoke_mxfp4_policy_a",
    "replay_mxfp4_policy_a",
    "profile_bf16",
    "profile_mxfp4_policy_a",
)
FORMAL_ARMS = ("profile_bf16", "profile_mxfp4_policy_a")
PROFILE_STEPS = (7, 8)
PHYSICAL_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
EXPECTED_PROGRESS = (
    "preflight_complete",
    "smoke_mxfp4_policy_a_complete",
    "replay_mxfp4_policy_a_complete",
    "profile_bf16_complete",
    "profile_mxfp4_policy_a_complete",
)
EXPECTED_STAGE_STATUS = (
    "preflight",
    "smoke_mxfp4_policy_a",
    "replay_mxfp4_policy_a",
    "profile_bf16",
    "profile_mxfp4_policy_a",
    "postflight",
)
EXPECTED_META = {
    "schema": "1",
    "branch": "dev/mxfp4",
    "arms": ",".join(ARMS),
    "formal_arms": ",".join(FORMAL_ARMS),
    "formal_steps": "8",
    "profile_steps": "7,8",
    "profile_rank": "0",
    "profile_device": "0",
    "nproc": "8",
    "micro_batch": "2",
    "global_batch": "128",
    "gradient_accumulation": "8",
    "sequence_length": "8192",
    "tail_bf16": "1",
    "expected_quantized_linears": "245",
    "expected_bf16_skipped_linears": "8",
    "expected_packed_qkv_layers": "35",
    "expected_split_swiglu_layers": "35",
    "lm_head_precision": "bf16",
    "eval_batches": "16",
    "val_samples": "256",
    "seed": "1234",
}
COMMON_RUN_META = {
    "global_batch": "128",
    "gradient_accumulation": "8",
    "tokens_per_update": "1048576",
    "tail_bf16": "1",
    "lm_head_precision": "bf16",
    "eval_batches": "16",
    "val_samples": "256",
    "seed": "1234",
    "numa_cpu_node": "0",
    "numa_memory_node": "0",
}
ZERO_METRICS = {
    "calls_total": 0,
    "calls_per_step": 0.0,
    "raw_total_ms": 0.0,
    "raw_per_step_ms": 0.0,
    "union_total_ms": 0.0,
    "union_per_step_ms": 0.0,
}
STEP_RE = re.compile(
    r"step (\d+)/(\d+) \| loss (\S+) \| grad_norm (\S+) "
    r"\| lr (\S+) \| step_time_ms (\S+) \| peak_mem_gib (\S+)"
)
PAIRING_RE = re.compile(
    r"PAIRING_EVIDENCE rank=(\d+) step=(\d+) microbatches=(\d+) "
    r"input_ids_labels_sha256=([0-9a-f]{64})"
)
VALIDATION_RE = re.compile(
    r"VALIDATION_EVIDENCE rank=(\d+) batches=(\d+) "
    r"input_ids_labels_sha256=([0-9a-f]{64})"
)
SHAPE_FLUSH_RE = re.compile(
    r"(?:INFO:)?lumen\.ops\.quantize\.mxfp4_autotune:MXFP4 shape log: wrote "
    r"\d+ distinct shapes to .+/mxfp4-shapes-rank0\.csv$"
)
MATRIX_RE = re.compile(r"^M=(\d+),N=(\d+),K=(\d+)$")


@dataclasses.dataclass
class Producer:
    name: str
    input_dims: Any
    input_strides: Any
    input_types: Any
    start_us: float = 0.0
    end_us: float = 0.0
    pid: Any = None
    tid: Any = None
    scope: str | None = None


@dataclasses.dataclass
class GPUEvent:
    start_us: float
    end_us: float
    trace_category: str
    name: str
    external_id: int | None
    correlation: int | None
    stream: Any
    producer: Producer | None = None
    category: str = "other"
    broad_category: str = "other"
    shape: str = "<unresolved>"


@dataclasses.dataclass(frozen=True)
class Scope:
    start_us: float
    end_us: float
    pid: Any
    tid: Any
    kind: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def first_token(path: Path) -> str:
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    return tokens[0] if tokens else ""


def interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((left, right) for left, right in intervals if right > left)
    if not ordered:
        return 0.0
    left, right = ordered[0]
    total = 0.0
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def summarize_intervals(
    intervals: Iterable[tuple[float, float]], steps: int = 2
) -> dict[str, float | int]:
    rows = list(intervals)
    raw_us = sum(right - left for left, right in rows if right > left)
    union_us = interval_union(rows)
    return {
        "calls_total": len(rows),
        "calls_per_step": len(rows) / steps,
        "raw_total_ms": raw_us / 1000.0,
        "raw_per_step_ms": raw_us / (1000.0 * steps),
        "union_total_ms": union_us / 1000.0,
        "union_per_step_ms": union_us / (1000.0 * steps),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def compact_dims(dims: Any) -> str:
    if dims is None:
        return "<unresolved>"
    encoded = json.dumps(dims, separators=(",", ":"))
    return encoded if len(encoded) <= 320 else encoded[:317] + "..."


def flattened_shapes(value: Any) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []

    def visit(item: Any) -> None:
        if not isinstance(item, list):
            return
        if item and all(isinstance(part, int) for part in item):
            shapes.append(tuple(item))
        else:
            for part in item:
                visit(part)

    visit(value)
    return shapes


def matrix_shape(producer: Producer | None) -> str | None:
    if producer is None or producer.input_dims is None:
        return None
    dims = producer.input_dims
    name = producer.name.lower()
    try:
        if (
            producer.name in {"aten::mm", "aten::matmul", "aten::bmm"}
            and len(dims) >= 2
        ):
            left, right = dims[0], dims[1]
            if len(left) == 2 and len(right) == 2:
                return f"M={left[0]},N={right[1]},K={left[1]}"
        if producer.name == "aten::addmm" and len(dims) >= 3:
            left, right = dims[1], dims[2]
            if len(left) == 2 and len(right) == 2:
                return f"M={left[0]},N={right[1]},K={left[1]}"
        if "gemm_a4w4_asm" in name and len(dims) >= 5:
            packed_activation, output = dims[0], dims[4]
            if len(packed_activation) == 2 and len(output) == 2:
                return f"M={output[0]},N={output[1]},K={2 * packed_activation[1]}"
        if "quantizedlinearfunction" in name and len(dims) >= 2:
            activation, weight = dims[0], dims[1]
            if activation and weight and len(weight) == 2:
                return (
                    f"M={math.prod(activation[:-1])},N={weight[0]},K={activation[-1]}"
                )
    except (IndexError, TypeError, ValueError):
        return None
    return None


def shape_label(producer: Producer | None, _category: str | None = None) -> str | None:
    if producer is None:
        return None
    return matrix_shape(producer) or compact_dims(producer.input_dims)


def normalize_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--mode":
            if index + 1 >= len(tokens):
                output.append(token)
                break
            output.extend((token, "<precision>"))
            index += 2
        elif token in {"--mxfp4-pack-qkv", "--mxfp4-fuse-swiglu"}:
            index += 1
        else:
            output.append(token)
            index += 1
    return output


def command_contract(
    bf16_command: str, policy_a_command: str
) -> tuple[bool, dict[str, Any]]:
    bf16 = shlex.split(bf16_command)
    policy_a = shlex.split(policy_a_command)

    def values(tokens: list[str], flag: str) -> list[str]:
        return [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == flag]

    evidence = {
        "normalized_equal": normalize_command(bf16_command)
        == normalize_command(policy_a_command),
        "bf16_modes": values(bf16, "--mode"),
        "policy_a_modes": values(policy_a, "--mode"),
        "bf16_tail": values(bf16, "--num-layers-at-end-in-bf16"),
        "policy_a_tail": values(policy_a, "--num-layers-at-end-in-bf16"),
        "bf16_pack_qkv": bf16.count("--mxfp4-pack-qkv"),
        "policy_a_pack_qkv": policy_a.count("--mxfp4-pack-qkv"),
        "bf16_split_swiglu": bf16.count("--mxfp4-fuse-swiglu"),
        "policy_a_split_swiglu": policy_a.count("--mxfp4-fuse-swiglu"),
    }
    passed = (
        evidence["normalized_equal"]
        and evidence["bf16_modes"] == ["bf16"]
        and evidence["policy_a_modes"] == ["mxfp4"]
        and evidence["bf16_tail"] == ["1"]
        and evidence["policy_a_tail"] == ["1"]
        and evidence["bf16_pack_qkv"] == 0
        and evidence["policy_a_pack_qkv"] == 1
        and evidence["bf16_split_swiglu"] == 0
        and evidence["policy_a_split_swiglu"] == 1
    )
    return bool(passed), evidence


def _stdlib_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"{path}: traceEvents array not found")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                position = match.end()
                break
        while True:
            while position >= len(buffer):
                buffer = ""
                position = 0
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise ValueError(f"{path}: unterminated traceEvents array")
                buffer = chunk
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position >= len(buffer):
                continue
            if buffer[position] == "]":
                return
            while True:
                try:
                    event, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        raise ValueError(f"{path}: truncated trace event") from None
                    buffer += chunk
                    continue
                if not isinstance(event, dict):
                    raise ValueError(f"{path}: non-object trace event")
                yield event
                position = end
                if position > 4 * 1024 * 1024:
                    buffer = buffer[position:]
                    position = 0
                break


def iter_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    if _ijson is not None:
        with path.open("rb") as handle:
            yield from _ijson.items(handle, "traceEvents.item")
    else:
        yield from _stdlib_trace_events(path)


def _strip_rank_prefix(line: str) -> str:
    return re.sub(r"^\[rank\d+\]:\s?", "", line)


def _allowed_teardown_block(lines: list[str]) -> bool:
    normalized = "\n".join(_strip_rank_prefix(line) for line in lines)
    required = (
        "Traceback (most recent call last):",
        "weakref.py",
        "_exitfunc",
        "__call__",
        "torch/library.py",
        "_del_library",
        "_clear_torch_ops_cache",
        'qualname.split("::")',
        "ValueError: too many values to unpack (expected 2)",
    )
    positions = [normalized.find(token) for token in required]
    return all(position >= 0 for position in positions) and positions == sorted(
        positions
    )


def analyze_post_completion(
    lines: list[str], completion_index: int, allow_shape_flush: bool
) -> dict[str, Any]:
    tracebacks: list[list[str]] = []
    shape_lines: list[str] = []
    unexpected: list[str] = []
    tail = lines[completion_index + 1 :]
    index = 0
    while index < len(tail):
        line = tail[index]
        if not line.strip():
            index += 1
        elif SHAPE_FLUSH_RE.search(line):
            shape_lines.append(line)
            index += 1
        elif "Traceback (most recent call last):" in line:
            block = [line]
            index += 1
            while index < len(tail):
                block.append(tail[index])
                terminal = (
                    "ValueError: too many values to unpack (expected 2)" in tail[index]
                )
                index += 1
                if terminal:
                    break
            tracebacks.append(block)
        else:
            unexpected.append(line)
            index += 1
    traceback_count_ok = len(tracebacks) in {0, 16}
    tracebacks_allowed = traceback_count_ok and all(
        _allowed_teardown_block(block) for block in tracebacks
    )
    shape_allowed = (allow_shape_flush and len(shape_lines) == 2) or (
        not allow_shape_flush and not shape_lines
    )
    return {
        "valid": tracebacks_allowed and shape_allowed and not unexpected,
        "traceback_count": len(tracebacks),
        "tracebacks_allowlisted": tracebacks_allowed,
        "shape_flush_count": len(shape_lines),
        "shape_flush_allowed": shape_allowed,
        "unexpected_lines": unexpected,
    }


def parse_training_log(
    path: Path, expected_steps: int, *, allow_shape_flush: bool = False
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    steps: list[dict[str, Any]] = []
    step_lines: dict[int, list[int]] = collections.defaultdict(list)
    for line_index, line in enumerate(lines):
        match = STEP_RE.search(line)
        if not match:
            continue
        values = [float(match.group(index)) for index in range(3, 8)]
        step = int(match.group(1))
        steps.append(
            {
                "step": step,
                "max_steps": int(match.group(2)),
                "loss": values[0],
                "grad_norm": values[1],
                "lr": values[2],
                "step_time_ms": values[3],
                "peak_mem_gib": values[4],
            }
        )
        step_lines[step].append(line_index)
    completion = [i for i, line in enumerate(lines) if "Training complete" in line]
    completion_index = completion[0] if completion else len(lines)
    before_completion = "\n".join(lines[:completion_index])
    teardown = (
        analyze_post_completion(lines, completion_index, allow_shape_flush)
        if completion
        else {"valid": False, "traceback_count": 0, "unexpected_lines": []}
    )
    failures = {
        "fallback": bool(
            re.search(
                r"\bfallback\b|falling back|backend failed .* trying next|"
                r"using two compact quantizations|compact quantizer.*fallback",
                before_completion,
                re.I,
            )
        ),
        "oom": bool(re.search(r"out of memory|\boom\b", before_completion, re.I)),
        "nan": bool(re.search(r"\bnan\b", before_completion, re.I)),
        "inf": bool(
            re.search(
                r"(?<![A-Za-z])[-+]?inf(?:inity)?(?![A-Za-z])", before_completion, re.I
            )
        ),
        "skipped_update": bool(
            re.search(r"skipped update|update skipped", before_completion, re.I)
        ),
        "kernel_failure": bool(
            re.search(
                r"kernel failure|kernel launch failed|illegal memory access",
                before_completion,
                re.I,
            )
        ),
        "runtime_error": bool(
            re.search(r"runtimeerror|runtime error", before_completion, re.I)
        ),
        "traceback": "Traceback (most recent call last):" in before_completion,
    }
    batch_rows = PAIRING_RE.findall(text)
    validation_rows = VALIDATION_RE.findall(text)
    validation_losses = [
        float(value) for value in re.findall(r"\|\s*val_loss\s+([-+0-9.eE]+)", text)
    ]
    # StepProfiler emits these concrete rank-0 messages; it does not emit the
    # synthetic ``profiler_start``/``profiler_stop`` strings used by an older
    # harness.  Bind the markers to this arm's exact output path so unrelated
    # log text cannot satisfy the ordering contract.
    profile_path = path.parent / "profile.txt"
    profiler_armed_marker = f"Profiler armed: steps 7-8 -> {profile_path}"
    profiler_written_marker = f"Profiler wrote {profile_path}"
    profiler_start = [
        i for i, line in enumerate(lines) if profiler_armed_marker in line
    ]
    profiler_stop = [
        i for i, line in enumerate(lines) if profiler_written_marker in line
    ]
    profiler_order = (
        len(profiler_start)
        == len(step_lines[7])
        == len(step_lines[8])
        == len(profiler_stop)
        == 1
        and profiler_start[0] < step_lines[7][0] < step_lines[8][0] < profiler_stop[0]
    )
    numeric = [
        value
        for row in steps
        for value in (
            row["loss"],
            row["grad_norm"],
            row["lr"],
            row["step_time_ms"],
            row["peak_mem_gib"],
        )
    ] + validation_losses
    batch_ranks = [int(row[0]) for row in batch_rows]
    validation_ranks = [int(row[0]) for row in validation_rows]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "steps": steps,
        "step_numbers_exact": [row["step"] for row in steps]
        == list(range(1, expected_steps + 1)),
        "max_steps_exact": all(row["max_steps"] == expected_steps for row in steps),
        "finite": bool(validation_losses)
        and all(math.isfinite(value) for value in numeric),
        "validation_nll": validation_losses[-1] if validation_losses else None,
        "validation_loss_count": len(validation_losses),
        "model_init_hashes": sorted(
            set(re.findall(r"model_init_sha256=([0-9a-f]{64})", text))
        ),
        "batch_evidence": {
            int(rank): {"step": int(step), "microbatches": int(count), "sha256": digest}
            for rank, step, count, digest in batch_rows
        },
        "batch_duplicate_ranks": sorted(
            rank
            for rank, count in collections.Counter(batch_ranks).items()
            if count != 1
        ),
        "validation_evidence": {
            int(rank): {"batches": int(count), "sha256": digest}
            for rank, count, digest in validation_rows
        },
        "validation_duplicate_ranks": sorted(
            rank
            for rank, count in collections.Counter(validation_ranks).items()
            if count != 1
        ),
        "training_complete_exact": len(completion) == 1
        and text.count(f"Training complete after {expected_steps} steps.") == 1,
        "failure_markers_before_completion": failures,
        "post_completion": teardown,
        "profile_marker_order_exact": profiler_order,
        "profile_marker_counts": {
            "start": len(profiler_start),
            "stop": len(profiler_stop),
        },
        "quantization_summaries": [
            (int(enabled), int(skipped))
            for enabled, skipped in re.findall(
                r"Quantization enabled on (\d+) nn\.Linear layers \([^\n]*bf16_layers_skipped=(\d+)\)",
                text,
            )
        ],
        "packed_qkv_enabled": [
            int(value)
            for value in re.findall(
                r"MXFP4 packed QKV enabled on (\d+) Qwen3 attention layers", text
            )
        ],
        "split_swiglu_enabled": [
            int(value)
            for value in re.findall(
                r"MXFP4 split SwiGLU enabled on (\d+) Qwen3 MLPs", text
            )
        ],
        "qkv_warning_count": text.count(
            "all Q/K/V projections must already be quantized"
        ),
        "swiglu_warning_count": text.count(
            "both projections must already be quantized"
        ),
        "loaded_decisions": [
            int(value)
            for value in re.findall(
                r"MXFP4 autotune: loaded (\d+) cached decisions", text
            )
        ],
        "online_autotune_lines": re.findall(r"MXFP4 autotune \d+x\d+x\d+:.*", text),
    }


def scope_kind(name: str) -> str | None:
    return {
        "quantizedlinearfunction": "quantized_linear_forward",
        "quantizedlinearfunctionbackward": "quantized_linear_backward",
        "mxfp4qkvfunction": "packed_qkv_forward",
        "mxfp4qkvfunctionbackward": "packed_qkv_backward",
        "splitswiglufunction": "split_swiglu_forward",
        "splitswiglufunctionbackward": "split_swiglu_backward",
        "packedswiglufunction": "split_swiglu_forward",
        "packedswiglufunctionbackward": "split_swiglu_backward",
    }.get(name.lower().lstrip("_"))


def host_category(name: str, dims: Any) -> str | None:
    kind = scope_kind(name)
    if kind:
        return kind
    lowered = name.lower()
    shapes = flattened_shapes(dims)
    if "parallelcrossentropy" in lowered or "crossentropy" in lowered:
        return (
            "cross_entropy_backward"
            if "backward" in lowered
            else "cross_entropy_forward"
        )
    if (
        "splitwithsizesbackward" in lowered or "split_with_sizes_backward" in lowered
    ) and any(6144 in shape for shape in shapes):
        return "packed_qkv_gradient_assembly"
    if name == "aten::cat" and (
        {4096, 1024} <= {shape[0] for shape in shapes if len(shape) == 2}
        or any(shape[-1:] == (6144,) for shape in shapes)
    ):
        return "packed_qkv_cat"
    if "split" in lowered and (
        (6144, 4096) in shapes or any(6144 in shape for shape in shapes)
    ):
        return "packed_qkv_split"
    if name == "aten::contiguous" and any(
        shape in {(6144, 4096), (4096, 4096), (1024, 4096)} for shape in shapes
    ):
        return "packed_qkv_contiguous"
    if "rmsnorm" in lowered or "rms_norm" in lowered:
        return (
            "q_rmsnorm"
            if (524288, 128) in shapes
            else "k_rmsnorm"
            if (131072, 128) in shapes
            else "norm_other"
        )
    if "rope" in lowered or "rotary" in lowered:
        return (
            "q_rope"
            if (2, 32, 8192, 128) in shapes
            else "k_rope"
            if (2, 8, 8192, 128) in shapes
            else "rope_other"
        )
    return None


def collective_op(event: GPUEvent) -> str:
    text = f"{event.name} {event.producer.name if event.producer else ''}".lower()
    if "all_gather" in text or "allgather" in text:
        return "all_gather"
    if "reduce_scatter" in text or "reducescatter" in text:
        return "reduce_scatter"
    if "all_reduce" in text or "allreduce" in text:
        return "all_reduce"
    return "broadcast" if "broadcast" in text else "other"


def is_collective(event: GPUEvent) -> bool:
    text = f"{event.name} {event.producer.name if event.producer else ''}".lower()
    return any(token in text for token in ("nccl", "rccl", "record_param_comms"))


def is_a4w4(event: GPUEvent) -> bool:
    text = f"{event.name} {event.producer.name if event.producer else ''}".lower()
    return any(
        token in text
        for token in ("f4gemm_bf16_per1x32fp4", "gemm_afp4wfp4", "gemm_a4w4")
    )


def _matrix_values(shape: str) -> tuple[int, int, int] | None:
    match = MATRIX_RE.fullmatch(shape)
    return tuple(int(match.group(i)) for i in (1, 2, 3)) if match else None


def _a4w4_phase(event: GPUEvent, token_rows: int | None) -> str:
    scope = event.producer.scope if event.producer else None
    if scope == "packed_qkv_forward":
        return "packed_qkv_forward"
    values = _matrix_values(event.shape)
    is_wgrad = bool(
        values and token_rows and values[2] == token_rows and values[0] != token_rows
    )
    if scope == "packed_qkv_backward":
        return "packed_qkv_wgrad" if is_wgrad else "packed_qkv_dgrad"
    if scope == "quantized_linear_forward":
        return "forward"
    if scope == "quantized_linear_backward":
        return "wgrad" if is_wgrad else "dgrad"
    return "wgrad" if is_wgrad else "unresolved_phase"


def classify_event(
    event: GPUEvent, *, precision: str = "mxfp4", token_rows: int | None = None
) -> str:
    name = event.name.lower()
    producer = event.producer.name.lower() if event.producer else ""
    scope = event.producer.scope if event.producer else None
    text = f"{name} {producer}"
    shapes = flattened_shapes(event.producer.input_dims) if event.producer else []
    if is_collective(event):
        return f"collective_{collective_op(event)}"
    if event.trace_category in {"gpu_memcpy", "gpu_memset"}:
        return (
            "packed_qkv_split_cat_copy"
            if scope and scope.startswith("packed_qkv")
            else "copy_memset"
        )
    if is_a4w4(event):
        return f"a4w4_{_a4w4_phase(event, token_rows)}"
    if "dual_layout_quant_mxfp4" in text:
        return (
            "mxfp4_dual_layout_gradient_quant"
            if scope and "backward" in scope
            else "mxfp4_dual_layout_activation_quant"
        )
    if "transpose_packed_fp4" in text:
        return "mxfp4_packed_transpose"
    if "swizzle_expanded_2d_scale" in text or "swizzle_mxfp4_scale" in text:
        return "mxfp4_scale_swizzle"
    if any(
        token in text for token in ("quantize_weight", "shuffle_weight", "weight_quant")
    ):
        return "mxfp4_weight_quant"
    if any(
        token in text
        for token in (
            "convert_to_mxfp4",
            "dynamic_mxfp4_quant",
            "dequant_hadamard_quant_mxfp4",
            "quant_mxfp4",
            "dequant_mxfp4",
        )
    ):
        return "mxfp4_conversion_helper"
    if (
        scope
        and scope.startswith("packed_qkv")
        and any(token in text for token in ("cat", "split", "copy", "contiguous"))
    ):
        return "packed_qkv_split_cat_copy"
    if any(token in text for token in ("fmha_bwd", "flash_attn_bwd")):
        return "attention_backward"
    if any(token in text for token in ("fmha_fwd", "flash_attn_fwd")):
        return "attention_forward"
    if "rmsnorm" in text or "rms_norm" in text:
        return (
            "q_rmsnorm"
            if (524288, 128) in shapes
            else "k_rmsnorm"
            if (131072, 128) in shapes
            else "norm_other"
        )
    if "rope" in text or "rotary" in text:
        return (
            "q_rope"
            if (2, 32, 8192, 128) in shapes
            else "k_rope"
            if (2, 8, 8192, 128) in shapes
            else "rope_other"
        )
    if "swiglu" in text or "silu" in text:
        return (
            "split_swiglu_backward"
            if scope == "split_swiglu_backward" or "bwd" in text or "backward" in text
            else "split_swiglu_forward"
        )
    if "cross_entropy" in text or "crossentropy" in text or "online_softmax" in text:
        return "cross_entropy"
    if "adam" in name or "optimizer" in producer or "_foreach_" in producer:
        return "optimizer"
    if producer in {
        "aten::mm",
        "aten::matmul",
        "aten::bmm",
        "aten::addmm",
    } or name.startswith("cijk_"):
        if precision == "bf16":
            return "bf16_gemm"
        values = _matrix_values(event.shape)
        return (
            "bf16_lm_head"
            if values and 151936 in values
            else "bf16_protected_tail"
            if values
            else "bf16_gemm_other"
        )
    if any(
        token in name
        for token in ("add_kernel", "mul_kernel", "gelu", "residual", "elementwise")
    ) or producer in {"aten::mul", "aten::add", "aten::add_", "aten::sum"}:
        return "activation_residual"
    if any(token in name for token in ("copy", "memset", "fillfunctor", "zero")):
        return "copy_memset"
    return "other"


def broad_category(detail: str) -> str:
    if detail.startswith("a4w4_"):
        return "a4w4_gemm"
    if detail.startswith("bf16_"):
        return "bf16_gemm"
    if detail.startswith("mxfp4_"):
        return "mxfp4_quant_layout"
    if detail.startswith("collective_"):
        return "collectives"
    if detail.startswith("attention_"):
        return "attention"
    if detail in {"q_rmsnorm", "k_rmsnorm", "norm_other"}:
        return "norm"
    if detail in {"q_rope", "k_rope", "rope_other"}:
        return "rope"
    if detail.startswith("split_swiglu") or detail == "activation_residual":
        return "activation_residual"
    return "packed_qkv_layout" if detail == "packed_qkv_split_cat_copy" else detail


def runtime_category(name: str) -> str:
    lowered = name.lower()
    if "pointer" in lowered or "getattribute" in lowered:
        return "pointer_query"
    if "launch" in lowered and ("kernel" in lowered or "graph" in lowered):
        return "kernel_launch"
    if "synchronize" in lowered:
        return "synchronization"
    if "memcpy" in lowered and ("sync" in lowered or "dtoh" in lowered):
        return "blocking_copy"
    return "other_runtime"


def _assign_scopes(producers: dict[int, Producer], scopes: list[Scope]) -> None:
    scopes_by_thread: dict[tuple[Any, Any], list[Scope]] = collections.defaultdict(list)
    producers_by_thread: dict[tuple[Any, Any], list[Producer]] = (
        collections.defaultdict(list)
    )
    for scope in scopes:
        scopes_by_thread[(scope.pid, scope.tid)].append(scope)
    for producer in producers.values():
        producers_by_thread[(producer.pid, producer.tid)].append(producer)
    for thread, thread_producers in producers_by_thread.items():
        timeline: list[tuple[float, int, int, Any]] = []
        for identity, scope in enumerate(scopes_by_thread.get(thread, [])):
            timeline.extend(
                (
                    (scope.start_us, 0, identity, scope),
                    (scope.end_us, 2, identity, scope),
                )
            )
        for identity, producer in enumerate(thread_producers):
            timeline.append((producer.start_us, 1, identity, producer))
        timeline.sort(key=lambda row: (row[0], row[1]))
        active: dict[int, Scope] = {}
        for _timestamp, kind, identity, value in timeline:
            if kind == 0:
                active[identity] = value
            elif kind == 2:
                active.pop(identity, None)
            else:
                candidates = [
                    scope
                    for scope in active.values()
                    if scope.start_us <= value.start_us and value.end_us <= scope.end_us
                ]
                if candidates:
                    value.scope = min(
                        candidates, key=lambda scope: scope.end_us - scope.start_us
                    ).kind


def _annotation_summary(
    annotations: dict[str, list[tuple[float, float]]],
    step_spans: dict[int, list[tuple[float, float]]],
) -> dict[str, Any]:
    def count(rows: list[tuple[float, float]], step: int) -> int:
        if len(step_spans[step]) != 1:
            return 0
        left, right = step_spans[step][0]
        return sum(left <= start and end <= right for start, end in rows)

    dataloaders = [
        row
        for name, rows in annotations.items()
        if name.startswith("enumerate(DataLoader)")
        for row in rows
    ]
    groups = {
        "dataloader": dataloaders,
        "root_forward": annotations.get("FSDP::root_pre_forward", []),
        "pre_backward": annotations.get("FSDP::pre_backward", []),
        "root_post_backward": annotations.get("FSDP::root_post_backward_callback", []),
        "optimizer_step": annotations.get("Optimizer.step#AdamW.step", []),
        "zero_grad": annotations.get("Optimizer.zero_grad#AdamW.zero_grad", []),
    }
    result: dict[str, Any] = {
        "step_spans": {str(step): step_spans[step] for step in PROFILE_STEPS}
    }
    for label, rows in groups.items():
        result[f"{label}_total"] = len(rows)
        result[f"{label}_by_step"] = {
            str(step): count(rows, step) for step in PROFILE_STEPS
        }
    return result


def load_cache(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    choices = payload.get("choices") if isinstance(payload.get("choices"), dict) else {}
    profiles = (
        payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
    )
    errors: list[str] = []
    if expected_sha256 and digest != expected_sha256:
        errors.append("sha256_mismatch")
    if payload.get("schema") != 6 or payload.get("arch") != "gfx950":
        errors.append("schema_or_arch")
    if not choices or set(choices) != set(profiles):
        errors.append("choice_profile_keys")
    selected: dict[str, Any] = {}
    for shape, backend in choices.items():
        identity = profiles.get(shape, {}).get("identities", {}).get(backend)
        if backend not in {"asm", "shuffled"} or not isinstance(identity, dict):
            errors.append(f"identity:{shape}:{backend}")
            identity = {}
        if backend == "asm" and any(
            key not in identity
            for key in (
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
        ):
            errors.append(f"asm_identity:{shape}")
        selected[shape] = {"backend": backend, **identity}
    return {
        "path": str(path),
        "sha256": digest,
        "schema": payload.get("schema"),
        "arch": payload.get("arch"),
        "choices": choices,
        "choice_count": len(choices),
        "selected_identities": selected,
        "valid": not errors,
        "errors": errors,
    }


def _cache_identity(cache: dict[str, Any], shape: str) -> dict[str, Any] | None:
    match = MATRIX_RE.fullmatch(shape)
    return (
        cache.get("selected_identities", {}).get(",".join(match.groups()))
        if match
        else None
    )


def scan_trace(
    path: Path,
    log: dict[str, Any],
    *,
    precision: str,
    token_rows: int,
    cache: dict[str, Any],
) -> dict[str, Any]:
    trace_spans: list[tuple[float, float]] = []
    profiler_instants: list[float] = []
    physical: list[GPUEvent] = []
    physical_devices: set[Any] = set()
    physical_pids: set[Any] = set()
    runtime_events: list[tuple[float, float, str, int | None]] = []
    runtime_by_correlation: dict[int, tuple[float, float, str]] = {}
    parsed_events = 0
    for event in iter_trace_events(path):
        parsed_events += 1
        phase = str(event.get("ph", ""))
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        args = event.get("args") or {}
        if category == "Trace" and phase == "X" and name == "PyTorch Profiler (0)":
            start = float(event["ts"])
            trace_spans.append((start, start + float(event["dur"])))
        if phase == "i" and name == "Iteration Start: PyTorch Profiler":
            profiler_instants.append(float(event["ts"]))
        if category in {"cuda_runtime", "hip_runtime"} and phase == "X":
            start = float(event["ts"])
            end = start + float(event.get("dur", 0.0))
            raw_correlation = args.get("correlation")
            correlation = int(raw_correlation) if raw_correlation is not None else None
            runtime_events.append((start, end, name, correlation))
            if correlation is not None:
                runtime_by_correlation[correlation] = (start, end, name)
        if category not in PHYSICAL_CATEGORIES or phase != "X":
            continue
        physical_devices.add(args.get("device"))
        physical_pids.add(event.get("pid"))
        if args.get("device") != 0:
            continue
        duration = float(event.get("dur", 0.0))
        if duration <= 0:
            continue
        raw_external = args.get("External id")
        raw_correlation = args.get("correlation")
        start = float(event["ts"])
        physical.append(
            GPUEvent(
                start,
                start + duration,
                category,
                name,
                int(raw_external) if raw_external is not None else None,
                int(raw_correlation) if raw_correlation is not None else None,
                args.get("stream"),
            )
        )
    if not trace_spans:
        raise AssertionError(f"{path}: no PyTorch Profiler span")
    trace_start, trace_end = trace_spans[0]

    direct_ids = {
        event.external_id for event in physical if event.external_id is not None
    }
    producers: dict[int, Producer] = {}
    duplicate_producer_ids: list[int] = []
    scopes: list[Scope] = []
    annotations: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    step_spans = {step: [] for step in PROFILE_STEPS}
    host_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    host_exact: dict[tuple[str, str, str], list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    for event in iter_trace_events(path):
        if event.get("ph") != "X" or event.get("cat") not in {
            "cpu_op",
            "user_annotation",
        }:
            continue
        start = float(event.get("ts", 0.0))
        end = start + float(event.get("dur", 0.0))
        if end <= trace_start or start >= trace_end:
            continue
        name = str(event.get("name", ""))
        args = event.get("args") or {}
        dims = args.get("Input Dims")
        raw_external = args.get("External id")
        if raw_external is not None and int(raw_external) in direct_ids:
            external_id = int(raw_external)
            producer = Producer(
                name,
                dims,
                args.get("Input Strides"),
                args.get("Input type"),
                start,
                end,
                event.get("pid"),
                event.get("tid"),
            )
            if external_id in producers:
                duplicate_producer_ids.append(external_id)
            else:
                producers[external_id] = producer
        kind = scope_kind(name)
        if kind:
            scopes.append(Scope(start, end, event.get("pid"), event.get("tid"), kind))
        clipped = (max(start, trace_start), min(end, trace_end))
        if event.get("cat") == "user_annotation":
            annotations[name].append(clipped)
            match = re.fullmatch(r"LUMEN_TRAIN_STEP#(7|8)", name)
            if match:
                step_spans[int(match.group(1))].append(clipped)
        selected = host_category(name, dims)
        if selected:
            host_intervals[selected].append(clipped)
            host_exact[(selected, name, compact_dims(dims))].append(clipped)
    _assign_scopes(producers, scopes)

    outside_window = 0
    unresolved_present_external = 0
    clipped_events: list[GPUEvent] = []
    missing_external: dict[tuple[str, str, str], list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    for event in physical:
        if event.start_us < trace_start or event.end_us > trace_end:
            outside_window += 1
        event.start_us = max(event.start_us, trace_start)
        event.end_us = min(event.end_us, trace_end)
        if event.end_us <= event.start_us:
            continue
        if event.external_id is not None:
            event.producer = producers.get(event.external_id)
            unresolved_present_external += event.producer is None
        event.shape = matrix_shape(event.producer) or compact_dims(
            event.producer.input_dims if event.producer else None
        )
        event.category = classify_event(
            event, precision=precision, token_rows=token_rows
        )
        event.broad_category = broad_category(event.category)
        if event.external_id is None:
            runtime_name = (
                runtime_by_correlation[event.correlation][2]
                if event.correlation in runtime_by_correlation
                else "<no-runtime-correlation>"
            )
            missing_external[(event.category, event.name, runtime_name)].append(
                (event.start_us, event.end_us)
            )
        clipped_events.append(event)
    if not clipped_events:
        raise AssertionError(f"{path}: no rank-0/device-0 physical GPU work")

    detail_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(
        list
    )
    broad_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(
        list
    )
    exact_intervals: dict[tuple[str, str, str, str, str], list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    kernel_intervals: dict[tuple[str, str], list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    collective_intervals: dict[str, list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    for event in clipped_events:
        interval = (event.start_us, event.end_us)
        detail_intervals[event.category].append(interval)
        broad_intervals[event.broad_category].append(interval)
        producer = event.producer.name if event.producer else "<unresolved>"
        scope = (
            event.producer.scope
            if event.producer and event.producer.scope
            else "<none>"
        )
        exact_intervals[
            (event.category, event.name, producer, event.shape, scope)
        ].append(interval)
        kernel_intervals[(event.name, event.category)].append(interval)
        if event.broad_category == "collectives":
            collective_intervals[collective_op(event)].append(interval)

    all_intervals = [(event.start_us, event.end_us) for event in clipped_events]
    envelope_start = min(left for left, _ in all_intervals)
    envelope_end = max(right for _, right in all_intervals)
    envelope_us = envelope_end - envelope_start
    busy_us = interval_union(all_intervals)
    raw_us = sum(right - left for left, right in all_intervals)
    detail_metrics = {
        name: summarize_intervals(rows)
        for name, rows in sorted(detail_intervals.items())
    }
    broad_metrics = {
        name: summarize_intervals(rows)
        for name, rows in sorted(broad_intervals.items())
    }

    exact_rows: list[dict[str, Any]] = []
    a4w4_rows: list[dict[str, Any]] = []
    for (category, kernel, producer, shape, scope), rows in exact_intervals.items():
        row: dict[str, Any] = {
            "category": category,
            "broad_category": broad_category(category),
            "kernel": kernel,
            "producer": producer,
            "shape": shape,
            "scope": scope,
            **summarize_intervals(rows),
        }
        identity = (
            _cache_identity(cache, shape) if category.startswith("a4w4_") else None
        )
        if identity:
            row.update(
                selected_backend=identity.get("backend"),
                selected_symbol=identity.get("kernel_name")
                or identity.get("entrypoint"),
                selected_tile=(
                    f"{identity.get('tile_m')}x{identity.get('tile_n')}"
                    if identity.get("tile_m") is not None
                    else None
                ),
                selected_log2_k_split=identity.get("log2_k_split"),
                selected_manifest_sha256=identity.get("manifest_sha256"),
                selected_code_object_sha256=identity.get("code_object_sha256"),
            )
        exact_rows.append(row)
        if category.startswith("a4w4_"):
            a4w4_rows.append(dict(row))
    exact_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["kernel"]))
    a4w4_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["shape"]))
    top_kernels = [
        {"kernel": kernel, "category": category, **summarize_intervals(rows)}
        for (kernel, category), rows in kernel_intervals.items()
    ]
    top_kernels.sort(key=lambda row: (-float(row["raw_total_ms"]), row["kernel"]))
    host_categories = {
        name: summarize_intervals(rows) for name, rows in sorted(host_intervals.items())
    }
    host_exact_rows = [
        {
            "category": category,
            "operator": operator,
            "input_dims": dims,
            **summarize_intervals(rows),
        }
        for (category, operator, dims), rows in host_exact.items()
    ]
    host_exact_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["category"]))

    runtime_by_api: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    runtime_by_kind: dict[str, list[tuple[float, float]]] = collections.defaultdict(
        list
    )
    launches: list[tuple[float, float, int | None]] = []
    for start, end, name, correlation in runtime_events:
        left, right = max(start, trace_start), min(end, trace_end)
        if right <= left:
            continue
        runtime_by_api[name].append((left, right))
        kind = runtime_category(name)
        runtime_by_kind[kind].append((left, right))
        if kind == "kernel_launch":
            launches.append((left, right, correlation))
    runtime_api_rows = [
        {"api": name, **summarize_intervals(rows)}
        for name, rows in runtime_by_api.items()
    ]
    runtime_api_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["api"]))
    runtime_categories = {
        name: summarize_intervals(rows)
        for name, rows in sorted(runtime_by_kind.items())
    }
    ordered_launches = sorted((left, right) for left, right, _ in launches)
    launch_gaps: list[float] = []
    if ordered_launches:
        previous_end = ordered_launches[0][1]
        for left, right in ordered_launches[1:]:
            if left > previous_end:
                launch_gaps.append(left - previous_end)
            previous_end = max(previous_end, right)
    queue_gaps: list[float] = []
    negative_queue_gaps = 0
    for event in clipped_events:
        if event.correlation is None or event.correlation not in runtime_by_correlation:
            continue
        gap = event.start_us - runtime_by_correlation[event.correlation][1]
        if gap >= 0:
            queue_gaps.append(gap)
        else:
            negative_queue_gaps += 1

    collective_all = [
        interval for rows in collective_intervals.values() for interval in rows
    ]
    noncollective = [
        interval
        for category, rows in broad_intervals.items()
        if category != "collectives"
        for interval in rows
    ]
    collective_union_us = interval_union(collective_all)
    collective_exposed_us = max(
        0.0,
        interval_union(collective_all + noncollective) - interval_union(noncollective),
    )
    collective_by_op: dict[str, Any] = {}
    for operation, rows in sorted(collective_intervals.items()):
        others = [
            interval
            for name, other_rows in collective_intervals.items()
            if name != operation
            for interval in other_rows
        ] + noncollective
        exposed_us = max(0.0, interval_union(rows + others) - interval_union(others))
        summary = summarize_intervals(rows)
        summary.update(
            exposed_total_ms=exposed_us / 1000.0,
            exposed_per_step_ms=exposed_us / 2000.0,
            overlapped_total_ms=summary["union_total_ms"] - exposed_us / 1000.0,
            overlapped_per_step_ms=summary["union_per_step_ms"] - exposed_us / 2000.0,
        )
        collective_by_op[operation] = summary

    annotations_summary = _annotation_summary(annotations, step_spans)
    log_by_step = {row["step"]: row["step_time_ms"] for row in log["steps"]}
    log_span_ms = sum(log_by_step.get(step, math.nan) for step in PROFILE_STEPS)
    trace_span_ms = (trace_end - trace_start) / 1000.0
    trace_log_error_ms = abs(trace_span_ms - log_span_ms)
    step_order = (
        all(len(step_spans[step]) == 1 for step in PROFILE_STEPS)
        and step_spans[7][0][0]
        < step_spans[7][0][1]
        <= step_spans[8][0][0]
        < step_spans[8][0][1]
    )
    profiler_order = (
        len(trace_spans) == len(profiler_instants) == 1
        and step_order
        and abs(profiler_instants[0] - trace_start) < 0.001
        and step_spans[8][0][1] <= trace_end
    )
    microbatch_contract = (
        all(
            annotations_summary[f"{name}_total"] == 16
            for name in (
                "dataloader",
                "root_forward",
                "pre_backward",
                "root_post_backward",
            )
        )
        and all(
            annotations_summary[f"{name}_by_step"] == {"7": 8, "8": 8}
            for name in (
                "dataloader",
                "root_forward",
                "pre_backward",
                "root_post_backward",
            )
        )
        and annotations_summary["optimizer_step_by_step"] == {"7": 1, "8": 1}
        and annotations_summary["zero_grad_by_step"] == {"7": 1, "8": 1}
    )
    contract_checks = {
        "one_profiler_span": len(trace_spans) == 1,
        "one_profiler_start_instant": len(profiler_instants) == 1,
        "exact_step_annotations": all(
            len(step_spans[step]) == 1 for step in PROFILE_STEPS
        ),
        "profiler_start_step7_step8_stop_order": profiler_order,
        "trace_log_span_within_3ms": math.isfinite(trace_log_error_ms)
        and trace_log_error_ms <= 3.0,
        "rank0_device0_physical_only": physical_devices == {0} and physical_pids == {0},
        "physical_events_inside_profiler_span": outside_window == 0,
        "microbatch_forward_backward_exact": microbatch_contract,
        "producer_external_ids_unique": not duplicate_producer_ids,
    }
    missing_rows = [
        {
            "category": category,
            "kernel": kernel,
            "runtime_resolution": runtime,
            **summarize_intervals(rows),
        }
        for (category, kernel, runtime), rows in missing_external.items()
    ]
    missing_rows.sort(key=lambda row: (-int(row["calls_total"]), row["kernel"]))
    joint = broad_intervals.get("a4w4_gemm", []) + broad_intervals.get(
        "mxfp4_quant_layout", []
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "stream_parser": "ijson" if _ijson is not None else "stdlib-jsondecoder",
        "parsed_to_eof": True,
        "trace_event_count": parsed_events,
        "contract": {
            "all_pass": all(contract_checks.values()),
            "checks": contract_checks,
            "evidence": {
                "trace_span_count": len(trace_spans),
                "profiler_instants": profiler_instants,
                "physical_devices": sorted(str(value) for value in physical_devices),
                "physical_pids": sorted(str(value) for value in physical_pids),
                "outside_window_count": outside_window,
                "duplicate_producer_ids": sorted(set(duplicate_producer_ids)),
                "trace_log_span_error_ms": trace_log_error_ms,
                "annotations": annotations_summary,
            },
        },
        "window": {
            "profiler_start_us": trace_start,
            "profiler_end_us": trace_end,
            "profiler_span_total_ms": trace_span_ms,
            "profiler_span_per_step_ms": trace_span_ms / 2.0,
            "log_steps_7_8_total_ms": log_span_ms,
            "trace_log_span_error_ms": trace_log_error_ms,
            "gpu_first_event_us": envelope_start,
            "gpu_last_event_us": envelope_end,
            "gpu_envelope_total_ms": envelope_us / 1000.0,
            "gpu_envelope_per_step_ms": envelope_us / 2000.0,
            "gpu_raw_duration_sum_total_ms": raw_us / 1000.0,
            "gpu_raw_duration_sum_per_step_ms": raw_us / 2000.0,
            "gpu_busy_union_total_ms": busy_us / 1000.0,
            "gpu_busy_union_per_step_ms": busy_us / 2000.0,
            "gpu_idle_inside_envelope_total_ms": (envelope_us - busy_us) / 1000.0,
            "gpu_idle_inside_envelope_per_step_ms": (envelope_us - busy_us) / 2000.0,
            "physical_events_total": len(clipped_events),
        },
        "annotations": annotations_summary,
        "detail_categories": detail_metrics,
        "broad_categories": broad_metrics,
        "top_kernels": top_kernels[:80],
        "exact_gpu_rows": exact_rows,
        "a4w4_exact_shapes": a4w4_rows,
        "joint_unions": {
            "a4w4_plus_quant_layout": {
                **summarize_intervals(joint),
                "interpretation": "Overlap-safe impossible-zero-cost ceiling, not predicted saving.",
            }
        },
        "collectives": {
            "overall": {
                **summarize_intervals(collective_all),
                "exposed_total_ms": collective_exposed_us / 1000.0,
                "exposed_per_step_ms": collective_exposed_us / 2000.0,
                "overlapped_total_ms": (collective_union_us - collective_exposed_us)
                / 1000.0,
                "overlapped_per_step_ms": (collective_union_us - collective_exposed_us)
                / 2000.0,
            },
            "by_operation": collective_by_op,
        },
        "host": {
            "categories": host_categories,
            "exact_rows": host_exact_rows,
            "runtime_categories": runtime_categories,
            "runtime_api_rows": runtime_api_rows,
            "hipModuleLaunchKernel": [
                row for row in runtime_api_rows if row["api"] == "hipModuleLaunchKernel"
            ],
            "hipPointerGetAttribute": [
                row
                for row in runtime_api_rows
                if row["api"] == "hipPointerGetAttribute"
            ],
            "launch_gaps": {
                "launch_calls_total": len(launches),
                "inter_launch_positive_gap_count": len(launch_gaps),
                "inter_launch_positive_gap_total_ms": sum(launch_gaps) / 1000.0,
                "inter_launch_gap_p50_us": percentile(launch_gaps, 0.5),
                "inter_launch_gap_p95_us": percentile(launch_gaps, 0.95),
                "inter_launch_gap_max_us": max(launch_gaps) if launch_gaps else None,
                "launch_to_device_gap_count": len(queue_gaps),
                "launch_to_device_gap_mean_us": sum(queue_gaps) / len(queue_gaps)
                if queue_gaps
                else None,
                "launch_to_device_gap_p50_us": percentile(queue_gaps, 0.5),
                "launch_to_device_gap_p95_us": percentile(queue_gaps, 0.95),
                "launch_to_device_gap_max_us": max(queue_gaps) if queue_gaps else None,
                "negative_gap_count": negative_queue_gaps,
            },
            "warning": "Overlapping host spans are not additive with GPU time.",
        },
        "missing_external_id": {
            "events_total": sum(len(rows) for rows in missing_external.values()),
            "external_id_present_but_producer_unresolved": unresolved_present_external,
            "rows": missing_rows,
            "resolution_method": "Direct External id to CPU producer; absent ids retained and grouped by category/kernel/correlated HIP API.",
        },
        "unclassified_gpu_work": {
            **detail_metrics.get("other", ZERO_METRICS),
            "kernels": [row for row in top_kernels if row["category"] == "other"],
        },
    }


parse_trace = scan_trace


def add_check(
    checks: dict[str, dict[str, Any]], name: str, passed: bool, evidence: Any
) -> None:
    checks[name] = {"pass": bool(passed), "evidence": evidence}


def status_is_zero(path: Path, required: tuple[str, ...]) -> bool:
    values = parse_kv(path)
    return all(values.get(key) == "0" for key in required)


def validate_status_file(
    path: Path, required: tuple[str, ...]
) -> tuple[bool, dict[str, str]]:
    values = parse_kv(path)
    return all(values.get(key) == "0" for key in required), values


def validate_kfd_file(path: Path) -> tuple[bool, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    idle_count = sum(line.strip() == "status=idle" for line in text.splitlines())
    bad = bool(
        re.search(
            r"^status=(?:busy|error)$|^non_service_kfd_client\b|^supplemental_known_workloads=.+",
            text,
            re.MULTILINE,
        )
    )
    return idle_count == 1 and not bad, {"idle_count": idle_count, "bad_marker": bad}


def validate_source_records(
    audit: dict[str, Any],
    run_meta: dict[str, str],
    tree: dict[str, str],
    before: str,
    after: str,
) -> tuple[bool, dict[str, Any]]:
    keys = (
        "lumen_commit",
        "aiter_commit",
        "lumen_tree_sha256",
        "aiter_tree_sha256",
        "source_bundle_sha256",
        "runtime_modules_sha256",
        "f4gemm_directory_sha256",
        "tuned_tables_sha256",
    )
    rows = {key: (audit.get(key), run_meta.get(key), tree.get(key)) for key in keys}
    passed = all(a == b == c for a, b, c in rows.values())
    passed &= before == after == audit.get("source_bundle_sha256")
    passed &= run_meta.get("workload_sha256") == audit.get("workload_sha256")
    return bool(passed), {"fields": rows, "before": before, "after": after}


EXPECTED_ROUTE_SHAPES = {
    "4096,4096,16384": 840,
    "4096,12288,16384": 840,
    "6144,4096,16384": 840,
    "12288,4096,16384": 1680,
    "16384,4096,4096": 2240,
    "16384,4096,6144": 840,
    "16384,4096,12288": 3080,
    "16384,6144,4096": 1400,
    "16384,12288,4096": 3640,
}


def _route_arm(root: Path, arm: str, cache: dict[str, Any]) -> dict[str, Any]:
    reports: dict[int, Any] = {}
    shape_bytes: dict[int, bytes] = {}
    parsed_shapes: dict[int, dict[str, tuple[str, int]]] = {}
    errors: list[str] = []
    for rank in range(8):
        report_path = root / arm / f"route-rank{rank}.json"
        shape_path = root / arm / f"mxfp4-shapes-rank{rank}.csv"
        reports[rank] = json.loads(report_path.read_text(encoding="utf-8"))
        shape_bytes[rank] = shape_path.read_bytes()
        lines = shape_path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "M,N,K,asm_available,backend,calls":
            errors.append(f"rank{rank}:shape_header")
            parsed_shapes[rank] = {}
        else:
            rows: dict[str, tuple[str, int]] = {}
            for line in lines[1:]:
                fields = line.split(",")
                if len(fields) != 6:
                    errors.append(f"rank{rank}:shape_row:{line}")
                    continue
                rows[",".join(fields[:3])] = (fields[4], int(fields[5]))
            parsed_shapes[rank] = rows
        report = reports[rank]
        counts = report.get("counts", {})
        exact_counts = {
            "rank": rank,
            "world_size": 8,
            "instrumentation_installed": 1,
            "enabled_qkv": 35,
            "enabled_swiglu": 35,
            "qkv_linear_success": 1400,
            "swiglu_fwd_success": 1400,
            "swiglu_bwd_success": 840,
        }
        for key, expected in exact_counts.items():
            if counts.get(key) != expected:
                errors.append(f"rank{rank}:{key}={counts.get(key)!r}")
        for key in (
            "qkv_linear_failure",
            "swiglu_fwd_failure",
            "swiglu_bwd_failure",
            "eligible_original_qkv_forward",
            "eligible_original_swiglu_forward",
        ):
            if counts.get(key, 0) != 0:
                errors.append(f"rank{rank}:unexpected_{key}")
        names = report.get("unquantized_linear_names", [])
        tail_names = [name for name in names if name != "lm_head"]
        if (
            len(names) != 8
            or names.count("lm_head") != 1
            or len(tail_names) != 7
            or any(re.search(r"layers\.35\.", name) is None for name in tail_names)
        ):
            errors.append(f"rank{rank}:bf16_inventory")
        if (
            report.get("lm_head_count"),
            report.get("lm_head_weight_dtype"),
            report.get("lm_head_quant_enabled"),
        ) != (1, "bfloat16", False):
            errors.append(f"rank{rank}:lm_head_contract")
        if (
            report.get("lumen_import") != "/home/xdai/Lumen/lumen/__init__.py"
            or report.get("aiter_import") != "/home/xdai/aiter/aiter/__init__.py"
        ):
            errors.append(f"rank{rank}:imports")
    if any(shape_bytes[rank] != shape_bytes[0] for rank in range(1, 8)):
        errors.append("rank_shape_csv_mismatch")
    expected_rows = {
        shape: (cache["choices"].get(shape), calls)
        for shape, calls in EXPECTED_ROUTE_SHAPES.items()
    }
    if parsed_shapes[0] != expected_rows:
        errors.append("shape_topology_or_backend")
    return {
        "valid": not errors,
        "errors": errors,
        "reports": reports,
        "shape_sha256_by_rank": {
            str(rank): hashlib.sha256(content).hexdigest()
            for rank, content in shape_bytes.items()
        },
        "shape_rows_rank0": parsed_shapes[0],
    }


def _differential(bf16: dict[str, Any], policy_a: dict[str, Any]) -> dict[str, Any]:
    def section(name: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for category in sorted(set(bf16[name]) | set(policy_a[name])):
            left = bf16[name].get(category, ZERO_METRICS)
            right = policy_a[name].get(category, ZERO_METRICS)
            result[category] = {
                "bf16": left,
                "policy_a": right,
                "policy_a_minus_bf16_raw_per_step_ms": right["raw_per_step_ms"]
                - left["raw_per_step_ms"],
                "policy_a_minus_bf16_union_per_step_ms": right["union_per_step_ms"]
                - left["union_per_step_ms"],
                "policy_a_minus_bf16_calls_per_step": right["calls_per_step"]
                - left["calls_per_step"],
            }
        return result

    bf16_span = bf16["window"]["profiler_span_per_step_ms"]
    policy_a_span = policy_a["window"]["profiler_span_per_step_ms"]
    joint = policy_a["joint_unions"]["a4w4_plus_quant_layout"]["union_per_step_ms"]
    target = bf16_span / 1.6
    remaining = max(policy_a_span - joint, 1e-12)
    return {
        "broad_categories": section("broad_categories"),
        "detail_categories": section("detail_categories"),
        "profile_window": {
            "bf16_profiler_span_per_step_ms": bf16_span,
            "policy_a_profiler_span_per_step_ms": policy_a_span,
            "diagnostic_speedup_bf16_over_policy_a": bf16_span / policy_a_span,
            "policy_a_target_for_1_6x_ms": target,
            "policy_a_gap_to_1_6x_ms": policy_a_span - target,
        },
        "amdahl": {
            "a4w4_plus_quant_union_per_step_ms": joint,
            "fraction_of_policy_a_profiler_span": joint / policy_a_span,
            "idealized_policy_a_internal_speedup_if_joint_vanished": policy_a_span
            / remaining,
            "idealized_bf16_over_policy_a_if_joint_vanished": bf16_span / remaining,
            "joint_union_covers_profile_gap": joint >= policy_a_span - target,
            "warning": "Impossible-zero-cost overlap-safe ceiling, not expected speedup.",
        },
    }


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
ROUTE_ARMS = ("smoke_mxfp4_policy_a", "replay_mxfp4_policy_a")
ARM_STEPS = {
    "smoke_mxfp4_policy_a": 3,
    "replay_mxfp4_policy_a": 3,
    "profile_bf16": 8,
    "profile_mxfp4_policy_a": 8,
}
STANDARD_ARM_FILES = (
    "train.log",
    "trace.json",
    "profile.txt",
    "run-meta.txt",
    "train-exit-status.txt",
    "postflight-status.txt",
    "kfd-before.txt",
    "kfd-prelaunch.txt",
    "kfd-after.txt",
    "kfd-after-attempts.txt",
    "cache-before.sha256",
    "cache-after.sha256",
    "source-bundle-before.sha256",
    "source-bundle-after.sha256",
    "tree-state-after.txt",
    "numa-policy.txt",
    "rocm-smi-before.txt",
    "rocm-smi-after.txt",
)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64_RE.fullmatch(value) is not None


def _is_hex40(value: Any) -> bool:
    return isinstance(value, str) and HEX40_RE.fullmatch(value) is not None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _required_files(root: Path) -> list[Path]:
    paths = [
        root / "profile-meta.txt",
        root / "source-audit.json",
        root / "imports-before.txt",
        root / "campaign-stage-status.txt",
        root / "campaign-progress.log",
        root / "kfd-campaign-before.txt",
        root / "kfd-campaign-after.txt",
    ]
    for arm in ARMS:
        paths.extend(root / arm / name for name in STANDARD_ARM_FILES)
        if arm in FORMAL_ARMS:
            paths.append(root / arm / "profile_shapes.txt")
        if arm in ROUTE_ARMS:
            for rank in range(8):
                paths.extend(
                    (
                        root / arm / f"route-rank{rank}.json",
                        root / arm / f"mxfp4-shapes-rank{rank}.csv",
                    )
                )
    paths.extend(
        (
            root / "cache/cache_build_policy_a/mxfp4-autotune.json",
            root / "cache/profile_bf16/mxfp4-autotune.json",
            root / "cache/profile_mxfp4_policy_a/mxfp4-autotune.json",
        )
    )
    return paths


def _validate_progress(
    path: Path, *, require_completion: bool
) -> tuple[bool, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stages: list[str] = []
    malformed: list[str] = []
    for line in lines:
        match = re.fullmatch(r"\S+ stage=([a-z0-9_]+)", line)
        if match:
            stages.append(match.group(1))
        elif line.strip():
            malformed.append(line)
    expected = (
        [*EXPECTED_PROGRESS, "manifest_ready"]
        if require_completion
        else list(EXPECTED_PROGRESS)
    )
    return stages == expected and not malformed, {
        "stages": stages,
        "expected": expected,
        "malformed": malformed,
    }


def _validate_source_audit(
    root: Path, audit: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    hash_fields = (
        "lumen_tree_sha256",
        "aiter_tree_sha256",
        "source_bundle_sha256",
        "runtime_modules_sha256",
        "f4gemm_directory_sha256",
        "tuned_tables_sha256",
        "workload_sha256",
    )
    harness = audit.get("harness_sha256", {})
    expected_harness = (
        root / "run_policy_a_profile.sh",
        root / "route_entry.py",
        root / "analyze_policy_a_profile.py",
        root / "test_analyze_policy_a_profile.py",
        root / "protocol.md",
        root / "test_runner.py",
    )
    harness_results: dict[str, Any] = {}
    harness_ok = isinstance(harness, dict) and set(harness) == {
        str(path) for path in expected_harness
    }
    for path in expected_harness:
        expected = harness.get(str(path)) if isinstance(harness, dict) else None
        actual = sha256_file(path) if path.is_file() else None
        harness_results[str(path)] = {"recorded": expected, "actual": actual}
        harness_ok &= expected == actual and _is_hex64(expected)
    imports = audit.get("imports", {})
    runtime = audit.get("runtime", {})
    checks = {
        "schema": audit.get("schema") == 1,
        "lumen_branch": audit.get("lumen_branch") == "dev/mxfp4",
        "lumen_commit": _is_hex40(audit.get("lumen_commit")),
        "aiter_commit": _is_hex40(audit.get("aiter_commit")),
        "state_hashes": all(_is_hex64(audit.get(key)) for key in hash_fields),
        "imports": imports
        == {
            "lumen": "/home/xdai/Lumen/lumen/__init__.py",
            "aiter": "/home/xdai/aiter/aiter/__init__.py",
        },
        "runtime": isinstance(runtime, dict)
        and runtime.get("devices") == 8
        and bool(runtime.get("torch"))
        and bool(runtime.get("hip")),
        "harness": harness_ok,
    }
    return all(checks.values()), {
        "checks": checks,
        "harness": harness_results,
        "runtime": runtime,
        "imports": imports,
    }


def _validate_profile_meta(
    root: Path, meta: dict[str, str], audit: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    dynamic_hashes = (
        "cache_sha256",
        "source_audit_sha256",
        "runner_sha256",
        "route_entry_sha256",
        "analyzer_sha256",
        "analyzer_test_sha256",
        "protocol_sha256",
        "test_sha256",
    )
    expected_files = {
        "runner_sha256": root / "run_policy_a_profile.sh",
        "route_entry_sha256": root / "route_entry.py",
        "analyzer_sha256": root / "analyze_policy_a_profile.py",
        "analyzer_test_sha256": root / "test_analyze_policy_a_profile.py",
        "protocol_sha256": root / "protocol.md",
        "test_sha256": root / "test_runner.py",
    }
    file_hashes = {
        key: {
            "recorded": meta.get(key),
            "actual": sha256_file(path) if path.is_file() else None,
        }
        for key, path in expected_files.items()
    }
    checks = {
        "fixed_fields": all(
            meta.get(key) == value for key, value in EXPECTED_META.items()
        ),
        "dynamic_hash_format": all(_is_hex64(meta.get(key)) for key in dynamic_hashes),
        "smoke_aiter_cache_digest": meta.get("smoke_aiter_config_cache_sha256")
        == "empty"
        or _is_hex64(meta.get("smoke_aiter_config_cache_sha256")),
        "source_audit_hash": meta.get("source_audit_sha256")
        == sha256_file(root / "source-audit.json"),
        "harness_hashes": all(
            values["recorded"] == values["actual"] for values in file_hashes.values()
        ),
        "audit_harness_agrees": all(
            audit.get("harness_sha256", {}).get(str(path)) == meta.get(key)
            for key, path in expected_files.items()
        ),
    }
    return all(checks.values()), {"checks": checks, "file_hashes": file_hashes}


def _log_contract(
    arm: str, log: dict[str, Any], cache_choice_count: int
) -> tuple[bool, dict[str, Any]]:
    is_mxfp4 = arm != "profile_bf16"
    route = arm in ROUTE_ARMS
    checks = {
        "steps": log["step_numbers_exact"] and log["max_steps_exact"],
        "finite": log["finite"],
        "one_validation": log["validation_loss_count"] == 1,
        "completion": log["training_complete_exact"],
        "no_failure_markers": not any(
            log["failure_markers_before_completion"].values()
        ),
        "teardown": log["post_completion"]["valid"],
        "profile_markers": (
            not route
            and log["profile_marker_order_exact"]
            and log["profile_marker_counts"] == {"start": 1, "stop": 1}
        )
        or (route and log["profile_marker_counts"] == {"start": 0, "stop": 0}),
    }
    if is_mxfp4:
        checks.update(
            {
                "quantized_linear_summary": log["quantization_summaries"]
                == [(245, 8)],
                "packed_qkv_summary": log["packed_qkv_enabled"] == [35],
                "split_swiglu_summary": log["split_swiglu_enabled"] == [35],
                "protected_tail_warnings": log["qkv_warning_count"] == 8
                and log["swiglu_warning_count"] == 8,
                "loaded_cache_decisions_valid": not log["loaded_decisions"]
                or all(
                    value == cache_choice_count for value in log["loaded_decisions"]
                ),
                "no_online_tune_after_smoke": arm == "smoke_mxfp4_policy_a"
                or not log["online_autotune_lines"],
            }
        )
    else:
        checks.update(
            {
                "no_quantized_linear_summary": not log["quantization_summaries"],
                "no_packed_qkv_summary": not log["packed_qkv_enabled"],
                "no_split_swiglu_summary": not log["split_swiglu_enabled"],
                "no_protected_tail_warnings": log["qkv_warning_count"] == 0
                and log["swiglu_warning_count"] == 0,
                "no_cache_decisions": not log["loaded_decisions"],
                "no_online_tune": not log["online_autotune_lines"],
            }
        )
    return all(checks.values()), checks


def _validate_pairing(logs: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    model_hashes = {arm: log["model_init_hashes"] for arm, log in logs.items()}
    batches = {arm: log["batch_evidence"] for arm, log in logs.items()}
    validations = {arm: log["validation_evidence"] for arm, log in logs.items()}
    first_arm = ARMS[0]
    model_ok = all(len(values) == 1 for values in model_hashes.values()) and all(
        values == model_hashes[first_arm] for values in model_hashes.values()
    )
    batch_shape_ok = all(
        set(values) == set(range(8))
        and all(
            row["step"] == 1 and row["microbatches"] == 8 for row in values.values()
        )
        and not logs[arm]["batch_duplicate_ranks"]
        for arm, values in batches.items()
    )
    batch_equal = all(values == batches[first_arm] for values in batches.values())
    validation_shape_ok = all(
        set(values) == set(range(8))
        and all(row["batches"] == 16 for row in values.values())
        and not logs[arm]["validation_duplicate_ranks"]
        for arm, values in validations.items()
    )
    validation_equal = all(
        values == validations[first_arm] for values in validations.values()
    )
    checks = {
        "same_model_initialization": model_ok,
        "first_update_evidence_shape": batch_shape_ok,
        "first_update_digests_equal": batch_equal,
        "validation_evidence_shape": validation_shape_ok,
        "validation_digests_equal": validation_equal,
    }
    return all(checks.values()), {
        "checks": checks,
        "model_init": model_hashes,
        "first_update": batches,
        "validation": validations,
    }


def _validate_trace_topology(trace: dict[str, Any], precision: str) -> tuple[bool, Any]:
    checks: dict[str, bool] = {"trace_contract": trace["contract"]["all_pass"]}
    if precision == "mxfp4":
        host = trace["host"]["categories"]
        detail = trace["detail_categories"]
        expected_host = {
            "packed_qkv_forward": 280,
            "packed_qkv_backward": 280,
            "quantized_linear_forward": 1120,
            "quantized_linear_backward": 1120,
        }
        for name, expected in expected_host.items():
            checks[f"host_{name}_calls_per_step"] = (
                host.get(name, ZERO_METRICS)["calls_per_step"] == expected
            )
        checks["protected_tail_bf16_calls_per_step"] = (
            detail.get("bf16_protected_tail", ZERO_METRICS)["calls_per_step"] == 168
        )
        checks["lm_head_bf16_calls_per_step"] = (
            detail.get("bf16_lm_head", ZERO_METRICS)["calls_per_step"] == 24
        )
        checks["a4w4_identity_resolved"] = bool(trace["a4w4_exact_shapes"]) and all(
            row.get("selected_backend") in {"asm", "shuffled"}
            and bool(row.get("selected_symbol"))
            for row in trace["a4w4_exact_shapes"]
        )
    return all(checks.values()), checks


def _manifest_entries(path: Path) -> tuple[dict[Path, str], list[str]]:
    entries: dict[Path, str] = {}
    errors: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            errors.append(f"malformed:{line}")
            continue
        target = Path(match.group(2))
        if target in entries:
            errors.append(f"duplicate:{target}")
        entries[target] = match.group(1)
    return entries, errors


def _validate_finalization(
    root: Path, meta: dict[str, str], *, require_completion: bool
) -> tuple[bool, dict[str, Any]]:
    manifest_path = root / "campaign-artifacts.sha256"
    complete_path = root / "profile-complete.txt"
    exit_path = root / "policy_a-profile-exit-status.txt"
    present = {
        "manifest": manifest_path.is_file(),
        "complete": complete_path.is_file(),
        "exit": exit_path.is_file(),
    }
    if not require_completion and not any(present.values()):
        return True, {"state": "post-analysis-pending", "present": present}
    if not require_completion:
        return False, {"state": "unexpected-prefinal-sentinel", "present": present}
    if not all(present.values()):
        return False, {"state": "partial-finalization", "present": present}
    entries, errors = _manifest_entries(manifest_path)
    excluded = {manifest_path, complete_path, exit_path}
    expected_paths = {
        path for path in root.rglob("*") if path.is_file() and path not in excluded
    }
    manifest_ok = set(entries) == expected_paths and not errors
    mismatches: dict[str, Any] = {}
    for path, expected in entries.items():
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches[str(path)] = {"expected": expected, "actual": actual}
            manifest_ok = False
    complete = parse_kv(complete_path)
    expected_complete_keys = {
        "schema",
        "completed",
        "cache_sha256",
        "source_audit_sha256",
        "profile_meta_sha256",
        "campaign_stage_status_sha256",
        "analysis_json_sha256",
        "analysis_markdown_sha256",
        "artifact_manifest_sha256",
    }
    complete_checks = {
        "exact_keys": set(complete) == expected_complete_keys,
        "schema": complete.get("schema") == "1",
        "completed": complete.get("completed") == ",".join(ARMS),
        "cache": complete.get("cache_sha256") == meta.get("cache_sha256"),
        "source_audit": complete.get("source_audit_sha256")
        == sha256_file(root / "source-audit.json"),
        "profile_meta": complete.get("profile_meta_sha256")
        == sha256_file(root / "profile-meta.txt"),
        "campaign_stage_status": complete.get("campaign_stage_status_sha256")
        == sha256_file(root / "campaign-stage-status.txt"),
        "analysis_json": complete.get("analysis_json_sha256")
        == sha256_file(root / "profile_analysis.json"),
        "analysis_markdown": complete.get("analysis_markdown_sha256")
        == sha256_file(root / "profile_analysis.md"),
        "manifest": complete.get("artifact_manifest_sha256")
        == sha256_file(manifest_path),
        "exit_zero": exit_path.read_text(encoding="utf-8").split() == ["0"],
    }
    signed_report = _read_json(root / "profile_analysis.json")
    complete_checks["signed_analysis_passed"] = (
        signed_report.get("schema") == 1
        and signed_report.get("root") == str(root)
        and signed_report.get("integrity", {}).get("all_pass") is True
    )
    return manifest_ok and all(complete_checks.values()), {
        "state": "complete",
        "present": present,
        "manifest_entries": len(entries),
        "manifest_errors": errors,
        "manifest_mismatches": mismatches,
        "complete_checks": complete_checks,
    }


def analyze_root(root: Path, *, require_completion: bool = False) -> dict[str, Any]:
    root = root.resolve()
    checks: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "schema": 1,
        "root": str(root),
        "mode": "verify-completion" if require_completion else "prefinal-analysis",
        "integrity": {"all_pass": False, "checks": checks},
        "arms": {},
    }
    required = _required_files(root)
    missing = [str(path) for path in required if not path.is_file()]
    empty = [
        str(path) for path in required if path.is_file() and path.stat().st_size == 0
    ]
    add_check(
        checks,
        "required_artifacts",
        not missing and not empty,
        {"missing": missing, "empty": empty},
    )
    if missing or empty:
        report["integrity"]["failed_checks"] = ["required_artifacts"]
        return report

    try:
        meta = parse_kv(root / "profile-meta.txt")
        audit = _read_json(root / "source-audit.json")
        report["profile_meta"] = meta
        report["source_audit"] = audit

        passed, evidence = _validate_profile_meta(root, meta, audit)
        add_check(checks, "profile_meta", passed, evidence)
        passed, evidence = _validate_source_audit(root, audit)
        add_check(checks, "source_audit", passed, evidence)
        imports_before = parse_kv(root / "imports-before.txt")
        imports_expected = {
            **audit["imports"],
            "torch": str(audit["runtime"]["torch"]),
            "hip": str(audit["runtime"]["hip"]),
            "devices": str(audit["runtime"]["devices"]),
        }
        add_check(
            checks,
            "imports_before",
            imports_before == imports_expected,
            {"actual": imports_before, "expected": imports_expected},
        )

        stage_ok, stages = validate_status_file(
            root / "campaign-stage-status.txt", EXPECTED_STAGE_STATUS
        )
        expected_stage_keys = set(EXPECTED_STAGE_STATUS)
        if require_completion:
            expected_stage_keys.add("analysis")
        stage_ok &= set(stages) == expected_stage_keys
        if require_completion:
            stage_ok &= stages.get("analysis") == "0"
        add_check(checks, "campaign_stage_status", stage_ok, stages)
        progress_ok, progress = _validate_progress(
            root / "campaign-progress.log", require_completion=require_completion
        )
        add_check(checks, "campaign_progress", progress_ok, progress)
        for phase in ("before", "after"):
            path = root / f"kfd-campaign-{phase}.txt"
            kfd_ok, evidence = validate_kfd_file(path)
            add_check(checks, f"campaign_kfd_{phase}", kfd_ok, evidence)

        cache_paths = {
            "build": root / "cache/cache_build_policy_a/mxfp4-autotune.json",
            "bf16": root / "cache/profile_bf16/mxfp4-autotune.json",
            "policy_a": root / "cache/profile_mxfp4_policy_a/mxfp4-autotune.json",
        }
        caches = {
            name: load_cache(path, meta.get("cache_sha256"))
            for name, path in cache_paths.items()
        }
        report["caches"] = caches
        cache_bytes = {name: path.read_bytes() for name, path in cache_paths.items()}
        cache_ok = all(cache["valid"] for cache in caches.values()) and all(
            value == cache_bytes["build"] for value in cache_bytes.values()
        )
        add_check(
            checks,
            "cache_schema_identity_and_copies",
            cache_ok,
            {
                "sha256": {name: value["sha256"] for name, value in caches.items()},
                "errors": {name: value["errors"] for name, value in caches.items()},
            },
        )
        cache = caches["build"]

        logs: dict[str, dict[str, Any]] = {}
        run_meta: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            arm_root = root / arm
            arm_meta = parse_kv(arm_root / "run-meta.txt")
            run_meta[arm] = arm_meta
            expected_precision = "bf16" if arm == "profile_bf16" else "mxfp4"
            is_formal = arm in FORMAL_ARMS
            meta_checks = {
                "case": arm_meta.get("case") == arm,
                "precision": arm_meta.get("precision") == expected_precision,
                "common": all(
                    arm_meta.get(key) == value for key, value in COMMON_RUN_META.items()
                ),
                "profile_window": arm_meta.get("profile_start")
                == ("7" if is_formal else "none")
                and arm_meta.get("profile_end") == ("8" if is_formal else "none"),
                "profile_shapes": arm_meta.get("profile_shapes")
                == ("1" if is_formal else "0"),
                "shape_log": arm_meta.get("shape_log_enabled")
                == ("1" if arm in ROUTE_ARMS else "0"),
                "kfd_permissions": re.fullmatch(
                    r"c[-rwx]{9}:root:render", arm_meta.get("kfd", "")
                )
                is not None,
            }
            add_check(checks, f"{arm}.run_meta", all(meta_checks.values()), meta_checks)

            train_ok, train_status = validate_status_file(
                arm_root / "train-exit-status.txt", ("torchrun", "tee")
            )
            add_check(checks, f"{arm}.train_status", train_ok, train_status)
            post_ok, post_status = validate_status_file(
                arm_root / "postflight-status.txt",
                ("torchrun", "tee", "rocm_smi", "post_idle", "provenance"),
            )
            add_check(checks, f"{arm}.postflight_status", post_ok, post_status)
            for phase in ("before", "prelaunch", "after"):
                kfd_ok, evidence = validate_kfd_file(arm_root / f"kfd-{phase}.txt")
                add_check(checks, f"{arm}.kfd_{phase}", kfd_ok, evidence)
            attempt_text = first_token(arm_root / "kfd-after-attempts.txt")
            attempt_ok = attempt_text in {"1", "2", "3", "4", "5"}
            add_check(checks, f"{arm}.kfd_after_attempts", attempt_ok, attempt_text)

            tree = parse_kv(arm_root / "tree-state-after.txt")
            source_ok, source_evidence = validate_source_records(
                audit,
                arm_meta,
                tree,
                first_token(arm_root / "source-bundle-before.sha256"),
                first_token(arm_root / "source-bundle-after.sha256"),
            )
            add_check(checks, f"{arm}.source_freeze", source_ok, source_evidence)

            log = parse_training_log(
                arm_root / "train.log",
                ARM_STEPS[arm],
                allow_shape_flush=arm in ROUTE_ARMS,
            )
            logs[arm] = log
            log_ok, log_evidence = _log_contract(arm, log, cache["choice_count"])
            add_check(checks, f"{arm}.training_log", log_ok, log_evidence)
            if arm in ROUTE_ARMS:
                trace_stub = _read_json(arm_root / "trace.json")
                profile_stub = parse_kv(arm_root / "profile.txt")
                stub_ok = trace_stub == {
                    "schema": 1,
                    "profiling_enabled": False,
                    "reason": "route-cache arm",
                } and profile_stub == {
                    "profiling_enabled": "0",
                    "reason": "route-cache-arm",
                }
                add_check(
                    checks,
                    f"{arm}.profile_disabled_sentinel",
                    stub_ok,
                    {"trace": trace_stub, "profile": profile_stub},
                )

            cache_before = first_token(arm_root / "cache-before.sha256")
            cache_after = first_token(arm_root / "cache-after.sha256")
            expected_before = (
                "absent" if arm == "smoke_mxfp4_policy_a" else meta["cache_sha256"]
            )
            cache_record_ok = (
                cache_before == expected_before and cache_after == meta["cache_sha256"]
            )
            cache_record_ok &= arm_meta.get("cache_sha256_before") == expected_before
            cache_record_ok &= (
                post_status.get("cache_sha256_after") == meta["cache_sha256"]
            )
            add_check(
                checks,
                f"{arm}.cache_before_after",
                cache_record_ok,
                {"before": cache_before, "after": cache_after},
            )

            aiter_before = arm_meta.get("aiter_config_cache_sha256_before")
            aiter_after = post_status.get("aiter_config_cache_sha256_after")
            if arm == "smoke_mxfp4_policy_a":
                aiter_ok = aiter_before == "empty" and bool(aiter_after)
            elif arm == "replay_mxfp4_policy_a":
                aiter_ok = (
                    aiter_before
                    == aiter_after
                    == meta.get("smoke_aiter_config_cache_sha256")
                )
            else:
                aiter_ok = aiter_before == aiter_after == "empty"
            add_check(
                checks,
                f"{arm}.aiter_config_cache",
                aiter_ok,
                {"before": aiter_before, "after": aiter_after},
            )

            report["arms"][arm] = {
                "run_meta": arm_meta,
                "log": log,
                "status": {"train": train_status, "postflight": post_status},
            }

        route_reports = {arm: _route_arm(root, arm, cache) for arm in ROUTE_ARMS}
        route_equal = (
            route_reports["smoke_mxfp4_policy_a"]["reports"]
            == route_reports["replay_mxfp4_policy_a"]["reports"]
            and route_reports["smoke_mxfp4_policy_a"]["shape_sha256_by_rank"]
            == route_reports["replay_mxfp4_policy_a"]["shape_sha256_by_rank"]
        )
        route_ok = (
            all(value["valid"] for value in route_reports.values()) and route_equal
        )
        add_check(
            checks,
            "route_smoke_replay_identity",
            route_ok,
            {
                "per_arm_valid": {
                    arm: value["valid"] for arm, value in route_reports.items()
                },
                "errors": {
                    arm: value["errors"] for arm, value in route_reports.items()
                },
                "equal": route_equal,
            },
        )
        report["route"] = route_reports

        pairing_ok, pairing = _validate_pairing(logs)
        add_check(checks, "paired_run_evidence", pairing_ok, pairing["checks"])
        report["pairing"] = pairing

        command_ok, command_evidence = command_contract(
            run_meta["profile_bf16"].get("command", ""),
            run_meta["profile_mxfp4_policy_a"].get("command", ""),
        )
        add_check(checks, "formal_command_pairing", command_ok, command_evidence)

        traces: dict[str, dict[str, Any]] = {}
        for arm, precision in (
            ("profile_bf16", "bf16"),
            ("profile_mxfp4_policy_a", "mxfp4"),
        ):
            trace = scan_trace(
                root / arm / "trace.json",
                logs[arm],
                precision=precision,
                token_rows=16384,
                cache=cache,
            )
            traces[arm] = trace
            topology_ok, topology = _validate_trace_topology(trace, precision)
            add_check(checks, f"{arm}.trace", topology_ok, topology)
            report["arms"][arm]["trace"] = trace

        differential = _differential(
            traces["profile_bf16"], traces["profile_mxfp4_policy_a"]
        )
        report["differential"] = differential
        bf16_loss = logs["profile_bf16"]["validation_nll"]
        policy_a_loss = logs["profile_mxfp4_policy_a"]["validation_nll"]
        report["accuracy"] = {
            "bf16_validation_nll": bf16_loss,
            "policy_a_validation_nll": policy_a_loss,
            "policy_a_minus_bf16": policy_a_loss - bf16_loss,
            "relative_delta": (policy_a_loss - bf16_loss) / bf16_loss,
            "interpretation": "Paired short-run diagnostic; not a substitute for long-horizon convergence validation.",
        }

        final_ok, finalization = _validate_finalization(
            root, meta, require_completion=require_completion
        )
        add_check(checks, "finalization", final_ok, finalization)
        report["finalization"] = finalization
    except Exception as error:  # fail closed while preserving a machine-readable report
        add_check(
            checks,
            "analysis_exception",
            False,
            {"type": type(error).__name__, "message": str(error)},
        )

    failed = [name for name, row in checks.items() if not row["pass"]]
    report["integrity"]["all_pass"] = not failed
    report["integrity"]["failed_checks"] = failed
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    integrity = report["integrity"]
    state = "PASS" if integrity["all_pass"] else "FAIL"
    lines = [
        "# Fresh BF16 / MXFP4 Policy A profiling analysis",
        "",
        f"Integrity result: **{state}**",
        "",
        "This report uses only artifacts under the campaign root. Profile timing is diagnostic; the 1.6x target still requires a fresh unprofiled symmetric confirmation.",
        "",
    ]
    failed = integrity.get("failed_checks", [])
    if failed:
        lines.extend(("## Failed integrity checks", ""))
        lines.extend(f"- `{name}`" for name in failed)
        lines.append("")
    if "differential" in report:
        diff = report["differential"]
        window = diff["profile_window"]
        accuracy = report["accuracy"]
        lines.extend(
            (
                "## Paired profile outcome",
                "",
                "| Metric | BF16 | MXFP4 policy_a | Delta / ratio |",
                "|---|---:|---:|---:|",
                f"| Profiler span per step (ms) | {_fmt(window['bf16_profiler_span_per_step_ms'])} | {_fmt(window['policy_a_profiler_span_per_step_ms'])} | speedup {_fmt(window['diagnostic_speedup_bf16_over_policy_a'])}x |",
                f"| Validation NLL | {_fmt(accuracy['bf16_validation_nll'], 6)} | {_fmt(accuracy['policy_a_validation_nll'], 6)} | {_fmt(accuracy['policy_a_minus_bf16'], 6)} |",
                f"| 1.6x target step time (ms) | n/a | {_fmt(window['policy_a_target_for_1_6x_ms'])} | gap {_fmt(window['policy_a_gap_to_1_6x_ms'])} ms |",
                "",
                "The validation delta is a paired eight-step diagnostic, not a long-horizon convergence claim.",
                "",
                "## Overlap-safe category comparison",
                "",
                "| Category | BF16 raw ms/step | PolicyA raw ms/step | BF16 union ms/step | PolicyA union ms/step |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for category, row in diff["broad_categories"].items():
            lines.append(
                f"| `{category}` | {_fmt(row['bf16']['raw_per_step_ms'])} | {_fmt(row['policy_a']['raw_per_step_ms'])} | {_fmt(row['bf16']['union_per_step_ms'])} | {_fmt(row['policy_a']['union_per_step_ms'])} |"
            )
        amdahl = diff["amdahl"]
        lines.extend(
            (
                "",
                "## Amdahl ceiling",
                "",
                f"A4W4 plus quant/layout occupies {_fmt(amdahl['a4w4_plus_quant_union_per_step_ms'])} ms/step by interval union ({_fmt(100 * amdahl['fraction_of_policy_a_profiler_span'], 2)}% of the policy_a profile span). Even removing that union entirely would yield an idealized BF16-over-policy_a ratio of {_fmt(amdahl['idealized_bf16_over_policy_a_if_joint_vanished'])}x. This is an impossible-zero-cost ceiling, not a predicted saving.",
                "",
            )
        )
    lines.extend(("## Integrity checks", "", "| Check | Result |", "|---|---|"))
    for name, row in integrity["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if row['pass'] else 'FAIL'} |")
    lines.append("")
    if "finalization" in report:
        lines.extend(
            (
                "## Finalization state",
                "",
                f"`{report['finalization']['state']}`",
                "",
            )
        )
    return "\n".join(lines)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def self_test() -> None:
    assert interval_union([]) == 0.0
    assert interval_union([(0, 2), (1, 3), (5, 7)]) == 5.0
    assert (
        matrix_shape(Producer("aten::mm", [[4, 8], [8, 16]], None, None))
        == "M=4,N=16,K=8"
    )
    event = GPUEvent(0.0, 1.0, "kernel", "unknown", None, None, 0)
    assert classify_event(event) == "other"
    bf16 = "torchrun train.py --mode bf16 --seed 1234"
    mxfp4 = "torchrun train.py --mode mxfp4 --seed 1234 --mxfp4-pack-qkv --mxfp4-fuse-swiglu"
    assert normalize_command(bf16) == normalize_command(mxfp4)
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        trace.write_text(
            json.dumps({"traceEvents": [{"name": "one"}, {"name": "two"}]}),
            encoding="utf-8",
        )
        assert [row["name"] for row in _stdlib_trace_events(trace)] == ["one", "two"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-completion", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print("self-test: PASS")
        return 0
    report = analyze_root(args.root, require_completion=args.verify_completion)
    final_paths = (
        args.root / "campaign-artifacts.sha256",
        args.root / "profile-complete.txt",
        args.root / "policy_a-profile-exit-status.txt",
    )
    if not args.verify_completion and not any(path.exists() for path in final_paths):
        atomic_write_text(
            args.root / "profile_analysis.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(args.root / "profile_analysis.md", render_markdown(report))
    else:
        print(
            "finalized campaign detected; signed analysis outputs were left unchanged"
        )
    if report["integrity"]["all_pass"]:
        print("analysis: PASS")
        return 0
    print(
        "analysis: FAIL: "
        + ", ".join(report["integrity"].get("failed_checks", ["unknown"]))
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

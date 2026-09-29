#!/usr/bin/env python3
"""CPU-only supplemental audit for the failed Policy A profiler campaign.

This script intentionally does not import torch, touch /dev/kfd, launch a GPU
process, or modify any original campaign artifact.  It corrects two analyzer
contracts in a separate report:

* source-audit harness keys are compared after campaign-root path normalization;
* pairing evidence is compared within the 640-sample route pair and within the
  1,024-sample formal pair, rather than across unlike workloads.

The two formal traces are streamed from their raw JSON files.  GPU collectives
are decoded from the trace event ``Collective name`` argument.
"""

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
except ImportError:  # pragma: no cover
    _ijson = None


ROOT = Path(__file__).resolve().parent
ARMS = (
    "smoke_mxfp4_policy_a",
    "replay_mxfp4_policy_a",
    "profile_bf16",
    "profile_mxfp4_policy_a",
)
ROUTE_ARMS = ("smoke_mxfp4_policy_a", "replay_mxfp4_policy_a")
FORMAL_ARMS = ("profile_bf16", "profile_mxfp4_policy_a")
PROFILE_STEPS = (7, 8)
PHYSICAL_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
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
MATRIX_RE = re.compile(r"^M=(\d+),N=(\d+),K=(\d+)$")


@dataclasses.dataclass
class Producer:
    name: str
    input_dims: Any
    input_strides: Any
    input_types: Any
    start_us: float
    end_us: float
    pid: Any
    tid: Any
    collective_name: str | None = None
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
    direct_collective_name: str | None
    collective_meta: dict[str, Any]
    producer: Producer | None = None
    collective_name: str | None = None
    collective_operation: str | None = None
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


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


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
        if producer.name in {"aten::mm", "aten::matmul", "aten::bmm"} and len(dims) >= 2:
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
                return f"M={math.prod(activation[:-1])},N={weight[0]},K={activation[-1]}"
    except (IndexError, TypeError, ValueError):
        return None
    return None


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
                for _ in iter(lambda: handle.read(1024 * 1024), ""):
                    pass
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
            for _ in iter(lambda: handle.read(1024 * 1024), b""):
                pass
    else:
        yield from _stdlib_trace_events(path)


def arg_value(args: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in args:
            return args[name]
    return None


def normalize_collective_name(value: Any) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower().replace("-", "_")
    if "allgather" in lowered or "all_gather" in lowered:
        return "all_gather"
    if "reduce_scatter" in lowered or "reducescatter" in lowered:
        return "reduce_scatter"
    if "allreduce" in lowered or "all_reduce" in lowered:
        return "all_reduce"
    if "broadcast" in lowered:
        return "broadcast"
    if lowered == "wait" or lowered.endswith("::wait"):
        return "wait"
    return "other"


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


def host_category(name: str, dims: Any, collective_name: Any = None) -> str | None:
    collective = normalize_collective_name(collective_name)
    if name == "record_param_comms" and collective:
        return f"collective_host_{collective}"
    kind = scope_kind(name)
    if kind:
        return kind
    lowered = name.lower()
    shapes = flattened_shapes(dims)
    if "parallelcrossentropy" in lowered or "crossentropy" in lowered:
        return "cross_entropy_backward" if "backward" in lowered else "cross_entropy_forward"
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


def assign_scopes(producers: dict[int, Producer], scopes: list[Scope]) -> None:
    scopes_by_thread: dict[tuple[Any, Any], list[Scope]] = collections.defaultdict(list)
    producers_by_thread: dict[tuple[Any, Any], list[Producer]] = collections.defaultdict(list)
    for scope in scopes:
        scopes_by_thread[(scope.pid, scope.tid)].append(scope)
    for producer in producers.values():
        producers_by_thread[(producer.pid, producer.tid)].append(producer)
    for thread, thread_producers in producers_by_thread.items():
        timeline: list[tuple[float, int, int, Any]] = []
        for identity, scope in enumerate(scopes_by_thread.get(thread, [])):
            timeline.extend(
                ((scope.start_us, 0, identity, scope), (scope.end_us, 2, identity, scope))
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


def _matrix_values(shape: str) -> tuple[int, int, int] | None:
    match = MATRIX_RE.fullmatch(shape)
    return tuple(int(match.group(index)) for index in (1, 2, 3)) if match else None


def is_a4w4(event: GPUEvent) -> bool:
    producer = event.producer.name if event.producer else ""
    text = f"{event.name} {producer}".lower()
    return any(
        token in text
        for token in ("f4gemm_bf16_per1x32fp4", "gemm_afp4wfp4", "gemm_a4w4")
    )


def a4w4_phase(event: GPUEvent, token_rows: int) -> str:
    scope = event.producer.scope if event.producer else None
    if scope == "packed_qkv_forward":
        return "packed_qkv_forward"
    values = _matrix_values(event.shape)
    is_wgrad = bool(values and values[2] == token_rows and values[0] != token_rows)
    if scope == "packed_qkv_backward":
        return "packed_qkv_wgrad" if is_wgrad else "packed_qkv_dgrad"
    if scope == "quantized_linear_forward":
        return "forward"
    if scope == "quantized_linear_backward":
        return "wgrad" if is_wgrad else "dgrad"
    return "wgrad" if is_wgrad else "unresolved_phase"


def collective_operation(event: GPUEvent) -> str | None:
    if event.collective_operation:
        return event.collective_operation
    producer = event.producer.name if event.producer else ""
    text = f"{event.name} {producer}".lower()
    if any(token in text for token in ("nccl", "rccl", "record_param_comms")):
        return normalize_collective_name(text) or "other"
    return None


def classify_event(event: GPUEvent, *, precision: str, token_rows: int) -> str:
    name = event.name.lower()
    producer = event.producer.name.lower() if event.producer else ""
    scope = event.producer.scope if event.producer else None
    text = f"{name} {producer}"
    shapes = flattened_shapes(event.producer.input_dims) if event.producer else []
    operation = collective_operation(event)
    if operation and operation != "wait":
        return f"collective_{operation}"
    if event.trace_category in {"gpu_memcpy", "gpu_memset"}:
        return (
            "packed_qkv_split_cat_copy"
            if scope and scope.startswith("packed_qkv")
            else "copy_memset"
        )
    if is_a4w4(event):
        return f"a4w4_{a4w4_phase(event, token_rows)}"
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
    if any(token in text for token in ("quantize_weight", "shuffle_weight", "weight_quant")):
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
    if scope and scope.startswith("packed_qkv") and any(
        token in text for token in ("cat", "split", "copy", "contiguous")
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
    if producer in {"aten::mm", "aten::matmul", "aten::bmm", "aten::addmm"} or name.startswith("cijk_"):
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
        token in name for token in ("add_kernel", "mul_kernel", "gelu", "residual", "elementwise")
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


def load_cache(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    choices = payload.get("choices", {})
    profiles = payload.get("profiles", {})
    selected: dict[str, Any] = {}
    errors: list[str] = []
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        errors.append("sha256_mismatch")
    if payload.get("schema") != 6 or payload.get("arch") != "gfx950":
        errors.append("schema_or_arch")
    if not isinstance(choices, dict) or not isinstance(profiles, dict) or set(choices) != set(profiles):
        errors.append("choice_profile_keys")
    for shape, backend in choices.items():
        identity = profiles.get(shape, {}).get("identities", {}).get(backend)
        if not isinstance(identity, dict):
            errors.append(f"missing_identity:{shape}:{backend}")
            identity = {}
        selected[shape] = {"backend": backend, **identity}
    return {
        "path": str(path),
        "sha256": digest,
        "schema": payload.get("schema"),
        "arch": payload.get("arch"),
        "choice_count": len(choices),
        "choices": choices,
        "selected_identities": selected,
        "valid": not errors,
        "errors": errors,
    }


def cache_identity(cache: dict[str, Any], shape: str) -> dict[str, Any] | None:
    match = MATRIX_RE.fullmatch(shape)
    if not match:
        return None
    return cache.get("selected_identities", {}).get(",".join(match.groups()))


def annotation_summary(
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


def parse_training_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    step_rows = [
        {
            "step": int(match.group(1)),
            "max_steps": int(match.group(2)),
            "loss": float(match.group(3)),
            "grad_norm": float(match.group(4)),
            "lr": float(match.group(5)),
            "step_time_ms": float(match.group(6)),
            "peak_mem_gib": float(match.group(7)),
        }
        for match in STEP_RE.finditer(text)
    ]
    batch_matches = PAIRING_RE.findall(text)
    validation_matches = VALIDATION_RE.findall(text)
    batch_counts = collections.Counter(int(rank) for rank, *_ in batch_matches)
    validation_counts = collections.Counter(int(rank) for rank, *_ in validation_matches)
    batches = {
        int(rank): {
            "step": int(step),
            "microbatches": int(microbatches),
            "sha256": digest,
        }
        for rank, step, microbatches, digest in batch_matches
    }
    validations = {
        int(rank): {"batches": int(count), "sha256": digest}
        for rank, count, digest in validation_matches
    }
    validation_losses = [
        float(value) for value in re.findall(r"\|\s*val_loss\s+([-+0-9.eE]+)", text)
    ]
    step_positions = {
        step: text.find(f"step {step}/") for step in PROFILE_STEPS
    }
    marker_positions = {
        "armed": text.find("> Profiler armed:"),
        "step7": step_positions[7],
        "step8": step_positions[8],
        "wrote": text.find("> Profiler wrote "),
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "model_init_hashes": sorted(set(re.findall(r"model_init_sha256=([0-9a-f]{64})", text))),
        "first_update": batches,
        "first_update_duplicate_ranks": sorted(rank for rank, count in batch_counts.items() if count != 1),
        "validation": validations,
        "validation_duplicate_ranks": sorted(rank for rank, count in validation_counts.items() if count != 1),
        "validation_nll": validation_losses[-1] if validation_losses else None,
        "validation_loss_count": len(validation_losses),
        "steps": step_rows,
        "step_times_ms": {row["step"]: row["step_time_ms"] for row in step_rows},
        "profiler_marker_positions": marker_positions,
        "profiler_marker_order": 0 <= marker_positions["armed"] < marker_positions["step7"] < marker_positions["step8"] < marker_positions["wrote"],
        "training_complete": bool(re.search(r"Training complete after \d+ steps\.", text)),
    }


def normalize_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--mode" and index + 1 < len(tokens):
            output.extend((token, "<precision>"))
            index += 2
        elif token in {"--mxfp4-pack-qkv", "--mxfp4-fuse-swiglu"}:
            index += 1
        else:
            output.append(token)
            index += 1
    return output


def formal_command_contract(left: str, right: str) -> dict[str, Any]:
    left_tokens = shlex.split(left)
    right_tokens = shlex.split(right)

    def values(tokens: list[str], flag: str) -> list[str]:
        return [tokens[index + 1] for index, token in enumerate(tokens[:-1]) if token == flag]

    checks = {
        "normalized_equal": normalize_command(left) == normalize_command(right),
        "bf16_mode": values(left_tokens, "--mode") == ["bf16"],
        "policy_a_mode": values(right_tokens, "--mode") == ["mxfp4"],
        "same_formal_train_samples": values(left_tokens, "--train-samples") == values(right_tokens, "--train-samples") == ["1024"],
        "same_tail": values(left_tokens, "--num-layers-at-end-in-bf16") == values(right_tokens, "--num-layers-at-end-in-bf16") == ["1"],
        "policy_only_flags": left_tokens.count("--mxfp4-pack-qkv") == 0
        and left_tokens.count("--mxfp4-fuse-swiglu") == 0
        and right_tokens.count("--mxfp4-pack-qkv") == 1
        and right_tokens.count("--mxfp4-fuse-swiglu") == 1,
    }
    return {"pass": all(checks.values()), "checks": checks}


def pair_evidence(
    left_name: str,
    right_name: str,
    logs: dict[str, dict[str, Any]],
    run_meta: dict[str, dict[str, str]],
    expected_samples: int,
) -> dict[str, Any]:
    left = logs[left_name]
    right = logs[right_name]
    rank_set = set(range(8))
    checks = {
        "train_samples": run_meta[left_name].get("train_samples")
        == run_meta[right_name].get("train_samples")
        == str(expected_samples),
        "single_matching_model_init": len(left["model_init_hashes"]) == 1
        and left["model_init_hashes"] == right["model_init_hashes"],
        "first_update_shape": set(left["first_update"]) == set(right["first_update"]) == rank_set
        and not left["first_update_duplicate_ranks"]
        and not right["first_update_duplicate_ranks"]
        and all(row["step"] == 1 and row["microbatches"] == 8 for row in left["first_update"].values())
        and all(row["step"] == 1 and row["microbatches"] == 8 for row in right["first_update"].values()),
        "first_update_digests_equal": left["first_update"] == right["first_update"],
        "validation_shape": set(left["validation"]) == set(right["validation"]) == rank_set
        and not left["validation_duplicate_ranks"]
        and not right["validation_duplicate_ranks"]
        and all(row["batches"] == 16 for row in left["validation"].values())
        and all(row["batches"] == 16 for row in right["validation"].values()),
        "validation_digests_equal": left["validation"] == right["validation"],
    }
    return {
        "arms": [left_name, right_name],
        "expected_train_samples": expected_samples,
        "pass": all(checks.values()),
        "checks": checks,
        "model_init": {left_name: left["model_init_hashes"], right_name: right["model_init_hashes"]},
        "first_update": {left_name: left["first_update"], right_name: right["first_update"]},
        "validation": {left_name: left["validation"], right_name: right["validation"]},
        "validation_nll": {left_name: left["validation_nll"], right_name: right["validation_nll"]},
    }


def normalized_source_audit(root: Path, meta: dict[str, str], audit: dict[str, Any]) -> dict[str, Any]:
    expected_files = {
        "runner_sha256": root / "run_policy_a_profile.sh",
        "route_entry_sha256": root / "route_entry.py",
        "analyzer_sha256": root / "analyze_policy_a_profile.py",
        "analyzer_test_sha256": root / "test_analyze_policy_a_profile.py",
        "protocol_sha256": root / "protocol.md",
        "test_sha256": root / "test_runner.py",
    }
    normalized: dict[Path, dict[str, Any]] = {}
    collisions: list[str] = []
    relative_keys: list[str] = []
    for recorded_key, recorded_digest in audit.get("harness_sha256", {}).items():
        recorded_path = Path(recorded_key)
        if not recorded_path.is_absolute():
            relative_keys.append(recorded_key)
            recorded_path = root / recorded_path
        resolved = recorded_path.resolve()
        if resolved in normalized:
            collisions.append(str(resolved))
        normalized[resolved] = {
            "recorded_key": recorded_key,
            "recorded_sha256": recorded_digest,
            "actual_sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    expected_resolved = {path.resolve() for path in expected_files.values()}
    harness_rows: dict[str, Any] = {}
    for meta_key, path in expected_files.items():
        row = normalized.get(path.resolve(), {})
        harness_rows[meta_key] = {
            "path": str(path.resolve()),
            "recorded_key": row.get("recorded_key"),
            "audit_sha256": row.get("recorded_sha256"),
            "meta_sha256": meta.get(meta_key),
            "actual_sha256": row.get("actual_sha256"),
            "pass": bool(row)
            and row.get("recorded_sha256") == row.get("actual_sha256") == meta.get(meta_key),
        }
    fixed_fields = {
        "schema": "1",
        "arms": ",".join(ARMS),
        "formal_arms": ",".join(FORMAL_ARMS),
        "formal_steps": "8",
        "profile_steps": "7,8",
        "nproc": "8",
        "micro_batch": "2",
        "global_batch": "128",
        "gradient_accumulation": "8",
        "sequence_length": "8192",
        "formal_train_samples": "1024",
        "eval_batches": "16",
        "val_samples": "256",
        "seed": "1234",
    }
    checks = {
        "fixed_profile_meta": all(meta.get(key) == value for key, value in fixed_fields.items()),
        "source_audit_sha256": meta.get("source_audit_sha256") == sha256_file(root / "source-audit.json"),
        "normalized_harness_path_set": set(normalized) == expected_resolved,
        "no_normalization_collisions": not collisions,
        "harness_hashes": all(row["pass"] for row in harness_rows.values()),
        "exactly_one_relative_key": relative_keys == ["run_policy_a_profile.sh"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "normalization_rule": "Absolute keys are resolved directly; relative keys are resolved against the campaign root before identity and hash comparison.",
        "relative_keys": relative_keys,
        "collisions": collisions,
        "harness": harness_rows,
    }


def scan_trace(
    path: Path,
    log: dict[str, Any],
    *,
    precision: str,
    cache: dict[str, Any],
) -> dict[str, Any]:
    trace_sha256 = sha256_file(path)
    trace_spans: list[tuple[float, float]] = []
    profiler_instants: list[float] = []
    physical: list[GPUEvent] = []
    physical_devices: set[Any] = set()
    physical_pids: set[Any] = set()
    runtime_events: list[tuple[float, float, str, int | None]] = []
    runtime_by_correlation: dict[int, tuple[float, float, str]] = {}
    parsed_events = 0
    for raw in iter_trace_events(path):
        parsed_events += 1
        phase = str(raw.get("ph", ""))
        category = str(raw.get("cat", ""))
        name = str(raw.get("name", ""))
        args = raw.get("args") or {}
        if category == "Trace" and phase == "X" and name == "PyTorch Profiler (0)":
            start = float(raw["ts"])
            trace_spans.append((start, start + float(raw["dur"])))
        if phase == "i" and name == "Iteration Start: PyTorch Profiler":
            profiler_instants.append(float(raw["ts"]))
        if category in {"cuda_runtime", "hip_runtime"} and phase == "X":
            start = float(raw["ts"])
            end = start + float(raw.get("dur", 0.0))
            raw_correlation = args.get("correlation")
            correlation = int(raw_correlation) if raw_correlation is not None else None
            runtime_events.append((start, end, name, correlation))
            if correlation is not None:
                runtime_by_correlation[correlation] = (start, end, name)
        if category not in PHYSICAL_CATEGORIES or phase != "X":
            continue
        physical_devices.add(args.get("device"))
        physical_pids.add(raw.get("pid"))
        if args.get("device") != 0:
            continue
        duration = float(raw.get("dur", 0.0))
        if duration <= 0:
            continue
        raw_external = arg_value(args, "External id", "External ID")
        raw_correlation = args.get("correlation")
        start = float(raw["ts"])
        direct_collective = arg_value(args, "Collective name", "Collective Name")
        collective_meta = {
            "dtype": args.get("dtype"),
            "in_msg_nelems": args.get("In msg nelems"),
            "out_msg_nelems": args.get("Out msg nelems"),
            "group_size": args.get("Group size"),
            "process_group": args.get("Process Group Name"),
        }
        physical.append(
            GPUEvent(
                start,
                start + duration,
                category,
                name,
                int(raw_external) if raw_external is not None else None,
                int(raw_correlation) if raw_correlation is not None else None,
                args.get("stream"),
                str(direct_collective) if direct_collective is not None else None,
                collective_meta,
            )
        )
    if len(trace_spans) != 1:
        raise AssertionError(f"{path}: expected one profiler span, found {len(trace_spans)}")
    trace_start, trace_end = trace_spans[0]

    direct_ids = {event.external_id for event in physical if event.external_id is not None}
    producers: dict[int, Producer] = {}
    duplicate_producer_ids: list[int] = []
    scopes: list[Scope] = []
    annotations: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    step_spans = {step: [] for step in PROFILE_STEPS}
    host_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    host_exact: dict[tuple[str, str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    second_pass_events = 0
    for raw in iter_trace_events(path):
        second_pass_events += 1
        if raw.get("ph") != "X" or raw.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        start = float(raw.get("ts", 0.0))
        end = start + float(raw.get("dur", 0.0))
        if end <= trace_start or start >= trace_end:
            continue
        name = str(raw.get("name", ""))
        args = raw.get("args") or {}
        dims = args.get("Input Dims")
        raw_external = arg_value(args, "External id", "External ID")
        collective_name = arg_value(args, "Collective name", "Collective Name")
        if raw_external is not None and int(raw_external) in direct_ids:
            external_id = int(raw_external)
            producer = Producer(
                name,
                dims,
                args.get("Input Strides"),
                args.get("Input type"),
                start,
                end,
                raw.get("pid"),
                raw.get("tid"),
                str(collective_name) if collective_name is not None else None,
            )
            if external_id in producers:
                duplicate_producer_ids.append(external_id)
            else:
                producers[external_id] = producer
        kind = scope_kind(name)
        if kind:
            scopes.append(Scope(start, end, raw.get("pid"), raw.get("tid"), kind))
        clipped = (max(start, trace_start), min(end, trace_end))
        if raw.get("cat") == "user_annotation":
            annotations[name].append(clipped)
            match = re.fullmatch(r"LUMEN_TRAIN_STEP#(7|8)", name)
            if match:
                step_spans[int(match.group(1))].append(clipped)
        selected = host_category(name, dims, collective_name)
        if selected:
            host_intervals[selected].append(clipped)
            host_exact[(selected, name, compact_dims(dims))].append(clipped)
    if second_pass_events != parsed_events:
        raise AssertionError(f"{path}: inconsistent trace event counts across passes")
    assign_scopes(producers, scopes)

    outside_window = 0
    unresolved_present_external = 0
    clipped_events: list[GPUEvent] = []
    missing_external: dict[tuple[str, str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    collective_resolution_counts: collections.Counter[str] = collections.Counter()
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
        producer_collective = event.producer.collective_name if event.producer else None
        if event.direct_collective_name is not None:
            event.collective_name = event.direct_collective_name
            collective_resolution_counts["physical_event_collective_name"] += 1
        elif producer_collective is not None:
            event.collective_name = producer_collective
            collective_resolution_counts["cpu_producer_collective_name"] += 1
        event.collective_operation = normalize_collective_name(event.collective_name)
        event.shape = matrix_shape(event.producer) or compact_dims(
            event.producer.input_dims if event.producer else None
        )
        event.category = classify_event(event, precision=precision, token_rows=16384)
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

    detail_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    broad_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    exact_intervals: dict[tuple[str, str, str, str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    matrix_intervals: dict[tuple[str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    kernel_intervals: dict[tuple[str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    collective_intervals: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    collective_exact: dict[tuple[str, str, str, str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    for event in clipped_events:
        interval = (event.start_us, event.end_us)
        detail_intervals[event.category].append(interval)
        broad_intervals[event.broad_category].append(interval)
        producer_name = event.producer.name if event.producer else "<unresolved>"
        scope = event.producer.scope if event.producer and event.producer.scope else "<none>"
        exact_intervals[(event.category, event.name, producer_name, event.shape, scope)].append(interval)
        if MATRIX_RE.fullmatch(event.shape):
            matrix_intervals[(event.category, event.shape)].append(interval)
        kernel_intervals[(event.name, event.category)].append(interval)
        if event.broad_category == "collectives":
            operation = event.collective_operation or "other"
            collective_intervals[operation].append(interval)
            meta = event.collective_meta
            collective_exact[
                (
                    operation,
                    str(meta.get("dtype")),
                    str(meta.get("in_msg_nelems")),
                    str(meta.get("out_msg_nelems")),
                    str(meta.get("group_size")),
                )
            ].append(interval)

    detail_metrics = {
        name: summarize_intervals(rows) for name, rows in sorted(detail_intervals.items())
    }
    broad_metrics = {
        name: summarize_intervals(rows) for name, rows in sorted(broad_intervals.items())
    }
    exact_rows: list[dict[str, Any]] = []
    a4w4_rows: list[dict[str, Any]] = []
    for (category, kernel, producer_name, shape, scope), rows in exact_intervals.items():
        row: dict[str, Any] = {
            "category": category,
            "broad_category": broad_category(category),
            "kernel": kernel,
            "producer": producer_name,
            "shape": shape,
            "scope": scope,
            **summarize_intervals(rows),
        }
        if category.startswith("a4w4_"):
            identity = cache_identity(cache, shape)
            if identity:
                selected_symbol = identity.get("kernel_name") or identity.get("entrypoint")
                row.update(
                    selected_backend=identity.get("backend"),
                    selected_symbol=selected_symbol,
                    selected_tile=(
                        f"{identity.get('tile_m')}x{identity.get('tile_n')}"
                        if identity.get("tile_m") is not None
                        else None
                    ),
                    selected_log2_k_split=identity.get("log2_k_split"),
                    selected_manifest_sha256=identity.get("manifest_sha256"),
                    selected_code_object_sha256=identity.get("code_object_sha256"),
                    selected_identity_matches_observed_kernel=bool(selected_symbol and selected_symbol in kernel),
                )
            a4w4_rows.append(dict(row))
        exact_rows.append(row)
    exact_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["kernel"], row["shape"]))
    a4w4_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["shape"], row["category"]))
    matrix_rows = [
        {"category": category, "shape": shape, **summarize_intervals(rows)}
        for (category, shape), rows in matrix_intervals.items()
    ]
    matrix_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["category"], row["shape"]))
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
    host_exact_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["category"], row["operator"]))

    runtime_by_api: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    runtime_by_kind: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
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
        {"api": name, **summarize_intervals(rows)} for name, rows in runtime_by_api.items()
    ]
    runtime_api_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["api"]))
    runtime_categories = {
        name: summarize_intervals(rows) for name, rows in sorted(runtime_by_kind.items())
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

    collective_all = [interval for rows in collective_intervals.values() for interval in rows]
    noncollective = [
        interval
        for category, rows in broad_intervals.items()
        if category != "collectives"
        for interval in rows
    ]
    collective_union_us = interval_union(collective_all)
    collective_exposed_us = max(
        0.0, interval_union(collective_all + noncollective) - interval_union(noncollective)
    )
    collective_by_operation: dict[str, Any] = {}
    for operation, rows in sorted(collective_intervals.items()):
        others = [
            interval
            for other_name, other_rows in collective_intervals.items()
            if other_name != operation
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
        collective_by_operation[operation] = summary
    collective_exact_rows = [
        {
            "operation": operation,
            "dtype": dtype,
            "in_msg_nelems": in_nelems,
            "out_msg_nelems": out_nelems,
            "group_size": group_size,
            **summarize_intervals(rows),
        }
        for (operation, dtype, in_nelems, out_nelems, group_size), rows in collective_exact.items()
    ]
    collective_exact_rows.sort(key=lambda row: (-float(row["raw_total_ms"]), row["operation"]))

    all_intervals = [(event.start_us, event.end_us) for event in clipped_events]
    envelope_start = min(left for left, _ in all_intervals)
    envelope_end = max(right for _, right in all_intervals)
    envelope_us = envelope_end - envelope_start
    busy_us = interval_union(all_intervals)
    raw_us = sum(right - left for left, right in all_intervals)
    trace_span_us = trace_end - trace_start

    annotations_summary = annotation_summary(annotations, step_spans)
    log_span_ms = sum(log["step_times_ms"].get(step, math.nan) for step in PROFILE_STEPS)
    trace_span_ms = trace_span_us / 1000.0
    trace_log_error_ms = abs(trace_span_ms - log_span_ms)
    step_order = (
        all(len(step_spans[step]) == 1 for step in PROFILE_STEPS)
        and step_spans[7][0][0] < step_spans[7][0][1] <= step_spans[8][0][0] < step_spans[8][0][1]
    )
    profiler_order = (
        len(profiler_instants) == 1
        and step_order
        and abs(profiler_instants[0] - trace_start) < 0.001
        and step_spans[8][0][1] <= trace_end
    )
    microbatch_contract = (
        all(
            annotations_summary[f"{name}_total"] == 16
            for name in ("dataloader", "root_forward", "pre_backward", "root_post_backward")
        )
        and all(
            annotations_summary[f"{name}_by_step"] == {"7": 8, "8": 8}
            for name in ("dataloader", "root_forward", "pre_backward", "root_post_backward")
        )
        and annotations_summary["optimizer_step_by_step"] == {"7": 1, "8": 1}
        and annotations_summary["zero_grad_by_step"] == {"7": 1, "8": 1}
    )
    physical_collective_count = sum(
        len(rows) for operation, rows in collective_intervals.items() if operation != "wait"
    )
    contract_checks = {
        "one_profiler_span": len(trace_spans) == 1,
        "one_profiler_start_instant": len(profiler_instants) == 1,
        "exact_step_annotations": all(len(step_spans[step]) == 1 for step in PROFILE_STEPS),
        "profiler_start_step7_step8_stop_order": profiler_order and log["profiler_marker_order"],
        "trace_log_span_within_3ms": math.isfinite(trace_log_error_ms) and trace_log_error_ms <= 3.0,
        "rank0_device0_physical_only": physical_devices == {0} and physical_pids == {0},
        "physical_events_inside_profiler_span": outside_window == 0,
        "microbatch_forward_backward_exact": microbatch_contract,
        "producer_external_ids_unique": not duplicate_producer_ids,
        "trace_event_pass_counts_match": second_pass_events == parsed_events,
        "collectives_use_direct_collective_name": collective_resolution_counts["physical_event_collective_name"] == physical_collective_count,
        "collectives_fully_decoded": physical_collective_count > 0
        and "other" not in collective_intervals
        and "wait" not in collective_intervals,
    }
    if precision == "mxfp4":
        contract_checks.update(
            a4w4_identities_resolved=bool(a4w4_rows)
            and all(row.get("selected_identity_matches_observed_kernel") for row in a4w4_rows),
            expected_quantized_linear_host_calls=host_categories.get("quantized_linear_forward", {}).get("calls_per_step") == 1120
            and host_categories.get("quantized_linear_backward", {}).get("calls_per_step") == 1120,
            expected_packed_qkv_host_calls=host_categories.get("packed_qkv_forward", {}).get("calls_per_step") == 280
            and host_categories.get("packed_qkv_backward", {}).get("calls_per_step") == 280,
        )
    else:
        contract_checks.update(
            no_a4w4_gpu_work=not a4w4_rows,
            no_mxfp4_gpu_category="mxfp4_quant_layout" not in broad_metrics,
        )

    missing_rows = [
        {
            "category": category,
            "kernel": kernel,
            "runtime_resolution": runtime_name,
            **summarize_intervals(rows),
        }
        for (category, kernel, runtime_name), rows in missing_external.items()
    ]
    missing_rows.sort(key=lambda row: (-int(row["calls_total"]), row["kernel"]))
    joint = broad_intervals.get("a4w4_gemm", []) + broad_intervals.get("mxfp4_quant_layout", [])
    return {
        "path": str(path),
        "sha256": trace_sha256,
        "stream_parser": "ijson" if _ijson is not None else "stdlib-jsondecoder",
        "trace_event_count": parsed_events,
        "trace_event_stream_exhausted": True,
        "whole_file_sha256_read_to_eof": True,
        "contract": {
            "all_pass": all(contract_checks.values()),
            "checks": contract_checks,
            "evidence": {
                "physical_devices": sorted(str(value) for value in physical_devices),
                "physical_pids": sorted(str(value) for value in physical_pids),
                "outside_window_count": outside_window,
                "duplicate_producer_ids": sorted(set(duplicate_producer_ids)),
                "trace_log_span_error_ms": trace_log_error_ms,
                "collective_resolution_counts": dict(sorted(collective_resolution_counts.items())),
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
            "profiler_nonbusy_total_ms": (trace_span_us - busy_us) / 1000.0,
            "profiler_nonbusy_per_step_ms": (trace_span_us - busy_us) / 2000.0,
            "profiler_edge_outside_gpu_envelope_total_ms": (trace_span_us - envelope_us) / 1000.0,
            "profiler_edge_outside_gpu_envelope_per_step_ms": (trace_span_us - envelope_us) / 2000.0,
            "overlap_raw_minus_busy_total_ms": (raw_us - busy_us) / 1000.0,
            "overlap_raw_minus_busy_per_step_ms": (raw_us - busy_us) / 2000.0,
            "physical_events_total": len(clipped_events),
        },
        "annotations": annotations_summary,
        "detail_categories": detail_metrics,
        "broad_categories": broad_metrics,
        "matrix_shapes": matrix_rows,
        "top_kernels": top_kernels[:100],
        "exact_gpu_rows": exact_rows,
        "a4w4_exact_shapes": a4w4_rows,
        "joint_unions": {
            "a4w4_plus_quant_layout": {
                **summarize_intervals(joint),
                "interpretation": "Overlap-safe impossible-zero-cost ceiling, not a predicted saving.",
            }
        },
        "collectives": {
            "decoder": "GPU trace args['Collective name']; CPU producer field is fallback only.",
            "overall": {
                **summarize_intervals(collective_all),
                "exposed_total_ms": collective_exposed_us / 1000.0,
                "exposed_per_step_ms": collective_exposed_us / 2000.0,
                "overlapped_total_ms": (collective_union_us - collective_exposed_us) / 1000.0,
                "overlapped_per_step_ms": (collective_union_us - collective_exposed_us) / 2000.0,
            },
            "by_operation": collective_by_operation,
            "exact_messages": collective_exact_rows,
            "resolution_counts": dict(sorted(collective_resolution_counts.items())),
        },
        "host": {
            "categories": host_categories,
            "exact_rows": host_exact_rows,
            "runtime_categories": runtime_categories,
            "runtime_api_rows": runtime_api_rows,
            "hipModuleLaunchKernel": [row for row in runtime_api_rows if row["api"] == "hipModuleLaunchKernel"],
            "hipPointerGetAttribute": [row for row in runtime_api_rows if row["api"] == "hipPointerGetAttribute"],
            "launch_gaps": {
                "launch_calls_total": len(launches),
                "inter_launch_positive_gap_count": len(launch_gaps),
                "inter_launch_positive_gap_total_ms": sum(launch_gaps) / 1000.0,
                "inter_launch_gap_p50_us": percentile(launch_gaps, 0.5),
                "inter_launch_gap_p95_us": percentile(launch_gaps, 0.95),
                "inter_launch_gap_max_us": max(launch_gaps) if launch_gaps else None,
                "launch_to_device_gap_count": len(queue_gaps),
                "launch_to_device_gap_mean_us": sum(queue_gaps) / len(queue_gaps) if queue_gaps else None,
                "launch_to_device_gap_p50_us": percentile(queue_gaps, 0.5),
                "launch_to_device_gap_p95_us": percentile(queue_gaps, 0.95),
                "launch_to_device_gap_max_us": max(queue_gaps) if queue_gaps else None,
                "negative_gap_count": negative_queue_gaps,
            },
            "warning": "Overlapping and nested host spans are not additive with GPU time.",
        },
        "missing_external_id": {
            "events_total": sum(len(rows) for rows in missing_external.values()),
            "external_id_present_but_producer_unresolved": unresolved_present_external,
            "rows": missing_rows,
            "resolution_method": "Direct External id to CPU producer; absent ids retained, classified by kernel name/scope-independent rules, and grouped with the correlated HIP API.",
        },
        "unclassified_gpu_work": {
            **detail_metrics.get("other", ZERO_METRICS),
            "kernels": [row for row in top_kernels if row["category"] == "other"],
        },
    }


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    output: dict[str, Any] = {}
    for key in keys:
        left_row = left.get(key, ZERO_METRICS)
        right_row = right.get(key, ZERO_METRICS)
        output[key] = {
            "bf16": left_row,
            "policy_a": right_row,
            "policy_a_minus_bf16_calls_per_step": right_row["calls_per_step"] - left_row["calls_per_step"],
            "policy_a_minus_bf16_raw_per_step_ms": right_row["raw_per_step_ms"] - left_row["raw_per_step_ms"],
            "policy_a_minus_bf16_union_per_step_ms": right_row["union_per_step_ms"] - left_row["union_per_step_ms"],
        }
    return output


def build_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    meta = parse_kv(root / "profile-meta.txt")
    audit = read_json(root / "source-audit.json")
    original_analysis = read_json(root / "profile_analysis.json")
    run_meta = {arm: parse_kv(root / arm / "run-meta.txt") for arm in ARMS}
    logs = {arm: parse_training_log(root / arm / "train.log") for arm in ARMS}

    source = normalized_source_audit(root, meta, audit)
    route_pair = pair_evidence(*ROUTE_ARMS, logs, run_meta, expected_samples=640)
    formal_pair = pair_evidence(*FORMAL_ARMS, logs, run_meta, expected_samples=1024)
    model_sets = {arm: logs[arm]["model_init_hashes"] for arm in ARMS}
    validation_sets = {arm: logs[arm]["validation"] for arm in ARMS}
    all_model_init_equal = all(
        values == model_sets[ARMS[0]] and len(values) == 1 for values in model_sets.values()
    )
    all_validation_equal = all(values == validation_sets[ARMS[0]] for values in validation_sets.values())
    route_vs_formal_first_update_distinct = (
        logs[ROUTE_ARMS[0]]["first_update"] != logs[FORMAL_ARMS[0]]["first_update"]
    )
    route_command_equal = shlex.split(run_meta[ROUTE_ARMS[0]]["command"]) == shlex.split(run_meta[ROUTE_ARMS[1]]["command"])
    formal_command = formal_command_contract(
        run_meta[FORMAL_ARMS[0]]["command"], run_meta[FORMAL_ARMS[1]]["command"]
    )

    cache_paths = {
        "build": root / "cache/cache_build_policy_a/mxfp4-autotune.json",
        "bf16": root / "cache/profile_bf16/mxfp4-autotune.json",
        "policy_a": root / "cache/profile_mxfp4_policy_a/mxfp4-autotune.json",
    }
    caches = {name: load_cache(path, meta["cache_sha256"]) for name, path in cache_paths.items()}
    cache_bytes = {name: path.read_bytes() for name, path in cache_paths.items()}
    cache_identity_pass = all(cache["valid"] for cache in caches.values()) and all(
        value == cache_bytes["build"] for value in cache_bytes.values()
    )
    cache = caches["build"]

    traces = {
        "profile_bf16": scan_trace(
            root / "profile_bf16/trace.json", logs["profile_bf16"], precision="bf16", cache=cache
        ),
        "profile_mxfp4_policy_a": scan_trace(
            root / "profile_mxfp4_policy_a/trace.json",
            logs["profile_mxfp4_policy_a"],
            precision="mxfp4",
            cache=cache,
        ),
    }
    bf16_trace = traces["profile_bf16"]
    policy_trace = traces["profile_mxfp4_policy_a"]
    bf16_span = bf16_trace["window"]["profiler_span_per_step_ms"]
    policy_span = policy_trace["window"]["profiler_span_per_step_ms"]
    joint_union = policy_trace["joint_unions"]["a4w4_plus_quant_layout"]["union_per_step_ms"]
    differential = {
        "profile_window": {
            "bf16_profiler_span_per_step_ms": bf16_span,
            "policy_a_profiler_span_per_step_ms": policy_span,
            "diagnostic_speedup_bf16_over_policy_a": bf16_span / policy_span,
            "policy_a_target_for_1_6x_ms": bf16_span / 1.6,
            "policy_a_gap_to_1_6x_ms": policy_span - bf16_span / 1.6,
        },
        "broad_categories": metric_delta(
            bf16_trace["broad_categories"], policy_trace["broad_categories"]
        ),
        "detail_categories": metric_delta(
            bf16_trace["detail_categories"], policy_trace["detail_categories"]
        ),
        "amdahl_zero_cost_ceiling": {
            "policy_a_a4w4_plus_quant_union_per_step_ms": joint_union,
            "fraction_of_policy_a_profiler_span": joint_union / policy_span,
            "hypothetical_policy_a_span_without_joint_union_ms": policy_span - joint_union,
            "bf16_over_hypothetical_policy_a_ratio": bf16_span / (policy_span - joint_union),
            "interpretation": "Impossible-zero-cost ceiling based on interval unions; not a predicted speedup.",
        },
    }

    original_failed_checks = original_analysis.get("integrity", {}).get("failed_checks", [])
    original_preservation = {
        "policy_a_profile_exit_status": (root / "policy_a-profile-exit-status.txt").read_text(encoding="utf-8").strip(),
        "campaign_stage_status": parse_kv(root / "campaign-stage-status.txt"),
        "original_analysis_all_pass": original_analysis.get("integrity", {}).get("all_pass"),
        "original_analysis_failed_checks": original_failed_checks,
        "original_analysis_sha256": sha256_file(root / "profile_analysis.json"),
        "original_markdown_sha256": sha256_file(root / "profile_analysis.md"),
        "original_analyzer_sha256": sha256_file(root / "analyze_policy_a_profile.py"),
    }
    original_failure_preserved = (
        original_preservation["policy_a_profile_exit_status"] == "1"
        and original_preservation["campaign_stage_status"].get("failed_stage") == "analysis"
        and original_preservation["campaign_stage_status"].get("failed_status") == "1"
        and original_preservation["original_analysis_all_pass"] is False
        and set(original_failed_checks) == {"profile_meta", "source_audit", "paired_run_evidence"}
    )
    status_pass = all(
        parse_kv(root / arm / "train-exit-status.txt") == {"torchrun": "0", "tee": "0"}
        for arm in ARMS
    ) and all(logs[arm]["training_complete"] and logs[arm]["validation_loss_count"] == 1 for arm in ARMS)

    checks = {
        "original_fail_preserved": original_failure_preserved,
        "normalized_source_audit": source["pass"],
        "route_pair_640_samples": route_pair["pass"],
        "formal_pair_1024_samples": formal_pair["pass"],
        "all_model_initialization_hashes_match": all_model_init_equal,
        "all_validation_data_digests_match": all_validation_equal,
        "route_and_formal_first_updates_are_distinct_populations": route_vs_formal_first_update_distinct,
        "route_commands_match": route_command_equal,
        "formal_commands_match_after_allowed_precision_flags": formal_command["pass"],
        "cache_identity_and_copies": cache_identity_pass,
        "arm_exit_and_completion_status": status_pass,
        "bf16_trace_contract": bf16_trace["contract"]["all_pass"],
        "policy_a_trace_contract": policy_trace["contract"]["all_pass"],
    }
    return {
        "schema": 1,
        "report_kind": "supplemental_non_mutating_audit",
        "root": str(root),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "gpu_used": False,
            "torch_imported": False,
        },
        "scope": {
            "original_result": "FAIL (preserved)",
            "supplemental_result": "PASS" if all(checks.values()) else "FAIL",
            "claim_boundary": "Profile timing is diagnostic only; no end-to-end throughput claim is made.",
        },
        "supplemental_integrity": {
            "all_pass": all(checks.values()),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        },
        "original_failure": original_preservation,
        "source_audit_normalized": source,
        "pairing": {
            "route_smoke_replay": route_pair,
            "formal_bf16_policy_a": formal_pair,
            "all_model_initialization_hashes_match": all_model_init_equal,
            "all_validation_data_digests_match": all_validation_equal,
            "route_and_formal_first_updates_are_distinct_populations": route_vs_formal_first_update_distinct,
            "model_init_by_arm": model_sets,
            "validation_by_arm": validation_sets,
            "route_commands_match": route_command_equal,
            "formal_command_contract": formal_command,
        },
        "run_meta": run_meta,
        "logs": logs,
        "cache": {
            "all_copies_identical": cache_identity_pass,
            "copies": caches,
        },
        "traces": traces,
        "differential": differential,
        "accuracy": {
            "bf16_validation_nll": logs["profile_bf16"]["validation_nll"],
            "policy_a_validation_nll": logs["profile_mxfp4_policy_a"]["validation_nll"],
            "policy_a_minus_bf16": logs["profile_mxfp4_policy_a"]["validation_nll"] - logs["profile_bf16"]["validation_nll"],
            "interpretation": "Paired eight-step diagnostic on identical validation data; not a long-horizon convergence result.",
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source_audit_normalized"]
    pairing = report["pairing"]
    bf16 = report["traces"]["profile_bf16"]
    policy = report["traces"]["profile_mxfp4_policy_a"]
    diff = report["differential"]
    window = diff["profile_window"]
    accuracy = report["accuracy"]
    lines = [
        "# Supplemental BF16 / MXFP4 Policy A profile audit",
        "",
        f"Supplemental integrity result: **{report['scope']['supplemental_result']}**",
        "",
        "The original analyzer result remains **FAIL** and no original campaign artifact was rewritten. This supplemental audit corrects only the two known analyzer contracts and independently streams both raw traces on CPU.",
        "",
        "## Corrected integrity contracts",
        "",
        f"- Source audit after path normalization: **{'PASS' if source['pass'] else 'FAIL'}**. Relative key: `{', '.join(source['relative_keys'])}`; it resolves against the campaign root and its recorded digest matches the file and `profile-meta.txt`.",
        f"- 640-sample smoke/replay pairing: **{'PASS' if pairing['route_smoke_replay']['pass'] else 'FAIL'}**.",
        f"- 1,024-sample BF16/Policy A formal pairing: **{'PASS' if pairing['formal_bf16_policy_a']['pass'] else 'FAIL'}**.",
        f"- Model initialization hash matches across all four arms: **{'PASS' if pairing['all_model_initialization_hashes_match'] else 'FAIL'}**.",
        f"- Validation input/label digests match across all four arms: **{'PASS' if pairing['all_validation_data_digests_match'] else 'FAIL'}**.",
        "- The 640-sample and 1,024-sample first-update digest sets are different, as expected for different formal data extents; they are not compared as one pair.",
        "",
        "## Paired formal outcome",
        "",
        "| Metric | BF16 | MXFP4 Policy A | Delta / ratio |",
        "|---|---:|---:|---:|",
        f"| Profiler span per step (ms) | {fmt(window['bf16_profiler_span_per_step_ms'])} | {fmt(window['policy_a_profiler_span_per_step_ms'])} | speedup {fmt(window['diagnostic_speedup_bf16_over_policy_a'])}x |",
        f"| Validation NLL | {fmt(accuracy['bf16_validation_nll'], 6)} | {fmt(accuracy['policy_a_validation_nll'], 6)} | {fmt(accuracy['policy_a_minus_bf16'], 6)} |",
        f"| 1.6x target step time (ms) | n/a | {fmt(window['policy_a_target_for_1_6x_ms'])} | gap {fmt(window['policy_a_gap_to_1_6x_ms'])} ms |",
        "",
        "Profiler timing remains diagnostic evidence only.",
        "",
        "## Trace windows and GPU occupancy",
        "",
        "| Metric per step (ms) | BF16 | MXFP4 Policy A |",
        "|---|---:|---:|",
    ]
    window_rows = (
        ("Profiler span", "profiler_span_per_step_ms"),
        ("GPU envelope", "gpu_envelope_per_step_ms"),
        ("GPU busy interval union", "gpu_busy_union_per_step_ms"),
        ("GPU idle inside envelope", "gpu_idle_inside_envelope_per_step_ms"),
        ("Profiler non-busy", "profiler_nonbusy_per_step_ms"),
        ("Profiler edge outside GPU envelope", "profiler_edge_outside_gpu_envelope_per_step_ms"),
        ("Raw GPU duration sum", "gpu_raw_duration_sum_per_step_ms"),
        ("Raw-minus-busy overlap", "overlap_raw_minus_busy_per_step_ms"),
    )
    for label, key in window_rows:
        lines.append(f"| {label} | {fmt(bf16['window'][key])} | {fmt(policy['window'][key])} |")
    lines.extend(
        [
            "",
            "## Overlap-safe GPU categories",
            "",
            "| Category | BF16 raw ms/step | Policy A raw ms/step | BF16 union ms/step | Policy A union ms/step |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, row in diff["broad_categories"].items():
        lines.append(
            f"| `{category}` | {fmt(row['bf16']['raw_per_step_ms'])} | {fmt(row['policy_a']['raw_per_step_ms'])} | {fmt(row['bf16']['union_per_step_ms'])} | {fmt(row['policy_a']['union_per_step_ms'])} |"
        )
    lines.extend(
        [
            "",
            "## Collectives decoded from `Collective name`",
            "",
            "| Operation | BF16 calls/step | BF16 raw ms/step | Policy A calls/step | Policy A raw ms/step |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    operations = sorted(
        set(bf16["collectives"]["by_operation"]) | set(policy["collectives"]["by_operation"])
    )
    for operation in operations:
        left = bf16["collectives"]["by_operation"].get(operation, ZERO_METRICS)
        right = policy["collectives"]["by_operation"].get(operation, ZERO_METRICS)
        lines.append(
            f"| `{operation}` | {fmt(left['calls_per_step'])} | {fmt(left['raw_per_step_ms'])} | {fmt(right['calls_per_step'])} | {fmt(right['raw_per_step_ms'])} |"
        )
    lines.extend(
        [
            "",
            "All physical collective kernels in both traces were resolved directly from the GPU event argument; none remained in an `other` collective bucket.",
            "",
            "## MXFP4 A4W4 exact shapes",
            "",
            "| Phase | Shape | Calls/step | Raw ms/step | Union ms/step | Backend | Symbol / tile / split |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in policy["a4w4_exact_shapes"]:
        symbol = str(row.get("selected_symbol", "n/a")).split("::")[-1]
        lines.append(
            f"| `{row['category']}` | `{row['shape']}` | {fmt(row['calls_per_step'])} | {fmt(row['raw_per_step_ms'])} | {fmt(row['union_per_step_ms'])} | `{row.get('selected_backend', 'n/a')}` | `{symbol}` / `{row.get('selected_tile')}` / `{row.get('selected_log2_k_split')}` |"
        )
    lines.extend(
        [
            "",
            "## Selected host/runtime calls",
            "",
            "| Call/category | BF16 calls/step | BF16 raw ms/step | Policy A calls/step | Policy A raw ms/step |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    host_names = (
        "quantized_linear_forward",
        "quantized_linear_backward",
        "packed_qkv_forward",
        "packed_qkv_backward",
        "split_swiglu_forward",
        "split_swiglu_backward",
    )
    for name in host_names:
        left = bf16["host"]["categories"].get(name, ZERO_METRICS)
        right = policy["host"]["categories"].get(name, ZERO_METRICS)
        lines.append(
            f"| `{name}` | {fmt(left['calls_per_step'])} | {fmt(left['raw_per_step_ms'])} | {fmt(right['calls_per_step'])} | {fmt(right['raw_per_step_ms'])} |"
        )
    for name in ("kernel_launch", "pointer_query", "synchronization", "blocking_copy"):
        left = bf16["host"]["runtime_categories"].get(name, ZERO_METRICS)
        right = policy["host"]["runtime_categories"].get(name, ZERO_METRICS)
        lines.append(
            f"| HIP `{name}` | {fmt(left['calls_per_step'])} | {fmt(left['raw_per_step_ms'])} | {fmt(right['calls_per_step'])} | {fmt(right['raw_per_step_ms'])} |"
        )
    amdahl = diff["amdahl_zero_cost_ceiling"]
    lines.extend(
        [
            "",
            "## Missing External id and Amdahl ceiling",
            "",
            f"- BF16 missing-External-id GPU events: {bf16['missing_external_id']['events_total']}.",
            f"- Policy A missing-External-id GPU events: {policy['missing_external_id']['events_total']}; all are retained and resolved by kernel-name rules plus correlated HIP API grouping.",
            f"- A4W4 plus quant/layout union: {fmt(amdahl['policy_a_a4w4_plus_quant_union_per_step_ms'])} ms/step ({fmt(100 * amdahl['fraction_of_policy_a_profiler_span'], 2)}% of the Policy A profiler span).",
            f"- Impossible-zero-cost BF16-over-Policy-A ceiling: {fmt(amdahl['bf16_over_hypothetical_policy_a_ratio'])}x.",
            "",
            "## Preservation and hashes",
            "",
            f"- Original analyzer JSON SHA256: `{report['original_failure']['original_analysis_sha256']}`",
            f"- Original analyzer Markdown SHA256: `{report['original_failure']['original_markdown_sha256']}`",
            f"- Original analyzer source SHA256: `{report['original_failure']['original_analyzer_sha256']}`",
            f"- Supplemental script SHA256: `{report['script']['sha256']}`",
            "- Original exit sentinel remains `1`; campaign stage remains `failed_stage=analysis`, `failed_status=1`.",
            "",
            "## Limits and risks",
            "",
            "- The traces cover optimizer steps 7 and 8 only. No `mxfp4_weight_quant` GPU event appears in that window, so this audit does not estimate one-time initialization, cache-build, or pre-window weight-quantization cost.",
            "- The GPU kernel-name `split_swiglu` bucket has 288 calls/step while the Policy A custom split-SwiGLU host scope has 280 calls/step; the extra eight calls/step come from work outside the 35 quantized-layer custom scope (including the protected BF16 tail), so the GPU bucket must not be read as a quantized-layer count.",
            "- The 1,540 Policy A events without `External id` are name-resolved helper kernels. Their categories are reliable at kernel-family level, but they have no producer shape/scope attribution.",
            "- These supplemental files are deliberately outside the original failed campaign's completion manifest and are not a replacement for a signed zero-exit campaign result.",
            "",
            "Raw duration sums and per-category unions can overlap across streams/categories. They are diagnostic and must not be added to predict step-time savings.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.root)
    if not args.no_write:
        atomic_write_text(
            args.root / "supplemental_audit.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(args.root / "supplemental_audit.md", render_markdown(report))
    print(f"supplemental_integrity={'PASS' if report['supplemental_integrity']['all_pass'] else 'FAIL'}")
    print("original_integrity=FAIL (preserved)")
    print(f"bf16_trace_sha256={report['traces']['profile_bf16']['sha256']}")
    print(f"policy_a_trace_sha256={report['traces']['profile_mxfp4_policy_a']['sha256']}")
    return 0 if report["supplemental_integrity"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

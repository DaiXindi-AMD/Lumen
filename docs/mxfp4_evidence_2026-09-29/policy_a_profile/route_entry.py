#!/usr/bin/env python3
"""Run Qwen3 training while collecting rank-local MXFP4 route evidence."""

from __future__ import annotations

import json
import os
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

import torch.nn as nn


TRAIN_ENTRY = Path("/home/xdai/Lumen/examples/qwen3/train_qwen3_fsdp.py")


def _shape_log_target() -> str:
    template = os.environ.pop("LUMEN_MXFP4_GEMM_SHAPE_LOG_TEMPLATE", "")
    if not template:
        return ""
    target = template.format(rank=int(os.environ["RANK"]))
    os.environ["LUMEN_MXFP4_GEMM_SHAPE_LOG"] = target
    return target


def _install_counters() -> tuple[Counter[str], dict[str, Any]]:
    import aiter.ops.triton.activation as activation
    import lumen.models.qwen3 as qwen3
    import lumen.ops.quantize.linear as linear

    counts: Counter[str] = Counter()
    details: dict[str, Any] = {}

    original_swiglu_fwd = activation.swiglu_fwd_split
    original_swiglu_bwd = activation.swiglu_bwd_split
    original_qkv_linear = linear.mxfp4_qkv_linear

    def counted_swiglu_fwd(*args, **kwargs):
        try:
            result = original_swiglu_fwd(*args, **kwargs)
        except BaseException:
            counts["swiglu_fwd_failure"] += 1
            raise
        counts["swiglu_fwd_success"] += 1
        return result

    def counted_swiglu_bwd(*args, **kwargs):
        try:
            result = original_swiglu_bwd(*args, **kwargs)
        except BaseException:
            counts["swiglu_bwd_failure"] += 1
            raise
        counts["swiglu_bwd_success"] += 1
        return result

    def counted_qkv_linear(*args, **kwargs):
        try:
            result = original_qkv_linear(*args, **kwargs)
        except BaseException:
            counts["qkv_linear_failure"] += 1
            raise
        counts["qkv_linear_success"] += 1
        return result

    activation.swiglu_fwd_split = counted_swiglu_fwd
    activation.swiglu_bwd_split = counted_swiglu_bwd
    linear.mxfp4_qkv_linear = counted_qkv_linear

    original_enable_swiglu = qwen3.enable_mxfp4_qwen_swiglu

    def counted_enable_swiglu(model, *args, **kwargs):
        enabled = original_enable_swiglu(model, *args, **kwargs)
        counts["enabled_swiglu"] += int(enabled)
        for module in model.modules():
            original = getattr(module, "_mxfp4_swiglu_original_forward", None)
            if original is None:
                continue

            def counted_original(*call_args, _original=original, **call_kwargs):
                counts["eligible_original_swiglu_forward"] += 1
                return _original(*call_args, **call_kwargs)

            module._mxfp4_swiglu_original_forward = counted_original
        return enabled

    original_enable_qkv = qwen3.enable_mxfp4_qwen_qkv

    def counted_enable_qkv(model, *args, **kwargs):
        if "unquantized_linear_names" not in details:
            unquantized = [
                (name, module)
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and not bool(getattr(module, "_quant_enabled", False))
            ]
            details["unquantized_linear_names"] = [name for name, _ in unquantized]
            lm_heads = [module for name, module in unquantized if name == "lm_head"]
            details["lm_head_count"] = len(lm_heads)
            if len(lm_heads) == 1:
                lm_head = lm_heads[0]
                details["lm_head_weight_dtype"] = str(
                    lm_head.weight.dtype
                ).removeprefix("torch.")
                details["lm_head_quant_enabled"] = bool(
                    getattr(lm_head, "_quant_enabled", False)
                )

        enabled = original_enable_qkv(model, *args, **kwargs)
        counts["enabled_qkv"] += int(enabled)
        for module in model.modules():
            original = getattr(module, "_mxfp4_qkv_original_forward", None)
            if original is None:
                continue

            def counted_original(*call_args, _original=original, **call_kwargs):
                counts["eligible_original_qkv_forward"] += 1
                return _original(*call_args, **call_kwargs)

            module._mxfp4_qkv_original_forward = counted_original
        return enabled

    qwen3.enable_mxfp4_qwen_swiglu = counted_enable_swiglu
    qwen3.enable_mxfp4_qwen_qkv = counted_enable_qkv
    counts["rank"] = int(os.environ["RANK"])
    counts["world_size"] = int(os.environ["WORLD_SIZE"])
    counts["instrumentation_installed"] = 1
    return counts, details


def _write_report(template: str, counts: Counter[str], details: dict[str, Any]) -> None:
    rank = int(os.environ["RANK"])
    target = Path(template.format(rank=rank))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "counts": dict(sorted(counts.items())),
        **details,
        "aiter_import": __import__("aiter").__file__,
        "lumen_import": __import__("lumen").__file__,
    }
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def main() -> None:
    shape_log_target = _shape_log_target()
    report_template = os.environ.get("ROUTE_REPORT_TEMPLATE", "")
    counts, details = _install_counters()
    shape_logger = None
    if shape_log_target:
        from lumen.ops.quantize import mxfp4_autotune

        mxfp4_autotune._SHAPE_LOG_PATH = shape_log_target
        mxfp4_autotune._register_hooks()
        shape_logger = mxfp4_autotune

    try:
        runpy.run_path(str(TRAIN_ENTRY), run_name="__main__")
    finally:
        # Do not rely on an interpreter atexit hook.  The distributed launcher
        # may tear libraries down before that hook has flushed the route file.
        if shape_logger is not None:
            shape_logger._save_shape_log()
        if report_template:
            _write_report(report_template, counts, details)

    if shape_log_target and not Path(shape_log_target).is_file():
        raise RuntimeError(f"MXFP4 shape log was not written: {shape_log_target}")


if __name__ == "__main__":
    main()

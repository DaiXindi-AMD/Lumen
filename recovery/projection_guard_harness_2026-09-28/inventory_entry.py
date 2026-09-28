#!/usr/bin/env python3
"""Run Qwen3 training and write rank-local precision-route inventory.

Only model-construction helpers are wrapped.  No forward/backward hot-path
call is instrumented, so formal step timing remains representative.
"""

from __future__ import annotations

import json
import os
import runpy
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


def _inventory(model: nn.Module) -> dict[str, Any]:
    unquantized = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not bool(getattr(module, "_quant_enabled", False))
    ]
    lm_heads = [module for name, module in unquantized if name == "lm_head"]
    result: dict[str, Any] = {
        "unquantized_linear_names": [name for name, _module in unquantized],
        "lm_head_count": len(lm_heads),
    }
    if len(lm_heads) == 1:
        result["lm_head_weight_dtype"] = str(lm_heads[0].weight.dtype).removeprefix(
            "torch."
        )
        result["lm_head_quant_enabled"] = bool(
            getattr(lm_heads[0], "_quant_enabled", False)
        )
    return result


def _install_inventory_hooks() -> tuple[dict[str, int], dict[str, Any]]:
    import lumen.models.qwen3 as qwen3

    counts = {
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "inventory_captured": 0,
        "enabled_qkv": -1,
        "enabled_swiglu": -1,
    }
    details: dict[str, Any] = {}
    original_enable_swiglu = qwen3.enable_mxfp4_qwen_swiglu
    original_enable_qkv = qwen3.enable_mxfp4_qwen_qkv

    def inventory_enable_swiglu(model, *args, **kwargs):
        if not details:
            details.update(_inventory(model))
            counts["inventory_captured"] = 1
        enabled = original_enable_swiglu(model, *args, **kwargs)
        counts["enabled_swiglu"] = int(enabled)
        return enabled

    def inventory_enable_qkv(model, *args, **kwargs):
        if not details:
            details.update(_inventory(model))
            counts["inventory_captured"] = 1
        enabled = original_enable_qkv(model, *args, **kwargs)
        counts["enabled_qkv"] = int(enabled)
        return enabled

    qwen3.enable_mxfp4_qwen_swiglu = inventory_enable_swiglu
    qwen3.enable_mxfp4_qwen_qkv = inventory_enable_qkv
    return counts, details


def _write_report(
    template: str, counts: dict[str, int], details: dict[str, Any]
) -> None:
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
    counts, details = _install_inventory_hooks()
    shape_logger = None
    if shape_log_target:
        from lumen.ops.quantize import mxfp4_autotune

        mxfp4_autotune._SHAPE_LOG_PATH = shape_log_target
        mxfp4_autotune._register_hooks()
        shape_logger = mxfp4_autotune

    try:
        runpy.run_path(str(TRAIN_ENTRY), run_name="__main__")
    finally:
        if shape_logger is not None:
            shape_logger._save_shape_log()
        if report_template:
            _write_report(report_template, counts, details)

    if shape_log_target and not Path(shape_log_target).is_file():
        raise RuntimeError(f"MXFP4 shape log was not written: {shape_log_target}")


if __name__ == "__main__":
    main()

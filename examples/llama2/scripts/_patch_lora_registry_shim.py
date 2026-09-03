#!/usr/bin/env python3
"""Deprecated — use ``examples/dsv4/patch_megatron_source.py <megatron-root> --tag llama,lora``."""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path


def _cli_main(argv: list[str]) -> int:
    _ensure_lumen_stub()
    from lumen.patches.cli import main as registry_main

    return registry_main([*argv, "--tag", "lora"])


def _ensure_lumen_stub() -> None:
    lumen = sys.modules.get("lumen")
    if lumen is not None and getattr(lumen, "__path__", None):
        return
    root = Path(__file__).resolve().parents[3]
    stub = types.ModuleType("lumen")
    stub.__path__ = [str(root / "lumen")]
    sys.modules["lumen"] = stub


def main(argv: list[str] | None = None) -> int:
    warnings.warn(
        f"{Path(__file__).name} is deprecated; use "
        "'PYTHONPATH=<Lumen> python3 examples/dsv4/patch_megatron_source.py <megatron-root> --tag llama,lora'",
        DeprecationWarning,
        stacklevel=2,
    )
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        return 1
    return _cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

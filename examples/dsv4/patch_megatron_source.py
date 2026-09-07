#!/usr/bin/env python3
"""Apply Lumen SOURCE patches to a Megatron-LM checkout (torch-less host entry).

Works on hosts without PyTorch (e.g. ``prepare_rocm_megatron.sh``) by stubbing
the top-level ``lumen`` package before loading the patch registry.

Examples::

    # Apply all default SOURCE patches (DSV4 + ROCm platform)
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM

    # List patches
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py --list
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py --list --tag rocm

    # Dry-run (no file changes)
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --dry-run --tag llama

    # Multiple tags in one flag (OR): DSV4 + standalone ROCm platform patches
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag dsv4,rocm

    # LLaMA + LoRA in one command (OR)
    PYTHONPATH=~/Lumen python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama,lora

    python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM
    python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
    python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4,rocm

See ``examples/dsv4/PATCHES.md`` for IMPORT / CONFIG_BUILD / TRAINING runtime patches.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _ensure_lumen_package_stub() -> None:
    """Avoid ``lumen/__init__.py`` (torch) when only patching Megatron source."""
    lumen = sys.modules.get("lumen")
    if lumen is not None and getattr(lumen, "__path__", None):
        return
    root = Path(__file__).resolve().parents[2]
    stub = types.ModuleType("lumen")
    stub.__path__ = [str(root / "lumen")]
    sys.modules["lumen"] = stub


def main(argv: list[str] | None = None) -> int:
    _ensure_lumen_package_stub()
    from lumen.patches.cli import main as cli_main

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        print(f"       {sys.argv[0]} --list [--tag dsv4] [--tag rocm]")
        return 1
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

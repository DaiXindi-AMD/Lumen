#!/usr/bin/env python3
"""MI308X tile_kernels patches for bootstrap site-packages in lumen/tests:latest."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Bootstrap layout: /bootstrap/site-packages/tile_kernels
SITE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/bootstrap/site-packages/tile_kernels"
)
SINKHORN_BWD_TOKEN_BLOCK = 16
NORM_FN_HIDDEN_BLOCK = 64


def patch_post_kernel() -> None:
    path = SITE / "mhc/post_kernel.py"
    if not path.is_file():
        return
    text = path.read_text()
    if "T.pdl_sync()" in text:
        path.write_text(text.replace("            T.pdl_sync()\n", ""))
        print(f"[mi308x] patched post_kernel: removed T.pdl_sync()")


def patch_norm_fn_kernel() -> None:
    path = SITE / "mhc/norm_fn_kernel.py"
    if not path.is_file():
        return
    text = path.read_text()
    patched = re.sub(
        r"hidden_block: int = \d+,(\s*#.*)?",
        f"hidden_block: int = {NORM_FN_HIDDEN_BLOCK},  # MI308X 64KB LDS",
        text,
    )
    if patched != text:
        path.write_text(patched)
        print(f"[mi308x] patched norm_fn_kernel hidden_block={NORM_FN_HIDDEN_BLOCK}")


def patch_sinkhorn_bwd_token_block() -> None:
    path = SITE / "modeling/mhc/ops/sinkhorn.py"
    if not path.is_file():
        return
    text = path.read_text()
    old = "_mhc_sinkhorn_bwd(hidden_size, 32, repeat, eps)"
    new = f"_mhc_sinkhorn_bwd(hidden_size, {SINKHORN_BWD_TOKEN_BLOCK}, repeat, eps)"
    if old in text:
        path.write_text(text.replace(old, new))
        print(f"[mi308x] patched sinkhorn bwd token_block 32->{SINKHORN_BWD_TOKEN_BLOCK}")


def patch_sitecustomize(miles_root: Path) -> None:
    patch_py = miles_root / "docker/usercustomize_mi308x.py"
    sc = Path("/usr/lib/python3.10/sitecustomize.py")
    if not patch_py.is_file() or not sc.is_file():
        return
    marker = "# MI308X (gfx942) compatibility patches"
    text = sc.read_text()
    if marker in text:
        text = text[: text.index(marker)]
    sc.write_text(text + patch_py.read_text())
    print("[mi308x] refreshed sitecustomize patches")


def main() -> None:
    miles_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/miles")
    if not SITE.is_dir():
        print(f"[mi308x] WARN: tile_kernels not found at {SITE}", file=sys.stderr)
        return
    patch_post_kernel()
    patch_norm_fn_kernel()
    patch_sinkhorn_bwd_token_block()
    patch_sitecustomize(miles_root)


if __name__ == "__main__":
    main()

"""MHC backend selection — delegates to TileKernels ``modeling/mhc/ops`` dispatch."""

from __future__ import annotations

import os


def get_mhc_backend() -> str:
    """Return ``triton`` or ``tilelang`` (TileKernels ``MHC_BACKEND`` env)."""
    return os.environ.get("MHC_BACKEND", "triton").lower()


def log_mhc_backend() -> str:
    """Import ops once and return the active backend (for bootstrap logging)."""
    backend = get_mhc_backend()
    from tile_kernels.modeling.mhc.ops.backend import get_backend

    active = get_backend()
    if active != backend:
        raise RuntimeError(f"MHC_BACKEND={backend!r} but tile_kernels loaded as {active!r}")
    return active

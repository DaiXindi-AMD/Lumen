"""MHC backend selection — delegates to TileKernels ``modeling/mhc/ops`` dispatch."""

from __future__ import annotations

import os


def get_mhc_backend() -> str:
    """Return ``triton`` or ``tilelang`` (TileKernels ``MHC_BACKEND`` env)."""
    return os.environ.get("MHC_BACKEND", "triton").lower()


def log_mhc_backend() -> str:
    """Import ops once and return the configured backend (for bootstrap logging)."""
    backend = get_mhc_backend()
    # TileKernels dispatch is via kernel modules under tile_kernels/mhc/; verify import path.
    import tile_kernels.modeling.mhc.ops  # noqa: F401

    return backend

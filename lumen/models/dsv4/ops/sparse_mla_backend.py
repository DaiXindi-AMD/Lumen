"""Sparse MLA backend selection for DSV4 training."""

from __future__ import annotations

import os
from typing import Callable

import torch


def get_sparse_mla_backend() -> str:
    return os.environ.get("V4_SPARSE_MLA_BACKEND", "triton").lower()


def get_sparse_attn_fn() -> Callable[..., torch.Tensor]:
    backend = get_sparse_mla_backend()
    if backend == "triton":
        from lumen.models.dsv4.ops.kernel.triton_sparse_mla import sparse_attn_triton

        return sparse_attn_triton
    if backend == "tilelang":
        from lumen.models.dsv4.ops.kernel.tilelang_sparse_mla import sparse_attn_tilelang

        return sparse_attn_tilelang
    raise ValueError(
        f"Unknown V4_SPARSE_MLA_BACKEND={backend!r}; expected 'triton' or 'tilelang'"
    )

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU activation dispatch.

Dispatches to AITER's Triton fused SwiGLU kernels which compute
``silu(y1) * y2`` (forward) and the full backward in single kernel
launches each, eliminating the ~6-8 separate elementwise launches that
Megatron's ``@jit_fuser`` swiglu/swiglu_back produce.

Enable globally via ``LUMEN_FUSED_SWIGLU=1`` (installed by megatron_patches).
"""

import logging

import torch
from torch.autograd.function import once_differentiable

logger = logging.getLogger(__name__)


def _probe_aiter_swiglu() -> bool:
    """Return True if AITER fused SwiGLU kernels are importable."""
    from lumen.ops.dispatch import _probe_aiter_swiglu as _probe

    return _probe()


def _probe_aiter_swiglu_split() -> bool:
    """Return True if AITER's eager-compatible split kernels are importable."""
    from lumen.ops.dispatch import _probe_aiter_swiglu_split as _probe

    return _probe()


def fused_swiglu(y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward via AITER Triton kernel."""
    from aiter.ops.triton.activation import swiglu_fwd

    return swiglu_fwd(y)


def fused_swiglu_backward(grad_output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU backward via AITER Triton kernel."""
    from aiter.ops.triton.activation import swiglu_bwd

    return swiglu_bwd(grad_output, y)


class _SplitSwiGLUFunction(torch.autograd.Function):
    """Apply eager-compatible SwiGLU to separate gate and up tensors."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        from aiter.ops.triton.activation import swiglu_fwd_split

        ctx.save_for_backward(gate, up)
        return swiglu_fwd_split(gate, up)

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from aiter.ops.triton.activation import swiglu_bwd_split

        gate, up = ctx.saved_tensors
        return swiglu_bwd_split(grad_output.contiguous(), gate, up)


def split_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Apply eager-compatible SwiGLU while preserving separate gradients."""
    return _SplitSwiGLUFunction.apply(gate, up)


class _PackedSwiGLUFunction(torch.autograd.Function):
    """Apply eager-compatible SwiGLU while returning one packed input gradient."""

    @staticmethod
    def forward(ctx, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim < 1 or packed.shape[-1] % 2:
            raise ValueError("packed SwiGLU requires an even last dimension")

        from aiter.ops.triton.activation import fused_silu_mul

        ctx.save_for_backward(packed)
        return fused_silu_mul(packed)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        from aiter.ops.triton.activation import swiglu_bwd_split

        (packed,) = ctx.saved_tensors
        width = packed.shape[-1] // 2
        gate, up = packed.split(width, dim=-1)
        packed_grad = torch.empty_like(packed)
        grad_gate, grad_up = packed_grad.split(width, dim=-1)
        swiglu_bwd_split(
            grad_output.contiguous(),
            gate,
            up,
            grad_gate,
            grad_up,
        )
        return (packed_grad,)


def packed_swiglu(packed: torch.Tensor) -> torch.Tensor:
    """Apply SiLU gating to a packed gate/up projection with packed backward."""
    return _PackedSwiGLUFunction.apply(packed)

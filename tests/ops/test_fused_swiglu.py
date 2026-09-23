"""Tests for the autograd-aware packed SwiGLU integration."""

import pytest
import torch
import torch.nn.functional as F

from conftest import compute_snr
from lumen.ops.fused_swiglu import packed_swiglu, split_swiglu


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@_CUDA
def test_packed_swiglu_forward_backward_matches_eager_bf16():
    """Packed backward writes one contiguous gradient without changing SwiGLU math."""
    torch.manual_seed(20260921)
    values = torch.randn(2, 257, 24_576, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(
        2, 257, 12_288, device="cuda", dtype=torch.bfloat16
    )

    reference_input = values.detach().clone().requires_grad_(True)
    gate, up = reference_input.split(12_288, dim=-1)
    reference = F.silu(gate) * up
    reference.backward(grad_output)

    actual_input = values.detach().clone().requires_grad_(True)
    actual = packed_swiglu(actual_input)
    actual.backward(grad_output)

    assert reference_input.grad is not None and actual_input.grad is not None
    assert torch.isfinite(actual).all() and torch.isfinite(actual_input.grad).all()
    assert compute_snr(reference, actual) >= 40.0
    assert compute_snr(reference_input.grad, actual_input.grad) >= 40.0
    assert actual_input.grad.is_contiguous()


@_CUDA
def test_packed_swiglu_accepts_noncontiguous_downstream_gradient():
    """Backward makes the downstream gradient launch-safe before AITER dispatch."""
    torch.manual_seed(20260921)
    values = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(
        2, 32, 3, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    assert not grad_output.is_contiguous()

    reference_input = values.detach().clone().requires_grad_(True)
    gate, up = reference_input.split(32, dim=-1)
    reference = F.silu(gate) * up
    reference.backward(grad_output)

    actual_input = values.detach().clone().requires_grad_(True)
    actual = packed_swiglu(actual_input)
    actual.backward(grad_output)

    assert reference_input.grad is not None and actual_input.grad is not None
    assert compute_snr(reference, actual) >= 40.0
    assert compute_snr(reference_input.grad, actual_input.grad) >= 40.0
    assert actual_input.grad.is_contiguous()


@_CUDA
def test_packed_swiglu_rejects_double_backward():
    """The custom AITER backward is intentionally first-order only."""
    values = torch.randn(
        2, 3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    first_grad = torch.autograd.grad(
        packed_swiglu(values).float().sum(), values, create_graph=True
    )[0]

    assert not first_grad.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(first_grad.float().sum(), values)


@_CUDA
def test_split_swiglu_forward_backward_matches_eager_bf16():
    """Separate-input integration keeps forward and both gradients accurate."""
    torch.manual_seed(20260921)
    gate = torch.randn(2, 257, 12_288, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    grad_output = torch.randn_like(gate)

    reference_gate = gate.detach().clone().requires_grad_(True)
    reference_up = up.detach().clone().requires_grad_(True)
    reference = F.silu(reference_gate) * reference_up
    reference.backward(grad_output)

    actual_gate = gate.detach().clone().requires_grad_(True)
    actual_up = up.detach().clone().requires_grad_(True)
    actual = split_swiglu(actual_gate, actual_up)
    actual.backward(grad_output)

    assert reference_gate.grad is not None and actual_gate.grad is not None
    assert reference_up.grad is not None and actual_up.grad is not None
    assert compute_snr(reference, actual) >= 40.0
    assert compute_snr(reference_gate.grad, actual_gate.grad) >= 40.0
    assert compute_snr(reference_up.grad, actual_up.grad) >= 40.0

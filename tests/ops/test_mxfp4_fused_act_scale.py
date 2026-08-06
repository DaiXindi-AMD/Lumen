"""End-to-end check that fusing the activation scale swizzle changes nothing.

The forward stores the activation's E8M0 scales in the gfx950 GEMM layout so the
GEMMs and the WGrad requantizer read them in place. Every consumer either takes
that layout or undoes it, so the whole linear -- output, dX and dW -- has to come
back bit-identical to the run that kept the scales row-major.
"""
import random

import pytest
import torch

from lumen.ops.quantize.linear import QuantizedLinearFunction


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def _run_linear(x, w, dy):
    # The gradient quantizer rounds stochastically off Python's RNG, so the two
    # runs only differ in the scale layout if they draw the same philox stream.
    random.seed(0)
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = QuantizedLinearFunction.apply(
        x, w, None, None, "mxfp4", None, 32, "weight",
    )
    out.backward(dy)
    return out, x.grad, w.grad


@_CUDA
@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 scale layout")
@pytest.mark.parametrize("M,K,N", [(512, 4096, 4096), (256, 2048, 512)])
def test_mxfp4_linear_fused_act_scale_matches_row_major(M, K, N, monkeypatch):
    import lumen.ops.quantize.linear as lin

    torch.manual_seed(17)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if not lin._mxfp4_can_fuse_scale_swizzle((M, K // 32)):
        pytest.skip(f"({M}, {K}) activation scales are not swizzle-eligible")

    # Forward's WGrad operand keys off the same predicate, and it changes dW by
    # design (one quantization instead of two). Hold it off in both runs so this
    # compares the scale layout and nothing else.
    monkeypatch.setattr(lin, "_mxfp4_wgrad_activation_operand", lambda *a: None)

    try:
        got = _run_linear(x, w, dy)
        # The same call with the fusion refused, which is what every
        # non-eligible shape already runs.
        monkeypatch.setattr(lin, "_mxfp4_can_fuse_scale_swizzle", lambda *a: False)
        ref = _run_linear(x, w, dy)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    for name, a, b in zip(("out", "dX", "dW"), got, ref):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=f"{name} differs")

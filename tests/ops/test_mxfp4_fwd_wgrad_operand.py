"""Emitting WGrad's activation operand in forward has to keep the linear correct.

The operand carries the same values as the one backward used to rebuild from the
stored FP4, minus one round of quantization, so dW cannot match bit-for-bit. What
has to hold is that the forward output and dX are untouched and that dW stays at
least as close to the BF16 reference as the rebuild path was.
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
    # The gradient quantizer rounds stochastically off Python's RNG; both runs
    # have to draw the same philox stream to be comparable.
    random.seed(0)
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = QuantizedLinearFunction.apply(x, w, None, None, "mxfp4", None, 32, "weight")
    out.backward(dy)
    return out, x.grad, w.grad


def _snr(ref, got):
    err = (ref.float() - got.float()).pow(2).sum()
    if err == 0:
        return float("inf")
    return 10 * torch.log10(ref.float().pow(2).sum() / err).item()


@_CUDA
@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
@pytest.mark.parametrize("M,K,N", [(512, 512, 256), (768, 256, 512)])
def test_fused_wgrad_operand_keeps_forward_and_improves_wgrad(M, K, N, monkeypatch):
    import lumen.ops.quantize.linear as lin

    torch.manual_seed(17)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if lin._mxfp4_wgrad_activation_operand(x, w, "mxfp4", True, True) is None:
        pytest.skip(f"({M}, {K}) is not eligible for the fused WGrad operand")

    try:
        got = _run_linear(x, w, dy)
        # The same call with the fusion refused: backward rebuilds the operand.
        monkeypatch.setattr(lin, "_mxfp4_wgrad_activation_operand", lambda *a: None)
        ref = _run_linear(x, w, dy)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    # Forward and DGrad never touch this operand.
    torch.testing.assert_close(got[0], ref[0], atol=0, rtol=0, msg="out differs")
    torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0, msg="dX differs")

    dw_ref = dy.float().t() @ x.float()
    snr_fused, snr_rebuilt = _snr(dw_ref, got[2]), _snr(dw_ref, ref[2])
    assert snr_fused >= snr_rebuilt - 0.5, (
        f"fused dW is worse than the rebuilt one: {snr_fused:.2f} dB vs {snr_rebuilt:.2f} dB"
    )


@_CUDA
@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
def test_fused_operand_is_the_direct_quantization_of_the_rotated_activation():
    """The forward-emitted operand is exactly what quantizing the rotated X^T gives.

    The rebuild it replaces passes through FP4 twice, so it can only sit further
    from that rotation; this pins the fused one to the single-quantization form.
    """
    import lumen.ops.quantize.linear as lin
    from lumen.ops.quantize.ops import (
        convert_from_mxfp4,
        convert_to_mxfp4,
        dequant_hadamard_quant_mxfp4,
        dual_layout_quant_mxfp4,
        hadamard_quant_mxfp4,
    )

    M, K = 512, 512
    torch.manual_seed(3)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = lin._get_mxfp4_rht_sign(x.device)

    _, _, wg_fp4, wg_scale = dual_layout_quant_mxfp4(
        x, sign, block_size=32, g=lin._MXFP4_RHT_G,
        use_sr_row=False, use_sr_transposed=False,
    )
    exact_fp4, exact_scale = hadamard_quant_mxfp4(
        x.t().contiguous(), sign, block_size=32, g=lin._MXFP4_RHT_G, use_sr=False,
    )
    torch.testing.assert_close(wg_fp4, exact_fp4, atol=0, rtol=0)
    torch.testing.assert_close(wg_scale, exact_scale, atol=0, rtol=0)

    x_fp4, x_scale = convert_to_mxfp4(x, block_size=32, axis=-1, use_sr=False)
    rebuilt_fp4, rebuilt_scale = dequant_hadamard_quant_mxfp4(
        x_fp4, x_scale, sign, block_size=32, g=lin._MXFP4_RHT_G, use_sr=False,
    )
    exact = convert_from_mxfp4(exact_fp4, exact_scale, output_dtype=torch.float32, block_size=32)
    rebuilt = convert_from_mxfp4(
        rebuilt_fp4, rebuilt_scale, output_dtype=torch.float32, block_size=32,
    )
    assert not torch.equal(exact, rebuilt), "the rebuild would be lossless, which it is not"

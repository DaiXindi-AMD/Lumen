###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ablation switches for the shared kernels `f01c39f` touched.

These three are not MXFP4-specific -- they are the RMSNorm and attention wins
from the same commit -- so they live apart from the MXFP4 switch tests. What
matters here is that each switch really reaches the pre-commit branch, since all
three legacy paths are still live at HEAD and a switch that missed its seam would
report a free optimization.
"""

import importlib
import importlib.util
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from lumen.utils import ablation

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_MEGATRON = pytest.mark.skipif(
    importlib.util.find_spec("megatron") is None,
    reason="needs Megatron, which only the training image has",
)


def _rmsnorm_module():
    """The submodule, not the same-named function the package re-exports."""
    return importlib.import_module("lumen.ops.normalization.rmsnorm")


def _snr(ref, got):
    err = (ref.float() - got.float()).pow(2).sum()
    if err == 0:
        return float("inf")
    return 10 * torch.log10(ref.float().pow(2).sum() / err).item()


# --------------------------------------------------------------------------- #
# A21 narrow-N RMSNorm backward
# --------------------------------------------------------------------------- #


@_CUDA
def test_narrow_rmsnorm_switch_routes_to_the_aiter_path():
    """The switch has to send short rows back to AITER's autograd.

    Lumen's row-tiling kernels exist because AITER specialises its narrow-N
    forward but not its backward. Turning the arm off must reach AITER again, not
    just recompute the same thing.
    """
    rms = _rmsnorm_module()

    if not rms._probe_aiter_triton_rmsnorm():
        pytest.skip("AITER Triton RMSNorm unavailable, so there is no legacy path")

    # N = 128 is Qwen3-8B's QK-norm width, which is what the specialisation
    # captures; the 4096-wide layer norms never reach it.
    x = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    calls = {"narrow": 0}
    real = rms.narrow_rmsnorm_forward

    def _counted(*a, **kw):
        calls["narrow"] += 1
        return real(*a, **kw)

    outs = {}
    for on in (True, False):
        calls["narrow"] = 0
        with mock.patch.object(rms, "narrow_rmsnorm_forward", _counted):
            with ablation.overridden(NARROW_N_RMSNORM=on):
                y = rms.rmsnorm(x, w, 1e-6)
        y.sum().backward()
        outs[on] = (y.detach().clone(), x.grad.detach().clone())
        x.grad = None
        w.grad = None
        assert calls["narrow"] == (1 if on else 0), (
            f"NARROW_N_RMSNORM={on} took the wrong path"
        )

    # Two implementations of the same normalization: close, not bit-identical.
    assert _snr(outs[True][0], outs[False][0]) > 40.0
    assert _snr(outs[True][1], outs[False][1]) > 40.0


def test_narrow_rmsnorm_switch_leaves_wide_rows_alone():
    """The arm must not touch the 4096-wide layer norms, on or off."""
    rms = _rmsnorm_module()

    x = torch.zeros(8, 4096)
    w = torch.zeros(4096)
    assert not rms._rows_are_short(x, w), (
        "a 4096-wide row is not narrow, so this arm cannot be credited for it"
    )


# --------------------------------------------------------------------------- #
# A22 / A23 attention operand and output layout
# --------------------------------------------------------------------------- #


class _MockAttnMaskType:
    causal = 1
    no_mask = 0


def _make_config():
    return SimpleNamespace(
        num_attention_heads=8,
        num_query_groups=8,
        kv_channels=64,
        tensor_model_parallel_size=1,
        apply_query_key_layer_scaling=False,
        attention_dropout=0.0,
        context_parallel_size=1,
    )


def _make_args():
    return SimpleNamespace(
        lumen_attn_backend="aiter_csrc",
        lumen_fp8_quant_type="blockwise",
        lumen_fp8_attn="none",
        mxfp8_block_m_fwd=128,
        mxfp8_block_n_fwd=128,
        mxfp8_block_m_dq_bwd=128,
        mxfp8_block_n_dq_bwd=128,
        mxfp8_block_m_dkv_bwd=128,
        mxfp8_block_n_dkv_bwd=128,
        mxfp8_quant_block_size=128,
        grad_quant_type=None,
    )


def _patched_megatron_init(self, config):
    torch.nn.Module.__init__(self)
    self.config = config


def _capture_attention_flags(**switches):
    """Build the Megatron attention module and record the two layout decisions.

    The real kernels are not called: what is under test is which layout the
    module asks for, and stubbing attention keeps the test off the CK JIT.
    """
    import lumen.modules.attention_megatron as am

    seen = {}

    def _stub_attention(q, k, v, **kw):
        seen["seq_major_out"] = kw.get("seq_major_out")
        seen["q_is_view"] = not q.is_contiguous()
        b, s, h, d = q.shape
        return torch.zeros((b, s, h, d), device=q.device, dtype=q.dtype)

    with mock.patch.object(am, "AttnMaskType", _MockAttnMaskType), \
         mock.patch.object(am, "divide", side_effect=lambda a, b: a // b), \
         mock.patch.object(am.MegatronModule, "__init__", _patched_megatron_init), \
         mock.patch.object(am, "get_args", side_effect=_make_args), \
         mock.patch.object(am, "is_aiter_available", lambda: True), \
         mock.patch.object(am, "attention", _stub_attention):
        attn = am.LumenDotProductAttention(
            _make_config(), layer_number=1,
            attn_mask_type=_MockAttnMaskType.causal,
            attention_type="self",
        )
        s, b, h, d = 16, 2, 8, 64
        q = torch.randn(s, b, h, d)
        with ablation.overridden(**switches):
            attn(q, q.clone(), q.clone(), None)
    return seen


@_MEGATRON
def test_attn_qkv_views_switch_controls_whether_qkv_is_copied():
    on = _capture_attention_flags(ATTN_QKV_VIEWS=True)
    off = _capture_attention_flags(ATTN_QKV_VIEWS=False)
    assert on["q_is_view"] is True, "the arm should hand the kernel a strided view"
    assert off["q_is_view"] is False, "with the arm off the operand must be copied"


@_MEGATRON
def test_attn_seq_major_switch_controls_the_output_layout_request():
    on = _capture_attention_flags(ATTN_SEQ_MAJOR=True)
    off = _capture_attention_flags(ATTN_SEQ_MAJOR=False)
    assert on["seq_major_out"] is True
    assert off["seq_major_out"] is False, (
        "with the arm off the kernel must write batch-major and pay the transpose"
    )

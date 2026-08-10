###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""MXFP4 weight operands built directly in the GEMM's layout.

Both operands the weight cache hands out -- the forward's and DGrad's -- are read
by nothing but an MXFP4 GEMM, so the quantizer and the transpose store them in
that GEMM's order and the separate permuting passes go away. A wrong permutation
does not raise: it multiplies the right numbers in the wrong places, so these
compare the GEMM's output against the operands built the two-pass way.
"""

import pytest
import torch
import torch.nn as nn


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout"),
]

BLOCK = 32


def _build(weight, gemm_rows):
    from lumen.quantize import _mxfp4_cached_weight

    module = nn.Module()
    data, scale = _mxfp4_cached_weight(
        module, weight, None, None, "mxfp4", None, BLOCK, gemm_rows=gemm_rows,
    )
    data_t, scale_t = data._mxfp4_wt_cached
    return (data, scale), (data_t, scale_t)


def _operand_cache_inputs():
    w = torch.zeros(64, 64, device="cuda", dtype=torch.uint8)
    scale = torch.zeros(64, 2, device="cuda", dtype=torch.uint8)
    return w, scale


def test_operand_cache_refuses_to_memoize_a_view_of_the_weight():
    """A reshape of an already-shuffled weight is a view, not a copy.

    Memoizing it on the weight makes the weight reference itself. The cache
    guarded against being handed the weight back, but not against being handed a
    view of it, which leaked one copy of every quantized weight per iteration.
    """
    from lumen.ops.quantize.linear import _cached_weight_operands

    w, scale = _operand_cache_inputs()
    key = "_test_alias_operands"

    data, _ = _cached_weight_operands(w, scale, key, lambda: (w.reshape(32, 128), scale))

    assert data.data_ptr() == w.data_ptr(), "the build result should still be returned"
    assert getattr(w, key, None) is None


def test_operand_cache_lets_the_weight_die_by_refcount_alone():
    """The leak was only visible as a leak because GPU bytes are invisible to gc.

    With the cyclic collector off, a weight that the cache has put in a cycle
    never goes away. That is the training behaviour: the collector's thresholds
    count Python allocations, so a 2 GiB tensor never triggers one.
    """
    import gc
    import weakref

    from lumen.ops.quantize.linear import _cached_weight_operands

    def build_and_drop():
        w, scale = _operand_cache_inputs()
        _cached_weight_operands(
            w, scale, "_test_alias_lifetime", lambda: (w.reshape(32, 128), scale)
        )
        return weakref.ref(w)

    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        ref = build_and_drop()
        assert ref() is None
    finally:
        if gc_was_on:
            gc.enable()


@pytest.mark.parametrize(
    "N_out,K_in", [(6144, 4096), (4096, 12288)], ids=["qkv", "fc2"],
)
def test_mxfp4_cached_weight_fused_operands_match_two_pass(N_out, K_in):
    from lumen.ops.quantize.linear import (
        _mxfp4_can_fuse_b_shuffle,
        gemm_mxfp4_dispatch,
    )
    from lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(23)
    M = 2048
    weight = torch.randn(N_out, K_in, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, K_in, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(M, N_out, device="cuda", dtype=torch.bfloat16)
    a_fp4, a_scale = convert_to_mxfp4(x, block_size=BLOCK, axis=-1, use_sr=False)
    g_fp4, g_scale = convert_to_mxfp4(g, block_size=BLOCK, axis=-1, use_sr=False)

    # gemm_rows=None keeps both operands row-major, which is also what the first
    # micro-batch of a run gets, before the backend for the shape is measured.
    (w_ref, sw_ref), (wt_ref, swt_ref) = _build(weight, None)
    fwd_ref = gemm_mxfp4_dispatch(a_fp4, w_ref, a_scale, sw_ref)
    dgrad_ref = gemm_mxfp4_dispatch(g_fp4, wt_ref, g_scale, swt_ref)

    if not (
        _mxfp4_can_fuse_b_shuffle((M, N_out, K_in), N_out, K_in // 2)
        and _mxfp4_can_fuse_b_shuffle((M, K_in, N_out), K_in, N_out // 2)
    ):
        pytest.skip("these shapes dispatch to the row-major kernel, which cannot fuse")

    (w, sw), (wt, swt) = _build(weight, M)
    torch.testing.assert_close(gemm_mxfp4_dispatch(a_fp4, w, a_scale, sw), fwd_ref,
                               atol=0, rtol=0)
    torch.testing.assert_close(gemm_mxfp4_dispatch(g_fp4, wt, g_scale, swt), dgrad_ref,
                               atol=0, rtol=0)

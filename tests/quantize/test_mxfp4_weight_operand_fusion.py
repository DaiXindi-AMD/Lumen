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

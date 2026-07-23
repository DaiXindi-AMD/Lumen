###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Quantization ops — functional API and autograd-aware quantized linear."""

from lumen.ops.quantize.linear import (
    QuantizedLinearFunction,
    quantized_linear,
)
from lumen.ops.quantize.ops import (
    convert_from_mxfp4,
    convert_from_mxfp4_2d,
    convert_from_mxfp8,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    convert_to_mxfp4_dual_axis,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    hadamard_quant_mxfp4,
    hadamard_transform,
    is_cdna4,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quant_fp8_tensorwise_impl,
    transpose_packed_fp4,
)

__all__ = [
    # Pure quant/dequant ops
    "convert_from_mxfp8",
    "convert_to_mxfp8",
    "dequant_fp8_tensorwise_impl",
    "is_cdna4",
    "quant_fp8_blockwise_impl",
    "quant_fp8_blockwise_segment_m_impl",
    "quant_fp8_tensorwise_impl",
    # MXFP4 quant/dequant ops
    "convert_to_mxfp4",
    "convert_to_mxfp4_2d",
    "convert_to_mxfp4_dual_axis",
    "convert_from_mxfp4",
    "convert_from_mxfp4_2d",
    "transpose_packed_fp4",
    "hadamard_transform",
    "hadamard_quant_mxfp4",
    # Quantized linear (autograd)
    "QuantizedLinearFunction",
    "quantized_linear",
]

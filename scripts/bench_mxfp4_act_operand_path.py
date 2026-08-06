"""Compare the two ways to get the activation's forward and WGrad MXFP4 operands.

Current: forward quantizes X row-major, backward reads that FP4 back and turns it
into the rotated, transposed operand (convert_to_mxfp4 + dequant_hadamard_quant).
Alternative: forward emits both layouts from one dense read (dual_layout_quant),
at the cost of holding the second operand until backward.
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumen.ops.quantize.ops import (  # noqa: E402
    convert_to_mxfp4,
    dequant_hadamard_quant_mxfp4,
    dual_layout_quant_mxfp4,
    mxfp4_scale_swizzle_supported,
)

M = 16384
BLOCK = 32
G = 16
# Qwen3-8B per layer: qkv, attn-proj and gate_up all read hidden=4096; down reads ffn=12288.
SHAPES = [(4096, 3), (12288, 1)]


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


def main():
    dev = "cuda"
    sign = (torch.randint(0, 2, (G,), device=dev) * 2 - 1).to(torch.bfloat16)
    total_cur = total_new = 0.0
    for K, per_layer in SHAPES:
        x = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
        swz_fwd = mxfp4_scale_swizzle_supported(M, K // BLOCK)
        fp4, sc = convert_to_mxfp4(x, block_size=BLOCK, axis=-1, use_sr=False,
                                   swizzle_scale=swz_fwd)
        swz_bwd = mxfp4_scale_swizzle_supported(K, M // BLOCK)

        t_fwd = bench(lambda: convert_to_mxfp4(x, block_size=BLOCK, axis=-1, use_sr=False,
                                               swizzle_scale=swz_fwd))
        t_bwd = bench(lambda: dequant_hadamard_quant_mxfp4(
            fp4, sc, sign, block_size=BLOCK, g=G, use_sr=False,
            swizzle_scale=swz_bwd, in_scale_swizzled=swz_fwd))
        t_dual = bench(lambda: dual_layout_quant_mxfp4(
            x, sign, block_size=BLOCK, g=G, use_sr_row=False, use_sr_transposed=False,
            swizzle_scale=swz_fwd and swz_bwd))

        cur = t_fwd + t_bwd
        print(f"\nX = ({M}, {K})   x{per_layer} per layer")
        print(f"  forward convert_to_mxfp4      {t_fwd:7.1f} us")
        print(f"  backward dequant_hadamard     {t_bwd:7.1f} us")
        print(f"  = current total               {cur:7.1f} us")
        print(f"  forward dual_layout (both)    {t_dual:7.1f} us   {cur/t_dual:.2f}x")
        print(f"  extra memory held             {M*K//2/2**20:.0f} MiB per linear")
        total_cur += cur * per_layer
        total_new += t_dual * per_layer
        del x, fp4, sc
        torch.cuda.empty_cache()

    layers = 31
    print(f"\nper layer: current {total_cur:.1f} us, dual {total_new:.1f} us")
    print(f"over {layers} quantized layers per micro-batch: "
          f"{total_cur*layers/1e3:.1f} ms -> {total_new*layers/1e3:.1f} ms "
          f"(save {(total_cur-total_new)*layers/1e3:.1f} ms)")


if __name__ == "__main__":
    main()

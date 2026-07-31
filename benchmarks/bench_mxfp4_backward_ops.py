# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cost of the MXFP4 backward-path ops that are not GEMMs.

``bench_mxfp4_gemm_models.py`` covers the three GEMMs of a training layer. Those
turned out to be only ~9% of a Qwen3-8B step, so this file measures the ops around
them -- the ones the weight cache and the fused dequant+transpose kernel removed:

    dequant+transpose   WGrad needs the saved activation as BF16 (K, M). Doing it
                        as convert_from_mxfp4 then .t().contiguous() writes the
                        full BF16 (M, K) once and reads it back; the fused kernel
                        writes only the transposed result.
    weight quantize     Under gradient accumulation the BF16 weight does not change
                        between micro-batches, so quantizing it every forward is
                        redundant. This measures what one redundant forward pays.
    FP4 all-gather      MXFP4CommTensor ships 0.5 byte/element instead of 2, at the
                        cost of a quantize before and a dequant after.

Every measurement alternates the two paths inside one process: run to run the
machine drifts ~1%, which is enough to swamp what is being measured.

Run:
    pytest benchmarks/bench_mxfp4_backward_ops.py -v -s
"""

import pytest
import torch

from benchmarks.bench_utils import cuda_timer
from benchmarks.conftest import AITER
from lumen.ops.quantize.ops import (
    convert_from_mxfp4,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    dequant_transpose_mxfp4,
    hadamard_quant_mxfp4,
    transpose_packed_fp4,
)

MXFP4_BLOCK = 32
RHT_G = 16
TOKENS = 8192

# Qwen3-8B: hidden 4096, intermediate 12288, 36 layers, last 5 kept in BF16.
HIDDEN = 4096
INTER = 12288
NUM_LAYERS = 36
TAIL_BF16 = 5
QUANTIZED_LAYERS = NUM_LAYERS - TAIL_BF16

# (name, N_out, K_in) per projection, matching bench_mxfp4_gemm_models.
PROJECTIONS = [
    ("q_proj", 32 * 128, HIDDEN),
    ("k_proj", 8 * 128, HIDDEN),
    ("v_proj", 8 * 128, HIDDEN),
    ("o_proj", HIDDEN, 32 * 128),
    ("gate_proj", INTER, HIDDEN),
    ("up_proj", INTER, HIDDEN),
    ("down_proj", HIDDEN, INTER),
]

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _median_ms(fn, label, warmup=5, iters=20):
    return cuda_timer(fn, warmup=warmup, iters=iters, label=label, trim_pct=10.0).median_ms


def _ab(fn_old, fn_new, label, repeats=3):
    """Alternate the two paths and return the median of each side's medians.

    Interleaved rather than back to back so clock drift and cache state hit both
    equally -- measured separately, whichever ran first can win or lose on drift
    alone by more than the effect being measured.
    """
    olds, news = [], []
    for _ in range(repeats):
        olds.append(_median_ms(fn_old, f"{label} old"))
        news.append(_median_ms(fn_new, f"{label} new"))
    olds.sort()
    news.sort()
    return olds[len(olds) // 2], news[len(news) // 2]


@_CUDA
@AITER
class TestWgradActivationPrep:
    """The three ways WGrad has prepared its activation operand, end to end.

    WGrad needs ``hadamard_quant_mxfp4(X^T)`` from the saved FP4 activation, and the
    chain has been rewritten twice. Timing the whole chain rather than the transpose
    alone matters, because the two rewrites trade against each other: passing a
    transposed *view* skips a copy but makes the quantizer read strided, while the
    fused kernel skips the BF16 intermediate *and* hands the quantizer a dense operand.

        materialise   convert_from_mxfp4 -> .t().contiguous() -> hadamard_quant
        view          convert_from_mxfp4 -> .t()              -> hadamard_quant
        fused         dequant_transpose_mxfp4                 -> hadamard_quant
    """

    def _variants(self, data, scales, sign):
        def _materialise():
            x = convert_from_mxfp4(
                data, scales, output_dtype=torch.bfloat16, block_size=MXFP4_BLOCK,
            )
            return hadamard_quant_mxfp4(
                x.t().contiguous(), sign, block_size=MXFP4_BLOCK, g=RHT_G, use_sr=False,
            )

        def _view():
            x = convert_from_mxfp4(
                data, scales, output_dtype=torch.bfloat16, block_size=MXFP4_BLOCK,
            )
            return hadamard_quant_mxfp4(
                x.t(), sign, block_size=MXFP4_BLOCK, g=RHT_G, use_sr=False,
            )

        def _fused():
            x_t = dequant_transpose_mxfp4(data, scales, block_size=MXFP4_BLOCK)
            return hadamard_quant_mxfp4(
                x_t, sign, block_size=MXFP4_BLOCK, g=RHT_G, use_sr=False,
            )

        return _materialise, _view, _fused

    def test_fused_dequant_transpose_is_bit_exact(self):
        """The fused kernel must reproduce dequant-then-transpose exactly."""
        for k_in in (HIDDEN, INTER):
            x = torch.randn(TOKENS, k_in, dtype=torch.bfloat16, device="cuda")
            data, scales = convert_to_mxfp4(x, block_size=MXFP4_BLOCK, axis=-1)
            ref = convert_from_mxfp4(
                data, scales, output_dtype=torch.bfloat16, block_size=MXFP4_BLOCK,
            ).t().contiguous()
            got = dequant_transpose_mxfp4(data, scales, block_size=MXFP4_BLOCK)
            assert torch.equal(ref, got), f"K={k_in}: fused kernel changed the result"

    def test_chain_cost_per_layer(self):
        sign = torch.ones(RHT_G, dtype=torch.bfloat16, device="cuda")
        rows = []
        for name, _n_out, k_in in PROJECTIONS:
            x = torch.randn(TOKENS, k_in, dtype=torch.bfloat16, device="cuda")
            data, scales = convert_to_mxfp4(x, block_size=MXFP4_BLOCK, axis=-1)
            mat, view, fused = self._variants(data, scales, sign)
            # Interleave all three so drift hits them equally.
            t_mat, t_view = _ab(mat, view, f"prep {name}")
            _, t_fused = _ab(mat, fused, f"prep {name} fused")
            rows.append((name, k_in, t_mat, t_view, t_fused))

        print("\n  WGrad activation prep (dequant + transpose + fused H+Q), "
              "Qwen3-8B @ 8192 tokens:")
        print(f"    {'proj':<12} {'K':>6}  {'materialise':>12} {'view':>12} {'fused':>12}")
        for name, k_in, t_mat, t_view, t_fused in rows:
            print(f"    {name:<12} {k_in:>6}  "
                  f"{t_mat * 1e3:9.1f} us {t_view * 1e3:9.1f} us {t_fused * 1e3:9.1f} us")
        tot = [sum(r[i] for r in rows) for i in (2, 3, 4)]
        print(f"    {'layer total':<12} {'':>6}  "
              f"{tot[0]:9.3f} ms {tot[1]:9.3f} ms {tot[2]:9.3f} ms")
        print(f"    {'vs materialise':<12} {'':>6}  "
              f"{'1.00x':>12} {tot[0] / tot[1]:11.2f}x {tot[0] / tot[2]:11.2f}x")
        print(f"    x{QUANTIZED_LAYERS} layers/step  "
              f"       {tot[0] * QUANTIZED_LAYERS:9.1f} ms "
              f"{tot[1] * QUANTIZED_LAYERS:9.1f} ms {tot[2] * QUANTIZED_LAYERS:9.1f} ms")


@_CUDA
@AITER
class TestWeightQuantCache:
    """What one redundant weight quantization costs.

    The BF16 weight is fixed within an optimizer step, and MXFP4 weight quant is
    RTN, so every micro-batch after the first re-derives a bit-identical FP4
    weight. This measures the work the module-level cache skips.
    """

    def test_weight_quant_cost(self):
        rows = []
        for name, n_out, k_in in PROJECTIONS:
            w = torch.randn(n_out, k_in, dtype=torch.bfloat16, device="cuda")

            def _quant(w=w):
                fp4, scale = convert_to_mxfp4_2d(w, block_size=MXFP4_BLOCK)
                return transpose_packed_fp4(fp4), scale.t().contiguous()

            t = _median_ms(_quant, f"wquant {name}")
            rows.append((name, n_out, k_in, t))

        # RTN is deterministic: the cached tensor is what a re-quantize would produce.
        w = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16, device="cuda")
        a_fp4, a_s = convert_to_mxfp4_2d(w, block_size=MXFP4_BLOCK)
        b_fp4, b_s = convert_to_mxfp4_2d(w, block_size=MXFP4_BLOCK)
        assert torch.equal(a_fp4, b_fp4) and torch.equal(a_s, b_s), (
            "weight quant is not deterministic -- caching it would change numerics"
        )

        layer = sum(r[3] for r in rows)
        print("\n  MXFP4 weight quantize + pre-transpose (what the cache skips):")
        for name, n_out, k_in, t in rows:
            print(f"    {name:<12} ({n_out}x{k_in})".ljust(34)
                  + f"{t * 1e3:7.1f} us")
        print(f"    {'layer total':<12}".ljust(34) + f"{layer:7.3f} ms")
        print(f"    x{QUANTIZED_LAYERS} layers".ljust(34)
              + f"{layer * QUANTIZED_LAYERS:7.1f} ms per redundant forward")


@_CUDA
@AITER
class TestFP4AllGather:
    """MXFP4CommTensor's trade: 4x fewer bytes on the wire, paid for with quant+dequant.

    Single GPU, so this measures the compute the hooks add and the bytes they save,
    not the collective itself.
    """

    def test_hook_cost_and_bytes(self):
        from lumen.quantize.comm_tensor import MXFP4CommTensor

        world = 8
        rows = []
        for name, n_out, k_in in PROJECTIONS:
            if n_out % (MXFP4_BLOCK * world) or k_in % MXFP4_BLOCK:
                rows.append((name, n_out, k_in, None, None, None))
                continue
            shard = torch.randn(
                n_out // world, k_in, dtype=torch.bfloat16, device="cuda",
            )
            t = MXFP4CommTensor(shard, MXFP4_BLOCK)
            (fp4, scale), meta = MXFP4CommTensor.fsdp_pre_all_gather(t)

            # The wire carries world_size shards' worth of each.
            bf16_bytes = n_out * k_in * 2
            fp4_bytes = (fp4.numel() + scale.numel()) * world

            full_fp4 = fp4.repeat(world, 1)
            full_scale = scale.repeat(world, 1)
            t_pre = _median_ms(
                lambda t=t: MXFP4CommTensor.fsdp_pre_all_gather(t), f"pre {name}",
            )
            t_post = _median_ms(
                lambda f=full_fp4, s=full_scale, m=meta: (
                    MXFP4CommTensor.fsdp_post_all_gather(
                        (f, s), m, torch.bfloat16,
                    )
                ),
                f"post {name}",
            )
            rows.append((name, n_out, k_in, t_pre, t_post, bf16_bytes / fp4_bytes))

        print("\n  MXFP4CommTensor hooks, Qwen3-8B weights, world_size=8:")
        tot_pre = tot_post = 0.0
        for name, n_out, k_in, t_pre, t_post, ratio in rows:
            if t_pre is None:
                print(f"    {name:<12} ({n_out}x{k_in}) skipped: not "
                      f"{MXFP4_BLOCK}x{world}-aligned on dim 0, stays BF16")
                continue
            tot_pre += t_pre
            tot_post += t_post
            print(f"    {name:<12} ({n_out}x{k_in})".ljust(34)
                  + f"pre {t_pre * 1e3:6.1f} us  post {t_post * 1e3:6.1f} us  "
                    f"wire {ratio:.2f}x smaller")
        print(f"    {'layer total':<12}".ljust(34)
              + f"pre {tot_pre:6.3f} ms  post {tot_post:6.3f} ms")
        print(f"    x{QUANTIZED_LAYERS} layers".ljust(34)
              + f"pre {tot_pre * QUANTIZED_LAYERS:6.1f} ms  "
                f"post {tot_post * QUANTIZED_LAYERS:6.1f} ms  "
                f"added per all-gather round")

    def test_roundtrip_is_rtn_quantization(self):
        """The gathered weight equals RTN(W) -- lossy, but not drifting.

        The optimizer keeps updating the full-precision sharded master weight; only
        the gathered copy is rounded, so the error does not accumulate across steps.
        And since the forward quantizes that copy to FP4 anyway with the same block
        size and rounding, the FP4 operand the GEMM sees is unchanged.
        """
        from lumen.quantize.comm_tensor import MXFP4CommTensor

        w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        t = MXFP4CommTensor(w, MXFP4_BLOCK)
        tensors, meta = MXFP4CommTensor.fsdp_pre_all_gather(t)
        gathered = MXFP4CommTensor.fsdp_post_all_gather(tensors, meta, torch.bfloat16)

        direct, direct_s = convert_to_mxfp4_2d(w, block_size=MXFP4_BLOCK)
        again, again_s = convert_to_mxfp4_2d(gathered, block_size=MXFP4_BLOCK)
        assert torch.equal(direct, again) and torch.equal(direct_s, again_s), (
            "re-quantizing the gathered weight must give the same FP4 as quantizing "
            "the original, or FP4 all-gather would change the forward"
        )

        err = (gathered.float() - w.float()).abs()
        snr = 20 * torch.log10(
            w.float().norm() / (gathered.float() - w.float()).norm()
        )
        print(f"\n  gathered weight vs BF16 master: max abs err {err.max():.4f}, "
              f"SNR {snr:.1f} dB (one RTN FP4 rounding, no accumulation)")

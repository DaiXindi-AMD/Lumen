"""Is fusing gradient accumulation into the wgrad GEMM worth it on gfx950?

Fusing means the wgrad GEMM writes ``main_grad`` directly instead of emitting a
bf16 gradient that a later elementwise kernel adds in. The asm backend cannot do
this -- its epilogue is prebuilt assembly, bf16 output only, and the ``beta*C``
term its header advertises is not wired up -- so a fused version has to run on a
source-compiled backend (CK or Triton) instead.

That makes the question a trade, not a win: fusing saves the add but gives up
whatever margin asm holds over the backend that can fuse. This measures both
sides at the shapes an actual Qwen3-8B step runs.
"""

import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

# (out_features, in_features) per transformer layer, and how many layers carry
# them. Qwen3-8B, TP=1: 36 layers, the last 5 held in bf16 by the run config.
QWEN3_8B_WEIGHTS = [
    ("qkv", 6144, 4096),
    ("o", 4096, 4096),
    ("gate_up", 24576, 4096),
    ("down", 4096, 12288),
]
MXFP4_LAYERS = 31
# micro_batch 2 x seq 8192, and the microbatches per rank at global batch 128, DP=8.
WGRAD_K = 16384
MICRO_BATCHES = 8


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_accumulate(m, n):
    """The kernel a fused epilogue would delete: main_grad(fp32) += wgrad(bf16)."""
    main_grad = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    wgrad = torch.randn((m, n), dtype=dtypes.bf16, device="cuda")
    ms = timed(lambda: main_grad.add_(wgrad))
    moved = main_grad.numel() * 8 + wgrad.numel() * 2  # rw fp32 + r bf16
    return ms, moved / (ms * 1e-3) / 1e12


def bench_backends(m, n, k):
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    a = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    b = torch.randn((n, k), dtype=dtypes.bf16, device="cuda")
    a_fp4, a_scale = quant(a, shuffle=True)
    b_fp4, b_scale = quant(b, shuffle=True)
    b_shuffled = shuffle_weight(b_fp4, layout=(16, 16))
    out = torch.empty((m, n), dtype=dtypes.bf16, device="cuda")

    results = {}
    results["asm"] = timed(
        lambda: aiter.gemm_a4w4_asm(
            a_fp4, b_shuffled, a_scale, b_scale, out, "", None, 1.0, 0.0, True, 0
        )
    )

    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        _, a_scale_flat = quant(a, shuffle=False)
        _, b_scale_flat = quant(b, shuffle=False)
        results["triton"] = timed(
            lambda: gemm_afp4wfp4(
                a_fp4,
                b_fp4,
                a_scale_flat.view(torch.uint8),
                b_scale_flat.view(torch.uint8),
                dtypes.bf16,
                out,
            )
        )
    except Exception as exc:
        results["triton"] = f"n/a ({type(exc).__name__}: {exc})"

    try:
        ck_out = torch.empty((m + 31) // 32 * 32, n, dtype=dtypes.bf16, device="cuda")
        results["ck"] = timed(
            lambda: aiter.gemm_a4w4_blockscale(
                a_fp4, b_shuffled, a_scale, b_scale, ck_out
            )
        )
    except Exception as exc:
        results["ck"] = f"n/a ({type(exc).__name__})"

    return results


def main():
    torch.manual_seed(0)

    print("=== cost of the add a fused epilogue would remove ===")
    print(f"{'weight':10} {'shape':16} {'ms':>8} {'TB/s':>7} {'ms/step':>9}")
    total_add = 0.0
    for name, out_f, in_f in QWEN3_8B_WEIGHTS:
        ms, tbs = bench_accumulate(out_f, in_f)
        per_step = ms * MXFP4_LAYERS * MICRO_BATCHES
        total_add += per_step
        print(f"{name:10} {f'{out_f}x{in_f}':16} {ms:8.3f} {tbs:7.2f} {per_step:9.1f}")
    print(f"{'total':10} {'':16} {'':8} {'':7} {total_add:9.1f} ms/step\n")

    print("=== wgrad GEMM by backend (the margin fusing would have to pay) ===")
    print(f"{'weight':10} {'M,N,K':22} {'asm ms':>9} {'triton ms':>11} {'ck ms':>9}")
    totals = {"asm": 0.0, "triton": 0.0, "ck": 0.0}
    for name, out_f, in_f in QWEN3_8B_WEIGHTS:
        # wgrad computes [out_features, in_features] from a K of batch tokens.
        res = bench_backends(out_f, in_f, WGRAD_K)
        cells = []
        for key in ("asm", "triton", "ck"):
            v = res[key]
            if isinstance(v, float):
                totals[key] += v * MXFP4_LAYERS * MICRO_BATCHES
                cells.append(f"{v:9.3f}" if key != "triton" else f"{v:11.3f}")
            else:
                totals[key] = float("nan")
                cells.append(f"{v:>11}" if key == "triton" else f"{v:>9}")
        print(f"{name:10} {f'{out_f},{in_f},{WGRAD_K}':22} {cells[0]} {cells[1]} {cells[2]}")
    print(
        f"\nper-step wgrad GEMM: asm {totals['asm']:.1f} ms, "
        f"triton {totals['triton']:.1f} ms, ck {totals['ck']:.1f} ms"
    )
    print(f"add that fusing removes: {total_add:.1f} ms/step")
    for key in ("triton", "ck"):
        penalty = totals[key] - totals["asm"]
        print(f"  fusing on {key}: {total_add:.1f} saved - {penalty:.1f} lost "
              f"= {total_add - penalty:+.1f} ms/step")


if __name__ == "__main__":
    main()

"""What the a4w4 asm GEMM's ``beta * C`` term can actually be used for.

The kernel advertises ``D = alpha*A*B + beta*C`` with ``C`` behind its own
pointer, which would let a wgrad GEMM accumulate straight into ``main_grad`` and
retire the separate elementwise add that dominates the gradient path. The host
wrapper asserts a bf16 ``D`` and its two headers disagree on ``C`` ([1, N] f32 in
the Python stub, [M, N] f32 in the .cu), so what is supported has to be measured.

Probes, in order: does beta*C apply at all, which dtype and shape C accepts, and
whether C may alias D (the in-place accumulate we actually want).
"""

import torch

import aiter
from aiter import dtypes
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32
from aiter.ops.shuffle import shuffle_weight

SCALE_GROUP_SIZE = 32


def reference(a_fp4, b_fp4, a_scales, b_scales, m, n):
    a = mxfp4_to_f32(a_fp4) * e8m0_to_f32(
        a_scales[:m].repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    )
    b = mxfp4_to_f32(b_fp4) * e8m0_to_f32(
        b_scales[:n].repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    )
    return torch.mm(a, b.T)[:m, :n]


def make_operands(m, n, k):
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    a = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    b = torch.randn((n, k), dtype=dtypes.bf16, device="cuda")
    _, a_scales = quant(a, shuffle=False)
    _, b_scales = quant(b, shuffle=False)
    a_fp4, a_scales_shuffled = quant(a, shuffle=True)
    b_fp4, b_scales_shuffled = quant(b, shuffle=True)
    ref = reference(
        a_fp4, b_fp4, a_scales.view(torch.uint8), b_scales.view(torch.uint8), m, n
    )
    return (
        a_fp4,
        shuffle_weight(b_fp4, layout=(16, 16)),
        a_scales_shuffled,
        b_scales_shuffled,
        ref,
    )


def relative_error(got, want):
    return (
        (got.float() - want.float()).abs().max() / want.float().abs().max().clamp(min=1e-6)
    ).item()


def main():
    torch.manual_seed(0)
    m, n, k = 512, 512, 1024
    a, b, a_scale, b_scale, ref = make_operands(m, n, k)

    def run(out, bias=None, beta=0.0):
        aiter.gemm_a4w4_asm(
            a, b, a_scale, b_scale, out, "", bias, 1.0, beta, True, 0
        )
        return out

    print(f"shape M={m} N={n} K={k}\n")

    out = run(torch.empty((m, n), dtype=dtypes.bf16, device="cuda"))
    print(f"[1] plain, beta=0            rel_err={relative_error(out, ref):.4f}")

    for name, c in (
        ("C bf16 [M,N]", torch.randn((m, n), dtype=dtypes.bf16, device="cuda")),
        ("C fp32 [M,N]", torch.randn((m, n), dtype=torch.float32, device="cuda")),
        ("C fp32 [1,N]", torch.randn((1, n), dtype=torch.float32, device="cuda")),
    ):
        out = torch.empty((m, n), dtype=dtypes.bf16, device="cuda")
        try:
            run(out, bias=c, beta=1.0)
        except Exception as exc:  # the wrapper rejects some of these outright
            print(f"[2] {name}, beta=1        rejected: {type(exc).__name__}: {exc}")
            continue
        # Which of the two candidate meanings, if either, the kernel implemented.
        print(
            f"[2] {name}, beta=1        vs ref+C={relative_error(out, ref + c):.4f}"
            f"   vs ref alone={relative_error(out, ref):.4f}"
        )

    # The one that matters: C and D as the same buffer, i.e. main_grad += A@B.
    acc = torch.randn((m, n), dtype=dtypes.bf16, device="cuda")
    want = ref + acc.float()
    try:
        run(acc, bias=acc, beta=1.0)
        print(f"[3] C aliases D, beta=1       rel_err={relative_error(acc, want):.4f}")
    except Exception as exc:
        print(f"[3] C aliases D, beta=1       rejected: {type(exc).__name__}: {exc}")

    # An fp32 D is what a real main_grad needs; expected to be refused.
    try:
        run(torch.empty((m, n), dtype=torch.float32, device="cuda"))
        print("[4] D fp32                    accepted")
    except Exception as exc:
        print(f"[4] D fp32                    rejected: {str(exc).strip()[:90]}")


if __name__ == "__main__":
    main()

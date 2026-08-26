# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What ``v_prng_b32`` computes, and whether it is fit to drive stochastic rounding.

Report §5.19 found Philox to be 39% of the dual-layout quantizer's VALU
instructions (796 of 2045 on ``grad gate_up``: 386 ``v_xor_b32``, 210
``v_mul_lo_u32``, 200 ``v_mul_hi_u32``), which is ~30% of the kernel. gfx950 has a
one-instruction PRNG, and ``v_cvt_scalef32_sr_pk_fp4_f32`` already uses it
internally to derive the second FP4 lane's random value from the seed it is
handed — so half the SR noise in production today is already ``v_prng_b32``
output. Driving *all* lanes from it would replace ~12 VALU per random word with 1.

Before building on that, this establishes what the instruction is. It is not in
the ROCm headers and its function is not documented here, so:

  1. is it a pure function of one operand (same input -> same output, no state)?
  2. is a single application uniform over 32 bits?
  3. does *chaining* it stay uniform, and are successive states independent?

(3) is the one that matters, because the cheap construction is a few Philox words
per lane expanded by independent chains. A weak avalanche would show up as
correlation between chain steps, which SR would turn into a rounding bias.

Result on gfx950 (MI350X): **the substitution does not work.** ``v_prng_b32`` is a
pure, bijective, *linear* map -- almost certainly an xorshift step:

  - pure and bijective: 2^20 distinct inputs give 2^20 distinct outputs, and
    ``prng(0) == 0`` is a fixed point.
  - one application of it to *already uniform* input is uniform (worst bit
    |p-0.5| = 0.0011 against a 0.00195 tolerance, byte chi2/df = 1.00) --
    but applied to the counter sequence 0,1,2,... it fails completely
    (|p-0.5| = 0.5, chi2/df = 4112). It cannot manufacture entropy from a
    counter the way Philox does; it can only stir entropy it is given.
  - chaining keeps each state marginally uniform, yet every pair of states
    tested (0-1, 0-2, 0-8, 3-4, 7-15) has some bit pair agreeing 100% of the
    time. Popcount is not preserved, so it is not a bit permutation; a linear
    GF(2) map with a pass-through field, i.e. ``x ^= x << k``, fits everything
    observed.

The entropy argument is the fatal one and does not depend on the exact form: a
bijection on 32 bits means the whole chain (s, f(s), f^2(s), ...) is a
deterministic function of s and carries 32 bits of entropy however long it runs.
Dithering a 32-element quantization block from one chain would give all 32
elements deterministically related noise, and independence across the block is
the property SR exists to provide. Philox's per-element cost is buying real
entropy and there is no hardware shortcut to it.

This does not impugn what the hardware does internally:
``v_cvt_scalef32_sr_pk_fp4_f32`` deriving lane 2's dither from lane 1's seed
shares 32 bits across *two* adjacent elements, a deliberate and bounded
trade. Spreading it across 8-32 is a different proposition.

What remains is to make the entropy cheaper rather than fake it -- see report
§5.20 on ``SR_PHILOX_ROUNDS``, which is 7 against Philox's standard 10.

Run:
    python benchmarks/probe_prng_b32.py
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@triton.jit
def _prng_b32(x):
    """One gfx950 hardware PRNG step. Pure, elementwise, 1 VALU."""
    return tl.inline_asm_elementwise(
        asm="v_prng_b32 $0, $1;",
        constraints="=v,v",
        args=[x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _apply_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    tl.store(y_ptr + offs, _prng_b32(tl.load(x_ptr + offs, mask=m)), mask=m)


@triton.jit
def _chain_kernel(seed_ptr, out_ptr, N, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    """Write STEPS successive states of each lane's chain, stride N apart."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    s = tl.load(seed_ptr + offs, mask=m).to(tl.uint32, bitcast=True)
    for i in tl.range(STEPS):
        s = _prng_b32(s)
        tl.store(out_ptr + i * N + offs, s, mask=m)


def apply_once(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    _apply_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return y


def chain(seed: torch.Tensor, steps: int) -> torch.Tensor:
    n = seed.numel()
    out = torch.empty((steps, n), dtype=torch.int32, device=seed.device)
    _chain_kernel[(triton.cdiv(n, 1024),)](seed, out, n, STEPS=steps, BLOCK=1024)
    return out


def bits_of(u: torch.Tensor) -> torch.Tensor:
    """(..., 32) float tensor of the bits of a uint32-valued int32 tensor."""
    v = u.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, device=u.device, dtype=torch.int64)
    return ((v.reshape(-1, 1) >> shifts) & 1).float()


def report_uniformity(name: str, u: torch.Tensor) -> None:
    n = u.numel()
    b = bits_of(u)
    p = b.mean(0)
    # Per-bit mean should be 0.5; 4 sigma of a fair coin over n draws.
    tol = 4.0 * (0.25 / n) ** 0.5
    worst = (p - 0.5).abs().max().item()
    # Byte-level chi-square, 256 bins, as a coarse whole-word check.
    v = u.to(torch.int64) & 0xFFFFFFFF
    counts = torch.bincount((v & 0xFF).reshape(-1), minlength=256).float()
    exp = n / 256
    chi2 = (((counts - exp) ** 2) / exp).sum().item()
    print(f"  {name:<34} n={n:>9}  worst bit |p-.5|={worst:.5f} (tol {tol:.5f})"
          f"  {'OK' if worst < tol else 'FAIL':<4}  chi2/df={chi2 / 255:.3f}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(0)
    dev = "cuda"

    print("1. is it a pure function of its operand?")
    x = torch.randint(-(2**31), 2**31 - 1, (1 << 20,), dtype=torch.int32, device=dev)
    y1 = apply_once(x)
    y2 = apply_once(x)
    print(f"  same input twice -> identical: {torch.equal(y1, y2)}")
    z = apply_once(torch.zeros_like(x))
    print(f"  prng(0) constant across lanes: {bool((z == z[0]).all())}, value={z[0].item() & 0xFFFFFFFF:#010x}")
    print(f"  distinct inputs -> distinct outputs: "
          f"{len(torch.unique(y1)) == len(torch.unique(x))} "
          f"({len(torch.unique(y1))} of {len(torch.unique(x))} unique)")

    print("\n2. is one application uniform?")
    report_uniformity("prng(uniform random input)", y1)
    seq = torch.arange(1 << 20, dtype=torch.int32, device=dev)
    report_uniformity("prng(0,1,2,...) [worst case]", apply_once(seq))

    print("\n3. does chaining stay uniform, and are steps independent?")
    steps = 16
    seeds = torch.randint(-(2**31), 2**31 - 1, (1 << 18,), dtype=torch.int32, device=dev)
    ch = chain(seeds, steps)
    for i in (0, 1, 7, steps - 1):
        report_uniformity(f"chain step {i}", ch[i])

    # Bit-level correlation between step i and step j. For independent words,
    # each of the 32x32 bit pairs should agree half the time.
    print("\n  pairwise bit agreement between chain steps (0.5 = independent)")
    n = ch.shape[1]
    tol = 4.0 * (0.25 / n) ** 0.5
    for i, j in ((0, 1), (0, 2), (0, 8), (3, 4), (7, 15)):
        bi = bits_of(ch[i]).reshape(n, 32)
        bj = bits_of(ch[j]).reshape(n, 32)
        agree = (bi.T @ bj + (1 - bi).T @ (1 - bj)) / n
        worst = (agree - 0.5).abs().max().item()
        print(f"    step {i:>2} vs {j:>2}: worst |agree-0.5| = {worst:.5f} "
              f"(tol {tol:.5f})  {'OK' if worst < tol else 'FAIL'}")

    # A chain that returns to its seed would silently shrink the noise space.
    print("\n  chain does not revisit its seed within 16 steps: "
          f"{not bool((ch == seeds.unsqueeze(0)).any())}")
    uniq_per_lane = (ch.unsqueeze(0) == ch.unsqueeze(1)).float()
    dup = (uniq_per_lane.sum(dim=(0, 1)) > steps).float().mean().item()
    print(f"  lanes with any repeat inside their own 16 states: {dup * 100:.3f}%")

    # If popcount is invariant the instruction is a *bit permutation*: it moves
    # entropy around a word but cannot create any. That would make chaining
    # useless for SR no matter how it benchmarks, so test it explicitly.
    print("\n4. is it a bit permutation (moves entropy) or a mixer (creates it)?")
    pc_in = bits_of(x).sum(1)
    pc_out = bits_of(y1).sum(1)
    same_pc = bool(torch.equal(pc_in, pc_out))
    print(f"  popcount preserved for every input: {same_pc}")
    if same_pc:
        # Recover the permutation from one-hot inputs: bit i of the input must
        # land on exactly one bit of the output.
        onehot = (1 << torch.arange(32, device=dev)).to(torch.int32)
        mapped = apply_once(onehot)
        perm = []
        for i in range(32):
            v = int(mapped[i].item()) & 0xFFFFFFFF
            perm.append(v.bit_length() - 1 if v else -1)
        print(f"  bit i -> bit perm[i]: {perm}")
        print(f"  is a pure permutation of all 32 bits: {sorted(perm) == list(range(32))}")
        rot = [(p - i) % 32 for i, p in enumerate(perm)]
        print(f"  constant rotation? {len(set(rot)) == 1}"
              + (f" (rotate left by {rot[0]})" if len(set(rot)) == 1 else ""))

    print("\n5. cost, against the Philox it would replace")
    big = torch.randint(-(2**31), 2**31 - 1, (1 << 24,), dtype=torch.int32, device=dev)
    from benchmarks.bench_utils import cuda_timer

    ms_chain = cuda_timer(lambda: chain(big, 8), warmup=5, iters=20).median_ms
    print(f"  8 chained states of 2^24 lanes: {ms_chain:.3f} ms")


if __name__ == "__main__":
    main()

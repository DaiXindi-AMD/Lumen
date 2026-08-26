# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the MXFP4 dual-layout quantizer compiles to, and where its cycles go.

Report §5.19 closed both launch-parameter sweeps as negative and attributed the
kernel's time to its features (stochastic rounding 22-26% of the gradient path,
the MFMA Hadamard load-bearing). That says *which computation* to attack but not
whether a hand-written HIP kernel can beat Triton at it. The deciding question is
what the compiler is already doing: a kernel that spills has an obvious target,
one that is ALU-saturated at full occupancy does not, and 2x is only plausible in
the first case.

So this prints, per production shape and recipe, the compiled resource usage
(VGPRs, spills, LDS, occupancy) and a breakdown of the AMDGCN by instruction
class. Occupancy is derived, not measured: gfx95x has 512 VGPRs per SIMD, so
waves/SIMD is floor(512 / vgprs_rounded_to_8), capped at 8 for a 4-warp launch.

Run:
    python benchmarks/inspect_dual_layout_asm.py [--dump-asm SHAPE]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.bench_dual_layout_tiles import (  # noqa: E402
    BLOCK,
    G,
    SHAPES,
    TOKENS,
    _alloc,
    _default_blocks,
)
from lumen.ops.quantize.ops import (  # noqa: E402
    _rht_matrix_bf16,
    is_cdna4,
    mxfp4_data_shuffle_supported,
    mxfp4_scale_swizzle_supported,
)

# gfx950 VGPR file per SIMD, and the granularity allocation rounds to.
VGPRS_PER_SIMD = 512
VGPR_ALLOC_GRAN = 8

# Instruction classes that tell us something about this kernel specifically.
# Order matters: first match wins, so put the specific patterns first.
CLASSES = [
    ("mfma", re.compile(r"^v_mfma")),
    ("transcendental", re.compile(r"^v_(rcp|sqrt|rsq|log|exp|sin|cos)")),
    ("permlane/swizzle", re.compile(r"^(v_permlane|v_perm_|ds_swizzle|ds_bpermute|ds_permute)")),
    ("cvt (pack/convert)", re.compile(r"^v_cvt")),
    ("global load", re.compile(r"^(global_load|buffer_load|flat_load)")),
    ("global store", re.compile(r"^(global_store|buffer_store|flat_store)")),
    ("LDS", re.compile(r"^ds_(read|write)")),
    ("valu", re.compile(r"^v_")),
    ("salu", re.compile(r"^s_(add|sub|mul|and|or|xor|lshl|lshr|ashr|cmp|cselect|mov|bfe|min|max|not|abs)")),
    ("branch/wait", re.compile(r"^s_(branch|cbranch|waitcnt|barrier|setprio|nop|endpgm|sleep|delay)")),
]


def classify(asm: str) -> tuple[Counter, int]:
    counts = Counter()
    total = 0
    for raw in asm.splitlines():
        line = raw.strip()
        if not line or line.startswith((".", ";", "/", "//")) or line.endswith(":"):
            continue
        mnemonic = line.split()[0]
        if not (mnemonic.startswith("v_") or mnemonic.startswith("s_")
                or mnemonic.startswith(("global_", "buffer_", "flat_", "ds_"))):
            continue
        total += 1
        for name, pat in CLASSES:
            if pat.match(mnemonic):
                counts[name] += 1
                break
        else:
            counts["other"] += 1
    return counts, total


def occupancy(vgprs: int) -> int:
    if vgprs <= 0:
        return 8
    rounded = -(-vgprs // VGPR_ALLOC_GRAN) * VGPR_ALLOC_GRAN
    return max(1, min(8, VGPRS_PER_SIMD // rounded))


def compile_one(M: int, N: int, recipe: str):
    """Compile at exactly what production launches for this shape."""
    from lumen.kernels.mxfp4 import _dual_layout_quant_mxfp4_kernel

    use_sr = recipe == "grad"
    swizzle = (mxfp4_scale_swizzle_supported(M, N // BLOCK)
               and mxfp4_scale_swizzle_supported(N, M // BLOCK))
    shuffle = recipe == "act" and mxfp4_data_shuffle_supported(N, M // 2)
    bm, bn = _default_blocks(M, N)

    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    a, a_s, b, b_s = _alloc(M, N, swizzle)
    sign = torch.randint(0, 2, (G,), device="cuda", dtype=torch.float32) * 2 - 1
    hmat = _rht_matrix_bf16(sign, G)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

    compiled = _dual_layout_quant_mxfp4_kernel[grid](
        x, a, a_s, b, b_s, sign, hmat,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1), a_s.stride(0), a_s.stride(1),
        b.stride(0), b.stride(1), b_s.stride(0), b_s.stride(1),
        1234, 0, 0x9E3779B9,
        BLOCK_M=bm, BLOCK_N=bn,
        QUANT_BLOCK_SIZE=BLOCK,
        USE_SR_A=use_sr, USE_SR_B=use_sr,
        USE_ASM=is_cdna4(),
        SWIZZLE_SCALE=swizzle,
        NUM_SCALE_COLS_A=N // BLOCK, NUM_SCALE_COLS_B=M // BLOCK,
        SHUFFLE_B=shuffle, NUM_PACKED_COLS_B=M // 2,
        USE_MFMA=True,
        num_warps=4,
    )
    del x, a, a_s, b, b_s
    torch.cuda.empty_cache()
    return compiled, (bm, bn), grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-asm", metavar="SUBSTR",
                    help="write the AMDGCN of the first shape whose label contains SUBSTR")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    print(f"{'shape':<22} {'tile':<10} {'vgpr':>5} {'agpr':>5} {'spill':>6} "
          f"{'lds':>6} {'occ':>4} {'insts':>7} {'waves':>7}")
    print("-" * 88)

    compiled_all = {}
    for label, N, recipe in SHAPES:
        M = TOKENS
        compiled, (bm, bn), grid = compile_one(M, N, recipe)
        meta = compiled.metadata
        n_regs = getattr(meta, "num_gprs", None) or getattr(compiled, "n_regs", 0)
        n_spills = getattr(compiled, "n_spills", 0)
        lds = getattr(meta, "shared", 0)
        # AGPRs matter here: the MFMA Hadamard puts its accumulator in them, and
        # on gfx95x they come out of the same file as VGPRs.
        agpr = getattr(meta, "num_agprs", 0) or 0
        asm = compiled.asm.get("amdgcn", "")
        counts, total = classify(asm)
        waves = grid[0] * grid[1] * 4
        print(f"{label:<22} ({bm:>3},{bn:>3}) {n_regs:>5} {agpr:>5} {n_spills:>6} "
              f"{lds:>6} {occupancy(n_regs + agpr):>4} {total:>7} {waves:>7}")
        compiled_all[label] = (asm, counts, total)

    print()
    print("instruction mix, share of counted instructions")
    names = [n for n, _ in CLASSES] + ["other"]
    header = "".join(f"{n.split()[0][:9]:>11}" for n in names)
    print(f"{'shape':<22}{header}")
    print("-" * (22 + 11 * len(names)))
    for label, (_, counts, total) in compiled_all.items():
        row = "".join(f"{counts.get(n, 0) / max(total, 1) * 100:>10.1f}%" for n in names)
        print(f"{label:<22}{row}")

    print()
    print("absolute counts (per wave, i.e. per 4-warp program instance)")
    print(f"{'shape':<22}{header}")
    print("-" * (22 + 11 * len(names)))
    for label, (_, counts, total) in compiled_all.items():
        row = "".join(f"{counts.get(n, 0):>11}" for n in names)
        print(f"{label:<22}{row}")

    if args.dump_asm:
        for label, (asm, _, _) in compiled_all.items():
            if args.dump_asm in label:
                path = f"/tmp/dual_layout_{label.replace('/', '_').replace(' ', '_')}.s"
                with open(path, "w") as f:
                    f.write(asm)
                print(f"\nwrote {len(asm.splitlines())} lines of AMDGCN to {path}")
                break

    print()
    print("Reading this: a spill count above zero, or an occupancy of 1-2 waves/SIMD,")
    print("means the compiler is losing to register pressure and a hand-written kernel")
    print("has somewhere to go. Zero spills at 4+ waves with a VALU-dominated mix means")
    print("the arithmetic is the wall and a rewrite has to remove work, not schedule it")
    print("better.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Get a model's MXFP4 GEMM shapes into AITER's tuned table, and prove they are correct.

The/tmp/qwen3-8b.untuned.csv prebuilt A4W4 ASM/CK kernels are only reachable for shapes present in
``a4w4_blockscale_tuned_gemm.csv``. Anything else falls back to Triton -- and must,
because for an untuned shape AITER picks a default kernel that is not validated:
at (64, 64, 128) it silently returns garbage. So widening the fast path for a new
model means adding rows to that table, then proving each new row is exact.

Four stages, run in order:

    # 1. Observe the shapes the model actually issues (all three GEMMs per linear)
    python scripts/mxfp4_tune_shapes.py collect --hf Qwen/Qwen3-8B --tokens 8192 \
        --out /tmp/qwen3-8b.shapes.csv

    # 2. Keep only what AITER has no tuned entry for
    python scripts/mxfp4_tune_shapes.py untuned --shapes /tmp/qwen3-8b.shapes.csv \
        --out 

    # 3. Tune them (long; needs a clang for the CK JIT build)
    python scripts/mxfp4_tune_shapes.py tune --untuned /tmp/qwen3-8b.untuned.csv \
        --out /tmp/qwen3-8b.tuned.csv --clang /path/to/llvm/bin

    # 4. Reject any row that is not bit-exact against the plain Triton kernel
    python scripts/mxfp4_tune_shapes.py verify --tuned /tmp/qwen3-8b.tuned.csv \
        --out lumen_a4w4_blockscale_tuned_gemm.csv

Then point AITER at the result, before importing it (the lookup is cached):

    export AITER_CONFIG_GEMM_A4W4=/abs/lumen_a4w4_blockscale_tuned_gemm.csv:\\
$AITER_ROOT/aiter/configs/a4w4_blockscale_tuned_gemm.csv
    export AITER_REBUILD=1   # once, so the CK kernels for the new rows get built

Stage 1 runs the real ``quantized_linear`` rather than deriving shapes on paper.
Every linear issues three GEMMs and the backward pair permutes the dimensions --
a wgrad's M is the output width and its K is the token count -- which is easy to
get wrong by hand and impossible to get wrong by observation.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MXFP4_BLOCK = 32


# ---------------------------------------------------------------------------
# Stage 1: collect
# ---------------------------------------------------------------------------


def _load_hf_dims(name_or_path):
    """(hidden, intermediate, q_dim, kv_dim) from a HuggingFace config.json."""
    import glob

    candidates = [
        os.path.join(name_or_path, "config.json"),
        *glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--"
                + name_or_path.replace("/", "--")
                + "/snapshots/*/config.json"
            )
        ),
        *glob.glob(
            "/root/.cache/huggingface/hub/models--"
            + name_or_path.replace("/", "--")
            + "/snapshots/*/config.json"
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            head_dim = cfg.get("head_dim") or (
                cfg["hidden_size"] // cfg["num_attention_heads"]
            )
            return (
                cfg["hidden_size"],
                cfg["intermediate_size"],
                cfg["num_attention_heads"] * head_dim,
                cfg.get("num_key_value_heads", cfg["num_attention_heads"]) * head_dim,
            )
    raise SystemExit(f"no config.json found for {name_or_path}")


def _projections(hidden, inter, q_dim, kv_dim, tp):
    """(name, out_features, in_features) per linear, sharded for tensor parallel.

    Column-parallel layers split the output width; row-parallel ones split the
    reduction dim. Both change the GEMM shapes, so tuning has to be done for the
    parallelism the job will actually run with.
    """
    return [
        ("q_proj", q_dim // tp, hidden),
        ("k_proj", kv_dim // tp, hidden),
        ("v_proj", kv_dim // tp, hidden),
        ("o_proj", hidden, q_dim // tp),
        ("gate_proj", inter // tp, hidden),
        ("up_proj", inter // tp, hidden),
        ("down_proj", hidden, inter // tp),
    ]


def cmd_collect(args):
    import torch

    os.environ["LUMEN_MXFP4_GEMM_SHAPE_LOG"] = args.out
    # Measuring backends here would waste time and pollute nothing useful; we only
    # want the shapes.
    os.environ.setdefault("LUMEN_MXFP4_AUTOTUNE", "0")

    from lumen.ops.quantize import mxfp4_autotune
    from lumen.ops.quantize.linear import quantized_linear

    if args.hf:
        hidden, inter, q_dim, kv_dim = _load_hf_dims(args.hf)
    else:
        hidden, inter, q_dim, kv_dim = (
            args.hidden, args.intermediate, args.q_dim, args.kv_dim
        )
    print(
        f"model dims: hidden={hidden} intermediate={inter} "
        f"q_dim={q_dim} kv_dim={kv_dim} tp={args.tp}"
    )

    for name, out_f, in_f in _projections(hidden, inter, q_dim, kv_dim, args.tp):
        for tokens in args.tokens:
            x = torch.randn(
                tokens, in_f, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            w = torch.randn(
                out_f, in_f, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            try:
                y = quantized_linear(x, w, scaling_type="mxfp4")
                y.sum().backward()
            except Exception as e:
                print(f"  {name} M={tokens}: skipped ({type(e).__name__}: {e})")
            del x, w
            torch.cuda.empty_cache()
        print(f"  {name}: out={out_f} in={in_f} done")

    mxfp4_autotune._save_shape_log()
    print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------
# Stage 2: untuned
# ---------------------------------------------------------------------------


def cmd_untuned(args):
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config

    rows = list(csv.DictReader(open(args.shapes)))
    missing, have = [], 0
    for r in rows:
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        if get_GEMM_config(M, N, K) is not None:
            have += 1
        else:
            missing.append((M, N, K))

    # The tuner keys on the exact triple, so duplicates just waste tuning time.
    missing = sorted(set(missing))
    with open(args.out, "w") as f:
        f.write("M,N,K\n")
        for M, N, K in missing:
            f.write(f"{M},{N},{K}\n")

    print(f"{len(rows)} shapes: {have} already tuned, {len(missing)} to tune")
    print(f"wrote {args.out}")
    for M, N, K in missing:
        print(f"  M={M:<7} N={N:<7} K={K:<7}  weight={N * K / 2 / 1024 / 1024:.1f} MiB")


# ---------------------------------------------------------------------------
# Stage 3: tune
# ---------------------------------------------------------------------------


def cmd_tune(args):
    aiter_root = args.aiter_root or os.environ.get("AITER_ROOT_DIR")
    if not aiter_root:
        import aiter

        aiter_root = str(Path(aiter.__file__).resolve().parent.parent)

    script = Path(aiter_root) / "csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py"
    if not script.exists():
        raise SystemExit(f"tuner not found at {script}")

    env = dict(os.environ)
    if args.clang:
        env["GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH"] = args.clang

    cmd = [
        sys.executable, str(script),
        "-i", os.path.abspath(args.untuned),
        "-o", os.path.abspath(args.out),
    ]
    if args.splitk:
        cmd.append("-k")
    print("running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=aiter_root, env=env))


# ---------------------------------------------------------------------------
# Stage 4: verify
# ---------------------------------------------------------------------------


def cmd_verify(args):
    import torch

    from lumen.ops.quantize.linear import _gemm_mxfp4_aiter, _gemm_mxfp4_aiter_asm
    from lumen.ops.quantize.ops import convert_to_mxfp4

    from aiter.ops.triton.utils._triton.arch_info import get_arch

    arch = get_arch()
    rows = list(csv.DictReader(open(args.tuned)))
    kept, dropped = [], []

    torch.manual_seed(0)
    for r in rows:
        if r.get("gfx") and r["gfx"] != arch:
            kept.append(r)  # not ours to judge on this GPU
            continue
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        try:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
            x_fp4, x_s = convert_to_mxfp4(x, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
            w_fp4, w_s = convert_to_mxfp4(w, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
            del x, w

            ref = _gemm_mxfp4_aiter(x_fp4, w_fp4, x_s, w_s)
            got = _gemm_mxfp4_aiter_asm(x_fp4, w_fp4, x_s, w_s)
            torch.cuda.synchronize()
            n_diff = (got != ref).sum().item()
            del x_fp4, x_s, w_fp4, w_s, ref, got
        except Exception as e:
            n_diff, note = -1, f"{type(e).__name__}: {str(e)[:60]}"
        else:
            note = "bit-exact" if n_diff == 0 else f"{n_diff} elems differ"
        torch.cuda.empty_cache()

        status = "keep" if n_diff == 0 else "DROP"
        print(f"  {status}  M={M:<7} N={N:<7} K={K:<7}  {note}")
        (kept if n_diff == 0 else dropped).append(r)

    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(kept)

    print(f"\n{len(kept)} kept, {len(dropped)} dropped -> {args.out}")
    if dropped:
        print(
            "dropped rows are excluded so the dispatcher never routes those shapes "
            "to a kernel that does not reproduce the Triton reference exactly."
        )
        return 1
    return 0


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="record the shapes a model's MXFP4 linears issue")
    c.add_argument("--hf", help="HuggingFace model id or local dir with config.json")
    c.add_argument("--hidden", type=int)
    c.add_argument("--intermediate", type=int)
    c.add_argument("--q-dim", type=int, dest="q_dim")
    c.add_argument("--kv-dim", type=int, dest="kv_dim")
    c.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    c.add_argument("--tokens", type=int, nargs="+", default=[8192],
                   help="token counts (micro_batch * seq_len) to record")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_collect)

    u = sub.add_parser("untuned", help="filter collected shapes to those AITER has not tuned")
    u.add_argument("--shapes", required=True)
    u.add_argument("--out", required=True)
    u.set_defaults(func=cmd_untuned)

    t = sub.add_parser("tune", help="run AITER's A4W4 tuner over the untuned shapes")
    t.add_argument("--untuned", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--clang", help="dir holding clang for the CK JIT build")
    t.add_argument("--aiter-root", dest="aiter_root")
    t.add_argument("--splitk", action="store_true", help="also try split-K variants")
    t.set_defaults(func=cmd_tune)

    v = sub.add_parser("verify", help="drop tuned rows that are not bit-exact vs plain Triton")
    v.add_argument("--tuned", required=True)
    v.add_argument("--out", required=True)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()

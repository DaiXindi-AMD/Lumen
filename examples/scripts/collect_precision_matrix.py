#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Turn a set of Megatron benchmark logs into one precision-comparison table.

Reads the ``iteration ... elapsed time per iteration (ms)`` lines every Megatron
run prints and reports, per (model, precision):

    median step time over the steps after the warmup, peak `mem usages`
    fraction, first and last lm loss, and the NaN / skipped-iteration counts.

The median rather than the mean, and only after the warmup, because the first
handful of steps carry JIT compilation and autotune probing and the validation
steps are several times a train step -- a mean over the whole run mostly measures
those (docs/mxfp4_feature_parity_plan.md §5).

Usage:
    python examples/scripts/collect_precision_matrix.py            # find logs
    python examples/scripts/collect_precision_matrix.py --warmup 10 --markdown
    python examples/scripts/collect_precision_matrix.py LOG [LOG ...]
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
MEM_RE = re.compile(r"mem usages:\s*([0-9.]+)")
LOSS_RE = re.compile(r"lm loss:\s*([0-9.eE+\-]+)")
SKIPPED_RE = re.compile(r"number of skipped iterations:\s*(\d+)")
NAN_RE = re.compile(r"number of nan iterations:\s*(\d+)")
# Megatron dumps its effective args at startup as "name ....... value". Reading
# the batch shape from there rather than assuming it is what stops a run measured
# at a different GBS from being divided by this matrix's BF16 baseline: step time
# scales with tokens/step, so such a ratio looks like a speedup and is not one.
ARG_RE = re.compile(r"^\s{2}(global_batch_size|micro_batch_size|seq_length|num_layers)\s\.+\s(\d+)\s*$")
PARAMS_RE = re.compile(r"number of parameters on \(tensor, pipeline\) model parallel rank \(0, 0\):\s*(\d+)")
# lumen_<model>[_<tag>]_<precision>.log
NAME_RE = re.compile(r"lumen_(llama2_7b|llama31_8b|qwen3_8b)(?:_(.+?))?_(bf16|fp8|mxfp4)\.log$")
# The BF16 / FP8 logs committed with the reference report, measured on 8xMI325X.
# Tagged rather than dropped: they are the only record of the FP8 path working as
# advertised, and every ratio here is computed within a tag so they never get
# silently averaged against a run from this machine.
REF_NAME_RE = re.compile(r"(llama2_7b|llama31_8b|qwen3_8b)_pretrain_(bf16|fp8)(?:_delayed)?\.log$")
REF_TAG = "mi325x-ref"

PRECISION_ORDER = {"bf16": 0, "fp8": 1, "mxfp4": 2}
MODEL_ORDER = {"llama2_7b": 0, "llama31_8b": 1, "qwen3_8b": 2}


def parse_log(path, warmup):
    steps, mems, losses = [], [], []
    skipped = nan = 0
    cfg = {}
    with open(path, errors="replace") as f:
        for line in f:
            am = ARG_RE.match(line)
            if am:
                cfg.setdefault(am.group(1), int(am.group(2)))
            m = ITER_RE.search(line)
            if m:
                steps.append((int(m.group(1)), float(m.group(3))))
                mm = MEM_RE.search(line)
                if mm:
                    mems.append(float(mm.group(1)))
                lm = LOSS_RE.search(line)
                if lm:
                    losses.append(float(lm.group(1)))
            sm = SKIPPED_RE.search(line)
            if sm:
                skipped = max(skipped, int(sm.group(1)))
            nm = NAN_RE.search(line)
            if nm:
                nan = max(nan, int(nm.group(1)))
            pm = PARAMS_RE.search(line)
            if pm:
                cfg.setdefault("params", int(pm.group(1)))

    steady = [ms for it, ms in steps if it > warmup]
    return {
        "iters": len(steps),
        "last_iter": steps[-1][0] if steps else 0,
        "median_ms": statistics.median(steady) if steady else None,
        "mean_ms": statistics.fmean(steady) if steady else None,
        "min_ms": min(steady) if steady else None,
        "n_steady": len(steady),
        "peak_mem": max(mems) if mems else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "skipped": skipped,
        "nan": nan,
        "gbs": cfg.get("global_batch_size"),
        "seq_len": cfg.get("seq_length"),
        "params": cfg.get("params"),
        "tokens": (cfg["global_batch_size"] * cfg["seq_length"]
                   if "global_batch_size" in cfg and "seq_length" in cfg else None),
    }


def gemm_backends(path, model, tag):
    """How many MXFP4 GEMM shapes reached AITER's ASM kernel, from the autotune cache.

    The cache is the only durable record of this: the per-shape probe runs once and
    a later run that finds the cache logs nothing about which backend it picked. It
    matters because the ASM kernels are only reachable for shapes present in the
    tuned table, so a model without one silently runs most of its GEMMs on Triton
    and its step time understates the path (docs/mxfp4_feature_parity_plan.md §13).
    """
    # An arm that pointed the cache somewhere of its own -- which any A/B over the
    # tuned table has to do, since the cache pins the backend choice -- names it
    # after its tag. Preferring that over the per-model default is what keeps such
    # a run from being credited with the default's ASM coverage.
    names = [f"mxfp4_autotune_{model}_{tag}.json"] if tag else []
    names.append(f"mxfp4_autotune_{model}.json")
    for name in names:
        try:
            with open(os.path.join(os.path.dirname(path), name)) as f:
                choices = json.load(f).get("choices", {})
        except (OSError, ValueError):
            continue
        if choices:
            break
    else:
        return None
    if not choices:
        return None
    return f"asm {sum(1 for v in choices.values() if v == 'asm')}/{len(choices)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", help="log files; default: examples/*/results/lumen_*.log")
    p.add_argument("--warmup", type=int, default=10,
                   help="ignore iterations <= this when taking the median")
    p.add_argument("--markdown", action="store_true", help="emit a markdown table")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--all-tags", action="store_true",
                   help="include the historical tagged A/B logs, not just the matrix")
    args = p.parse_args()

    logs = args.logs
    if not logs:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        logs = sorted(glob.glob(os.path.join(root, "*", "results", "lumen_*.log")))
        logs += sorted(glob.glob(os.path.join(root, "*", "results", "*_pretrain_*.log")))
    if not logs:
        sys.exit("no logs found")

    rows = []
    for path in logs:
        name = os.path.basename(path)
        m = NAME_RE.search(name)
        if m:
            model, tag, precision = m.group(1), m.group(2) or "", m.group(3)
        elif REF_NAME_RE.search(name):
            m = REF_NAME_RE.search(name)
            model, tag, precision = m.group(1), REF_TAG, m.group(2)
        else:
            continue
        # Smoke passes exist to prove a recipe runs at all; including them would
        # mix 3-step runs into a table of 50-step medians.
        if tag.startswith("smoke"):
            continue
        st = parse_log(path, args.warmup)
        st.update(model=model, precision=precision, tag=tag, path=path,
                  gemm=gemm_backends(path, model, tag) if precision == "mxfp4" else None)
        st["tok_s_gpu"] = (
            st["tokens"] / (st["median_ms"] / 1000.0) / args.gpus
            if st["median_ms"] and st["tokens"] else None
        )
        # 6*P*tokens is the reference report's formula, kept identical so its TE and
        # Lumen columns stay comparable with these. It counts only the GEMMs' 2 FLOPs
        # per MAC over fwd + 2x bwd and ignores attention, so it is a throughput
        # index rather than achieved FLOPs -- fine for ratios, low by a few percent
        # in absolute terms, and increasingly so at long sequence length.
        st["tflops"] = (
            6 * st["params"] * st["tokens"] / (st["median_ms"] / 1000.0) / args.gpus / 1e12
            if st["median_ms"] and st["tokens"] and st["params"] else None
        )
        rows.append(st)

    if not args.all_tags:
        rows = [r for r in rows if not r["tag"] or r["tag"] in (REF_TAG,)
                or r["tag"].startswith("rep")]

    rows.sort(key=lambda r: (MODEL_ORDER.get(r["model"], 9),
                             PRECISION_ORDER.get(r["precision"], 9), r["tag"]))

    # BF16 is the only baseline every model has, so speedups are quoted against it,
    # matched on tag first: a repeat or a run from other hardware must be divided
    # by its own BF16, never by another arm's.
    bf16 = {(r["model"], r["tag"]): r for r in rows if r["precision"] == "bf16"}

    def baseline_for(row):
        base = bf16.get((row["model"], row["tag"])) or bf16.get((row["model"], ""))
        if base is None or base["tokens"] != row["tokens"]:
            return None
        return base["median_ms"]

    def fmt(v, spec=".1f"):
        return "--" if v is None else format(v, spec)

    hdr = ["model", "precision", "tag", "iters", "gbs x seq", "median ms", "vs BF16",
           "TFLOP/s/GPU", "tok/s/GPU", "peak mem", "loss", "nan/skip", "gemm"]
    table = []
    for r in rows:
        base = baseline_for(r)
        spd = (f"{base / r['median_ms']:.3f}x"
               if base and r["median_ms"] else "--")
        table.append([
            r["model"], r["precision"], r["tag"] or "-", str(r["iters"]),
            f"{r['gbs'] or '?'}x{r['seq_len'] or '?'}",
            fmt(r["median_ms"]), spd, fmt(r["tflops"], ".1f"), fmt(r["tok_s_gpu"], ".0f"),
            fmt(r["peak_mem"], ".4f"),
            f"{fmt(r['loss_first'], '.3f')}->{fmt(r['loss_last'], '.3f')}",
            f"{r['nan']}/{r['skipped']}",
            r["gemm"] or "-",
        ])

    if args.markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for row in table:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(hdr[i]), *(len(r[i]) for r in table)) if table else len(hdr[i])
                  for i in range(len(hdr))]
        print("  ".join(h.ljust(w) for h, w in zip(hdr, widths)))
        print("  ".join("-" * w for w in widths))
        for row in table:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    incomplete = [r for r in rows if r["n_steady"] == 0]
    if incomplete:
        print("\nruns with no steady-state steps (crashed or too short):", file=sys.stderr)
        for r in incomplete:
            print(f"  {r['path']}", file=sys.stderr)


if __name__ == "__main__":
    main()

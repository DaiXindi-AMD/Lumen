###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Read step times out of the ablation ladder's logs.

Two modes: ``--arm-dir`` summarises one run right after it finishes, and
``--ladder`` builds the staircase table across arms.

The measurement follows docs/mxfp4_ablation_plan.md §6: drop the first
``--warmup`` iterations, which carry Triton compilation and the autotune probes,
and take the median of the rest rather than the mean, so one stalled step cannot
move the number.
"""

import argparse
import json
import pathlib
import re
import statistics

_ITER = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
)
_MEM = re.compile(r"mem usages:\s*([\d.]+)")
_LOSS = re.compile(r"lm loss:\s*([\dE.+-]+)")
_BACKEND = re.compile(r"mxfp4.*?(asm|shuffled|plain)", re.IGNORECASE)


def parse_log(path):
    """Per-iteration step time, memory and loss, each tagged with its iteration.

    Everything is keyed by iteration so the warmup can be dropped consistently.
    Memory especially: the first iterations spike while the allocator settles --
    one arm touched 1.0 there and held 0.6179 for the rest of the run -- so a peak
    taken over the whole log describes the warmup, not the arm.
    """
    steps, mems, losses = [], [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _ITER.search(line)
            if not m:
                continue
            it = int(m.group(1))
            steps.append((it, float(m.group(2))))
            mm = _MEM.search(line)
            if mm:
                mems.append((it, float(mm.group(1))))
            ml = _LOSS.search(line)
            if ml:
                losses.append((it, float(ml.group(1))))
    return steps, mems, losses


def summarize(path, warmup):
    steps, mems, losses = parse_log(path)
    kept = [ms for it, ms in steps if it > warmup]
    kept_mem = [v for it, v in mems if it > warmup]
    kept_loss = [v for it, v in losses if it > warmup]
    if not kept:
        return None
    median = statistics.median(kept)
    ordered = sorted(kept)
    # Interquartile, not min-max or p10-p90: the eval interval lands a 1.3x step
    # every few iterations, up to ~15% of the window, and anything wider than the
    # quartiles reports those as instability in an otherwise flat run.
    p25 = ordered[int(0.25 * (len(ordered) - 1))]
    p75 = ordered[int(0.75 * (len(ordered) - 1))]
    return {
        "iterations": len(steps),
        "measured": len(kept),
        "median_ms": round(median, 1),
        "p25_ms": round(p25, 1),
        "p75_ms": round(p75, 1),
        "max_ms": round(max(kept), 1),
        "spread_pct": round(100 * (p75 - p25) / median, 1),
        # The mean is kept alongside the median because they disagree when an arm
        # stalls periodically rather than uniformly: the median then describes the
        # good steps and hides a real cost. A ratio far from its neighbours' is
        # the signal to go read the step sequence.
        "mean_ms": round(statistics.mean(kept), 1),
        "mean_over_median": round(statistics.mean(kept) / median, 3),
        "slow_steps": sum(1 for x in kept if x > 1.3 * median),
        "peak_mem_frac": max(kept_mem) if kept_mem else None,
        "final_loss": kept_loss[-1] if kept_loss else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", type=pathlib.Path)
    ap.add_argument("--ladder", type=pathlib.Path)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    if args.arm_dir:
        log = args.arm_dir / "train.log"
        if not log.is_file():
            print(f"  no train.log in {args.arm_dir}")
            return
        s = summarize(log, args.warmup)
        if s is None:
            print(f"  no iterations logged in {log}")
            return
        (args.arm_dir / "summary.json").write_text(json.dumps(s, indent=2))
        print(
            f"  {s['median_ms']} ms/step median over {s['measured']} steps"
            f" (spread {s['spread_pct']}%), peak mem {s['peak_mem_frac']},"
            f" loss {s['final_loss']}"
        )
        return

    root = args.ladder or pathlib.Path(__file__).resolve().parent.parent / "results" / "ablation"
    arms = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.startswith("S")),
        key=lambda d: int(d.name[1:]),
    )
    rows, baseline = [], None
    for arm in arms:
        runs, label = [], ""
        for run in sorted(arm.glob("run*")):
            if not label and (run / "arm.txt").is_file():
                label = (run / "arm.txt").read_text().strip()
            log = run / "train.log"
            if log.is_file():
                s = summarize(log, args.warmup)
                if s:
                    runs.append(s)
        if not runs:
            continue
        best = min(r["median_ms"] for r in runs)
        if baseline is None:
            baseline = best
        rows.append((arm.name, label, runs, best))

    print(f"{'arm':5} {'ms/step':>9} {'vs S0':>8} {'step':>7} {'runs':>5}  what it turns on")
    prev = None
    for name, label, runs, best in rows:
        cum = f"{baseline / best:.3f}x" if baseline else "-"
        step = f"{prev / best:.3f}x" if prev else "-"
        spread = ""
        if len(runs) > 1:
            times = [r["median_ms"] for r in runs]
            spread = f" [{min(times)}-{max(times)}]"
        print(f"{name:5} {best:9.1f} {cum:>8} {step:>7} {len(runs):5}  {label}{spread}")
        prev = best


if __name__ == "__main__":
    main()

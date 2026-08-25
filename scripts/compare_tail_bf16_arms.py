"""Compare BF16-tail arms on step time and on the loss curve.

The two questions are answered by different parts of a run and have different
noise floors, so they are reported separately.

Step time: a single arm's median over 15 steady iterations was measured to move
by 54 ms run to run, which is the same size as the per-layer effect being looked
for. Longer runs help, so the spread across the steady window is printed next to
the median rather than the median alone.

Loss: the arms share seed, corpus and sample order, so the curves are paired and
small differences are meaningful. What they cannot show is the failure the BF16
tail exists to prevent -- report §4.3 saw divergence only after a long horizon at
a higher learning rate.
"""

import argparse
import re
import statistics

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([\d.]+)"
    r".*?lm loss: ([\d.E+-]+).*?grad norm: ([\d.naN]+)"
    r".*?number of skipped iterations:\s*(\d+)"
    r".*?number of nan iterations:\s*(\d+)"
)
VAL_RE = re.compile(
    r"validation loss at iteration (\d+) \| lm loss value: ([\d.E+-]+)"
)


def parse(path):
    iters, val = {}, {}
    with open(path, errors="replace") as handle:
        for line in handle:
            m = ITER_RE.search(line)
            if m:
                iters[int(m.group(1))] = {
                    "ms": float(m.group(2)),
                    "loss": float(m.group(3)),
                    "gnorm": m.group(4),
                    "skipped": int(m.group(5)),
                    "nan": int(m.group(6)),
                }
            m = VAL_RE.search(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    return iters, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", metavar="LABEL=LOG")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Iterations to drop before timing (compile and autotune).")
    args = parser.parse_args()

    data = {}
    for spec in args.arms:
        label, path = spec.split("=", 1)
        data[label] = parse(path)

    print("=== step time ===")
    print(f"{'arm':10} {'iters':>6} {'median':>9} {'p25':>9} {'p75':>9} {'min':>9} {'vs base':>9}")
    base = None
    for label, (iters, _) in data.items():
        times = [v["ms"] for k, v in sorted(iters.items()) if k > args.warmup]
        if not times:
            print(f"{label:10} {'-':>6}  no steady iterations yet")
            continue
        med = statistics.median(times)
        quartiles = statistics.quantiles(times, n=4) if len(times) >= 4 else [med] * 3
        if base is None:
            base = med
        print(f"{label:10} {len(times):>6} {med:9.1f} {quartiles[0]:9.1f} "
              f"{quartiles[2]:9.1f} {min(times):9.1f} {med - base:+9.1f}")

    print("\n=== training loss ===")
    checkpoints = [1, 10, 25, 50, 100, 150, 200, 250, 300]
    header = "".join(f"{c:>10}" for c in checkpoints)
    print(f"{'arm':10}{header}")
    for label, (iters, _) in data.items():
        cells = "".join(
            f"{iters[c]['loss']:10.4f}" if c in iters else f"{'-':>10}"
            for c in checkpoints
        )
        print(f"{label:10}{cells}")

    print("\n=== held-out validation loss ===")
    all_val_steps = sorted({s for _, (_, val) in data.items() for s in val})
    header = "".join(f"{s:>10}" for s in all_val_steps)
    print(f"{'arm':10}{header}")
    for label, (_, val) in data.items():
        cells = "".join(
            f"{val[s]:10.4f}" if s in val else f"{'-':>10}" for s in all_val_steps
        )
        print(f"{label:10}{cells}")

    print("\n=== stability ===")
    print(f"{'arm':10} {'nan':>5} {'skipped':>8} {'max loss jump':>14} {'at iter':>8}")
    for label, (iters, _) in data.items():
        if not iters:
            continue
        ordered = [iters[k]["loss"] for k in sorted(iters)]
        keys = sorted(iters)
        jumps = [(ordered[i] - ordered[i - 1], keys[i]) for i in range(1, len(ordered))]
        worst = max(jumps, default=(0.0, 0))
        last = iters[max(iters)]
        print(f"{label:10} {last['nan']:>5} {last['skipped']:>8} "
              f"{worst[0]:>14.4f} {worst[1]:>8}")


if __name__ == "__main__":
    main()

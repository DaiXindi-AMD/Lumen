#!/usr/bin/env python3
"""Loss regression gate for MXFP4 training runs.

MXFP4's WGrad uses stochastic rounding, so two runs of an identical
configuration do not agree bit for bit and a determinism gate would be red
forever. What they do agree on is where the loss ends up, and that is what this
gates on -- but only after the noise floor has been measured, because the
threshold is otherwise a guess.

Which part of the curve to gate on matters more than it looks. Two identical
200-step runs of the current default differ by 7.0% at iteration 15 and 0.10%
in the mean of the last 50, so a gate on the early curve or on a single final
step reports noise, while the tail mean is stable enough to catch a real
regression.

Usage::

    mxfp4_loss_regression.py calibrate LOG [LOG ...]
    mxfp4_loss_regression.py record -o baseline.json LOG [LOG ...]
    mxfp4_loss_regression.py check -b baseline.json LOG

``calibrate`` reports the run-to-run spread across logs of the same
configuration. ``record`` freezes those runs into a baseline whose threshold is
a multiple of the spread it measured. ``check`` is the gate and exits non-zero
when a run falls outside it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

# Megatron's per-iteration line. The fields are positional in practice but the
# set between them changes between branches, so each is matched on its label.
_ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+)")
_LOSS = re.compile(r"lm loss:\s*([\d.eE+-]+)")
_ELAPSED = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
_SKIPPED = re.compile(r"number of skipped iterations:\s*(\d+)")
_NAN = re.compile(r"number of nan iterations:\s*(\d+)")

# The first iterations include compilation, autotune and allocator warm-up, and
# the loss itself is still chaotic there. Nothing before this is comparable.
DEFAULT_WARMUP = 5
DEFAULT_TAIL = 50

# How many standard deviations of the measured spread a threshold gets.
#
# The multiple applies to the standard deviation rather than the range because
# the range grows with the sample count -- going from two runs to five widened
# it from 0.096% to 0.223% without anything changing -- so a threshold set from
# it moves every time another run is added.
DEFAULT_MARGIN = 4.0


@dataclass
class Run:
    path: str
    iterations: int
    total_iterations: int
    tail_mean: float
    final_loss: float
    median_step_ms: float
    skipped: int
    nan: int

    def summary(self) -> str:
        return (
            f"{Path(self.path).name}: {self.iterations} iters, "
            f"tail mean {self.tail_mean:.6f}, final {self.final_loss:.6f}, "
            f"step {self.median_step_ms:.1f} ms"
        )


def parse_log(path: Path, warmup: int, tail: int) -> Run:
    losses: Dict[int, float] = {}
    steps: Dict[int, float] = {}
    skipped = nan = 0
    total = 0

    with path.open(errors="ignore") as fh:
        for line in fh:
            it = _ITER.search(line)
            loss = _LOSS.search(line)
            if not (it and loss):
                continue
            i = int(it.group(1))
            total = int(it.group(2))
            losses[i] = float(loss.group(1))
            elapsed = _ELAPSED.search(line)
            if elapsed:
                steps[i] = float(elapsed.group(1))
            for pattern, name in ((_SKIPPED, "skipped"), (_NAN, "nan")):
                m = pattern.search(line)
                if m and int(m.group(1)):
                    if name == "skipped":
                        skipped = int(m.group(1))
                    else:
                        nan = int(m.group(1))

    if not losses:
        raise SystemExit(f"{path}: no iteration lines found")

    ordered = sorted(losses)
    tail_iters = ordered[-tail:]
    warm = [i for i in ordered if i > warmup]

    return Run(
        path=str(path),
        iterations=len(ordered),
        total_iterations=total,
        tail_mean=statistics.mean(losses[i] for i in tail_iters),
        final_loss=losses[ordered[-1]],
        median_step_ms=statistics.median(steps[i] for i in warm if i in steps),
        skipped=skipped,
        nan=nan,
    )


def spread(values: List[float]) -> Dict[str, float]:
    """Absolute and relative spread of a set of same-configuration runs."""
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "range_pct": (max(values) - min(values)) / mean * 100 if mean else 0.0,
        "stdev": stdev,
        "stdev_pct": stdev / mean * 100 if mean else 0.0,
        "n": len(values),
    }


def _load_runs(paths: List[str], warmup: int, tail: int) -> List[Run]:
    runs = [parse_log(Path(p), warmup, tail) for p in paths]
    counts = {r.iterations for r in runs}
    if len(counts) > 1:
        raise SystemExit(f"runs disagree on iteration count: {sorted(counts)}")
    return runs


def cmd_calibrate(args) -> int:
    runs = _load_runs(args.logs, args.warmup, args.tail)
    for r in runs:
        print("  " + r.summary())

    loss = spread([r.tail_mean for r in runs])
    step = spread([r.median_step_ms for r in runs])

    print(f"\nnoise floor over {len(runs)} runs "
          f"(tail-{args.tail} mean, warmup {args.warmup}):")
    print(f"  loss : mean {loss['mean']:.6f}  stdev {loss['stdev_pct']:.3f}%  "
          f"range {loss['range_pct']:.3f}%")
    print(f"  step : mean {step['mean']:.1f} ms  stdev {step['stdev_pct']:.3f}%  "
          f"range {step['range']:.1f} ms ({step['range_pct']:.3f}%)")

    if len(runs) < 3:
        print("\n  Two runs give a range, not a distribution. Three or more "
              "before setting a gate.")
    print(f"\nsuggested thresholds at {args.margin} stdev:")
    print(f"  loss tail mean : {loss['stdev_pct'] * args.margin:.3f}%")
    print(f"  median step    : {max(step['stdev_pct'] * args.margin, 1.0):.3f}%")
    return 0


def cmd_record(args) -> int:
    runs = _load_runs(args.logs, args.warmup, args.tail)
    loss = spread([r.tail_mean for r in runs])
    step = spread([r.median_step_ms for r in runs])

    baseline = {
        "runs": [asdict(r) for r in runs],
        "n_runs": len(runs),
        "warmup": args.warmup,
        "tail": args.tail,
        "iterations": runs[0].iterations,
        "loss_tail_mean": loss["mean"],
        "loss_noise_pct": loss["stdev_pct"],
        "loss_range_pct": loss["range_pct"],
        "loss_threshold_pct": max(loss["stdev_pct"] * args.margin, args.floor),
        "step_median_ms": step["mean"],
        "step_noise_pct": step["stdev_pct"],
        "step_range_pct": step["range_pct"],
        "step_threshold_pct": max(step["stdev_pct"] * args.margin, 1.0),
    }
    Path(args.out).write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"wrote {args.out} from {len(runs)} runs")
    print(f"  loss tail mean {baseline['loss_tail_mean']:.6f} "
          f"+- {baseline['loss_threshold_pct']:.3f}% "
          f"(stdev {baseline['loss_noise_pct']:.3f}%)")
    print(f"  median step    {baseline['step_median_ms']:.1f} ms "
          f"+- {baseline['step_threshold_pct']:.3f}% "
          f"(stdev {baseline['step_noise_pct']:.3f}%)")
    return 0


def cmd_check(args) -> int:
    baseline = json.loads(Path(args.baseline).read_text())
    run = parse_log(Path(args.log), baseline["warmup"], baseline["tail"])
    print("  " + run.summary())

    failures = []

    if run.iterations != baseline["iterations"]:
        failures.append(
            f"iteration count {run.iterations} != baseline "
            f"{baseline['iterations']} -- not comparable"
        )
    if run.nan:
        failures.append(f"{run.nan} NaN iterations")
    if run.skipped:
        failures.append(f"{run.skipped} skipped iterations")

    loss_delta = (run.tail_mean - baseline["loss_tail_mean"]) / baseline["loss_tail_mean"] * 100
    step_delta = (run.median_step_ms - baseline["step_median_ms"]) / baseline["step_median_ms"] * 100

    print(f"\n  loss tail mean : {run.tail_mean:.6f} vs {baseline['loss_tail_mean']:.6f} "
          f"({loss_delta:+.3f}%, gate +-{baseline['loss_threshold_pct']:.3f}%, "
          f"stdev {baseline['loss_noise_pct']:.3f}%)")
    print(f"  median step    : {run.median_step_ms:.1f} vs {baseline['step_median_ms']:.1f} ms "
          f"({step_delta:+.3f}%, gate +{baseline['step_threshold_pct']:.3f}%)")

    if abs(loss_delta) > baseline["loss_threshold_pct"]:
        failures.append(
            f"loss tail mean {loss_delta:+.3f}% outside "
            f"+-{baseline['loss_threshold_pct']:.3f}%"
        )
    # Only a slowdown is a regression; a speedup is the point of the work.
    if step_delta > baseline["step_threshold_pct"]:
        failures.append(
            f"median step {step_delta:+.3f}% over "
            f"+{baseline['step_threshold_pct']:.3f}%"
        )

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="iterations to ignore when taking the step median")
    common.add_argument("--tail", type=int, default=DEFAULT_TAIL,
                        help="iterations averaged for the loss metric")
    common.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="standard deviations of measured spread a threshold gets")

    c = sub.add_parser("calibrate", parents=[common],
                       help="report run-to-run spread across identical runs")
    c.add_argument("logs", nargs="+")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("record", parents=[common], help="write a baseline")
    r.add_argument("logs", nargs="+")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--floor", type=float, default=0.2,
                   help="lower bound on the loss threshold, in percent")
    r.set_defaults(func=cmd_record)

    k = sub.add_parser("check", help="gate a run against a baseline")
    k.add_argument("log")
    k.add_argument("-b", "--baseline", required=True)
    k.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

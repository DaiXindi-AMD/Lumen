"""Where a training step is *not* running a compute kernel, and what is there instead.

``summarize_torch_trace.py`` ranks kernels, which answers "what does the GPU spend
its time on" but not "why is the step longer than that". On Qwen3-8B MXFP4 the two
differ by ~360 ms/iter: the compute stream is busy 5623 ms of a 5985 ms step. That
gap is either exposed communication or the host failing to stay ahead of the
device, and the two have completely different fixes, so it has to be attributed
rather than guessed at.

For each idle interval on the busiest kernel stream this reports how long it is,
whether a collective was in flight on the comm stream at the time, and which host
operator was on the CPU. Gaps are bucketed by size because the two causes look
different: launch-bound host code produces thousands of gaps a few microseconds
wide, exposed communication produces a handful of millisecond-scale ones.

Usage:
    python scripts/trace_compute_stream_gaps.py <trace.json[.gz]> [--iters N]
      [--min-gap-us N] [--top N]
"""

import argparse
import gzip
import json
from bisect import bisect_right
from collections import defaultdict


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    return trace.get("traceEvents", trace)


def merge(intervals):
    """Union of possibly-overlapping [start, end) intervals, sorted by start."""
    if not intervals:
        return []
    intervals.sort()
    out = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo > out[-1][1]:
            out.append([lo, hi])
        else:
            out[-1][1] = max(out[-1][1], hi)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--min-gap-us", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    events = load_events(args.trace)

    gpu_by_stream = defaultdict(list)
    cpu_ops = []
    for event in events:
        cat = event.get("cat")
        ts = event.get("ts")
        dur = event.get("dur")
        if ts is None or dur is None:
            continue
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu_by_stream[(event.get("pid"), event.get("tid"))].append(
                (ts, ts + dur, event["name"])
            )
        elif cat in ("cpu_op", "user_annotation"):
            cpu_ops.append((ts, ts + dur, event["name"]))

    if not gpu_by_stream:
        raise SystemExit(f"no GPU kernel events in {args.trace}")

    # Busiest stream by summed kernel time is the compute stream; the next one
    # down on these traces is RCCL's.
    ranked_streams = sorted(
        gpu_by_stream.items(), key=lambda kv: -sum(hi - lo for lo, hi, _ in kv[1])
    )
    compute_key, compute_events = ranked_streams[0]
    comm_intervals = merge(
        [
            (lo, hi)
            for key, evs in ranked_streams[1:]
            for lo, hi, name in evs
            if "nccl" in name.lower() or "rccl" in name.lower()
        ]
    )
    comm_starts = [lo for lo, _ in comm_intervals]

    busy = merge([(lo, hi) for lo, hi, _ in compute_events])
    window_start, window_end = busy[0][0], busy[-1][1]
    span = window_end - window_start
    busy_total = sum(hi - lo for lo, hi in busy)

    def comm_in_flight(lo, hi):
        """Fraction of [lo, hi) that overlaps a collective on the comm stream."""
        idx = bisect_right(comm_starts, hi) - 1
        covered = 0.0
        while idx >= 0 and comm_intervals[idx][1] > lo:
            covered += max(0.0, min(hi, comm_intervals[idx][1]) - max(lo, comm_intervals[idx][0]))
            idx -= 1
        return covered

    # Host operators, longest first, so the innermost enclosing op is not
    # shadowed by the top-level `ProfilerStep` that spans everything.
    cpu_ops.sort(key=lambda op: (op[0], -(op[1] - op[0])))
    cpu_starts = [op[0] for op in cpu_ops]

    def host_op(lo, hi):
        idx = bisect_right(cpu_starts, lo)
        best, best_span = None, float("inf")
        for op_lo, op_hi, name in cpu_ops[max(0, idx - 400): idx]:
            if op_lo <= lo and op_hi >= hi and (op_hi - op_lo) < best_span:
                best, best_span = name, op_hi - op_lo
        return best or "-"

    gaps = []
    for (_, prev_end), (next_start, _) in zip(busy, busy[1:]):
        width = next_start - prev_end
        if width >= args.min_gap_us:
            gaps.append((width, prev_end, next_start))

    idle_total = span - busy_total
    counted = sum(g[0] for g in gaps)

    print(f"trace: {args.trace}")
    print(f"compute stream: pid/tid {compute_key[0]}/{compute_key[1]}   iterations: {args.iters}")
    print(f"window {span / 1000 / args.iters:.1f} ms/iter   "
          f"busy {busy_total / 1000 / args.iters:.1f}   "
          f"idle {idle_total / 1000 / args.iters:.1f} "
          f"({idle_total / span * 100:.1f}%)")

    buckets = [(0, 5), (5, 20), (20, 100), (100, 1000), (1000, float("inf"))]
    all_gaps = [
        (next_start - prev_end, prev_end, next_start)
        for (_, prev_end), (next_start, _) in zip(busy, busy[1:])
    ]
    print("\n=== idle intervals by width ===")
    print(f"{'width':>16} {'count/iter':>11} {'ms/iter':>9} {'share of idle':>14} {'ms/iter w/ comm':>16}")
    for lo_us, hi_us in buckets:
        sel = [g for g in all_gaps if lo_us <= g[0] < hi_us]
        ms = sum(g[0] for g in sel) / 1000 / args.iters
        with_comm = sum(comm_in_flight(g[1], g[2]) for g in sel) / 1000 / args.iters
        label = f"{lo_us:g}-{hi_us:g} us" if hi_us != float("inf") else f">{lo_us:g} us"
        print(f"{label:>16} {len(sel) / args.iters:11.0f} {ms:9.1f} "
              f"{ms / (idle_total / 1000 / args.iters) * 100:13.1f}% {with_comm:16.1f}")

    print(f"\n=== top {args.top} single idle intervals (>= {args.min_gap_us:g} us) ===")
    print(f"{'ms':>8} {'comm%':>6}  host op")
    for width, lo, hi in sorted(gaps, reverse=True)[: args.top]:
        print(f"{width / 1000:8.2f} {comm_in_flight(lo, hi) / width * 100:5.0f}%  {host_op(lo, hi)[:88]}")

    print(f"\ngaps >= {args.min_gap_us:g} us account for "
          f"{counted / 1000 / args.iters:.1f} ms/iter of "
          f"{idle_total / 1000 / args.iters:.1f} ms idle")


if __name__ == "__main__":
    main()

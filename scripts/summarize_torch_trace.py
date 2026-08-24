"""Rank GPU kernels by total time from a Megatron ``--use-pytorch-profiler`` trace.

The MXFP4 training report's kernel table (§5.5) is the basis for most of the
remaining optimization estimates, and at least one of those estimates did not
survive measurement, so the ranking needs to be reproducible rather than quoted.

Usage:
    python scripts/summarize_torch_trace.py <trace.json[.gz]> [--iters N] [--top N]

``--iters`` is how many training iterations the trace covers (Megatron's
``profile_step_end - profile_step_start``); every number is reported per
iteration so it can be compared against a step time directly. Kernel times are
device time and overlap freely with each other, so the per-category totals sum to
more than the step's critical path.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

# First match wins, so the specific patterns precede the generic ones.
CATEGORIES = [
    ("fp4 gemm", r"f4gemm|a4w4|mxfp4.*gemm|gemm.*mxfp4|afp4wfp4"),
    ("other gemm", r"Cijk|gemm|matmul|_mm_|dot_kernel"),
    ("attention", r"fmha|flash|attn|attention"),
    ("quantize / layout", r"quant|hadamard|rht|shuffle|swizzle|fp4|e8m0"),
    ("collectives", r"nccl|rccl|AllReduce|ReduceScatter|AllGather|all_reduce"),
    ("grad accumulate", r"CUDAFunctor_add|add_kernel|add_stride"),
    ("optimizer", r"adam|multi_tensor|foreach|clip_grad|lamb"),
    ("norm", r"rmsnorm|layer_norm|LayerNorm|norm_kernel"),
    ("rope / embedding", r"rope|rotary|embedding|Indexing"),
    ("activation", r"swiglu|silu|gelu|glu_kernel"),
    ("copy / reshape", r"copy|contiguous|CatArrayBatched|transpose|permute|cat_kernel"),
    ("reduce / loss", r"reduce|softmax|cross_entropy|sum_kernel|norm2"),
    ("elementwise other", r"elementwise|Functor|scalar|fill|zero"),
]


def classify(name):
    for label, pattern in CATEGORIES:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "unclassified"


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    return trace.get("traceEvents", trace)


def busy_time(intervals):
    """Wall time a stream is occupied, i.e. the union of its kernel intervals.

    Summed kernel duration overcounts whenever kernels overlap, so it cannot say
    how much of a step a category actually holds. The union can.
    """
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > end:
            total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    total += end - start
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--step-ms", type=float, default=None,
                        help="Wall-clock step time, to report each kernel as a share of it.")
    args = parser.parse_args()

    events = load_events(args.trace)

    by_name = defaultdict(lambda: [0.0, 0])
    by_stream = defaultdict(list)
    all_intervals = []
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        entry = by_name[event["name"]]
        duration = event.get("dur", 0) / 1000.0  # us -> ms
        entry[0] += duration
        entry[1] += 1
        start = event.get("ts", 0) / 1000.0
        by_stream[(event.get("pid"), event.get("tid"))].append((start, start + duration))
        all_intervals.append((start, start + duration))

    if not by_name:
        raise SystemExit(f"no GPU kernel events in {args.trace}")

    total = sum(v[0] for v in by_name.values()) / args.iters

    def share(ms):
        base = args.step_ms if args.step_ms else total
        return f"{ms / base * 100:5.1f}%"

    print(f"trace: {args.trace}")
    print(f"iterations: {args.iters}   distinct kernels: {len(by_name)}")
    print(f"total GPU kernel time: {total:.1f} ms/iter", end="")
    if args.step_ms:
        print(f"   ({share(total)} of a {args.step_ms:.0f} ms step)")
    else:
        print()

    print("\n=== streams (summed kernel time vs wall time the stream is busy) ===")
    print(f"{'stream':>16} {'summed ms/iter':>15} {'busy ms/iter':>13} {'kernels/iter':>13}")
    for (pid, tid), intervals in sorted(
        by_stream.items(), key=lambda kv: -sum(hi - lo for lo, hi in kv[1])
    ):
        summed = sum(hi - lo for lo, hi in intervals) / args.iters
        if summed < 1.0:
            continue
        print(f"{f'{pid}/{tid}':>16} {summed:15.1f} "
              f"{busy_time(intervals) / args.iters:13.1f} {len(intervals) / args.iters:13.0f}")
    print(f"{'all streams':>16} {total:15.1f} "
          f"{busy_time(all_intervals) / args.iters:13.1f} "
          f"{sum(v[1] for v in by_name.values()) / args.iters:13.0f}")

    by_category = defaultdict(lambda: [0.0, 0])
    for name, (ms, calls) in by_name.items():
        entry = by_category[classify(name)]
        entry[0] += ms
        entry[1] += calls

    print("\n=== by category ===")
    print(f"{'category':22} {'ms/iter':>9} {'share':>7} {'calls/iter':>11}")
    for label, (ms, calls) in sorted(by_category.items(), key=lambda kv: -kv[1][0]):
        print(f"{label:22} {ms / args.iters:9.1f} {share(ms / args.iters):>7} "
              f"{calls / args.iters:11.0f}")

    print(f"\n=== top {args.top} kernels ===")
    print(f"{'ms/iter':>9} {'share':>7} {'calls':>8}  {'category':20} name")
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[: args.top]
    for name, (ms, calls) in ranked:
        per_iter = ms / args.iters
        display = name if len(name) <= 78 else name[:75] + "..."
        print(f"{per_iter:9.2f} {share(per_iter):>7} {calls / args.iters:8.0f}  "
              f"{classify(name):20} {display}")


if __name__ == "__main__":
    main()

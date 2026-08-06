#!/usr/bin/env python3
"""Replay a finished Megatron training log into W&B.

For runs that were launched without ``--wandb-project``: Megatron only creates the
wandb writer when that argument is non-empty, so those runs leave their metrics in
the stdout log and nowhere else. This parses the log back out and creates an
equivalent run after the fact.

The resulting run carries ``backfilled_from_log`` in its config so it is
distinguishable from a live one — the wall-clock timestamps are the replay's, not
the training run's.

    python wandb_backfill_megatron_log.py results/lumen_qwen3_8b_c4_mxfp4.log \
        --project qwen3-8b-mxfp4 --entity daixindi-amd --name megatron-mxfp4-8b-c4-200
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

# " [ts] iteration    7/  200 | consumed samples: 224 | elapsed time per iteration
#   (ms): 2505.0 | mem usages: 0.6170 | learning rate: 1.0E-05 | ... | lm loss:
#   7.2E+00 | loss scale: 1.0 | grad norm: 1.234 | number of skipped iterations: 0 |
#   number of nan iterations: 0 |"
ITER_RE = re.compile(
    r"iteration +(?P<iteration>\d+)/ *(?P<total>\d+) *\|"
    r".*?consumed samples: *(?P<samples>\d+)"
    r".*?elapsed time per iteration \(ms\): *(?P<step_ms>[\d.]+)"
)
FIELD_RES = {
    "mem_usage_fraction": re.compile(r"mem usages?: *([\d.]+)"),
    "learning_rate": re.compile(r"learning rate: *([\d.eE+-]+)"),
    "lm_loss": re.compile(r"lm loss: *([\d.eE+-]+)"),
    "grad_norm": re.compile(r"grad norm: *([\d.]+)"),
    "skipped_iterations": re.compile(r"number of skipped iterations: *(\d+)"),
    "nan_iterations": re.compile(r"number of nan iterations: *(\d+)"),
}
# Bare "at iteration N |" is the periodic eval; the "on validation set" / "on test
# set" variants are the extra end-of-training passes. Keep them apart: they draw
# different batches and land on different values.
VAL_RE = re.compile(
    r"validation loss at iteration (?P<iteration>\d+)(?P<split> on \w+ set)? \|"
    r" lm loss value: (?P<loss>[\d.eE+-]+)"
)
# Megatron's argparse dump: "  seq_length ...... 8192"
ARG_RE = re.compile(r"^ {2}(\w+) \.+ (.*)$")

CONFIG_KEYS = (
    "num_layers hidden_size ffn_hidden_size num_attention_heads num_query_groups "
    "seq_length micro_batch_size global_batch_size train_iters lr min_lr "
    "lr_decay_style lr_warmup_iters weight_decay clip_grad seed world_size "
    "tensor_model_parallel_size pipeline_model_parallel_size context_parallel_size "
    "use_distributed_optimizer linear_fp8 linear_fp8_format linear_fp8_scaling "
    "linear_fp8_block_size first_last_layers_bf16 num_layers_at_end_in_bf16 "
    "lumen_attn_backend transformer_impl train_data_path"
).split()


def parse(path: Path) -> dict:
    iters: list[dict] = []
    vals: list[tuple[int, str, float]] = []
    config: dict[str, str] = {}
    quantized_layers = None

    for line in path.read_text(errors="ignore").splitlines():
        m = ITER_RE.search(line)
        if m:
            row = {
                "iteration": int(m.group("iteration")),
                "consumed_samples": int(m.group("samples")),
                "step_time_ms": float(m.group("step_ms")),
            }
            for name, regex in FIELD_RES.items():
                got = regex.search(line)
                if got:
                    row[name] = float(got.group(1))
            iters.append(row)
            continue

        m = VAL_RE.search(line)
        if m:
            split = (m.group("split") or "periodic").strip()
            split = split.removeprefix("on ").removesuffix(" set")
            vals.append((int(m.group("iteration")), split, float(m.group("loss"))))
            continue

        m = ARG_RE.match(line)
        if m and m.group(1) in CONFIG_KEYS:
            config[m.group(1)] = m.group(2).strip()
            continue

        if "Quantization enabled on" in line:
            got = re.search(r"on (\d+) nn.Linear layers", line)
            skipped = re.search(r"bf16_layers_skipped=(\d+)", line)
            if got:
                quantized_layers = int(got.group(1))
            if skipped:
                config["bf16_layers_skipped"] = skipped.group(1)

    if not iters:
        raise SystemExit(f"{path}: no iteration lines found")
    if quantized_layers is not None:
        config["quantized_linear_layers"] = str(quantized_layers)
    return {"iters": iters, "vals": vals, "config": config}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--project", required=True)
    p.add_argument("--entity", default=None)
    p.add_argument("--name", default=None, help="Defaults to the log's stem.")
    p.add_argument("--mode", default="online", choices=["online", "offline", "dryrun"])
    p.add_argument("--dry-run", action="store_true", help="Parse and print, no upload.")
    p.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Log of the run to diff against. Adds per-iteration delta series. Only "
        "meaningful when both runs used the same seed and data order, which makes "
        "the two losses at a given iteration a paired comparison on identical batches.",
    )
    p.add_argument(
        "--delta-only",
        action="store_true",
        help="Log only the delta/* series, not this run's own train/* curves. Use to "
        "publish the comparison as its own run without duplicating curves that are "
        "already in the two runs being compared.",
    )
    args = p.parse_args()
    if args.delta_only and not args.baseline_log:
        p.error("--delta-only requires --baseline-log")

    parsed = parse(args.log)
    iters, vals, config = parsed["iters"], parsed["vals"], parsed["config"]
    seq_len = int(config.get("seq_length", 0))

    base_loss: dict[int, float] = {}
    base_step_ms: dict[int, float] = {}
    if args.baseline_log:
        base = parse(args.baseline_log)
        base_loss = {r["iteration"]: r["lm_loss"] for r in base["iters"] if "lm_loss" in r}
        base_step_ms = {r["iteration"]: r["step_time_ms"] for r in base["iters"]}
        for seed_key in ("seed", "global_batch_size", "seq_length", "train_data_path"):
            mine, theirs = config.get(seed_key), base["config"].get(seed_key)
            if mine != theirs:
                print(
                    f"  WARNING: {seed_key} differs ({mine!r} vs {theirs!r}) — the "
                    f"delta series is not a paired comparison"
                )
        config["delta_baseline_log"] = str(args.baseline_log.resolve())

    steady = [r["step_time_ms"] for r in iters[10:]] or [r["step_time_ms"] for r in iters]
    summary = {
        "step_time_ms_median_steady": statistics.median(steady),
        "step_time_ms_mean_steady": statistics.fmean(steady),
        "step_time_ms_min": min(r["step_time_ms"] for r in iters),
        "iterations_logged": len(iters),
        "final_lm_loss": iters[-1].get("lm_loss"),
    }
    if seq_len:
        summary["tokens_per_second_median"] = (
            iters[-1]["consumed_samples"] / max(iters[-1]["iteration"], 1) * seq_len
        ) / (summary["step_time_ms_median_steady"] / 1000.0)
    # val/* and train/* land in the summary on their own as the last logged value,
    # so only derived quantities need adding here.
    if base_loss:
        tail = [
            iters[i]["lm_loss"] - base_loss[iters[i]["iteration"]]
            for i in range(len(iters) // 2, len(iters))
            if iters[i]["iteration"] in base_loss and "lm_loss" in iters[i]
        ]
        if tail:
            summary["lm_loss_delta_second_half_mean"] = statistics.fmean(tail)
        if base_step_ms:
            summary["speedup_vs_baseline_median"] = statistics.median(
                base_step_ms[r["iteration"]] / r["step_time_ms"]
                for r in iters[10:]
                if r["iteration"] in base_step_ms
            )

    print(f"{args.log}: {len(iters)} iterations, {len(vals)} eval lines")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if args.dry_run:
        return

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name or args.log.stem,
        mode=args.mode,
        config={
            **config,
            "backend": "megatron",
            "backfilled_from_log": str(args.log.resolve()),
        },
    )
    # Eval lines are folded into the iteration they belong to rather than logged
    # after the loop: wandb ignores any log at a step below the highest one seen,
    # so a trailing val pass would silently drop every point.
    val_at = {}
    if not args.delta_only:
        for iteration, split, loss in vals:
            val_at.setdefault(iteration, {})[f"val/lm_loss_{split}"] = loss

    # step_time_ms is per-iteration wall time as Megatron reports it, so it already
    # excludes nothing — iteration 1 carries the process warmup.
    for row in iters:
        it = row["iteration"]
        metrics = {}
        if not args.delta_only:
            metrics = {f"train/{k}": v for k, v in row.items() if k != "iteration"}
            if seq_len:
                metrics["train/tokens"] = row["consumed_samples"] * seq_len
        if it in base_loss and "lm_loss" in row:
            metrics["delta/lm_loss_vs_baseline"] = row["lm_loss"] - base_loss[it]
        if it in base_step_ms:
            metrics["delta/speedup_vs_baseline"] = base_step_ms[it] / row["step_time_ms"]
        metrics.update(val_at.pop(it, {}))
        if metrics:
            wandb.log(metrics, step=it)
    # Eval passes that carry no matching iteration line (the end-of-training
    # validation and test passes) still have to go somewhere.
    for iteration in sorted(val_at):
        wandb.log(val_at[iteration], step=max(iteration, iters[-1]["iteration"]))
    run.summary.update(summary)
    run.finish()
    print(f"uploaded as {run.url}")


if __name__ == "__main__":
    main()

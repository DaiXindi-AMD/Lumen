#!/usr/bin/env python3
"""Capture activations with and without Hadamard transform in Qwen3-8B MXFP4.

Hooks into the MXFP4-patched nn.Linear layers to capture the BF16 activation
before quantization, then computes:
  1. Per-row inf-norm / Frobenius norm / max absolute value — raw vs. Hadamard
  2. Side-by-side heatmaps of |activation| (PNG via matplotlib + HTML fallback)
  3. Per-row inf-norm distribution histogram

Usage (single-GPU, a few forward steps):
    python hadamard_outlier_analysis.py \
        --model-name-or-path Qwen/Qwen3-8B \
        --train-data-path /data/alpaca_data_cleaned.json \
        --steps 2 \
        --layers-to-capture "model.layers.0.mlp.gate_proj,model.layers.0.mlp.up_proj" \
        --output-dir ./hadamard_analysis

Requires: a CUDA GPU with Lumen + AITER installed.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lumen.ops.quantize.ops import hadamard_transform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# -- Hadamard parameters matching the MXFP4 training path --
_RHT_G = 16


def _get_sign(device):
    return torch.ones(_RHT_G, device=device, dtype=torch.float32)


# ---- Capture storage ----
_captures: dict[str, list[dict]] = {}


def _make_hook(layer_name):
    """Return a forward pre-hook that records the BF16 input before quant."""

    def hook(module, args, kwargs=None):
        x = args[0] if isinstance(args, tuple) else args
        if not isinstance(x, torch.Tensor):
            return
        x_2d = x.detach().reshape(-1, x.shape[-1]).float()
        sign = _get_sign(x.device)
        x_had = hadamard_transform(x_2d, sign, g=_RHT_G)

        record = {
            "raw": x_2d.cpu(),
            "hadamard": x_had.cpu(),
        }
        _captures.setdefault(layer_name, []).append(record)

    return hook


# ---- Norm computation ----

def compute_norms(t: torch.Tensor) -> dict:
    """Compute summary statistics for a 2D (M, K) tensor."""
    abs_t = t.abs()
    return {
        "max_abs": abs_t.max().item(),
        "mean_abs": abs_t.mean().item(),
        "frobenius": t.norm(p="fro").item(),
        "inf_norm_per_row_mean": abs_t.max(dim=-1).values.mean().item(),
        "inf_norm_per_row_max": abs_t.max(dim=-1).values.max().item(),
        "kurtosis_per_row_mean": _kurtosis_mean(t),
    }


def _kurtosis_mean(t):
    """Mean excess kurtosis across rows — measures outlier-heaviness."""
    mu = t.mean(dim=-1, keepdim=True)
    var = t.var(dim=-1, keepdim=True).clamp(min=1e-12)
    k4 = ((t - mu) ** 4).mean(dim=-1) / (var.squeeze(-1) ** 2) - 3.0
    return k4.mean().item()


# ---- Matplotlib heatmap + histogram ----

def generate_heatmap_png(
    raw: torch.Tensor,
    had: torch.Tensor,
    layer_name: str,
    out_dir: Path,
    max_rows: int = 64,
    max_cols: int = 128,
):
    """Side-by-side |activation| heatmaps saved as PNG."""
    if not HAS_MPL:
        return
    raw_s = raw[:max_rows, :max_cols].abs().numpy()
    had_s = had[:max_rows, :max_cols].abs().numpy()
    vmax = max(raw_s.max(), had_s.max(), 1e-12)
    norm = Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im0 = axes[0].imshow(raw_s, aspect="auto", cmap="inferno", norm=norm)
    axes[0].set_title("Raw activation |x|", fontsize=13)
    axes[0].set_xlabel("Hidden dim")
    axes[0].set_ylabel("Token (row)")

    im1 = axes[1].imshow(had_s, aspect="auto", cmap="inferno", norm=norm)
    axes[1].set_title("After Hadamard |Hx|", fontsize=13)
    axes[1].set_xlabel("Hidden dim")

    fig.suptitle(f"{layer_name}  [{raw_s.shape[0]}×{raw_s.shape[1]}]", fontsize=14, y=1.02)
    fig.tight_layout()

    safe = layer_name.replace(".", "_")
    path = out_dir / f"heatmap_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Heatmap PNG saved: %s", path)


def generate_inf_norm_histogram(
    raw: torch.Tensor,
    had: torch.Tensor,
    layer_name: str,
    out_dir: Path,
):
    """Per-row inf-norm distribution histogram comparing raw vs. Hadamard."""
    if not HAS_MPL:
        return
    raw_inf = raw.abs().max(dim=-1).values.numpy()
    had_inf = had.abs().max(dim=-1).values.numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(raw_inf, bins=50, alpha=0.6, label="Raw", color="#e63946", edgecolor="white", linewidth=0.5)
    ax.hist(had_inf, bins=50, alpha=0.6, label="After Hadamard", color="#457b9d", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Per-row inf-norm (max |x|)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Inf-norm distribution — {layer_name}", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()

    safe = layer_name.replace(".", "_")
    path = out_dir / f"inf_norm_hist_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Inf-norm histogram saved: %s", path)


# ---- Heatmap (pure HTML fallback, no matplotlib dependency) ----

def generate_heatmap_html(
    raw: torch.Tensor,
    had: torch.Tensor,
    layer_name: str,
    max_rows: int = 64,
    max_cols: int = 128,
) -> str:
    """Generate a side-by-side heatmap as a self-contained HTML page."""
    raw_slice = raw[:max_rows, :max_cols].abs()
    had_slice = had[:max_rows, :max_cols].abs()
    global_max = max(raw_slice.max().item(), had_slice.max().item(), 1e-12)

    def _cell_color(v, vmax):
        frac = min(v / vmax, 1.0)
        r = int(255 * frac)
        b = int(255 * (1 - frac))
        return f"rgb({r},0,{b})"

    def _table(data, vmax, title):
        rows_html = []
        R, C = data.shape
        for r in range(R):
            cells = []
            for c in range(C):
                v = data[r, c].item()
                color = _cell_color(v, vmax)
                cells.append(
                    f'<td style="background:{color};width:4px;height:4px;padding:0;border:none;" '
                    f'title="[{r},{c}] {v:.4f}"></td>'
                )
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        return (
            f"<div style='display:inline-block;margin:10px;'>"
            f"<h3>{title}</h3>"
            f"<table style='border-collapse:collapse;'>"
            + "".join(rows_html)
            + "</table></div>"
        )

    raw_tbl = _table(raw_slice, global_max, f"Raw activation (|x|)")
    had_tbl = _table(had_slice, global_max, f"After Hadamard (|Hx|)")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Hadamard Outlier Analysis — {layer_name}</title>
<style>body{{font-family:sans-serif;background:#1a1a2e;color:#eee;}}
h2{{text-align:center;}} .legend{{margin:20px auto;width:300px;height:20px;
background:linear-gradient(to right, rgb(0,0,255), rgb(255,0,0));border-radius:4px;}}
.legend-labels{{display:flex;justify-content:space-between;width:300px;margin:0 auto;font-size:12px;}}
</style></head><body>
<h2>{layer_name} — [{raw_slice.shape[0]}×{raw_slice.shape[1]}] slice</h2>
<div class="legend"></div>
<div class="legend-labels"><span>0</span><span>max |x| = {global_max:.4f}</span></div>
<div style="display:flex;justify-content:center;flex-wrap:wrap;">
{raw_tbl}
{had_tbl}
</div>
</body></html>"""


# ---- Main ----

def parse_args():
    p = argparse.ArgumentParser(description="Hadamard outlier analysis for Qwen3-8B MXFP4")
    p.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
    p.add_argument("--train-data-path", required=True, help="Alpaca-style JSON or pretrain text")
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=2, help="Forward steps to capture")
    p.add_argument("--layers-to-capture", type=str, default=None,
                   help="Comma-separated module names (default: first decoder layer's MLP)")
    p.add_argument("--heatmap-rows", type=int, default=64)
    p.add_argument("--heatmap-cols", type=int, default=128)
    p.add_argument("--output-dir", type=str, default="./hadamard_analysis")
    p.add_argument("--mode", default="mxfp4", choices=["mxfp4"])
    p.add_argument("--init-from-scratch", action="store_true",
                   help="Initialize model from config (no pretrained weights)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model %s ...", args.model_name_or_path)
    if args.init_from_scratch:
        model_cfg = AutoConfig.from_pretrained(args.model_name_or_path)
        model_cfg.torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_config(
            model_cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model = model.to(device)

    # ---- Apply Lumen MXFP4 patching ----
    from lumen.config import LumenConfig
    from argparse import Namespace

    cfg = LumenConfig.from_args(Namespace(
        linear_fp8=False, linear_fp4=True,
        linear_fp8_format="fp8_e4m3", linear_fp8_scaling="delayed",
        linear_fp8_block_size=128, linear_fp8_amax_algo="max", linear_fp8_amax_history=16,
        linear_fp8_reduce_amax=False, linear_fp8_activation=True, linear_fp8_wgrad=True,
        linear_fp8_cache_frozen_weight=False, linear_fp8_bpreshuffle=False,
        grad_quant_type=None,
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=5,
        num_layers_at_end_in_bf16=0,
        num_layers=model.config.num_hidden_layers,
        lumen_norm=False,
        hf_attn_patch=False,
        lora_rank=0, lora_alpha=16, lora_dropout=0.0,
    ))
    _manager, model = cfg.enable(model)
    log.info("Lumen MXFP4 patching applied")

    # ---- Determine which layers to capture ----
    if args.layers_to_capture:
        target_names = [n.strip() for n in args.layers_to_capture.split(",")]
    else:
        target_names = [
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.self_attn.q_proj",
        ]
    log.info("Capturing layers: %s", target_names)

    # ---- Register hooks ----
    hooks = []
    for name, module in model.named_modules():
        if name in target_names:
            h = module.register_forward_pre_hook(_make_hook(name))
            hooks.append(h)
            log.info("  Hook attached: %s", name)

    if not hooks:
        available = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        log.error("No hooks attached! Available nn.Linear layers (first 20):")
        for n in available[:20]:
            log.error("  %s", n)
        return

    # ---- Prepare data ----
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.train_data_path.endswith(".json") or args.train_data_path.endswith(".jsonl"):
        with open(args.train_data_path) as f:
            if args.train_data_path.endswith(".jsonl"):
                data = [json.loads(line) for _, line in zip(range(args.steps * args.micro_batch_size + 1), f)]
            else:
                data = json.load(f)
        texts = []
        for item in data[:args.steps * args.micro_batch_size]:
            if "text" in item:
                texts.append(item["text"])
            else:
                parts = []
                if item.get("instruction"):
                    parts.append(item["instruction"])
                if item.get("input"):
                    parts.append(item["input"])
                if item.get("output"):
                    parts.append(item["output"])
                texts.append(" ".join(parts))
    else:
        with open(args.train_data_path) as f:
            texts = [f.read(args.seq_length * 10)]

    # ---- Forward passes ----
    model.eval()
    with torch.no_grad():
        for step in range(args.steps):
            idx = step % len(texts)
            enc = tok(
                texts[idx], return_tensors="pt",
                max_length=args.seq_length, truncation=True, padding="max_length",
            )
            input_ids = enc["input_ids"].to(device)
            _ = model(input_ids=input_ids)
            log.info("Step %d/%d done", step + 1, args.steps)

    # ---- Remove hooks ----
    for h in hooks:
        h.remove()

    # ---- Compute and report ----
    results = {}
    for layer_name, records in _captures.items():
        log.info("=" * 60)
        log.info("Layer: %s  (%d captures)", layer_name, len(records))
        log.info("=" * 60)

        raw = records[0]["raw"]
        had = records[0]["hadamard"]

        raw_norms = compute_norms(raw)
        had_norms = compute_norms(had)

        log.info("  Shape: %s", list(raw.shape))
        log.info("  ---- Raw (no Hadamard) ----")
        for k, v in raw_norms.items():
            log.info("    %-25s %12.6f", k, v)
        log.info("  ---- After Hadamard (g=%d) ----", _RHT_G)
        for k, v in had_norms.items():
            log.info("    %-25s %12.6f", k, v)
        log.info("  ---- Reduction ratios (raw / hadamard) ----")
        for k in raw_norms:
            r = raw_norms[k]
            h = had_norms[k]
            ratio = r / h if h > 1e-12 else float("inf")
            log.info("    %-25s %12.4fx", k, ratio)

        results[layer_name] = {
            "shape": list(raw.shape),
            "raw": raw_norms,
            "hadamard": had_norms,
        }

        # Generate matplotlib heatmap + histogram (when available)
        generate_heatmap_png(
            raw, had, layer_name, out_dir,
            max_rows=args.heatmap_rows, max_cols=args.heatmap_cols,
        )
        generate_inf_norm_histogram(raw, had, layer_name, out_dir)

        # Generate HTML heatmap (always available, no dependencies)
        html = generate_heatmap_html(
            raw, had, layer_name,
            max_rows=args.heatmap_rows, max_cols=args.heatmap_cols,
        )
        safe_name = layer_name.replace(".", "_")
        html_path = out_dir / f"heatmap_{safe_name}.html"
        html_path.write_text(html, encoding="utf-8")
        log.info("  HTML heatmap saved: %s", html_path)

    # Save JSON summary
    json_path = out_dir / "norm_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Summary saved: %s", json_path)


if __name__ == "__main__":
    main()

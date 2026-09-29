#!/usr/bin/env python3
"""Compare WGrad quality with vs. without Hadamard in the MXFP4 path.

For each target linear layer, captures the BF16 activation (X) and gradient (dY),
then computes:
  1. WGrad_ref   = dY^T @ X^T                           (BF16 reference)
  2. WGrad_had   = mxfp4_had(dY^T) @ mxfp4_had(X^T)^T  (Hadamard + MXFP4)
  3. WGrad_noHad = mxfp4(dY^T)     @ mxfp4(X^T)^T       (plain MXFP4, no rotation)

Outputs: norms, error vs reference, heatmaps, histograms.
"""

import argparse, json, logging, os, sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, SymLogNorm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lumen.ops.quantize.ops import (
    hadamard_transform,
    hadamard_quant_mxfp4,
    convert_to_mxfp4,
    convert_from_mxfp4,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_RHT_G = 16

def _get_sign(device):
    return torch.ones(_RHT_G, device=device, dtype=torch.float32)


# ---- Capture storage ----
_fwd_activations: dict[str, torch.Tensor] = {}
_bwd_gradients: dict[str, torch.Tensor] = {}


_fwd_weights: dict[str, torch.Tensor] = {}


def _fwd_hook(name):
    def hook(module, args, output):
        x = args[0] if isinstance(args, tuple) else args
        if isinstance(x, torch.Tensor):
            _fwd_activations[name] = x.detach().clone()
        w = getattr(module, "weight", None)
        if w is not None:
            _fwd_weights[name] = w.detach().clone()
    return hook


def _bwd_hook(name):
    def hook(module, grad_input, grad_output):
        dy = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if isinstance(dy, torch.Tensor):
            _bwd_gradients[name] = dy.detach().clone()
    return hook


# ---- MXFP4 WGrad simulation ----

def wgrad_bf16(dy_2d, x_2d):
    """Reference: dW = dY^T @ X  (weight is TN layout: [N, K])"""
    return (dy_2d.float().t() @ x_2d.float()).bfloat16()


def wgrad_mxfp4_with_hadamard(dy_2d, x_2d):
    """WGrad via Hadamard + MXFP4: rotate both operands, quantize, dequant, matmul."""
    sign = _get_sign(dy_2d.device)
    block = 32

    # dY^T: Hadamard-rotate then quantize to MXFP4
    dy_t = dy_2d.t().contiguous()
    dy_t_fp4, dy_t_scale = hadamard_quant_mxfp4(dy_t, sign, block_size=block, g=_RHT_G, use_sr=False)
    dy_t_deq = convert_from_mxfp4(dy_t_fp4, dy_t_scale, output_dtype=torch.bfloat16, block_size=block)

    # X^T: Hadamard-rotate then quantize to MXFP4
    x_t = x_2d.t().contiguous()
    x_t_fp4, x_t_scale = hadamard_quant_mxfp4(x_t, sign, block_size=block, g=_RHT_G, use_sr=False)
    x_t_deq = convert_from_mxfp4(x_t_fp4, x_t_scale, output_dtype=torch.bfloat16, block_size=block)

    return (dy_t_deq.float() @ x_t_deq.float().t()).bfloat16()


def wgrad_mxfp4_no_hadamard(dy_2d, x_2d):
    """WGrad via plain MXFP4 (no Hadamard rotation)."""
    block = 32

    # dY^T: plain MXFP4
    dy_t = dy_2d.t().contiguous()
    dy_t_fp4, dy_t_scale = convert_to_mxfp4(dy_t, block_size=block, axis=-1, use_sr=False)
    dy_t_deq = convert_from_mxfp4(dy_t_fp4, dy_t_scale, output_dtype=torch.bfloat16, block_size=block)

    # X^T: plain MXFP4
    x_t = x_2d.t().contiguous()
    x_t_fp4, x_t_scale = convert_to_mxfp4(x_t, block_size=block, axis=-1, use_sr=False)
    x_t_deq = convert_from_mxfp4(x_t_fp4, x_t_scale, output_dtype=torch.bfloat16, block_size=block)

    return (dy_t_deq.float() @ x_t_deq.float().t()).bfloat16()


# ---- MXFP4 DGrad simulation ----

def dgrad_bf16(dy_2d, w):
    """Reference: dX = dY @ W  (w is [N, K], Y = X @ W^T, so dX = dY @ W)"""
    return (dy_2d.float() @ w.float()).bfloat16()


def dgrad_mxfp4_with_hadamard(dy_2d, w):
    """DGrad via Hadamard + MXFP4: rotate both operands.

    GEMM is TN: A @ B^T. With A = H(dY) shape (M, N) and B = H(W) shape (K, N):
    H(dY) @ H(W)^T = dY @ H^T @ H @ W^T = dY @ W^T  -- but we want dY @ W.
    So use B = W^T, and GEMM(H(dY), H(W^T)) = H(dY) @ H(W^T)^T = dY @ W.
    """
    sign = _get_sign(dy_2d.device)
    block = 32
    wt = w.t().contiguous()  # W^T: (K, N) — same dim as dY along N

    dy_fp4, dy_s = hadamard_quant_mxfp4(dy_2d, sign, block_size=block, g=_RHT_G, use_sr=False)
    dy_deq = convert_from_mxfp4(dy_fp4, dy_s, output_dtype=torch.float32, block_size=block)

    wt_fp4, wt_s = hadamard_quant_mxfp4(wt, sign, block_size=block, g=_RHT_G, use_sr=False)
    wt_deq = convert_from_mxfp4(wt_fp4, wt_s, output_dtype=torch.float32, block_size=block)

    # TN GEMM: A @ B^T = H(dY) @ H(W^T)^T = dY @ W
    return (dy_deq @ wt_deq.t()).bfloat16()


def dgrad_mxfp4_no_hadamard(dy_2d, w):
    """DGrad via plain MXFP4 (no Hadamard rotation)."""
    block = 32
    wt = w.t().contiguous()  # (K, N)

    dy_fp4, dy_s = convert_to_mxfp4(dy_2d, block_size=block, axis=-1, use_sr=False)
    dy_deq = convert_from_mxfp4(dy_fp4, dy_s, output_dtype=torch.float32, block_size=block)

    wt_fp4, wt_s = convert_to_mxfp4(wt, block_size=block, axis=-1, use_sr=False)
    wt_deq = convert_from_mxfp4(wt_fp4, wt_s, output_dtype=torch.float32, block_size=block)

    return (dy_deq @ wt_deq.t()).bfloat16()


# ---- Also show the quantized operand distributions ----

def operand_stats(dy_2d, x_2d):
    """Compare the quantized operands with/without Hadamard."""
    sign = _get_sign(dy_2d.device)
    block = 32
    dy_t = dy_2d.t().contiguous()
    x_t = x_2d.t().contiguous()

    # With Hadamard
    dy_had_fp4, dy_had_s = hadamard_quant_mxfp4(dy_t, sign, block_size=block, g=_RHT_G, use_sr=False)
    dy_had_deq = convert_from_mxfp4(dy_had_fp4, dy_had_s, output_dtype=torch.float32, block_size=block)
    x_had_fp4, x_had_s = hadamard_quant_mxfp4(x_t, sign, block_size=block, g=_RHT_G, use_sr=False)
    x_had_deq = convert_from_mxfp4(x_had_fp4, x_had_s, output_dtype=torch.float32, block_size=block)

    # Without Hadamard
    dy_plain_fp4, dy_plain_s = convert_to_mxfp4(dy_t, block_size=block, axis=-1, use_sr=False)
    dy_plain_deq = convert_from_mxfp4(dy_plain_fp4, dy_plain_s, output_dtype=torch.float32, block_size=block)
    x_plain_fp4, x_plain_s = convert_to_mxfp4(x_t, block_size=block, axis=-1, use_sr=False)
    x_plain_deq = convert_from_mxfp4(x_plain_fp4, x_plain_s, output_dtype=torch.float32, block_size=block)

    # Reference
    dy_t_ref = dy_t.float()
    x_t_ref = x_t.float()

    return {
        "dY_T": {
            "ref": dy_t_ref.cpu(),
            "had": dy_had_deq.cpu(),
            "plain": dy_plain_deq.cpu(),
        },
        "X_T": {
            "ref": x_t_ref.cpu(),
            "had": x_had_deq.cpu(),
            "plain": x_plain_deq.cpu(),
        },
    }


# ---- Metrics ----

def compute_snr(ref, approx):
    """Signal-to-noise ratio in dB."""
    ref_f = ref.float()
    approx_f = approx.float()
    noise = ref_f - approx_f
    signal_power = (ref_f ** 2).sum()
    noise_power = (noise ** 2).sum()
    if noise_power < 1e-30:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def compute_metrics(ref, tensor, label):
    t = tensor.float()
    r = ref.float()
    err = (r - t)
    return {
        "label": label,
        "max_abs": t.abs().max().item(),
        "mean_abs": t.abs().mean().item(),
        "frobenius": t.norm(p="fro").item(),
        "inf_norm_per_row_max": t.abs().max(dim=-1).values.max().item(),
        "snr_vs_ref_dB": compute_snr(r, t),
        "max_abs_error": err.abs().max().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "cosine_sim": nn.functional.cosine_similarity(
            r.reshape(1, -1), t.reshape(1, -1)
        ).item(),
    }


# ---- Visualization ----

def plot_grad_comparison(ref, had, noHad, layer_name, out_dir, max_r=128, max_c=128, kind="WGrad"):
    if not HAS_MPL:
        return
    ref_s = ref[:max_r, :max_c].float().numpy()
    had_s = had[:max_r, :max_c].float().numpy()
    noh_s = noHad[:max_r, :max_c].float().numpy()
    vmax = max(abs(ref_s).max(), abs(had_s).max(), abs(noh_s).max(), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    norm = SymLogNorm(linthresh=vmax * 0.01, vmin=-vmax, vmax=vmax)

    for ax, data, title in [
        (axes[0], ref_s, f"{kind} BF16 ref"),
        (axes[1], had_s, f"{kind} MXFP4 + Hadamard"),
        (axes[2], noh_s, f"{kind} MXFP4 (no Hadamard)"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("K (input dim)")
        ax.set_ylabel("N (output dim)")

    fig.suptitle(f"{kind} — {layer_name}  [{ref_s.shape[0]}×{ref_s.shape[1]}]", fontsize=13, y=1.02)
    fig.tight_layout()

    safe = layer_name.replace(".", "_")
    prefix = kind.lower()
    path = out_dir / f"{prefix}_heatmap_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  %s heatmap saved: %s", kind, path)


def plot_error_heatmap(ref, had, noHad, layer_name, out_dir, max_r=128, max_c=128, kind="WGrad"):
    if not HAS_MPL:
        return
    err_had = (ref[:max_r, :max_c].float() - had[:max_r, :max_c].float()).numpy()
    err_noh = (ref[:max_r, :max_c].float() - noHad[:max_r, :max_c].float()).numpy()
    vmax = max(abs(err_had).max(), abs(err_noh).max(), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    norm = SymLogNorm(linthresh=vmax * 0.01, vmin=-vmax, vmax=vmax)

    im0 = axes[0].imshow(err_had, aspect="auto", cmap="RdBu_r", norm=norm)
    axes[0].set_title("Error: BF16 − (Hadamard+MXFP4)", fontsize=12)
    im1 = axes[1].imshow(err_noh, aspect="auto", cmap="RdBu_r", norm=norm)
    axes[1].set_title("Error: BF16 − (plain MXFP4)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("K"); ax.set_ylabel("N")
    fig.suptitle(f"{kind} error vs BF16 — {layer_name}", fontsize=13, y=1.02)
    fig.tight_layout()

    safe = layer_name.replace(".", "_")
    prefix = kind.lower()
    path = out_dir / f"{prefix}_error_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  %s error heatmap saved: %s", kind, path)


def plot_element_hist(ref, had, noHad, layer_name, out_dir, kind="WGrad"):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    bins = 100
    r_np = ref.float().reshape(-1).numpy()
    h_np = had.float().reshape(-1).numpy()
    n_np = noHad.float().reshape(-1).numpy()
    lo = min(r_np.min(), h_np.min(), n_np.min())
    hi = max(r_np.max(), h_np.max(), n_np.max())
    edges = torch.linspace(lo, hi, bins + 1).numpy()
    ax.hist(r_np, bins=edges, alpha=0.5, label="BF16 ref", color="#2a9d8f", edgecolor="white", linewidth=0.3)
    ax.hist(h_np, bins=edges, alpha=0.5, label="Hadamard+MXFP4", color="#457b9d", edgecolor="white", linewidth=0.3)
    ax.hist(n_np, bins=edges, alpha=0.5, label="plain MXFP4", color="#e63946", edgecolor="white", linewidth=0.3)
    ax.set_xlabel(f"{kind} element value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{kind} element distribution", fontsize=12)
    ax.legend(fontsize=10)

    ax = axes[1]
    rmse_had = (ref.float() - had.float()).pow(2).mean(dim=-1).sqrt().numpy()
    rmse_noh = (ref.float() - noHad.float()).pow(2).mean(dim=-1).sqrt().numpy()
    ax.hist(rmse_had, bins=50, alpha=0.6, label="Hadamard+MXFP4", color="#457b9d", edgecolor="white", linewidth=0.3)
    ax.hist(rmse_noh, bins=50, alpha=0.6, label="plain MXFP4", color="#e63946", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Per-row RMSE vs BF16 ref", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{kind} per-row error distribution", fontsize=12)
    ax.legend(fontsize=10)

    fig.suptitle(f"{kind} — {layer_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    safe = layer_name.replace(".", "_")
    prefix = kind.lower()
    path = out_dir / f"{prefix}_hist_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  %s histogram saved: %s", kind, path)


# ---- Main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", default="/home/xdai/models/Qwen3-8B")
    p.add_argument("--train-data-path", default="/home/xdai/data/alpaca/train.jsonl")
    p.add_argument("--seq-length", type=int, default=256)
    p.add_argument("--layers-to-capture", type=str, default=None)
    p.add_argument("--heatmap-size", type=int, default=128)
    p.add_argument("--output-dir", type=str, default="./hadamard_wgrad_analysis")
    p.add_argument("--init-from-scratch", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model ...")
    if args.init_from_scratch:
        cfg = AutoConfig.from_pretrained(args.model_name_or_path)
        cfg.torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(device)
    model.train()

    if args.layers_to_capture:
        targets = [n.strip() for n in args.layers_to_capture.split(",")]
    else:
        targets = [
            "model.layers.5.mlp.gate_proj",
            "model.layers.5.mlp.down_proj",
            "model.layers.5.self_attn.q_proj",
            "model.layers.15.mlp.down_proj",
        ]

    hooks = []
    for name, mod in model.named_modules():
        if name in targets:
            hooks.append(mod.register_forward_hook(_fwd_hook(name)))
            hooks.append(mod.register_full_backward_hook(_bwd_hook(name)))
            log.info("  Hook: %s", name)

    if not hooks:
        log.error("No hooks — check layer names")
        return

    # Prepare one batch with mostly real tokens
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.train_data_path) as f:
        lines = [json.loads(next(f)) for _ in range(8)]
    text = " ".join(item.get("text", item.get("instruction", "")) for item in lines)

    enc = tok(text, return_tensors="pt", max_length=args.seq_length,
              truncation=True, padding=False)
    input_ids = enc["input_ids"].to(device)
    real_tokens = input_ids.shape[1]
    log.info("Input: %d real tokens (no padding)", real_tokens)

    # Forward + backward
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    log.info("Forward + backward done, loss=%.4f", out.loss.item())

    for h in hooks:
        h.remove()

    # ---- Compute WGrad three ways per layer ----
    results = {}
    for name in targets:
        if name not in _fwd_activations or name not in _bwd_gradients:
            log.warning("  %s: missing activation or gradient, skipping", name)
            continue

        x = _fwd_activations[name]       # (B*S, K) or (B, S, K)
        dy = _bwd_gradients[name]         # (B*S, N) or (B, S, N)

        x_2d = x.reshape(-1, x.shape[-1]).to(device)
        dy_2d = dy.reshape(-1, dy.shape[-1]).to(device)
        M, K = x_2d.shape
        _, N = dy_2d.shape

        # Pad to multiples of 32 for MXFP4
        def pad32(t, dim):
            s = t.shape[dim]
            if s % 32 == 0:
                return t
            pad_n = 32 - (s % 32)
            pads = [0] * (2 * t.dim())
            pads[-(2 * dim + 1)] = pad_n
            return torch.nn.functional.pad(t, list(reversed(pads)))

        x_2d = pad32(x_2d, 0)
        x_2d = pad32(x_2d, 1)
        dy_2d = pad32(dy_2d, 0)
        dy_2d = pad32(dy_2d, 1)

        log.info("=" * 60)
        log.info("Layer: %s", name)
        log.info("  X: %s, dY: %s  (padded)", list(x_2d.shape), list(dy_2d.shape))

        wg_ref = wgrad_bf16(dy_2d, x_2d)
        wg_had = wgrad_mxfp4_with_hadamard(dy_2d, x_2d)
        wg_noh = wgrad_mxfp4_no_hadamard(dy_2d, x_2d)
        torch.cuda.synchronize()

        # Trim back to original weight shape
        wg_ref = wg_ref[:N, :K].cpu()
        wg_had = wg_had[:N, :K].cpu()
        wg_noh = wg_noh[:N, :K].cpu()

        log.info("  WGrad shape: %s", list(wg_ref.shape))

        m_ref = compute_metrics(wg_ref, wg_ref, "BF16 ref")
        m_had = compute_metrics(wg_ref, wg_had, "Hadamard+MXFP4")
        m_noh = compute_metrics(wg_ref, wg_noh, "plain MXFP4")

        for m in [m_ref, m_had, m_noh]:
            log.info("  ---- %s ----", m["label"])
            for k, v in m.items():
                if k == "label": continue
                log.info("    %-22s %14.6f", k, v)

        results[name] = {"ref": m_ref, "hadamard": m_had, "no_hadamard": m_noh}

        sz = args.heatmap_size
        plot_grad_comparison(wg_ref, wg_had, wg_noh, name, out_dir, sz, sz, kind="WGrad")
        plot_error_heatmap(wg_ref, wg_had, wg_noh, name, out_dir, sz, sz, kind="WGrad")
        plot_element_hist(wg_ref, wg_had, wg_noh, name, out_dir, kind="WGrad")

        # ---- DGrad analysis ----
        if name in _fwd_weights:
            w = _fwd_weights[name].to(device)
            w_N, w_K = w.shape
            # Pad weight dims to multiples of 32
            w = pad32(w, 0)
            w = pad32(w, 1)

            log.info("  ---- DGrad (%s) ----", name)
            log.info("  W: %s, dY: %s", list(w.shape), list(dy_2d.shape))

            dg_ref = dgrad_bf16(dy_2d, w)
            dg_had = dgrad_mxfp4_with_hadamard(dy_2d, w)
            dg_noh = dgrad_mxfp4_no_hadamard(dy_2d, w)
            torch.cuda.synchronize()

            dg_ref = dg_ref[:M, :w_K].cpu()
            dg_had = dg_had[:M, :w_K].cpu()
            dg_noh = dg_noh[:M, :w_K].cpu()

            m_dg_ref = compute_metrics(dg_ref, dg_ref, "DGrad BF16 ref")
            m_dg_had = compute_metrics(dg_ref, dg_had, "DGrad Hadamard+MXFP4")
            m_dg_noh = compute_metrics(dg_ref, dg_noh, "DGrad plain MXFP4")

            for m in [m_dg_ref, m_dg_had, m_dg_noh]:
                log.info("  ---- %s ----", m["label"])
                for k, v in m.items():
                    if k == "label": continue
                    log.info("    %-22s %14.6f", k, v)

            results[name]["dgrad_ref"] = m_dg_ref
            results[name]["dgrad_hadamard"] = m_dg_had
            results[name]["dgrad_no_hadamard"] = m_dg_noh

            plot_grad_comparison(dg_ref, dg_had, dg_noh, name, out_dir, sz, sz, kind="DGrad")
            plot_error_heatmap(dg_ref, dg_had, dg_noh, name, out_dir, sz, sz, kind="DGrad")
            plot_element_hist(dg_ref, dg_had, dg_noh, name, out_dir, kind="DGrad")

    json_path = out_dir / "wgrad_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Summary: %s", json_path)


if __name__ == "__main__":
    main()

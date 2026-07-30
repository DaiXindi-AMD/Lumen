#!/usr/bin/env python3
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3-0.6B pretraining from random init — MXFP4 / FP8 / BF16 + TensorBoard.

Validates MXFP4 training by comparing against BF16 baseline on the same data.
Model is randomly initialized (no pretrained weights needed).

Usage (run both, then compare in TensorBoard):
    # BF16 baseline
    torchrun --nproc_per_node=8 examples/qwen3/pretrain_qwen3_mxfp4.py \
        --mode bf16 --dataset c4 --max-steps 10000 \
        --tensorboard-dir ./runs/bf16_qwen3

    # MXFP4
    torchrun --nproc_per_node=8 examples/qwen3/pretrain_qwen3_mxfp4.py \
        --mode mxfp4 --dataset c4 --max-steps 10000 \
        --tensorboard-dir ./runs/mxfp4_qwen3

    # Compare
    tensorboard --logdir ./runs --port 6006 --bind_all
"""
import argparse
import itertools
import logging
import math
import os
import time
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler, Dataset, IterableDataset

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from lumen.config import LumenConfig
from lumen.models.fsdp import _rank0_print as rank0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class WikitextDataset(Dataset):
    """Tokenize wikitext into fixed-length chunks (small, fits in memory)."""

    def __init__(self, split, tokenizer, seq_length, max_samples=None):
        from datasets import load_dataset
        raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(row["text"] for row in raw if row["text"].strip())
        tokens = tokenizer(text, return_attention_mask=False)["input_ids"]
        chunk_len = seq_length + 1
        n_chunks = len(tokens) // chunk_len
        if max_samples:
            n_chunks = min(n_chunks, max_samples)
        self.chunks = [tokens[i * chunk_len:(i + 1) * chunk_len] for i in range(n_chunks)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": torch.LongTensor(self.chunks[idx])}


class C4StreamingDataset(IterableDataset):
    """C4 streaming — no download, each rank sees distinct data."""

    def __init__(self, tokenizer, seq_length, rank=0, world_size=1, split="train",
                 val_samples=200):
        self.tokenizer = tokenizer
        self.chunk_len = seq_length + 1
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.val_samples = val_samples

    def _token_stream(self):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split=self.split, streaming=True)
        # Each rank skips to its shard
        ds = ds.skip(self.rank)
        buf = []
        for i, row in enumerate(ds):
            if i % self.world_size != 0:
                continue
            tokens = self.tokenizer(row["text"], return_attention_mask=False)["input_ids"]
            buf.extend(tokens)
            while len(buf) >= self.chunk_len:
                yield {"input_ids": torch.LongTensor(buf[:self.chunk_len])}
                buf = buf[self.chunk_len:]

    def __iter__(self):
        if self.split == "validation":
            count = 0
            for item in self._token_stream():
                yield item
                count += 1
                if count >= self.val_samples:
                    return
        else:
            yield from self._token_stream()


def build_dataloaders(args, tokenizer, global_rank, world_size):
    if args.dataset == "wikitext":
        train_ds = WikitextDataset("train", tokenizer, args.seq_length)
        val_ds = WikitextDataset("validation", tokenizer, args.seq_length)
        rank0(f"> Data: wikitext-2, {len(train_ds)} train / {len(val_ds)} val chunks")
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=True)
    else:
        rank0(f"> Data: C4 streaming (each rank sees distinct shards)")
        train_ds = C4StreamingDataset(tokenizer, args.seq_length, rank=global_rank,
                                      world_size=world_size, split="train")
        val_ds = C4StreamingDataset(tokenizer, args.seq_length, rank=global_rank,
                                    world_size=world_size, split="validation",
                                    val_samples=args.val_batches)
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size,
                                num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def _configure_mxfp4_dispatch(args):
    """Hand Lumen the Qwen3 tuned A4W4 table and a place to keep autotune results.

    The table only widens which shapes can reach the prebuilt ASM kernels; every
    MXFP4 backend produces bit-identical output, so this changes speed and nothing
    else. The autotune cache mainly buys reproducibility: a few shapes sit close
    enough to the switch margin that two processes can otherwise rank the backends
    differently.

    Both are skipped if the corresponding environment variable is already set.
    """
    from lumen.ops.quantize import mxfp4_autotune

    applied = mxfp4_autotune.configure(
        tuned_config=os.path.join(_CONFIG_DIR, "a4w4_blockscale_tuned_gemm.csv"),
        autotune_cache=args.mxfp4_autotune_cache or None,
    )
    rank0(f"> MXFP4 tuned config: {applied['tuned_config'] or '(aiter default)'}")
    rank0(f"> MXFP4 autotune cache: {applied['autotune_cache'] or '(not persisted)'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                   help="HF model id for config + tokenizer (weights are NOT downloaded, model is randomly initialized)")
    p.add_argument("--mode", choices=["bf16", "fp8_blockwise2d", "mxfp4"], default="mxfp4")
    p.add_argument("--dataset", choices=["wikitext", "c4"], default="c4")
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--sharding", choices=["full_shard", "shard_grad_op"], default="full_shard")
    p.add_argument("--fsdp-version", type=int, choices=[1, 2], default=2)
    p.add_argument("--aiter-attn", action="store_true")
    p.add_argument("--mxfp4-autotune-cache", type=str, default="",
                   help="JSON file to persist which GEMM backend won per shape, so "
                        "the choice is reproducible across runs (mxfp4 mode only)")
    p.add_argument("--lumen-norm", action="store_true")
    p.add_argument("--fuse-rope", action="store_true")
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    p.set_defaults(grad_checkpointing=True)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--profile-steps", type=int, default=0,
                   help="Profile this many steps (after a short warmup), print a "
                        "GPU-time breakdown, and stop. 0 disables profiling.")
    p.add_argument("--profile-out", type=str, default=None,
                   help="Optional chrome trace path for --profile-steps.")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tensorboard-dir", type=str, default="auto",
                   help="TensorBoard log dir. 'auto' generates ./runs/<mode>_<model>_<MMDD_HHMM>. 'none' disables.")
    p.add_argument("--num-workers", type=int, default=2)
    # --- Weights & Biases (optional; runs alongside TensorBoard) ---
    p.add_argument("--wandb-project", type=str, default=None,
                   help="W&B project name. If set, metrics are also logged to wandb.")
    p.add_argument("--wandb-name", type=str, default=None,
                   help="W&B run name. Defaults to the TensorBoard dir basename.")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="W&B entity (team/user). Defaults to your logged-in default.")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B mode. 'offline' logs locally (sync later); 'disabled' is a no-op.")
    args = p.parse_args()

    if args.tensorboard_dir == "auto":
        from datetime import datetime
        ts = datetime.now().strftime("%m%d_%H%M")
        model_short = args.model.split("/")[-1].lower().replace("-", "")
        args.tensorboard_dir = f"./runs/{args.mode}_{model_short}_{ts}"
    elif args.tensorboard_dir == "none":
        args.tensorboard_dir = None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)

    # --- Model: random init from HF config ---
    rank0(f"> Initializing {args.model} from random weights ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to(torch.device("cuda", local_rank))
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    rank0(f"> Model initialized: {param_count:.0f}M params, head_dim={config.head_dim}")

    if args.fuse_rope:
        import transformers.models.qwen3.modeling_qwen3 as _q3
        from lumen.ops.rope import apply_rotary_qk_autograd
        def _lumen_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rotary_qk_autograd(q, k, cos, sin)
        _q3.apply_rotary_pos_emb = _lumen_rope
        rank0("> Fused RoPE enabled")

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        rank0("> Gradient checkpointing enabled")

    # --- Lumen quantization ---
    if args.mode == "mxfp4":
        _configure_mxfp4_dispatch(args)
        use_quant, fmt, scaling, blk = True, "mxfp4", "blockwise", 32
    elif args.mode == "fp8_blockwise2d":
        use_quant, fmt, scaling, blk = True, "fp8_e4m3", "blockwise2d", 128
    else:
        use_quant, fmt, scaling, blk = False, "fp8_e4m3", "delayed", 128

    # MXFP4: keep last ~15% layers in BF16 (NVFP4 paper §4:末尾层最敏感)
    tail_bf16 = args.mode == "mxfp4"
    num_layers = getattr(config, "num_hidden_layers", 0)
    tail_count = max(1, round(num_layers * 0.15)) if tail_bf16 else 0

    cfg = LumenConfig.from_args(Namespace(
        linear_fp8=use_quant, linear_fp8_format=fmt, linear_fp8_scaling=scaling,
        linear_fp8_block_size=blk, linear_fp8_amax_algo="max", linear_fp8_amax_history=16,
        linear_fp8_reduce_amax=False, linear_fp8_activation=True, linear_fp8_wgrad=True,
        linear_fp8_cache_frozen_weight=False, linear_fp8_bpreshuffle=False,
        grad_quant_type=None,
        first_last_layers_bf16=tail_bf16,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=tail_count,
        num_layers=num_layers,
        lumen_norm=args.lumen_norm, hf_attn_patch=args.aiter_attn,
        lora_rank=0, lora_alpha=32.0, lora_dropout=0.0,
    ))
    _manager, model = cfg.enable(model)
    if tail_bf16:
        rank0(f"> Lumen enabled: mode={args.mode}, format={fmt}, scaling={scaling}, block_size={blk}, last {tail_count}/{num_layers} layers BF16")
    else:
        rank0(f"> Lumen enabled: mode={args.mode}, format={fmt}, scaling={scaling}, block_size={blk}")

    # --- FSDP ---
    if args.fsdp_version == 2:
        from lumen.models.fsdp import apply_fsdp2
        apply_fsdp2(model, Namespace(
            linear_fp8=use_quant, sharding_strategy=args.sharding,
            fsdp_fp8_param_storage=False,
        ))
        rank0(f"> FSDP2 ready (sharding={args.sharding})")
    else:
        _shard = {"full_shard": ShardingStrategy.FULL_SHARD,
                  "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP}[args.sharding]
        from functools import partial
        model = FSDP(
            model,
            auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16),
            sharding_strategy=_shard, device_id=local_rank, use_orig_params=True,
        )
        rank0(f"> FSDP1 ready (sharding={args.sharding})")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_loader, val_loader = build_dataloaders(args, tokenizer, global_rank, world_size)

    # --- Optimizer + scheduler ---
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    def lr_lambda(step):
        w, T = args.lr_warmup_steps, args.max_steps
        if step < w:
            return float(step) / max(w, 1)
        progress = (step - w) / max(T - w, 1)
        return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # --- TensorBoard ---
    tb_writer = None
    if global_rank == 0 and args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        rank0(f"> TensorBoard logging to {args.tensorboard_dir}")

    # --- Weights & Biases (rank 0 only) ---
    wb_run = None
    if global_rank == 0 and args.wandb_project:
        import wandb
        run_name = args.wandb_name
        if run_name is None and args.tensorboard_dir:
            run_name = os.path.basename(args.tensorboard_dir.rstrip("/"))
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )
        rank0(f"> W&B logging to project '{args.wandb_project}' as run '{run_name}'")

    def log_metrics(metrics: dict, step: int):
        """Log a flat {name: value} dict to TensorBoard and/or W&B."""
        if tb_writer:
            for k, v in metrics.items():
                tb_writer.add_scalar(k, v, step)
        if wb_run:
            wb_run.log(metrics, step=step)

    # --- Helpers ---
    def compute_loss(batch):
        ids = batch["input_ids"][:, :-1].to(local_rank)
        labels = batch["input_ids"][:, 1:].to(local_rank)
        logits = model(input_ids=ids).logits
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))

    @torch.no_grad()
    def validate():
        model.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            total += compute_loss(batch).item()
            count += 1
            if count >= args.val_batches:
                break
        model.train()
        avg = total / max(count, 1)
        if dist.is_initialized():
            t = torch.tensor([avg], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg

    # --- Training loop ---
    model.train()
    ga = args.gradient_accumulation_steps
    it = iter(train_loader)
    rank0(f"> Training starts: {args.max_steps} steps, batch_size={args.micro_batch_size}x{ga}x{world_size}")

    prof = None
    profile_until = 0
    if args.profile_steps > 0:
        from torch.profiler import ProfilerActivity, profile
        # Skipped steps cover allocator warmup and the one-off MXFP4 autotune
        # measurement, neither of which is representative of a steady step.
        profile_warmup = 3
        profile_until = profile_warmup + args.profile_steps
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )

    for step in range(1, args.max_steps + 1):
        if prof is not None and step == profile_warmup + 1:
            prof.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        acc_loss = 0.0
        for _ in range(ga):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            loss = compute_loss(batch)
            (loss / ga).backward()
            acc_loss += loss.item()
        if args.max_grad_norm > 0:
            if args.fsdp_version == 2:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                model.clip_grad_norm_(args.max_grad_norm)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1e3

        if step % args.log_interval == 0:
            train_loss = acc_loss / ga
            lr = sched.get_last_lr()[0]
            mem_gb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)
            rank0(f"  step {step}/{args.max_steps} | loss {train_loss:.4f} | lr {lr:.2e} | step_time_ms {step_time_ms:.1f} | mem {mem_gb:.1f}GB")
            log_metrics({
                "train/loss": train_loss,
                "train/lr": lr,
                "train/step_time_ms": step_time_ms,
                "gpu/peak_memory_gb": mem_gb,
            }, step)

        if prof is not None and step == profile_until:
            prof.stop()
            rank0(prof.key_averages().table(
                sort_by="self_device_time_total", row_limit=40, max_name_column_width=90
            ))
            rank0("\n>>> same data, split by operand shape:")
            rank0(prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_device_time_total", row_limit=30, max_name_column_width=45
            ))
            if args.profile_out:
                prof.export_chrome_trace(args.profile_out)
                rank0(f"> chrome trace written to {args.profile_out}")
            break

        if step % args.eval_interval == 0:
            val_loss = validate()
            rank0(f"  step {step}/{args.max_steps} | val_loss {val_loss:.4f}")
            log_metrics({"val/loss": val_loss}, step)

    if tb_writer:
        tb_writer.close()
    if wb_run:
        wb_run.finish()
    rank0(f"> Training complete after {args.max_steps} steps.")


if __name__ == "__main__":
    main()

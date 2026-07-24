#!/usr/bin/env python3
"""Pure PyTorch BF16 pretraining baseline — zero Lumen dependency.

No Lumen, no AITER, no custom kernels. Standard HuggingFace model +
PyTorch FSDP2 + C4 streaming. Used as a clean reference for MXFP4 comparison.

Usage:
    torchrun --nproc_per_node=8 examples/qwen3/pretrain_qwen3_bf16_baseline.py \
        --model Qwen/Qwen3-8B --max-steps 5000 \
        --tensorboard-dir ./runs/bf16_baseline_8b_0723
"""
import argparse
import logging
import math
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, IterableDataset
from functools import partial

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

logger = logging.getLogger(__name__)


def rank0_print(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


class C4StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, seq_length, rank=0, world_size=1, split="train", max_samples=None):
        self.tokenizer = tokenizer
        self.chunk_len = seq_length + 1
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.max_samples = max_samples

    def _token_stream(self):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split=self.split, streaming=True)
        ds = ds.skip(self.rank)
        buf = []
        count = 0
        for i, row in enumerate(ds):
            if i % self.world_size != 0:
                continue
            tokens = self.tokenizer(row["text"], return_attention_mask=False)["input_ids"]
            buf.extend(tokens)
            while len(buf) >= self.chunk_len:
                yield {"input_ids": torch.LongTensor(buf[:self.chunk_len])}
                buf = buf[self.chunk_len:]
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return

    def __iter__(self):
        yield from self._token_stream()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tensorboard-dir", type=str, default="auto")
    args = p.parse_args()

    if args.tensorboard_dir == "auto":
        ts = datetime.now().strftime("%m%d_%H%M")
        model_short = args.model.split("/")[-1].lower().replace("-", "")
        args.tensorboard_dir = f"./runs/bf16_baseline_{model_short}_{ts}"

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

    # --- Model ---
    rank0_print(f"> Initializing {args.model} from random weights (pure PyTorch, no Lumen) ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to(torch.device("cuda", local_rank))
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    rank0_print(f"> Model: {param_count:.0f}M params")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- FSDP2 ---
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    rank0_print(f"> FSDP2 applied (pure PyTorch)")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_ds = C4StreamingDataset(tokenizer, args.seq_length, rank=global_rank, world_size=world_size)
    val_ds = C4StreamingDataset(tokenizer, args.seq_length, rank=global_rank, world_size=world_size,
                                split="validation", max_samples=args.val_batches)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, num_workers=0, pin_memory=True)
    rank0_print(f"> Data: C4 streaming")

    # --- Optimizer ---
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
        rank0_print(f"> TensorBoard: {args.tensorboard_dir}")

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

    # --- Training ---
    model.train()
    it = iter(train_loader)
    rank0_print(f"> Training starts: {args.max_steps} steps, lr={args.lr}")

    for step in range(1, args.max_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        loss = compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1e3

        if step % args.log_interval == 0:
            train_loss = loss.item()
            lr = sched.get_last_lr()[0]
            mem_gb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)
            rank0_print(f"  step {step}/{args.max_steps} | loss {train_loss:.4f} | lr {lr:.2e} | step_time_ms {step_time_ms:.1f} | mem {mem_gb:.1f}GB")
            if tb_writer:
                tb_writer.add_scalar("train/loss", train_loss, step)
                tb_writer.add_scalar("train/lr", lr, step)
                tb_writer.add_scalar("train/step_time_ms", step_time_ms, step)
                tb_writer.add_scalar("gpu/peak_memory_gb", mem_gb, step)

        if step % args.eval_interval == 0:
            val_loss = validate()
            rank0_print(f"  step {step}/{args.max_steps} | val_loss {val_loss:.4f}")
            if tb_writer:
                tb_writer.add_scalar("val/loss", val_loss, step)

    if tb_writer:
        tb_writer.close()
    rank0_print(f"> Training complete after {args.max_steps} steps.")


if __name__ == "__main__":
    main()

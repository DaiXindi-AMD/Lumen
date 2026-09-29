# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3-8B training — PyTorch FSDP + Lumen FP8 blockwise2d or MXFP4.

Supports both full-parameter causal-LM pretraining and LoRA SFT:
  - AutoModelForCausalLM + Qwen3DecoderLayer FSDP wrap policy
  - pretraining from a pretrained checkpoint or random weights from its config
  - raw text / ``{"text": ...}`` jsonl pretraining data
  - alpaca-style jsonl SFT data with an answer-only loss mask
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from argparse import Namespace
from collections import Counter, defaultdict
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from lumen.config import LumenConfig

# Reuse the FSDP trainer's rank0 logger so startup/quant/LoRA lines share the
# `INFO:lumen.models.fsdp:` / `INFO:lumen.quantize:` style of the llama2 runs.
from lumen.models.fsdp import (
    _rank0_print as rank0,
    register_quant_optimizer_hooks,
    validate_fsdp_quant_args,
)

_PAIRED_EVIDENCE_ENV = "LUMEN_PAIRED_RUN_EVIDENCE"
_QWEN3_LAST_LAYER_BF16_PROJECTION_NAMES = (
    "o_proj",
    "down_proj",
)
_MXFP4_LINEAR_METADATA_ATTRS = (
    "_quant_manager",
    "_quant_backend",
    "_quant_tensor_id",
    "_quant_scaling_type",
    "_lumen_scaling_manager",
    "_lumen_scaling_type",
    "_lumen_fp8_dtype",
    "_lumen_block_size",
    "_lumen_quantize_activation",
    "_lumen_act_tensor_id",
    "_lumen_frozen",
    "_lumen_cache_frozen",
    "_lumen_bpreshuffle",
    "_fp8_weight_data",
    "_fp8_weight_scale",
    "_fp8_weight_dtype",
)
_MXFP4_WEIGHT_CACHE_ATTRS = (
    "_mxfp4_w_cache",
    "_mxfp4_w_cache_version",
    "_mxfp4_w_cache_sources",
    "_mxfp4_w_cache_metadata",
)


def _paired_evidence_enabled():
    """Whether to emit one-time evidence for strict BF16/MXFP4 pairing."""
    return os.environ.get(_PAIRED_EVIDENCE_ENV, "0") == "1"


def _update_sha256_tensor(digest, tensor):
    """Add a CPU tensor's metadata and bytes to a SHA256 digest."""
    if tensor.device.type != "cpu":
        raise ValueError("paired-run evidence expects CPU tensors")
    value = tensor.detach().contiguous()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))


def _model_init_sha256(model):
    """Hash model parameters before quantization changes the module path."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = parameter.detach()
        if value.device.type != "cpu":
            value = value.cpu()
        _update_sha256_tensor(digest, value)
    return digest.hexdigest()


def _update_pretrain_batch_sha256(digest, batch, microbatch):
    """Roll one CPU pretraining micro-batch into a rank-local digest."""
    digest.update(microbatch.to_bytes(8, byteorder="little", signed=False))
    for key in ("input_ids", "labels"):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        _update_sha256_tensor(digest, batch[key])


def _validation_evidence_line(global_rank, batches, digest):
    """Format rank-local validation identity without changing evaluation math."""
    return (
        f"VALIDATION_EVIDENCE rank={global_rank} batches={batches} "
        f"input_ids_labels_sha256={digest.hexdigest()}"
    )


def _pretrain_shuffle_generator(seed, global_rank):
    """Return a sampler RNG independent of precision-specific Torch RNG use."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + global_rank)
    return generator


def _restore_last_layer_bf16_projections(model, projection_names):
    """Restore selected last-layer residual projections after MXFP4 patching.

    This is an experimental precision guard for narrowing the coarse
    whole-layer BF16 policy. It runs before FSDP wrapping and before the QKV or
    SwiGLU integrations, so those integrations still observe the final
    per-projection quantization policy. Validate every requested target before
    mutating any module so a bad request cannot leave a partially restored
    model.
    """
    requested = tuple(dict.fromkeys(projection_names))
    if not requested:
        return ()

    unknown = sorted(
        set(requested) - set(_QWEN3_LAST_LAYER_BF16_PROJECTION_NAMES)
    )
    if unknown:
        raise ValueError(
            "unsupported Qwen3 last-layer BF16 projections: "
            f"{unknown}; supported residual projections are "
            f"{list(_QWEN3_LAST_LAYER_BF16_PROJECTION_NAMES)}"
        )

    last_layer = _transformer_layers(model)[-1]
    projections = {
        "o_proj": ("self_attn.o_proj", last_layer.self_attn.o_proj),
        "down_proj": ("mlp.down_proj", last_layer.mlp.down_proj),
    }
    targets = []
    for name in requested:
        relative_name, module = projections[name]
        if not isinstance(module, nn.Linear):
            raise RuntimeError(
                f"last-layer projection {relative_name!r} is not nn.Linear"
            )
        if not getattr(module, "_quant_enabled", False) or not hasattr(
            module, "_original_forward"
        ):
            raise RuntimeError(
                f"last-layer projection {relative_name!r} is not MXFP4-patched"
            )
        if getattr(module, "_quant_scaling_type", None) != "mxfp4":
            raise RuntimeError(
                f"last-layer projection {relative_name!r} is not MXFP4-patched"
            )
        targets.append((relative_name, module))

    for _relative_name, module in targets:
        module.forward = module._original_forward
        del module._original_forward
        for attr in _MXFP4_LINEAR_METADATA_ATTRS:
            if hasattr(module, attr):
                delattr(module, attr)
        for owner in (module, module.weight):
            for attr in _MXFP4_WEIGHT_CACHE_ATTRS:
                if hasattr(owner, attr):
                    delattr(owner, attr)
        if hasattr(module.weight, "_lumen_frozen"):
            del module.weight._lumen_frozen
        module._quant_enabled = False
    return tuple(relative_name for relative_name, _module in targets)


def _init_wandb(args, world_size):
    """Open a rank0 wandb run, or return None when --wandb-project is unset.

    The import lives here so a run without the flag never needs wandb installed.
    """
    if not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            **vars(args),
            "world_size": world_size,
            "global_batch_size": world_size
            * args.micro_batch_size
            * args.gradient_accumulation_steps,
        },
    )


class StepProfiler:
    """Rank-0 torch profiler, enabled only when ``LUMEN_PROF_START`` is set."""

    def __init__(self, enabled):
        self.start = int(os.environ.get("LUMEN_PROF_START") or 0)
        self.enabled = enabled and self.start > 0
        self.end = int(os.environ.get("LUMEN_PROF_END") or (self.start + 3))
        self.out = os.environ.get("LUMEN_PROF_OUTPUT") or "/results/qwen3_profile.txt"
        self.trace = os.environ.get("LUMEN_PROF_TRACE") or ""
        self.copy_trace = os.environ.get("LUMEN_COPY_TRACE", "0") == "1"
        self.shapes = os.environ.get("LUMEN_PROF_SHAPES", "0") == "1"
        self.prof = None
        self._step_range = None
        self._tracing = False
        self._copy_n = Counter()
        self._copy_mb = defaultdict(float)
        self._contig_n = Counter()
        self._contig_mb = defaultdict(float)
        if self.enabled:
            rank0(
                f"> Profiler armed: steps {self.start}-{self.end} -> {self.out}"
                + (" (+copy trace)" if self.copy_trace else "")
            )

    def _short_stack(self):
        import traceback

        rel = []
        for frame in traceback.extract_stack()[:-2]:
            if any(part in frame.filename for part in ("lumen/", "examples/")):
                rel.append(f"{frame.filename}:{frame.lineno} {frame.name}")
        return " <- ".join(rel[-3:]) if rel else "(non-lumen)"

    def _install_copy_patches(self):
        self._orig_copy = torch.Tensor.copy_
        self._orig_contig = torch.Tensor.contiguous
        profiler = self

        def _copy(tensor, src, *args, **kwargs):
            if profiler._tracing:
                key = profiler._short_stack()
                profiler._copy_n[key] += 1
                profiler._copy_mb[key] += (
                    tensor.nelement() * tensor.element_size() / 1e6
                )
            return profiler._orig_copy(tensor, src, *args, **kwargs)

        def _contiguous(tensor, *args, **kwargs):
            if profiler._tracing and not tensor.is_contiguous(*args, **kwargs):
                key = profiler._short_stack()
                profiler._contig_n[key] += 1
                profiler._contig_mb[key] += (
                    tensor.nelement() * tensor.element_size() / 1e6
                )
            return profiler._orig_contig(tensor, *args, **kwargs)

        torch.Tensor.copy_ = _copy
        torch.Tensor.contiguous = _contiguous

    def _dump_copy_trace(self):
        torch.Tensor.copy_ = self._orig_copy
        torch.Tensor.contiguous = self._orig_contig
        nsteps = max(1, self.end - self.start + 1)
        path = self.out.replace(".txt", "_copy_trace.txt")
        with open(path, "w") as output:
            output.write(
                f"copy_/contiguous trace, steps {self.start}-{self.end} "
                f"({nsteps} steps)\n\n"
            )
            for title, counts, megabytes in (
                ("copy_", self._copy_n, self._copy_mb),
                ("contiguous", self._contig_n, self._contig_mb),
            ):
                output.write(f"--- {title} by total MB (top 25) ---\n")
                output.write(f"{'calls/step':>10} {'MB/step':>10}  site\n")
                for key, total_mb in sorted(
                    megabytes.items(), key=lambda item: -item[1]
                )[:25]:
                    output.write(
                        f"{counts[key] / nsteps:>10.0f} "
                        f"{total_mb / nsteps:>10.1f}  {key}\n"
                    )
                output.write("\n")
        rank0(f"> Profiler wrote copy trace {path}")

    def step_begin(self, step):
        if self.enabled and step == self.start:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=self.shapes,
            )
            self.prof.__enter__()
            if self.copy_trace:
                self._install_copy_patches()
                self._tracing = True
        if self.prof is not None and self.start <= step <= self.end:
            if self._step_range is not None:
                raise RuntimeError("profiler step range was not closed")
            self._step_range = torch.profiler.record_function(
                f"LUMEN_TRAIN_STEP#{step}"
            )
            self._step_range.__enter__()

    def step_end(self, step):
        if self._step_range is not None:
            self._step_range.__exit__(None, None, None)
            self._step_range = None
        if self.enabled and self.prof is not None and step == self.end:
            self._tracing = False
            self.prof.__exit__(None, None, None)
            table = self.prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40
            )
            with open(self.out, "w") as output:
                output.write(
                    f"Qwen3 FSDP profile, steps {self.start}-{self.end}\n\n{table}"
                )
            rank0(f"> Profiler wrote {self.out}")
            if self.shapes:
                shape_table = self.prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=80
                )
                shape_path = self.out.replace(".txt", "_shapes.txt")
                with open(shape_path, "w") as output:
                    output.write(
                        f"Qwen3 FSDP per-shape profile, steps "
                        f"{self.start}-{self.end}\n\n{shape_table}"
                    )
                rank0(f"> Profiler wrote per-shape table {shape_path}")
            if self.trace:
                self.prof.export_chrome_trace(self.trace)
                rank0(f"> Profiler wrote chrome trace {self.trace}")
            if self.copy_trace:
                self._dump_copy_trace()
            self.prof = None


class AlpacaDataset(Dataset):
    """jsonl rows {instruction, input, output} -> Qwen3 chat, answer-only mask."""

    def __init__(self, path, tokenizer, seq_length, num_samples=None):
        self.tok = tokenizer
        self.seq_length = seq_length
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.n = num_samples or len(self.rows)
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __len__(self):
        return self.n

    def _chat(self, msgs, add_gen):
        # transformers 5.x returns a BatchEncoding when tokenize=True; return_dict
        # gives the flat input_ids list.
        try:
            o = self.tok.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=add_gen,
                enable_thinking=False,
                return_dict=True,
            )
        except TypeError:
            o = self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=add_gen, return_dict=True
            )
        return list(o["input_ids"])

    def __getitem__(self, idx):
        r = self.rows[idx % len(self.rows)]
        prompt = r["instruction"].strip()
        if r.get("input", "").strip():
            prompt += "\n" + r["input"].strip()
        p_ids = self._chat([{"role": "user", "content": prompt}], add_gen=True)
        f_ids = self._chat(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": r["output"]},
            ],
            add_gen=False,
        )
        mask = [0] * len(p_ids) + [1] * max(0, len(f_ids) - len(p_ids))
        ids = list(f_ids)
        L = self.seq_length + 1
        ids, mask = ids[:L], mask[:L]
        if len(ids) < L:
            pad = L - len(ids)
            ids += [self.pad_id] * pad
            mask += [0] * pad
        return {"input_ids": torch.LongTensor(ids), "loss_mask": torch.LongTensor(mask)}


_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def _configure_mxfp4_dispatch():
    """Hand Lumen the Qwen3 tuned A4W4 tables before the first MXFP4 GEMM.

    Widens which shapes can reach the prebuilt ASM kernels: Lumen only lets a
    shape reach ASM only when the tuned table names the exact direct ASM family;
    all other rows are ignored and fall back safely. Skipped if the environment
    variable is already set.

    The per-model table was tuned for Megatron's fused QKV and gate_up, so only
    the few rows whose shapes this path's unfused projections happen to share
    apply here.
    """
    from lumen.ops.quantize import mxfp4_autotune

    applied = mxfp4_autotune.configure(
        tuned_config=[
            os.path.join(_CONFIG_DIR, "qwen3_8b_a4w4_blockscale_tuned_gemm.csv"),
            os.path.join(_CONFIG_DIR, "a4w4_blockscale_tuned_gemm.csv"),
        ],
        autotune_cache=os.environ.get("LUMEN_MXFP4_AUTOTUNE_CACHE") or None,
    )
    rank0(f"> MXFP4 tuned config: {applied['tuned_config'] or '(aiter default)'}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--tokenizer-name-or-path", default=None)
    p.add_argument("--task", choices=["sft", "pretrain"], default="sft")
    p.add_argument(
        "--init-from-scratch",
        action="store_true",
        help="Build random weights from --model-name-or-path's config (pretrain only).",
    )
    p.add_argument(
        "--mode",
        choices=["bf16", "fp8_blockwise2d", "mxfp4"],
        default="fp8_blockwise2d",
    )
    p.add_argument(
        "--mxfp4-pack-gate-up",
        action="store_true",
        help="Run each eligible Qwen3 gate/up pair as one packed MXFP4 projection.",
    )
    p.add_argument(
        "--mxfp4-fuse-swiglu",
        action="store_true",
        help="Fuse separate Qwen3 gate/up SwiGLU forward and backward kernels.",
    )
    p.add_argument(
        "--mxfp4-pack-qkv",
        action="store_true",
        help="Run each eligible Qwen3 Q/K/V group as one packed MXFP4 projection.",
    )
    p.add_argument(
        "--mxfp4-last-layer-bf16-projections",
        nargs="+",
        choices=_QWEN3_LAST_LAYER_BF16_PROJECTION_NAMES,
        default=(),
        help="Experimental tail-0 precision guard: keep only final-layer "
        "o_proj and/or down_proj in BF16.",
    )
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--val-data-path", default=None)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank. Defaults to 16 for SFT and 0 for full pretraining.",
    )
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--cache-frozen-weight",
        action="store_true",
        help="cache the frozen base weight's FP8 quant (skip per-fwd re-quant)",
    )
    p.add_argument(
        "--bpreshuffle",
        action="store_true",
        help="use the ~2.5x-faster B-preshuffle blockscale GEMM (needs --cache-frozen-weight)",
    )
    p.add_argument(
        "--sharding",
        choices=["full_shard", "shard_grad_op"],
        default="full_shard",
        help="FSDP sharding: shard_grad_op (ZeRO-2) avoids per-step param all-gather",
    )
    p.add_argument(
        "--fp8-scaling",
        choices=["blockwise2d", "delayed", "dynamic"],
        default="blockwise2d",
        help="FP8 linear scaling: blockwise2d (128x128, accurate) vs delayed/dynamic (per-tensor, faster GEMM)",
    )
    p.add_argument(
        "--aiter-attn",
        action="store_true",
        help="route SDPA attention through AITER (CK FMHA) instead of PyTorch AOTriton (hf_attn_patch)",
    )
    p.add_argument(
        "--lumen-norm",
        action="store_true",
        help="replace HF Qwen3RMSNorm with Lumen fused RMSNorm (AITER)",
    )
    p.add_argument(
        "--fuse-rope",
        action="store_true",
        help="replace HF apply_rotary_pos_emb with AITER autograd RoPE (fwd+bwd)",
    )
    p.add_argument(
        "--no-grad-checkpointing",
        dest="grad_checkpointing",
        action="store_false",
        help="disable activation checkpointing (no backward forward-recompute; more memory)",
    )
    p.add_argument(
        "--fused-cross-entropy",
        action="store_true",
        help="Compute the pretraining loss with AITER's online-softmax kernel "
        "instead of an FP32 log-softmax. Avoids materializing three "
        "logit-sized tensors, so it saves both time and peak memory.",
    )
    p.add_argument(
        "--grad-checkpoint-layers",
        type=int,
        default=None,
        help="Recompute only this many transformer layers instead of every "
        "one. The recompute forward is a quarter of the MXFP4 GEMMs and "
        "half the attention forwards per step, so this dials that cost "
        "against activation memory. Unset means every layer.",
    )
    p.add_argument(
        "--no-limit-all-gathers",
        dest="limit_all_gathers",
        action="store_false",
        help="allow FSDP to overlap consecutive all-gathers with compute (more memory)",
    )
    p.add_argument(
        "--forward-prefetch",
        action="store_true",
        help="FSDP forward_prefetch: prefetch next unit's all-gather during compute",
    )
    p.add_argument(
        "--fsdp-version",
        type=int,
        choices=[1, 2],
        default=1,
        help="FSDP version: 1 (FullyShardedDataParallel) or 2 (fully_shard)",
    )
    p.add_argument(
        "--fsdp-reduce-dtype",
        choices=["auto", "bf16", "fp32"],
        default="auto",
        help="FSDP2 gradient reduction and accumulation dtype. auto preserves "
        "the existing policy: BF16 uses bf16; FP8/MXFP4 uses fp32.",
    )
    p.add_argument(
        "--fsdp-fp8-param-storage",
        action="store_true",
        help="FSDP2 only: store the frozen blockwise2d base weight as FP8 and "
        "all-gather it as FP8 (no per-step re-quant)",
    )
    p.add_argument(
        "--fsdp-mxfp4-comm",
        action="store_true",
        help="FSDP2 only: all-gather MXFP4-patched weights as packed FP4 "
        "(requires --mode mxfp4 --fsdp-version 2)",
    )
    p.add_argument(
        "--fsdp-retain-accumulated-params",
        action="store_true",
        help="FSDP2 gradient accumulation only: keep parameters unsharded across "
        "the accumulation window and reshard on the final micro-batch. "
        "Trades the memory of one unsharded model for the per-micro-batch "
        "all-gathers.",
    )
    p.add_argument(
        "--fsdp-retain-root-params",
        action="store_true",
        help="Experimental FSDP2 gradient-accumulation optimization: keep the "
        "root full-shard FSDP unit unsharded between forward and backward while "
        "--fsdp-retain-accumulated-params retains decoder-layer parameters. "
        "The final backward still reshards all parameters.",
    )
    p.add_argument(
        "--first-last-layers-bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep configured edge transformer layers in BF16. Defaults on for MXFP4 pretraining.",
    )
    p.add_argument("--num-layers-at-start-in-bf16", type=int, default=0)
    p.add_argument("--num-layers-at-end-in-bf16", type=int, default=5)
    p.set_defaults(grad_checkpointing=True, limit_all_gathers=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=200)
    p.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="Micro-batches per rank to average the validation loss over. "
        "Raise it when the val loss is the measurement, not a progress readout.",
    )
    p.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Cap training samples (0 = use the whole corpus). Pretraining only; "
        "keeps startup tokenization proportional to the requested step count.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--wandb-project",
        default=None,
        help="Log the step metrics to this Weights & Biases project. "
        "Unset means wandb is never imported.",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Run name within --wandb-project. Defaults to wandb's own.",
    )
    return p


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.lora_rank is None:
        args.lora_rank = 0 if args.task == "pretrain" else 16
    if args.mxfp4_pack_gate_up and args.mode != "mxfp4":
        raise ValueError("--mxfp4-pack-gate-up requires --mode mxfp4")
    if args.mxfp4_pack_gate_up and args.lora_rank != 0:
        raise ValueError("--mxfp4-pack-gate-up requires --lora-rank 0")
    if args.mxfp4_fuse_swiglu and args.mode != "mxfp4":
        raise ValueError("--mxfp4-fuse-swiglu requires --mode mxfp4")
    if args.mxfp4_fuse_swiglu and args.lora_rank != 0:
        raise ValueError("--mxfp4-fuse-swiglu requires --lora-rank 0")
    if args.mxfp4_fuse_swiglu and args.mxfp4_pack_gate_up:
        raise ValueError(
            "--mxfp4-fuse-swiglu and --mxfp4-pack-gate-up are mutually exclusive"
        )
    if args.mxfp4_pack_qkv and args.mode != "mxfp4":
        raise ValueError("--mxfp4-pack-qkv requires --mode mxfp4")
    if args.mxfp4_pack_qkv and args.lora_rank != 0:
        raise ValueError("--mxfp4-pack-qkv requires --lora-rank 0")
    if args.mxfp4_last_layer_bf16_projections:
        if args.mode != "mxfp4":
            raise ValueError(
                "--mxfp4-last-layer-bf16-projections requires --mode mxfp4"
            )
        if args.lora_rank != 0:
            raise ValueError(
                "--mxfp4-last-layer-bf16-projections requires --lora-rank 0"
            )
        if args.num_layers_at_end_in_bf16 != 0:
            raise ValueError(
                "--mxfp4-last-layer-bf16-projections requires "
                "--num-layers-at-end-in-bf16 0"
            )
        args.mxfp4_last_layer_bf16_projections = tuple(
            dict.fromkeys(args.mxfp4_last_layer_bf16_projections)
        )
    if args.task == "pretrain" and args.lora_rank != 0:
        raise ValueError(
            "--task pretrain requires --lora-rank 0 (full-parameter training)"
        )
    if args.task != "pretrain" and args.init_from_scratch:
        raise ValueError("--init-from-scratch is only valid with --task pretrain")
    if args.task != "pretrain" and args.train_samples:
        raise ValueError("--train-samples is only valid with --task pretrain")
    if args.train_samples < 0:
        raise ValueError("--train-samples must be >= 0")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")
    if args.fsdp_retain_accumulated_params and args.fsdp_version != 2:
        raise ValueError("--fsdp-retain-accumulated-params requires --fsdp-version 2")
    if args.fsdp_retain_root_params:
        if args.fsdp_version != 2:
            raise ValueError("--fsdp-retain-root-params requires --fsdp-version 2")
        if not args.fsdp_retain_accumulated_params:
            raise ValueError(
                "--fsdp-retain-root-params requires "
                "--fsdp-retain-accumulated-params"
            )
        if args.gradient_accumulation_steps <= 1:
            raise ValueError(
                "--fsdp-retain-root-params requires "
                "--gradient-accumulation-steps > 1"
            )
        if args.sharding != "full_shard":
            raise ValueError(
                "--fsdp-retain-root-params requires --sharding full_shard"
            )
    if args.fsdp_reduce_dtype != "auto" and args.fsdp_version != 2:
        raise ValueError("--fsdp-reduce-dtype requires --fsdp-version 2")
    if args.fused_cross_entropy and args.task != "pretrain":
        # The kernel's reduction divides by every row. Only the packed
        # pretraining labels have no ignored positions; the SFT path weights by
        # its own loss mask and would silently get a different denominator.
        raise ValueError("--fused-cross-entropy requires --task pretrain")
    if args.grad_checkpoint_layers is not None:
        if not args.grad_checkpointing:
            raise ValueError(
                "--grad-checkpoint-layers conflicts with --no-grad-checkpointing"
            )
        if args.grad_checkpoint_layers < 0:
            raise ValueError("--grad-checkpoint-layers must be >= 0")
    if args.first_last_layers_bf16 is None:
        args.first_last_layers_bf16 = args.task == "pretrain" and args.mode == "mxfp4"
    args.linear_fp8 = args.mode == "fp8_blockwise2d"
    args.linear_fp4 = args.mode == "mxfp4"
    validate_fsdp_quant_args(args)
    return args


def _set_fsdp2_gradient_sync(model, enabled):
    """Enable the final accumulation sync and suppress earlier reduce-scatters."""
    setter = getattr(model, "set_requires_gradient_sync", None)
    if setter is None:
        raise RuntimeError("FSDP2 model does not expose set_requires_gradient_sync")
    setter(enabled, recurse=True)


def _transformer_layers(model):
    """The decoder-layer list, found the way apply_fsdp2 finds it to wrap."""
    for module in model.modules():
        if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
            return module.layers
    raise RuntimeError("No transformer layer ModuleList found on the model")


def _select_recompute_layers(num_layers, requested):
    """Evenly spaced indices of the layers to recompute.

    Every layer costs about the same recompute and holds about the same
    activation memory, so the only dial that matters is how many. Spreading
    them keeps the subset from being an arbitrary prefix.
    """
    if requested >= num_layers:
        return list(range(num_layers))
    if requested <= 0:
        return []
    stride = num_layers / requested
    return [int(i * stride) for i in range(requested)]


def _apply_selective_grad_checkpointing(model, requested):
    """Recompute only `requested` transformer layers; return their indices.

    transformers keeps ``gradient_checkpointing`` on each layer, so this only
    clears the flag where recompute is not wanted. The checkpoint function and
    the module call it wraps stay as ``gradient_checkpointing_enable`` left
    them, which is what keeps FSDP's unshard hooks running on recompute.
    """
    layers = _transformer_layers(model)
    selected = set(_select_recompute_layers(len(layers), requested))
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "gradient_checkpointing"):
            raise RuntimeError(
                "Transformer layer has no gradient_checkpointing attribute; "
                "selective recompute needs the per-layer flag"
            )
        layer.gradient_checkpointing = idx in selected
    return sorted(selected)


def _set_fsdp2_reshard_after_backward(model, enabled):
    """Free the unsharded parameters after backward, or keep them for reuse."""
    setter = getattr(model, "set_reshard_after_backward", None)
    if setter is None:
        raise RuntimeError("FSDP2 model does not expose set_reshard_after_backward")
    setter(enabled, recurse=True)


def _set_fsdp2_reshard_after_forward(model, enabled):
    """Set the post-forward reshard policy on every wrapped submodule.

    The root is skipped on purpose: its lifetime is independent of the decoder
    layers and the experimental root-retention path configures it once after
    wrapping rather than toggling it on every accumulation micro-batch.
    """
    applied = 0
    for module in model.modules():
        if module is model:
            continue
        setter = getattr(module, "set_reshard_after_forward", None)
        if setter is None:
            continue
        setter(enabled, recurse=False)
        applied += 1
    if applied == 0:
        raise RuntimeError("No FSDP2 submodule exposes set_reshard_after_forward")


def _set_fsdp2_root_reshard_after_forward(model, enabled):
    """Set the root FSDP unit's post-forward reshard policy exactly once.

    Accumulation-time policy changes intentionally skip the root. Keeping the
    root unsharded after the final forward lets backward reuse those parameters;
    the existing recursive reshard-after-backward policy still frees them after
    the final micro-batch.
    """
    setter = getattr(model, "set_reshard_after_forward", None)
    if setter is None:
        raise RuntimeError("FSDP2 root does not expose set_reshard_after_forward")
    setter(enabled, recurse=False)


def _set_fsdp2_accumulation_state(
    model, final_micro, retain_params, reshard_after_forward
):
    """Configure communication and parameter lifetime for one micro-batch.

    reshard_after_forward is the policy the model was wrapped with, so the
    final micro-batch restores it instead of assuming full_shard.
    """
    _set_fsdp2_gradient_sync(model, final_micro)
    if not retain_params:
        return
    _set_fsdp2_reshard_after_backward(model, final_micro)
    _set_fsdp2_reshard_after_forward(
        model, reshard_after_forward if final_micro else False
    )


def _pretrain_loss(model, batch, device, fused=False):
    """Compute next-token loss for an already shifted pretraining batch."""
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    logits = model(input_ids=input_ids, use_cache=False).logits
    if fused:
        from lumen.ops.cross_entropy import parallel_cross_entropy

        # The online-softmax kernel reads the BF16 logits and writes the
        # gradient back into them, so the FP32 copy below, the log-softmax
        # output and that output's gradient never exist -- three tensors the
        # size of the logits, which at vocab 151936 and 16384 tokens is ~10 GiB
        # each. Its mean divides by every row, which equals
        # nn.functional.cross_entropy's mean over unignored rows only because
        # packed pretraining labels have no ignored positions; parse_args
        # rejects the flag for the masked SFT loss.
        return parallel_cross_entropy(logits, labels, 0.0, True, None, -100)
    # Match HuggingFace's causal-LM loss: log-softmax and its backward need
    # FP32 even when model activations are BF16.
    logits = logits.float()
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )


def _validation_batch_count(local_batches, requested, device):
    """Return one batch count all ranks can execute without FSDP deadlock."""
    count = torch.tensor(
        min(local_batches, requested), device=device, dtype=torch.int64
    )
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
    count = int(count.item())
    if count == 0:
        raise ValueError(
            "Validation loader has no full micro-batch on at least one rank; "
            "increase --val-samples or reduce --micro-batch-size"
        )
    return count


def _seed_training_rngs(seed, global_rank):
    """Seed shared model RNGs and rank-local stochastic rounding deterministically."""
    random.seed(seed + global_rank)
    torch.manual_seed(seed)


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    # Surface lumen.* INFO logs (quant enable, LoRA) on rank0 only, like FSDPTrainer.
    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    _seed_training_rngs(args.seed, global_rank)

    if args.init_from_scratch:
        rank0(f"> Building Qwen3 from config {args.model_name_or_path} ...")
        model_cfg = AutoConfig.from_pretrained(args.model_name_or_path)
        model_cfg.torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_config(
            model_cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    else:
        rank0(f"> Loading Qwen3 from {args.model_name_or_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    paired_evidence = _paired_evidence_enabled()
    if paired_evidence and global_rank == 0:
        rank0(f"> Paired-run evidence: model_init_sha256={_model_init_sha256(model)}")

    if args.fuse_rope:
        # Replace HF NEOX rope (mul+rotate_half+mul+add) with AITER autograd RoPE.
        import transformers.models.qwen3.modeling_qwen3 as _q3
        from lumen.ops.rope import apply_rotary_qk_autograd

        def _lumen_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return apply_rotary_qk_autograd(q, k, cos, sin)

        _q3.apply_rotary_pos_emb = _lumen_rope
        rank0("> Fused RoPE: HF apply_rotary_pos_emb -> AITER autograd RoPE")
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if args.grad_checkpoint_layers is None:
            rank0("> Gradient checkpointing enabled")
        else:
            recomputed = _apply_selective_grad_checkpointing(
                model, args.grad_checkpoint_layers
            )
            rank0(f"> Gradient checkpointing on {len(recomputed)} layers: {recomputed}")
    else:
        rank0(
            "> Gradient checkpointing DISABLED (more activation memory, no backward recompute)"
        )

    # ---- Lumen full-parameter/LoRA training (+ optional quantised linears) ----
    if args.mode == "mxfp4":
        _configure_mxfp4_dispatch()
    # Reached only by --linear-fp8: from_args overwrites format/scaling/
    # block_size with mxfp4/blockwise/32 whenever linear_fp4 is set.
    fmt, scaling, blk = "fp8_e4m3", args.fp8_scaling, 128
    use_fp8, use_fp4 = args.linear_fp8, args.linear_fp4
    # linear_fp8_* are LumenConfig's argument names, not a precision. Activation,
    # wgrad, cache_frozen_weight and bpreshuffle reach the MXFP4 path as well;
    # the amax fields stay inert there because fp4 forces blockwise scaling.
    cfg = LumenConfig.from_args(
        Namespace(
            linear_fp8=use_fp8,
            linear_fp4=use_fp4,
            linear_fp8_format=fmt,
            linear_fp8_scaling=scaling,
            linear_fp8_block_size=blk,
            linear_fp8_amax_algo="max",
            linear_fp8_amax_history=16,
            linear_fp8_reduce_amax=False,
            linear_fp8_activation=True,
            linear_fp8_wgrad=True,
            linear_fp8_cache_frozen_weight=args.cache_frozen_weight,
            linear_fp8_bpreshuffle=args.bpreshuffle,
            grad_quant_type=None,
            first_last_layers_bf16=args.first_last_layers_bf16,
            num_layers_at_start_in_bf16=args.num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=args.num_layers_at_end_in_bf16,
            num_layers=model.config.num_hidden_layers,
            lumen_norm=args.lumen_norm,
            hf_attn_patch=args.aiter_attn,  # route SDPA -> AITER CK FMHA when set
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    )
    _manager, model = cfg.enable(
        model
    )  # logs INFO:lumen.quantize + INFO:lumen.config (LoRA + trainable)
    restored_last_layer = _restore_last_layer_bf16_projections(
        model, args.mxfp4_last_layer_bf16_projections
    )
    if restored_last_layer:
        rank0(
            "> MXFP4 last-layer BF16 projection guard: "
            + ", ".join(restored_last_layer)
        )
    if args.mxfp4_pack_gate_up:
        from lumen.models.qwen3 import enable_mxfp4_qwen_gate_up

        packed_mlps = enable_mxfp4_qwen_gate_up(model, strict=True)
        rank0(f"> MXFP4 packed gate/up enabled on {packed_mlps} Qwen3 MLPs")
    if args.mxfp4_fuse_swiglu:
        from lumen.models.qwen3 import enable_mxfp4_qwen_swiglu

        fused_mlps = enable_mxfp4_qwen_swiglu(model, strict=True)
        rank0(f"> MXFP4 split SwiGLU enabled on {fused_mlps} Qwen3 MLPs")
    if args.mxfp4_pack_qkv:
        from lumen.models.qwen3 import enable_mxfp4_qwen_qkv

        packed_attentions = enable_mxfp4_qwen_qkv(model, strict=True)
        rank0(
            f"> MXFP4 packed QKV enabled on {packed_attentions} "
            "Qwen3 attention layers"
        )
    # FSDP1 flatten needs uniform dtype: cast PEFT LoRA adapters (fp32) to bf16.
    for nme, prm in model.named_parameters():
        if "lora_" in nme and prm.dtype == torch.float32:
            prm.data = prm.data.to(torch.bfloat16)

    # LoRA can prefer SHARD_GRAD_OP because its base weights are frozen. Full
    # pretraining normally needs FULL_SHARD to shard parameters, gradients, and
    # optimizer states.
    if args.fsdp_version == 2:
        # FSDP2 (fully_shard, per-param sharding). Shards each Qwen3 decoder layer
        # via apply_fsdp2's `.layers` detection; with --fsdp-fp8-param-storage the
        # frozen blockwise2d base weights are wrapped as Blockwise2DFP8Param and
        # all-gathered as FP8 (no per-step re-quant). --fsdp-mxfp4-comm wraps
        # MXFP4 linears so param all-gather ships packed FP4.
        from lumen.models.fsdp import apply_fsdp2

        apply_fsdp2(
            model,
            Namespace(
                linear_fp8=use_fp8,
                linear_fp4=use_fp4,
                fsdp_version=2,
                sharding_strategy=args.sharding,
                fsdp_fp8_param_storage=args.fsdp_fp8_param_storage,
                fsdp_mxfp4_comm=args.fsdp_mxfp4_comm,
                fsdp_reduce_dtype=args.fsdp_reduce_dtype,
            ),
        )
        if args.fsdp_retain_root_params:
            _set_fsdp2_root_reshard_after_forward(model, False)
            rank0(
                "> Experimental FSDP2 root parameter retention enabled: "
                "root parameters remain unsharded from forward through backward"
            )
        rank0(
            f"> FSDP2 model ready (sharding={args.sharding}, "
            f"fp8_param_storage={args.fsdp_fp8_param_storage}, "
            f"mxfp4_comm={args.fsdp_mxfp4_comm}, grad_ckpt={args.grad_checkpointing}, "
            f"retain_accumulated_params={args.fsdp_retain_accumulated_params}, "
            f"retain_root_params={args.fsdp_retain_root_params}, "
            f"world_size={world_size})"
        )
    else:
        _shard = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }[args.sharding]
        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy=_shard,
            device_id=local_rank,
            limit_all_gathers=args.limit_all_gathers,
            forward_prefetch=args.forward_prefetch,
            use_orig_params=True,
        )
        rank0(
            f"> FSDP model ready (sharding={args.sharding}, limit_all_gathers={args.limit_all_gathers}, "
            f"forward_prefetch={args.forward_prefetch}, grad_ckpt={args.grad_checkpointing}, world_size={world_size})"
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    # Must follow optimizer construction: without it the MXFP4 weight cache is
    # never invalidated under FSDP and every quantized layer trains against its
    # step-0 weights.
    register_quant_optimizer_hooks(model, opt, args)

    def lr_lambda(step):
        w, T, mx, mn = args.lr_warmup_steps, args.max_steps, args.lr, args.min_lr
        if step < w:
            return float(step) / max(w, 1)
        prog = float(step - w) / max(T - w, 1)
        return (mn / mx if mx > 0 else 0.0) + (
            1 - (mn / mx if mx > 0 else 0.0)
        ) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path
    )

    def make_loader(path, n, shuffle=True):
        generator = None
        if args.task == "pretrain":
            from lumen.models.llama31.dataset import PretrainTextDataset

            ds = PretrainTextDataset(
                data_path=path,
                seq_length=args.seq_length,
                tokenizer=tok,
                is_hf_tokenizer=True,
                max_samples=(math.ceil(n / world_size) if n is not None else None),
                rank=global_rank,
                world_size=world_size,
            )
            sampler = None
            if shuffle:
                generator = _pretrain_shuffle_generator(args.seed, global_rank)
        else:
            ds = AlpacaDataset(path, tok, args.seq_length, num_samples=n)
            sampler = (
                DistributedSampler(
                    ds, num_replicas=world_size, rank=global_rank, shuffle=shuffle
                )
                if world_size > 1
                else None
            )
        rank0(f"Loaded {len(ds)} samples from {path}")
        return DataLoader(
            ds,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    train_loader = make_loader(args.train_data_path, args.train_samples or None)
    # Validation does not shuffle: comparing two runs only means something when
    # both are scored on the same tokens, and a shuffled loader hands each
    # eval -- and each run -- a different subset of the held-out set.
    val_loader = (
        make_loader(args.val_data_path, args.val_samples, shuffle=False)
        if args.val_data_path
        else None
    )

    def loss_on(b):
        if args.task == "pretrain":
            # PretrainTextDataset already pairs token[t] with token[t+1].
            # HuggingFace's labels= path shifts internally, so use explicit CE
            # to avoid accidentally learning token[t+2].
            return _pretrain_loss(model, b, local_rank, args.fused_cross_entropy)
        ids = b["input_ids"][:, :-1].to(local_rank, non_blocking=True)
        labels = b["input_ids"][:, 1:].to(local_rank, non_blocking=True)
        lm = b["loss_mask"][:, 1:].to(local_rank, non_blocking=True).float()
        logits = model(input_ids=ids, use_cache=False).logits.float()
        per = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        )
        return (per * lm.reshape(-1)).sum() / lm.sum().clamp(min=1)

    @torch.no_grad()
    def validate():
        model.eval()
        target_batches = _validation_batch_count(
            len(val_loader), args.eval_batches, local_rank
        )
        tot_l, nb = 0.0, 0
        validation_digest = (
            hashlib.sha256() if paired_evidence and args.task == "pretrain" else None
        )
        for b in val_loader:
            if validation_digest is not None:
                _update_pretrain_batch_sha256(validation_digest, b, nb)
            tot_l += loss_on(b).item()
            nb += 1
            if nb >= target_batches:
                break
        model.train()
        if validation_digest is not None:
            print(
                _validation_evidence_line(global_rank, nb, validation_digest),
                flush=True,
            )
        totals = torch.tensor([tot_l, nb], device=local_rank, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return (totals[0] / totals[1]).item()

    model.train()
    ga = args.gradient_accumulation_steps
    it = iter(train_loader)
    profiler = StepProfiler(enabled=(global_rank == 0))
    wandb_run = _init_wandb(args, world_size) if global_rank == 0 else None
    first_step_batch_digest = (
        hashlib.sha256() if paired_evidence and args.task == "pretrain" else None
    )
    for step in range(1, args.max_steps + 1):
        profiler.step_begin(step)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        acc = 0.0
        for micro in range(ga):
            try:
                b = next(it)
            except StopIteration:
                it = iter(train_loader)
                b = next(it)
            if step == 1 and first_step_batch_digest is not None:
                _update_pretrain_batch_sha256(first_step_batch_digest, b, micro)
            final_micro = micro == ga - 1
            if args.fsdp_version == 2 and ga > 1:
                _set_fsdp2_accumulation_state(
                    model,
                    final_micro,
                    args.fsdp_retain_accumulated_params,
                    args.sharding != "shard_grad_op",
                )
                sync_context = nullcontext()
            elif args.fsdp_version == 1 and not final_micro:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()
            with sync_context:
                l = loss_on(b)
                (l / ga).backward()
            acc += l.item()
        grad_norm = None
        if args.max_grad_norm > 0:
            # FSDP1 wraps the model and exposes .clip_grad_norm_; FSDP2 (fully_shard)
            # returns the bare model → clip via the param-list utility (handles DTensor).
            if args.fsdp_version == 2:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            else:
                grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1e3
        mean_loss_value = acc / ga
        peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30
        if step == 1 and first_step_batch_digest is not None:
            print(
                f"PAIRING_EVIDENCE rank={global_rank} step=1 microbatches={ga} "
                f"input_ids_labels_sha256={first_step_batch_digest.hexdigest()}",
                flush=True,
            )
            first_step_batch_digest = None
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss": mean_loss_value,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "lr": sched.get_last_lr()[0],
                    "step_time_ms": step_time_ms,
                    "peak_mem_gib": peak_mem_gib,
                },
                step=step,
            )
        if step % args.log_interval == 0:
            # Log the grad norm: it is the cheapest signal that a precision or
            # comm variant is not actually training the same parameters.
            gn = f"{float(grad_norm):.3e}" if grad_norm is not None else "n/a"
            # Peak memory belongs next to step time: the checkpointing and
            # sharding knobs buy speed with memory, so a step time on its own
            # cannot say whether a setting is affordable.
            rank0(
                f"  step {step}/{args.max_steps} | loss {mean_loss_value:.4f} | grad_norm {gn} "
                f"| lr {sched.get_last_lr()[0]:.2e} | step_time_ms {step_time_ms:.1f} "
                f"| peak_mem_gib {peak_mem_gib:.1f}"
            )
        profiler.step_end(step)
        if val_loader and step % args.eval_interval == 0:
            val_loss = validate()
            rank0(f"  step {step}/{args.max_steps} | val_loss {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val_loss": val_loss}, step=step)

    rank0(f"> Training complete after {args.max_steps} steps.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

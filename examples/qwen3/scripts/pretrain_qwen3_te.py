#!/usr/bin/env python3
"""Qwen3-8B pretrain on stock Megatron + TransformerEngine, with no Lumen model code.

This is the control arm for the Lumen-MXFP4-vs-TE-MXFP4 comparison. Lumen's own
Megatron entry point cannot serve as that control: it forces
``transformer_impl="local"`` (``_TE_FORCE_OVERRIDES`` in ``lumen/models/megatron.py``)
and swaps Lumen modules into the layer spec, so TE's own layers never run.

The model here is built entirely by Megatron's ``model_provider`` / ``gpt_builder``,
which is what gives TE its quantized linear layers and fused kernels. The *data*,
however, has to match the Lumen run token for token, or the loss curves are not
comparable — so this reuses Lumen's ``PretrainTextDataset`` and reproduces its
batch construction exactly:

* The dataset class is loaded straight from its file rather than imported as
  ``lumen.models.llama31.dataset``, because that import would execute
  ``lumen/__init__.py`` and ``lumen/models/llama31/__init__.py``, and the latter
  pulls in Lumen's Megatron patches. Loading the file keeps the tokenization and
  chunking byte-identical while leaving Megatron untouched.
* ``get_batch`` below mirrors Lumen's, down to calling the same
  ``get_ltor_masks_and_position_ids`` with the same arguments, so both runs see the
  same supervision.
* The loss is Megatron's ``loss_func``, which is numerically the same expression as
  Lumen's (``sum(losses * mask)`` reported alongside the token count).

Run it from the Megatron checkout so ``model_provider``, ``gpt_builders`` and
``pretrain_gpt`` are importable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import partial
from pathlib import Path

import torch

from gpt_builders import gpt_builder
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import get_patch_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider
from pretrain_gpt import loss_func


def _load_lumen_dataset_class():
    """Load ``PretrainTextDataset`` from Lumen's file without importing the package."""
    lumen_root = os.environ.get("LUMEN_ROOT")
    if not lumen_root:
        raise SystemExit("LUMEN_ROOT must point at the Lumen checkout")
    path = Path(lumen_root) / "lumen" / "models" / "llama31" / "dataset.py"
    if not path.is_file():
        raise SystemExit(f"not found: {path}")
    spec = importlib.util.spec_from_file_location("_lumen_pretrain_dataset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PretrainTextDataset


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the datasets from raw jsonl, matching Lumen's provider."""
    PretrainTextDataset = _load_lumen_dataset_class()
    args = get_args()

    tokenizer_obj = get_tokenizer()
    is_hf = hasattr(tokenizer_obj, "_tokenizer") and hasattr(tokenizer_obj._tokenizer, "encode")
    raw_tokenizer = tokenizer_obj._tokenizer if is_hf else tokenizer_obj

    paths = (
        args.train_data_path[0] if args.train_data_path else None,
        args.valid_data_path[0] if args.valid_data_path else None,
        args.test_data_path[0] if args.test_data_path else None,
    )

    print_rank_0("> building train, validation, and test datasets (raw jsonl) ...")
    datasets = tuple(
        PretrainTextDataset(
            path,
            args.seq_length,
            raw_tokenizer,
            is_hf,
            max_samples=n_samples,
        )
        for path, n_samples in zip(paths, train_val_test_num_samples)
    )
    print_rank_0("> finished creating datasets ...")
    return datasets


def get_batch(data_iterator, vp_stage=None):
    """Standard LM batch — every token contributes. Mirrors Lumen's get_batch."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    args = get_args()
    data = next(data_iterator) if data_iterator is not None else None
    data_b = tensor_parallel.broadcast_data(["input_ids", "labels"], data, torch.int64)

    tokens = data_b["input_ids"].contiguous()
    labels = data_b["labels"].contiguous()

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "eos_token_id"):
        eod_token = tokenizer._tokenizer.eos_token_id
    else:
        eod_token = tokenizer.eod

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        False,
    )

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


def forward_step(data_iterator, model):
    """Forward pass for one micro-batch."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, getattr(model, "vp_stage", None)
    )
    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
    )
    return output_tensor, partial(loss_func, loss_mask, model=model)


def _install_gc_freeze(warmup_steps: int = 20) -> None:
    """Match the Lumen arm's GC policy, so the comparison is not measuring it.

    A step allocates enough short-lived Python objects to reach a generation-2
    collection every few steps, and that collection walks every tracked object
    in the process, which stalls whichever step it lands in.  Freezing after
    warmup leaves only the step's own garbage to scan.  Lumen's entry point does
    the same in ``lumen.models.megatron.install_gc_freeze_hook``; this is a copy
    rather than an import because the control arm deliberately runs without
    Lumen on its path.
    """
    import gc

    import megatron.training.training as _mt_training

    original = _mt_training.train_step
    state = {"calls": 0}

    def _train_step(*args, **kwargs):
        out = original(*args, **kwargs)
        state["calls"] += 1
        if state["calls"] == warmup_steps:
            gc.collect()
            gc.freeze()
            print_rank_0(
                f"> GC: froze {gc.get_freeze_count()} objects after "
                f"{warmup_steps} steps (later collections skip them)"
            )
            _mt_training.train_step = original
        return out

    _mt_training.train_step = _train_step


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    if os.environ.get("LUMEN_GC_FREEZE", "1") != "0":
        _install_gc_freeze()

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        extra_args_provider=get_patch_args,
    )

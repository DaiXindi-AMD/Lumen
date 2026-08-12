###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Framework-agnostic pretraining dataset for LLaMA 3.1.

Provides :class:`PretrainTextDataset` that reads raw text files (one document
per line, or jsonl with a ``"text"`` field) and produces fixed-length token
sequences for causal language-model pretraining.

No dependency on Megatron, HuggingFace Transformers, or any training
framework — only PyTorch and the Python standard library.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

__all__ = ["PretrainTextDataset"]

logger = logging.getLogger(__name__)


class PretrainTextDataset(Dataset):
    """Pretraining dataset that concatenates tokenized documents and chunks
    into fixed-length sequences.

    Supports two input formats:

    - **plain text** (``.txt``): one document per line.
    - **jsonl** (``.jsonl`` / ``.json``): each line is a JSON object with a
      ``"text"`` field.

    Tokens from all documents are concatenated with EOS separators, then
    split into contiguous chunks of ``seq_length + 1`` tokens (the extra
    token provides the label for the last position).

    Args:
        data_path: Path to a ``.txt``, ``.jsonl``, or ``.json`` file.
        seq_length: Context window (number of input tokens per sample).
        tokenizer: Any object with ``encode(text) -> list[int]`` and
            ``eos_token_id``, **or** a Megatron tokenizer with
            ``tokenize()`` / ``eod``.
        is_hf_tokenizer: ``True`` for HuggingFace-style tokenizers.
        max_samples: If set, cap ``__len__`` at this value.
        allow_repeat: Serve ``max_samples`` samples even when the corpus holds
            fewer, wrapping back to chunk 0. Off by default so a dataset never
            silently trains on repeated data; turn it on for a training set that
            must reach a fixed step count on a corpus smaller than one epoch,
            which otherwise ends in a ``StopIteration`` mid-run.
    """

    def __init__(
        self,
        data_path: Optional[str],
        seq_length: int,
        tokenizer,
        is_hf_tokenizer: bool = False,
        max_samples: Optional[int] = None,
        allow_repeat: bool = False,
    ):
        self._allow_repeat = allow_repeat
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer
        self._chunks: List[List[int]] = []

        if data_path is None:
            self._max_samples = max_samples or 0
            return

        all_ids = self._load_and_tokenize(data_path)

        chunk_len = seq_length + 1
        n_chunks = len(all_ids) // chunk_len
        for i in range(n_chunks):
            self._chunks.append(all_ids[i * chunk_len : (i + 1) * chunk_len])

        logger.info(
            "Built %d pretraining samples (seq_length=%d) from %s " "(%d tokens total)",
            len(self._chunks),
            seq_length,
            data_path,
            len(all_ids),
        )

        self._max_samples = max_samples

        if allow_repeat and max_samples is not None and n_chunks and max_samples > n_chunks:
            logger.warning(
                "Corpus holds %d samples but %d were requested: data repeats %.2f times",
                n_chunks,
                max_samples,
                max_samples / n_chunks,
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        n = len(self._chunks)
        if self._max_samples is None:
            return n
        if self._allow_repeat and n:
            return self._max_samples
        return min(n, self._max_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = idx % len(self._chunks)
        chunk = self._chunks[idx]
        return {
            "input_ids": torch.LongTensor(chunk[:-1]),
            "labels": torch.LongTensor(chunk[1:]),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[int]:
        if self.is_hf_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.tokenize(text)

    def _get_eos_id(self) -> int:
        if self.is_hf_tokenizer:
            return self.tokenizer.eos_token_id
        return self.tokenizer.eod

    def _load_and_tokenize(self, data_path: str) -> List[int]:
        """Read documents, tokenize, and concatenate with EOS separators."""
        path = Path(data_path)
        eos_id = self._get_eos_id()
        all_ids: List[int] = []

        if path.suffix in (".jsonl", ".json"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        all_ids.extend(self._tokenize(text))
                        all_ids.append(eos_id)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        all_ids.extend(self._tokenize(text))
                        all_ids.append(eos_id)

        return all_ids

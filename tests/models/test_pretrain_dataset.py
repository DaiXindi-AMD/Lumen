###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for lumen.models.llama31.dataset.PretrainTextDataset.

Covers:
  - chunking of a jsonl corpus into seq_length + 1 token samples
  - max_samples capping one pass over the corpus
  - allow_repeat serving a full step budget from a short corpus by wrapping,
    which is what keeps a fixed-length run from ending in StopIteration
"""

import json
import tempfile
from pathlib import Path

from lumen.models.llama31.dataset import PretrainTextDataset


class FakeTokenizer:
    """HF-style tokenizer emitting one id per character."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def write_corpus(tmpdir, n_docs, doc_len):
    path = Path(tmpdir) / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_docs):
            f.write(json.dumps({"text": "a" * doc_len}) + "\n")
    return str(path)


def build(tmpdir, seq_length=4, n_docs=10, doc_len=9, **kwargs):
    return PretrainTextDataset(
        write_corpus(tmpdir, n_docs, doc_len),
        seq_length,
        FakeTokenizer(),
        is_hf_tokenizer=True,
        **kwargs,
    )


class TestChunking:
    def test_splits_corpus_into_fixed_length_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 10 docs x (9 chars + 1 eos) = 100 ids -> 20 chunks of 5
            ds = build(tmpdir, seq_length=4)
            assert len(ds) == 20
            sample = ds[0]
            assert sample["input_ids"].shape[0] == 4
            assert sample["labels"].shape[0] == 4

    def test_labels_are_inputs_shifted_by_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4)
            sample = ds[0]
            assert sample["input_ids"][1:].tolist() == sample["labels"][:-1].tolist()


class TestMaxSamples:
    def test_caps_below_corpus_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=5)
            assert len(ds) == 5

    def test_does_not_extend_past_corpus_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100)
            assert len(ds) == 20


class TestAllowRepeat:
    def test_serves_full_budget_from_short_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100, allow_repeat=True)
            assert len(ds) == 100

    def test_wraps_back_to_the_first_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100, allow_repeat=True)
            assert ds[20]["input_ids"].tolist() == ds[0]["input_ids"].tolist()

    def test_still_caps_when_corpus_is_larger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=5, allow_repeat=True)
            assert len(ds) == 5

    def test_empty_corpus_stays_empty(self):
        ds = PretrainTextDataset(
            None,
            4,
            FakeTokenizer(),
            is_hf_tokenizer=True,
            max_samples=100,
            allow_repeat=True,
        )
        assert len(ds) == 0

# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the Qwen3 FSDP full-pretraining entrypoint."""

import hashlib
import os
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers.modeling_layers import (
    GradientCheckpointingLayer as _GRAD_CKPT_LAYER,
)
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from examples.qwen3.train_qwen3_fsdp import (
    StepProfiler,
    _apply_selective_grad_checkpointing,
    _model_init_sha256,
    _paired_evidence_enabled,
    _pretrain_loss,
    _pretrain_shuffle_generator,
    _restore_last_layer_bf16_projections,
    _seed_training_rngs,
    _select_recompute_layers,
    _set_fsdp2_accumulation_state,
    _set_fsdp2_gradient_sync,
    _set_fsdp2_reshard_after_backward,
    _set_fsdp2_reshard_after_forward,
    _set_fsdp2_root_reshard_after_forward,
    _update_pretrain_batch_sha256,
    _validation_evidence_line,
    _validation_batch_count,
    parse_args,
)
from lumen.models.qwen3 import (
    disable_mxfp4_qwen_swiglu,
    disable_mxfp4_qwen_qkv,
    enable_mxfp4_qwen_gate_up,
    enable_mxfp4_qwen_qkv,
    enable_mxfp4_qwen_swiglu,
)

_BASE = [
    "--model-name-or-path",
    "Qwen/Qwen3-8B",
    "--train-data-path",
    "train.jsonl",
]


def _fake_torchrun_env(tmp_path):
    """Run the launcher against a torchrun that just echoes its arguments."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    launcher = (
        Path(__file__).resolve().parents[2]
        / "examples/qwen3/run_qwen3_fsdp_mxfp4_pretrain.sh"
    )
    env = os.environ.copy()
    env.update(
        MODEL_PATH=str(tmp_path / "model"),
        TRAIN_DATA_PATH=str(tmp_path / "train.jsonl"),
        RESULTS_DIR=str(tmp_path / "results"),
        NPROC="1",
        MBS="1",
        GBS="1",
        TRAIN_STEPS="1",
        PATH=f"{fake_bin}:{env['PATH']}",
    )
    return env, launcher


@pytest.mark.parametrize(
    ("mxfp4_comm", "expect_compressed_comm"),
    [(None, False), ("1", True)],
)
def test_mxfp4_launcher_comm_compression_is_opt_in(
    tmp_path, mxfp4_comm, expect_compressed_comm
):
    env, launcher = _fake_torchrun_env(tmp_path)
    if mxfp4_comm is None:
        env.pop("MXFP4_COMM", None)
    else:
        env["MXFP4_COMM"] = mxfp4_comm

    result = subprocess.run(
        ["bash", str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert ("--fsdp-mxfp4-comm" in args) is expect_compressed_comm


@pytest.mark.parametrize(
    ("retain", "gbs", "expect_flag", "expect_warning"),
    [
        (None, "16", False, False),
        ("1", "16", True, False),
        # Retention has nothing to save without accumulation, so the launcher
        # must not quietly pass a flag that changes memory for no benefit.
        ("1", "1", False, True),
    ],
)
def test_launcher_retains_accumulated_params_only_when_accumulating(
    tmp_path, retain, gbs, expect_flag, expect_warning
):
    env, launcher = _fake_torchrun_env(tmp_path)
    env["GBS"] = gbs
    env["MBS"] = "1"
    env["NPROC"] = "1"
    if retain is None:
        env.pop("RETAIN_ACCUM_PARAMS", None)
    else:
        env["RETAIN_ACCUM_PARAMS"] = retain

    result = subprocess.run(
        ["bash", str(launcher)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert ("--fsdp-retain-accumulated-params" in args) is expect_flag
    assert ("RETAIN_ACCUM_PARAMS ignored" in result.stderr) is expect_warning


def test_launcher_forwards_fsdp_reduce_dtype_override(tmp_path):
    env, launcher = _fake_torchrun_env(tmp_path)
    env["PRECISION"] = "bf16"
    env["EXTRA_TRAIN_ARGS"] = "--fsdp-reduce-dtype fp32"

    result = subprocess.run(
        ["bash", str(launcher)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    index = args.index("--fsdp-reduce-dtype")
    assert args[index + 1] == "fp32"


def test_pretrain_mxfp4_defaults_to_full_parameters_and_bf16_tail():
    args = parse_args(_BASE + ["--task", "pretrain", "--mode", "mxfp4"])

    assert args.lora_rank == 0
    assert args.linear_fp4 is True
    assert args.linear_fp8 is False
    assert args.first_last_layers_bf16 is True
    assert args.num_layers_at_start_in_bf16 == 0
    assert args.num_layers_at_end_in_bf16 == 5


def test_mxfp4_pack_gate_up_cli_validation():
    with pytest.raises(ValueError, match="requires --mode mxfp4"):
        parse_args(_BASE + ["--mxfp4-pack-gate-up", "--mode", "bf16"])

    with pytest.raises(ValueError, match="requires --lora-rank 0"):
        parse_args(
            _BASE
            + [
                "--task",
                "sft",
                "--mode",
                "mxfp4",
                "--lora-rank",
                "8",
                "--mxfp4-pack-gate-up",
            ]
        )

    args = parse_args(
        _BASE
        + [
            "--task",
            "pretrain",
            "--mode",
            "mxfp4",
            "--mxfp4-pack-gate-up",
        ]
    )
    assert args.mxfp4_pack_gate_up is True


def test_mxfp4_fuse_swiglu_cli_validation():
    with pytest.raises(ValueError, match="requires --mode mxfp4"):
        parse_args(_BASE + ["--mxfp4-fuse-swiglu", "--mode", "bf16"])

    with pytest.raises(ValueError, match="requires --lora-rank 0"):
        parse_args(
            _BASE
            + [
                "--task",
                "sft",
                "--mode",
                "mxfp4",
                "--lora-rank",
                "8",
                "--mxfp4-fuse-swiglu",
            ]
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(
            _BASE
            + [
                "--task",
                "pretrain",
                "--mode",
                "mxfp4",
                "--mxfp4-fuse-swiglu",
                "--mxfp4-pack-gate-up",
            ]
        )

    args = parse_args(
        _BASE
        + [
            "--task",
            "pretrain",
            "--mode",
            "mxfp4",
            "--mxfp4-fuse-swiglu",
        ]
    )
    assert args.mxfp4_fuse_swiglu is True


def test_mxfp4_pack_qkv_cli_validation():
    with pytest.raises(ValueError, match="requires --mode mxfp4"):
        parse_args(_BASE + ["--mxfp4-pack-qkv", "--mode", "bf16"])

    with pytest.raises(ValueError, match="requires --lora-rank 0"):
        parse_args(
            _BASE
            + [
                "--task",
                "sft",
                "--mode",
                "mxfp4",
                "--lora-rank",
                "8",
                "--mxfp4-pack-qkv",
            ]
        )

    args = parse_args(
        _BASE
        + [
            "--task",
            "pretrain",
            "--mode",
            "mxfp4",
            "--mxfp4-pack-qkv",
        ]
    )
    assert args.mxfp4_pack_qkv is True


def test_mxfp4_last_layer_bf16_projection_cli_validation():
    option = [
        "--mxfp4-last-layer-bf16-projections",
        "o_proj",
        "down_proj",
    ]
    with pytest.raises(ValueError, match="requires --mode mxfp4"):
        parse_args(_BASE + option + ["--mode", "bf16"])

    with pytest.raises(ValueError, match="requires --num-layers-at-end-in-bf16 0"):
        parse_args(_BASE + ["--task", "pretrain", "--mode", "mxfp4"] + option)

    args = parse_args(
        _BASE
        + [
            "--task",
            "pretrain",
            "--mode",
            "mxfp4",
            "--num-layers-at-end-in-bf16",
            "0",
        ]
        + option
    )
    assert args.mxfp4_last_layer_bf16_projections == ("o_proj", "down_proj")

    with pytest.raises(SystemExit):
        parse_args(
            _BASE
            + [
                "--task",
                "pretrain",
                "--mode",
                "mxfp4",
                "--num-layers-at-end-in-bf16",
                "0",
                "--mxfp4-last-layer-bf16-projections",
                "q_proj",
            ]
        )


def test_restore_last_layer_bf16_projections_is_exact_and_local():
    model = _tiny_qwen_layers(count=2)
    original_forwards = {}
    for layer_index, layer in enumerate(model.layers):
        for name, projection in (
            ("o_proj", layer.self_attn.o_proj),
            ("down_proj", layer.mlp.down_proj),
        ):
            original_forwards[(layer_index, name)] = projection.forward
            projection._original_forward = projection.forward
            projection.forward = MagicMock(name=f"quant_{layer_index}_{name}")
            projection._quant_enabled = True
            projection._quant_scaling_type = "mxfp4"
            projection._quant_manager = object()
            projection._lumen_quantize_activation = True
            projection._mxfp4_w_cache = object()
            projection.weight._mxfp4_w_cache = object()
            projection.weight._lumen_frozen = True

    restored = _restore_last_layer_bf16_projections(
        model, ("o_proj", "down_proj", "o_proj")
    )

    assert restored == ("self_attn.o_proj", "mlp.down_proj")
    for name, projection in (
        ("o_proj", model.layers[1].self_attn.o_proj),
        ("down_proj", model.layers[1].mlp.down_proj),
    ):
        assert projection.forward == original_forwards[(1, name)]
        assert projection._quant_enabled is False
        assert not hasattr(projection, "_original_forward")
        assert not hasattr(projection, "_quant_scaling_type")
        assert not hasattr(projection, "_quant_manager")
        assert not hasattr(projection, "_lumen_quantize_activation")
        assert not hasattr(projection, "_mxfp4_w_cache")
        assert not hasattr(projection.weight, "_mxfp4_w_cache")
        assert not hasattr(projection.weight, "_lumen_frozen")
    assert model.layers[0].self_attn.o_proj._quant_enabled is True
    assert model.layers[0].mlp.down_proj._quant_enabled is True


def test_restore_last_layer_bf16_projections_validates_atomically():
    model = _tiny_qwen_layers(count=1)
    o_proj = model.layers[0].self_attn.o_proj
    down_proj = model.layers[0].mlp.down_proj
    original_o_forward = o_proj.forward
    o_proj._original_forward = original_o_forward
    quant_o_forward = MagicMock(name="quant_o_proj")
    o_proj.forward = quant_o_forward
    o_proj._quant_enabled = True
    o_proj._quant_scaling_type = "mxfp4"

    with pytest.raises(RuntimeError, match="mlp.down_proj.*not MXFP4-patched"):
        _restore_last_layer_bf16_projections(model, ("o_proj", "down_proj"))

    assert o_proj.forward is quant_o_forward
    assert o_proj._quant_enabled is True
    assert o_proj._quant_scaling_type == "mxfp4"
    assert hasattr(o_proj, "_original_forward")
    assert getattr(down_proj, "_quant_enabled", False) is False


def _tiny_qwen_layers(count=1):
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=count,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
    )
    model = nn.Module()
    model.layers = nn.ModuleList(
        [Qwen3DecoderLayer(config, layer_idx=index) for index in range(count)]
    )
    return model


def _mark_mxfp4_mlp(mlp):
    for projection in (mlp.gate_proj, mlp.up_proj):
        projection._quant_enabled = True
        projection._quant_scaling_type = "mxfp4"
        projection._quant_manager = None
        projection._lumen_fp8_dtype = torch.float8_e4m3fn
        projection._lumen_block_size = 32
        projection._lumen_quantize_activation = True


def _tiny_qwen_qkv_layers(count=1, *, attention_bias=False):
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=count,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        attention_bias=attention_bias,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = nn.Module()
    model.layers = nn.ModuleList(
        [Qwen3DecoderLayer(config, layer_idx=index) for index in range(count)]
    )
    return model


def _mark_mxfp4_qkv(attention):
    manager = object()
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
    ):
        projection._quant_enabled = True
        projection._quant_scaling_type = "mxfp4"
        projection._quant_manager = manager
        projection._lumen_fp8_dtype = torch.float8_e4m3fn
        projection._lumen_block_size = 32
        projection._lumen_quantize_activation = True


def _qkv_inputs(attention, *, batch=2, sequence=4):
    hidden_states = torch.randn(batch, sequence, attention.q_proj.in_features)
    angles = torch.randn(batch, sequence, attention.head_dim)
    position_embeddings = (angles.cos(), angles.sin())
    attention_mask = torch.zeros(batch, 1, sequence, sequence)
    attention_mask[..., -1] = -10_000.0
    return hidden_states, position_embeddings, attention_mask


def _patch_exact_qkv(monkeypatch, call_log=None):
    import lumen.ops.quantize.linear as linear_mod
    import lumen.quantize as quantize_mod

    def _cache(_owner, q_weight, k_weight, v_weight, *_args, **_kwargs):
        rows = q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0]
        width = q_weight.shape[1]
        return (
            torch.empty((rows, width // 2), dtype=torch.uint8),
            torch.empty((rows // 32, width // 32), dtype=torch.uint8),
        )

    def _linear(x, q_weight, k_weight, v_weight, *_args, **_kwargs):
        if call_log is not None:
            call_log.append(tuple(x.shape))
        return torch.cat(
            tuple(
                nn.functional.linear(x, weight)
                for weight in (q_weight, k_weight, v_weight)
            ),
            dim=-1,
        )

    monkeypatch.setattr(quantize_mod, "_mxfp4_cached_weight_qkv", _cache)
    monkeypatch.setattr(linear_mod, "mxfp4_qkv_linear", _linear)


def test_mxfp4_qkv_patch_preserves_parameters_state_dict_and_attention(monkeypatch):
    model = _tiny_qwen_qkv_layers()
    attention = model.layers[0].self_attn
    _mark_mxfp4_qkv(attention)
    _patch_exact_qkv(monkeypatch)
    attention.eval()
    hidden_states, position_embeddings, attention_mask = _qkv_inputs(attention)
    original_forward = attention.forward
    reference_output, reference_weights = original_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
    )
    parameter_ids = {name: id(param) for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_ids = tuple(
        id(param) for group in optimizer.param_groups for param in group["params"]
    )
    state_before = {name: value.clone() for name, value in model.state_dict().items()}

    assert enable_mxfp4_qwen_qkv(model, strict=True) == 1
    output, weights = attention(hidden_states, position_embeddings, attention_mask)

    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
    assert {
        name: id(param) for name, param in model.named_parameters()
    } == parameter_ids
    assert (
        tuple(
            id(param) for group in optimizer.param_groups for param in group["params"]
        )
        == optimizer_ids
    )
    assert set(model.state_dict()) == set(state_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name])
    attention._mxfp4_w_cache = ((False, False), "stale-data", "stale-scale")
    attention._mxfp4_w_cache_version = (0, 0, 0)
    attention._mxfp4_w_cache_sources = tuple(
        id(projection.weight)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    attention._mxfp4_w_cache_metadata = ("stale",)
    model.load_state_dict(state_before, strict=True)
    for attr in (
        "_mxfp4_w_cache",
        "_mxfp4_w_cache_version",
        "_mxfp4_w_cache_sources",
        "_mxfp4_w_cache_metadata",
    ):
        assert not hasattr(attention, attr)
    assert not any("packed" in name for name, _ in model.named_parameters())
    assert not any("packed" in name for name, _ in model.named_buffers())

    assert disable_mxfp4_qwen_qkv(model) == 1
    assert attention.forward == original_forward


def test_mxfp4_qkv_preserves_parameter_gradient_hooks(monkeypatch):
    model = _tiny_qwen_qkv_layers()
    attention = model.layers[0].self_attn
    _mark_mxfp4_qkv(attention)
    _patch_exact_qkv(monkeypatch)
    seen = {}
    for name, projection in (
        ("q", attention.q_proj),
        ("k", attention.k_proj),
        ("v", attention.v_proj),
    ):
        projection.weight.register_hook(
            lambda grad, _name=name: seen.setdefault(_name, grad)
        )
    hidden_states, position_embeddings, attention_mask = _qkv_inputs(attention)
    hidden_states.requires_grad_(True)

    assert enable_mxfp4_qwen_qkv(model, strict=True) == 1
    output, _ = attention(hidden_states, position_embeddings, attention_mask)
    output.square().mean().backward()

    assert set(seen) == {"q", "k", "v"}
    for name, projection in (
        ("q", attention.q_proj),
        ("k", attention.k_proj),
        ("v", attention.v_proj),
    ):
        assert seen[name].shape == projection.weight.shape
        assert seen[name].is_contiguous()


def test_mxfp4_qkv_strict_rejects_child_hooks_atomically():
    model = _tiny_qwen_qkv_layers(2)
    for layer in model.layers:
        _mark_mxfp4_qkv(layer.self_attn)
    model.layers[1].self_attn.k_proj.register_forward_hook(lambda *_args: None)

    with pytest.raises(RuntimeError, match="enable on 1 of 2"):
        enable_mxfp4_qwen_qkv(model, strict=True)

    assert not any(
        hasattr(layer.self_attn, "_mxfp4_qkv_original_forward")
        for layer in model.layers
    )


def test_mxfp4_qkv_strict_rejects_attention_bias():
    model = _tiny_qwen_qkv_layers(attention_bias=True)
    _mark_mxfp4_qkv(model.layers[0].self_attn)

    with pytest.raises(RuntimeError, match="enable on 0 of 1"):
        enable_mxfp4_qwen_qkv(model, strict=True)


def test_mxfp4_qkv_runtime_falls_back_for_replaced_projection(monkeypatch):
    import lumen.ops.quantize.linear as linear_mod

    model = _tiny_qwen_qkv_layers()
    attention = model.layers[0].self_attn
    _mark_mxfp4_qkv(attention)
    assert enable_mxfp4_qwen_qkv(model, strict=True) == 1
    packed = MagicMock(side_effect=AssertionError("packed path must not run"))
    monkeypatch.setattr(linear_mod, "mxfp4_qkv_linear", packed)
    hidden_states, position_embeddings, attention_mask = _qkv_inputs(attention)
    attention.q_proj = nn.Sequential(attention.q_proj)

    output, weights = attention(hidden_states, position_embeddings, attention_mask)

    assert output.shape == hidden_states.shape
    assert weights.shape[-2:] == (hidden_states.shape[1], hidden_states.shape[1])
    packed.assert_not_called()


def test_mxfp4_qkv_propagates_packed_runtime_failures(monkeypatch):
    import lumen.ops.quantize.linear as linear_mod
    import lumen.quantize as quantize_mod

    model = _tiny_qwen_qkv_layers()
    attention = model.layers[0].self_attn
    _mark_mxfp4_qkv(attention)
    monkeypatch.setattr(
        quantize_mod,
        "_mxfp4_cached_weight_qkv",
        lambda _owner, q_weight, k_weight, v_weight, *_args, **_kwargs: (
            torch.empty(
                (
                    q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0],
                    q_weight.shape[1] // 2,
                ),
                dtype=torch.uint8,
            ),
            torch.empty(
                (
                    (q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0]) // 32,
                    q_weight.shape[1] // 32,
                ),
                dtype=torch.uint8,
            ),
        ),
    )
    packed = MagicMock(side_effect=RuntimeError("packed QKV kernel failed"))
    monkeypatch.setattr(linear_mod, "mxfp4_qkv_linear", packed)
    hidden_states, position_embeddings, attention_mask = _qkv_inputs(attention)

    assert enable_mxfp4_qwen_qkv(model, strict=True) == 1
    with pytest.raises(RuntimeError, match="packed QKV kernel failed"):
        attention(hidden_states, position_embeddings, attention_mask)
    packed.assert_called_once()


def test_mxfp4_qkv_preserves_dynamic_cache_updates(monkeypatch):
    model = _tiny_qwen_qkv_layers()
    attention = model.layers[0].self_attn
    _mark_mxfp4_qkv(attention)
    _patch_exact_qkv(monkeypatch)
    attention.eval()
    original_forward = attention.forward
    hidden_states, position_embeddings, _ = _qkv_inputs(attention, sequence=3)
    prefill_mask = torch.triu(torch.full((3, 3), -10_000.0), diagonal=1)
    prefill_mask = prefill_mask.view(1, 1, 3, 3).expand(2, 1, 3, 3)
    decode_states = torch.randn(2, 1, attention.q_proj.in_features)
    decode_angles = torch.randn(2, 1, attention.head_dim)
    decode_embeddings = (decode_angles.cos(), decode_angles.sin())
    decode_mask = torch.zeros(2, 1, 1, 4)
    reference_cache = DynamicCache(config=attention.config)
    packed_cache = DynamicCache(config=attention.config)

    reference_prefill = original_forward(
        hidden_states,
        position_embeddings,
        prefill_mask,
        reference_cache,
    )
    reference_decode = original_forward(
        decode_states,
        decode_embeddings,
        decode_mask,
        reference_cache,
    )
    assert enable_mxfp4_qwen_qkv(model, strict=True) == 1
    packed_prefill = attention(
        hidden_states,
        position_embeddings,
        prefill_mask,
        packed_cache,
    )
    packed_decode = attention(
        decode_states,
        decode_embeddings,
        decode_mask,
        packed_cache,
    )

    for reference, actual in zip(reference_prefill, packed_prefill):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for reference, actual in zip(reference_decode, packed_decode):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert reference_cache.get_seq_length(0) == packed_cache.get_seq_length(0) == 4
    torch.testing.assert_close(
        packed_cache.layers[0].keys,
        reference_cache.layers[0].keys,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        packed_cache.layers[0].values,
        reference_cache.layers[0].values,
        rtol=0,
        atol=0,
    )


def test_mxfp4_qkv_checkpoint_recompute_preserves_gradients(monkeypatch):
    model = _tiny_qwen_qkv_layers()
    attention = model.layers[0].self_attn
    _mark_mxfp4_qkv(attention)
    calls = []
    _patch_exact_qkv(monkeypatch, calls)
    attention.eval()
    hidden_states, position_embeddings, attention_mask = _qkv_inputs(attention)
    assert enable_mxfp4_qwen_qkv(model, strict=True) == 1

    direct_input = hidden_states.detach().clone().requires_grad_(True)
    direct_output, _ = attention(
        direct_input,
        position_embeddings,
        attention_mask,
    )
    direct_output.square().mean().backward()
    direct_input_grad = direct_input.grad.clone()
    direct_weight_grads = {
        name: projection.weight.grad.clone()
        for name, projection in (
            ("q", attention.q_proj),
            ("k", attention.k_proj),
            ("v", attention.v_proj),
        )
    }
    model.zero_grad(set_to_none=True)

    checkpoint_input = hidden_states.detach().clone().requires_grad_(True)

    def _forward(value):
        return attention(value, position_embeddings, attention_mask)[0]

    checkpoint_output = torch.utils.checkpoint.checkpoint(
        _forward,
        checkpoint_input,
        use_reentrant=False,
    )
    checkpoint_output.square().mean().backward()

    torch.testing.assert_close(checkpoint_output, direct_output, rtol=0, atol=0)
    torch.testing.assert_close(
        checkpoint_input.grad,
        direct_input_grad,
        rtol=0,
        atol=0,
    )
    for name, projection in (
        ("q", attention.q_proj),
        ("k", attention.k_proj),
        ("v", attention.v_proj),
    ):
        torch.testing.assert_close(
            projection.weight.grad,
            direct_weight_grads[name],
            rtol=0,
            atol=0,
        )
    assert len(calls) == 3


def test_mxfp4_gate_up_patch_preserves_parameters_state_dict_and_forward(monkeypatch):
    import lumen.ops.fused_swiglu as swiglu_mod
    import lumen.ops.quantize.linear as linear_mod
    import lumen.quantize as quantize_mod

    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    parameter_ids = {name: id(param) for name, param in model.named_parameters()}
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    original_forward = mlp.forward

    monkeypatch.setattr(
        quantize_mod,
        "_mxfp4_cached_weight_pair",
        lambda _owner, gate, up, *_args, **_kwargs: (
            torch.empty((gate.shape[0] + up.shape[0], gate.shape[1] // 2), dtype=torch.uint8),
            torch.empty(((gate.shape[0] + up.shape[0]) // 32, gate.shape[1] // 32), dtype=torch.uint8),
        ),
    )

    def _reference_pair(x, gate, up, *_args, **_kwargs):
        return torch.cat((nn.functional.linear(x, gate), nn.functional.linear(x, up)), dim=-1)

    monkeypatch.setattr(linear_mod, "mxfp4_gate_up_linear", _reference_pair)
    monkeypatch.setattr(
        swiglu_mod,
        "packed_swiglu",
        lambda packed: mlp.act_fn(packed[..., :64]) * packed[..., 64:],
    )
    x = torch.randn(2, 3, 32)
    reference = original_forward(x)

    assert enable_mxfp4_qwen_gate_up(model) == 1
    torch.testing.assert_close(mlp(x), reference)
    assert {name: id(param) for name, param in model.named_parameters()} == parameter_ids
    assert set(model.state_dict()) == set(state_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name])
    assert not any("packed" in name for name, _ in model.named_parameters())

    quantize_mod.disable(model)
    assert mlp.forward == original_forward


def test_mxfp4_split_swiglu_preserves_modules_hooks_and_state(monkeypatch):
    import lumen.ops.fused_swiglu as swiglu_mod

    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    monkeypatch.setattr(swiglu_mod, "_probe_aiter_swiglu_split", lambda: True)
    monkeypatch.setattr(
        swiglu_mod,
        "split_swiglu",
        lambda gate, up: mlp.act_fn(gate) * up,
    )

    parameter_ids = {name: id(param) for name, param in model.named_parameters()}
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    original_forward = mlp.forward
    x = torch.randn(2, 3, 32)
    reference = original_forward(x)

    hook_calls = {"gate": 0, "up": 0, "down": 0}
    handles = [
        mlp.gate_proj.register_forward_hook(
            lambda *_args: hook_calls.__setitem__("gate", hook_calls["gate"] + 1)
        ),
        mlp.up_proj.register_forward_hook(
            lambda *_args: hook_calls.__setitem__("up", hook_calls["up"] + 1)
        ),
        mlp.down_proj.register_forward_hook(
            lambda *_args: hook_calls.__setitem__("down", hook_calls["down"] + 1)
        ),
    ]
    try:
        assert enable_mxfp4_qwen_swiglu(model, strict=True) == 1
        torch.testing.assert_close(mlp(x), reference)
    finally:
        for handle in handles:
            handle.remove()

    assert hook_calls == {"gate": 1, "up": 1, "down": 1}
    assert {name: id(param) for name, param in model.named_parameters()} == parameter_ids
    assert set(model.state_dict()) == set(state_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name])

    assert disable_mxfp4_qwen_swiglu(model) == 1
    assert mlp.forward == original_forward


def test_mxfp4_split_swiglu_patches_quantized_prefix_only(monkeypatch):
    import lumen.ops.fused_swiglu as swiglu_mod

    model = _tiny_qwen_layers(4)
    for layer in model.layers[:3]:
        _mark_mxfp4_mlp(layer.mlp)
    monkeypatch.setattr(swiglu_mod, "_probe_aiter_swiglu_split", lambda: True)

    assert enable_mxfp4_qwen_swiglu(model, strict=True) == 3
    assert sum(
        hasattr(layer.mlp, "_mxfp4_swiglu_original_forward")
        for layer in model.layers
    ) == 3


def test_mxfp4_gate_up_patches_quantized_prefix_only():
    model = _tiny_qwen_layers(36)
    for layer in model.layers[:31]:
        _mark_mxfp4_mlp(layer.mlp)

    assert enable_mxfp4_qwen_gate_up(model) == 31
    assert sum(
        hasattr(layer.mlp, "_mxfp4_gate_up_original_forward")
        for layer in model.layers
    ) == 31


def test_mxfp4_gate_up_runtime_falls_back_for_wrapped_child(monkeypatch):
    import lumen.ops.quantize.linear as linear_mod

    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    assert enable_mxfp4_qwen_gate_up(model) == 1
    packed = MagicMock(side_effect=AssertionError("packed path must not run"))
    monkeypatch.setattr(linear_mod, "mxfp4_gate_up_linear", packed)

    mlp.gate_proj = nn.Sequential(mlp.gate_proj)
    result = mlp(torch.randn(2, 3, 32))

    assert result.shape == (2, 3, 32)
    packed.assert_not_called()


def test_mxfp4_gate_up_rejects_projection_hooks_and_strict_partial_enable():
    model = _tiny_qwen_layers(2)
    for layer in model.layers:
        _mark_mxfp4_mlp(layer.mlp)
    hook_calls = []
    model.layers[1].mlp.gate_proj.register_forward_hook(
        lambda *_args: hook_calls.append(True)
    )

    with pytest.raises(RuntimeError, match="enable on 1 of 2"):
        enable_mxfp4_qwen_gate_up(model, strict=True)

    assert not hasattr(
        model.layers[0].mlp, "_mxfp4_gate_up_original_forward"
    )
    model.layers[1].mlp(torch.randn(2, 3, 32))
    assert hook_calls == [True]


def test_mxfp4_gate_up_strict_rejects_weight_only_quantization():
    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    mlp.gate_proj._lumen_quantize_activation = False
    mlp.up_proj._lumen_quantize_activation = False

    with pytest.raises(RuntimeError, match="enable on 0 of 1"):
        enable_mxfp4_qwen_gate_up(model, strict=True)
    assert not hasattr(mlp, "_mxfp4_gate_up_original_forward")


def test_mxfp4_gate_up_strict_rejects_non_silu_activation():
    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    mlp.act_fn = nn.GELU()

    with pytest.raises(RuntimeError, match="enable on 0 of 1"):
        enable_mxfp4_qwen_gate_up(model, strict=True)
    assert not hasattr(mlp, "_mxfp4_gate_up_original_forward")


def test_mxfp4_gate_up_strict_requires_split_swiglu_capability(monkeypatch):
    import lumen.ops.fused_swiglu as swiglu_mod

    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    monkeypatch.setattr(swiglu_mod, "_probe_aiter_swiglu_split", lambda: False)

    with pytest.raises(RuntimeError, match="enable on 0 of 1"):
        enable_mxfp4_qwen_gate_up(model, strict=True)
    assert not hasattr(mlp, "_mxfp4_gate_up_original_forward")


def test_mxfp4_gate_up_propagates_packed_runtime_failures(monkeypatch):
    import lumen.ops.quantize.linear as linear_mod
    import lumen.quantize as quantize_mod

    model = _tiny_qwen_layers()
    mlp = model.layers[0].mlp
    _mark_mxfp4_mlp(mlp)
    monkeypatch.setattr(
        quantize_mod,
        "_mxfp4_cached_weight_pair",
        lambda _owner, gate, up, *_args, **_kwargs: (
            torch.empty(
                (gate.shape[0] + up.shape[0], gate.shape[1] // 2),
                dtype=torch.uint8,
            ),
            torch.empty(
                ((gate.shape[0] + up.shape[0]) // 32, gate.shape[1] // 32),
                dtype=torch.uint8,
            ),
        ),
    )
    packed = MagicMock(side_effect=RuntimeError("packed kernel failed"))
    monkeypatch.setattr(linear_mod, "mxfp4_gate_up_linear", packed)

    assert enable_mxfp4_qwen_gate_up(model, strict=True) == 1
    with pytest.raises(RuntimeError, match="packed kernel failed"):
        mlp(torch.randn(2, 3, 32))
    packed.assert_called_once()


def test_sft_defaults_remain_lora_and_no_bf16_tail():
    args = parse_args(_BASE)

    assert args.task == "sft"
    assert args.lora_rank == 16
    assert args.first_last_layers_bf16 is False


def test_pretrain_rejects_lora():
    with pytest.raises(ValueError, match="requires --lora-rank 0"):
        parse_args(_BASE + ["--task", "pretrain", "--lora-rank", "16"])


def test_init_from_scratch_is_pretrain_only():
    with pytest.raises(ValueError, match="only valid with --task pretrain"):
        parse_args(_BASE + ["--init-from-scratch"])


def test_fsdp2_accumulation_sync_recurses():
    model = MagicMock()

    _set_fsdp2_gradient_sync(model, False)
    _set_fsdp2_gradient_sync(model, True)

    assert model.set_requires_gradient_sync.call_args_list[0].args == (False,)
    assert model.set_requires_gradient_sync.call_args_list[0].kwargs == {
        "recurse": True
    }
    assert model.set_requires_gradient_sync.call_args_list[1].args == (True,)


def test_fsdp2_reshard_after_backward_recurses():
    model = MagicMock()

    _set_fsdp2_reshard_after_backward(model, False)

    model.set_reshard_after_backward.assert_called_once_with(False, recurse=True)


def test_fsdp2_reshard_after_forward_skips_the_root_module():
    root = MagicMock()
    child_a, child_b = MagicMock(), MagicMock()
    plain = SimpleNamespace()  # a non-FSDP submodule must be ignored
    root.modules.return_value = [root, child_a, plain, child_b]

    _set_fsdp2_reshard_after_forward(root, False)

    root.set_reshard_after_forward.assert_not_called()
    child_a.set_reshard_after_forward.assert_called_once_with(False, recurse=False)
    child_b.set_reshard_after_forward.assert_called_once_with(False, recurse=False)


def test_fsdp2_reshard_after_forward_requires_a_wrapped_submodule():
    root = MagicMock()
    root.modules.return_value = [root, SimpleNamespace()]

    with pytest.raises(RuntimeError, match="set_reshard_after_forward"):
        _set_fsdp2_reshard_after_forward(root, False)


def test_fsdp2_root_reshard_after_forward_only_updates_the_root():
    root = MagicMock()
    child = MagicMock()
    root.modules.return_value = [root, child]

    _set_fsdp2_root_reshard_after_forward(root, False)

    root.set_reshard_after_forward.assert_called_once_with(False, recurse=False)
    child.set_reshard_after_forward.assert_not_called()


def test_fsdp2_root_reshard_after_forward_requires_the_root_setter():
    root = SimpleNamespace()

    with pytest.raises(RuntimeError, match="FSDP2 root"):
        _set_fsdp2_root_reshard_after_forward(root, False)


def test_fsdp2_root_reshard_after_forward_propagates_setter_failures():
    root = MagicMock()
    root.set_reshard_after_forward.side_effect = RuntimeError("root policy failed")

    with pytest.raises(RuntimeError, match="root policy failed"):
        _set_fsdp2_root_reshard_after_forward(root, False)


def test_step_profiler_marks_each_profiled_training_step(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_PROF_START", "2")
    monkeypatch.setenv("LUMEN_PROF_END", "3")
    monkeypatch.setenv("LUMEN_PROF_OUTPUT", str(tmp_path / "profile.txt"))
    profile_context = MagicMock()
    profile_context.key_averages.return_value.table.return_value = "table"
    step_contexts = [MagicMock(), MagicMock()]

    with patch(
        "examples.qwen3.train_qwen3_fsdp.torch.profiler.profile",
        return_value=profile_context,
    ) as profile, patch(
        "examples.qwen3.train_qwen3_fsdp.torch.profiler.record_function",
        side_effect=step_contexts,
    ) as record_function:
        profiler = StepProfiler(enabled=True)
        profiler.step_begin(1)
        profiler.step_end(1)
        profiler.step_begin(2)
        profiler.step_end(2)
        profiler.step_begin(3)
        profiler.step_end(3)

    profile.assert_called_once()
    assert [call.args for call in record_function.call_args_list] == [
        ("LUMEN_TRAIN_STEP#2",),
        ("LUMEN_TRAIN_STEP#3",),
    ]
    for context in step_contexts:
        context.__enter__.assert_called_once_with()
        context.__exit__.assert_called_once_with(None, None, None)


def _accumulation_calls(model, name):
    return [(c.args, c.kwargs) for c in getattr(model, name).call_args_list]


def test_fsdp2_retained_params_are_freed_only_on_the_final_microbatch():
    model = MagicMock()
    child = MagicMock()
    model.modules.return_value = [model, child]

    for micro in range(3):
        _set_fsdp2_accumulation_state(
            model,
            final_micro=micro == 2,
            retain_params=True,
            reshard_after_forward=True,
        )

    assert _accumulation_calls(model, "set_requires_gradient_sync") == [
        ((False,), {"recurse": True}),
        ((False,), {"recurse": True}),
        ((True,), {"recurse": True}),
    ]
    assert _accumulation_calls(model, "set_reshard_after_backward") == [
        ((False,), {"recurse": True}),
        ((False,), {"recurse": True}),
        ((True,), {"recurse": True}),
    ]
    # The wrapped policy is restored on the final micro-batch, not assumed.
    assert _accumulation_calls(child, "set_reshard_after_forward") == [
        ((False,), {"recurse": False}),
        ((False,), {"recurse": False}),
        ((True,), {"recurse": False}),
    ]
    model.set_reshard_after_forward.assert_not_called()


def test_fsdp2_root_retention_is_not_reversed_on_the_final_microbatch():
    model = MagicMock()
    child = MagicMock()
    model.modules.return_value = [model, child]

    _set_fsdp2_root_reshard_after_forward(model, False)
    for micro in range(3):
        _set_fsdp2_accumulation_state(
            model,
            final_micro=micro == 2,
            retain_params=True,
            reshard_after_forward=True,
        )

    model.set_reshard_after_forward.assert_called_once_with(False, recurse=False)
    assert _accumulation_calls(model, "set_reshard_after_backward") == [
        ((False,), {"recurse": True}),
        ((False,), {"recurse": True}),
        ((True,), {"recurse": True}),
    ]
    assert _accumulation_calls(child, "set_reshard_after_forward") == [
        ((False,), {"recurse": False}),
        ((False,), {"recurse": False}),
        ((True,), {"recurse": False}),
    ]


def test_fsdp2_retention_restores_shard_grad_op_policy():
    model = MagicMock()
    child = MagicMock()
    model.modules.return_value = [model, child]

    _set_fsdp2_accumulation_state(
        model, final_micro=True, retain_params=True, reshard_after_forward=False
    )

    # shard_grad_op never reshards after forward; retention must not turn that
    # into full_shard behaviour on the final micro-batch.
    child.set_reshard_after_forward.assert_called_once_with(False, recurse=False)


def test_fsdp2_accumulation_leaves_lifetime_untouched_by_default():
    model = MagicMock()
    child = MagicMock()
    model.modules.return_value = [model, child]

    _set_fsdp2_accumulation_state(
        model, final_micro=False, retain_params=False, reshard_after_forward=True
    )

    model.set_requires_gradient_sync.assert_called_once_with(False, recurse=True)
    model.set_reshard_after_backward.assert_not_called()
    model.set_reshard_after_forward.assert_not_called()
    child.set_reshard_after_forward.assert_not_called()


def test_fsdp2_retain_accumulated_params_requires_fsdp2():
    with pytest.raises(ValueError, match="requires --fsdp-version 2"):
        parse_args(_BASE + ["--fsdp-retain-accumulated-params"])

    args = parse_args(
        _BASE + ["--fsdp-version", "2", "--fsdp-retain-accumulated-params"]
    )
    assert args.fsdp_retain_accumulated_params is True


def test_fsdp2_root_retention_defaults_off():
    args = parse_args(_BASE + ["--fsdp-version", "2"])

    assert args.fsdp_retain_root_params is False


def test_fsdp2_root_retention_accepts_the_supported_accumulation_path():
    args = parse_args(
        _BASE
        + [
            "--fsdp-version",
            "2",
            "--gradient-accumulation-steps",
            "2",
            "--fsdp-retain-accumulated-params",
            "--fsdp-retain-root-params",
        ]
    )

    assert args.fsdp_retain_root_params is True


def test_fsdp2_root_retention_requires_fsdp2():
    with pytest.raises(ValueError, match="requires --fsdp-version 2"):
        parse_args(_BASE + ["--fsdp-retain-root-params"])


def test_fsdp2_root_retention_requires_accumulated_param_retention():
    with pytest.raises(ValueError, match="--fsdp-retain-accumulated-params"):
        parse_args(
            _BASE
            + [
                "--fsdp-version",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--fsdp-retain-root-params",
            ]
        )


def test_fsdp2_root_retention_requires_gradient_accumulation():
    with pytest.raises(ValueError, match="--gradient-accumulation-steps > 1"):
        parse_args(
            _BASE
            + [
                "--fsdp-version",
                "2",
                "--fsdp-retain-accumulated-params",
                "--fsdp-retain-root-params",
            ]
        )


def test_fsdp2_root_retention_requires_full_shard():
    with pytest.raises(ValueError, match="--sharding full_shard"):
        parse_args(
            _BASE
            + [
                "--fsdp-version",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--fsdp-retain-accumulated-params",
                "--fsdp-retain-root-params",
                "--sharding",
                "shard_grad_op",
            ]
        )


def test_fsdp_reduce_dtype_defaults_to_auto():
    args = parse_args(_BASE + ["--fsdp-version", "2", "--mode", "bf16"])

    assert args.fsdp_reduce_dtype == "auto"


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_fsdp2_reduce_dtype_accepts_explicit_override(dtype):
    args = parse_args(_BASE + ["--fsdp-version", "2", "--fsdp-reduce-dtype", dtype])

    assert args.fsdp_reduce_dtype == dtype


def test_fsdp_reduce_dtype_override_requires_fsdp2():
    with pytest.raises(ValueError, match="requires --fsdp-version 2"):
        parse_args(_BASE + ["--fsdp-reduce-dtype", "fp32"])


@pytest.mark.parametrize(
    ("num_layers", "requested", "expected"),
    [
        (36, 36, list(range(36))),
        (36, 40, list(range(36))),
        (36, 0, []),
        (36, -1, []),
        # Spread across the depth rather than a prefix: every layer costs the
        # same recompute, so a prefix would only bias where memory is held.
        (36, 9, [0, 4, 8, 12, 16, 20, 24, 28, 32]),
        (36, 1, [0]),
    ],
)
def test_select_recompute_layers_spreads_across_depth(num_layers, requested, expected):
    assert _select_recompute_layers(num_layers, requested) == expected


class _CountingLayer(_GRAD_CKPT_LAYER):
    """A layer that records how many times its forward actually runs."""

    def __init__(self, dim, counter):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)
        self.counter = counter

    def forward(self, x):
        self.counter["forwards"] += 1
        return torch.tanh(self.lin(x))


class _CountingStack(torch.nn.Module):
    def __init__(self, dim, depth, counter):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _CountingLayer(dim, counter) for _ in range(depth)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run_counting_stack(requested, depth=4, dim=8):
    """Train-mode forward/backward with `requested` layers recomputed."""
    torch.manual_seed(0)
    counter = {"forwards": 0}
    model = _CountingStack(dim, depth, counter)
    model.train()
    for layer in model.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
    if requested is not None:
        _apply_selective_grad_checkpointing(model, requested)

    x = torch.randn(2, dim, requires_grad=True)
    model(x).square().sum().backward()
    grads = [layer.lin.weight.grad.clone() for layer in model.layers]
    return counter["forwards"], grads


@pytest.mark.parametrize("requested", [0, 2, 4])
def test_selective_recompute_runs_exactly_the_selected_layers_twice(requested):
    depth = 4
    forwards, grads = _run_counting_stack(requested, depth=depth)

    # One forward per layer, plus one extra for each recomputed layer. This is
    # what fails if the per-layer flag is set but never read, or if the whole
    # stack keeps recomputing.
    assert forwards == depth + requested

    _, reference = _run_counting_stack(0, depth=depth)
    for grad, expected in zip(grads, reference):
        torch.testing.assert_close(grad, expected)


def test_selective_recompute_rejects_layers_without_the_flag():
    model = SimpleNamespace()
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    model.modules = lambda: [SimpleNamespace(layers=layers)]

    with pytest.raises(RuntimeError, match="gradient_checkpointing"):
        _apply_selective_grad_checkpointing(model, 1)


def test_selective_recompute_requires_a_transformer_layer_list():
    model = SimpleNamespace(modules=lambda: [SimpleNamespace()])

    with pytest.raises(RuntimeError, match="ModuleList"):
        _apply_selective_grad_checkpointing(model, 1)


def test_grad_checkpoint_layers_conflicts_with_disabled_checkpointing():
    with pytest.raises(ValueError, match="--no-grad-checkpointing"):
        parse_args(_BASE + ["--no-grad-checkpointing", "--grad-checkpoint-layers", "9"])

    with pytest.raises(ValueError, match=">= 0"):
        parse_args(_BASE + ["--grad-checkpoint-layers", "-2"])

    assert parse_args(_BASE).grad_checkpoint_layers is None
    assert (
        parse_args(_BASE + ["--grad-checkpoint-layers", "9"]).grad_checkpoint_layers
        == 9
    )


def test_pretrain_loss_does_not_shift_dataset_labels_twice():
    labels = torch.tensor([[1, 2, 3]])
    logits = torch.full((1, 3, 4), -20.0)
    logits.scatter_(-1, labels.unsqueeze(-1), 20.0)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": labels,
    }

    loss = _pretrain_loss(model, batch, "cpu")

    assert loss.item() < 1e-6
    model.assert_called_once_with(input_ids=batch["input_ids"], use_cache=False)


def test_pretrain_loss_upcasts_bf16_logits_for_cross_entropy():
    logits = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = {
        "input_ids": torch.randint(0, 8, (2, 3)),
        "labels": torch.randint(0, 8, (2, 3)),
    }

    loss = _pretrain_loss(model, batch, "cpu")

    assert loss.dtype == torch.float32


@patch("examples.qwen3.train_qwen3_fsdp.torch.manual_seed")
@patch("examples.qwen3.train_qwen3_fsdp.random.seed")
def test_seed_training_rngs_uses_shared_model_and_rank_local_rounding_seeds(
    random_seed, torch_seed
):
    _seed_training_rngs(1234, global_rank=7)

    torch_seed.assert_called_once_with(1234)
    random_seed.assert_called_once_with(1241)


def _pretrain_shuffle_order(seed, global_rank):
    loader = torch.utils.data.DataLoader(
        torch.arange(32),
        batch_size=4,
        shuffle=True,
        generator=_pretrain_shuffle_generator(seed, global_rank),
        num_workers=0,
    )
    return torch.cat(list(loader))


def test_pretrain_shuffle_is_independent_of_global_torch_rng():
    torch.manual_seed(11)
    torch.rand(257)
    first = _pretrain_shuffle_order(seed=1234, global_rank=3)

    torch.manual_seed(999)
    torch.rand(19)
    second = _pretrain_shuffle_order(seed=1234, global_rank=3)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, _pretrain_shuffle_order(seed=1234, global_rank=4))


def test_paired_evidence_switch_is_opt_in(monkeypatch):
    monkeypatch.delenv("LUMEN_PAIRED_RUN_EVIDENCE", raising=False)
    assert _paired_evidence_enabled() is False

    monkeypatch.setenv("LUMEN_PAIRED_RUN_EVIDENCE", "1")
    assert _paired_evidence_enabled() is True

    monkeypatch.setenv("LUMEN_PAIRED_RUN_EVIDENCE", "true")
    assert _paired_evidence_enabled() is False


def test_model_init_sha256_tracks_parameter_values_not_global_rng():
    torch.manual_seed(7)
    reference = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(3, 2, dtype=torch.bfloat16),
    )
    same = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(3, 2, dtype=torch.bfloat16),
    )
    same.load_state_dict(reference.state_dict())

    expected = _model_init_sha256(reference)
    torch.manual_seed(999)
    torch.rand(31)
    assert _model_init_sha256(same) == expected

    with torch.no_grad():
        same[0].weight.view(-1)[0] += 1
    assert _model_init_sha256(same) != expected


def _rolling_batch_digest(batches):
    digest = hashlib.sha256()
    for microbatch, batch in enumerate(batches):
        _update_pretrain_batch_sha256(digest, batch, microbatch)
    return digest.hexdigest()


def test_pretrain_batch_sha256_is_deterministic_and_order_sensitive():
    first = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
    }
    second = {
        "input_ids": torch.tensor([[8, 9, 10], [11, 12, 13]]),
        "labels": torch.tensor([[9, 10, 11], [12, 13, 14]]),
    }

    expected = _rolling_batch_digest([first, second])
    assert _rolling_batch_digest([first, second]) == expected
    assert _rolling_batch_digest([second, first]) != expected

    changed = {key: value.clone() for key, value in second.items()}
    changed["labels"][0, 0] += 1
    assert _rolling_batch_digest([first, changed]) != expected


def test_validation_evidence_line_records_rank_count_and_digest():
    digest = hashlib.sha256(b"validation")

    assert _validation_evidence_line(3, 16, digest) == (
        "VALIDATION_EVIDENCE rank=3 batches=16 "
        f"input_ids_labels_sha256={digest.hexdigest()}"
    )


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@_CUDA
def test_fused_cross_entropy_matches_the_fp32_reference():
    """The fused loss and its logit gradient must track the FP32 reference.

    Both arms read the same BF16 logits; the kernel accumulates the softmax in
    FP32 internally, so this is a rounding difference, not a precision change.
    """
    torch.manual_seed(0)
    batch = {
        "input_ids": torch.randint(0, 4096, (2, 128), device="cuda"),
        "labels": torch.randint(0, 4096, (2, 128), device="cuda"),
    }
    base = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16)

    def run(fused):
        leaf = base.detach().clone().requires_grad_(True)
        # Non-leaf logits, as the real lm_head produces: the fused path writes
        # its gradient into that buffer in place.
        model = MagicMock(return_value=SimpleNamespace(logits=leaf * 1.0))
        loss = _pretrain_loss(model, batch, "cuda", fused=fused)
        loss.backward()
        return loss.detach(), leaf.grad

    try:
        fused_loss, fused_grad = run(True)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"AITER cross-entropy unavailable: {e}")
    ref_loss, ref_grad = run(False)

    torch.testing.assert_close(fused_loss.float(), ref_loss, rtol=1e-3, atol=1e-3)
    err = (fused_grad.float() - ref_grad.float()).pow(2).sum()
    snr = 10 * torch.log10(ref_grad.float().pow(2).sum() / err.clamp(min=1e-30))
    assert snr > 30, f"logit-gradient SNR {snr:.1f} dB"


def test_fused_cross_entropy_rejects_the_masked_sft_loss():
    with pytest.raises(ValueError, match="requires --task pretrain"):
        parse_args(_BASE + ["--task", "sft", "--fused-cross-entropy"])

    pretrain = _BASE + ["--task", "pretrain", "--lora-rank", "0"]
    assert parse_args(pretrain + ["--fused-cross-entropy"]).fused_cross_entropy is True
    assert parse_args(pretrain).fused_cross_entropy is False


def test_eval_batches_must_be_positive():
    with pytest.raises(ValueError, match="--eval-batches must be >= 1"):
        parse_args(_BASE + ["--eval-batches", "0"])


def test_validation_batch_count_rejects_empty_local_loader():
    with pytest.raises(ValueError, match="no full micro-batch"):
        _validation_batch_count(0, 10, "cpu")


def test_validation_batch_count_uses_global_minimum():
    def set_remote_min(count, op):
        assert op == torch.distributed.ReduceOp.MIN
        count.fill_(3)

    with (
        patch("examples.qwen3.train_qwen3_fsdp.dist.is_initialized", return_value=True),
        patch(
            "examples.qwen3.train_qwen3_fsdp.dist.all_reduce",
            side_effect=set_remote_min,
        ),
    ):
        count = _validation_batch_count(8, 10, "cpu")

    assert count == 3


def test_pretrain_dataset_import_does_not_require_megatron():
    from lumen.models.llama31.dataset import PretrainTextDataset

    assert PretrainTextDataset.__name__ == "PretrainTextDataset"


def test_pretrain_dataset_tokenization_is_sharded_by_input_line(tmp_path):
    from lumen.models.llama31.dataset import PretrainTextDataset

    data_path = tmp_path / "train.txt"
    data_path.write_text("1\n2\n3\n4\n", encoding="utf-8")
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        encode=lambda text, add_special_tokens=False: [int(text)],
    )

    rank0 = PretrainTextDataset(
        str(data_path),
        seq_length=1,
        tokenizer=tokenizer,
        is_hf_tokenizer=True,
        rank=0,
        world_size=2,
    )
    rank1 = PretrainTextDataset(
        str(data_path),
        seq_length=1,
        tokenizer=tokenizer,
        is_hf_tokenizer=True,
        rank=1,
        world_size=2,
    )

    assert [rank0[i]["input_ids"].item() for i in range(len(rank0))] == [1, 3]
    assert [rank1[i]["input_ids"].item() for i in range(len(rank1))] == [2, 4]


def test_pretrain_dataset_stops_reading_once_sample_budget_is_met(tmp_path):
    from lumen.models.llama31.dataset import PretrainTextDataset

    data_path = tmp_path / "train.txt"
    data_path.write_text("".join(f"{i}\n" for i in range(1, 1001)), encoding="utf-8")
    seen = []

    def encode(text, add_special_tokens=False):
        seen.append(text)
        return [int(text)]

    tokenizer = SimpleNamespace(eos_token_id=0, encode=encode)

    ds = PretrainTextDataset(
        str(data_path),
        seq_length=1,
        tokenizer=tokenizer,
        is_hf_tokenizer=True,
        max_samples=2,
    )

    assert len(ds) == 2
    # Two samples need 4 tokens; the scan must not walk the whole file.
    assert len(seen) <= 4


def test_train_samples_is_pretrain_only():
    with pytest.raises(ValueError, match="--train-samples"):
        parse_args(_BASE + ["--task", "sft", "--train-samples", "8"])

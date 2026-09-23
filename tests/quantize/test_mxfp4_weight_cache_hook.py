###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""MXFP4 weight cache invalidation across optimizer and load boundaries.

The cache holds each layer's quantized weight so that gradient-accumulation
micro-batches reuse it. A hook that fails to invalidate it after an optimizer
step or checkpoint load leaves the run using stale weights without raising.
"""

import pytest
import torch
import torch.nn as nn

from lumen.quantize import register_mxfp4_weight_optimizer_hooks


class _MegatronStyleOptimizer:
    """Stands in for Megatron's ChainedOptimizer / DistributedOptimizer.

    Those wrap the torch optimizers rather than subclassing them, so they
    expose ``step()`` but not ``register_step_post_hook``.
    """

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return "step-result"


def _model_with_cache():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    for layer in model:
        layer._mxfp4_w_cache = ((False, False), "fp4", "scale")
    return model


def _cached(model):
    return [hasattr(layer, "_mxfp4_w_cache") for layer in model]


class TestMXFP4WeightCacheHook:
    @staticmethod
    def _patch_pair_builder(monkeypatch):
        from lumen.ops import dispatch as dispatch_mod
        from lumen.ops.quantize import linear as linear_mod
        from lumen.ops.quantize import ops as ops_mod

        calls = []

        def _quantize(weight, **_kwargs):
            calls.append(weight)
            fill = 1 if len(calls) % 2 else 2
            return (
                torch.full(
                    (weight.shape[0], weight.shape[1] // 2),
                    fill,
                    dtype=torch.uint8,
                ),
                torch.full(
                    (weight.shape[0] // 32, weight.shape[1] // 32),
                    fill,
                    dtype=torch.uint8,
                ),
            )

        monkeypatch.setattr(
            dispatch_mod,
            "_probe_aiter_triton_quant_mxfp4_2way",
            lambda: False,
        )
        monkeypatch.setattr(ops_mod, "convert_to_mxfp4_2d", _quantize)
        monkeypatch.setattr(
            ops_mod,
            "transpose_packed_fp4",
            lambda data, **_kwargs: data.t().contiguous(),
        )
        monkeypatch.setattr(
            linear_mod, "_mxfp4_can_fuse_b_shuffle", lambda *_args: False
        )
        monkeypatch.setattr(
            linear_mod, "_mxfp4_can_fuse_scale_swizzle", lambda *_args: False
        )
        return calls

    @staticmethod
    def _patch_qkv_builder(monkeypatch):
        from lumen.ops.quantize import linear as linear_mod
        from lumen.ops.quantize import ops as ops_mod

        calls = []

        def _quantize(weight, **_kwargs):
            calls.append(weight)
            fill = len(calls)
            return (
                torch.full(
                    (weight.shape[0], weight.shape[1] // 2),
                    fill,
                    dtype=torch.uint8,
                ),
                torch.full(
                    (weight.shape[0] // 32, weight.shape[1] // 32),
                    fill,
                    dtype=torch.uint8,
                ),
            )

        monkeypatch.setattr(ops_mod, "convert_to_mxfp4_2d", _quantize)
        monkeypatch.setattr(
            ops_mod,
            "transpose_packed_fp4",
            lambda data, **_kwargs: data.t().contiguous(),
        )
        monkeypatch.setattr(
            linear_mod, "_mxfp4_can_fuse_b_shuffle", lambda *_args: False
        )
        monkeypatch.setattr(
            linear_mod, "_mxfp4_can_fuse_scale_swizzle", lambda *_args: False
        )
        return calls

    def test_qkv_cache_supports_unequal_rows_and_tracks_full_metadata(
        self, monkeypatch
    ):
        from lumen.quantize import _mxfp4_cached_weight_qkv

        calls = self._patch_qkv_builder(monkeypatch)
        owner = nn.Module()
        q = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        k = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))
        v = nn.Parameter(torch.randn(96, 64, dtype=torch.bfloat16))

        first_data, first_scale = _mxfp4_cached_weight_qkv(
            owner, q, k, v, None, 32, gemm_rows=32
        )
        reused_data, reused_scale = _mxfp4_cached_weight_qkv(
            owner, q, k, v, None, 32, gemm_rows=32
        )

        assert reused_data is first_data
        assert reused_scale is first_scale
        assert len(calls) == 3
        assert first_data.shape == (192, 32)
        assert first_scale.shape == (6, 2)
        assert torch.all(first_data[:64] == 1)
        assert torch.all(first_data[64:96] == 2)
        assert torch.all(first_data[96:] == 3)
        assert owner._mxfp4_w_cache_version == (
            q._version,
            k._version,
            v._version,
        )
        assert owner._mxfp4_w_cache_sources == (id(q), id(k), id(v))
        assert owner._mxfp4_w_cache_metadata == (
            ((64, 64), (32, 64), (96, 64)),
            torch.bfloat16,
            q.device,
            32,
            (False, False),
        )

        with torch.no_grad():
            k.add_(1)
        rebuilt_data, _ = _mxfp4_cached_weight_qkv(
            owner, q, k, v, None, 32, gemm_rows=32
        )
        assert rebuilt_data is not first_data
        assert len(calls) == 6

    def test_qkv_cache_rebuilds_for_source_and_layout_changes(self, monkeypatch):
        from lumen.ops.quantize import linear as linear_mod
        from lumen.quantize import _mxfp4_cached_weight_qkv

        calls = self._patch_qkv_builder(monkeypatch)
        monkeypatch.setattr(
            linear_mod,
            "_mxfp4_can_fuse_b_shuffle",
            lambda gemm_key, *_args: gemm_key[0] == 64,
        )
        monkeypatch.setattr(linear_mod, "_shuffle_mxfp4_weight", lambda data: data)
        owner = nn.Module()
        q = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        k = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))
        v = nn.Parameter(torch.randn(96, 64, dtype=torch.bfloat16))

        first, _ = _mxfp4_cached_weight_qkv(
            owner, q, k, v, None, 32, gemm_rows=32
        )
        replacement = nn.Parameter(v.detach().clone())
        replaced, _ = _mxfp4_cached_weight_qkv(
            owner, q, k, replacement, None, 32, gemm_rows=32
        )
        layout_changed, _ = _mxfp4_cached_weight_qkv(
            owner, q, k, replacement, None, 32, gemm_rows=64
        )

        assert replaced is not first
        assert layout_changed is not replaced
        assert len(calls) == 9
        assert owner._mxfp4_w_cache[0] == (True, True)
        assert owner._mxfp4_w_cache_sources == (id(q), id(k), id(replacement))
        assert owner._mxfp4_w_cache_metadata[-1] == (True, True)

    def test_qkv_builder_only_concatenates_compact_uint8(self, monkeypatch):
        from lumen.quantize import _mxfp4_cached_weight_qkv

        self._patch_qkv_builder(monkeypatch)
        q = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        k = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))
        v = nn.Parameter(torch.randn(96, 64, dtype=torch.bfloat16))
        real_cat = torch.cat
        cat_dtypes = []

        def _cat(tensors, *args, **kwargs):
            tensors = tuple(tensors)
            cat_dtypes.append(tuple(tensor.dtype for tensor in tensors))
            return real_cat(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", _cat)
        data, scale = _mxfp4_cached_weight_qkv(
            nn.Module(), q, k, v, None, 32, gemm_rows=32
        )

        assert data.dtype == torch.uint8
        assert scale.dtype == torch.uint8
        assert cat_dtypes == [
            (torch.uint8, torch.uint8, torch.uint8),
            (torch.uint8, torch.uint8, torch.uint8),
        ]

    def test_qkv_state_dict_load_preserves_parameters_and_rebuilds_cache(
        self, monkeypatch
    ):
        from lumen.quantize import _mxfp4_cached_weight_qkv

        calls = self._patch_qkv_builder(monkeypatch)
        owner = nn.Module()
        owner.q_proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        owner.k_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        owner.v_proj = nn.Linear(64, 96, bias=False, dtype=torch.bfloat16)
        weights = (
            owner.q_proj.weight,
            owner.k_proj.weight,
            owner.v_proj.weight,
        )
        parameter_ids = tuple(id(weight) for weight in weights)

        cached_data, _ = _mxfp4_cached_weight_qkv(
            owner, *weights, None, 32, gemm_rows=32
        )
        cached_versions = owner._mxfp4_w_cache_version
        checkpoint = {
            name: torch.full_like(value, fill_value=index)
            for index, (name, value) in enumerate(owner.state_dict().items(), start=1)
        }

        assert set(checkpoint) == {
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
        }
        assert not any("_mxfp4_w_cache" in name for name in checkpoint)
        owner.load_state_dict(checkpoint, strict=True)

        assert tuple(id(weight) for weight in weights) == parameter_ids
        live_versions = tuple(weight._version for weight in weights)
        assert live_versions != cached_versions
        assert owner._mxfp4_w_cache_version == cached_versions
        for name, expected in checkpoint.items():
            torch.testing.assert_close(owner.state_dict()[name], expected)

        rebuilt_data, _ = _mxfp4_cached_weight_qkv(
            owner, *weights, None, 32, gemm_rows=32
        )

        assert rebuilt_data is not cached_data
        assert len(calls) == 6
        assert owner._mxfp4_w_cache_version == live_versions
        assert owner._mxfp4_w_cache_sources == parameter_ids

    def test_pair_cache_reuses_and_tracks_both_versions(self, monkeypatch):
        from lumen.quantize import _mxfp4_cached_weight_pair

        calls = self._patch_pair_builder(monkeypatch)
        owner = nn.Module()
        gate = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        up = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))

        first, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )
        reused, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )
        assert reused is first
        assert len(calls) == 2
        assert owner._mxfp4_w_cache_version == (gate._version, up._version)

        with torch.no_grad():
            up.add_(1)
        rebuilt, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )
        assert rebuilt is not first
        assert len(calls) == 4

    def test_pair_cache_rebuilds_for_replaced_parameter(self, monkeypatch):
        from lumen.quantize import _mxfp4_cached_weight_pair

        calls = self._patch_pair_builder(monkeypatch)
        owner = nn.Module()
        gate = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        up = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        first, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )

        replacement = nn.Parameter(up.detach().clone())
        assert replacement._version == up._version == 0
        rebuilt, _ = _mxfp4_cached_weight_pair(
            owner, gate, replacement, None, 32, gemm_rows=32
        )

        assert rebuilt is not first
        assert len(calls) == 4
        assert owner._mxfp4_w_cache_sources == (id(gate), id(replacement))

    def test_pair_cache_rebuilds_when_layout_changes(self, monkeypatch):
        from lumen.ops.quantize import linear as linear_mod
        from lumen.quantize import _mxfp4_cached_weight_pair

        calls = self._patch_pair_builder(monkeypatch)
        monkeypatch.setattr(
            linear_mod,
            "_mxfp4_can_fuse_b_shuffle",
            lambda gemm_key, *_args: gemm_key[0] == 64,
        )
        monkeypatch.setattr(linear_mod, "_shuffle_mxfp4_weight", lambda data: data)
        owner = nn.Module()
        gate = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        up = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))

        first, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )
        rebuilt, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=64
        )

        assert rebuilt is not first
        assert len(calls) == 4
        assert owner._mxfp4_w_cache[0] == (True, True)

    def test_pair_cache_honors_disable_environment(self, monkeypatch):
        from lumen.quantize import _mxfp4_cached_weight_pair

        calls = self._patch_pair_builder(monkeypatch)
        monkeypatch.setenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", "1")
        owner = nn.Module()
        gate = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        up = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))

        first, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )
        rebuilt, _ = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )

        assert rebuilt is not first
        assert len(calls) == 4
        assert not hasattr(owner, "_mxfp4_w_cache")
        assert not hasattr(owner, "_mxfp4_w_cache_version")
        assert not hasattr(owner, "_mxfp4_w_cache_sources")

    def test_pair_fallback_only_concatenates_compact_uint8(self, monkeypatch):
        from lumen.quantize import _mxfp4_cached_weight_pair

        self._patch_pair_builder(monkeypatch)
        owner = nn.Module()
        gate = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        up = nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16))
        real_cat = torch.cat
        cat_dtypes = []

        def _cat(tensors, *args, **kwargs):
            cat_dtypes.append(tuple(t.dtype for t in tensors))
            return real_cat(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", _cat)
        data, scale = _mxfp4_cached_weight_pair(
            owner, gate, up, None, 32, gemm_rows=32
        )

        assert data.dtype == torch.uint8
        assert scale.dtype == torch.uint8
        assert cat_dtypes == [
            (torch.uint8, torch.uint8),
            (torch.uint8, torch.uint8),
        ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pair_builder_matches_aligned_concatenated_reference(self, monkeypatch):
        from lumen.ops.quantize import linear as linear_mod
        from lumen.ops.quantize.ops import convert_to_mxfp4_2d
        from lumen.quantize import _mxfp4_cached_weight_pair

        monkeypatch.setattr(
            linear_mod, "_mxfp4_can_fuse_b_shuffle", lambda *_args: False
        )
        monkeypatch.setattr(
            linear_mod, "_mxfp4_can_fuse_scale_swizzle", lambda *_args: False
        )
        gate = nn.Parameter(torch.randn(32, 64, device="cuda", dtype=torch.bfloat16))
        up = nn.Parameter(torch.randn(32, 64, device="cuda", dtype=torch.bfloat16))

        data, scale = _mxfp4_cached_weight_pair(
            nn.Module(), gate, up, None, 32, gemm_rows=32
        )
        reference_data, reference_scale = convert_to_mxfp4_2d(
            torch.cat((gate.detach(), up.detach()), dim=0), block_size=32
        )

        torch.testing.assert_close(data, reference_data, rtol=0, atol=0)
        torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)

    def test_optimizer_hook_clears_parent_pair_cache(self):
        owner = nn.Module()
        owner.gate = nn.Linear(8, 8, bias=False)
        owner.up = nn.Linear(8, 8, bias=False)
        owner._mxfp4_w_cache = ((False, False), "fp4", "scale")
        owner._mxfp4_w_cache_version = (0, 0)
        owner._mxfp4_w_cache_sources = (1, 2)
        owner._mxfp4_w_cache_metadata = ("metadata",)
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(owner, optimizer)

        optimizer.step()

        assert not hasattr(owner, "_mxfp4_w_cache")
        assert not hasattr(owner, "_mxfp4_w_cache_version")
        assert not hasattr(owner, "_mxfp4_w_cache_sources")
        assert not hasattr(owner, "_mxfp4_w_cache_metadata")

    def test_load_state_dict_hook_clears_module_and_parameter_caches(self):
        cache_attrs = (
            "_mxfp4_w_cache",
            "_mxfp4_w_cache_version",
            "_mxfp4_w_cache_sources",
            "_mxfp4_w_cache_metadata",
        )
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        model._mxfp4_w_cache = ((False, False), "root-fp4", "root-scale")
        model._mxfp4_w_cache_version = (0,)
        model._mxfp4_w_cache_sources = (1,)
        model._mxfp4_w_cache_metadata = ("root",)
        for index, layer in enumerate(model):
            layer._mxfp4_w_cache = ((False, False), "fp4", "scale")
            layer._mxfp4_w_cache_version = index
            layer.weight._mxfp4_w_cache = ((False, False), "fp4", "scale")
            layer.weight._mxfp4_w_cache_metadata = (index,)

        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)
        hook_handle = model._mxfp4_weight_load_state_dict_post_hook_handle
        checkpoint = {
            name: torch.full_like(value, index + 1)
            for index, (name, value) in enumerate(model.state_dict().items())
        }

        model.load_state_dict(checkpoint, strict=True)

        assert (
            tuple(id(parameter) for parameter in model.parameters())
            == parameter_ids
        )
        for module in model.modules():
            for attr in cache_attrs:
                assert not hasattr(module, attr)
            for parameter in module._parameters.values():
                if parameter is not None:
                    for attr in cache_attrs:
                        assert not hasattr(parameter, attr)

        # Re-registering for another optimizer must not stack load hooks.
        register_mxfp4_weight_optimizer_hooks(model, _MegatronStyleOptimizer())
        assert model._mxfp4_weight_load_state_dict_post_hook_handle is hook_handle

    def test_weight_version_invalidates_cache_without_optimizer_hook(self, monkeypatch):
        """Generic nn.Linear training must not depend on Megatron's setup hook."""
        from lumen.ops.quantize import linear as linear_mod
        from lumen.ops.quantize import ops as ops_mod
        from lumen.quantize import _mxfp4_cached_weight

        builds = 0

        def _quantize(*_args, **_kwargs):
            nonlocal builds
            builds += 1
            desc = type("_Desc", (), {})()
            desc.data = torch.full((32, 16), builds, dtype=torch.uint8)
            desc.scale = torch.ones((1, 1), dtype=torch.uint8)
            return desc

        monkeypatch.setattr(linear_mod, "quantize_input", _quantize)
        monkeypatch.setattr(linear_mod, "_mxfp4_can_fuse_b_shuffle", lambda *_args: False)
        monkeypatch.setattr(linear_mod, "_mxfp4_can_fuse_scale_swizzle", lambda *_args: False)
        monkeypatch.setattr(
            ops_mod,
            "transpose_packed_fp4",
            lambda data, **_kwargs: data.t().contiguous(),
        )

        module = nn.Linear(32, 32, bias=False)
        first, _ = _mxfp4_cached_weight(
            module, module.weight, None, None, "mxfp4", None, 32, gemm_rows=32,
        )
        reused, _ = _mxfp4_cached_weight(
            module, module.weight, None, None, "mxfp4", None, 32, gemm_rows=32,
        )
        assert reused is first and builds == 1

        with torch.no_grad():
            module.weight.add_(1)

        rebuilt, _ = _mxfp4_cached_weight(
            module, module.weight, None, None, "mxfp4", None, 32, gemm_rows=32,
        )
        assert rebuilt is not first and builds == 2

    def test_torch_optimizer_post_step_hook_clears_cache(self):
        model = _model_with_cache()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        assert all(_cached(model))
        optimizer.step()
        assert not any(_cached(model))

    def test_optimizer_without_post_step_hook_clears_cache(self):
        model = _model_with_cache()
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        optimizer.step()
        assert not any(_cached(model))
        assert optimizer.steps == 1, "wrapping must still run the original step()"

    def test_native_parameter_cache_is_cleared(self):
        """Native parallel linears keep the cache on weight, not the module."""
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        for layer in model:
            layer.weight._mxfp4_w_cache = ((False, False), "fp4", "scale")
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        optimizer.step()

        assert not any(hasattr(layer.weight, "_mxfp4_w_cache") for layer in model)

    def test_grouped_expert_weights_are_cleared(self):
        """A grouped MoE layer holds its experts as weight0..weightN.

        The linear's forward takes the weight as an argument, so the cache
        lands on a Parameter the module does not expose as ``.weight``. Sweeping
        only that name left every expert quantized from the step-0 masters for
        the whole run while the dense layers updated.
        """
        experts = nn.Module()
        for i in range(3):
            experts.register_parameter(f"weight{i}", nn.Parameter(torch.zeros(8, 8)))
            getattr(experts, f"weight{i}")._mxfp4_w_cache = ((False, False), "fp4", "scale")
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(experts, optimizer)

        optimizer.step()

        assert not any(
            hasattr(getattr(experts, f"weight{i}"), "_mxfp4_w_cache") for i in range(3)
        )

    def test_wrapped_step_returns_original_result(self):
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(_model_with_cache(), optimizer)

        assert optimizer.step() == "step-result"

    def test_model_chunk_list_is_walked(self):
        """Megatron hands out a list of chunks under virtual pipeline parallelism."""
        chunks = [_model_with_cache(), _model_with_cache()]
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(chunks, optimizer)

        optimizer.step()
        assert not any(flag for chunk in chunks for flag in _cached(chunk))

    def test_cache_is_recreated_and_cleared_each_step(self):
        model = _model_with_cache()
        optimizer = _MegatronStyleOptimizer()
        register_mxfp4_weight_optimizer_hooks(model, optimizer)

        for _ in range(3):
            for layer in model:
                layer._mxfp4_w_cache = ((False, False), "fp4", "scale")
            optimizer.step()
            assert not any(_cached(model))

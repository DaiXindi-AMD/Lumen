###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""MXFP4 weight cache invalidation across optimizer shapes.

The cache holds each layer's quantized weight so that gradient-accumulation
micro-batches reuse it. Nothing else invalidates it, so a hook that fails to
fire leaves the run training against the step-0 weights without raising.
"""

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
        layer._mxfp4_w_cache = ("fp4", "scale")
    return model


def _cached(model):
    return [hasattr(layer, "_mxfp4_w_cache") for layer in model]


class TestMXFP4WeightCacheHook:
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
                layer._mxfp4_w_cache = ("fp4", "scale")
            optimizer.step()
            assert not any(_cached(model))

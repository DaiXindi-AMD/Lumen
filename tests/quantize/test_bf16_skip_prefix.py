###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BF16 skip-prefix matching for first/last-layer BF16."""

import torch.nn as nn

from lumen.quantize import ScalingManager, _patch_linear_layers, is_under_bf16_prefix
from lumen.quantize.config import QuantConfig


class TestIsUnderBf16Prefix:
    def test_exact_layer_matches(self):
        prefixes = {"decoder.layers.1"}
        assert is_under_bf16_prefix("decoder.layers.1", prefixes)
        assert is_under_bf16_prefix("decoder.layers.1.mlp.linear_fc1", prefixes)

    def test_layer_1_does_not_match_layer_10(self):
        prefixes = {"decoder.layers.1"}
        assert not is_under_bf16_prefix("decoder.layers.10", prefixes)
        assert not is_under_bf16_prefix("decoder.layers.10.mlp.linear_fc1", prefixes)
        assert not is_under_bf16_prefix("decoder.layers.11", prefixes)

    def test_tail_indices_do_not_collide(self):
        prefixes = {"decoder.layers.31", "decoder.layers.32"}
        assert is_under_bf16_prefix("decoder.layers.31.self_attention", prefixes)
        assert not is_under_bf16_prefix("decoder.layers.3", prefixes)
        assert not is_under_bf16_prefix("decoder.layers.310", prefixes)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4)


class _Decoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])


class _GPT(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.decoder = _Decoder(n)


class TestPatchLinearLayersBf16Skip:
    def test_start_of_2_quantizes_layer_10(self):
        model = _GPT(12)
        config = QuantConfig(
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=1,
            num_layers=12,
        )
        _patch_linear_layers(model, ScalingManager(config), "none", config)

        assert not getattr(model.decoder.layers[1].mlp, "_quant_enabled", False)
        assert model.decoder.layers[10].mlp._quant_enabled is True
        assert not getattr(model.decoder.layers[0].mlp, "_quant_enabled", False)
        assert not getattr(model.decoder.layers[11].mlp, "_quant_enabled", False)

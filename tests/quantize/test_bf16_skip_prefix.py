###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Matching of the BF16-skip prefixes against module names.

The prefixes are layer module paths ending in an index, so a plain
``startswith`` makes ``decoder.layers.1`` match ``decoder.layers.10`` through
``.19`` as well. Nothing raises when that happens -- the run simply keeps ten
more layers in BF16 than asked for, and the only symptom is a step time that
looks like a slow kernel. Both ``_patch_linear_layers`` and
``enable_fp8_for_parallel_linear`` select layers through this predicate.
"""

from lumen.quantize import is_under_bf16_prefix


class TestIsUnderBF16Prefix:
    def test_exact_name_matches(self):
        assert is_under_bf16_prefix("decoder.layers.1", {"decoder.layers.1"})

    def test_submodule_of_a_bf16_layer_matches(self):
        prefixes = {"decoder.layers.1"}
        assert is_under_bf16_prefix("decoder.layers.1.mlp.linear_fc1", prefixes)
        assert is_under_bf16_prefix("decoder.layers.1.self_attention.linear_qkv", prefixes)

    def test_sibling_index_sharing_the_prefix_does_not_match(self):
        """The regression: .1 must not capture .10 through .19."""
        prefixes = {"decoder.layers.1"}
        for index in range(10, 20):
            name = f"decoder.layers.{index}"
            assert not is_under_bf16_prefix(name, prefixes), name
            assert not is_under_bf16_prefix(f"{name}.mlp.linear_fc1", prefixes), name

    def test_no_prefixes_matches_nothing(self):
        assert not is_under_bf16_prefix("decoder.layers.0", set())

    def test_start_of_model_skip_holds_only_its_own_layers(self):
        """A start-of-2 skip on 20 layers: exactly layers 0 and 1 stay BF16."""
        prefixes = {"decoder.layers.0", "decoder.layers.1"}
        held = [i for i in range(20) if is_under_bf16_prefix(f"decoder.layers.{i}", prefixes)]
        assert held == [0, 1]

    def test_tail_skip_holds_only_its_own_layers(self):
        """The default recipe, which the collision happens to spare."""
        prefixes = {f"decoder.layers.{i}" for i in range(31, 36)}
        held = [i for i in range(36) if is_under_bf16_prefix(f"decoder.layers.{i}", prefixes)]
        assert held == [31, 32, 33, 34, 35]

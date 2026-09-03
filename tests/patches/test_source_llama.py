"""Tests for LLaMA SOURCE patch registration."""

import importlib
import sys
import types
from pathlib import Path

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

from lumen.patches import PatchPhase, PatchRegistry, list_patches
from lumen.patches.source import llama


class TestLlamaSourcePatches:
    def setup_method(self):
        PatchRegistry.clear()
        importlib.reload(llama)

    EXPECTED = {
        "llama_megatron_fused_rmsnorm",
        "llama_gpt_layer_specs_rmsnorm",
        "llama_transformer_block_rmsnorm",
    }

    def test_all_llama_source_patches_registered(self):
        names = {spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"llama"})}
        assert self.EXPECTED <= names

    def test_gpt_layer_specs_depends_on_wrapper(self):
        spec = PatchRegistry.get("llama_gpt_layer_specs_rmsnorm")
        assert "llama_megatron_fused_rmsnorm" in spec.depends_on

    def test_transformer_block_depends_on_wrapper(self):
        spec = PatchRegistry.get("llama_transformer_block_rmsnorm")
        assert "llama_megatron_fused_rmsnorm" in spec.depends_on

    def test_llama_patches_not_dsv4_only(self):
        dsv4_only = {
            spec.name
            for spec in list_patches(PatchPhase.SOURCE, tags={"dsv4"})
            if "llama" not in spec.tags
        }
        assert not self.EXPECTED & dsv4_only

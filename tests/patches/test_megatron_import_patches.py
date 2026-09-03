"""Tests for Megatron IMPORT patch registration and installers."""

import sys
import types
from pathlib import Path

import pytest

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

from lumen.patches import PatchPhase, PatchRegistry


def _load_moe_fused_router():
    for key in ("lumen.patches.runtime.moe_fused_router", "lumen.patches.runtime"):
        sys.modules.pop(key, None)
    from lumen.patches.runtime import moe_fused_router

    return moe_fused_router


class TestMegatronImportPatches:
    def setup_method(self):
        PatchRegistry.clear()
        self.moe_fused_router = _load_moe_fused_router()

    def test_moe_fused_router_registered(self):
        spec = PatchRegistry.get("moe_fused_router")
        assert spec.phase is PatchPhase.IMPORT
        assert spec.tags == frozenset({"core", "moe", "megatron"})
        assert spec.default is True
        assert spec.fn is self.moe_fused_router.install_moe_fused_router

    def test_install_moe_fused_router_patches_moe_utils(self):
        pytest.importorskip("torch")
        from lumen.ops.moe.fused_router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )

        moe_utils = types.ModuleType("megatron.core.transformer.moe.moe_utils")
        te_ext = types.ModuleType("megatron.core.extensions.transformer_engine")

        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.extensions",
        ):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["megatron.core.transformer.moe.moe_utils"] = moe_utils
        sys.modules["megatron.core.extensions.transformer_engine"] = te_ext

        self.moe_fused_router.install_moe_fused_router()

        assert moe_utils.fused_topk_with_score_function is fused_topk_with_score_function
        assert moe_utils.fused_compute_score_for_moe_aux_loss is fused_compute_score_for_moe_aux_loss
        assert moe_utils.fused_moe_aux_loss is fused_moe_aux_loss
        assert te_ext.fused_topk_with_score_function is fused_topk_with_score_function
        assert getattr(moe_utils, "_lumen_fused_router_patched") is True

        self.moe_fused_router.install_moe_fused_router()

    def test_install_moe_fused_router_skips_without_megatron(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block_moe_utils_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "megatron.core.transformer.moe.moe_utils":
                raise ImportError("megatron not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _block_moe_utils_import)
        self.moe_fused_router.install_moe_fused_router()


def _load_megatron_import():
    for key in (
        "lumen.patches.runtime.megatron_import",
        "lumen.patches.runtime",
        "lumen.models.megatron_patches",
    ):
        sys.modules.pop(key, None)
    from lumen.patches.runtime import megatron_import

    return megatron_import


class TestMegatronImportModuleLocation:
    def setup_method(self):
        PatchRegistry.clear()

    def test_megatron_import_registers_core_patches(self):
        pytest.importorskip("torch")
        _load_megatron_import()
        names = {spec.name for spec in PatchRegistry.all() if spec.phase is PatchPhase.IMPORT}
        assert "fused_layer_norm" in names
        assert "mmap_checkpoint" in names

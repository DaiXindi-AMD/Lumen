"""Tests for Megatron TRAINING-phase patch registration and apply order."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

from lumen.patches import PatchPhase, PatchRegistry, apply_training_patches
from lumen.patches.registry import _topological_sort
from lumen.patches.types import PatchSpec


def _load_training_hooks():
    for key in (
        "lumen.patches.training.megatron_hooks",
        "lumen.patches.training",
    ):
        sys.modules.pop(key, None)
    from lumen.patches.training import megatron_hooks

    return megatron_hooks


class TestMegatronTrainingPatches:
    def setup_method(self):
        PatchRegistry.clear()
        self.hooks = _load_training_hooks()

    def test_all_training_hooks_registered(self):
        expected = {
            "fp8_param_gather_hook",
            "mxfp4_weight_cache_hook",
            "gc_freeze_hook",
            "fp8_param_storage_hook",
            "hip_graphs_hook",
            "val_loss_early_stop_hook",
        }
        registered = {
            spec.name
            for spec in PatchRegistry.all()
            if spec.phase is PatchPhase.TRAINING
        }
        assert registered == expected
        for name in expected:
            spec = PatchRegistry.get(name)
            assert spec.default is False
            assert "megatron" in spec.tags
            assert "training" in spec.tags

    def test_dependency_order_for_llama2_bundle(self):
        specs = [
            PatchRegistry.get(name)
            for name in (
                "fp8_param_gather_hook",
                "fp8_param_storage_hook",
                "hip_graphs_hook",
                "val_loss_early_stop_hook",
            )
        ]
        ordered = [spec.name for spec in _topological_sort(specs)]
        assert ordered.index("fp8_param_gather_hook") < ordered.index("fp8_param_storage_hook")
        assert ordered.index("fp8_param_storage_hook") < ordered.index("hip_graphs_hook")

    def test_storage_depends_on_gather(self):
        spec = PatchRegistry.get("fp8_param_storage_hook")
        assert spec.depends_on == ("fp8_param_gather_hook",)

    def test_hip_graphs_depends_on_fp8_hooks(self):
        spec = PatchRegistry.get("hip_graphs_hook")
        assert spec.depends_on == ("fp8_param_gather_hook", "fp8_param_storage_hook")

    def test_apply_training_patches_invokes_installers(self):
        calls = []

        def _stub(patch_name, label):
            spec = PatchRegistry.get(patch_name)

            def _installer():
                calls.append(label)

            PatchRegistry._patches[patch_name] = PatchSpec(
                name=spec.name,
                phase=spec.phase,
                fn=_installer,
                description=spec.description,
                enabled=spec.enabled,
                depends_on=spec.depends_on,
                tags=spec.tags,
                config_fields=spec.config_fields,
                default=spec.default,
            )

        _stub("fp8_param_gather_hook", "install_fp8_param_gather_hook")
        _stub("fp8_param_storage_hook", "install_fp8_param_storage_hook")
        _stub("hip_graphs_hook", "install_hip_graphs_hook")
        _stub("val_loss_early_stop_hook", "install_val_loss_early_stop_hook")

        apply_training_patches(
            names={
                "fp8_param_gather_hook",
                "fp8_param_storage_hook",
                "hip_graphs_hook",
                "val_loss_early_stop_hook",
            }
        )

        assert calls.index("install_fp8_param_gather_hook") < calls.index(
            "install_fp8_param_storage_hook"
        )
        assert calls.index("install_fp8_param_storage_hook") < calls.index("install_hip_graphs_hook")
        assert "install_val_loss_early_stop_hook" in calls

    def test_val_loss_hook_skips_without_megatron(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block_training_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "megatron.training.training":
                raise ImportError("megatron not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _block_training_import)
        self.hooks.install_val_loss_early_stop_hook()

    def test_gc_freeze_hook_skips_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LUMEN_GC_FREEZE", "0")
        self.hooks.install_gc_freeze_hook()

    def _stub_megatron_training(self, monkeypatch, *, setup=None, get_args=None):
        """Point TRAINING installers at in-process stubs, not a real Megatron."""
        training_mod = types.ModuleType("megatron.training.training")
        training_mod.setup_model_and_optimizer = setup or (
            lambda *a, **k: ("model", "optimizer", "scheduler")
        )
        training_mod.train_step = lambda *a, **k: None

        training_pkg = types.ModuleType("megatron.training")
        training_pkg.get_args = get_args or (lambda: SimpleNamespace())
        training_pkg.print_rank_0 = lambda *a, **k: None
        training_pkg.training = training_mod

        megatron_pkg = types.ModuleType("megatron")
        megatron_pkg.training = training_pkg

        monkeypatch.setitem(sys.modules, "megatron", megatron_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)
        return training_mod

    def _stub_mxfp4_hook_deps(self, monkeypatch, register):
        """Avoid importing real lumen.quantize / megatron.py (AITER, ROCm)."""
        megatron_compat = types.ModuleType("lumen.models.megatron")
        megatron_compat.resolve_quant_format = lambda args: getattr(
            args, "linear_fp8_format", None
        )
        quant = types.ModuleType("lumen.quantize")
        quant.register_mxfp4_weight_optimizer_hooks = register
        monkeypatch.setitem(sys.modules, "lumen.models.megatron", megatron_compat)
        monkeypatch.setitem(sys.modules, "lumen.quantize", quant)

    def test_mxfp4_weight_cache_hook_registers_on_mxfp4_run(self, monkeypatch):
        model, optimizer = object(), object()
        training_mod = self._stub_megatron_training(
            monkeypatch,
            setup=lambda *a, **k: (model, optimizer, None),
            get_args=lambda: SimpleNamespace(linear_fp8=True, linear_fp8_format="mxfp4"),
        )
        register = mock.MagicMock()
        self._stub_mxfp4_hook_deps(monkeypatch, register)
        monkeypatch.delenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", raising=False)

        self.hooks.install_mxfp4_weight_cache_hook()
        training_mod.setup_model_and_optimizer()

        register.assert_called_once_with(model, optimizer)

    def test_mxfp4_weight_cache_hook_skips_when_disabled(self, monkeypatch):
        training_mod = self._stub_megatron_training(
            monkeypatch,
            get_args=lambda: SimpleNamespace(linear_fp8=True, linear_fp8_format="mxfp4"),
        )
        register = mock.MagicMock()
        self._stub_mxfp4_hook_deps(monkeypatch, register)
        monkeypatch.setenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", "1")

        self.hooks.install_mxfp4_weight_cache_hook()
        training_mod.setup_model_and_optimizer()

        register.assert_not_called()

    def test_mxfp4_weight_cache_hook_skips_non_mxfp4_format(self, monkeypatch):
        training_mod = self._stub_megatron_training(
            monkeypatch,
            get_args=lambda: SimpleNamespace(linear_fp8=True, linear_fp8_format="fp8_e4m3"),
        )
        register = mock.MagicMock()
        self._stub_mxfp4_hook_deps(monkeypatch, register)

        self.hooks.install_mxfp4_weight_cache_hook()
        training_mod.setup_model_and_optimizer()

        register.assert_not_called()

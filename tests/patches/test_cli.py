"""CLI tests for SOURCE patch apply/list/dry-run."""

import importlib
import sys
import types
from pathlib import Path

if "lumen" not in sys.modules:
    _lumen = types.ModuleType("lumen")
    _lumen.__path__ = [str(Path(__file__).resolve().parents[2] / "lumen")]
    sys.modules["lumen"] = _lumen

from lumen.patches import PatchPhase, PatchRegistry
from lumen.patches.cli import main, parse_cli_tags
from lumen.patches.source import dsv4, llama, llama_lora, rocm


class TestParseCliTags:
    def test_comma_separated_uses_any_mode(self):
        tags, mode = parse_cli_tags(["dsv4,rocm"])
        assert tags == {"dsv4", "rocm"}
        assert mode == "any"

    def test_repeated_flags_use_all_mode(self):
        tags, mode = parse_cli_tags(["dsv4", "rocm"])
        assert tags == {"dsv4", "rocm"}
        assert mode == "all"

    def test_single_tag(self):
        tags, mode = parse_cli_tags(["llama"])
        assert tags == {"llama"}
        assert mode == "all"


class TestSourcePatchCli:
    def setup_method(self):
        PatchRegistry.clear()
        for mod in (dsv4, llama, llama_lora, rocm):
            importlib.reload(mod)

    def test_dry_run_llama_tag(self, capsys, tmp_path):
        megatron = tmp_path / "Megatron-LM"
        megatron.mkdir()
        rc = main([str(megatron), "--tag", "llama", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Would patch" in out
        assert "would apply: llama_megatron_fused_rmsnorm" in out
        assert "would apply: lora_requires_grad" not in out

    def test_dry_run_lora_tag_includes_opt_in(self, capsys, tmp_path):
        megatron = tmp_path / "Megatron-LM"
        megatron.mkdir()
        rc = main([str(megatron), "--tag", "lora", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "would apply: lora_checkpoint_load" in out

    def test_list_llama_patches(self, capsys):
        rc = main(["--list", "--tag", "llama"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "llama_megatron_fused_rmsnorm" in out

    def test_dry_run_dsv4_rocm_comma_includes_standalone_rocm(self, capsys, tmp_path):
        megatron = tmp_path / "Megatron-LM"
        megatron.mkdir()
        rc = main([str(megatron), "--tag", "dsv4,rocm", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "would apply: dsv4_transformer_config" in out
        assert "would apply: disable_batch_p2p_comm" in out

    def test_dry_run_llama_lora_comma(self, capsys, tmp_path):
        megatron = tmp_path / "Megatron-LM"
        megatron.mkdir()
        rc = main([str(megatron), "--tag", "llama,lora", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "would apply: llama_megatron_fused_rmsnorm" in out
        assert "would apply: lora_checkpoint_load" in out

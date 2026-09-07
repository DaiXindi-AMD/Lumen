"""Contracts copied from the MILES DSV4 Megatron training flow."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
ATTENTION = REPO_ROOT / "lumen/models/dsv4/megatron/deepseek_v4.py"
COMPRESSOR = REPO_ROOT / "lumen/models/dsv4/ops/compressor.py"
INDEXER = REPO_ROOT / "lumen/models/dsv4/megatron/v4_indexer.py"
QAT = REPO_ROOT / "lumen/ops/dsv4/qat.py"
HYPER_CONNECTION = REPO_ROOT / "lumen/models/dsv4/ops/hyper_connection.py"


def _call_block_sizes(path: Path, function_name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        ast.literal_eval(node.args[1])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == function_name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == function_name
        )
    ]


def test_qat_is_gated_by_megatron_fp8_config():
    for path in (ATTENTION, COMPRESSOR, INDEXER):
        source = path.read_text(encoding="utf-8")
        assert "self.use_fp8_qat = config.fp8 is not None" in source
        assert "if self.use_fp8_qat:" in source


def test_miles_uses_fp8_block64_and_block128_only():
    assert _call_block_sizes(ATTENTION, "fp8_simulate_qat") == [64]
    assert _call_block_sizes(COMPRESSOR, "_fp8_simulate_qat") == [128, 64]
    assert _call_block_sizes(INDEXER, "fp8_simulate_qat") == [128]
    sources = "".join(
        path.read_text(encoding="utf-8")
        for path in (ATTENTION, COMPRESSOR, INDEXER, QAT)
    )
    assert "fp4" not in sources


def test_dsv4_learned_biases_start_from_finite_identity_semantics():
    compressor_source = COMPRESSOR.read_text(encoding="utf-8")
    attention_source = ATTENTION.read_text(encoding="utf-8")
    hc_source = HYPER_CONNECTION.read_text(encoding="utf-8")

    assert "self.ape = nn.Parameter(torch.zeros(" in compressor_source
    assert "self.attn_sink = nn.Parameter(torch.zeros(" in attention_source
    assert "self.hc_head_fn = torch.nn.Parameter(torch.zeros(" in hc_source
    assert "self.hc_head_base = torch.nn.Parameter(torch.zeros(" in hc_source
    assert "self.hc_head_scale = torch.nn.Parameter(torch.zeros(" in hc_source

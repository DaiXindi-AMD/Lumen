"""DSV4 kernels and helpers (vendored from Miles ``miles_plugins``)."""

from lumen.models.dsv4.ops.compressor import DeepSeekV4Compressor
from lumen.models.dsv4.ops.cp_utils import (
    all_gather_cp,
    get_compress_topk_idxs_cp,
    get_freqs_cis_for_cp,
    get_q_positions_for_cp,
    get_window_topk_idxs_cp,
)
from lumen.models.dsv4.ops.dsa_topk import get_dsa_topk_fn
from lumen.models.dsv4.ops.kernel.tilelang_sparse_mla import sparse_attn_tilelang
from lumen.models.dsv4.ops.qat import fp8_simulate_qat
from lumen.models.dsv4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from lumen.models.dsv4.ops.utils import rotate_activation

__all__ = [
    "DeepSeekV4Compressor",
    "all_gather_cp",
    "get_compress_topk_idxs_cp",
    "get_dsa_topk_fn",
    "get_freqs_cis_for_cp",
    "get_q_positions_for_cp",
    "get_window_topk_idxs_cp",
    "rotate_activation",
    "sparse_attn_tilelang",
    "fp8_simulate_qat",
    "apply_rotary_emb",
    "wrapped_precompute_freqs_cis",
]

"""DSV4 sparse MLA via aiter Triton kernels (MI308X-optimised).

Layout adapter: miles uses [B,S,H,D] batched tensors; aiter expects flat [N,H,D].
"""

import torch

from aiter.ops.triton.attention.sparse_mla_dsv4_train import sparse_mla_dsv4_train


def _to_flat(q, kv, topk_idxs):
    """Convert miles batched layout to aiter flat layout."""
    B, S, H, D = q.shape
    S_kv = kv.shape[1]
    N = B * S
    q_flat = q.reshape(N, H, D).contiguous()
    kv_flat = kv.reshape(B * S_kv, D).contiguous()
    idx = topk_idxs.reshape(N, -1).contiguous()
    if B > 1:
        offsets = (torch.arange(B, device=idx.device, dtype=idx.dtype) * S_kv).view(B, 1, 1)
        idx_b = idx.view(B, S, -1) + offsets
        valid = idx_b >= 0
        idx = torch.where(valid, idx_b, torch.full_like(idx_b, -1)).reshape(N, -1)
    return q_flat, kv_flat, idx, (B, S, H, D)


def sparse_attn_triton(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Drop-in replacement for sparse_attn_tilelang with identical signature."""
    q_flat, kv_flat, idx_flat, shape = _to_flat(q, kv, topk_idxs)
    out_flat = sparse_mla_dsv4_train(q_flat, kv_flat, attn_sink, idx_flat, sm_scale)
    B, S, H, D = shape
    return out_flat.reshape(B, S, H, D)

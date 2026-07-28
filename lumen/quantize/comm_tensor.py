###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

import torch
import torch.utils._pytree as pytree

from lumen.quantize.fp8_params import quantize_param_to_fp8

__all__ = ["FP8CommTensor", "Blockwise2DFP8Param", "Blockwise2DFP8Gathered"]


class FP8CommTensor(torch.Tensor):
    """Thin tensor subclass for FSDP2 FP8 all-gather communication.

    Wraps a BF16 parameter and provides FSDP2 hooks that quantize the
    local shard to FP8 before all-gather, then dequantize after.  This
    halves the all-gather bandwidth without changing optimizer or
    gradient behaviour.
    """

    _fp8_dtype: torch.dtype

    @staticmethod
    def __new__(cls, data: torch.Tensor, fp8_dtype: torch.dtype):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
            requires_grad=data.requires_grad,
        )

    def __init__(self, data: torch.Tensor, fp8_dtype: torch.dtype):
        self._data = data
        self._fp8_dtype = fp8_dtype

    def __repr__(self):
        return f"FP8CommTensor(shape={list(self.shape)}, dtype={self.dtype}, fp8_dtype={self._fp8_dtype})"

    def __tensor_flatten__(self):
        return ["_data"], {"fp8_dtype": self._fp8_dtype}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_data"], metadata["fp8_dtype"])

    @staticmethod
    def fsdp_pre_all_gather(tensor) -> tuple[tuple[torch.Tensor, ...], dict]:
        fp8_data, scale = quantize_param_to_fp8(tensor._data, tensor._fp8_dtype)
        return (fp8_data, scale.float().reshape(1)), {}

    @staticmethod
    def fsdp_post_all_gather(
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: dict,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fp8_gathered, scales_gathered = all_gather_outputs
        world_size = scales_gathered.numel()
        rows_per_rank = fp8_gathered.shape[0] // world_size
        # Write each rank's dequant directly into pre-allocated output;
        # avoids N intermediate BF16 tensors + torch.cat (+ an extra copy when out!=None).
        result = out if out is not None else torch.empty(
            fp8_gathered.shape, dtype=param_dtype, device=fp8_gathered.device
        )
        for i in range(world_size):
            row_s = i * rows_per_rank
            row_e = row_s + rows_per_rank
            result[row_s:row_e].copy_(
                fp8_gathered[row_s:row_e].to(torch.float32).div_(scales_gathered[i])
            )
        return result

    _FSDP2_SAFE_OPS = {
        torch.ops.aten.copy_.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.detach.default,
    }

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            return x._data if isinstance(x, FP8CommTensor) else x

        if func in cls._FSDP2_SAFE_OPS:
            unwrapped_args = pytree.tree_map(_unwrap, args)
            unwrapped_kwargs = pytree.tree_map(_unwrap, kwargs)
            result = func(*unwrapped_args, **unwrapped_kwargs)
            if isinstance(result, torch.Tensor) and not isinstance(result, FP8CommTensor):
                source = next(
                    (a for a in pytree.tree_leaves(args) if isinstance(a, FP8CommTensor)),
                    None,
                )
                if source is not None:
                    return FP8CommTensor(result, source._fp8_dtype)
            return result

        unwrapped_args = pytree.tree_map(_unwrap, args)
        unwrapped_kwargs = pytree.tree_map(_unwrap, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)


class Blockwise2DFP8Gathered(torch.Tensor):
    """The all-gathered, FP8 form of a frozen blockwise2d base weight.

    Returned by :meth:`Blockwise2DFP8Param.fsdp_post_all_gather` and set as the
    unsharded ``module.weight`` for the forward. Carries the full FP8 weight
    ``(N, K)`` plus its 2D ``(N/block, K/block)`` tile scale; the patched Linear
    forward detects this type and feeds ``(_fp8, _scale)`` straight to the FP8
    GEMM — no re-quantization in the forward.
    """

    @staticmethod
    def __new__(cls, fp8_data, scale, orig_dtype, block_size=128):
        return torch.Tensor._make_wrapper_subclass(
            cls, fp8_data.shape, dtype=orig_dtype, device=fp8_data.device, requires_grad=False
        )

    def __init__(self, fp8_data, scale, orig_dtype, block_size=128):
        self._fp8 = fp8_data
        self._scale = scale
        self._orig_dtype = orig_dtype
        self._block_size = block_size

    def __repr__(self):
        return (
            f"Blockwise2DFP8Gathered(shape={list(self.shape)}, fp8={self._fp8.dtype}, "
            f"scale={list(self._scale.shape)})"
        )

    def __tensor_flatten__(self):
        return ["_fp8", "_scale"], {"orig_dtype": self._orig_dtype, "block_size": self._block_size}

    @classmethod
    def __tensor_unflatten__(cls, inner, meta, outer_size, outer_stride):
        return cls(inner["_fp8"], inner["_scale"], meta["orig_dtype"], meta["block_size"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            return x._fp8 if isinstance(x, Blockwise2DFP8Gathered) else x

        source = next(
            (a for a in pytree.tree_leaves(args) if isinstance(a, Blockwise2DFP8Gathered)),
            None,
        )
        result = func(*pytree.tree_map(_unwrap, args), **pytree.tree_map(_unwrap, kwargs))
        if (
            source is not None
            and isinstance(result, torch.Tensor)
            and not isinstance(result, Blockwise2DFP8Gathered)
            and result.shape == source._fp8.shape
        ):
            return Blockwise2DFP8Gathered(result, source._scale, source._orig_dtype, source._block_size)
        return result


class Blockwise2DFP8Param(torch.Tensor):
    """FSDP2 parameter wrapper for a **frozen** blockwise2d base weight.

    Unlike :class:`FP8CommTensor` (per-tensor scale, wraps *trainable* params,
    dequantizes back to BF16 so the GEMM re-quantizes), this wraps a LoRA-frozen
    base weight to all-gather it as FP8 and feed the FP8 GEMM directly.

    Design (mirrors torchao's float8 FSDP2 integration): the param holds a single
    BF16 inner tensor (``_tensor``) whose dtype/shape match the outer wrapper, so
    FSDP2 shards it like any BF16 param and keeps the subclass as the sharded-local
    tensor (a wrapper subclass with two inner tensors of *different* shapes is not
    preserved by FSDP2 — it collapses to a plain DTensor and the extension never
    fires). The FP8 conversion happens in the all-gather extension:

      - ``fsdp_pre_all_gather`` quantizes the **local BF16 shard** to FP8 + 2D scale
        (1/world_size of the work, overlapped with the all-gather) and ships FP8 on
        the wire (~half the param-comm bytes);
      - ``fsdp_post_all_gather`` returns a :class:`Blockwise2DFP8Gathered` carrying
        the gathered FP8 + 2D scale — the forward GEMM consumes it with no re-quant.

    Per-shard blockwise2d quant is exact iff ``N % (block * world_size) == 0`` (dim-0
    shards never split a 128-row tile).
    """

    _fp8_dtype: torch.dtype
    _block_size: int

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, fp8_dtype: torch.dtype, block_size: int = 128):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, fp8_dtype: torch.dtype, block_size: int = 128):
        self._tensor = tensor           # BF16 master (sharded by FSDP2)
        self._fp8_dtype = fp8_dtype
        self._block_size = block_size

    def __repr__(self):
        return f"Blockwise2DFP8Param(shape={list(self.shape)}, master={self._tensor.dtype}, fp8={self._fp8_dtype})"

    def __tensor_flatten__(self):
        return ["_tensor"], {"fp8_dtype": self._fp8_dtype, "block_size": self._block_size}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_tensor"], metadata["fp8_dtype"], metadata["block_size"])

    # NOTE: FSDP2 (torch 2.8) introspects the parameter count of these hooks and only
    # accepts the *instance-method* forms (1 param after ``self`` = old BC signature,
    # or 5 = new). A ``@staticmethod`` taking ``(tensor)`` is read as the 1-param form
    # and FSDP2 passes the *mesh* in place of the tensor — silently mis-invoking it.
    def fsdp_pre_all_gather(self, mesh) -> tuple[tuple[torch.Tensor, ...], dict]:
        from lumen.ops.quantize.linear import _quant_blockwise2d_weight

        fp8, scale = _quant_blockwise2d_weight(
            self._tensor.contiguous(), self._fp8_dtype, self._block_size
        )
        return (fp8, scale), {"orig_dtype": self._tensor.dtype, "block_size": self._block_size}

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: dict,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        fp8_full, scale_full = all_gather_outputs
        if out is not None:
            assert isinstance(out, Blockwise2DFP8Gathered)
            if out._fp8.data_ptr() != fp8_full.data_ptr():
                out._fp8.copy_(fp8_full)
            if out._scale.data_ptr() != scale_full.data_ptr():
                out._scale.copy_(scale_full)
            return
        result = Blockwise2DFP8Gathered(
            fp8_full, scale_full, metadata["orig_dtype"], metadata["block_size"]
        )
        return result, (fp8_full, scale_full)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            return x._tensor if isinstance(x, Blockwise2DFP8Param) else x

        source = next(
            (a for a in pytree.tree_leaves(args) if isinstance(a, Blockwise2DFP8Param)),
            None,
        )
        result = func(*pytree.tree_map(_unwrap, args), **pytree.tree_map(_unwrap, kwargs))
        if source is None:
            return result

        # Re-wrap every master-dtype tensor output so the subclass SURVIVES FSDP2's
        # sharding ops (chunk/narrow/view/clone/copy_/detach all change shape or alias
        # but keep dtype). A shape check here would drop the subclass on the sharded
        # shard → the all-gather extension would never attach.
        def _wrap(t):
            if (
                isinstance(t, torch.Tensor)
                and not isinstance(t, Blockwise2DFP8Param)
                and t.dtype == source._tensor.dtype
            ):
                return Blockwise2DFP8Param(t, source._fp8_dtype, source._block_size)
            return t

        return pytree.tree_map(_wrap, result)

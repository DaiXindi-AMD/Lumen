###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Qwen3-specific MXFP4 model adaptations."""

import logging
import os
import weakref

from torch import nn

logger = logging.getLogger(__name__)


def _has_module_call_hooks(module):
    """Return whether calling ``module.forward`` directly would skip hooks."""
    return any(
        getattr(module, hook_name, None)
        for hook_name in (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_backward_pre_hooks",
            "_backward_hooks",
        )
    )


def _forward_identity(module):
    """Return a stable identity for class-bound or instance-assigned forwards."""
    forward = module.forward
    return getattr(forward, "__func__", forward)


def _qkv_static_ineligibility(attention):
    projections = tuple(
        getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj")
    )
    if any(type(projection) is not nn.Linear for projection in projections):
        return "q_proj, k_proj, and v_proj must be plain nn.Linear modules"
    if any(_has_module_call_hooks(projection) for projection in projections):
        return "Q/K/V module hooks would be bypassed by the packed projection"
    if any(
        not getattr(projection, "_quant_enabled", False) for projection in projections
    ):
        return "all Q/K/V projections must already be quantized"
    if any(
        getattr(projection, "_quant_scaling_type", None) != "mxfp4"
        for projection in projections
    ):
        return "all Q/K/V projections must use MXFP4"
    managers = tuple(
        getattr(projection, "_quant_manager", None) for projection in projections
    )
    if any(manager is not managers[0] for manager in managers[1:]):
        return "Q/K/V projections must share one scaling manager"
    if any(
        not getattr(projection, "_lumen_quantize_activation", False)
        for projection in projections
    ):
        return "packed QKV requires activation quantization on every projection"
    fp8_dtypes = tuple(
        getattr(projection, "_lumen_fp8_dtype", None) for projection in projections
    )
    block_sizes = tuple(
        getattr(projection, "_lumen_block_size", 32) for projection in projections
    )
    if any(dtype != fp8_dtypes[0] for dtype in fp8_dtypes[1:]) or any(
        block_size != block_sizes[0] for block_size in block_sizes[1:]
    ):
        return "Q/K/V quantization settings differ"
    if block_sizes[0] != 32:
        return "packed QKV requires MXFP4 block_size=32"
    if any(projection.bias is not None for projection in projections):
        return "biased Q/K/V projections are unsupported"
    weights = tuple(projection.weight for projection in projections)
    if any(weight.dim() != 2 for weight in weights):
        return "Q/K/V weights must all be 2D"
    input_widths = {weight.shape[1] for weight in weights}
    if len(input_widths) != 1:
        return "Q/K/V input widths differ"
    if any(weight.shape[0] % 32 for weight in weights) or weights[0].shape[1] % 32:
        return "Q/K/V dimensions must be multiples of 32"
    if any(
        weight.dtype != weights[0].dtype or weight.device != weights[0].device
        for weight in weights[1:]
    ):
        return "Q/K/V weights must share dtype and device"
    if any(weight.requires_grad != weights[0].requires_grad for weight in weights[1:]):
        return "Q/K/V weights must have compatible requires_grad values"
    if any(getattr(projection, "delay_wgrad", False) for projection in projections):
        return "delay_wgrad is unsupported"
    if any(
        getattr(projection, "gradient_accumulation_fusion", False)
        for projection in projections
    ):
        return "gradient accumulation fusion is unsupported"
    if os.environ.get("LUMEN_MXFP4_DGRAD_HADAMARD", "0") == "1":
        return "LUMEN_MXFP4_DGRAD_HADAMARD=1 is unsupported"
    if any(hasattr(projection, "_mxfp4_w_cache") for projection in projections):
        return (
            "Q/K/V child weight caches already exist; enable packing before first "
            "forward"
        )
    return None


def _qkv_runtime_ineligibility(attention, hidden_states):
    projections = tuple(
        getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj")
    )
    references = tuple(reference() for reference in attention._mxfp4_qkv_module_refs)
    if projections != references:
        return "a Q/K/V projection module was replaced after packing was enabled"
    reason = _qkv_static_ineligibility(attention)
    if reason is not None:
        return reason
    if tuple(_forward_identity(projection) for projection in projections) != (
        attention._mxfp4_qkv_forward_identities
    ):
        return "a Q/K/V projection forward was replaced after packing was enabled"
    weights = tuple(projection.weight for projection in projections)
    if (
        tuple(tuple(weight.shape) for weight in weights)
        != attention._mxfp4_qkv_weight_shapes
    ):
        return "Q/K/V weights are not currently full gathered matrices"
    if hidden_states.shape[-1] != weights[0].shape[1]:
        return "input width does not match the packed QKV weights"

    from lumen.ops.quantize.linear import _mxfp4_shape_operands_fit_int32

    if not _mxfp4_shape_operands_fit_int32(
        hidden_states,
        n_out=sum(weight.shape[0] for weight in weights),
        k_in=weights[0].shape[1],
    ):
        return "packed QKV operands exceed AITER's 32-bit index limit"
    return None


def _warn_qkv_fallback(attention, reason):
    warned = getattr(attention, "_mxfp4_qkv_warned", None)
    if warned is None:
        warned = set()
        attention._mxfp4_qkv_warned = warned
    if reason in warned:
        return
    warned.add(reason)
    logger.warning(
        "MXFP4 packed QKV disabled for %s: %s; using the original "
        "Qwen3Attention forward",
        getattr(attention, "_mxfp4_qkv_name", type(attention).__name__),
        reason,
    )


def enable_mxfp4_qwen_qkv(model: nn.Module, *, strict: bool = False) -> int:
    """Pack eligible Qwen3 Q/K/V projections without replacing Parameters.

    ``strict`` first validates every attention marked for MXFP4, then patches
    atomically. This prevents an explicitly requested performance run from
    silently benchmarking a mixture of packed and unpacked attention layers.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            Qwen3Attention,
            Qwen3DecoderLayer,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except ImportError as error:
        raise ImportError("MXFP4 QKV packing requires transformers Qwen3") from error

    modules = dict(model.named_modules())
    qwen_attentions = tuple(
        (name, module)
        for name, module in modules.items()
        if type(module) is Qwen3Attention
    )

    def _is_marked_for_mxfp4(attention):
        return any(
            getattr(projection, "_quant_enabled", False)
            or getattr(projection, "_quant_scaling_type", None) == "mxfp4"
            for projection in (
                getattr(attention, "q_proj", None),
                getattr(attention, "k_proj", None),
                getattr(attention, "v_proj", None),
            )
        )

    def _owner_reason(name, attention):
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        if (
            type(parent) is not Qwen3DecoderLayer
            or getattr(parent, "self_attn", None) is not attention
        ):
            return "attention is not owned by a Qwen3 decoder layer"
        if hasattr(attention, "_mxfp4_qkv_original_forward"):
            current = tuple(
                getattr(attention, projection_name, None)
                for projection_name in ("q_proj", "k_proj", "v_proj")
            )
            expected = tuple(
                reference() for reference in attention._mxfp4_qkv_module_refs
            )
            if current != expected:
                return "a Q/K/V module was replaced after packing was enabled"
        return _qkv_static_ineligibility(attention)

    if strict:
        requested = tuple(
            (name, attention)
            for name, attention in qwen_attentions
            if _is_marked_for_mxfp4(attention)
        )
        eligible = 0
        for name, attention in requested:
            reason = _owner_reason(name, attention)
            if reason is None:
                eligible += 1
            else:
                _warn_qkv_fallback(attention, reason)
        if not requested or eligible != len(requested):
            raise RuntimeError(
                "MXFP4 packed QKV was explicitly requested but can enable on "
                f"{eligible} of {len(requested)} marked Qwen3 attention layers"
            )

    patched = 0
    for name, attention in qwen_attentions:
        if hasattr(attention, "_mxfp4_qkv_original_forward"):
            patched += 1
            continue
        reason = _owner_reason(name, attention)
        if reason is not None:
            _warn_qkv_fallback(attention, reason)
            continue

        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        weights = tuple(projection.weight for projection in projections)
        attention._mxfp4_qkv_original_forward = attention.forward
        attention._mxfp4_qkv_module_refs = tuple(
            weakref.ref(projection) for projection in projections
        )
        attention._mxfp4_qkv_forward_identities = tuple(
            _forward_identity(projection) for projection in projections
        )
        attention._mxfp4_qkv_weight_shapes = tuple(
            tuple(weight.shape) for weight in weights
        )
        attention._mxfp4_qkv_rows = tuple(weight.shape[0] for weight in weights)
        attention._mxfp4_qkv_name = name

        def packed_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            _attention=attention,
            **kwargs,
        ):
            reason = _qkv_runtime_ineligibility(_attention, hidden_states)
            if reason is not None:
                _warn_qkv_fallback(_attention, reason)
                return _attention._mxfp4_qkv_original_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    **kwargs,
                )

            q_projection = _attention.q_proj
            k_projection = _attention.k_proj
            v_projection = _attention.v_proj
            from lumen.ops.quantize.linear import mxfp4_qkv_linear
            from lumen.quantize import _mxfp4_cached_weight_qkv

            packed_weight, packed_scale = _mxfp4_cached_weight_qkv(
                _attention,
                q_projection.weight,
                k_projection.weight,
                v_projection.weight,
                q_projection._lumen_fp8_dtype,
                getattr(q_projection, "_lumen_block_size", 32),
                gemm_rows=hidden_states.numel() // hidden_states.shape[-1],
            )
            packed = mxfp4_qkv_linear(
                hidden_states,
                q_projection.weight,
                k_projection.weight,
                v_projection.weight,
                packed_weight,
                packed_scale,
                scaling_manager=q_projection._quant_manager,
                fp8_dtype=q_projection._lumen_fp8_dtype,
                block_size=getattr(q_projection, "_lumen_block_size", 32),
                tensor_id=f"{_attention._mxfp4_qkv_name}.qkv.weight",
            )
            query_projection, key_projection, value_projection = (
                part.contiguous()
                for part in packed.split(_attention._mxfp4_qkv_rows, dim=-1)
            )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, _attention.head_dim)
            query_states = _attention.q_norm(
                query_projection.view(hidden_shape)
            ).transpose(1, 2)
            key_states = _attention.k_norm(key_projection.view(hidden_shape)).transpose(
                1, 2
            )
            value_states = value_projection.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, _attention.layer_idx
                )

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                _attention.config._attn_implementation,
                eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                _attention,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=(
                    0.0 if not _attention.training else _attention.attention_dropout
                ),
                scaling=_attention.scaling,
                sliding_window=_attention.sliding_window,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return _attention.o_proj(attn_output), attn_weights

        attention.forward = packed_forward
        patched += 1
    if patched:
        from lumen.quantize import _register_mxfp4_load_state_dict_hooks

        _register_mxfp4_load_state_dict_hooks(model)
    return patched


def disable_mxfp4_qwen_qkv(model: nn.Module) -> int:
    """Restore Qwen3Attention forwards replaced by packed QKV integration."""
    restored = 0
    for module in model.modules():
        original = getattr(module, "_mxfp4_qkv_original_forward", None)
        if original is None:
            continue
        module.forward = original
        for attr in (
            "_mxfp4_qkv_original_forward",
            "_mxfp4_qkv_module_refs",
            "_mxfp4_qkv_forward_identities",
            "_mxfp4_qkv_weight_shapes",
            "_mxfp4_qkv_rows",
            "_mxfp4_qkv_name",
            "_mxfp4_qkv_warned",
            "_mxfp4_w_cache",
            "_mxfp4_w_cache_version",
            "_mxfp4_w_cache_sources",
            "_mxfp4_w_cache_metadata",
        ):
            if hasattr(module, attr):
                delattr(module, attr)
        restored += 1
    return restored


def _swiglu_static_ineligibility(mlp):
    projections = tuple(
        getattr(mlp, name, None) for name in ("gate_proj", "up_proj")
    )
    if any(not isinstance(projection, nn.Linear) for projection in projections):
        return "gate_proj and up_proj must be nn.Linear modules"
    if any(
        not getattr(projection, "_quant_enabled", False)
        for projection in projections
    ):
        return "both projections must already be quantized"
    if any(
        getattr(projection, "_quant_scaling_type", None) != "mxfp4"
        for projection in projections
    ):
        return "both projections must use MXFP4"

    activation = getattr(mlp, "act_fn", None)
    activation_type = type(activation)
    if not (
        isinstance(activation, nn.SiLU)
        or (
            activation_type.__module__ == "transformers.activations"
            and activation_type.__name__ == "SiLUActivation"
        )
    ):
        return "split SwiGLU requires SiLU activation"

    from lumen.ops.fused_swiglu import _probe_aiter_swiglu_split

    if not _probe_aiter_swiglu_split():
        return "AITER eager-compatible split SwiGLU kernels are unavailable"
    return None


def _swiglu_runtime_ineligibility(mlp, _input_tensor):
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if gate is not mlp._mxfp4_swiglu_gate_ref() or up is not mlp._mxfp4_swiglu_up_ref():
        return "gate/up modules were replaced after split SwiGLU was enabled"
    reason = _swiglu_static_ineligibility(mlp)
    if reason is not None:
        return reason
    return None


def _warn_swiglu_fallback(mlp, reason):
    warned = getattr(mlp, "_mxfp4_swiglu_warned", None)
    if warned is None:
        warned = set()
        mlp._mxfp4_swiglu_warned = warned
    if reason in warned:
        return
    warned.add(reason)
    logger.warning(
        "MXFP4 split SwiGLU disabled for %s: %s; using the original Qwen3MLP forward",
        getattr(mlp, "_mxfp4_swiglu_name", type(mlp).__name__),
        reason,
    )


def enable_mxfp4_qwen_swiglu(model: nn.Module, *, strict: bool = False) -> int:
    """Fuse eligible Qwen3 SwiGLU activations without replacing Parameters."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3MLP,
        )
    except ImportError as error:
        raise ImportError("MXFP4 split SwiGLU requires transformers Qwen3") from error

    modules = dict(model.named_modules())
    qwen_mlps = tuple(
        (name, module) for name, module in modules.items() if isinstance(module, Qwen3MLP)
    )

    def _is_marked_for_mxfp4(mlp):
        return any(
            getattr(projection, "_quant_enabled", False)
            or getattr(projection, "_quant_scaling_type", None) == "mxfp4"
            for projection in (
                getattr(mlp, "gate_proj", None),
                getattr(mlp, "up_proj", None),
            )
        )

    def _owner_reason(name, mlp):
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        if (
            not isinstance(parent, Qwen3DecoderLayer)
            or getattr(parent, "mlp", None) is not mlp
        ):
            return "MLP is not owned by a Qwen3 decoder layer"
        if hasattr(mlp, "_mxfp4_gate_up_original_forward"):
            return "packed gate/up and split SwiGLU are mutually exclusive"
        return _swiglu_static_ineligibility(mlp)

    if strict:
        requested = tuple(
            (name, mlp) for name, mlp in qwen_mlps if _is_marked_for_mxfp4(mlp)
        )
        eligible = 0
        for name, mlp in requested:
            reason = _owner_reason(name, mlp)
            if reason is None:
                eligible += 1
            else:
                _warn_swiglu_fallback(mlp, reason)
        if not requested or eligible != len(requested):
            raise RuntimeError(
                "MXFP4 split SwiGLU was explicitly requested but can enable on "
                f"{eligible} of {len(requested)} marked Qwen3 MLPs"
            )

    patched = 0
    for name, mlp in qwen_mlps:
        if hasattr(mlp, "_mxfp4_swiglu_original_forward"):
            patched += 1
            continue
        reason = _owner_reason(name, mlp)
        if reason is not None:
            _warn_swiglu_fallback(mlp, reason)
            continue

        original_forward = mlp.forward
        mlp._mxfp4_swiglu_original_forward = original_forward
        mlp._mxfp4_swiglu_gate_ref = weakref.ref(mlp.gate_proj)
        mlp._mxfp4_swiglu_up_ref = weakref.ref(mlp.up_proj)
        mlp._mxfp4_swiglu_name = name

        def fused_forward(input_tensor, *args, _mlp=mlp, **kwargs):
            if args or kwargs:
                reason = "unexpected Qwen3MLP forward arguments"
            else:
                reason = _swiglu_runtime_ineligibility(_mlp, input_tensor)
            if reason is not None:
                _warn_swiglu_fallback(_mlp, reason)
                return _mlp._mxfp4_swiglu_original_forward(
                    input_tensor, *args, **kwargs
                )

            from lumen.ops.fused_swiglu import split_swiglu

            gate = _mlp.gate_proj(input_tensor)
            up = _mlp.up_proj(input_tensor)
            return _mlp.down_proj(split_swiglu(gate, up))

        mlp.forward = fused_forward
        patched += 1
    return patched


def disable_mxfp4_qwen_swiglu(model: nn.Module) -> int:
    """Restore Qwen3MLP forwards replaced by split SwiGLU integration."""
    restored = 0
    for module in model.modules():
        original = getattr(module, "_mxfp4_swiglu_original_forward", None)
        if original is None:
            continue
        module.forward = original
        for attr in (
            "_mxfp4_swiglu_original_forward",
            "_mxfp4_swiglu_gate_ref",
            "_mxfp4_swiglu_up_ref",
            "_mxfp4_swiglu_name",
            "_mxfp4_swiglu_warned",
        ):
            if hasattr(module, attr):
                delattr(module, attr)
        restored += 1
    return restored


def _gate_up_static_ineligibility(mlp):
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if type(gate) is not nn.Linear or type(up) is not nn.Linear:
        return "gate_proj and up_proj must be plain nn.Linear modules"
    if _has_module_call_hooks(gate) or _has_module_call_hooks(up):
        return "gate/up module hooks would be bypassed by the packed projection"
    if not getattr(gate, "_quant_enabled", False) or not getattr(
        up, "_quant_enabled", False
    ):
        return "both projections must already be quantized"
    if (
        getattr(gate, "_quant_scaling_type", None) != "mxfp4"
        or getattr(up, "_quant_scaling_type", None) != "mxfp4"
    ):
        return "both projections must use MXFP4"
    if getattr(gate, "_quant_manager", None) is not getattr(up, "_quant_manager", None):
        return "gate/up projections must share one scaling manager"
    if not getattr(gate, "_lumen_quantize_activation", False) or not getattr(
        up, "_lumen_quantize_activation", False
    ):
        return "packed gate/up requires activation quantization on both projections"
    if getattr(gate, "_lumen_fp8_dtype", None) != getattr(
        up, "_lumen_fp8_dtype", None
    ) or getattr(gate, "_lumen_block_size", 32) != getattr(up, "_lumen_block_size", 32):
        return "gate/up quantization settings differ"
    if gate.bias is not None or up.bias is not None:
        return "biased gate/up projections are unsupported"
    if gate.weight.shape != up.weight.shape:
        return "gate/up weight shapes differ"
    if gate.weight.dtype != up.weight.dtype or gate.weight.device != up.weight.device:
        return "gate/up weights must share dtype and device"
    if gate.weight.requires_grad != up.weight.requires_grad:
        return "gate/up weights must have compatible requires_grad values"
    if gate.weight.dim() != 2 or any(dim % 32 for dim in gate.weight.shape):
        return "gate/up dimensions must be multiples of 32"
    if getattr(gate, "delay_wgrad", False) or getattr(up, "delay_wgrad", False):
        return "delay_wgrad is unsupported"
    if getattr(gate, "gradient_accumulation_fusion", False) or getattr(
        up, "gradient_accumulation_fusion", False
    ):
        return "gradient accumulation fusion is unsupported"
    if os.environ.get("LUMEN_MXFP4_DGRAD_HADAMARD", "0") == "1":
        return "LUMEN_MXFP4_DGRAD_HADAMARD=1 is unsupported"
    activation = getattr(mlp, "act_fn", None)
    activation_type = type(activation)
    if not (
        isinstance(activation, nn.SiLU)
        or (
            activation_type.__module__ == "transformers.activations"
            and activation_type.__name__ == "SiLUActivation"
        )
    ):
        return "packed gate/up requires SiLU activation"

    from lumen.ops.fused_swiglu import _probe_aiter_swiglu_split

    if not _probe_aiter_swiglu_split():
        return "AITER eager-compatible split SwiGLU kernels are unavailable"
    return None


def _gate_up_runtime_ineligibility(mlp, input_tensor):
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if gate is not mlp._mxfp4_gate_module_ref() or up is not mlp._mxfp4_up_module_ref():
        return "gate/up modules were replaced after packing was enabled"
    if type(gate) is not nn.Linear or type(up) is not nn.Linear:
        return "an individually wrapped gate/up projection would bypass its hooks"
    reason = _gate_up_static_ineligibility(mlp)
    if reason is not None:
        return reason
    if tuple(gate.weight.shape) != mlp._mxfp4_gate_up_weight_shape:
        return "gate/up weights are not currently full gathered matrices"
    if input_tensor.shape[-1] != gate.weight.shape[1]:
        return "input width does not match the packed gate/up weights"

    from lumen.ops.quantize.linear import _mxfp4_shape_operands_fit_int32

    if not _mxfp4_shape_operands_fit_int32(
        input_tensor,
        n_out=gate.weight.shape[0] + up.weight.shape[0],
        k_in=gate.weight.shape[1],
    ):
        return "packed projection operands exceed AITER's 32-bit index limit"
    return None


def _warn_gate_up_fallback(mlp, reason):
    warned = getattr(mlp, "_mxfp4_gate_up_warned", None)
    if warned is None:
        warned = set()
        mlp._mxfp4_gate_up_warned = warned
    if reason in warned:
        return
    warned.add(reason)
    logger.warning(
        "MXFP4 packed gate/up disabled for %s: %s; using the original Qwen3MLP forward",
        getattr(mlp, "_mxfp4_gate_up_name", type(mlp).__name__),
        reason,
    )


def enable_mxfp4_qwen_gate_up(model: nn.Module, *, strict: bool = False) -> int:
    """Pack eligible Qwen3 gate/up projections without replacing Parameters.

    ``strict`` requires every Qwen MLP marked for MXFP4 to use the packed path,
    so an explicitly requested performance run cannot silently benchmark a
    partially enabled candidate.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3MLP,
        )
    except ImportError as error:
        raise ImportError(
            "MXFP4 gate/up packing requires transformers Qwen3"
        ) from error

    modules = dict(model.named_modules())
    qwen_mlps = tuple(
        (name, module)
        for name, module in modules.items()
        if isinstance(module, Qwen3MLP)
    )

    def _is_marked_for_mxfp4(mlp):
        return any(
            getattr(projection, "_quant_enabled", False)
            or getattr(projection, "_quant_scaling_type", None) == "mxfp4"
            for projection in (
                getattr(mlp, "gate_proj", None),
                getattr(mlp, "up_proj", None),
            )
        )

    if strict:
        requested = tuple(
            (name, mlp) for name, mlp in qwen_mlps if _is_marked_for_mxfp4(mlp)
        )
        eligible = 0
        for name, mlp in requested:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = modules.get(parent_name)
            reason = None
            if (
                not isinstance(parent, Qwen3DecoderLayer)
                or getattr(parent, "mlp", None) is not mlp
            ):
                reason = "MLP is not owned by a Qwen3 decoder layer"
            elif hasattr(mlp, "_mxfp4_gate_up_original_forward") and (
                getattr(mlp, "gate_proj", None)
                is not mlp._mxfp4_gate_module_ref()
                or getattr(mlp, "up_proj", None) is not mlp._mxfp4_up_module_ref()
            ):
                reason = "gate/up modules were replaced after packing was enabled"
            else:
                reason = _gate_up_static_ineligibility(mlp)
            if reason is None:
                eligible += 1
            else:
                _warn_gate_up_fallback(mlp, reason)
        if not requested or eligible != len(requested):
            raise RuntimeError(
                "MXFP4 packed gate/up was explicitly requested but can enable on "
                f"{eligible} of {len(requested)} marked Qwen3 MLPs"
            )

    patched = 0
    for name, mlp in qwen_mlps:
        if hasattr(mlp, "_mxfp4_gate_up_original_forward"):
            patched += 1
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        if (
            not isinstance(parent, Qwen3DecoderLayer)
            or getattr(parent, "mlp", None) is not mlp
        ):
            _warn_gate_up_fallback(mlp, "MLP is not owned by a Qwen3 decoder layer")
            continue
        reason = _gate_up_static_ineligibility(mlp)
        if reason is not None:
            _warn_gate_up_fallback(mlp, reason)
            continue

        gate = mlp.gate_proj
        up = mlp.up_proj
        original_forward = mlp.forward
        mlp._mxfp4_gate_up_original_forward = original_forward
        mlp._mxfp4_gate_module_ref = weakref.ref(gate)
        mlp._mxfp4_up_module_ref = weakref.ref(up)
        mlp._mxfp4_gate_up_weight_shape = tuple(gate.weight.shape)
        mlp._mxfp4_gate_up_name = name

        def packed_forward(input_tensor, *args, _mlp=mlp, **kwargs):
            if args or kwargs:
                reason = "unexpected Qwen3MLP forward arguments"
            else:
                reason = _gate_up_runtime_ineligibility(_mlp, input_tensor)
            if reason is not None:
                _warn_gate_up_fallback(_mlp, reason)
                return _mlp._mxfp4_gate_up_original_forward(
                    input_tensor, *args, **kwargs
                )

            gate_module = _mlp.gate_proj
            up_module = _mlp.up_proj
            from lumen.ops.quantize.linear import mxfp4_gate_up_linear
            from lumen.ops.fused_swiglu import packed_swiglu
            from lumen.quantize import _mxfp4_cached_weight_pair

            packed_weight, packed_scale = _mxfp4_cached_weight_pair(
                _mlp,
                gate_module.weight,
                up_module.weight,
                gate_module._lumen_fp8_dtype,
                getattr(gate_module, "_lumen_block_size", 32),
                gemm_rows=input_tensor.numel() // input_tensor.shape[-1],
            )
            projected = mxfp4_gate_up_linear(
                input_tensor,
                gate_module.weight,
                up_module.weight,
                packed_weight,
                packed_scale,
                scaling_manager=gate_module._quant_manager,
                fp8_dtype=gate_module._lumen_fp8_dtype,
                block_size=getattr(gate_module, "_lumen_block_size", 32),
                tensor_id=f"{_mlp._mxfp4_gate_up_name}.gate_up.weight",
            )

            return _mlp.down_proj(packed_swiglu(projected))

        mlp.forward = packed_forward
        patched += 1
    if patched:
        from lumen.quantize import _register_mxfp4_load_state_dict_hooks

        _register_mxfp4_load_state_dict_hooks(model)
    return patched


def disable_mxfp4_qwen_gate_up(model: nn.Module) -> int:
    """Restore Qwen3MLP forwards replaced by :func:`enable_mxfp4_qwen_gate_up`."""
    restored = 0
    for module in model.modules():
        original = getattr(module, "_mxfp4_gate_up_original_forward", None)
        if original is None:
            continue
        module.forward = original
        for attr in (
            "_mxfp4_gate_up_original_forward",
            "_mxfp4_gate_module_ref",
            "_mxfp4_up_module_ref",
            "_mxfp4_gate_up_weight_shape",
            "_mxfp4_gate_up_name",
            "_mxfp4_gate_up_warned",
            "_mxfp4_pair_quant_fallback_warned",
            "_mxfp4_w_cache",
            "_mxfp4_w_cache_version",
            "_mxfp4_w_cache_sources",
        ):
            if hasattr(module, attr):
                delattr(module, attr)
        restored += 1
    return restored


__all__ = [
    "disable_mxfp4_qwen_gate_up",
    "disable_mxfp4_qwen_qkv",
    "disable_mxfp4_qwen_swiglu",
    "enable_mxfp4_qwen_gate_up",
    "enable_mxfp4_qwen_qkv",
    "enable_mxfp4_qwen_swiglu",
]

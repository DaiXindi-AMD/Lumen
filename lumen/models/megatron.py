###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared Megatron-LM training components for Lumen models.

This module consolidates the building blocks that are common to all
Megatron-LM-based training scripts (LLaMA2 SFT, LLaMA 3.1 pretraining, etc.).

Model-specific code (batch construction, dataset providers, model-specific CLI
arguments) remains in the per-model subpackages.
"""

import logging
import os
from functools import partial
from typing import Callable, Optional

import torch

# ---------------------------------------------------------------------------
# Megatron compatibility patches (must run before any Megatron model imports)
# ---------------------------------------------------------------------------
from lumen.models.megatron_patches import install_all as _install_megatron_patches
from lumen.quantize.descriptor import FP8Descriptor

_install_megatron_patches()


from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from lumen.models.training_contract import add_shared_checkpoint_args, add_shared_experiment_args
from lumen.models.utils import safe_add_argument
from lumen.modules.attention_megatron import (
    LumenDotProductAttention,
)
from lumen.modules.attention_mla import LumenDotProductAttentionMLA

logger = logging.getLogger(__name__)


def _patch_moe_fused_router():
    """Monkey-patch Megatron-Core's moe_utils with Lumen fused router ops."""
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils

        from lumen.ops.moe.fused_router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )

        moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
        moe_utils.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
        moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

        try:
            import megatron.core.extensions.transformer_engine as te_ext

            te_ext.fused_topk_with_score_function = fused_topk_with_score_function
            te_ext.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
            te_ext.fused_moe_aux_loss = fused_moe_aux_loss
        except ImportError:
            pass

        logger.info("Patched Megatron-Core moe_utils with Lumen fused router ops")
    except ImportError:
        logger.debug("Megatron-Core moe_utils not found, skipping MoE router patch")


_patch_moe_fused_router()

stimer = StragglerDetector()


# ---------------------------------------------------------------------------
# Layer-spec patching helpers
# ---------------------------------------------------------------------------


def _patch_core_attention(spec):
    """Recursively walk a ModuleSpec tree and replace every ``core_attention``
    submodule with ``LumenDotProductAttention``."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(module=LumenDotProductAttention)
        if hasattr(subs, "layer_specs"):
            for layer_spec in subs.layer_specs:
                _patch_core_attention(layer_spec)


class _MegatronCompatibleTLRMSNorm(torch.nn.Module):
    """Wrapper that adapts :class:`LumenRMSNorm` to the Megatron-Core
    norm construction signature ``(config, hidden_size, eps=...)``.

    Unlike ``WrappedTorchNorm``, this does **not** assert on
    ``persist_layer_norm`` or ``sequence_parallel`` — RMSNorm normalises
    over the hidden dimension only, so each position is independent and
    works correctly with SP-scattered inputs.
    """

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        from lumen.ops.normalization import LumenRMSNorm

        self._norm = LumenRMSNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class _MegatronCompatibleTLLayerNorm(torch.nn.Module):
    """Wrapper that adapts :class:`LumenLayerNorm` to the Megatron-Core
    norm construction signature ``(config, hidden_size, eps=...)``."""

    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        from lumen.ops.normalization import LumenLayerNorm

        self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


class _MegatronCompatibleTLNorm(torch.nn.Module):
    """Auto-detect norm type from Megatron config and dispatch."""

    def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        norm_type = getattr(config, "normalization", "LayerNorm")
        if norm_type == "RMSNorm":
            from lumen.ops.normalization import LumenRMSNorm

            self._norm = LumenRMSNorm(hidden_size, eps=eps)
        else:
            from lumen.ops.normalization import LumenLayerNorm

            self._norm = LumenLayerNorm(hidden_size, eps=eps)
        self.weight = self._norm.weight

    def forward(self, x):
        return self._norm(x)


_NORM_ATTRS = (
    "input_layernorm",
    "pre_mlp_layernorm",
    "pre_cross_attn_layernorm",
    "post_cross_attn_layernorm",
    "final_layernorm",
)


def _patch_norms_in_spec(spec, norm_cls=None):
    """Replace **all** norm classes in a spec tree with Lumen norm modules.

    When *norm_cls* is ``None``, uses :class:`_MegatronCompatibleTLNorm`
    which auto-detects RMSNorm vs LayerNorm from the Megatron config.

    Handles both:
    - Block-level specs (``TransformerBlockSubmodules`` with
      ``layer_specs`` and ``final_layernorm``)
    - Layer-level specs (``ModuleSpec`` with ``submodules``)

    IdentityOp placeholders (used for disabled modules like cross-attention
    norms in decoder-only models) are preserved and never replaced.
    """
    from megatron.core.transformer.identity_op import IdentityOp

    if norm_cls is None:
        norm_cls = _MegatronCompatibleTLNorm

    for attr in _NORM_ATTRS:
        cur = getattr(spec, attr, None)
        if cur is not None and cur is not IdentityOp:
            setattr(spec, attr, norm_cls)

    if hasattr(spec, "submodules") and spec.submodules is not None:
        for attr in _NORM_ATTRS:
            cur = getattr(spec.submodules, attr, None)
            if cur is not None and cur is not IdentityOp:
                setattr(spec.submodules, attr, norm_cls)

    layer_specs = getattr(spec, "layer_specs", None)
    if layer_specs is None and hasattr(spec, "submodules"):
        layer_specs = getattr(spec.submodules, "layer_specs", None)
    if layer_specs:
        for layer_spec in layer_specs:
            _patch_norms_in_spec(layer_spec, norm_cls)


def _patch_rmsnorm(model, grad_quant_type=None):
    """Replace all Megatron-Core RMSNorm modules with Lumen's
    Triton-accelerated :class:`LumenRMSNorm`."""
    from lumen.ops.normalization import LumenRMSNorm

    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "RMSNorm",
                "MegatronRMSNorm",
                "TENorm",
                "_MegatronCompatibleTLRMSNorm",
                "_MegatronCompatibleTLNorm",
            ):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-6))
                replacement = LumenRMSNorm(
                    hidden_size,
                    eps=eps,
                    grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                setattr(module, attr_name, replacement)
                count += 1

    print_rank_0(f"> Replaced {count} RMSNorm modules with LumenRMSNorm")


def _patch_layernorm(model, grad_quant_type=None):
    """Replace all Megatron-Core LayerNorm modules with Lumen's
    :class:`LumenLayerNorm`."""
    from lumen.ops.normalization import LumenLayerNorm

    count = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in (
                "LayerNorm",
                "FusedLayerNorm",
                "WrappedTorchNorm",
                "_MegatronCompatibleTLLayerNorm",
                "_MegatronCompatibleTLNorm",
            ):
                hidden_size = child.weight.shape[0]
                eps = getattr(child, "eps", getattr(child, "epsilon", 1e-5))
                replacement = LumenLayerNorm(
                    hidden_size,
                    eps=eps,
                    grad_quant_type=grad_quant_type,
                )
                replacement.weight.data.copy_(child.weight.data)
                if hasattr(child, "bias") and child.bias is not None and replacement.bias is not None:
                    replacement.bias.data.copy_(child.bias.data)
                setattr(module, attr_name, replacement)
                count += 1

    print_rank_0(f"> Replaced {count} LayerNorm modules with LumenLayerNorm")


def _patch_all_norms(model, normalization="RMSNorm", grad_quant_type=None):
    """Replace all norm modules with the appropriate Lumen implementation."""
    if normalization == "RMSNorm":
        _patch_rmsnorm(model, grad_quant_type)
    else:
        _patch_layernorm(model, grad_quant_type)


# ---------------------------------------------------------------------------
# Override defaults for Lumen
# ---------------------------------------------------------------------------

_TE_FORCE_OVERRIDES = {
    "transformer_impl": "local",
    "fp8_param_gather": False,
    "keep_fp8_weight_transpose_cache": False,
    "deprecated_keep_fp8_weight_transpose_cache": False,
    "fp4": None,
    "fp4_param": False,
    "te_rng_tracker": False,
    "inference_rng_tracker": False,
}

_FP8_FORMAT_MAP = {"e4m3": "fp8_e4m3", "hybrid": "hybrid"}

_BACKEND_MAP = {
    "auto": ("aiter_csrc", "aiter_triton_fp8"),
    "triton": ("aiter_triton", "aiter_triton_fp8"),
    "csrc": ("aiter_csrc", "aiter_csrc_fp8"),
    "asm": ("aiter_csrc", "aiter_asm_fp8"),
}


def resolve_attn_backend(backend: str, fp8_attn: str) -> str:
    """Derive the concrete ``aiter_*`` backend string from user-facing flags.

    Args:
        backend: One of ``auto``, ``triton``, ``csrc``, ``asm``.
        fp8_attn: One of ``none``, ``dpa``, ``mha``.

    Returns:
        A concrete backend name like ``aiter_triton_fp8``.
    """
    bf16_be, fp8_be = _BACKEND_MAP.get(backend, ("aiter_csrc", "aiter_triton_fp8"))
    return fp8_be if fp8_attn in ("dpa", "mha") else bf16_be


def _override_te_args_for_lumen(args):
    """Configure Lumen FP8 settings from Megatron args.

    The ``--fp8-format`` value (``args.fp8``) is mapped to the Lumen
    :class:`QuantFormat` string and stored as ``args.lumen_fp8_format`` for
    :func:`apply_fp8_training`.  ``args.fp8`` is then set to ``None`` so
    that ``TransformerConfig`` uses Lumen's own FP8 code-paths.

    All other shared parameters (``fp8_margin``, ``fp8_recipe``,
    ``fp8_amax_history_len``, ``fp8_amax_compute_algo``, ``fp8_wgrad``,
    ``first_last_layers_bf16``, etc.) are kept as-is.
    """
    te_fp8 = getattr(args, "fp8", None)
    if te_fp8 is not None:
        args.lumen_fp8_format = _FP8_FORMAT_MAP.get(te_fp8, te_fp8)
    args.fp8 = None

    for attr, value in _TE_FORCE_OVERRIDES.items():
        setattr(args, attr, value)

    fp8_attn = getattr(args, "lumen_fp8_attn", "none")
    if getattr(args, "fp8_multi_head_attention", False):
        fp8_attn = "mha"
    elif getattr(args, "fp8_dot_product_attention", False) and fp8_attn == "none":
        fp8_attn = "dpa"
    args.lumen_fp8_attn = fp8_attn

    backend_base = getattr(args, "lumen_attn_backend", "auto")
    args.lumen_attn_backend = resolve_attn_backend(backend_base, fp8_attn)

    if getattr(args, "lumen_cross_entropy", False):
        _patch_cross_entropy()


_cross_entropy_patched = False


def _patch_cross_entropy():
    """Monkey-patch Megatron's cross-entropy so the GPTModel loss computation
    goes through Lumen's Triton kernel.

    ``LanguageModule.compute_language_model_loss`` dispatches through
    ``te_parallel_cross_entropy`` when ``cross_entropy_loss_fusion`` is
    enabled.  We replace that symbol with ``lumen_parallel_cross_entropy``
    (same signature) and also wrap it as ``vocab_parallel_cross_entropy``
    for the non-fusion fallback path.
    """
    global _cross_entropy_patched
    if _cross_entropy_patched:
        return

    from lumen.modules.cross_entropy import lumen_parallel_cross_entropy

    try:
        import megatron.core.models.common.language_module.language_module as _lm_mod

        _lm_mod.te_parallel_cross_entropy = lumen_parallel_cross_entropy
    except (ImportError, AttributeError):
        pass

    try:
        import megatron.core.extensions.transformer_engine as _te_ext

        _te_ext.te_parallel_cross_entropy = lumen_parallel_cross_entropy
    except (ImportError, AttributeError):
        pass

    def _vocab_parallel_ce_adapter(logits, labels, label_smoothing=0.0):
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        return lumen_parallel_cross_entropy(logits, labels, tp_group)

    try:
        import megatron.core.tensor_parallel as _tp_mod
        import megatron.core.tensor_parallel.cross_entropy as _ce_mod

        _tp_mod.vocab_parallel_cross_entropy = _vocab_parallel_ce_adapter
        _ce_mod.vocab_parallel_cross_entropy = _vocab_parallel_ce_adapter
    except (ImportError, AttributeError):
        pass

    _cross_entropy_patched = True
    print_rank_0("> Patched cross-entropy with Lumen Triton kernel")


# ---------------------------------------------------------------------------
# Custom GPT builder that injects Lumen attention
# ---------------------------------------------------------------------------


def lumen_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, model_name="GPT"):
    """Build a GPTModel with Lumen attention replacing the default
    DotProductAttention in every layer.

    When ``--lumen-linear`` is set, uses the Lumen spec provider for all
    linear, norm, and attention modules.  Otherwise, uses the Megatron-Core
    local spec and patches attention/norms post-hoc.

    Args:
        model_name: Label used in the startup log message (e.g. ``"LLaMA 3.1"``).
    """
    if getattr(args, "lumen_linear", False):
        return lumen_gpt_builder_with_spec(
            args,
            pre_process,
            post_process,
            vp_stage=vp_stage,
            config=config,
            model_name=model_name,
        )

    print_rank_0(f"building {model_name} model with Lumen attention ...")

    _override_te_args_for_lumen(args)

    if config is None:
        args.apply_rope_fusion = getattr(args, "lumen_fused_rope", False)
        config = core_transformer_config_from_args(args)
        config.persist_layer_norm = False
        config.bias_swiglu_fusion = False
        if getattr(args, "lumen_fp8_activation_store", False):
            config.activation_func_fp8_input_store = True

    transformer_layer_spec = get_gpt_layer_local_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
        use_kitchen=config.use_kitchen,
    )

    _patch_core_attention(transformer_layer_spec)
    _patch_norms_in_spec(transformer_layer_spec)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    grad_quant_type = getattr(args, "grad_quant_type", None)
    normalization = getattr(args, "normalization", "RMSNorm")

    if getattr(args, "lumen_rmsnorm", False) or getattr(args, "lumen_norm", False):
        _patch_all_norms(model, normalization, grad_quant_type)

    return model


def lumen_gpt_builder_with_spec(args, pre_process, post_process, vp_stage=None, config=None, model_name="GPT"):
    """Build a GPTModel using the Lumen spec provider.

    Instead of patching individual modules post-hoc, this builder uses
    :class:`~lumen.models.spec_provider.LumenSpecProvider` to produce a
    layer spec where *all* linear, norm, and attention modules are Lumen
    classes from the start.  This is the recommended path for full Lumen
    integration including FP8 parallel linear layers.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    from lumen.models.spec_provider import LumenSpecProvider

    print_rank_0(f"building {model_name} model with Lumen spec provider ...")

    _override_te_args_for_lumen(args)

    if config is None:
        args.apply_rope_fusion = getattr(args, "lumen_fused_rope", False)
        config = core_transformer_config_from_args(args)
        config.persist_layer_norm = False
        config.bias_swiglu_fusion = False

    # Monkey-patch the TE spec provider lookup so get_gpt_layer_with_transformer_engine_spec
    # picks up Lumen modules without modifying Megatron source.
    import megatron.core.models.gpt.gpt_layer_specs as _gls

    _orig_te_spec = getattr(_gls, "TESpecProvider", None)
    _gls.TESpecProvider = LumenSpecProvider
    _gls.HAVE_TE = True

    try:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=getattr(args, "num_experts", None),
            moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
            qk_layernorm=getattr(args, "qk_layernorm", False),
            multi_latent_attention=getattr(args, "multi_latent_attention", False),
            moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
            use_kitchen=getattr(config, "use_kitchen", False),
        )
    finally:
        if _orig_te_spec is not None:
            _gls.TESpecProvider = _orig_te_spec

    _subs = transformer_layer_spec.submodules
    _existing_map = getattr(_subs, "sharded_state_dict_keys_map", None) or {}
    _existing_map.update(
        {
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        }
    )
    _subs.sharded_state_dict_keys_map = _existing_map

    # Patch MLA attention if needed
    # (note: checkpoint key remapping for legacy checkpoints is handled
    # via _install_layernorm_linear_ckpt_hook below)
    if getattr(args, "multi_latent_attention", False):
        _patch_mla_attention(transformer_layer_spec)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )

    if getattr(args, "lumen_fused_mlp", False):
        _patch_fused_swiglu_mlp(model)

    return model


_LN_KEY_REMAP_CANDIDATES = [
    (
        "input_layernorm.",
        [
            "self_attention.linear_qkv.layer_norm_",
            "self_attention.linear_qkv.base_layer.layer_norm_",
        ],
    ),
    (
        "pre_mlp_layernorm.",
        [
            "mlp.linear_fc1.layer_norm_",
            "mlp.linear_fc1.base_layer.layer_norm_",
        ],
    ),
]


def _install_layernorm_linear_ckpt_hook(model):
    """Remap legacy checkpoint keys for fused LayerNormLinear modules.

    Legacy Megatron checkpoints store norm weights under
    ``decoder.layers.N.input_layernorm.weight`` etc., but the fused
    ``LumenLayerNormLinear`` (used in the TE-style spec) expects them at
    ``decoder.layers.N.self_attention.linear_qkv.layer_norm_weight``.

    When LoRA is applied, the target has an extra ``base_layer.`` segment
    (e.g. ``linear_qkv.base_layer.layer_norm_weight``).
    """

    from megatron.core.transformer.transformer_layer import TransformerLayer

    model_keys = {n for n, _ in model.named_parameters()}
    model_keys |= {n for n, _ in model.named_buffers()}

    first_tl_prefix = None
    for mod_name, mod in model.named_modules():
        if isinstance(mod, TransformerLayer):
            first_tl_prefix = f"{mod_name}." if mod_name else ""
            break

    if first_tl_prefix is None:
        return

    _resolved_map: dict = {}
    for old_pfx, candidates in _LN_KEY_REMAP_CANDIDATES:
        for cand in candidates:
            probe = f"{first_tl_prefix}{cand}weight"
            if probe in model_keys:
                _resolved_map[old_pfx] = cand
                break

    if not _resolved_map:
        return

    def _remap_hook(state_dict, prefix, *_args, **_kwargs):
        for old_pfx, new_pfx in _resolved_map.items():
            for suffix in ("weight", "bias"):
                old_key = f"{prefix}{old_pfx}{suffix}"
                new_key = f"{prefix}{new_pfx}{suffix}"
                if old_key in state_dict and new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

    for module in model.modules():
        if isinstance(module, TransformerLayer):
            module._register_load_state_dict_pre_hook(_remap_hook)


def _patch_fused_swiglu_mlp(model):
    """Patch Megatron MLP forward to use AITER fused SwiGLU when available.

    Replaces the fc1 → SwiGLU → fc2 pipeline with a single AITER Triton
    kernel call (``ff_a16w16_fused_gated``) that fuses gate+up GEMM,
    SiLU activation, element-wise multiply, and down GEMM.

    Only activates when: gated_linear_unit=True, no MLP bias, the AITER
    fused gated kernel is available, AND batch size M <= 64 (the fused
    kernel is slower than decomposed GEMMs for large M).  For training
    with large sequence lengths (M=8192), this will fall back to the
    original path.  Main benefit is for inference or small-batch scenarios.
    """
    from lumen.ops.dispatch import _probe_aiter_fused_gated

    if not _probe_aiter_fused_gated():
        print_rank_0("WARNING: --lumen-fused-mlp requested but AITER fused gated kernel unavailable")
        return

    from megatron.core.transformer.mlp import MLP

    patched = 0
    for module in model.modules():
        if not isinstance(module, MLP):
            continue
        if not getattr(module.config, "gated_linear_unit", False):
            continue
        if getattr(module.config, "add_bias_linear", False):
            continue

        _orig_forward = module.forward

        def _make_fused_forward(mlp_module, orig_fwd):
            _w_down_cache = [None]

            def _fused_forward(hidden_states, per_token_scale=None):
                try:
                    from aiter.ops.triton.gemm.feed_forward import ff_a16w16_fused_gated

                    w_fc1 = mlp_module.linear_fc1.weight
                    w_fc2 = mlp_module.linear_fc2.weight

                    orig_shape = hidden_states.shape
                    x_2d = hidden_states.reshape(-1, orig_shape[-1]).contiguous()

                    M = x_2d.shape[0]
                    if M > 64:
                        return orig_fwd(hidden_states, per_token_scale=per_token_scale)

                    x_bf16 = x_2d.bfloat16() if x_2d.dtype != torch.bfloat16 else x_2d
                    w1_bf16 = w_fc1.bfloat16() if w_fc1.dtype != torch.bfloat16 else w_fc1

                    w2_data = w_fc2.data if not hasattr(w_fc2, "data") else w_fc2
                    w2_bf16 = w2_data.bfloat16() if w2_data.dtype != torch.bfloat16 else w2_data
                    if _w_down_cache[0] is None or _w_down_cache[0].data_ptr() != w2_bf16.data_ptr():
                        _w_down_cache[0] = w2_bf16.t().contiguous()
                    w_down = _w_down_cache[0]

                    out = ff_a16w16_fused_gated(
                        x_bf16,
                        w1_bf16,
                        w_down,
                        dtype=torch.bfloat16,
                        activation="silu",
                    )
                    out = out.reshape(orig_shape[:-1] + (out.shape[-1],))
                    return out, None
                except Exception:
                    return orig_fwd(hidden_states, per_token_scale=per_token_scale)

            return _fused_forward

        module.forward = _make_fused_forward(module, _orig_forward)
        patched += 1

    print_rank_0(f"Patched {patched} MLP modules with AITER fused SwiGLU forward")


def _patch_mla_attention(spec):
    """Replace core_attention with LumenDotProductAttentionMLA in MLA specs."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    if hasattr(spec, "submodules") and spec.submodules is not None:
        subs = spec.submodules
        if hasattr(subs, "self_attention") and subs.self_attention is not None:
            sa = subs.self_attention
            if hasattr(sa, "submodules") and sa.submodules is not None:
                sa_subs = sa.submodules
                if hasattr(sa_subs, "core_attention"):
                    sa_subs.core_attention = ModuleSpec(module=LumenDotProductAttentionMLA)
        if hasattr(subs, "layer_specs"):
            for layer_spec in subs.layer_specs:
                _patch_mla_attention(layer_spec)


def enable_fp8_for_parallel_linear(
    model,
    scaling_manager=None,
    scaling_type="dynamic",
    fp8_dtype=None,
    block_size=None,
    fp8_mha=False,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    quant_config=None,
):
    """Enable FP8 GEMM on all Lumen parallel linear modules in the model.

    When *fp8_mha* is True, a shared :class:`Blockwise2DScaleManager` is
    attached to each ``LumenDotProductAttention`` (or MLA variant) so that
    QKV projection, dot-product attention and output projection share the
    same FP8 scale context within a single MHA block.

    When *delay_wgrad* is True, backward passes compute only dgrad
    immediately and defer wgrad to a later ``backward_dw()`` call.

    When *gradient_accumulation_fusion* is True, weight gradients
    accumulate directly into ``param.main_grad``.

    When *quant_config* is provided, per-module ScalingManagers are created
    with this config (ensuring correct amax_algo, history_len, etc.).
    """
    from lumen.modules.attention_megatron import LumenDotProductAttention
    from lumen.modules.attention_mla import LumenDotProductAttentionMLA
    from lumen.modules.grouped_linear import LumenGroupedLinear
    from lumen.modules.layernorm_linear import LumenLayerNormLinear
    from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear

    if fp8_dtype is None:
        from lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()

    # Tell the fused SwiGLU quant bridge (LUMEN_FUSED_SWIGLU_QUANT) the global
    # activation scale granularity so its cached scale layout matches the fc2
    # GEMM that consumes it (blockwise2d needs a 2D 1×block scale, not 1D).
    from lumen.models._swiglu_fp8_fuse import set_fused_swiglu_scaling

    set_fused_swiglu_scaling(scaling_type, block_size)

    count = 0
    for module in model.modules():
        if isinstance(
            module, (LumenColumnParallelLinear, LumenRowParallelLinear, LumenLayerNormLinear, LumenGroupedLinear)
        ):
            _mgr = scaling_manager
            if _mgr is None and quant_config is not None:
                from lumen.quantize import ScalingManager

                _mgr = ScalingManager(quant_config)
            module.enable_fp8(
                scaling_manager=_mgr,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
            )
            if hasattr(module, "gradient_accumulation_fusion"):
                module.gradient_accumulation_fusion = gradient_accumulation_fusion
            if hasattr(module, "delay_wgrad"):
                module.delay_wgrad = delay_wgrad
            count += 1

    if fp8_mha:
        from lumen.quantize.scaling_manager import Blockwise2DScaleManager

        attn_count = 0
        for module in model.modules():
            if isinstance(module, (LumenDotProductAttention, LumenDotProductAttentionMLA)):
                module.scale_manager = Blockwise2DScaleManager()
                attn_count += 1
        if attn_count > 0:
            print_rank_0(f"> Attached Blockwise2DScaleManager to {attn_count} attention modules for FP8 MHA")

    if count > 0:
        print_rank_0(f"> Enabled FP8 (scaling={scaling_type}) on {count} Lumen parallel linear modules")


# ---------------------------------------------------------------------------
# LoRA (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------------------------


def _patch_lora_for_layernorm_linear(model):
    """Fix LoRA input for LumenLayerNormLinear base layers.

    When a LumenLayerNormLinear is wrapped by LoRA, the LoRA adapter's
    forward passes the *raw* (pre-norm) input to lora_a. But lora_a
    expects the *normalized* input — matching what a standalone
    ColumnParallelLinear would receive after a separate layernorm.

    This patch replaces the LoRA adapter forward to retrieve the cached
    normalized output from the base layer's forward (stored in thread-local
    by ``LumenLayerNormLinear.forward``), avoiding a redundant RMSNorm.
    Falls back to recomputing the norm if the cache is empty.
    """
    from megatron.core.transformer.lora_adapter import LoraAdapter

    from lumen.modules.layernorm_linear import LumenLayerNormLinear, _pop_cached_ln_out

    patched = 0
    for module in model.modules():
        if not isinstance(module, LoraAdapter):
            continue
        base = module.base_layer
        if not isinstance(base, LumenLayerNormLinear):
            continue
        if module.lora_a is None:
            continue

        def _make_patched_forward(adapter, base_layer):
            def _patched_forward(input_tensor, *args, **kwargs):
                output = base_layer(input_tensor, *args, **kwargs)
                if adapter.lora_a is None:
                    return output

                normed_input = _pop_cached_ln_out()
                if normed_input is None:
                    normed_input = base_layer._norm(input_tensor)

                lora_a_out, _ = adapter.lora_a(normed_input)
                lora_b_out, _ = adapter.lora_b(lora_a_out)
                lora_drop_out = adapter.lora_dropout(lora_b_out)
                lora_out = adapter.lora_alpha * lora_drop_out

                if type(output) is torch.Tensor:
                    return output + lora_out

                out_tensor, bias = output
                return out_tensor + lora_out, bias

            return _patched_forward

        module.forward = _make_patched_forward(module, base)
        patched += 1

    if patched > 0:
        print_rank_0(
            f"> Patched {patched} LoRA adapters for LumenLayerNormLinear " f"(cached normalized input for lora_a)"
        )


def apply_lora(model: GPTModel, args) -> None:
    """Wrap linear layers with LoRA adapters for parameter-efficient fine-tuning.

    Target modules controlled by ``--lora-target-modules``:

    * ``"attention"`` — QKV + output projection only (NeMo reference).
    * ``"attention_mlp"`` — attention + MLP (gate/up + down).
    * ``"all"`` (default) — attention + MLP + embedding + output layer.
    """
    from megatron.core.transformer.lora_adapter import (
        COLUMN_PARALLEL_LAYERS,
        LORA_LAYERS_MAPPING,
        ROW_PARALLEL_LAYERS,
        LoraAdapter,
    )

    from lumen.modules.layernorm_linear import LumenLayerNormLinear
    from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear

    if LumenLayerNormLinear not in LORA_LAYERS_MAPPING:
        LORA_LAYERS_MAPPING[LumenLayerNormLinear] = COLUMN_PARALLEL_LAYERS
    if LumenColumnParallelLinear not in LORA_LAYERS_MAPPING:
        LORA_LAYERS_MAPPING[LumenColumnParallelLinear] = COLUMN_PARALLEL_LAYERS
    if LumenRowParallelLinear not in LORA_LAYERS_MAPPING:
        LORA_LAYERS_MAPPING[LumenRowParallelLinear] = ROW_PARALLEL_LAYERS

    common = {
        "config": model.config,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }

    target = getattr(args, "lora_target_modules", "all")

    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "decoder") and model.decoder is not None:
        for layer in model.decoder.layers:
            layer.self_attention.linear_qkv = LoraAdapter(layer.self_attention.linear_qkv, **common)
            layer.self_attention.linear_proj = LoraAdapter(layer.self_attention.linear_proj, **common)
            if target in ("all", "attention_mlp") and hasattr(layer, "mlp") and layer.mlp is not None:
                layer.mlp.linear_fc1 = LoraAdapter(layer.mlp.linear_fc1, **common)
                layer.mlp.linear_fc2 = LoraAdapter(layer.mlp.linear_fc2, **common)

    if target == "all":
        if hasattr(model, "embedding") and model.embedding is not None:
            if hasattr(model.embedding, "word_embeddings"):
                model.embedding.word_embeddings = LoraAdapter(model.embedding.word_embeddings, **common)
        if hasattr(model, "output_layer") and model.output_layer is not None:
            model.output_layer = LoraAdapter(model.output_layer, **common)

    _patch_lora_for_layernorm_linear(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"> LoRA applied (rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"target={target}) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )


# ---------------------------------------------------------------------------
# FP8 quantised training
# ---------------------------------------------------------------------------


def apply_fp8_training(model: GPTModel, args) -> None:
    """Enable FP8 quantised training via Lumen's non-invasive patching.

    .. deprecated:: Prefer :meth:`LumenConfig.enable` directly.
    """
    from lumen.config import LumenConfig

    cfg = LumenConfig.from_args(args)

    dp_group = None
    if cfg.reduce_amax:
        import torch.distributed as dist
        from megatron.core import parallel_state

        if dist.is_initialized():
            dp_group = parallel_state.get_data_parallel_group()

    import lumen.quantize as quant

    qcfg = cfg.quant_config
    quant.enable(model, config=qcfg, dp_group=dp_group)
    print_rank_0(
        f"> FP8 training enabled (format={cfg.format}, scaling={cfg.scaling}, "
        f"block_size={cfg.block_size}, amax_algo={cfg.amax_algo}, "
        f"reduce_amax={cfg.reduce_amax}, history={cfg.history_len}, "
        f"activation={cfg.quantize_activation}, grad_quant={cfg.quantize_grad})"
    )


def _find_scaling_manager(model):
    """Retrieve the ScalingManager from quant-patched modules."""
    for module in model.modules():
        sm = getattr(module, "_quant_manager", None)
        if sm is not None:
            return sm
    return None


def _enable_lumen_fp8_checkpoint(scaling_manager):
    """Monkey-patch tensor_parallel.checkpoint to preserve FP8 scaling state."""
    import megatron.core.tensor_parallel as tp_module
    import megatron.core.tensor_parallel.random as tp_random

    from lumen.utils.checkpoint import _FP8ScalingContext

    if hasattr(tp_module, "_lumen_fp8_checkpoint_patched"):
        return

    _original = tp_random.checkpoint

    def _patched(function, distribute_saved_activations, *args):
        ctx = _FP8ScalingContext()
        ctx.save(scaling_manager)
        orig_fn = function

        def wrapped(*a, **kw):
            ctx.restore(scaling_manager)
            return orig_fn(*a, **kw)

        return _original(wrapped, distribute_saved_activations, *args)

    tp_random.checkpoint = _patched
    tp_module.checkpoint = _patched
    tp_module._lumen_fp8_checkpoint_patched = True
    tp_module._lumen_fp8_checkpoint_original = _original
    print_rank_0("> FP8-aware activation checkpointing enabled (Lumen)")


def apply_lumen_pre_quant(model: GPTModel, args) -> None:
    """Phase 1: Set module attributes BEFORE quant.enable() captures them.

    .. deprecated:: Prefer :meth:`LumenConfig.enable` which handles this automatically.
    """
    from lumen.config import LumenConfig

    LumenConfig.from_args(args)._apply_pre_quant(model)


apply_lumen_optimizations = apply_lumen_pre_quant


def apply_lumen_post_quant(model: GPTModel, args) -> None:
    """Phase 2: Features requiring ScalingManager (created by quant.enable).

    .. deprecated:: Prefer :meth:`LumenConfig.enable` which handles this automatically.
    """
    from lumen.config import LumenConfig

    cfg = LumenConfig.from_args(args)
    sm = _find_scaling_manager(model) if (cfg.fp8_checkpoint or cfg.fp8_param_gather) else None
    cfg._apply_post_quant(model, sm)


def get_cpu_offload_context(args):
    """Return CPU offload context manager (no-op if disabled)."""
    from lumen.utils.cpu_offload import lumen_cpu_offload_context

    enabled = getattr(args, "lumen_cpu_offload", False)
    return lumen_cpu_offload_context(enabled=enabled)


def register_fp8_param_optimizer_hook(model, optimizer):
    """Register optimizer post-step hook for FP8 param staleness marking.

    Must be called AFTER optimizer creation (outside model_provider).
    """
    sm = _find_scaling_manager(model)
    if sm is not None and sm.num_fp8_params > 0:
        sm.register_fp8_optimizer_hook(optimizer)
        print_rank_0("> FP8 param optimizer hook registered")


def make_lumen_model_provider(
    model_builder: Callable,
    *,
    lora_applier: Callable = apply_lora,
    fp8_applier: Callable = apply_fp8_training,
):
    """Build the canonical Megatron model-provider assembly for Lumen.

    The returned callable keeps task semantics in the supplied ``model_builder``
    while applying the shared infrastructure assembly in a fixed order:

    1. Megatron LoRA (uses ``megatron.core.transformer.lora_adapter``,
       separate from PEFT — handled by *lora_applier*)
    2. ``LumenConfig.enable()`` — FP8ParamManager, norm patching, pre-quant,
       ``quant.enable``, post-quant (PEFT LoRA skipped via ``lora_rank=0``)
    3. Megatron-specific ``enable_fp8_for_parallel_linear`` (optional)
    """

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        import os
        from dataclasses import replace as _replace

        from lumen.config import LumenConfig

        from dataclasses import replace as _replace

        from lumen.config import LumenConfig

        args = get_args()
        model = model_builder(args, pre_process, post_process, vp_stage)

        # 1. Megatron LoRA (not PEFT — stays separate)
        if getattr(args, "lora_rank", 0) > 0:
            lora_applier(model, args)
            if getattr(args, "lora_a2a", False):
                os.environ["LORA_A2A"] = "1"
                print_rank_0("> LoRA A2A communication optimisation enabled")

        # 1b. Install checkpoint key remapping for fused LayerNormLinear
        if getattr(args, "lumen_linear", False):
            _install_layernorm_linear_ckpt_hook(model)

        # 2. Unified LumenConfig.enable() — skip PEFT LoRA (handled above)
        cfg = LumenConfig.from_args(args)
        cfg = _replace(cfg, lora_rank=0)

        dp_group = None
        if cfg.reduce_amax:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if dist.is_initialized():
                dp_group = parallel_state.get_data_parallel_group()

        _original_model = model
        _manager, model = cfg.enable(model, dp_group=dp_group)
        assert model is _original_model, (
            f"quant.enable() returned a different model object "
            f"(type {type(model).__name__} vs {type(_original_model).__name__}). "
            f"This indicates unexpected model wrapping (e.g. LoRA) that may "
            f"break Megatron's parameter management."
        )

        # 3. Megatron-specific parallel linear FP8 (not covered by LumenConfig)
        if getattr(args, "linear_fp8", False) and getattr(args, "lumen_linear", False):
            scaling_type = getattr(args, "linear_fp8_scaling", "dynamic")
            enable_fp8_for_parallel_linear(
                model,
                scaling_type=scaling_type,
                fp8_mha=getattr(args, "lumen_fp8_attn", "none") == "mha",
                gradient_accumulation_fusion=getattr(args, "lumen_gradient_accumulation_fusion", False),
                delay_wgrad=getattr(args, "lumen_delay_wgrad", False),
                quant_config=cfg.quant_config,
            )

        if getattr(args, "fp8_param_storage", False):
            _shrink_frozen_weights_to_fp8(model)

        return model

    return model_provider


def _shrink_frozen_weights_to_fp8(model) -> None:
    """Tag frozen 2-D weights for FP8 storage.

    At this point the model might still be on meta device or already on CUDA.
    We tag the weights with metadata and install load/forward hooks.  If the
    weight is already materialized on CUDA, we also shrink it to a 1-element
    FP8 placeholder.  If on meta device, we just tag it — the patched
    materializer will create FP8-sized tensors later.
    """
    import torch

    fp8_dtype = torch.float8_e4m3fnuz
    count = 0
    for _name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, torch.nn.Parameter):
            continue
        if w.requires_grad:
            continue
        if w.ndim < 2:
            continue

        orig_shape = w.shape
        orig_dtype = w.dtype
        w._fp8_orig_shape = orig_shape
        w._fp8_original_dtype = orig_dtype
        w._fp8_dtype = fp8_dtype
        w._fp8_storage_enabled = True

        if str(w.device) != "meta":
            device = w.device
            tiny = torch.zeros(1, dtype=fp8_dtype, device=device)
            w.data = tiny

        _wrap_load_from_state_dict(module, fp8_dtype)
        _install_fp8_forward_hooks(module, fp8_dtype)
        count += 1

    print_rank_0(f"> FP8 param storage: tagged {count} frozen weights for FP8 storage")


def install_fp8_param_gather_hook() -> None:
    """Install the canonical Megatron optimizer hook for FP8 param gather."""
    import megatron.training.training as _mt_training

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_fp8_param_gather_hook", False):
        return

    def _setup_with_fp8_hook(*args, **kwargs):
        model, optimizer, scheduler = current_setup(*args, **kwargs)
        train_args = get_args()
        # Register the optimizer post-step hook whenever the FP8 param cache is
        # active — either for FP8 DP gather, or for per-step weight quantization
        # (LUMEN_WEIGHT_QUANT_ONCE). The hook fires mark_fp8_params_stale each
        # optimizer step, the reliable per-step invalidation signal.
        _weight_quant_once = os.environ.get("LUMEN_WEIGHT_QUANT_ONCE", "0") == "1"
        if (getattr(train_args, "lumen_fp8_param_gather", False) or _weight_quant_once) and model:
            target = model[0] if isinstance(model, list) else model
            register_fp8_param_optimizer_hook(target, optimizer)
        return model, optimizer, scheduler

    _setup_with_fp8_hook._lumen_fp8_param_gather_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_fp8_hook


def install_val_loss_early_stop_hook() -> None:
    """Stop training when the *reduced* validation loss reaches val_loss_target.

    Wraps ``megatron.training.training.evaluate`` whose returned
    ``total_loss_dict['lm loss']`` is already averaged AND all-reduced across
    ranks, so it is identical on every rank. When that value <=
    ``args.val_loss_target`` we set ``args.train_iters = args.iteration`` on ALL
    ranks at the same eval point, so the train loop exits collectively. This
    replaces the old per-rank training-loss EMA stop in ``loss_func`` that
    desynced DP ranks (one rank exiting while others kept running -> mismatched
    collectives -> NCCL deadlock).
    """
    try:
        import megatron.training.training as _train_mod
        from megatron.training import get_args
    except ImportError:
        return
    if getattr(_train_mod, "_lumen_val_early_stop_patched", False):
        return

    _orig_evaluate = _train_mod.evaluate

    def _evaluate_with_early_stop(*a, **kw):
        result = _orig_evaluate(*a, **kw)
        try:
            args = get_args()
            target = getattr(args, "val_loss_target", None)
            if target is not None and result and isinstance(result[0], dict):
                v = result[0].get("lm loss")
                if v is not None:
                    val = v.item() if hasattr(v, "item") else float(v)
                    if val <= float(target):
                        cur = getattr(args, "iteration", None)
                        if cur:
                            args.train_iters = cur
                        print_rank_0(
                            f"> [Early Stop] validation loss ({val:.4f}) <= "
                            f"target ({float(target):.4f}) -> stopping at iter {cur}."
                        )
        except Exception as e:  # never let the hook break eval
            print_rank_0(f"> [Early Stop] hook skipped ({e})")
        return result

    _train_mod.evaluate = _evaluate_with_early_stop
    _train_mod._lumen_val_early_stop_patched = True


def install_fp8_param_storage_hook() -> None:
    """Hook the training setup to enable FP8 parameter storage.

    When ``--fp8-param-storage`` is active, this:

    1. Forces ``--init-model-with-meta-device`` so the model skeleton is
       created without allocating GPU memory.
    2. Patches ``to_empty_if_meta_device`` so tagged frozen weights are
       materialized as tiny FP8 placeholders (~0 MB) instead of full-size
       BF16 tensors (~140 GB for 70B).
    3. Patches ``Float16Module.__init__`` so ``.bfloat16()`` skips params
       that are already in FP8.
    4. Patches ``load_checkpoint`` to log FP8 statistics after loading.
    """
    import megatron.training.training as _mt_training

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_fp8_param_storage_hook", False):
        return

    def _setup_with_fp8_storage(*a, **kw):
        train_args = get_args()
        if not getattr(train_args, "fp8_param_storage", False):
            return current_setup(*a, **kw)

        _fmt = (
            getattr(train_args, "lumen_fp8_format", "")
            or getattr(train_args, "fp8", "")
            or getattr(train_args, "linear_fp8_format", "")
        )
        _want_hipblaslt = _fmt == "hybrid" or os.environ.get("LUMEN_PREFER_HIPBLASLT", "0") == "1"
        if _want_hipblaslt:
            os.environ["LUMEN_PREFER_HIPBLASLT"] = "1"
            try:
                import lumen.ops.quantize.linear as _qlinear

                _qlinear._PREFER_HIPBLASLT = True
                ensure_hipblaslt_ready = _qlinear.ensure_hipblaslt_ready
                ensure_hipblaslt_ready()
                _reason = (
                    "LUMEN_PREFER_HIPBLASLT"
                    if os.environ.get("LUMEN_PREFER_HIPBLASLT") == "1"
                    else "hybrid FP8 backward"
                )
                print_rank_0(f"> hipBLASLt workspace pre-allocated for {_reason}")
            except Exception as e:
                print_rank_0(f"> WARNING: hipBLASLt pre-init failed: {e}")

        train_args.init_model_with_meta_device = True
        print_rank_0("> FP8 param storage: forcing init_model_with_meta_device=True")
        _patch_meta_materializer()
        _patch_float16_module()
        _patch_load_checkpoint_for_fp8()
        model, optimizer, scheduler = current_setup(*a, **kw)

        targets = model if isinstance(model, list) else [model]
        for m in targets:
            unwrapped = m
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            _install_embedding_output_fp8_hooks(unwrapped)

        print_rank_0(
            "> FP8 param storage: linear layers handled inline by "
            "FP8StoredLinearFunction (no per-layer forward hooks needed)"
        )

        return model, optimizer, scheduler

    _setup_with_fp8_storage._lumen_fp8_param_storage_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_fp8_storage


def install_hip_graphs_hook() -> None:
    """Hook setup_model_and_optimizer to capture HIP graphs for transformer layers.

    When ``--lumen-hip-graphs`` is active, this wraps each transformer layer's
    forward+backward in pre-captured CUDA/HIP graphs to eliminate kernel launch
    overhead (~30K launches/step reduced to ~10K).

    Must be installed after all other setup hooks (fp8_param_gather, fp8_param_storage)
    so graph capture sees the final model structure.
    """
    import megatron.training.training as _mt_training

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_hip_graphs_hook", False):
        return

    def _setup_with_hip_graphs(*args, **kwargs):
        model, optimizer, scheduler = current_setup(*args, **kwargs)
        train_args = get_args()

        if not getattr(train_args, "lumen_hip_graphs", False):
            return model, optimizer, scheduler

        if not model:
            return model, optimizer, scheduler

        from lumen.utils.hip_graphs import install_lazy_graph_capture

        warmup_steps = getattr(train_args, "warmup_steps", 5)
        num_warmup = max(warmup_steps, 3)

        recompute_num = 0
        if (
            getattr(train_args, "recompute_granularity", None) == "full"
            and getattr(train_args, "recompute_method", None) == "block"
        ):
            recompute_num = getattr(train_args, "recompute_num_layers", 0)

        targets = model if isinstance(model, list) else [model]
        for m in targets:
            unwrapped = m
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            max_graphed = int(os.environ.get("LUMEN_HIP_GRAPHS_MAX_LAYERS", "10"))
            count = install_lazy_graph_capture(
                unwrapped,
                num_warmup=num_warmup,
                skip_recomputed_layers=recompute_num,
                max_graphed_layers=max_graphed,
            )
            if count > 0:
                print_rank_0(
                    f"> HIP graphs: installed lazy capture on {count} "
                    f"transformer layers (skipped {recompute_num} recomputed, "
                    f"capture after step {num_warmup})"
                )

        return model, optimizer, scheduler

    _setup_with_hip_graphs._lumen_hip_graphs_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_hip_graphs



def _patch_meta_materializer() -> None:
    """Replace to_empty_if_meta_device with a version that materializes
    FP8-tagged parameters as tiny 1-element FP8 tensors (saving ~70GB).

    The trick: ``Module._apply`` iterates parameters in order.  We build
    a lookup of which Parameter objects are FP8-tagged, then inside the
    per-tensor callback we look up the enclosing Parameter via the module
    tree to decide whether to shrink it.

    We also directly patch the local name binding in the already-imported
    ``megatron.training.training`` module via ``sys.modules``.
    """
    import sys

    import megatron.training.utils as _mu
    import torch

    _orig_to_empty = _mu.to_empty_if_meta_device
    if getattr(_orig_to_empty, "_fp8_patched", False):
        return

    def _fp8_aware_to_empty(module, *, device, recurse=True):
        fp8_data_map = {}
        for _n, p in module.named_parameters(recurse=recurse):
            if getattr(p, "_fp8_storage_enabled", False):
                fp8_data_map[id(p)] = p

        orig_apply = torch.nn.Module._apply

        def _custom_apply(mod, fn, recurse_inner=True):
            for key, param in mod._parameters.items():
                if param is None:
                    continue
                if id(param) in fp8_data_map:
                    if param.data.device == torch.device("meta"):
                        fp8_dtype = torch.float8_e4m3fnuz
                        new_data = torch.zeros(1, dtype=fp8_dtype, device=device)
                    else:
                        new_data = param.data.to(device)
                    param_out = torch.nn.Parameter(new_data, requires_grad=param.requires_grad)
                    for attr in (
                        "_fp8_storage_enabled",
                        "_fp8_orig_shape",
                        "_fp8_original_dtype",
                        "_fp8_dtype",
                        "_fp8_scale",
                    ):
                        if hasattr(param, attr):
                            setattr(param_out, attr, getattr(param, attr))
                    if (
                        getattr(param_out, "_fp8_scale", None) is not None
                        and getattr(param_out, "_fp8_dtype", None) is not None
                    ):
                        sc = param_out._fp8_scale
                        if torch.is_tensor(sc):
                            sc = sc.to(param_out.device)
                            param_out._fp8_scale = sc
                        param_out._fp8_desc = FP8Descriptor(
                            data=param_out.data,
                            scale=sc,
                            fp8_dtype=param_out._fp8_dtype,
                        )
                    mod._parameters[key] = param_out
                    fp8_data_map[id(param_out)] = param_out
                else:
                    with torch.no_grad():
                        new_data = fn(param.data)
                    if new_data is not param.data:
                        param_out = torch.nn.Parameter(new_data, requires_grad=param.requires_grad)
                        mod._parameters[key] = param_out
            for key, buf in mod._buffers.items():
                if buf is not None:
                    mod._buffers[key] = fn(buf)
            if recurse_inner:
                for child in mod.children():
                    _custom_apply(child, fn, recurse_inner)
            return mod

        def _empty_fn(tensor):
            if tensor.device == torch.device("meta"):
                return torch.empty_like(tensor, device=device)
            return tensor.to(device)

        _custom_apply(module, _empty_fn, recurse)
        torch.nn.Module._apply = orig_apply
        return module

    _fp8_aware_to_empty._fp8_patched = True
    _mu.to_empty_if_meta_device = _fp8_aware_to_empty
    training_mod = sys.modules.get("megatron.training.training")
    if training_mod is not None:
        training_mod.to_empty_if_meta_device = _fp8_aware_to_empty


def _patch_float16_module() -> None:
    """Patch Float16Module.__init__ so .bfloat16() skips FP8-tagged params.

    Float16Module wraps the model via ``module.bfloat16()``, which casts
    every parameter to BF16.  For FP8-tagged weights (tiny placeholders),
    we collect them, let .bfloat16() run, then restore FP8 data and
    re-attach the custom attributes.
    """
    import torch
    from megatron.core.transformer.module import Float16Module

    _orig_init = Float16Module.__init__
    if getattr(_orig_init, "_fp8_patched", False):
        return

    def _fp8_safe_init(self, config, module):
        fp8_info = {}
        for name, mod in module.named_modules():
            for pname, p in mod._parameters.items():
                if p is not None and getattr(p, "_fp8_storage_enabled", False):
                    fp8_info[(name, pname)] = {
                        "data": p.data.clone(),
                        "_fp8_storage_enabled": True,
                        "_fp8_orig_shape": getattr(p, "_fp8_orig_shape", None),
                        "_fp8_original_dtype": getattr(p, "_fp8_original_dtype", None),
                        "_fp8_dtype": getattr(p, "_fp8_dtype", None),
                        "_fp8_scale": getattr(p, "_fp8_scale", None),
                    }

        _orig_init(self, config, module)

        inner = self.module if hasattr(self, "module") else module
        for (mod_name, pname), info in fp8_info.items():
            parts = mod_name.split(".") if mod_name else []
            target = inner
            for part in parts:
                target = getattr(target, part, target)
            p = target._parameters.get(pname)
            if p is not None:
                p.data = info["data"].to(p.device)
                for attr in (
                    "_fp8_storage_enabled",
                    "_fp8_orig_shape",
                    "_fp8_original_dtype",
                    "_fp8_dtype",
                    "_fp8_scale",
                ):
                    if info.get(attr) is not None:
                        setattr(p, attr, info[attr])
                if info.get("_fp8_scale") is not None and info.get("_fp8_dtype") is not None:
                    sc = p._fp8_scale
                    if torch.is_tensor(sc):
                        sc = sc.to(p.device)
                        p._fp8_scale = sc
                    p._fp8_desc = FP8Descriptor(data=p.data, scale=sc, fp8_dtype=info["_fp8_dtype"])

    _fp8_safe_init._fp8_patched = True
    Float16Module.__init__ = _fp8_safe_init


def _get_fp8_store_scaling() -> str:
    """FP8 scaling type for param storage (from Megatron args; default per-tensor)."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "linear_fp8_scaling", "delayed") or "delayed"
    except Exception:
        return "delayed"


def _fp8_store_quantize_weight(weight_bf16, fp8_dtype, scaling_type, block_size: int = 128):
    """Quantize a frozen weight for FP8 param storage.

    Returns ``(fp8_data, scale, transpose)``:
      - ``blockwise2d``: ``scale`` is the 2D ``(ceil(N/bs), ceil(K/bs))`` dequant
        factor consumed directly by ``gemm_blockscale``; ``transpose`` is None
        (the blockscale kernel transposes internally).
      - otherwise: per-tensor scalar quant factor (``fp8_max/amax``) plus a
        precomputed transpose for the hipBLASLt per-tensor path.
    """
    import torch

    if scaling_type == "blockwise2d":
        from lumen.ops.quantize.linear import _quant_blockwise2d_weight

        fp8_data, scale = _quant_blockwise2d_weight(weight_bf16, fp8_dtype, block_size)
        return fp8_data, scale, None

    amax = weight_bf16.float().abs().amax().clamp(min=1e-12)
    scale = torch.finfo(fp8_dtype).max / amax
    fp8_data = (weight_bf16.float() * scale).to(fp8_dtype)
    return fp8_data, scale, _precompute_fp8_transpose(fp8_data)


def _precompute_fp8_transpose(fp8_data: "torch.Tensor") -> "Optional[torch.Tensor]":
    """Pre-compute the transposed layout for an FP8 weight tensor.

    Uses the fast Triton transpose when available, otherwise falls back
    to ``t().contiguous()``.  Called once at checkpoint load time so that
    ``FP8Descriptor.transpose_cached`` never needs to compute it lazily.

    When ``LUMEN_PREFER_HIPBLASLT=1`` we skip the allocation entirely.
    hipBLASLt's C++ kernel (``hipbsolgemm.cu``) detects non-contiguous
    strides from a metadata-only ``.t()`` view and applies ``HIPBLAS_OP_T``
    internally.  ``_gemm_per_tensor_hipblas`` passes ``w.t()`` (zero-cost
    view, no memory copy) directly to ``hipb_mm``.  Storing both NxK and
    KxN would add ~37 GiB on Llama2-70B, causing OOM.
    """
    import os

    if os.environ.get("LUMEN_PREFER_HIPBLASLT", "0") == "1":
        return None
    if fp8_data.dim() == 2 and fp8_data.is_cuda:
        try:
            from lumen.ops.quantize.fast_transpose import fast_transpose_fp8

            return fast_transpose_fp8(fp8_data)
        except (ImportError, OSError, RuntimeError):
            pass
    return fp8_data.t().contiguous()


def _patch_load_checkpoint_for_fp8() -> None:
    """Monkey-patch Megatron's load_checkpoint to convert weights to FP8 after loading.

    Also integrates LoRA base_layer key remapping and mmap loading, so
    external ``patch_checkpointing.py`` is no longer needed.
    """
    import sys

    import megatron.training.checkpointing as _ckpt

    _original_load = _ckpt.load_checkpoint
    if getattr(_original_load, "_fp8_patched", False):
        return

    from lumen.models.megatron_patches import remap_lora_state_dict as _remap_lora_state_dict

    def _load_with_fp8(ddp_model, optimizer, opt_param_scheduler, **kwargs):
        import gc

        import torch

        _orig_module_load_sd = torch.nn.Module.load_state_dict

        def _remap_load_state_dict(self_mod, state_dict, strict=True, **kw):
            state_dict = _remap_lora_state_dict(self_mod, state_dict)
            try:
                return _orig_module_load_sd(self_mod, state_dict, strict=strict, **kw)
            except Exception:
                if strict:
                    return _orig_module_load_sd(self_mod, state_dict, strict=False, **kw)
                raise

        torch.nn.Module.load_state_dict = _remap_load_state_dict
        try:
            result = _original_load(ddp_model, optimizer, opt_param_scheduler, **kwargs)
        finally:
            torch.nn.Module.load_state_dict = _orig_module_load_sd

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print_rank_0(f"> GPU memory right after ckpt load: {alloc:.2f}GB")

        targets = ddp_model if isinstance(ddp_model, list) else [ddp_model]
        fp8_dtype = torch.float8_e4m3fnuz
        converted = 0
        freed_bytes = 0
        already_fp8 = 0
        already_fp8_no_desc = 0

        for m in targets:
            unwrapped = m
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            for _name, mod in unwrapped.named_modules():
                w = getattr(mod, "weight", None)
                if w is None or w.requires_grad or w.dim() != 2:
                    continue
                if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz):
                    already_fp8 += 1
                    if not hasattr(w, "_fp8_desc"):
                        already_fp8_no_desc += 1
                        if already_fp8_no_desc <= 5:
                            print_rank_0(
                                f"  [FP8 BUG] {_name}.weight: FP8 but NO _fp8_desc! "
                                f"shape={tuple(w.shape)} fp8_amax={w.data.float().abs().amax():.4f}"
                            )
                        amax = w.data.float().abs().amax().clamp(min=1e-12)
                        w._fp8_scale = (torch.finfo(fp8_dtype).max / amax).to(w.device)
                        w._fp8_scale_reciprocal = (1.0 / w._fp8_scale).to(w.device)
                        fp8_data_t = _precompute_fp8_transpose(w.data)
                        w._fp8_desc = FP8Descriptor(
                            data=w.data,
                            scale=w._fp8_scale,
                            fp8_dtype=fp8_dtype,
                            _transpose=fp8_data_t,
                        )
                        w._fp8_orig_shape = w.shape
                        w._fp8_original_dtype = torch.bfloat16
                        w._fp8_storage_enabled = True
                        _install_fp8_forward_hooks(mod, fp8_dtype)
                    continue
                if w.dtype == torch.bfloat16:
                    old_bytes = w.numel() * w.element_size()
                    _scaling = _get_fp8_store_scaling()
                    fp8_data, scale, fp8_data_t = _fp8_store_quantize_weight(
                        w.data, fp8_dtype, _scaling
                    )
                    w.data = fp8_data
                    w._fp8_scale = scale.to(w.device)
                    if _scaling not in ("blockwise", "blockwise2d"):
                        w._fp8_scale_reciprocal = (1.0 / scale).to(w.device)
                    w._fp8_desc = FP8Descriptor(
                        data=w.data,
                        scale=w._fp8_scale,
                        fp8_dtype=fp8_dtype,
                        _transpose=fp8_data_t,
                    )
                    w._fp8_orig_shape = fp8_data.shape
                    w._fp8_original_dtype = torch.bfloat16
                    w._fp8_storage_enabled = True
                    freed_bytes += old_bytes - fp8_data.numel() * fp8_data.element_size()
                    converted += 1
                    if not getattr(mod, "_fp8_hooks_installed", False):
                        _install_fp8_forward_hooks(mod, fp8_dtype)
                        mod._fp8_hooks_installed = True

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print_rank_0(f"> GPU memory after FP8 conversion: {alloc:.2f}GB")

        print_rank_0(
            f"> FP8 param storage: {converted} BF16 weights converted to FP8, "
            f"{already_fp8} already FP8 ({already_fp8_no_desc} had NO _fp8_desc!), "
            f"freed {freed_bytes/(1024**3):.1f}GB"
        )
        if already_fp8_no_desc > 0:
            print_rank_0(
                f"  *** WARNING: {already_fp8_no_desc} FP8 weights lost _fp8_desc "
                f"and got WRONG scale (fp8_max/fp8_amax ≈ 1.0 instead of correct value)! ***"
            )
        return result

    _load_with_fp8._fp8_patched = True
    _ckpt.load_checkpoint = _load_with_fp8
    training_mod = sys.modules.get("megatron.training.training")
    if training_mod is not None:
        training_mod.load_checkpoint = _load_with_fp8


def _wrap_load_from_state_dict(module, fp8_dtype):
    """Override _load_from_state_dict to quantize 'weight' on the fly."""
    import torch

    original_load = module._load_from_state_dict

    _fp8_hook_call_count = [0]

    def _fp8_load_from_state_dict(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight_key = prefix + "weight"
        _fp8_hook_call_count[0] += 1
        if _fp8_hook_call_count[0] <= 3:
            import torch.distributed as _dist

            rank = _dist.get_rank() if _dist.is_initialized() else 0
            if rank == 0:
                has_key = weight_key in state_dict
                has_attr = hasattr(module.weight, "_fp8_orig_shape") if hasattr(module, "weight") else False
                print(
                    f"[FP8 HOOK] prefix={prefix!r} key={weight_key!r} "
                    f"found={has_key} has_fp8_shape={has_attr} "
                    f"w.dtype={module.weight.dtype if hasattr(module, 'weight') else 'N/A'} "
                    f"w.shape={tuple(module.weight.shape) if hasattr(module, 'weight') else 'N/A'}",
                    flush=True,
                )

        if weight_key in state_dict:
            w = module.weight
            if hasattr(w, "_fp8_orig_shape"):
                incoming = state_dict[weight_key]
                if isinstance(incoming, torch.Tensor):
                    device = w.device if str(w.device) != "meta" else torch.device("cuda")
                    _scaling = _get_fp8_store_scaling()
                    fp8_w, scale, fp8_w_t = _fp8_store_quantize_weight(
                        incoming.to(device), fp8_dtype, _scaling
                    )
                    w.data = fp8_w
                    w._fp8_scale = scale.to(device)
                    if _scaling not in ("blockwise", "blockwise2d"):
                        w._fp8_scale_reciprocal = (1.0 / scale).to(device)
                    w._fp8_desc = FP8Descriptor(
                        data=w.data,
                        scale=w._fp8_scale,
                        fp8_dtype=fp8_dtype,
                        _transpose=fp8_w_t,
                    )
                    del state_dict[weight_key]

                    remaining = {k: v for k, v in state_dict.items() if k.startswith(prefix) and k != weight_key}
                    if remaining:
                        original_load(
                            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
                        )
                    return

        original_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    module._load_from_state_dict = _fp8_load_from_state_dict


def _install_fp8_forward_hooks(module, fp8_dtype):
    """No-op per-linear hooks — layer-level hooks are installed by
    _install_layer_level_fp8_hooks instead, to handle FP8 wrappers
    that bypass individual module forward()."""
    pass


def _install_embedding_output_fp8_hooks(model):
    """Install dequant hooks on embedding and output_layer only.

    Transformer layer linears are handled by ``FP8StoredLinearFunction``
    inside ``_do_gemm`` — no hooks needed there.  But embedding (index
    lookup) and Megatron's output_layer (``ColumnParallelLinear``) don't
    go through ``_do_gemm``, so they still need pre/post hooks.  These
    are only two weights (~500 MB BF16 each), so the temporary is small.
    """
    import torch

    embed_mod = None
    output_mod = None
    for name, mod in model.named_modules():
        if "word_embeddings" in name and hasattr(mod, "weight"):
            w = mod.weight
            if hasattr(w, "_fp8_desc"):
                embed_mod = mod
        if name.endswith("output_layer") and hasattr(mod, "weight"):
            w = mod.weight
            if hasattr(w, "_fp8_desc"):
                output_mod = mod

    count = 0
    for mod in [embed_mod, output_mod]:
        if mod is None:
            continue

        def _pre(m, inputs, _mod=mod):
            w = _mod.weight
            if hasattr(w, "_fp8_desc"):
                orig_dtype = getattr(w, "_fp8_original_dtype", torch.bfloat16)
                _mod._fp8_emb_saved = w.data
                _sc = w._fp8_desc.scale
                if _sc.numel() > 1:
                    # blockwise2d 2D block scale (dequant factor) -> expand+multiply
                    from lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight

                    w.data = _dequant_fp8_weight(w.data, _sc, 128).to(orig_dtype)
                else:
                    w.data = (w.data.to(torch.float32) / _sc).to(orig_dtype)

        def _post(m, inputs, output, _mod=mod):
            if hasattr(_mod, "_fp8_emb_saved"):
                _mod.weight.data = _mod._fp8_emb_saved
                del _mod._fp8_emb_saved

        mod.register_forward_pre_hook(_pre)
        mod.register_forward_hook(_post)
        count += 1

    print_rank_0(f"> FP8 param storage: installed embedding/output dequant hooks " f"on {count} modules")


# ---------------------------------------------------------------------------
# Synthetic warmup + FP8 state reset
# ---------------------------------------------------------------------------

_warmup_step_counter = 0
_warmup_completed = False


def _get_synthetic_batch(args, *, zero_last_loss_mask=False):
    """Generate a synthetic batch for GPU kernel warmup.

    The loss_mask is zeroed out entirely so that the optimizer step
    receives zero gradients and trainable weights (e.g. LoRA) are not
    corrupted by synthetic data.  Forward + backward still execute
    (warming up GPU kernels and calibrating FP8 amax history).

    Args:
        zero_last_loss_mask: Legacy flag (kept for API compat); the
            entire loss_mask is now always zeroed.
    """
    seq_length = args.seq_length
    mbs = args.micro_batch_size

    tokens = torch.ones(mbs, seq_length, dtype=torch.long, device="cuda") * 3545
    tokens[:, -1] = 2
    labels = tokens.clone()
    loss_mask = torch.zeros(mbs, seq_length, dtype=torch.float, device="cuda")
    attention_mask = torch.ones(mbs, 1, seq_length, seq_length, dtype=torch.bool, device="cuda")
    position_ids = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(mbs, -1)

    return tokens, labels, loss_mask, attention_mask, position_ids


def reset_fp8_state(model):
    """Reset FP8 scaling state in all Lumen quantised layers."""

    def _reset(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
        if hasattr(m, "_quant_manager"):
            m._quant_manager.reset()
        if hasattr(m, "_tl_scaling_manager"):
            m._tl_scaling_manager.reset()

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.apply(_reset)
    print_rank_0("> FP8 state reset after warmup")


_WARMUP_EVAL_STEPS = int(os.environ.get("LUMEN_WARMUP_EVAL_STEPS", "0"))


def _run_warmup_eval_pass(model, args):
    """Run synthetic forward passes in eval mode to prime the GPU allocator.

    The MLPerf reference runs ``warmup_validation_steps`` before real
    training so that the eval-time allocation pattern is already present
    in the allocator's block cache.  This prevents the first real eval
    from fragmenting the cache and causing a permanent step-time
    regression.

    The forward pass includes the loss computation path so that all
    eval-specific tensors (loss gather, metric buffers) are also
    pre-allocated in the cache.

    Controlled by ``LUMEN_WARMUP_EVAL_STEPS`` (default 0 = disabled).
    """
    n = _WARMUP_EVAL_STEPS
    if n <= 0:
        return

    print_rank_0(f"> Running {n} warmup eval forward passes to prime allocator ...")

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    unwrapped.eval()
    with torch.no_grad():
        for _ in range(n):
            tokens, labels, loss_mask, attention_mask, position_ids = _get_synthetic_batch(
                args, zero_last_loss_mask=True
            )
            output = unwrapped(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
            if output is not None:
                losses = output.view(-1).float()
                lm = loss_mask.view(-1).float()
                _ = torch.sum(losses * lm)
                _ = lm.sum()
    unwrapped.train()

    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_rank_0(f"> Warmup eval pass done ({n} steps). Allocator primed.")


# ---------------------------------------------------------------------------
# Loss function + early stopping
# ---------------------------------------------------------------------------

_val_loss_ema: Optional[float] = None
_early_stop_logged = False


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """LM loss. Early stopping is handled collectively on the *reduced*
    validation loss by ``install_val_loss_early_stop_hook`` — NOT here. The
    previous per-step, per-rank training-loss EMA stop caused a DP desync
    (one rank crossed the threshold on its local loss and exited the train
    loop while others continued -> mismatched collectives -> NCCL deadlock)."""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
    return loss, num_tokens, {"lm loss": reporting}


# ---------------------------------------------------------------------------
# Forward step factory
# ---------------------------------------------------------------------------


def make_forward_step(get_batch_fn: Callable, loss_fn: Callable = loss_func, zero_last_loss_mask: bool = False):
    """Return a ``forward_step`` function suitable for :func:`megatron.training.pretrain`.

    Args:
        get_batch_fn: Model-specific batch constructor
            ``(data_iterator, vp_stage) -> (tokens, labels, loss_mask, attention_mask, position_ids)``.
        loss_fn: Loss function (defaults to :func:`loss_func`).
        zero_last_loss_mask: Forwarded to :func:`_get_synthetic_batch`.
    """

    def forward_step(data_iterator, model: GPTModel):
        global _warmup_step_counter, _warmup_completed

        args = get_args()
        timers = get_timers()
        warmup_steps = getattr(args, "warmup_steps", 0)

        timers("batch-generator", log_level=2).start()
        with stimer(bdata=True):
            if warmup_steps <= 0 and not _warmup_completed:
                _warmup_completed = True
                from lumen.ops.quantize.linear import set_warmup_mode

                set_warmup_mode(False)
            if warmup_steps > 0 and not _warmup_completed:
                _warmup_step_counter += 1
                if _warmup_step_counter <= warmup_steps:
                    tokens, labels, loss_mask, attention_mask, position_ids = _get_synthetic_batch(
                        args, zero_last_loss_mask=zero_last_loss_mask
                    )
                    if data_iterator is not None:
                        try:
                            next(data_iterator)
                        except StopIteration:
                            pass
                else:
                    if getattr(args, "linear_fp8", False):
                        reset_fp8_state(model)
                    _run_warmup_eval_pass(model, args)
                    if getattr(args, "linear_fp8", False):
                        reset_fp8_state(model)
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    _warmup_completed = True
                    from lumen.ops.quantize.linear import set_warmup_mode

                    set_warmup_mode(False)
                    print_rank_0(f"> Synthetic warmup complete ({warmup_steps} steps). " f"Resuming with real data.")
                    vp_stage = get_attr_wrapped_model(model, "vp_stage")
                    tokens, labels, loss_mask, attention_mask, position_ids = get_batch_fn(data_iterator, vp_stage)
            else:
                vp_stage = get_attr_wrapped_model(model, "vp_stage")
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch_fn(data_iterator, vp_stage)
        timers("batch-generator").stop()

        with stimer:
            with get_cpu_offload_context(args):
                output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(loss_fn, loss_mask, model=model)

    return forward_step


# ---------------------------------------------------------------------------
# Common CLI argument groups
# ---------------------------------------------------------------------------


def add_common_megatron_args(parser):
    """Register CLI argument groups shared by all Megatron model scripts.

    Registers: ``--backend``, lumen, mxfp8-block-config, lora,
    fp8-training, and warmup/early-stop groups.

    Uses :func:`safe_add_argument` so that model-specific scripts can
    pre-register any of these flags with different defaults **before**
    calling this function.
    """
    safe_add_argument(
        parser, "--backend", type=str, default="megatron", choices=["megatron", "fsdp"], help="Training backend."
    )

    lumen = parser.add_argument_group(title="Lumen")
    safe_add_argument(
        lumen,
        "--lumen-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "csrc", "asm"],
        help="Lumen attention kernel backend. 'auto' prefers csrc with triton fallback. "
        "'asm' uses ASM kernels with fallback chain: asm -> csrc -> triton.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-attn",
        type=str,
        default="none",
        choices=["none", "dpa", "mha"],
        help="FP8 attention scope: 'none' = BF16 attention, "
        "'dpa' = FP8 dot-product attention only, "
        "'mha' = FP8 for full Multi-Head Attention block "
        "(QKV projection + attention + output projection).",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-quant-type",
        type=str,
        default="blockwise",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none", "mxfp8"],
        help="FP8 quantisation type for FP8 attention backends.",
    )
    safe_add_argument(
        lumen,
        "--lumen-rmsnorm",
        action="store_true",
        default=False,
        help="Replace RMSNorm with Lumen Triton-accelerated RMSNorm.",
    )
    safe_add_argument(
        lumen,
        "--lumen-norm",
        action="store_true",
        default=False,
        help="Replace all norm modules (RMSNorm and LayerNorm) with Lumen implementations.",
    )
    safe_add_argument(
        lumen,
        "--lumen-linear",
        action="store_true",
        default=False,
        help="Use Lumen parallel linear modules (LumenColumnParallelLinear, "
        "LumenRowParallelLinear, LumenLayerNormLinear) via the Lumen spec provider.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cross-entropy",
        action="store_true",
        default=False,
        help="Compute loss using Lumen's Triton parallel cross-entropy kernel.",
    )
    safe_add_argument(
        lumen,
        "--lumen-ce-chunk-rows",
        type=int,
        default=0,
        help=(
            "Row chunk size for chunked cross-entropy (0 = disabled). "
            "Splits B*SQ into chunks of this size so each chunk's allgather "
            "transfers chunk_rows*3 floats instead of B*SQ*3, reducing peak "
            "activation memory. Effective only when --lumen-cross-entropy is set. "
            "Typical value: 2048 (= 2 chunks for MBS=1 seq_len=4096)."
        ),
    )
    safe_add_argument(
        lumen,
        "--lumen-cpu-offload",
        action="store_true",
        default=False,
        help="Offload activations to CPU pinned memory during forward, prefetch in backward.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-checkpoint",
        action="store_true",
        default=False,
        help="Use FP8-aware activation checkpointing that preserves scaling state.",
    )
    safe_add_argument(
        lumen,
        "--lumen-hip-graphs",
        action="store_true",
        default=False,
        help="Graph-capture training steps to reduce kernel launch overhead.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-activation-store",
        action="store_true",
        default=False,
        help="Store MLP activations in FP8 during forward for reduced memory in backward.",
    )
    safe_add_argument(
        lumen,
        "--lumen-cp-comm-type",
        type=str,
        default="a2a",
        choices=["a2a", "p2p"],
        help="Context parallelism communication type: 'a2a' (all-to-all) or 'p2p' (ring).",
    )
    safe_add_argument(
        lumen,
        "--lumen-delay-wgrad",
        action="store_true",
        default=False,
        help="Defer weight gradient computation to overlap with next layer comm.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fp8-param-gather",
        action="store_true",
        default=False,
        help="Store and all-gather parameters in FP8 for reduced communication volume.",
    )
    safe_add_argument(
        lumen,
        "--fp8-param-storage",
        action="store_true",
        default=False,
        help="Store frozen base-model weights in FP8 after checkpoint loading. "
        "Halves model weight memory (~140GB→~70GB for 70B) enabling TP=1 on "
        "192GB GPUs. Weights are dequantized on-the-fly during forward pass.",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap",
        action="store_true",
        default=False,
        help="Overlap TP communication with GEMM computation. "
        "Mode is set by --lumen-tp-comm-overlap-mode (default: none, which uses "
        "SDMA async overlap when --use-sdma is set). Use 'pipeline' for chunked "
        "NCCL fused pipelining (requires sequence_parallel, BF16/scaling_type=none).",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-mode",
        type=str,
        default="none",
        choices=["none", "pipeline"],
        help="TP comm-GEMM overlap mode. 'none': legacy SDMA async overlap (requires "
        "--use-sdma). 'pipeline': chunked NCCL fused pipelining with user-buffer "
        "double-buffering (requires sequence_parallel, BF16).",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-chunks",
        type=int,
        default=4,
        help="Number of pipeline chunks for 'pipeline' overlap mode. Sequence length "
        "must be divisible by this value. More chunks = finer overlap granularity "
        "but higher scheduling overhead.",
    )
    safe_add_argument(
        lumen,
        "--lumen-tp-comm-overlap-method",
        type=str,
        default="nccl",
        choices=["nccl"],
        help="Communication backend for 'pipeline' overlap mode. Currently only 'nccl' " "is supported.",
    )
    safe_add_argument(
        lumen,
        "--use-sdma",
        action="store_true",
        default=False,
        help="Use mori SDMA instead of torch.distributed for supported collectives "
        "(TP comm, amax all-reduce, CP all-to-all) when available.",
    )
    safe_add_argument(
        lumen,
        "--lumen-fused-rope",
        action="store_true",
        default=False,
        help="Use AITER fused RoPE kernel for rotary positional embeddings.",
    )
    safe_add_argument(
        lumen,
        "--lumen-gradient-accumulation-fusion",
        action="store_true",
        default=False,
        help="Fuse weight gradient accumulation into GEMM backward (accumulate into main_grad).",
    )
    safe_add_argument(
        lumen,
        "--lumen-fused-mlp",
        action="store_true",
        default=False,
        help="Use fused MLP modules (LumenFusedMLP / LumenGatedMLP) for reduced kernel launch overhead.",
    )
    mxfp8 = parser.add_argument_group(title="mxfp8-block-config")
    safe_add_argument(mxfp8, "--mxfp8-block-m-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-fwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dq-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-m-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-block-n-dkv-bwd", type=int, default=128)
    safe_add_argument(mxfp8, "--mxfp8-quant-block-size", type=int, default=128)

    lora = parser.add_argument_group(title="lora")
    safe_add_argument(lora, "--lora-rank", type=int, default=0, help="LoRA rank. 0 = disabled.")
    safe_add_argument(lora, "--lora-alpha", type=float, default=32.0)
    safe_add_argument(lora, "--lora-dropout", type=float, default=0.1)
    safe_add_argument(
        lora,
        "--lora-target-modules",
        type=str,
        default="all",
        choices=["attention", "attention_mlp", "all"],
        help="LoRA target scope: 'attention' (QKV+proj, NeMo reference), "
        "'attention_mlp' (attention+MLP), 'all' (attention+MLP+emb+output).",
    )
    safe_add_argument(
        lora,
        "--lora-a2a",
        action="store_true",
        default=False,
        help="Enable LoRA all-to-all communication optimisation.",
    )

    lfp8 = parser.add_argument_group(title="linear-fp8")
    safe_add_argument(
        lfp8,
        "--linear-fp8",
        action="store_true",
        default=False,
        help="Enable FP8 quantised training for Linear layers.",
    )
    safe_add_argument(
        lfp8,
        "--linear-fp8-scaling",
        type=str,
        default="delayed",
        choices=["dynamic", "delayed", "blockwise", "blockwise2d", "per_token", "none"],
    )
    safe_add_argument(lfp8, "--linear-fp8-block-size", type=int, default=128)
    safe_add_argument(lfp8, "--linear-fp8-amax-algo", type=str, default="max", choices=["max", "most_recent"])
    safe_add_argument(lfp8, "--linear-fp8-reduce-amax", action="store_true", default=False)
    safe_add_argument(lfp8, "--linear-fp8-amax-history", type=int, default=16)
    safe_add_argument(
        lfp8, "--linear-fp8-margin", type=int, default=0, help="Margin for FP8 scaling factor computation."
    )
    safe_add_argument(lfp8, "--linear-fp8-activation", action="store_true", default=True)
    safe_add_argument(lfp8, "--no-linear-fp8-activation", dest="linear_fp8_activation", action="store_false")
    safe_add_argument(lfp8, "--linear-fp8-wgrad", action="store_true", default=True)
    safe_add_argument(
        lfp8,
        "--no-linear-fp8-wgrad",
        dest="linear_fp8_wgrad",
        action="store_false",
        help="Execute weight gradient GEMM in higher precision (BF16) even for FP8 runs.",
    )
    safe_add_argument(
        lfp8,
        "--linear-fp8-cache-frozen-weight",
        dest="linear_fp8_cache_frozen_weight",
        action="store_true",
        default=False,
        help="Cache FP8-quantised frozen base weights to avoid re-quantisation on every forward/recompute.",
    )
    safe_add_argument(
        lfp8,
        "--grad-quant-type",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "fp4"],
        help="Gradient quantization type (None=disabled). Applies to Linear, Attention, and RMSNorm.",
    )

    wes = parser.add_argument_group(title="warmup-early-stop")
    safe_add_argument(wes, "--warmup-steps", type=int, default=0)
    safe_add_argument(wes, "--val-loss-target", type=float, default=None)

    ckpt = parser.add_argument_group(title="checkpoint")
    add_shared_checkpoint_args(ckpt)

    experiment = parser.add_argument_group(title="experiment")
    add_shared_experiment_args(experiment)

    parser.set_defaults(**_TE_FORCE_OVERRIDES)

    return parser

"""Lumen FP8 hooks for DSV4 on the Miles/Megatron training path."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Optional

_HOOK_INSTALLED = False


def dsv4_linear_fp8_enabled() -> bool:
    return os.environ.get("LUMEN_DSV4_LINEAR_FP8", "0") == "1"


def _fp8_scaling_type() -> str:
    return os.environ.get("LUMEN_DSV4_FP8_SCALING", "blockwise")


def _fp8_block_size() -> int:
    return int(os.environ.get("LUMEN_DSV4_FP8_BLOCK_SIZE", "128"))


def _build_fp8_args() -> Namespace:
    """Minimal args namespace for :func:`enable_fp8_for_dsv4_model`."""
    return Namespace(
        linear_fp8=True,
        linear_fp8_scaling=_fp8_scaling_type(),
        linear_fp8_block_size=_fp8_block_size(),
        linear_fp8_amax_algo=os.environ.get("LUMEN_DSV4_FP8_AMAX_ALGO", "max"),
        linear_fp8_reduce_amax=os.environ.get("LUMEN_DSV4_FP8_REDUCE_AMAX", "0") == "1",
        linear_fp8_amax_history=int(os.environ.get("LUMEN_DSV4_FP8_AMAX_HISTORY", "16")),
        linear_fp8_margin=int(os.environ.get("LUMEN_DSV4_FP8_MARGIN", "0")),
        linear_fp8_activation=os.environ.get("LUMEN_DSV4_FP8_ACTIVATION", "1") != "0",
        linear_fp8_wgrad=os.environ.get("LUMEN_DSV4_FP8_WGRAD", "1") != "0",
        lumen_fp8_attn=os.environ.get("LUMEN_DSV4_FP8_ATTN", "none"),
        lumen_gradient_accumulation_fusion=False,
        lumen_delay_wgrad=False,
    )


def enable_fp8_for_dsv4_model(model, args: Optional[Namespace] = None) -> None:
    """Enable Lumen FP8 on all DSV4 Lumen linear modules."""
    from megatron.training.utils import print_rank_0

    from lumen.config import LumenConfig
    from lumen.models.dsv4.megatron.layers import LumenDuplicatedLinear
    from lumen.models.megatron import enable_fp8_for_parallel_linear
    from lumen.modules.grouped_linear import LumenGroupedLinear
    from lumen.modules.parallel_linear import LumenColumnParallelLinear, LumenRowParallelLinear
    from lumen.quantize import ScalingManager

    fp8_args = args if args is not None else _build_fp8_args()
    cfg = LumenConfig.from_args(fp8_args)
    scaling_type = getattr(fp8_args, "linear_fp8_scaling", "blockwise")
    block_size = getattr(fp8_args, "linear_fp8_block_size", 128)

    enable_fp8_for_parallel_linear(
        model,
        scaling_type=scaling_type,
        block_size=block_size,
        quant_config=cfg.quant_config,
        fp8_mha=getattr(fp8_args, "lumen_fp8_attn", "none") == "mha",
        gradient_accumulation_fusion=getattr(fp8_args, "lumen_gradient_accumulation_fusion", False),
        delay_wgrad=getattr(fp8_args, "lumen_delay_wgrad", False),
    )

    dup_count = 0
    for module in model.modules():
        if isinstance(module, LumenDuplicatedLinear):
            module.enable_fp8(
                scaling_manager=ScalingManager(cfg.quant_config),
                scaling_type=scaling_type,
                block_size=block_size,
            )
            dup_count += 1

    linear_types = (LumenColumnParallelLinear, LumenRowParallelLinear, LumenGroupedLinear, LumenDuplicatedLinear)
    total = sum(1 for m in model.modules() if isinstance(m, linear_types))
    print_rank_0(
        f"> Lumen DSV4 FP8 enabled (scaling={scaling_type}, block={block_size}, "
        f"modules={total}, duplicated={dup_count})"
    )


def install_dsv4_fp8_model_provider_hook(args=None) -> None:
    """Wrap Miles model provider to enable Lumen FP8 after GPTModel construction.

    Invoked via ``--custom-megatron-init-path`` when ``LUMEN_DSV4_LINEAR_FP8=1``.
    Does not re-check the env var on Ray workers (init path is the enable signal).
    """
    del args
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return

    import miles.backends.megatron_utils.model_provider as model_provider_mod

    _orig_get_provider = model_provider_mod.get_model_provider_func

    def get_model_provider_func(args, role="actor"):
        provider = _orig_get_provider(args, role)

        def wrapped_provider(
            pre_process=True,
            post_process=True,
            vp_stage=None,
            config=None,
            pg_collection=None,
        ):
            model = provider(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
                config=config,
                pg_collection=pg_collection,
            )
            enable_fp8_for_dsv4_model(model, _build_fp8_args())
            return model

        return wrapped_provider

    model_provider_mod.get_model_provider_func = get_model_provider_func
    _HOOK_INSTALLED = True


def register_dsv4_megatron_cli(parser) -> None:
    """Register Lumen FP8 CLI flags on the Miles/Megatron parser."""
    from lumen.models.megatron import add_common_megatron_args

    add_common_megatron_args(parser)


def install_dsv4_miles_cli_hook() -> None:
    """Extend Miles Megatron arg parser with Lumen ``--linear-fp8`` flags."""
    if not dsv4_linear_fp8_enabled():
        return

    from miles.utils import arguments as miles_arguments

    _orig_provider = miles_arguments.get_miles_extra_args_provider

    def get_miles_extra_args_provider(add_custom_arguments=None):
        inner = _orig_provider(add_custom_arguments)

        def provider(parser):
            inner(parser)
            register_dsv4_megatron_cli(parser)

        return provider

    miles_arguments.get_miles_extra_args_provider = get_miles_extra_args_provider

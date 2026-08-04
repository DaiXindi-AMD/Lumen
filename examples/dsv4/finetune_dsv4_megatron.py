"""Native torchrun GRPO finetune entry for DSV4 (debug-train-only, no Ray/Miles train.py).

Reads ``fake_rollout.pt``, runs GRPO policy loss updates via ``megatron.training.pretrain``
with a patched train loop (see ``lumen.models.dsv4.megatron.finetune_loop``).

Launched by ``run_dsv4_4layer_finetune_inner.sh`` or ``run_dsv4_flash_finetune_inner.sh``.
"""

from __future__ import annotations

import os
import sys
from functools import partial

MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
if MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from lumen.models.dsv4.megatron.spec import get_dsv4_spec  # noqa: E402, F401

from model_provider import model_provider as _megatron_model_provider  # noqa: E402

from lumen.models.dsv4.megatron.finetune_loop import (  # noqa: E402
    add_dsv4_finetune_args,
    dsv4_grpo_forward_step,
    run_dsv4_grpo_finetune,
)
from lumen.models.dsv4.megatron.pretrain import (  # noqa: E402
    add_dsv4_pretrain_args,
    dsv4_gpt_builder,
    dsv4_model_provider,
    install_dsv4_safe_mock_data,
)

install_dsv4_safe_mock_data()


def _build_model_provider():
    return partial(
        dsv4_model_provider,
        _megatron_model_provider,
        dsv4_gpt_builder,
    )


def _extra_args_provider(parser):
    add_dsv4_pretrain_args(parser)
    add_dsv4_finetune_args(parser)
    return parser


def main() -> None:
    if os.environ.get("LUMEN_DSV4_LINEAR_FP8", "0") == "1":
        from lumen.models.megatron import install_fp8_param_gather_hook

        install_fp8_param_gather_hook()

    run_dsv4_grpo_finetune(
        _build_model_provider(),
        extra_args_provider=_extra_args_provider,
    )


if __name__ == "__main__":
    main()

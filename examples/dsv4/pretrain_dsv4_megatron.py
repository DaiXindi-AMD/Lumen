"""Shared Megatron pretrain entry for DSV4 (4-layer and Flash full model).

Uses ``megatron.training.pretrain`` with ``--spec lumen.models.dsv4.megatron.spec get_dsv4_spec``.

Launched by ``run_dsv4_4layer_pretrain_inner.sh`` or ``run_dsv4_flash_pretrain_inner.sh``.
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

from megatron.core.enums import ModelType  # noqa: E402
from megatron.training import pretrain  # noqa: E402
from model_provider import model_provider as _megatron_model_provider  # noqa: E402
from pretrain_gpt import train_valid_test_datasets_provider  # noqa: E402

from lumen.models.dsv4.megatron.pretrain import (  # noqa: E402
    add_dsv4_pretrain_args,
    dsv4_forward_step,
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


def main() -> None:
    if os.environ.get("LUMEN_DSV4_LINEAR_FP8", "0") == "1":
        from lumen.models.megatron import install_fp8_param_gather_hook

        install_fp8_param_gather_hook()

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        _build_model_provider(),
        ModelType.encoder_or_decoder,
        dsv4_forward_step,
        extra_args_provider=add_dsv4_pretrain_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )


if __name__ == "__main__":
    main()

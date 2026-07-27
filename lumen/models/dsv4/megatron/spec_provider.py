"""Lumen backend spec provider for DSV4 transformer blocks (MoE/MLP, no TE)."""

from typing import Optional, Tuple

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

from lumen.models.spec_provider import LumenSpecProvider
from lumen.modules.grouped_linear import (
    LumenColumnParallelGroupedLinear,
    LumenRowParallelGroupedLinear,
)


class LumenDSV4SpecProvider(LumenSpecProvider):
    """Backend for DSV4 MoE/MLP/dense layers — Lumen linear/norm only.

    ``TEGroupedMLP`` here is Megatron's grouped-expert *container*; fc1/fc2
    are ``LumenColumnParallelGroupedLinear`` / ``LumenRowParallelGroupedLinear``,
    not Transformer Engine grouped linear modules.
    """

    def grouped_mlp_modules(
        self,
        moe_use_grouped_gemm: bool = False,
        moe_use_legacy_grouped_gemm: bool = False,
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=LumenColumnParallelGroupedLinear,
                linear_fc2=LumenRowParallelGroupedLinear,
            )

        return SequentialMLP, MLPSubmodules(
            linear_fc1=self.column_parallel_linear(),
            linear_fc2=self.row_parallel_linear(),
        )

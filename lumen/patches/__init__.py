###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unified patch registry for Lumen.

Patches are registered with :func:`register_patch` and applied via
:func:`apply_patches` according to their :class:`PatchPhase`.
"""

from lumen.patches.cli import apply_megatron_source_patches, print_patch_report
from lumen.patches.builders import apply_args_patches, apply_config_build, apply_model_build
from lumen.patches.training import apply_training_patches
from lumen.patches.registry import (
    PatchRegistry,
    apply_patches,
    list_patches,
    register_patch,
)
from lumen.patches.types import PatchPhase, PatchResult, PatchSpec

__all__ = [
    "PatchPhase",
    "PatchRegistry",
    "PatchResult",
    "PatchSpec",
    "apply_args_patches",
    "apply_config_build",
    "apply_megatron_source_patches",
    "apply_training_patches",
    "apply_patches",
    "list_patches",
    "print_patch_report",
    "register_patch",
]

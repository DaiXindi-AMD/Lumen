###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Shims for Megatron-LM APIs that only exist in newer checkouts.

Lumen is imported against whatever Megatron the user has on PYTHONPATH, so a
symbol that a recent Megatron added cannot be imported unconditionally: the
failure lands as an ImportError at ``import lumen.modules``, which takes down
every entry point, including the ones that never touch the missing API.
"""

from __future__ import annotations

__all__ = ["ensure_metadata_has_dp_cp_group"]


try:
    from megatron.core.transformer.utils import (  # type: ignore[attr-defined]
        ensure_metadata_has_dp_cp_group,
    )
except ImportError:
    # Megatron gained this helper along with the checkpoint APIs that expect a
    # data-parallel/context-parallel group in the sharded_state_dict metadata.
    # Older checkouts have neither, so passing the metadata through unchanged is
    # what their checkpoint code already assumes.
    def ensure_metadata_has_dp_cp_group(metadata):
        """Return *metadata* unchanged; this Megatron has no group to inject."""
        return metadata

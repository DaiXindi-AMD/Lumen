"""Megatron spec entry point for Lumen DSV4."""

from lumen.models.dsv4.megatron.deepseek_v4 import _patch_megatron_no_te, get_dsv4_spec

_patch_megatron_no_te()

__all__ = ["get_dsv4_spec"]

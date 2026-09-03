"""Runtime (import-time) patch registrations."""

from lumen.patches.runtime import moe_fused_router as moe_fused_router  # noqa: F401

__all__ = ["moe_fused_router"]


def apply_import_patches(*, default_only: bool = True, dry_run: bool = False):
    """Apply registered IMPORT-phase Megatron monkey-patches."""
    import lumen.patches.runtime.megatron_import  # noqa: F401

    from lumen.patches import PatchPhase, apply_patches

    return apply_patches(
        PatchPhase.IMPORT,
        default_only=default_only,
        dry_run=dry_run,
    )

"""On-disk Megatron source patches (PatchPhase.SOURCE)."""

from lumen.patches.source import dsv4 as dsv4  # noqa: F401 — DSV4 model patches
from lumen.patches.source import llama as llama  # noqa: F401 — LLaMA/GPT patches
from lumen.patches.source import llama_lora as llama_lora  # noqa: F401 — LoRA finetune patches
from lumen.patches.source import rocm as rocm  # noqa: F401 — ROCm platform patches

__all__ = ["dsv4", "llama", "llama_lora", "rocm"]

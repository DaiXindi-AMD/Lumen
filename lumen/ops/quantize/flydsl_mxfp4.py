###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Explicit FlyDSL backends for dense MXFP4 GEMM on gfx950.

The kernel sources are pinned under :mod:`lumen.kernels.flydsl`.  This module
deliberately exposes named configurations instead of a generic dispatcher: the
MXFP4 autotuner benchmarks each configuration against the strict ASM incumbent
and persists the exact per-shape winner.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import os
import re
import threading
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import torch


FLYDSL_KERNEL_REVISION = "67e3b9d3dd2fb20abe87dc959c8f10f9824ff50b"
_PRESHUFFLE_FAMILY = "mxfp4_preshuffle"
_FOUR_WAVE_FAMILY = "mxfp4_4wave"
_FOUR_WAVE_MIN_RUNTIME = (0, 3, 2)
_UPSTREAM_SOURCE_SHA256 = {
    _PRESHUFFLE_FAMILY: "7af8d7a09620099527aba096218efc4cd0e1d1221cde5a5fe638f762c2156102",
    _FOUR_WAVE_FAMILY: "54c3b209a8518d20a32a42011060ed369bc6aa965b32e2d71c1c1dcf4517f243",
}


def _installed_runtime_version() -> str:
    """Version of the compiler/runtime that turns the vendored source into ISA."""
    try:
        return importlib.metadata.version("flydsl")
    except Exception:
        # A source-only checkout without distribution metadata is not a stable,
        # replayable compiler identity. Keep it out of production selection.
        return "unavailable"


FLYDSL_RUNTIME_VERSION = _installed_runtime_version()


def _configured_rocm_root() -> str:
    for name in ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME"):
        value = os.environ.get(name)
        if value:
            return os.path.realpath(os.path.expanduser(value))
    return os.path.realpath("/opt/rocm")


@functools.lru_cache(maxsize=4)
def _rocm_toolchain_identity(root: str) -> dict:
    """Compiler/linker identity behind FlyDSL's generated gfx950 code."""
    linker = Path(root) / "llvm" / "bin" / "ld.lld"
    try:
        digest = hashlib.sha256(linker.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = "unavailable"
    return {
        "rocm_root": root,
        "rocm_lld": os.path.realpath(linker),
        "rocm_lld_sha256": digest,
    }


class FlyDSLConfig(NamedTuple):
    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    waves_per_eu: int = 0
    xcd_swizzle: int = 0
    kernel_family: str = _PRESHUFFLE_FAMILY
    use_xcd_remap: bool = False


# This is the complete bounded set exercised by the upstream kernel's gfx950
# correctness/performance harness. Every shape profiles every legal member;
# selection must come from measurements rather than an unmeasured M heuristic.
_CONFIGS: Tuple[FlyDSLConfig, ...] = (
    FlyDSLConfig("flydsl_32x128x256", 32, 128, 256),
    FlyDSLConfig("flydsl_64x128x128", 64, 128, 128),
    FlyDSLConfig("flydsl_64x128x256", 64, 128, 256),
    FlyDSLConfig("flydsl_64x256x256", 64, 256, 256),
    FlyDSLConfig(
        "flydsl_4wave_256x256x256",
        256,
        256,
        256,
        waves_per_eu=1,
        kernel_family=_FOUR_WAVE_FAMILY,
        use_xcd_remap=True,
    ),
)
_CONFIG_BY_NAME = {config.name: config for config in _CONFIGS}
# Opt in for profiling first. A fresh FlyDSL revision must earn per-shape cache
# entries before it can participate in a production run.
_ENABLED = os.environ.get("LUMEN_MXFP4_FLYDSL", "0") == "1"


@functools.lru_cache(maxsize=2)
def _kernel_source_sha256(filename: str = "mxfp4_preshuffle.py") -> str:
    """Hash the vendored source so edited kernels cannot reuse old profiles."""
    source = (
        Path(__file__).resolve().parents[2]
        / "kernels"
        / "flydsl"
        / filename
    )
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


@functools.lru_cache(maxsize=1)
def _wrapper_source_sha256() -> str:
    """Hash launch/validation overhead that can change a profiled winner."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def _source_filename(config: FlyDSLConfig) -> str:
    if config.kernel_family == _FOUR_WAVE_FAMILY:
        return "mxfp4_4wave.py"
    return "mxfp4_preshuffle.py"


def _runtime_release(version: str) -> Optional[Tuple[int, int, int]]:
    """Parse the numeric release prefix without accepting unknown metadata."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?$", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def four_wave_runtime_supported(version: Optional[str] = None) -> bool:
    """Whether ``version`` can compile the vendored gfx950 4-wave kernel."""
    release = _runtime_release(
        FLYDSL_RUNTIME_VERSION if version is None else version
    )
    return release is not None and release >= _FOUR_WAVE_MIN_RUNTIME


def _fp4_dma_intrinsic_mode() -> int:
    """Match the vendored kernel's exact opt-in environment semantics."""
    return int(os.environ.get("FP4_DMA_INTRINSIC", "0") == "1")


def _runtime_supports(config: FlyDSLConfig) -> bool:
    release = _runtime_release(FLYDSL_RUNTIME_VERSION)
    if release is None:
        return False
    if config.kernel_family == _FOUR_WAVE_FAMILY:
        return four_wave_runtime_supported()
    return True


def config_identity(config: FlyDSLConfig) -> dict:
    """Exact implementation identity persisted with a per-shape result."""
    dma_intrinsic = (
        _fp4_dma_intrinsic_mode()
        if config.kernel_family == _FOUR_WAVE_FAMILY
        else "not_applicable"
    )
    return {
        "implementation": "flydsl",
        "kernel_family": config.kernel_family,
        "source_revision": FLYDSL_KERNEL_REVISION,
        "source_sha256": _kernel_source_sha256(_source_filename(config)),
        "wrapper_source_sha256": _wrapper_source_sha256(),
        "upstream_source_sha256": _UPSTREAM_SOURCE_SHA256[config.kernel_family],
        "runtime_version": FLYDSL_RUNTIME_VERSION,
        "tile_m": config.tile_m,
        "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "waves_per_eu": config.waves_per_eu,
        "xcd_swizzle": config.xcd_swizzle,
        "use_xcd_remap": config.use_xcd_remap,
        "fp4_dma_intrinsic": dma_intrinsic,
        **_rocm_toolchain_identity(_configured_rocm_root()),
    }


def backend_fingerprint() -> str:
    """Stable identity included in persisted per-shape profile decisions."""
    configs = ";".join(
        f"{c.name}:{c.kernel_family}:{c.tile_m},{c.tile_n},{c.tile_k},"
        f"{c.waves_per_eu},{c.xcd_swizzle},{int(c.use_xcd_remap)}"
        for c in _CONFIGS
    )
    sources = ";".join(
        f"{family}={_kernel_source_sha256(filename)}"
        for family, filename in (
            (_PRESHUFFLE_FAMILY, "mxfp4_preshuffle.py"),
            (_FOUR_WAVE_FAMILY, "mxfp4_4wave.py"),
        )
    )
    toolchain = _rocm_toolchain_identity(_configured_rocm_root())
    return (
        f"flydsl:{FLYDSL_KERNEL_REVISION}:sources={sources}:"
        f"wrapper={_wrapper_source_sha256()}:"
        f"fp4_dma_intrinsic={_fp4_dma_intrinsic_mode()}:"
        f"runtime={FLYDSL_RUNTIME_VERSION}:"
        f"rocm={toolchain['rocm_root']}:lld={toolchain['rocm_lld_sha256']}:"
        f"enabled={int(_ENABLED)}:{configs}"
    )


def is_backend_name(name: object) -> bool:
    return isinstance(name, str) and name in _CONFIG_BY_NAME


def backend_names() -> Tuple[str, ...]:
    return tuple(config.name for config in _CONFIGS)


def get_config(name: str) -> FlyDSLConfig:
    try:
        return _CONFIG_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown MXFP4 FlyDSL configuration: {name!r}") from exc


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """Return whether the pinned kernel and its FlyDSL runtime can be imported."""
    if not _ENABLED or FLYDSL_RUNTIME_VERSION == "unavailable":
        return False
    toolchain = _rocm_toolchain_identity(_configured_rocm_root())
    if toolchain["rocm_lld_sha256"] == "unavailable":
        return False
    try:
        if not torch.cuda.is_available():
            return False
        from flydsl.runtime.device import get_rocm_arch

        if get_rocm_arch() != "gfx950":
            return False
        import flydsl.compiler  # noqa: F401
        import flydsl.expr  # noqa: F401
        from lumen.kernels.flydsl.mxfp4_preshuffle import launch_gemm  # noqa: F401

        return True
    except Exception:
        return False


def supported_configs(M: int, N: int, K: int) -> Tuple[FlyDSLConfig, ...]:
    """Return every upstream-covered configuration legal for ``(M, N, K)``.

    The original preshuffle kernel masks a ragged M dimension.  The 4-wave
    kernel is deliberately narrower: it has no edge masks, and its peeled
    depth-2 pipeline requires at least four K iterations with every iteration
    after the first four occurring in pairs.
    """
    if not _ENABLED or M <= 0 or N <= 0 or K < 256 or K % 256:
        return ()

    configs = []
    for config in _CONFIGS:
        if not _runtime_supports(config):
            continue
        if config.kernel_family == _FOUR_WAVE_FAMILY:
            if M % 256 or N % 256 or K < 1024 or K % 512:
                continue
        elif N % config.tile_n or K % config.tile_k:
            continue
        configs.append(config)
    return tuple(configs)


_FOUR_WAVE_COMPILED = {}
_FOUR_WAVE_COMPILE_LOCK = threading.Lock()


def _run_four_wave(
    config: FlyDSLConfig,
    a_fp4: torch.Tensor,
    w_fp4_shuffled: torch.Tensor,
    scale_a_shuffled: torch.Tensor,
    scale_w_shuffled: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compile once per device/shape, then call the upstream 4-wave launcher."""
    import flydsl.compiler as flyc
    from lumen.kernels.flydsl.mxfp4_4wave import compile_fp4_gemm_4w

    M, N, K = a_fp4.shape[0], w_fp4_shuffled.shape[0], a_fp4.shape[1] * 2
    stream = torch.cuda.current_stream(device=a_fp4.device)
    args = (
        a_fp4.view(torch.uint8).view(-1),
        w_fp4_shuffled.view(torch.uint8).view(-1),
        out.view(-1),
        scale_a_shuffled.view(torch.uint8).view(-1),
        scale_w_shuffled.view(torch.uint8).view(-1),
        M,
        N,
        stream,
    )
    cache_key = (a_fp4.device.index, M, N, K, config.use_xcd_remap)
    compiled = _FOUR_WAVE_COMPILED.get(cache_key)
    if compiled is None:
        with _FOUR_WAVE_COMPILE_LOCK:
            compiled = _FOUR_WAVE_COMPILED.get(cache_key)
            if compiled is None:
                launcher = compile_fp4_gemm_4w(
                    K=K,
                    MN=(M, N),
                    use_xcd_remap=config.use_xcd_remap,
                )
                compiled = flyc.compile(launcher, *args)
                _FOUR_WAVE_COMPILED[cache_key] = compiled
    compiled(*args)


def run(
    config: FlyDSLConfig,
    a_fp4: torch.Tensor,
    w_fp4_shuffled: torch.Tensor,
    scale_a_shuffled: torch.Tensor,
    scale_w_shuffled: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compile/cache and launch one explicit FlyDSL configuration."""
    if not available():
        raise RuntimeError("FlyDSL MXFP4 backend is unavailable")

    M = a_fp4.shape[0]
    N = w_fp4_shuffled.shape[0]
    K = a_fp4.shape[1] * 2
    if config not in supported_configs(M, N, K):
        raise NotImplementedError(
            f"FlyDSL config {config.name} does not support MXFP4 shape {(M, N, K)}"
        )
    for label, tensor in (
        ("A", a_fp4),
        ("B", w_fp4_shuffled),
        ("scale A", scale_a_shuffled),
        ("scale B", scale_w_shuffled),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"FlyDSL MXFP4 {label} operand must be contiguous")

    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a_fp4.device)
    elif out.shape != (M, N) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"FlyDSL MXFP4 output must be bf16 {(M, N)}, got {out.dtype} {tuple(out.shape)}"
        )
    if not out.is_contiguous():
        raise ValueError("FlyDSL MXFP4 output must be contiguous")

    if config.kernel_family == _FOUR_WAVE_FAMILY:
        _run_four_wave(
            config,
            a_fp4,
            w_fp4_shuffled,
            scale_a_shuffled,
            scale_w_shuffled,
            out,
        )
        return out

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from lumen.kernels.flydsl.mxfp4_preshuffle import launch_gemm

    def _ptr(tensor: torch.Tensor):
        return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())

    launch_gemm(
        _ptr(out),
        _ptr(a_fp4),
        _ptr(w_fp4_shuffled),
        _ptr(scale_a_shuffled),
        _ptr(scale_w_shuffled),
        M,
        N,
        torch.cuda.current_stream(device=a_fp4.device),
        N,
        K,
        config.tile_m,
        config.tile_n,
        config.tile_k,
        "fp4",
        "bf16",
        "fp4",
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        config.waves_per_eu,
        config.xcd_swizzle,
        1,
        "none",
    )
    return out

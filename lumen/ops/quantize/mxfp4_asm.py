###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Strict metadata lookup for Lumen's prebuilt MXFP4 ASM backend.

The upstream A4W4 tuning table may contain several implementation families.
Lumen does not hand that mixed table to a generic dispatcher.  This module
builds an ASM-only view, validates every symbol against the installed manifest
and code object, and exposes only an exact ``(symbol, log2_split_k)`` pair.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)

TUNED_CONFIG_ENV = "AITER_CONFIG_GEMM_A4W4"
REGISTRY_POLICY_REVISION = "asm-only-v1"
_TUNED_FILENAME = "a4w4_blockscale_tuned_gemm.csv"
_MANIFEST_FILENAME = "f4gemm_bf16_per1x32Fp4.csv"
_ASM_SYMBOL_RE = re.compile(
    r"^_ZN5aiter\d+f4gemm_bf16_per1x32Fp4_BpreShuffle_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)E$"
)

AsmConfig = Tuple[str, int]
TableKey = Tuple[str, int, int, int, int]
FileSignature = Tuple[str, int, int, int, int]
IdentityItems = Tuple[Tuple[str, object], ...]
ManifestSnapshot = Tuple[Optional[Path], Optional[FileSignature]]
RuntimeArtifactKey = Tuple[str, str]
RuntimeArtifactBaseline = Tuple[str, IdentityItems]
_MANIFEST_SNAPSHOT_UNSET = object()


class RuntimeSnapshot(NamedTuple):
    """One coherent view of the ASM registry for a single GEMM dispatch.

    ``validation_token`` contains the same filesystem signatures that back
    the parser and hash caches below.  It is intentionally cheap to compare and
    changes whenever a tuned table, manifest, or selected code object changes.
    ``identity_items`` keeps the full persisted identity lazy: the dispatcher
    only materialises its dictionary after the lightweight token misses.
    """

    arch: str
    cu_num: int
    config: Optional[AsmConfig]
    validation_token: Tuple[object, ...]
    identity_items: Optional[IdentityItems]
    cacheable: bool = True


_registry_epoch = 0
_runtime_artifact_lock = threading.Lock()
_runtime_artifact_baselines: Dict[RuntimeArtifactKey, RuntimeArtifactBaseline] = {}
_runtime_artifact_warnings: set[Tuple[object, ...]] = set()


@functools.lru_cache(maxsize=1)
def _aiter_package_dir() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def default_tuned_config() -> Optional[str]:
    package_dir = _aiter_package_dir()
    if package_dir is None:
        return None
    path = package_dir / "configs" / _TUNED_FILENAME
    return str(path) if path.is_file() else None


def tuned_table_paths() -> Tuple[str, ...]:
    configured = os.environ.get(TUNED_CONFIG_ENV, "")
    if configured:
        return tuple(path for path in configured.split(os.pathsep) if path)
    default = default_tuned_config()
    return (default,) if default else ()


def _asm_roots() -> Iterable[Path]:
    configured = os.environ.get("AITER_ASM_DIR")
    if configured:
        # Keep this lexical rather than resolving symlinks on every GEMM. File
        # signatures below still follow the path and therefore observe a
        # retargeted symlink on the very next dispatch, while avoiding several
        # lstat calls in this extremely hot path.
        yield Path(os.path.abspath(os.path.expanduser(configured)))
    package_dir = _aiter_package_dir()
    if package_dir is not None:
        yield package_dir.parent / "hsa"
        yield package_dir.parent / "aiter_meta" / "hsa"


def _file_signature(path: Path) -> FileSignature:
    try:
        stat = path.stat()
    except OSError:
        return str(path), -1, -1, -1, -1
    return (
        str(path),
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
        stat.st_ino,
    )


@functools.lru_cache(maxsize=128)
def _file_sha256(signature: FileSignature) -> str:
    path = signature[0]
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def _strict_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _manifest_path(arch: str) -> Optional[Path]:
    seen = set()
    for root in _asm_roots():
        if root in seen:
            continue
        seen.add(root)
        path = root / arch / "f4gemm" / _MANIFEST_FILENAME
        if path.is_file():
            return path
    return None


@functools.lru_cache(maxsize=16)
def _load_manifest(
    signature: FileSignature,
) -> Dict[str, Dict[str, object]]:
    path = Path(signature[0])
    entries: Dict[str, Dict[str, object]] = {}
    ambiguous: set[str] = set()
    try:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                symbol = row.get("knl_name", "")
                match = _ASM_SYMBOL_RE.fullmatch(symbol)
                tile_m = _strict_int(row.get("tile_M"))
                tile_n = _strict_int(row.get("tile_N"))
                split_capable = _strict_int(row.get("splitK"))
                bpreshuffle = _strict_int(row.get("bpreshuffle"))
                code_object = row.get("co_name", "")
                if (
                    match is None
                    or tile_m != int(match.group("tile_m"))
                    or tile_n != int(match.group("tile_n"))
                    or split_capable not in (0, 1)
                    or bpreshuffle != 1
                    or not code_object
                    or symbol in ambiguous
                ):
                    continue
                if symbol in entries:
                    # Duplicate symbols are ambiguous too; omitting them is the
                    # fail-closed choice even if a third copy follows.
                    entries.pop(symbol, None)
                    ambiguous.add(symbol)
                    continue
                if _file_sha256(_file_signature(path.parent / code_object)) == "unavailable":
                    continue
                entries[symbol] = {
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "split_k_capable": bool(split_capable),
                    "manifest_sha256": _file_sha256(signature),
                    "code_object": code_object,
                }
    except (OSError, csv.Error):
        return {}
    return entries


def _kernel_artifact_snapshot(
    symbol: str,
    arch: str,
    *,
    manifest_snapshot: object = _MANIFEST_SNAPSHOT_UNSET,
) -> Tuple[Optional[IdentityItems], Tuple[object, ...]]:
    """Return immutable artifact identity plus its live filesystem token."""
    if arch != "gfx950" or _ASM_SYMBOL_RE.fullmatch(symbol) is None:
        return None, ("invalid", arch, symbol)
    if manifest_snapshot is _MANIFEST_SNAPSHOT_UNSET:
        manifest = _manifest_path(arch)
        manifest_signature = (
            _file_signature(manifest) if manifest is not None else None
        )
    else:
        manifest, manifest_signature = manifest_snapshot
    if manifest is None:
        # ``_manifest_path`` is retried on every call, so a manifest appearing
        # later changes this token on the next dispatch.
        return None, ("missing_manifest", arch)
    assert manifest_signature is not None
    entry = _load_manifest(manifest_signature).get(symbol)
    if entry is None:
        return None, ("missing_symbol", manifest_signature, symbol)
    code_signature = _file_signature(manifest.parent / str(entry["code_object"]))
    code_hash = _file_sha256(code_signature)
    if code_hash == "unavailable":
        return None, ("missing_code_object", manifest_signature, code_signature)
    identity_items: IdentityItems = (
        ("tile_m", entry["tile_m"]),
        ("tile_n", entry["tile_n"]),
        ("split_k_capable", entry["split_k_capable"]),
        ("manifest_sha256", entry["manifest_sha256"]),
        ("code_object", entry["code_object"]),
        ("code_object_sha256", code_hash),
    )
    return identity_items, ("artifact", manifest_signature, code_signature)


def _artifact_manifest_path(token: Tuple[object, ...]) -> Optional[str]:
    """Extract the selected manifest path from an artifact-state token."""
    if token and token[0] in ("artifact", "missing_symbol", "missing_code_object"):
        signature = token[1]
        if isinstance(signature, tuple) and signature:
            return str(signature[0])
    return None


def _runtime_kernel_artifact_snapshot(
    symbol: str,
    arch: str,
    *,
    manifest_snapshot: object = _MANIFEST_SNAPSHOT_UNSET,
) -> Tuple[Optional[IdentityItems], Tuple[object, ...]]:
    """Freeze the first runnable artifact identity for the life of the process.

    AITER caches an already loaded ASM kernel by symbol and does not provide a
    loader-eviction API. Consequently, replacing a manifest or code object in a
    running process cannot safely be treated as a live kernel update: the files
    would describe the new implementation while the launcher may still execute
    the old one. Once an arch/symbol pair has been observed, any content change
    fails closed until process restart, even if the manifest root changes.

    An artifact that was unavailable before its first valid observation is not
    frozen. This lets installation complete before the first launch without
    requiring a table edit or an explicit cache clear.
    """
    identity_items, artifact_token = _kernel_artifact_snapshot(
        symbol,
        arch,
        manifest_snapshot=manifest_snapshot,
    )
    manifest_path = _artifact_manifest_path(artifact_token)
    # AITER's process-local loader cache is keyed by symbol, not manifest path.
    # Keep one baseline across registry-root changes so a new path cannot make
    # metadata describe different bytes than the already loaded code object.
    baseline_key = (arch, symbol)
    warning = None

    with _runtime_artifact_lock:
        baseline_entry = _runtime_artifact_baselines.get(baseline_key)
        if identity_items is not None and manifest_path is not None:
            if baseline_entry is None:
                _runtime_artifact_baselines[baseline_key] = (
                    manifest_path,
                    identity_items,
                )
            else:
                baseline_path, baseline = baseline_entry
                if baseline != identity_items:
                    warning_token = (
                        "changed",
                        baseline_key,
                        baseline,
                        identity_items,
                    )
                    if warning_token not in _runtime_artifact_warnings:
                        _runtime_artifact_warnings.add(warning_token)
                        warning = (
                            baseline_path,
                            "changed on disk after it was first validated",
                        )
                    identity_items = None
                    artifact_token = (
                        "restart_required",
                        baseline_key,
                        baseline_path,
                        baseline,
                        artifact_token,
                    )
        elif identity_items is None and baseline_entry is not None:
            # Disappearance fails closed just like a content replacement.
            baseline_path, baseline = baseline_entry
            warning_token = ("unavailable", baseline_key, artifact_token)
            if warning_token not in _runtime_artifact_warnings:
                _runtime_artifact_warnings.add(warning_token)
                warning = (
                    baseline_path,
                    "became unavailable after it was first validated",
                )
            artifact_token = (
                "restart_required",
                baseline_key,
                baseline_path,
                baseline,
                artifact_token,
            )

    if warning is not None:
        baseline_path, reason = warning
        logger.warning(
            "MXFP4 ASM artifact %s (%s) %s; disabling this kernel until process restart",
            symbol,
            baseline_path,
            reason,
        )
    return identity_items, artifact_token


def kernel_artifact(symbol: str, arch: str) -> Optional[Dict[str, object]]:
    """Return the installed code-object identity for an approved symbol."""
    identity_items, _token = _kernel_artifact_snapshot(symbol, arch)
    return dict(identity_items) if identity_items is not None else None


def validate_tuned_entry(
    entry: object, K: int, arch: str
) -> Optional[AsmConfig]:
    """Validate one table row against the exact installed ASM implementation."""
    return _validate_tuned_entry(
        entry,
        K,
        arch,
        manifest_snapshot=_MANIFEST_SNAPSHOT_UNSET,
    )


def _validate_tuned_entry(
    entry: object,
    K: int,
    arch: str,
    *,
    manifest_snapshot: object,
) -> Optional[AsmConfig]:
    """Validate one row against a caller-provided coherent manifest view."""
    if not isinstance(entry, Mapping) or not entry:
        return None
    if "libtype" in entry and str(entry.get("libtype", "")).strip().lower() != "asm":
        return None

    symbol = entry.get("kernelName")
    if not isinstance(symbol, str) or _ASM_SYMBOL_RE.fullmatch(symbol) is None:
        return None
    artifact_items, _artifact_token = _kernel_artifact_snapshot(
        symbol,
        arch,
        manifest_snapshot=manifest_snapshot,
    )
    if artifact_items is None:
        return None
    artifact = dict(artifact_items)

    split_k = _strict_int(entry.get("splitK"))
    if split_k is None or not 0 <= split_k <= 3 or K % (1 << split_k):
        return None
    if split_k and not artifact["split_k_capable"]:
        return None
    return symbol, split_k


def _parse_table(
    path: str,
    *,
    manifest_snapshot: object,
) -> Tuple[Dict[TableKey, AsmConfig], set[TableKey]]:
    """Parse one source table; duplicate valid ASM keys are rejected."""
    accepted: Dict[TableKey, AsmConfig] = {}
    ambiguous: set[TableKey] = set()
    try:
        with open(path, newline="") as source:
            reader = csv.DictReader(source)
            has_gfx = bool(reader.fieldnames and "gfx" in reader.fieldnames)
            for row in reader:
                cu_num = _strict_int(row.get("cu_num"))
                m = _strict_int(row.get("M"))
                n = _strict_int(row.get("N"))
                k = _strict_int(row.get("K"))
                if None in (cu_num, m, n, k):
                    continue
                # Legacy tables predate the architecture column. Lumen's direct
                # A4W4 path is gfx950-only, so normalise those rows to gfx950
                # instead of keeping a second key namespace whose precedence
                # could accidentally outrank an explicitly configured table.
                gfx = (
                    str(row.get("gfx", "")).strip().lower()
                    if has_gfx
                    else "gfx950"
                )
                if has_gfx and not gfx:
                    continue
                if cu_num <= 0 or m <= 0 or n <= 0 or k <= 0:
                    continue
                config = _validate_tuned_entry(
                    row,
                    k,
                    gfx,
                    manifest_snapshot=manifest_snapshot,
                )
                if config is None:
                    continue
                key = (gfx, cu_num, m, n, k)
                if key in accepted or key in ambiguous:
                    accepted.pop(key, None)
                    ambiguous.add(key)
                else:
                    accepted[key] = config
    except (OSError, csv.Error):
        return {}, set()
    return accepted, ambiguous


@functools.lru_cache(maxsize=32)
def _merged_asm_map(
    signatures: Tuple[FileSignature, ...],
    artifact_registry_token: Tuple[object, ...],
    manifest_snapshot: ManifestSnapshot,
) -> Dict[TableKey, AsmConfig]:
    """Merge source tables in priority order after filtering to ASM only."""
    # The token is intentionally consumed only by functools.lru_cache: it makes
    # a previously missing manifest/code object trigger a fresh table parse.
    _ = artifact_registry_token
    merged: Dict[TableKey, AsmConfig] = {}
    claimed: set[TableKey] = set()
    for signature in signatures:
        path = signature[0]
        accepted, ambiguous = _parse_table(
            path,
            manifest_snapshot=manifest_snapshot,
        )
        for key in ambiguous:
            if key not in claimed:
                claimed.add(key)
                logger.warning(
                    "MXFP4 ASM table %s has duplicate approved rows for %s; "
                    "disabling that key",
                    path,
                    key,
                )
        for key, config in accepted.items():
            if key not in claimed:
                claimed.add(key)
                merged[key] = config
    return merged


def _table_signatures(paths: Optional[Sequence[str]] = None):
    source_paths = tuned_table_paths() if paths is None else paths
    return tuple(_file_signature(Path(path)) for path in source_paths)


def _artifact_registry_snapshot(
    arch: str,
) -> Tuple[ManifestSnapshot, Tuple[object, ...]]:
    """State that can make a previously unavailable table row become runnable.

    Manifest edits are represented by the manifest signature itself. A missing
    code object appearing changes the containing directory's metadata, so the
    cached table view is rebuilt without stat-ing every installed code object on
    every dispatch.
    """
    manifest = _manifest_path(arch)
    if manifest is not None:
        manifest_signature = _file_signature(manifest)
        return (
            (manifest, manifest_signature),
            (
                "selected_manifest",
                manifest_signature,
                _file_signature(manifest.parent),
            ),
        )

    candidates = []
    for root in _asm_roots():
        path = root / arch / "f4gemm" / _MANIFEST_FILENAME
        candidates.append((_file_signature(path), _file_signature(path.parent)))
    return (None, None), ("missing_manifest", tuple(candidates))


def padded_m(M: int, N: int, level: int) -> int:
    """Pure-Python equivalent of the A4W4 tuned lookup's M padding."""
    if level == 0:
        multiple = 16 if M <= 256 else 32 if M <= 1024 else 64 if M <= 4096 else 128
        return ((M + multiple - 1) // multiple) * multiple
    if level == 1:
        if M > 8192 and N > 4096:
            return 8192
        return 1 if M <= 1 else 1 << (M - 1).bit_length()
    raise ValueError(f"unsupported MXFP4 ASM padding level: {level}")


def lookup(
    M: int,
    N: int,
    K: int,
    *,
    arch: str,
    cu_num: int,
    paths: Optional[Sequence[str]] = None,
) -> Optional[AsmConfig]:
    """Find the first exact/fine/coarse ASM row for the current GPU."""
    if arch != "gfx950":
        return None
    manifest_snapshot, artifact_registry_token = _artifact_registry_snapshot(arch)
    return _lookup_from_signatures(
        M,
        N,
        K,
        arch=arch,
        cu_num=cu_num,
        signatures=_table_signatures(paths),
        artifact_registry_token=artifact_registry_token,
        manifest_snapshot=manifest_snapshot,
    )


def _lookup_from_signatures(
    M: int,
    N: int,
    K: int,
    *,
    arch: str,
    cu_num: int,
    signatures: Tuple[FileSignature, ...],
    artifact_registry_token: Tuple[object, ...],
    manifest_snapshot: ManifestSnapshot,
) -> Optional[AsmConfig]:
    """Lookup helper for callers that already captured the live table state."""
    if arch != "gfx950":
        return None
    table = _merged_asm_map(
        signatures,
        artifact_registry_token,
        manifest_snapshot,
    )
    lookup_m: List[int] = []
    for value in (M, padded_m(M, N, 0), padded_m(M, N, 1)):
        if value not in lookup_m:
            lookup_m.append(value)
    for candidate_m in lookup_m:
        config = table.get((arch, cu_num, candidate_m, N, K))
        if config is not None:
            return config
    return None


def lookup_runtime(M: int, N: int, K: int) -> Optional[AsmConfig]:
    """Look up a shape for the active GPU without invoking a GEMM dispatcher."""
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        arch = get_gfx_runtime()
        cu_num = get_cu_num()
    except Exception:
        return None
    return lookup(M, N, K, arch=arch, cu_num=cu_num)


def _runtime_snapshot_for_identity(
    M: int,
    N: int,
    K: int,
    *,
    epoch: int,
    arch: str,
    cu_num: int,
) -> RuntimeSnapshot:
    """Capture one coherent registry view for an already discovered device."""
    signatures = _table_signatures()
    manifest_snapshot, artifact_registry_token = _artifact_registry_snapshot(arch)
    config = _lookup_from_signatures(
        M,
        N,
        K,
        arch=arch,
        cu_num=cu_num,
        signatures=signatures,
        artifact_registry_token=artifact_registry_token,
        manifest_snapshot=manifest_snapshot,
    )
    artifact_items: Optional[IdentityItems] = None
    artifact_token: Tuple[object, ...] = ("no_config",)
    identity_items: Optional[IdentityItems] = None
    if config is not None:
        symbol, split_k = config
        artifact_items, artifact_token = _runtime_kernel_artifact_snapshot(
            symbol,
            arch,
            manifest_snapshot=manifest_snapshot,
        )
        if artifact_items is None:
            # A parsed table row is not runnable after its manifest/code object
            # disappears. Fail closed rather than carrying a stale config.
            config = None
        elif split_k and not dict(artifact_items)["split_k_capable"]:
            # The selected split policy must be legal in the exact manifest
            # snapshot whose artifact identity is carried through launch.
            config = None
            artifact_token = (
                "unsupported_split_k",
                split_k,
                artifact_token,
            )
        else:
            identity_items = (
                ("implementation", "asm"),
                ("kernel_name", symbol),
                ("log2_k_split", split_k),
                *artifact_items,
            )
    token = (
        "asm_runtime_snapshot_v1",
        epoch,
        arch,
        cu_num,
        signatures,
        config,
        artifact_token,
    )
    return RuntimeSnapshot(arch, cu_num, config, token, identity_items, True)


def runtime_snapshot(M: int, N: int, K: int) -> RuntimeSnapshot:
    """Capture one file-change-aware ASM resolution for a GEMM dispatch.

    A dispatch used to perform the same registry lookup up to three times: for
    legality, for profile identity validation, and again immediately before the
    direct ASM launch.  Besides repeated Python work, a table edit between those
    reads could make the launched config differ from the one just validated.

    This reads every live file signature on every call so an explicit table or
    artifact reconfiguration is visible without relying on a process-local
    frozen-registry contract.
    """
    epoch = _registry_epoch
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        arch = get_gfx_runtime()
        cu_num = get_cu_num()
    except Exception as exc:
        # Retry transient discovery failures on every dispatch rather than
        # caching an unavailable registry forever.
        return RuntimeSnapshot(
            "unknown",
            0,
            None,
            ("runtime_identity_unavailable", epoch, type(exc).__name__),
            None,
            False,
        )

    return _runtime_snapshot_for_identity(
        M, N, K, epoch=epoch, arch=arch, cu_num=cu_num
    )


def snapshot_identity(snapshot: RuntimeSnapshot) -> Optional[Dict[str, object]]:
    """Materialise the persisted identity only on a validation-cache miss."""
    return (
        dict(snapshot.identity_items)
        if snapshot.identity_items is not None
        else None
    )


def table_fingerprint(paths: Optional[Sequence[str]] = None) -> str:
    """Hash the effective ASM-only map, independent of source file paths."""
    signatures = _table_signatures(paths)
    manifest_snapshot, artifact_registry_token = _artifact_registry_snapshot("gfx950")
    table = _merged_asm_map(
        signatures,
        artifact_registry_token,
        manifest_snapshot,
    )
    canonical = []
    for key, (symbol, split_k) in sorted(table.items()):
        artifact_items, _artifact_token = _kernel_artifact_snapshot(
            symbol,
            key[0],
            manifest_snapshot=manifest_snapshot,
        )
        artifact = dict(artifact_items) if artifact_items is not None else None
        canonical.append(
            {
                "key": key,
                "symbol": symbol,
                "log2_k_split": split_k,
                "artifact": artifact,
            }
        )
    unavailable = sorted(signature[0] for signature in signatures if signature[1] < 0)
    payload = {"rows": canonical, "unavailable": unavailable}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()[:16]


def registry_fingerprint() -> str:
    """Identity of the parser and fail-closed ASM admission policy."""
    source_hash = _file_sha256(_file_signature(Path(__file__).resolve()))
    return f"{REGISTRY_POLICY_REVISION}:source={source_hash}"


def identity(config: Optional[AsmConfig], arch: str) -> Optional[Dict[str, object]]:
    """Exact symbol, split policy, manifest and code-object identity."""
    if config is None:
        return None
    symbol, split_k = config
    artifact_items, _token = _kernel_artifact_snapshot(symbol, arch)
    if artifact_items is None:
        return None
    return {
        "implementation": "asm",
        "kernel_name": symbol,
        "log2_k_split": split_k,
        **dict(artifact_items),
    }


def clear_caches() -> None:
    """Drop parsed metadata caches. Intended for tests and explicit reconfigure."""
    global _registry_epoch
    _aiter_package_dir.cache_clear()
    _load_manifest.cache_clear()
    _file_sha256.cache_clear()
    _merged_asm_map.cache_clear()
    _registry_epoch += 1
    # Do not clear _runtime_artifact_baselines: AITER may still hold the old
    # loaded kernel, so only a process restart can safely accept changed bytes.

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0
###############################################################################

"""Pick an MXFP4 GEMM backend per shape, with ASM as the protected incumbent.

Lumen measures explicit FlyDSL configurations against the prebuilt ASM kernel.
When an ASM entry exists it stays selected unless a FlyDSL configuration is at
least five percent faster. Triton kernels remain fallbacks for shapes without a
valid ASM entry and for runtime failures.

A hand-measured byte threshold does not survive a change of model: the constant
tuned on Llama 3.1 8B (28 MiB MLP weights) excludes Qwen3-8B (24 MiB) entirely.
So the first call for a shape times the legal backends and remembers the winner;
a model issues only a few dozen shapes, so this costs a second once.

Environment:
    ``LUMEN_MXFP4_AUTOTUNE=0``       fall back to the static byte thresholds
    ``LUMEN_MXFP4_FLYDSL=1``         admit FlyDSL challengers during profiling
    ``LUMEN_MXFP4_PROFILE_WARMUP``   warmup launches per candidate (default 3)
    ``LUMEN_MXFP4_PROFILE_ITERS``    timed launches per candidate (default 11)
    ``LUMEN_MXFP4_AUTOTUNE_CACHE``   JSON file to persist and reuse decisions
    ``LUMEN_MXFP4_GEMM_SHAPE_LOG``   CSV to record every shape the model issues
    ``LUMEN_MXFP4_REQUIRE_CONSENSUS`` require offline unanimous multi-GPU
                                      evidence when ``WORLD_SIZE > 1`` (default 1)
"""

import atexit
import hashlib
import importlib.metadata
import importlib.util
import json
import logging as _logging
import math
import os
import statistics
import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from lumen.ops.quantize import mxfp4_asm

_logger = _logging.getLogger(__name__)

AUTOTUNE_ENABLED = os.environ.get("LUMEN_MXFP4_AUTOTUNE", "1") == "1"
_CACHE_PATH = os.environ.get("LUMEN_MXFP4_AUTOTUNE_CACHE", "")
_SHAPE_LOG_PATH = os.environ.get("LUMEN_MXFP4_GEMM_SHAPE_LOG", "")

# Public alias used by launch scripts and tests.
AITER_TUNED_CONFIG_ENV = mxfp4_asm.TUNED_CONFIG_ENV


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


# Past the first-call JIT and cache warmup without the measurement itself
# stalling. The median of the timed iterations decides, so a stray slow one does
# not.
_WARMUP_ITERS = _positive_env_int("LUMEN_MXFP4_PROFILE_WARMUP", 3)
_TIMED_ITERS = _positive_env_int("LUMEN_MXFP4_PROFILE_ITERS", 11)
_VERIFY_RTOL = 0.1
_VERIFY_ATOL = 0.1
# Keep the temporary host transfer used by full-output verification bounded.
# The full reference remains on CPU while each candidate is copied one row
# chunk at a time, so validation does not retain two complete GPU outputs.
_VERIFY_CHUNK_BYTES = 8 * 1024 * 1024

# A FlyDSL challenger must beat an available ASM incumbent by at least this
# factor. The guard absorbs ordinary run-to-run noise and prevents a marginal
# first-call result from replacing a known-fast ASM kernel.
_SWITCH_MARGIN = 1.05
_CACHE_SCHEMA = 6
_SINGLE_DEVICE_SCOPE = "single_device"
_CONSENSUS_SCOPE = "multi_device_consensus"
_PROTECTED_ASM_POLICY = "protected_asm_consensus_required"

ShapeKey = Tuple[int, int, int]
Candidate = Tuple[str, Callable[[], torch.Tensor]]

_lock = threading.Lock()
_choice: Dict[ShapeKey, str] = {}
_profiles: Dict[ShapeKey, Dict[str, object]] = {}
_shape_log: Dict[ShapeKey, Dict[str, object]] = {}
_cache_loaded = False
_cache_dirty = False
_hooks_registered = False
_cache_decision_scope = _SINGLE_DEVICE_SCOPE
_cache_profile_device_count = 1
_decision_epoch = 0


def _arch() -> str:
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        return get_arch()
    except Exception:
        return "unknown"


def _world_size() -> int:
    """Return the launcher world size, treating malformed values as local."""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _consensus_required() -> bool:
    """Whether cached FlyDSL promotion needs unanimous multi-device evidence."""
    return (
        _world_size() > 1
        and os.environ.get("LUMEN_MXFP4_REQUIRE_CONSENSUS", "1") != "0"
    )


def replay_context() -> Tuple[int, bool]:
    """Small dynamic token for cached multi-device replay requirements."""
    world_size = _world_size()
    return (
        world_size,
        world_size > 1
        and os.environ.get("LUMEN_MXFP4_REQUIRE_CONSENSUS", "1") != "0",
    )


def decision_epoch() -> int:
    """Generation of the in-memory choice/profile maps."""
    return _decision_epoch


# ---------------------------------------------------------------------------
# Persisted decisions
# ---------------------------------------------------------------------------


def _cache_key(key: ShapeKey) -> str:
    return "{},{},{}".format(*key)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _module_origin(name: str) -> str:
    """Resolve the imported source tree so profiles cannot mix AITER installs."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return "unavailable"
    if spec is None:
        return "unavailable"
    if spec.origin:
        return os.path.realpath(spec.origin)
    locations = spec.submodule_search_locations
    if locations:
        return os.path.realpath(next(iter(locations)))
    return "unavailable"


def _file_sha256(path: str) -> str:
    if path == "unavailable" or not os.path.isfile(path):
        return "unavailable"
    try:
        with open(path, "rb") as source:
            return hashlib.sha256(source.read()).hexdigest()[:16]
    except OSError:
        return "unavailable"


def _runtime_metadata() -> Dict[str, str]:
    aiter_entrypoint = _module_origin("aiter.ops.gemm_op_a4w4")
    metadata = {
        "aiter_origin": _module_origin("aiter"),
        "aiter_a4w4_origin": aiter_entrypoint,
        "aiter_a4w4_sha256": _file_sha256(aiter_entrypoint),
        "aiter_version": _package_version("amd-aiter"),
        "flydsl_origin": _module_origin("flydsl"),
        "flydsl_version": _package_version("flydsl"),
        "torch_version": str(torch.__version__),
        "torch_hip": str(torch.version.hip),
    }
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        metadata["device_name"] = props.name
    except Exception:
        metadata["device_name"] = "unavailable"
    try:
        # Match the CU identity used by the ASM registry, including AITER's
        # CU_NUM override for partitioned or binned devices.
        from aiter.jit.utils.chip_info import get_cu_num

        metadata["device_cu_count"] = str(get_cu_num())
    except Exception:
        metadata["device_cu_count"] = "unavailable"
    return metadata


def _backend_fingerprint() -> str:
    """Identity of implementations and policy that produced a decision."""
    from lumen.ops.quantize.flydsl_mxfp4 import backend_fingerprint

    runtime = _runtime_metadata()
    source_dir = os.path.dirname(os.path.realpath(__file__))
    autotune_source = _file_sha256(os.path.realpath(__file__))
    linear_source = _file_sha256(os.path.join(source_dir, "linear.py"))
    identity = "|".join(
        [
            f"schema={_CACHE_SCHEMA}",
            f"switch_margin={_SWITCH_MARGIN}",
            f"profile_iters={_WARMUP_ITERS},{_TIMED_ITERS}",
            f"verify=full,{_VERIFY_RTOL},{_VERIFY_ATOL},{_VERIFY_CHUNK_BYTES}",
            f"lumen_mxfp4_autotune_sha256={autotune_source}",
            f"lumen_mxfp4_linear_sha256={linear_source}",
            *(f"{key}={runtime[key]}" for key in sorted(runtime)),
            mxfp4_asm.registry_fingerprint(),
            backend_fingerprint(),
        ]
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _known_backend(name: object) -> bool:
    if not isinstance(name, str):
        return False
    if name in ("asm", "shuffled", "plain"):
        return True
    from lumen.ops.quantize.flydsl_mxfp4 import is_backend_name

    return is_backend_name(name)


def _is_flydsl_backend(name: object) -> bool:
    if not isinstance(name, str):
        return False
    from lumen.ops.quantize.flydsl_mxfp4 import is_backend_name

    return is_backend_name(name)


def _positive_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _profile_has_unanimous_consensus(
    profile: object,
    name: str,
    *,
    world_size: Optional[int] = None,
    decision_scope: Optional[str] = None,
    profile_device_count: Optional[int] = None,
) -> bool:
    """Validate the offline evidence required to replay FlyDSL on many GPUs.

    ``worst_ratio`` is the largest per-device ``FlyDSL / ASM`` pairwise median.
    Requiring every profiled device to pass correctness and the protected-ASM
    gate avoids turning one rank's favourable noise into a job-wide decision.
    """
    if not isinstance(profile, dict) or profile.get("winner") != name:
        return False
    entry_scope = profile.get("decision_scope")
    entry_device_count = profile.get("profile_device_count")
    if (
        decision_scope is not None
        and entry_scope is not None
        and entry_scope != decision_scope
    ):
        return False
    if (
        profile_device_count is not None
        and entry_device_count is not None
        and entry_device_count != profile_device_count
    ):
        return False
    if decision_scope is None:
        decision_scope = (
            entry_scope if entry_scope is not None else _cache_decision_scope
        )
    if profile_device_count is None:
        profile_device_count = (
            entry_device_count
            if entry_device_count is not None
            else _cache_profile_device_count
        )
    if world_size is None:
        world_size = _world_size()
    top_count = _positive_int(profile_device_count)
    if decision_scope != _CONSENSUS_SCOPE or top_count is None:
        return False

    evidence = profile.get("consensus")
    if not isinstance(evidence, dict):
        return False
    device_count = _positive_int(evidence.get("device_count"))
    if (
        evidence.get("policy") != "unanimous"
        or evidence.get("backend") != name
        or evidence.get("arch") != _arch()
        or device_count is None
        or device_count != top_count
        or device_count < world_size
    ):
        return False

    for field in ("correctness", "gate"):
        result = evidence.get(field)
        if not isinstance(result, dict):
            return False
        passed = _positive_int(result.get("passed"))
        total = _positive_int(result.get("total"))
        if passed != device_count or total != device_count:
            return False

    worst_ratio = evidence.get("worst_ratio")
    return (
        isinstance(worst_ratio, (int, float))
        and not isinstance(worst_ratio, bool)
        and math.isfinite(float(worst_ratio))
        and float(worst_ratio) > 0
        and float(worst_ratio) <= 1.0 / _SWITCH_MARGIN
    )


def _tuned_table_paths() -> List[str]:
    """Source tables Lumen filters into its strict ASM-only lookup."""
    return list(mxfp4_asm.tuned_table_paths())


def _tuned_table_fingerprint() -> str:
    """Identity of the effective ASM-only rows, independent of file paths."""
    return mxfp4_asm.table_fingerprint()


def _load_cache() -> None:
    """Read previously measured decisions, if they still apply to this process."""
    global _cache_decision_scope, _cache_loaded, _cache_profile_device_count
    global _decision_epoch
    _cache_loaded = True
    # Even an empty or rejected load is a state transition.  A resolved dispatch
    # created before an explicit reload must not survive it.
    _decision_epoch += 1
    if not _CACHE_PATH or not os.path.exists(_CACHE_PATH):
        return
    try:
        with open(_CACHE_PATH) as f:
            blob = json.load(f)
    except (OSError, ValueError) as e:
        _logger.warning("MXFP4 autotune cache %s unreadable: %s", _CACHE_PATH, e)
        return
    if not isinstance(blob, dict):
        _logger.warning("MXFP4 autotune cache %s is not a JSON object", _CACHE_PATH)
        return
    if blob.get("schema") != _CACHE_SCHEMA:
        _logger.info(
            "MXFP4 autotune cache %s has schema %s, expected %s; ignoring",
            _CACHE_PATH, blob.get("schema"), _CACHE_SCHEMA,
        )
        return
    live_backends = _backend_fingerprint()
    if blob.get("backends") != live_backends:
        _logger.info(
            "MXFP4 autotune cache %s was measured with backend set %s, "
            "ignoring with %s",
            _CACHE_PATH, blob.get("backends"), live_backends,
        )
        return
    # A decision measured on another GPU says nothing about this one.
    if blob.get("arch") != _arch():
        _logger.info(
            "MXFP4 autotune cache %s was measured on %s, ignoring on %s",
            _CACHE_PATH, blob.get("arch"), _arch(),
        )
        return
    # Nor does one measured against a different tuned table. Caches written
    # before this field existed carry None and are discarded for the same
    # reason: there is no way to tell what they were measured against.
    live_tables = _tuned_table_fingerprint()
    if blob.get("tuned_tables") != live_tables:
        _logger.info(
            "MXFP4 autotune cache %s was measured against tuned tables %s, "
            "ignoring against %s -- decisions will be re-measured",
            _CACHE_PATH, blob.get("tuned_tables"), live_tables,
        )
        return
    decision_scope = blob.get("decision_scope", _SINGLE_DEVICE_SCOPE)
    profile_device_count = _positive_int(blob.get("profile_device_count")) or 1
    _cache_decision_scope = (
        decision_scope
        if decision_scope in (_SINGLE_DEVICE_SCOPE, _CONSENSUS_SCOPE)
        else _SINGLE_DEVICE_SCOPE
    )
    _cache_profile_device_count = profile_device_count

    loaded_profiles: Dict[ShapeKey, Dict[str, object]] = {}
    profile_items = blob.get("profiles", {})
    if not isinstance(profile_items, dict):
        profile_items = {}
    for k, profile in profile_items.items():
        try:
            m, n, kk = (int(x) for x in k.split(","))
        except (AttributeError, ValueError):
            continue
        if isinstance(profile, dict):
            loaded_profiles[(m, n, kk)] = profile

    choice_items = blob.get("choices", {})
    if not isinstance(choice_items, dict):
        choice_items = {}
    for k, name in choice_items.items():
        try:
            m, n, kk = (int(x) for x in k.split(","))
        except (AttributeError, ValueError):
            continue
        key = (m, n, kk)
        if not _known_backend(name):
            continue
        profile = loaded_profiles.get(key)
        if (
            _consensus_required()
            and _is_flydsl_backend(name)
            and not _profile_has_unanimous_consensus(
                profile,
                name,
                decision_scope=_cache_decision_scope,
                profile_device_count=_cache_profile_device_count,
            )
        ):
            _logger.warning(
                "MXFP4 autotune: ignoring cached %s for %s because it lacks "
                "unanimous %d-device consensus evidence; protected ASM will be used",
                name,
                key,
                _world_size(),
            )
            continue
        _choice.setdefault(key, name)
        if profile is not None:
            _profiles.setdefault(key, profile)
    _logger.info(
        "MXFP4 autotune: loaded %d cached decisions from %s", len(_choice), _CACHE_PATH
    )


def _is_cache_writer() -> bool:
    """Only global rank zero writes a cache shared by distributed workers."""
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        # Preserve the historical single-process behaviour for unrelated or
        # malformed environments rather than silently disabling persistence.
        return True


def _save_cache() -> None:
    if not _CACHE_PATH or not _cache_dirty or not _is_cache_writer():
        return
    blob = {
        "schema": _CACHE_SCHEMA,
        "arch": _arch(),
        "decision_scope": _cache_decision_scope,
        "profile_device_count": _cache_profile_device_count,
        "backends": _backend_fingerprint(),
        "runtime": _runtime_metadata(),
        "profile_settings": {
            "warmup_iters": _WARMUP_ITERS,
            "timed_iters": _TIMED_ITERS,
        },
        "tuned_tables": _tuned_table_fingerprint(),
        "choices": {_cache_key(k): v for k, v in sorted(_choice.items())},
        "profiles": {_cache_key(k): v for k, v in sorted(_profiles.items())},
    }
    # Rank zero writes privately, then renames: the reader either sees the old
    # file or the new one, never a partially written cache.
    tmp = "{}.{}.tmp".format(_CACHE_PATH, os.getpid())
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(blob, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _CACHE_PATH)
    except OSError as e:
        _logger.warning("could not write MXFP4 autotune cache %s: %s", _CACHE_PATH, e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Shape collection
# ---------------------------------------------------------------------------


def record_shape(key: ShapeKey, asm_available: bool, backend: str) -> None:
    """Note that the model issued this GEMM. Off unless the env var is set.

    The CSV is the input inventory for an offline ASM-versus-FlyDSL profile.
    Collecting beats deriving the shapes by hand: every linear issues three
    GEMMs, and the backward pair permutes the dims in ways that are easy to get
    wrong (a wgrad's M is the output width and its K is the token count).
    """
    if not _SHAPE_LOG_PATH:
        return
    with _lock:
        entry = _shape_log.get(key)
        if entry is None:
            _shape_log[key] = {
                "asm_available": asm_available,
                "backend": backend,
                "calls": 1,
            }
        else:
            entry["calls"] = int(entry["calls"]) + 1
            entry["backend"] = backend


def _save_shape_log() -> None:
    if not _SHAPE_LOG_PATH or not _shape_log:
        return
    try:
        os.makedirs(os.path.dirname(_SHAPE_LOG_PATH) or ".", exist_ok=True)
        with open(_SHAPE_LOG_PATH, "w") as f:
            f.write("M,N,K,asm_available,backend,calls\n")
            for (m, n, k), e in sorted(_shape_log.items()):
                f.write(
                    f"{m},{n},{k},{int(bool(e['asm_available']))},"
                    f"{e['backend']},{e['calls']}\n"
                )
        _logger.info(
            "MXFP4 shape log: wrote %d distinct shapes to %s",
            len(_shape_log), _SHAPE_LOG_PATH,
        )
    except OSError as e:
        _logger.warning("could not write MXFP4 shape log %s: %s", _SHAPE_LOG_PATH, e)


def _register_hooks() -> None:
    global _hooks_registered
    if _hooks_registered:
        return
    _hooks_registered = True
    atexit.register(_save_cache)
    atexit.register(_save_shape_log)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _select_winner(
    timings: Dict[str, float], incumbent: Optional[str]
) -> Tuple[str, str]:
    """Apply the stable-switch policy to already measured timings."""
    if not timings:
        return incumbent or "", "no successful measurements"

    def fastest(names: Sequence[str]) -> str:
        # Include the name as a tie breaker so candidate insertion order cannot
        # change a persisted decision.
        return min(names, key=lambda name: (timings[name], name))

    if incumbent == "asm":
        challengers = [name for name in timings if _is_flydsl_backend(name)]
        if incumbent not in timings:
            existing = [name for name in timings if not _is_flydsl_backend(name)]
            if existing:
                return (
                    fastest(existing),
                    "ASM measurement failed; using an existing fallback without "
                    "promoting FlyDSL",
                )
            return (
                incumbent,
                "ASM measurement failed; retaining ASM without comparative evidence",
            )
        if not challengers:
            return incumbent, "no successful FlyDSL challenger"

        # Triton remains a fallback, not a participant in the decision to
        # replace an existing ASM kernel. In particular, a faster Triton timing
        # must not hide a FlyDSL result that independently clears the margin.
        best = fastest(challengers)
        if timings[best] * _SWITCH_MARGIN > timings[incumbent]:
            return incumbent, f"FlyDSL challenger did not clear {_SWITCH_MARGIN:.3f}x margin"
        return best, f"FlyDSL challenger cleared {_SWITCH_MARGIN:.3f}x margin"

    best = fastest(tuple(timings))
    if incumbent not in timings or best == incumbent:
        return best, "fastest measured backend"
    if timings[best] * _SWITCH_MARGIN > timings[incumbent]:
        return incumbent, f"challenger did not clear {_SWITCH_MARGIN:.3f}x margin"
    return best, f"challenger cleared {_SWITCH_MARGIN:.3f}x margin"


def _median_timings(samples: Dict[str, List[float]]) -> Dict[str, float]:
    """Return one deterministic median for every backend with samples."""
    return {
        name: float(statistics.median(values))
        for name, values in samples.items()
        if values
    }


def _balanced_round_count(rounds: int) -> int:
    """Round a pairwise run up so each backend occupies each position equally."""
    return rounds + rounds % 2


def _confirm_asm_flydsl(
    key: ShapeKey,
    asm: Candidate,
    challenger: Candidate,
) -> Tuple[str, Dict[str, object]]:
    """Independently confirm an apparent FlyDSL win with a balanced ABBA race.

    The full candidate race is useful for finding the best FlyDSL configuration,
    but its launch position can still bias a close result. This second race has
    only ASM and that one challenger. Odd configured counts are rounded up so
    both kernels have exactly the same number of first- and second-position
    launches. Any launch or synchronization failure retains ASM.
    """
    asm_name, asm_fn = asm
    challenger_name, challenger_fn = challenger
    if asm_name != "asm" or not _is_flydsl_backend(challenger_name):
        raise ValueError("pairwise confirmation requires ASM and one FlyDSL challenger")

    canonical = [(asm_name, asm_fn), (challenger_name, challenger_fn)]
    warmup_rounds = _balanced_round_count(_WARMUP_ITERS)
    timed_rounds = _balanced_round_count(_TIMED_ITERS)
    warmup_counts = {name: 0 for name, _fn in canonical}
    samples: Dict[str, List[float]] = {name: [] for name, _fn in canonical}
    confirmation: Dict[str, object] = {
        "challenger": challenger_name,
        "policy": "balanced_abba",
        "switch_margin": _SWITCH_MARGIN,
        "configured_warmup_iters": _WARMUP_ITERS,
        "configured_timed_iters": _TIMED_ITERS,
        "warmup_rounds": warmup_rounds,
        "timed_rounds": timed_rounds,
    }

    def fail_closed(name: str, phase: str, iteration: int, exc: Exception):
        reason = (
            f"pairwise confirmation failed closed: {name} failed during "
            f"{phase} round {iteration}: {exc}"
        )
        confirmation.update(
            {
                "status": "failed",
                "winner": "asm",
                "reason": reason,
                "failed_backend": name,
                "failed_phase": phase,
                "failed_round": iteration,
                "warmup_launch_counts": dict(warmup_counts),
                "timed_sample_counts": {
                    backend: len(values) for backend, values in samples.items()
                },
                "timings_ms": _median_timings(samples),
            }
        )
        _logger.warning("MXFP4 autotune %s: %s", key, reason)
        return "asm", confirmation

    for iteration in range(warmup_rounds):
        launch_order = canonical if iteration % 2 == 0 else reversed(canonical)
        for name, fn in launch_order:
            try:
                fn()
                torch.cuda.synchronize()
                warmup_counts[name] += 1
            except Exception as exc:
                return fail_closed(name, "warmup", iteration, exc)

    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    except Exception as exc:
        return fail_closed("timer", "setup", -1, exc)
    for iteration in range(timed_rounds):
        launch_order = canonical if iteration % 2 == 0 else reversed(canonical)
        for name, fn in launch_order:
            try:
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples[name].append(start.elapsed_time(end))
            except Exception as exc:
                return fail_closed(name, "timed", iteration, exc)

    timings = _median_timings(samples)
    sample_counts = {name: len(values) for name, values in samples.items()}
    if set(timings) != {asm_name, challenger_name} or any(
        not math.isfinite(value) or value <= 0 for value in timings.values()
    ):
        reason = "pairwise confirmation failed closed: incomplete timing evidence"
        confirmation.update(
            {
                "status": "failed",
                "winner": "asm",
                "reason": reason,
                "warmup_launch_counts": dict(warmup_counts),
                "timed_sample_counts": sample_counts,
                "timings_ms": timings,
            }
        )
        return "asm", confirmation

    passed = timings[challenger_name] * _SWITCH_MARGIN <= timings[asm_name]
    winner = challenger_name if passed else "asm"
    reason = (
        f"pairwise confirmation {'cleared' if passed else 'did not clear'} "
        f"{_SWITCH_MARGIN:.3f}x margin"
    )
    confirmation.update(
        {
            "status": "passed" if passed else "rejected",
            "winner": winner,
            "reason": reason,
            "warmup_launch_counts": dict(warmup_counts),
            "timed_sample_counts": sample_counts,
            "timings_ms": timings,
        }
    )
    return winner, confirmation


def _verification_reference(
    output: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int], torch.dtype, torch.device]:
    """Copy a complete reference output to CPU without retaining it on GPU."""
    if not torch.is_tensor(output) or output.ndim != 2:
        raise TypeError("MXFP4 GEMM candidates must return a 2D tensor")
    rows, cols = output.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("MXFP4 GEMM candidates must return a non-empty tensor")
    reference = output.detach().to(device="cpu", copy=True)
    return reference, (rows, cols), output.dtype, output.device


def _assert_full_output_close(
    output: torch.Tensor,
    reference: torch.Tensor,
    reference_shape: Tuple[int, int],
    reference_dtype: torch.dtype,
    reference_device: torch.device,
) -> None:
    """Compare every output element while bounding temporary GPU/CPU memory."""
    if not torch.is_tensor(output) or output.ndim != 2:
        raise TypeError("MXFP4 GEMM candidates must return a 2D tensor")
    if tuple(output.shape) != reference_shape:
        raise AssertionError(
            f"MXFP4 GEMM output shape mismatch: {tuple(output.shape)} != "
            f"{reference_shape}"
        )
    if output.dtype != reference_dtype:
        raise AssertionError(
            f"MXFP4 GEMM output dtype mismatch: {output.dtype} != "
            f"{reference_dtype}"
        )
    if output.device != reference_device:
        raise AssertionError(
            f"MXFP4 GEMM output device mismatch: {output.device} != "
            f"{reference_device}"
        )

    row_bytes = max(1, output.shape[1] * output.element_size())
    rows_per_chunk = max(1, _VERIFY_CHUNK_BYTES // row_bytes)
    candidate = output.detach()
    for start in range(0, output.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, output.shape[0])
        candidate_chunk = candidate[start:stop]
        if candidate_chunk.device.type != "cpu":
            candidate_chunk = candidate_chunk.to(device="cpu")
        torch.testing.assert_close(
            candidate_chunk,
            reference[start:stop],
            rtol=_VERIFY_RTOL,
            atol=_VERIFY_ATOL,
        )


def _validate_flydsl_candidates(
    key: ShapeKey,
    candidates: Sequence[Candidate],
    incumbent: Optional[str],
) -> Tuple[List[Candidate], Dict[str, str]]:
    """Admit FlyDSL candidates only after comparing them with an existing path."""
    live = list(candidates)
    flydsl = [(name, fn) for name, fn in live if _is_flydsl_backend(name)]
    if not flydsl:
        return live, {}

    by_name = dict(live)
    reference_names = []
    if incumbent in by_name and not _is_flydsl_backend(incumbent):
        reference_names.append(incumbent)
    reference_names.extend(
        name
        for name in ("asm", "plain", "shuffled")
        if name in by_name and name not in reference_names
    )

    reference_name = None
    reference = None
    reference_shape = None
    reference_dtype = None
    reference_device = None
    for name in reference_names:
        reference_output = None
        try:
            reference_output = by_name[name]()
            (
                reference,
                reference_shape,
                reference_dtype,
                reference_device,
            ) = _verification_reference(reference_output)
            reference_name = name
            break
        except Exception as exc:
            _logger.debug(
                "MXFP4 autotune: correctness reference %s failed on %s: %s",
                name,
                key,
                exc,
            )
        finally:
            reference_output = None

    validation: Dict[str, str] = {}
    admitted = [item for item in live if not _is_flydsl_backend(item[0])]
    if reference is None:
        for name, _fn in flydsl:
            validation[name] = "rejected: no working reference backend"
        _logger.warning(
            "MXFP4 autotune %s: no existing backend produced a correctness "
            "reference; FlyDSL challengers were not profiled",
            key,
        )
        return admitted, validation

    for name, fn in flydsl:
        candidate_output = None
        try:
            candidate_output = fn()
            _assert_full_output_close(
                candidate_output,
                reference,
                reference_shape,
                reference_dtype,
                reference_device,
            )
        except Exception as exc:
            validation[name] = f"rejected against {reference_name}: {exc}"
            _logger.warning(
                "MXFP4 autotune %s: rejected %s against %s before timing: %s",
                key,
                name,
                reference_name,
                exc,
            )
        else:
            validation[name] = f"passed against {reference_name}"
            admitted.append((name, fn))
        finally:
            candidate_output = None
    return admitted, validation


def _measure(
    key: ShapeKey,
    candidates: Sequence[Candidate],
    incumbent: Optional[str] = None,
    profile_evidence: Optional[Dict[str, object]] = None,
) -> Tuple[str, Dict[str, float], str, Dict[str, str]]:
    """Time every candidate and return the stable-policy result plus evidence.

    The candidates are timed round-robin rather than one after another, and every
    iteration is synchronised. Interleaving matters because the decision is cached
    for the life of the process: measured back to back, whichever backend went
    first would absorb the cold caches and clock ramp-up and could lose a contest
    it deserves to win, permanently. Syncing each call charges every backend for
    its own launch chain rather than letting the CPU run ahead and hide it, which
    is the honest comparison for a training step issuing hundreds of GEMMs.
    """
    live, validation = _validate_flydsl_candidates(key, candidates, incumbent)
    if not live:
        raise RuntimeError(
            f"MXFP4 autotune {key}: no existing backend is available to validate "
            "a FlyDSL candidate"
        )
    samples: Dict[str, List[float]] = {name: [] for name, _ in live}

    # Warmup uses the same alternating canonical/reverse order as measurement.
    # Synchronizing each launch both prevents one candidate from running ahead
    # and attributes an asynchronous failure to the kernel that caused it.
    canonical_live = list(live)
    for iteration in range(_WARMUP_ITERS):
        launch_order = (
            canonical_live if iteration % 2 == 0 else reversed(canonical_live)
        )
        failed = set()
        for name, fn in launch_order:
            try:
                fn()
                torch.cuda.synchronize()
            except Exception as exc:
                # A backend that cannot run this shape simply loses the contest.
                samples.pop(name, None)
                failed.add(name)
                _logger.debug(
                    "MXFP4 autotune: %s failed during warmup on %s: %s",
                    name,
                    key,
                    exc,
                )
        if failed:
            canonical_live = [
                (name, fn) for name, fn in canonical_live if name not in failed
            ]
        if not canonical_live:
            break
    if not canonical_live:
        raise RuntimeError(
            f"MXFP4 autotune {key}: no backend survived profile warmup"
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Alternate canonical and reversed launch order. For two candidates this
    # produces ABBA across each pair of iterations; for three it produces
    # ABC/CBA. This balances clock/cache drift without changing the canonical
    # order used for failure removal or deterministic tie-breaking.
    for iteration in range(_TIMED_ITERS):
        launch_order = (
            canonical_live if iteration % 2 == 0 else reversed(canonical_live)
        )
        failed = set()
        for name, fn in launch_order:
            try:
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples[name].append(start.elapsed_time(end))
            except Exception as exc:
                samples.pop(name, None)
                failed.add(name)
                _logger.warning(
                    "MXFP4 autotune: %s failed during timed profile on %s: %s",
                    name,
                    key,
                    exc,
                )
        if failed:
            canonical_live = [
                (name, fn) for name, fn in canonical_live if name not in failed
            ]
        if not canonical_live:
            break

    timings = _median_timings(samples)
    if not timings:
        raise RuntimeError(
            f"MXFP4 autotune {key}: no backend completed measurement"
        )

    best, reason = _select_winner(timings, incumbent)
    if incumbent == "asm" and _is_flydsl_backend(best):
        live_by_name = dict(canonical_live)
        confirmed, confirmation = _confirm_asm_flydsl(
            key,
            ("asm", live_by_name["asm"]),
            (best, live_by_name[best]),
        )
        confirmation["preliminary_reason"] = reason
        if profile_evidence is not None:
            profile_evidence["confirmation"] = confirmation
        best = confirmed
        reason = str(confirmation["reason"])

    _logger.info(
        "MXFP4 autotune %dx%dx%d: %s -> %s",
        *key,
        ", ".join(f"{n}={timings[n]:.3f}ms" for n in sorted(timings, key=timings.get)),
        f"{best} ({reason})",
    )
    return best, timings, reason, validation


def _capturing() -> bool:
    """True inside a CUDA graph capture, where extra launches are not allowed."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _remember_protected_asm(
    key: ShapeKey,
    identities: Optional[Dict[str, object]],
) -> str:
    """Cache the no-profile multi-GPU fallback to the protected ASM incumbent."""
    global _cache_decision_scope, _cache_dirty, _cache_profile_device_count
    global _decision_epoch
    with _lock:
        existing = _choice.get(key)
        if existing is not None:
            return existing
        _choice[key] = "asm"
        _profiles[key] = {
            "winner": "asm",
            "incumbent": "asm",
            "identities": dict(identities or {}),
            "switch_margin": _SWITCH_MARGIN,
            "selection_policy": _PROTECTED_ASM_POLICY,
            "decision_scope": _SINGLE_DEVICE_SCOPE,
            "profile_device_count": 1,
            "reason": (
                "multi-device consensus is required; retaining protected ASM "
                "without per-rank online profiling"
            ),
        }
        if _cache_decision_scope != _CONSENSUS_SCOPE:
            _cache_decision_scope = _SINGLE_DEVICE_SCOPE
            _cache_profile_device_count = 1
        _cache_dirty = True
        _decision_epoch += 1
    return "asm"


def pick_backend(
    key: ShapeKey,
    candidates: Sequence[Candidate],
    fallback: Optional[str] = None,
    incumbent: Optional[str] = None,
    identities: Optional[Dict[str, object]] = None,
) -> str:
    """Return the name of the backend to use for this shape.

    ``candidates`` are the backends that can legally run the shape, cheapest
    fallback last. The first call for a shape measures them; later calls reuse
    the answer. ``fallback`` is the static policy's pick, used when autotune is
    off or cannot run. ``identities`` stores the exact kernel/config behind each
    human-readable backend label alongside the timing evidence.
    """
    _register_hooks()
    global _cache_decision_scope, _cache_dirty, _cache_profile_device_count
    global _decision_epoch

    if not _cache_loaded:
        with _lock:
            if not _cache_loaded:
                _load_cache()

    name = _choice.get(key)
    if name is not None:
        return name

    if not candidates:
        return fallback or ""
    if _consensus_required() and incumbent == "asm":
        # Do not let each rank independently promote a challenger: even a
        # balanced pairwise race has measurable device-to-device variance. A
        # multi-device job may replay an offline unanimous decision loaded
        # above, but a cold or rejected cache stays on its known-good incumbent.
        return _remember_protected_asm(key, identities)
    if len(candidates) == 1:
        if _is_flydsl_backend(candidates[0][0]):
            raise RuntimeError(
                f"MXFP4 autotune {key}: FlyDSL cannot be selected without an "
                "existing backend for correctness and performance comparison"
            )
        return candidates[0][0]
    if not AUTOTUNE_ENABLED or _capturing():
        return fallback or candidates[0][0]

    profile_evidence: Dict[str, object] = {}
    name, timings, reason, validation = _measure(
        key,
        candidates,
        incumbent=incumbent,
        profile_evidence=profile_evidence,
    )
    if incumbent == "asm" and _is_flydsl_backend(name):
        confirmation = profile_evidence.get("confirmation")
        if not (
            isinstance(confirmation, dict)
            and confirmation.get("status") == "passed"
            and confirmation.get("winner") == name
        ):
            name = "asm"
            reason = "missing successful pairwise confirmation; retaining ASM"
    # A FlyDSL result may replace ASM only after both sides produced timings.
    # If the incumbent failed to measure, return a safe existing fallback for
    # this call but leave the shape uncached so the next call retries the race.
    cacheable = name in timings and not (
        incumbent == "asm" and "asm" not in timings
    )
    with _lock:
        if cacheable:
            _choice[key] = name
        else:
            _choice.pop(key, None)
        profile = {
            "winner": name,
            "incumbent": incumbent,
            "identities": dict(identities or {}),
            "switch_margin": _SWITCH_MARGIN,
            "warmup_iters": _WARMUP_ITERS,
            "timed_iters": _TIMED_ITERS,
            "timings_ms": timings,
            "validation": validation,
            "reason": reason,
            "decision_scope": _SINGLE_DEVICE_SCOPE,
            "profile_device_count": 1,
        }
        if "confirmation" in profile_evidence:
            profile["confirmation"] = profile_evidence["confirmation"]
        _profiles[key] = profile
        if _cache_decision_scope != _CONSENSUS_SCOPE:
            _cache_decision_scope = _SINGLE_DEVICE_SCOPE
            _cache_profile_device_count = 1
        _cache_dirty = True
        _decision_epoch += 1
    return name


def configure(
    tuned_config: Optional[Union[str, Sequence[str]]] = None,
    autotune_cache: Optional[str] = None,
    merge_aiter_default: bool = True,
) -> Dict[str, str]:
    """Configure Lumen's ASM registry and persist per-shape decisions.

    ``tuned_config`` takes one path or several, highest priority first; a model's
    own table has to come ahead of the generic one. The environment-variable
    name is retained for deployment compatibility, but Lumen parses the CSVs
    itself and never hands backend selection to a mixed implementation dispatcher.

    Call once at process start, before the first MXFP4 GEMM. Lumen reads these
    source CSVs itself and admits only rows that resolve to an installed ASM
    code object.

    A tuned table only widens which shapes can reach the direct ASM kernels:
    rows outside Lumen's exact ASM symbol contract are ignored by dispatch.
    Backend correctness is covered independently; this function only controls
    availability and persistence of measured performance decisions.

    Environment variables already set win, so a job can override without edits.
    Returns what ended up in effect, for logging.
    """
    global _CACHE_PATH

    if tuned_config and not os.environ.get(AITER_TUNED_CONFIG_ENV):
        requested = [tuned_config] if isinstance(tuned_config, str) else list(tuned_config)
        paths = []
        for path in requested:
            if not os.path.exists(path):
                _logger.warning("MXFP4 tuned config %s does not exist, ignoring", path)
            else:
                paths.append(os.path.abspath(path))
        if paths:
            # Keep the installed stock table; the tables cover different shapes.
            default = mxfp4_asm.default_tuned_config() if merge_aiter_default else None
            if default and os.path.exists(default):
                paths.append(default)
            os.environ[AITER_TUNED_CONFIG_ENV] = os.pathsep.join(paths)
            mxfp4_asm.clear_caches()

    if autotune_cache and not _CACHE_PATH:
        _CACHE_PATH = os.path.abspath(autotune_cache)
        _register_hooks()
    if _CACHE_PATH and not _cache_loaded:
        # Load now rather than on the first GEMM, so the decisions are visible
        # (and logged) at startup instead of appearing mid-step.
        with _lock:
            if not _cache_loaded:
                _load_cache()

    return {
        "tuned_config": os.environ.get(AITER_TUNED_CONFIG_ENV, ""),
        "autotune_cache": _CACHE_PATH,
    }


def cached(key: ShapeKey) -> Optional[str]:
    """The decision for this shape if one is already in memory, else None.

    Lets the dispatcher skip building the candidate list on the hot path.
    """
    return _choice.get(key)


def cached_profile(key: ShapeKey) -> Optional[Dict[str, object]]:
    """The timing/identity evidence associated with a cached decision."""
    return _profiles.get(key)


def cached_profile_supports(
    key: ShapeKey,
    name: str,
    *,
    expected_identities: Optional[Dict[str, object]] = None,
    required_incumbent: Optional[str] = None,
) -> bool:
    """Whether a protected backend choice has sufficient replay evidence.

    ``expected_identities`` is deliberately checked per shape in addition to
    the cache-wide fingerprint. It catches a tuned ASM row changing within a
    process and prevents a malformed FlyDSL cache entry from naming a tile it
    never measured. When ASM is currently available, ``required_incumbent``
    makes an older FlyDSL-vs-Triton result re-enter the race against ASM.
    """
    profile = cached_profile(key)
    if not isinstance(profile, dict) or profile.get("winner") != name:
        return False
    identities = profile.get("identities")

    def valid_time(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0
        )

    if (
        not isinstance(identities, dict)
        or name not in identities
        or profile.get("switch_margin") != _SWITCH_MARGIN
    ):
        return False

    if (
        required_incumbent is not None
        and profile.get("incumbent") != required_incumbent
    ):
        return False
    if expected_identities is not None:
        for backend, identity in expected_identities.items():
            if identities.get(backend) != identity:
                return False

    # Keeping ASM is safe without inventing a timing. This profile is created
    # only when a multi-device job has no replayable unanimous promotion and
    # lets subsequent calls remain on the hot cached path.
    if profile.get("selection_policy") == _PROTECTED_ASM_POLICY:
        return name == "asm" and profile.get("incumbent") == "asm"

    timings = profile.get("timings_ms")
    if not isinstance(timings, dict) or not valid_time(timings.get(name)):
        return False

    if not _is_flydsl_backend(name):
        if name == "asm":
            return profile.get("incumbent") == "asm"
        # Triton labels are only trusted here when the caller supplies the
        # exact implementation identity. This helper gates decisions that can
        # change an operand's irreversible stored layout; a label by itself is
        # not enough evidence.
        return (
            name in ("plain", "shuffled")
            and expected_identities is not None
            and name in expected_identities
        )

    if _consensus_required() and not _profile_has_unanimous_consensus(
        profile, name
    ):
        return False

    validation = profile.get("validation")
    if not isinstance(validation, dict) or not str(validation.get(name, "")).startswith(
        "passed"
    ):
        return False
    if profile.get("incumbent") == "asm":
        asm_time = timings.get("asm")
        if not valid_time(asm_time):
            return False
        if float(timings[name]) * _SWITCH_MARGIN > float(asm_time):
            return False
        confirmation = profile.get("confirmation")
        if not isinstance(confirmation, dict):
            return False
        confirmation_timings = confirmation.get("timings_ms")
        sample_counts = confirmation.get("timed_sample_counts")
        expected_samples = _balanced_round_count(_TIMED_ITERS)
        if (
            confirmation.get("status") != "passed"
            or confirmation.get("winner") != name
            or confirmation.get("challenger") != name
            or confirmation.get("switch_margin") != _SWITCH_MARGIN
            or confirmation.get("timed_rounds") != expected_samples
            or not isinstance(confirmation_timings, dict)
            or not valid_time(confirmation_timings.get("asm"))
            or not valid_time(confirmation_timings.get(name))
            or not isinstance(sample_counts, dict)
            or sample_counts.get("asm") != expected_samples
            or sample_counts.get(name) != expected_samples
        ):
            return False
        return (
            float(confirmation_timings[name]) * _SWITCH_MARGIN
            <= float(confirmation_timings["asm"])
        )
    return True


def forget(key: ShapeKey) -> None:
    """Drop one shape's decision so the next call re-measures it.

    The dispatcher calls this when a loaded decision names a backend that is
    not legal for these operands -- a cache written against a tuned table this
    process does not have, say.
    """
    global _cache_dirty, _decision_epoch
    with _lock:
        choice_removed = _choice.pop(key, None) is not None
        profile_removed = _profiles.pop(key, None) is not None
        if choice_removed or profile_removed:
            _cache_dirty = True
            _decision_epoch += 1


def clear() -> None:
    """Drop measured decisions. For tests."""
    global _cache_decision_scope, _cache_dirty, _cache_loaded
    global _cache_profile_device_count
    global _decision_epoch
    with _lock:
        _choice.clear()
        _profiles.clear()
        _shape_log.clear()
        _cache_loaded = False
        _cache_dirty = False
        _cache_decision_scope = _SINGLE_DEVICE_SCOPE
        _cache_profile_device_count = 1
        _decision_epoch += 1
    mxfp4_asm.clear_caches()

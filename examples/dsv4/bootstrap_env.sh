#!/usr/bin/env bash
# Runtime env setup inside lumen/tests:latest for DSV4 smoke.
set -euo pipefail

BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/bootstrap}"
MILES_DIR="${MILES_DIR:-/workspace/miles}"
LUMEN_DIR="${LUMEN_DIR:-/workspace/Lumen}"
AITER_ROOT="${AITER_ROOT:-${LUMEN_DIR}/third_party/aiter}"
TILEKERNELS_DIR="${TILEKERNELS_DIR:-}"
WRITABLE_ROOT="${WRITABLE_ROOT:-/tmp/lumen-dsv4-runtime}"
LUMEN_DSV4_PRETRAIN="${LUMEN_DSV4_PRETRAIN:-0}"

export MEGATRON_PATH="${MEGATRON_PATH:-${BOOTSTRAP_DIR}/Megatron-LM}"
export MILES_SCRIPT_MEGATRON_PATH="${MEGATRON_PATH}"

mkdir -p "${WRITABLE_ROOT}"

# Writable copy of bootstrap site-packages (bootstrap mount is read-only).
# Skip when image pre-staged at ${WRITABLE_ROOT}/site-packages (lumen/dsv4-lumen:mi308x).
SITE_PKGS="${SITE_PKGS:-${WRITABLE_ROOT}/site-packages}"
if [[ ! -d "${SITE_PKGS}/tile_kernels" ]]; then
    echo "[bootstrap_env] copying site-packages -> ${SITE_PKGS} (no TE)"
    rm -rf "${SITE_PKGS}"
    mkdir -p "${SITE_PKGS}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a \
            --exclude='transformer_engine' \
            --exclude='transformer_engine*.dist-info' \
            --exclude='transformer_engine*.egg-info' \
            "${BOOTSTRAP_DIR}/site-packages/." "${SITE_PKGS}/"
    else
        cp -a "${BOOTSTRAP_DIR}/site-packages/." "${SITE_PKGS}/"
        rm -rf "${SITE_PKGS}/transformer_engine" \
            "${SITE_PKGS}"/transformer_engine*.dist-info \
            "${SITE_PKGS}"/transformer_engine*.egg-info 2>/dev/null || true
    fi
fi
export SITE_PKGS

# Disable Megatron torch.compile before fusion modules import (Lumen triton != Miles).
if [[ ! -f "${SITE_PKGS}/lumen_dsv4_bootstrap.pth" ]]; then
    cat > "${SITE_PKGS}/lumen_dsv4_bootstrap.pth" <<'PY'
import megatron.core.jit as _jit
_jit.disable_jit_fuser()
PY
fi

# tilelang + sglang: use PYTHONPATH only (avoid editable pip build on ro mount).
TILELANG_ROOT="${TILELANG_ROOT:-${WRITABLE_ROOT}/tilelang}"
if [[ ! -f "${TILELANG_ROOT}/tilelang/__init__.py" ]]; then
    echo "[bootstrap_env] copying tilelang -> ${TILELANG_ROOT}"
    rm -rf "${TILELANG_ROOT}"
    cp -a "${BOOTSTRAP_DIR}/tilelang" "${TILELANG_ROOT}"
    # Dev layout: built libs under ${TILELANG_ROOT}/build/lib. Nested 3rdparty under
    # tilelang/tilelang/ makes env.py pick the wrong (empty) lib path.
    if [[ -d "${TILELANG_ROOT}/build/lib" && -d "${TILELANG_ROOT}/tilelang/3rdparty" ]]; then
        rm -rf "${TILELANG_ROOT}/tilelang/3rdparty"
    fi
fi

SGLANG_ROOT="${SGLANG_ROOT:-${WRITABLE_ROOT}/sglang-python}"
if [[ "${LUMEN_DSV4_PRETRAIN}" != "1" ]]; then
    if [[ ! -f "${SGLANG_ROOT}/sglang/__init__.py" ]]; then
        echo "[bootstrap_env] copying sglang -> ${SGLANG_ROOT}"
        rm -rf "${SGLANG_ROOT}"
        cp -a "${BOOTSTRAP_DIR}/sglang-python" "${SGLANG_ROOT}"
    fi
fi

if [[ "${LUMEN_DSV4_PRETRAIN}" == "1" ]]; then
    if [[ -d "${MILES_DIR}" ]] && ! python -c "import miles" 2>/dev/null; then
        echo "[bootstrap_env] installing miles (no-deps, for Megatron router hook) ..."
        pip install -e "${MILES_DIR}" --no-deps -q
    fi
    if [[ -d "${MILES_DIR}" ]] && ! python -c "import backports.strenum" 2>/dev/null; then
        echo "[bootstrap_env] installing backports.strenum for Miles convert helper ..."
        pip install backports.strenum -q
    fi
    export PYTHONPATH="${MILES_DIR}:${LUMEN_DIR}:${AITER_ROOT}:${MEGATRON_PATH}:${SITE_PKGS}:${TILELANG_ROOT}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${MILES_DIR}:${LUMEN_DIR}:${AITER_ROOT}:${MEGATRON_PATH}:${SITE_PKGS}:${SGLANG_ROOT}:${TILELANG_ROOT}:${PYTHONPATH:-}"
fi

NATIVE_LIBS="${BOOTSTRAP_DIR}/native-libs"
if [[ -d "${NATIVE_LIBS}" ]]; then
    export LD_LIBRARY_PATH="${NATIVE_LIBS}:${LD_LIBRARY_PATH:-}"
fi

# mooncake engine.so needs libglog (present in Miles image, not always in lumen/tests:latest).
if ! ldconfig -p 2>/dev/null | grep -q 'libglog.so.0'; then
    echo "[bootstrap_env] installing libgoogle-glog0v5 for mooncake..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgoogle-glog0v5 >/dev/null
fi

echo "[bootstrap_env] MEGATRON_PATH=${MEGATRON_PATH}"

# Miles python deps (skip for native Megatron pretrain — no Ray/train.py).
if [[ "${LUMEN_DSV4_PRETRAIN}" != "1" ]]; then
    if ! python -c "import ray" 2>/dev/null; then
        echo "[bootstrap_env] installing miles python deps..."
        grep -vE 'nvidia-resiliency-ext|torchft-nightly' "${MILES_DIR}/requirements.txt" \
            | pip install -r /dev/stdin -q
    fi

    if ! python -c "import miles" 2>/dev/null; then
        pip install -e "${MILES_DIR}" --no-deps -q
    fi
fi

echo "[bootstrap_env] installing bootstrap runtime deps..."
pip install -r "${LUMEN_DIR}/examples/dsv4/requirements-bootstrap.txt" -q

# MORI EP: host mount replaces image-built libmori_pybinds.so — restore or rebuild.
if [[ "${LUMEN_DSV4_MOE_MORI:-0}" == "1" ]]; then
    MORI_PKG="${LUMEN_DIR}/third_party/mori/python/mori"
    MORI_CACHE=""
    for candidate in \
        "${WRITABLE_ROOT}/mori-libs" \
        "${BOOTSTRAP_DIR}/mori-libs" \
        "/opt/dsv4-runtime/mori-libs"; do
        if [[ -f "${candidate}/libmori_pybinds.so" ]]; then
            MORI_CACHE="${candidate}"
            break
        fi
    done
    if [[ ! -f "${MORI_PKG}/libmori_pybinds.so" && -n "${MORI_CACHE}" ]]; then
        echo "[bootstrap_env] restoring mori libs from ${MORI_CACHE}"
        cp -a "${MORI_CACHE}/." "${MORI_PKG}/"
    fi
    if ! python -c "import mori.cpp" 2>/dev/null; then
        echo "[bootstrap_env] building mori for LUMEN_DSV4_MOE_MORI..."
        git config --global --add safe.directory "${LUMEN_DIR}/third_party/mori" || true
        cd "${LUMEN_DIR}/third_party/mori"
        pip install setuptools-scm -q
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MORI=0.1.0 pip install -e . -q
    fi
fi

# Local TileKernels mHC: rsync onto bootstrap site-packages. Do NOT put the dev tree on
# PYTHONPATH — dev tile_kernels/__init__.py eagerly imports engram kernels that require
# a newer tilelang PassConfigKey than the bootstrap image provides.
if [[ -n "${TILEKERNELS_DIR:-}" && -d "${TILEKERNELS_DIR}/tile_kernels/mhc" && -n "${SITE_PKGS:-}" ]]; then
    _tk_dest="${SITE_PKGS}/tile_kernels"
    echo "[bootstrap_env] overlay local mHC: ${TILEKERNELS_DIR}/tile_kernels -> ${_tk_dest}"
    mkdir -p "${_tk_dest}/mhc" "${_tk_dest}/modeling/mhc"
    rsync -a "${TILEKERNELS_DIR}/tile_kernels/mhc/" "${_tk_dest}/mhc/"
    rsync -a "${TILEKERNELS_DIR}/tile_kernels/modeling/mhc/" "${_tk_dest}/modeling/mhc/"
fi

python - <<'PY'
import importlib
import os

required = ["tile_kernels", "megatron.core", "tilelang"]
optional = ["transformer_engine", "fast_hadamard_transform"]
if os.environ.get("LUMEN_DSV4_PRETRAIN", "0") == "1":
    required = ["miles", *required]
else:
    required = ["ray", "miles", "sglang", *required]
if os.environ.get("LUMEN_DSV4_MOE_MORI", "0") == "1":
    required.append("mori")

for m in required:
    try:
        mod = importlib.import_module(m)
        print(f"[bootstrap_env] OK {m}: {getattr(mod, '__file__', mod)}")
    except Exception as e:
        print(f"[bootstrap_env] FAIL {m}: {e}")
        raise SystemExit(1)

for m in optional:
    try:
        mod = importlib.import_module(m)
        print(f"[bootstrap_env] OK {m} (optional): {getattr(mod, '__file__', mod)}")
    except Exception as e:
        print(f"[bootstrap_env] SKIP {m} (optional): {e}")

if os.environ.get("V4_SPARSE_MLA_BACKEND", "triton").lower() == "triton":
    try:
        from aiter.ops.triton.attention.sparse_mla_dsv4_train import sparse_mla_dsv4_train
        from lumen.models.dsv4.ops.kernel.triton_sparse_mla import sparse_attn_triton

        print(f"[bootstrap_env] OK sparse MLA triton: {sparse_mla_dsv4_train.__name__} -> {sparse_attn_triton.__name__}")
    except Exception as e:
        print(f"[bootstrap_env] FAIL sparse MLA triton: {e}")
        raise SystemExit(1)

_mhc_backend = os.environ.get("MHC_BACKEND", "triton").lower()
try:
    import tile_kernels

    print(f"[bootstrap_env] tile_kernels: {tile_kernels.__file__}")
    from lumen.models.dsv4.ops.mhc_backend import log_mhc_backend

    print(f"[bootstrap_env] OK MHC backend={log_mhc_backend()} (MHC_BACKEND={_mhc_backend})")
except Exception as e:
    print(f"[bootstrap_env] FAIL MHC/tile_kernels: {e}")
    raise SystemExit(1)
PY

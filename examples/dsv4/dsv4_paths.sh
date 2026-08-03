#!/usr/bin/env bash
# Shared path defaults for examples/dsv4/*.sh — source, do not execute directly.
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/dsv4_paths.sh"
#
# Override via env: WORKSPACE_ROOT, DATA_ROOT, MILES_DIR, TILEKERNELS_DIR,
# MODEL_DIR, LOG_DIR, TVM_CACHE_DIR, BOOTSTRAP_DIR, NFS_ROOT, MEGATRON_PATH.

: "${SCRIPT_DIR:?dsv4_paths.sh: set SCRIPT_DIR before sourcing}"

LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${LUMEN_DIR}/.." && pwd)}"

MILES_DIR="${MILES_DIR:-${WORKSPACE_ROOT}/miles}"
TILEKERNELS_DIR="${TILEKERNELS_DIR:-${WORKSPACE_ROOT}/TileKernels}"

if [[ -z "${DATA_ROOT:-}" ]]; then
    for _dsv4_data_candidate in \
        "/nfs/data/${USER}" \
        "/mnt/data/${USER}" \
        "${WORKSPACE_ROOT}/dsv4-data"; do
        if [[ -d "${_dsv4_data_candidate}" ]]; then
            DATA_ROOT="${_dsv4_data_candidate}"
            break
        fi
    done
    DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/dsv4-data}"
fi

NFS_ROOT="${NFS_ROOT:-/nfs/data}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${DATA_ROOT}/lumen-dsv4-bootstrap}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
TVM_CACHE_DIR="${TVM_CACHE_DIR:-${DATA_ROOT}/tvm-cache}"
MEGATRON_PATH="${MEGATRON_PATH:-${DATA_ROOT}/Megatron-LM-miles-main}"

unset _dsv4_data_candidate

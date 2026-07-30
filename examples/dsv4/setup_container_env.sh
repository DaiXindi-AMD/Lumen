#!/usr/bin/env bash
# Shared in-container bootstrap for DSV4 pretrain (source, do not execute).
#
# Sets MEGATRON_PATH, PYTHONPATH (via bootstrap_env.sh), and applies MI308X tile_kernels patches.
# Optional first argument: miles tree for patch_mi308x (default /workspace/miles).

setup_dsv4_container_env() {
    local patch_miles="${1:-/workspace/miles}"
    if [[ ! -d "${patch_miles}" ]]; then
        patch_miles="/tmp"
        mkdir -p "${patch_miles}"
    fi

    if [[ -d /opt/dsv4-bootstrap && -f /opt/dsv4-bootstrap/.ready ]]; then
        unset MEGATRON_PATH PYTHONPATH
        export BOOTSTRAP_DIR=/opt/dsv4-bootstrap
        # shellcheck source=examples/dsv4/bootstrap_env.sh
        source examples/dsv4/bootstrap_env.sh
        export MEGATRON_PATH=/opt/dsv4-bootstrap/Megatron-LM
    elif [[ -d /bootstrap && -f /bootstrap/.ready ]]; then
        unset MEGATRON_PATH PYTHONPATH
        export BOOTSTRAP_DIR=/bootstrap
        # shellcheck source=examples/dsv4/bootstrap_env.sh
        source examples/dsv4/bootstrap_env.sh
        export MEGATRON_PATH=/bootstrap/Megatron-LM
    else
        export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
    fi

    if [[ -f examples/dsv4/patch_mi308x_tile_kernels.py && -n "${SITE_PKGS:-}" ]]; then
        TK_PATCH="${SITE_PKGS}/tile_kernels"
        if [[ -n "${TILEKERNELS_DIR:-}" && -d "${TILEKERNELS_DIR}/tile_kernels" ]]; then
            TK_PATCH="${TILEKERNELS_DIR}/tile_kernels"
        fi
        if [[ "${MHC_BACKEND:-triton}" == "tilelang" ]]; then
            python3 examples/dsv4/patch_mi308x_tile_kernels.py "${TK_PATCH}" "${patch_miles}"
        else
            echo "[setup] skip MI308X tilelang MHC patches (MHC_BACKEND=${MHC_BACKEND:-triton})"
        fi
    elif [[ -f /workspace/miles/docker/patch_mi308x_tile_kernels.py ]]; then
        python3 /workspace/miles/docker/patch_mi308x_tile_kernels.py /workspace/miles
    fi

    local datasets_dir="${MEGATRON_PATH}/megatron/core/datasets"
    if [[ -d "${datasets_dir}" ]] && ! compgen -G "${datasets_dir}/helpers_cpp*.so" >/dev/null; then
        echo "[setup] building Megatron helpers_cpp in ${datasets_dir} ..."
        make -C "${datasets_dir}" -j"$(nproc)"
    fi
}

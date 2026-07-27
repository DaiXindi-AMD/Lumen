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
        python3 examples/dsv4/patch_mi308x_tile_kernels.py "${SITE_PKGS}/tile_kernels" "${patch_miles}"
    elif [[ -f /workspace/miles/docker/patch_mi308x_tile_kernels.py ]]; then
        python3 /workspace/miles/docker/patch_mi308x_tile_kernels.py /workspace/miles
    fi
}

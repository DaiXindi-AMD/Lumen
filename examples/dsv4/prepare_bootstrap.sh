#!/usr/bin/env bash
# Extract Miles runtime artifacts needed by DSV4 smoke on lumen/tests:latest.
#
# Usage:
#   bash examples/dsv4/prepare_bootstrap.sh
#
# Output: ${BOOTSTRAP_DIR}/Megatron-LM, site-packages/*, tilelang, sglang

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/mnt/data/leiwu/lumen-dsv4-bootstrap}"
MILES_IMAGE="${MILES_IMAGE:-rlsys/miles:rocm7.2-mi308x}"
CONTAINER="lumen-dsv4-bootstrap-extract-$$"

mkdir -p "${BOOTSTRAP_DIR}/site-packages"

if [[ -f "${BOOTSTRAP_DIR}/.ready" ]]; then
    echo "[bootstrap] already prepared at ${BOOTSTRAP_DIR}"
    exit 0
fi

if ! docker image inspect "${MILES_IMAGE}" &>/dev/null; then
    echo "[ERROR] Miles image not found: ${MILES_IMAGE}"
    exit 1
fi

echo "[bootstrap] extracting from ${MILES_IMAGE} -> ${BOOTSTRAP_DIR}"

docker rm -f "${CONTAINER}" 2>/dev/null || true
docker create --name "${CONTAINER}" "${MILES_IMAGE}" >/dev/null

copy_site_pkg() {
    local pkg="$1"
    echo "  - site-packages/${pkg}"
    docker cp "${CONTAINER}:/opt/venv/lib/python3.10/site-packages/${pkg}" \
        "${BOOTSTRAP_DIR}/site-packages/${pkg}"
}

echo "[bootstrap] Megatron-LM"
docker cp "${CONTAINER}:/root/Megatron-LM" "${BOOTSTRAP_DIR}/Megatron-LM"

echo "[bootstrap] native libs (mooncake etcd wrapper)"
mkdir -p "${BOOTSTRAP_DIR}/native-libs"
docker cp "${CONTAINER}:/usr/local/lib/libetcd_wrapper.so" \
    "${BOOTSTRAP_DIR}/native-libs/" 2>/dev/null \
    || echo "    (skip missing libetcd_wrapper.so)"

echo "[bootstrap] python packages"
while IFS= read -r pkg || [[ -n "${pkg}" ]]; do
    [[ -z "${pkg}" || "${pkg}" =~ ^# ]] && continue
    copy_site_pkg "${pkg}" || echo "    (skip missing ${pkg})"
done < "${SCRIPT_DIR}/miles_overlay_packages.txt"

echo "[bootstrap] fast_hadamard_transform"
docker cp "${CONTAINER}:/opt/venv/lib/python3.10/site-packages/fast_hadamard_transform-1.0.4.post1-py3.10-linux-x86_64.egg" \
    "${BOOTSTRAP_DIR}/site-packages/" 2>/dev/null \
    || echo "    (skip missing fast_hadamard_transform)"

echo "[bootstrap] dist-info metadata"
docker cp "${CONTAINER}:/opt/venv/lib/python3.10/site-packages/." "${BOOTSTRAP_DIR}/site-packages-dist/" 2>/dev/null || true
# Keep only metadata wheels for packages we vendored (TE sanity check needs dist-info).
for meta in transformer_engine tile_kernels mbridge apache_tvm_ffi tvm_ffi; do
    find "${BOOTSTRAP_DIR}/site-packages-dist" -maxdepth 1 -name "${meta}*.dist-info" -exec cp -a {} "${BOOTSTRAP_DIR}/site-packages/" \; 2>/dev/null || true
    find "${BOOTSTRAP_DIR}/site-packages-dist" -maxdepth 1 -name "${meta}*.egg-info" -exec cp -a {} "${BOOTSTRAP_DIR}/site-packages/" \; 2>/dev/null || true
done
rm -rf "${BOOTSTRAP_DIR}/site-packages-dist"

echo "[bootstrap] tilelang"
docker cp "${CONTAINER}:/opt/tilelang" "${BOOTSTRAP_DIR}/tilelang"

echo "[bootstrap] sglang source tree"
docker cp "${CONTAINER}:/sgl-workspace/sglang/python" "${BOOTSTRAP_DIR}/sglang-python"

docker rm -f "${CONTAINER}" >/dev/null

touch "${BOOTSTRAP_DIR}/.ready"
echo "[bootstrap] done: ${BOOTSTRAP_DIR}"

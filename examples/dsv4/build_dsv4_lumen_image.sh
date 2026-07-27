#!/usr/bin/env bash
# Build lumen/dsv4-lumen:mi308x from lumen/tests:latest + bootstrap (no Miles base image).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/mnt/data/leiwu/lumen-dsv4-bootstrap}"
LUMEN_IMAGE="${LUMEN_IMAGE:-lumen/tests:latest}"
IMAGE="${IMAGE:-lumen/dsv4-lumen:mi308x}"
STAGING="${LUMEN_DIR}/examples/dsv4/.bootstrap-build"

if ! docker image inspect "${LUMEN_IMAGE}" &>/dev/null; then
    echo "[ERROR] Lumen base image missing: ${LUMEN_IMAGE}"
    exit 1
fi

if [[ ! -f "${BOOTSTRAP_DIR}/.ready" ]]; then
    echo "[prepare] bootstrap missing — running prepare_bootstrap.sh"
    bash "${SCRIPT_DIR}/prepare_bootstrap.sh"
fi

echo "[staging] ${BOOTSTRAP_DIR} -> ${STAGING}"
rm -rf "${STAGING}"
mkdir -p "${STAGING}"
cp -a "${BOOTSTRAP_DIR}/." "${STAGING}/"

echo "==> Building ${IMAGE} (base=${LUMEN_IMAGE})"
docker build -f "${LUMEN_DIR}/Dockerfile.dsv4-lumen" \
    --build-arg "LUMEN_IMAGE=${LUMEN_IMAGE}" \
    -t "${IMAGE}" \
    "${LUMEN_DIR}"

echo "==> Done: ${IMAGE}"
echo "Run: SKIP_PREPARE=1 LOAD_CKPT=1 TRAIN_ITERS=10 IMAGE=${IMAGE} bash examples/dsv4/run_dsv4_pretrain.sh"

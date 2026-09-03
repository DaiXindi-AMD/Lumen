# DSV4 training image: Lumen base + bootstrap artifacts (TileKernels + pip deps).
#
# Build:
#   bash examples/dsv4/build_dsv4_lumen_image.sh
#
# Requires:
#   lumen/tests:latest
#   bootstrap dir from examples/dsv4/prepare_bootstrap.sh

ARG LUMEN_IMAGE=lumen/tests:latest
FROM ${LUMEN_IMAGE}

WORKDIR /workspace/Lumen

# Bootstrap artifacts (ROCm Megatron rocm_dev, TileKernels, mbridge via pip).
# Replace bootstrap entirely (COPY merges; stale miles files remain from old base layers).
RUN rm -rf /opt/dsv4-bootstrap
COPY examples/dsv4/.bootstrap-build/ /opt/dsv4-bootstrap/
RUN test -f /opt/dsv4-bootstrap/.ready

# mooncake + native libs; Megatron fused CE needs libglog on lumen/tests:latest.
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgoogle-glog0v5 rsync \
    && rm -rf /var/lib/apt/lists/*

# Pre-stage writable runtime tree (no TE — ABI mismatch with Lumen PyTorch 2.8).
RUN bash -c 'set -euo pipefail; \
    mkdir -p /opt/dsv4-runtime/site-packages; \
    rsync -a \
        --exclude=transformer_engine \
        --exclude=transformer_engine*.dist-info \
        --exclude=transformer_engine*.egg-info \
        /opt/dsv4-bootstrap/site-packages/. /opt/dsv4-runtime/site-packages/; \
    printf "import megatron.core.jit as _jit\n_jit.disable_jit_fuser()\n" \
        > /opt/dsv4-runtime/site-packages/lumen_dsv4_bootstrap.pth'

# Lumen source (submodules must be checked out on host).
COPY . /workspace/Lumen

RUN cd /workspace/Lumen/third_party/aiter && PREBUILD_KERNELS=1 pip install -e . -q
RUN cd /workspace/Lumen/third_party/mori && \
    pip install setuptools-scm -q && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MORI=0.1.0 pip install -e . -q
RUN mkdir -p /opt/dsv4-runtime/mori-libs && \
    cp /workspace/Lumen/third_party/mori/python/mori/*.so /opt/dsv4-runtime/mori-libs/
RUN cd /workspace/Lumen && pip install -e ".[dev]" -q
RUN pip install pyarrow -q \
    && (pip install fast-hadamard-transform -q \
        || echo "[dsv4-lumen] skip fast-hadamard-transform (optional on ROCm)")

ENV V4_INDEXER_IMPL=aiter
ENV V4_SPARSE_MLA_BACKEND=triton
ENV BOOTSTRAP_DIR=/opt/dsv4-bootstrap
ENV WRITABLE_ROOT=/opt/dsv4-runtime
ENV LD_LIBRARY_PATH="/opt/dsv4-bootstrap/native-libs:${LD_LIBRARY_PATH:-}"
ENV TORCHDYNAMO_DISABLE=1
ENV PYTHONPATH="/workspace/Lumen:/workspace/Lumen/third_party/aiter:${PYTHONPATH:-}"
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WORKDIR /workspace/Lumen

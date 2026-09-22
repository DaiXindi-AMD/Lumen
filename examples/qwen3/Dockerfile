# Qwen3 MXFP4 training image: Lumen base + baked source + Qwen3 deps.
#
# Build:
#   bash examples/qwen3/build_mxfp4_lumen_image.sh
#
# Requires:
#   lumen/tests:latest   (bash build.sh)

ARG LUMEN_IMAGE=lumen/tests:latest
FROM ${LUMEN_IMAGE}

WORKDIR /workspace/Lumen

# The base image installed mori and Lumen editable against this path, so
# re-copying moves both onto this commit without reinstalling them. aiter is
# installed at /opt/aiter instead and is refreshed separately below.
# Submodules have to be checked out on the host: .dockerignore drops .git/, so
# an in-image `git submodule update` has no metadata to work from.
COPY . /workspace/Lumen

# aiter is not editable against this tree, so a moved submodule pin needs to be
# carried over explicitly. The compiled extensions stay: only Python is
# refreshed, and reinstalling would cost the PREBUILD_KERNELS .so.
RUN cp -a /workspace/Lumen/third_party/aiter/aiter/. /opt/aiter/aiter/

# Tokenizer and dataset deps; the base only installs the top-level requirements.
RUN pip install -r /workspace/Lumen/examples/qwen3/requirements.txt -q

# Re-resolve in case the copied tree moved a pin.
RUN cd /workspace/Lumen && pip install -e ".[dev]" -q

# MXFP4 defaults. Only knobs that hold for every launcher and every parallel
# layout are baked; see the block below for the ones that must not be.
ENV LUMEN_FAST_QUANT_DISPATCH=1
ENV LUMEN_FUSED_SWIGLU=1
ENV LUMEN_FUSED_RESIDUAL_NORM=1
ENV LUMEN_FUSED_RES_BWD=1
ENV LUMEN_SKIP_BACKEND_SYNC=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV GPU_MAX_HW_QUEUES=8
ENV TORCHDYNAMO_DISABLE=1
ENV USE_HIPBLASLT=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=1
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV OMP_NUM_THREADS=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV LUMEN_ROOT=/workspace/Lumen
ENV TOKENIZER_PATH=/workspace/Lumen/examples/qwen3/tokenizer

# Deliberately not baked:
#
#   AITER_CONFIG_GEMM_A4W4 — train_pretrain.sh builds the path list only while
#     this is empty, and that list is what puts the model's own tuned table in
#     front of the stock one. The two are disjoint, so any value here costs the
#     M=16384 shapes their asm kernels.
#   CUDA_DEVICE_MAX_CONNECTIONS — the tuned value is 8, which Megatron asserts
#     is exactly 1 once tensor or context parallelism is on.
#   NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME — single-node values.
#   LUMEN_MXFP4_AUTOTUNE_CACHE — a cache records which backends were reachable
#     when it was written, so a fixed path lets one bad run pin "use Triton"
#     for every shape and every later run inherits it.

WORKDIR /workspace/Lumen/examples/qwen3

CMD ["bash", "run_pretrain_qwen3_8b_mxfp4.sh"]

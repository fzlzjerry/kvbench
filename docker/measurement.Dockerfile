# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 nvidia/cuda@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c

LABEL org.kvbench.measurement.lane="phase3-phase4-bf16" \
      org.kvbench.measurement.base.manifest="sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c" \
      org.kvbench.measurement.requirements-e00.sha256="aafe68e54cb316d6bb673dbc42087b2f971ac94668973cc3f8cc555d8a0dbb29" \
      org.kvbench.measurement.requirements-phase3.sha256="cebe254a3e03a48e3e67100ce11d5623fc0dc722dc43e2f482152beb644a08e9"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-13.0 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/c++ \
    TORCH_CUDA_ARCH_LIST=12.0+PTX \
    CUDAARCHS=120 \
    CMAKE_CUDA_ARCHITECTURES=120 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    LD_LIBRARY_PATH= \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

SHELL ["/bin/bash", "--noprofile", "--norc", "-eu", "-o", "pipefail", "-c"]

# These explicit package versions preserve the validated Phase 3/4 stack.
# A distinct observed container package lock is recorded only after a real
# build; the native-host E00 lock is not container authority.
# The host driver, NVML, and nvidia-smi are supplied only by NVIDIA Container
# Toolkit at runtime; no host driver package is installed in this image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash=5.2.21-2ubuntu4 \
       binutils=2.42-4ubuntu2.10 \
       coreutils=9.4-3ubuntu6.2 \
       build-essential=12.10ubuntu1 \
       cuda-cccl-13-0=13.0.85-1 \
       cuda-crt-13-0=13.0.88-1 \
       cuda-cudart-13-0=13.0.96-1 \
       cuda-cudart-dev-13-0=13.0.96-1 \
       cuda-culibos-dev-13-0=13.0.85-1 \
       cuda-cuobjdump-13-0=13.0.85-1 \
       cuda-nvdisasm-13-0=13.0.85-1 \
       cuda-driver-dev-13-0=13.0.96-1 \
       cuda-nvcc-13-0=13.0.88-1 \
       cuda-sanitizer-13-0=13.0.85-1 \
       dash=0.5.12-6ubuntu5 \
       dpkg=1.22.6ubuntu6.6 \
       g++=4:13.2.0-7ubuntu1 \
       g++-13=13.3.0-6ubuntu2~24.04.1 \
       gcc=4:13.2.0-7ubuntu1 \
       gcc-13=13.3.0-6ubuntu2~24.04.1 \
       git=1:2.43.0-1ubuntu7.3 \
       hostname=3.23+nmu2ubuntu2 \
       libc6-dev=2.39-0ubuntu8.7 \
       libpython3.12-dev=3.12.3-1ubuntu0.15 \
       libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1 \
       ninja-build=1.11.1-2 \
       nsight-compute-2026.2.1=2026.2.1.5-1 \
       nsight-systems-2026.1.3=2026.1.3.425-261338342291v0 \
       python3.12=3.12.3-1ubuntu0.15 \
       python3.12-dev=3.12.3-1ubuntu0.15 \
       python3.12-minimal=3.12.3-1ubuntu0.15 \
       python3.12-venv=3.12.3-1ubuntu0.15 \
       systemd=255.4-1ubuntu8.16 \
       util-linux=2.39.3-9ubuntu6.5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kvbench

COPY preflight/requirements-e00.txt preflight/requirements-e00.txt
COPY preflight/requirements-phase3.txt preflight/requirements-phase3.txt

RUN python3.12 -m venv /opt/kvbench/.venv \
    && /opt/kvbench/.venv/bin/python -m pip install \
       --disable-pip-version-check \
       --no-input \
       --no-cache-dir \
       --require-hashes \
       -r preflight/requirements-e00.txt \
    && mkdir -p /opt/kvbench/.phase3/site-packages \
    && /opt/kvbench/.venv/bin/python -m pip install \
       --disable-pip-version-check \
       --no-input \
       --no-cache-dir \
       --no-deps \
       --require-hashes \
       --only-binary=:all: \
       --target /opt/kvbench/.phase3/site-packages \
       -r preflight/requirements-phase3.txt \
    && mkdir -p /run/kvbench

ENV PATH=/opt/kvbench/.venv/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN python -c 'import platform, torch, triton; assert platform.python_version() == "3.12.3"; assert torch.__version__ == "2.12.1+cu130"; assert torch.version.cuda == "13.0"; assert triton.__version__ == "3.7.1"' \
    && PYTHONPATH=/opt/kvbench/.phase3/site-packages python -c 'import numpy, transformers; assert numpy.__version__ == "2.5.1"; assert transformers.__version__ == "4.57.6"' \
    && nvcc --version | grep -F "V13.0.88" \
    && nvdisasm --version | grep -F "V13.0.85" \
    && compute-sanitizer --version | grep -F "2025.3.1.0" \
    && gcc-13 -dumpfullversion | grep -Fx "13.3.0" \
    && g++-13 -dumpfullversion | grep -Fx "13.3.0"

CMD ["/bin/bash"]

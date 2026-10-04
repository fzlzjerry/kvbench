# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268

ARG VLLM_COMMIT=752a3a504485790a2e8491cacbb35c137339ad34
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CUDAARCHS=120 \
    CMAKE_CUDA_ARCHITECTURES=120 \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_NO_USAGE_STATS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/kvbench/src

WORKDIR /opt/kvbench
COPY reference/turboquant/python-freeze.txt /opt/kvbench/reference/turboquant/python-freeze.txt
RUN python3 -m venv /opt/turboquant-reference \
    && /opt/turboquant-reference/bin/python -m pip install \
       --disable-pip-version-check --no-input --no-cache-dir \
       -r /opt/kvbench/reference/turboquant/python-freeze.txt
RUN git init /opt/vllm-source \
    && git -C /opt/vllm-source remote add origin https://github.com/vllm-project/vllm.git \
    && git -C /opt/vllm-source fetch --depth=1 origin "${VLLM_COMMIT}" \
    && git -C /opt/vllm-source update-ref refs/turboquant/locked FETCH_HEAD

COPY src /opt/kvbench/src
COPY reference/turboquant /opt/kvbench/reference/turboquant

ENTRYPOINT ["/opt/turboquant-reference/bin/python", "reference/turboquant/generate_fixtures.py", "--venv", "/opt/turboquant-reference", "--source", "/opt/vllm-source"]

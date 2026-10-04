# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44

LABEL org.kvbench.phase="9-kvquant-calibration" \
      org.kvbench.base.image="kvbench-phase9p-validation@sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44" \
      org.kvbench.method.identifier="kvquant_gqa_upstream_patch_v1" \
      org.kvbench.authority.decision="0021" \
      org.kvbench.authority.base-commit="57a238357f0ffe50084670fcd5781c9848f80ea2" \
      org.kvbench.authority.patch-sha256="db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6" \
      org.kvbench.authority.patched-tree="c4f1490c9c0c4ec46099f1e95c092516df2adb4e"

ENV PYTHONPATH=/opt/kvbench/.phase3/site-packages \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONHASHSEED=20260721 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NVIDIA_TF32_OVERRIDE=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /opt/kvbench-calibration

COPY docker/calibration-kvquant.requirements.txt /opt/kvbench-calibration/requirements.txt

RUN python3 -m pip install \
      --disable-pip-version-check \
      --no-input \
      --no-cache-dir \
      --no-deps \
      --only-binary=:all: \
      --require-hashes \
      -r /opt/kvbench-calibration/requirements.txt \
    && python3 -m pip check

RUN python3 -c 'import accelerate, datasets, numpy, pandas, pyarrow, safetensors, scipy, sklearn, torch, transformers; assert torch.__version__ == "2.12.1+cu130"; assert torch.version.cuda == "13.0"; assert transformers.__version__ == "4.57.6"; assert accelerate.__version__ == "1.10.1"; assert datasets.__version__ == "5.0.0"; assert numpy.__version__ == "2.5.1"; assert scipy.__version__ == "1.16.1"; assert sklearn.__version__ == "1.7.1"; assert pyarrow.__version__ == "25.0.0"; assert pandas.__version__ == "3.0.5"; assert safetensors.__version__ == "0.8.0"' \
    && nvcc --version | grep -F "V13.0.88" \
    && gcc-13 -dumpfullversion | grep -Fx "13.3.0"

CMD ["/bin/bash"]

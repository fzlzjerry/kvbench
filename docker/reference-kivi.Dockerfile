# syntax=docker/dockerfile:1.7
# The Make target verifies this local tag resolves to the exact config digest
# recorded below before Docker receives the build context.
FROM --platform=linux/amd64 kvbench-measurement:phase6a

ARG KIVI_SOURCE_COMMIT=876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
ARG KIVI_BUILD_REVISION

LABEL org.kvbench.reference.lane="phase7-kivi" \
      org.kvbench.reference.source.commit="876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6" \
      org.kvbench.reference.source.patched_tree="b617493dea5aff1a754cd27ad6be12ac512b2aee" \
      org.kvbench.reference.parent.config_digest="sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e" \
      org.opencontainers.image.revision="${KIVI_BUILD_REVISION}"

ENV TORCH_CUDA_ARCH_LIST=12.0+PTX \
    CUDAARCHS=120 \
    CMAKE_CUDA_ARCHITECTURES=120 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH=/opt/kivi-source/quant:/opt/kivi-source:/opt/kvbench/.phase3/site-packages \
    TRITON_CACHE_DIR=/tmp/kivi-triton-cache

WORKDIR /opt/kvbench-reference

COPY third_party/LOCK.json third_party/LOCK.json
COPY third_party/patches/kivi/manifest.json third_party/patches/kivi/manifest.json
COPY third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch
COPY scripts/validate_kivi_b019_patch.py scripts/validate_kivi_b019_patch.py
COPY reference/kivi/source_manifest.json reference/kivi/source_manifest.json
COPY reference/kivi/generate_fixtures.py reference/kivi/generate_fixtures.py
COPY reference/kivi/python-freeze.txt reference/kivi/python-freeze.txt

RUN /opt/kvbench/.venv/bin/python -c 'import importlib.metadata as metadata; from pathlib import Path; records={f"{distribution.metadata[chr(78)+chr(97)+chr(109)+chr(101)]}=={distribution.version}" for distribution in metadata.distributions()}; observed="\n".join(sorted(records,key=str.lower))+"\n"; expected=Path("reference/kivi/python-freeze.txt").read_text(encoding="utf-8"); assert observed == expected'

RUN git init /opt/kivi-source \
    && git -C /opt/kivi-source remote add origin https://github.com/jy-yuan/KIVI.git \
    && git -C /opt/kivi-source fetch --depth=1 origin "${KIVI_SOURCE_COMMIT}" \
    && git -C /opt/kivi-source checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/kivi-source rev-parse HEAD)" = "${KIVI_SOURCE_COMMIT}" \
    && git -C /opt/kivi-source apply --unidiff-zero --check \
       /opt/kvbench-reference/third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch \
    && git -C /opt/kivi-source apply --unidiff-zero \
       /opt/kvbench-reference/third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch \
    && /opt/kvbench/.venv/bin/python scripts/validate_kivi_b019_patch.py \
       --device cpu --source-root /opt/kivi-source \
    && /opt/kvbench/.venv/bin/python reference/kivi/generate_fixtures.py \
       source-probe --source-root /opt/kivi-source \
       --source-manifest reference/kivi/source_manifest.json

WORKDIR /opt/kivi-source/quant

RUN /opt/kvbench/.venv/bin/python setup.py build_ext --inplace --verbose \
    && test "$(find . -maxdepth 1 -type f -name 'kivi_gemv*.so' | wc -l)" = 1 \
    && cuobjdump --list-elf kivi_gemv*.so | grep -F '.sm_120.cubin' \
    && cuobjdump --dump-ptx kivi_gemv*.so | grep -F '.target sm_120'

WORKDIR /opt/kvbench-reference

ENTRYPOINT ["/opt/kvbench/.venv/bin/python", "reference/kivi/generate_fixtures.py"]

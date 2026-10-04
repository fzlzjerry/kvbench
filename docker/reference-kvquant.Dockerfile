ARG BASE_IMAGE=sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d
FROM ${BASE_IMAGE}

ARG TOKENIZERS_WHEEL_SHA256=9e0480c452217edd35eca56fafe2029fb4d368b7c0475f8dfa3c5c9c400a7456
COPY tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN printf '%s  %s\n' "${TOKENIZERS_WHEEL_SHA256}" \
        /tmp/tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
        | sha256sum -c - \
    && mkdir -p /opt/kvbench-reference/deps \
    && /opt/kvbench/.venv/bin/python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-deps \
        --no-index \
        --target /opt/kvbench-reference/deps \
        /tmp/tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    && rm -f \
        /tmp/tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

ENV PYTHONPATH=/opt/kvbench-reference/deps:/opt/kvbench/.phase3/site-packages

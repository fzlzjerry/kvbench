# syntax=docker/dockerfile:1.7
FROM kvbench-measurement@sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e

LABEL org.kvbench.quality.phase="QP-0" \
      org.kvbench.quality.base.digest="sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e" \
      org.kvbench.quality.dependency.delta="none-pending-QP-1-lock" \
      org.kvbench.quality.execution="locked"

# QP-0 freezes the measured runtime without guessing unresolved evaluation
# package versions. QP-1 owns the reviewed evaluation dependency lock.
RUN python -c 'import platform, torch, triton; assert platform.python_version() == "3.12.3"; assert torch.__version__ == "2.12.1+cu130"; assert torch.version.cuda == "13.0"; assert triton.__version__ == "3.7.1"' \
    && PYTHONPATH=/opt/kvbench/.phase3/site-packages python -c 'import numpy, transformers; assert numpy.__version__ == "2.5.1"; assert transformers.__version__ == "4.57.6"'

CMD ["/bin/bash"]

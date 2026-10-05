#!/bin/bash
# Task 3 retry (amendment Section 10): waits for the timing chain, installs vLLM
# 0.25.1 with uv from the mirror (new 3 h cap from the install start), then runs
# addendum/vllm_fp8.py.  Run as root in tmux.
set -u
R=/home/rockrock/kvbench-addendum-20261005
L=$R/results/addendum-20261005/logs
V=/home/rockrock/addendum-vllm-env
MIRROR=https://mirrors.aliyun.com/pypi/simple/
until grep -q CHAIN_DONE "$L/task2.log" 2>/dev/null; do sleep 60; done
START=$(date +%s); DEADLINE=$((START + 3 * 3600))
echo "task3 retry start $(date -u +%FT%TZ) start_epoch=$START deadline_epoch=$DEADLINE" >> "$L/task3-retry.log"
sudo -u rockrock -H timeout 900 "$V/bin/pip" install --progress-bar off -i "$MIRROR" uv >> "$L/task3-retry.log" 2>&1
BUDGET=$((DEADLINE - $(date +%s) - 3600))
sudo -u rockrock -H env UV_HTTP_TIMEOUT=120 timeout "$BUDGET" "$V/bin/uv" pip install --python "$V/bin/python" \
  --index-url "$MIRROR" vllm==0.25.1 >> "$L/task3-retry-install.log" 2>&1
rc=$?
echo "install rc=$rc at $(date -u +%FT%TZ)" >> "$L/task3-retry.log"
if [ "$rc" -eq 0 ]; then
  "$V/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" >> "$L/task3-retry.log" 2>&1
  cd "$R" && python3 addendum/vllm_fp8.py run --venv "$V" --deadline-epoch "$DEADLINE" >> "$L/task3-run.log" 2>&1
  echo "vllm_fp8 rc=$? at $(date -u +%FT%TZ)" >> "$L/task3-retry.log"
else
  printf '\n## %s task3 (retry)\n\nRetry installation failed or timed out (uv exit %s; budget %s s). Machine-readable reason: `retry_installation_failed`. See logs/task3-retry-install.log.\n' \
    "$(date -u +%FT%TZ)" "$rc" "$BUDGET" >> "$R/results/addendum-20261005/FAILURES.md"
fi
echo T3_RETRY_DONE >> "$L/task3-retry.log"

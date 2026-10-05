
## 2026-10-05T10:37:45.822689Z smoke

process `smoke-r0-o000-tq_k3v4_nc-s32-b1-l4096` attempt 0: failed (worker_failed) RuntimeError: split scratch was not allocated at cache construction

## 2026-10-05T10:39:59.410335Z smoke

process `smoke-r0-o000-tq_k3v4_nc-s32-b1-l4096` attempt 1: failed (worker_failed) RuntimeError: split scratch was not allocated at cache construction: cache id 136244955519056 type weakref.ProxyType init object.__init__ class is patched class False recorded ids [136245011205200] records 1

## 2026-10-05T10:41:10.966404Z task1

check `t1-greedy-s4` failed: returncode 3 No module named 'scripts.q0_cache_sensitive_correctness'

## 2026-10-05T10:41:24.784297Z task1

check `t1-greedy-s32` failed: returncode 3 No module named 'scripts.q0_cache_sensitive_correctness'

## 2026-10-05T13:07:20Z task3

vLLM 0.25.1 install attempt 1 (pypi.org, started 20:05:23 CST) stopped after 60 min by the session background-task limit while still downloading dependencies (files.pythonhosted.org throughput near zero); no GPU run affected. Attempt 2 uses the PyPI mirror https://mirrors.aliyun.com/pypi/simple/ (same packages). Task 3 cap (3 h including installation) counts from 20:05:23 CST: deadline 23:05:23 CST.

## 2026-10-05T13:25:00Z task2 (gate)

Task 2 gate 1 FAILED: 100 greedy tokens at B=1, 4K (frozen Q0 core-l4096 probe) are not identical between the existing adapter and the KV-head-grouped adapter. Machine-readable reason: `gate_failed_greedy_tokens_not_identical`.

| comparison | step-0 max abs logit diff | agreeing positions | first divergence |
|---|---|---|---|
| k4v4 existing vs grouped (m4) | 0.125 | 32/100 | 31 |
| k2v2 existing vs grouped (m4) | 0.133 | 81/100 | 81 |
| k4v4 existing vs grouped (m4, FP16 reduced-precision reduction off) | 0.125 | 32/100 | 31 |
| k2v2 existing vs grouped (m4, FP16 reduced-precision reduction off) | 0.133 | 81/100 | 81 |
| k4v4 existing vs grouped (m1x4: 4 bmm calls of M=1 over B*8 KV heads) | 0.125 | 32/100 | 31 |
| k2v2 existing vs grouped (m1x4) | 0.125 | 86/100 | 86 |
| k4v4 existing vs existing (repeat, determinism control) | 0.0 | 100/100 | none |

All other gates pass for the grouped adapter (m4): G3 items with the semantic replacement for the literal `native_gqa` source-string check (literal check fails by construction; the frozen-adapter control reproduces the literal pass), G4 (capture/replay, zero replay allocation, no fallback), G1 fixture conformance (2/2) and graph harness (1/1) unit tests, per-head output equivalence within 0.02 (FP16 max abs 6.1e-5, not bitwise), Compute Sanitizer memcheck 0 errors (k4v4, k2v2). Cause: batching the four query heads of a KV group (or batching over KV heads) changes cuBLAS kernel selection and FP16 rounding order; the existing adapter is deterministic. No Task 2 timing has run. Decision on how to proceed requested from the author (2026-10-05 ~20:20 CST).

## 2026-10-05T15:13:27Z task3 (cap reached)

Task 3 stopped at its 3 h cap (20:05:23 to 23:05:23 CST, installation included) before any vLLM run. Machine-readable reason: `cap_reached_during_environment_installation`. Completed: an isolated virtual environment (/home/rockrock/addendum-vllm-env, outside the measurement container). Attempt 1 of `pip install vllm==0.25.1` from pypi.org downloaded dependencies for 60 min at near-zero throughput and was stopped by the session's background-task limit. Attempt 2 from the mirror mirrors.aliyun.com (21:07 CST) was still downloading dependencies at 23:12 CST (last: triton 3.8.0 wheel, 248 MB; pip cache 916 MB in total) and was stopped so that no installation runs during the next timing task. vLLM was never imported: there is no sm_120 compatibility result and no FP8 or BF16 vLLM measurement. Logs: logs/task3-install-vllm-0.25.1.log, logs/task3-install-vllm-0.25.1-attempt2-aliyun.log.

## 2026-10-05T20:16:43.244582Z task3

`task3-r0-o000-auto-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:17:15.142168Z task3

`task3-r0-o001-auto-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:17:43.390449Z task3

`task3-r0-o002-auto-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:18:11.704208Z task3

`task3-r0-o003-auto-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:18:40.090522Z task3

`task3-r0-o004-auto-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:19:08.477212Z task3

`task3-r0-o005-auto-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:20:13.036682Z task3

`task3-r0-o006-fp8-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:20:42.162652Z task3

`task3-r0-o007-fp8-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:21:11.005677Z task3

`task3-r0-o008-fp8-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:21:39.916479Z task3

`task3-r0-o009-fp8-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:22:08.789391Z task3

`task3-r0-o010-fp8-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:22:38.101819Z task3

`task3-r0-o011-fp8-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:23:06.465237Z task3

`task3-r1-o000-auto-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:23:34.743481Z task3

`task3-r1-o001-auto-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:24:03.170395Z task3

`task3-r1-o002-auto-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:24:31.629257Z task3

`task3-r1-o003-auto-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:25:00.048394Z task3

`task3-r1-o004-auto-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:25:28.493022Z task3

`task3-r1-o005-auto-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:25:57.321896Z task3

`task3-r1-o006-fp8-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:26:26.193012Z task3

`task3-r1-o007-fp8-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:26:55.032620Z task3

`task3-r1-o008-fp8-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:27:23.938364Z task3

`task3-r1-o009-fp8-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:27:52.754623Z task3

`task3-r1-o010-fp8-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:28:21.601711Z task3

`task3-r1-o011-fp8-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:28:50.387741Z task3

`task3-r2-o000-fp8-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:29:19.202612Z task3

`task3-r2-o001-fp8-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:29:48.117977Z task3

`task3-r2-o002-fp8-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:30:16.940279Z task3

`task3-r2-o003-fp8-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:30:45.797836Z task3

`task3-r2-o004-fp8-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:31:14.552045Z task3

`task3-r2-o005-fp8-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:31:42.939596Z task3

`task3-r2-o006-auto-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:32:11.300148Z task3

`task3-r2-o007-auto-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:32:39.653049Z task3

`task3-r2-o008-auto-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:33:08.027128Z task3

`task3-r2-o009-auto-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:33:36.355581Z task3

`task3-r2-o010-auto-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:34:04.810940Z task3

`task3-r2-o011-auto-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:36:01Z task3 (retry, first pass)

All 36 Task 3 processes of the retry's first pass (04:14-04:34 CST) failed before measuring: the vLLM engine did not start because FlashInfer 0.6.13 JIT-compiles its sampling ops for sm_120 with the host nvcc (/usr/local/cuda-13.0), whose toolkit has no curand headers (`sampling.cuh: fatal error: curand.h: No such file or directory`). Machine-readable reason: `engine_start_failed_missing_curand_headers`. These are environment (infrastructure) failures: no latency was measured. Fix: a directory with symlinks to the curand headers of the venv's own nvidia-curand (cu13) package, passed as CPATH to the vLLM processes (nothing in /usr/local/cuda or the measurement container changed; FlashInfer sampler left at the vLLM default). Each process is rerun once (attempt1) within the retry's cap (deadline 06:35:35 CST). The attention backend vLLM selected for the BF16 runs was FLASH_ATTN.

## 2026-10-05T20:58:13Z task3 (retry, second pass stopped)

The retry's second pass (prefix caching off, the `vllm bench latency` default) completed 2 of 36 processes (BF16, B=1, 128K, output_len 1 and 65; 22.85 s and 24.23 s per iteration, full prefill every iteration) and was stopped at 04:57 CST by author decision because the remaining processes could not finish before the cap. Machine-readable reason: `stopped_protocol_change_prefix_caching`. The process in flight (task3-r0-o002) was killed and has no result. Task 3 continues as task3b with --enable-prefix-caching (amendment Section 10).

## 2026-10-05T21:39:09Z task5 (optional; not run)

Optional Task 5 (KVQuant identity pass-through diagnostic) was not started. Machine-readable reason: `implementation_estimate_exceeds_1h`. Read-only assessment: an exact-codebook identity is impossible (per-channel Key tables and per-token Value tables of 2-16 levels; the deterministic Value kernels require exactly 12 outliers); the most faithful feasible design (a proxy around the KVQuant extension that forwards all store kernels and replaces only the Key-RoPE and Value decode kernels with exact FP32 computation from a BF16 side cache) needs about 150-200 lines plus validation and a GPU run, estimated at 2-3 h end to end, above the preregistered 1 h condition.

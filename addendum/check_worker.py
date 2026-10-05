#!/usr/bin/env python3
"""Addendum correctness worker (never timing): greedy decode with a growing cache.

Runs inside the measurement container like timing_worker.py.  Builds the
method exactly as the Q0 cache-sensitive correctness stage does
(scripts/q0_cache_sensitive_correctness.py _allocate_state/_prefill/_finish_step),
prefills the frozen Q0 core-l4096 probe, feeds its burn-in token, and then
decodes greedily for --steps tokens, each step's input being the previous
step's argmax.  Logits of every step and the generated tokens are saved; the
comparison between variants is done offline (compare_checks.py).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import overrides  # noqa: E402
from timing_worker import install_variant  # noqa: E402

RESULT_PREFIX = "ADDENDUM_CHECK_RESULT="
SCHEMA = "kvbench-addendum-20261005-check-1.0.0"


def greedy(loaded, configuration: str, prefix_ids: list[int], first_token: int, steps: int):
    import torch
    import scripts.q0_cache_sensitive_correctness as q0

    length = len(prefix_ids)
    method, cache, endpoint, positions, rope = q0._allocate_state(
        loaded, configuration, batch=1, capacity=length + steps, mode="growing",
        prefix_length=length, output_steps=steps)
    selected = q0.family(configuration)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=cache.device).reshape(1, length)
    token = torch.tensor([[first_token]], dtype=torch.long, device=cache.device)
    logits, tokens = [], []
    with torch.inference_mode():
        q0._prefill(endpoint, prefix, selected)
        cache.prepare_growing(length, steps)
        for step in range(steps):
            cache.select_growing_step(step)
            output = endpoint.decode(token, positions[step], rope[step])
            row = output.detach().to(device="cpu", dtype=torch.float32, copy=True).reshape(-1)
            logits.append(row)
            tokens.append(int(row.argmax().item()))
            q0._finish_step(cache, selected)
            token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=cache.device)
    evidence = {
        "method": selected,
        "adapter_config_fingerprint": method.config_fingerprint(cache.layout_fingerprint()),
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "prefix_length": length,
        "decode_steps": steps,
        "active_context": int(cache.active_context),
        "finite": all(bool(torch.isfinite(row).all()) for row in logits),
    }
    return torch.stack(logits), tokens, evidence


def run(args: argparse.Namespace) -> dict:
    variant = json.loads(args.variant)
    state = install_variant(variant)
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    probe = json.loads(Path(args.probe).read_text())
    prefix = [int(v) for v in probe["input_ids"]][-args.prefix_tokens:]
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    with torch.inference_mode(), forced_flash_execution():
        logits, tokens, evidence = greedy(loaded, args.configuration, prefix,
                                          int(probe["burn_in_token"]), args.steps)
    out = Path(args.output_dir)
    torch.save({"logits": logits, "tokens": tokens}, out / "greedy.pt")
    return {
        "schema_version": SCHEMA, "status": "completed", "mode": "greedy",
        "configuration": args.configuration, "variant": variant,
        "variant_state": overrides.describe(state),
        "probe_sha256": common.sha256_file(Path(args.probe)),
        "probe_stage": probe["stage"], "prefix_tokens": len(prefix), "steps": args.steps,
        "tokens": tokens, "evidence": evidence,
        "greedy_pt_sha256": common.sha256_file(out / "greedy.pt"),
        "execution_git_sha": common.EXECUTION_SHA,
        "addendum_git_sha": args.addendum_commit,
        "container_digest": os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST"),
        "worker_sha256": common.sha256_file(Path(__file__)),
        "overrides_sha256": common.sha256_file(Path(overrides.__file__)),
        "torch_version": torch.__version__,
        "performance_timing_collected": False,
        "finished_at_utc": common.utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--variant", default="{}")
    parser.add_argument("--probe", default=f"{common.CONTAINER_INPUTS}/q0_core_l4096.json")
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--addendum-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    code = 0
    try:
        result = run(args)
    except BaseException as error:
        result = {"schema_version": SCHEMA, "status": "failed", "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "finished_at_utc": common.utc_now()}
        code = 3
    text = json.dumps(result, sort_keys=True, default=str)
    common.write_new(Path(args.output_dir) / "worker_result.json", text + "\n")
    sys.stdout.write(RESULT_PREFIX + text + "\n")
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()

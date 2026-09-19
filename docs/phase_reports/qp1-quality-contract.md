# QP-1 Quality Contract

Status: **PASS — preparation complete; human approval pending**

The six technical QP-1 items are complete. The approval target is contract
`quality-qp1-20260918t170812117127z-170b638c-e8f4a2`, SHA-256
`b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
Quality execution remains **LOCKED**; Q0, PPL, LongBench inference, and all
quality scoring remain **NOT RUN**.

## Binding and runtime

- Corrected performance freeze tag: `perf-freeze-20260917-83536c37-r1`.
- Final performance manifest SHA-256:
  `91db28a33940e9bbdda6c723a2678ae9459e73f813b20bfecb6c97222ba38fc5`.
- Freeze root: `9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f`.
- Correction root: `44c9594079e372d708d509449589a428f4b27c0392e2f1b1e491acc3d17e2559`.
- The exact ten measured configurations and fingerprints are retained. Quality
  uses growing-context/cache-sensitive execution and must prove Q0 equivalence;
  fixed-L performance fingerprints are not asserted as universal geometry.
- Measurement image remains
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
  Quality derivative remains
  `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`.
  No measured CUDA/PyTorch/Triton or method runtime changed.
- CPU preparation is locked to Python 3.12.3, Transformers 4.57.6,
  tokenizers 0.22.2, Hugging Face Hub 0.36.2, safetensors 0.8.0,
  Jinja2 3.1.6, NumPy 2.5.1, and PyArrow 25.0.0. Official future metric
  dependencies `rouge==1.0.1`, `jieba==0.42.1`, and
  `fuzzywuzzy==0.18.0` are version- and artifact-hash locked.

## Tokenizer, datasets, and selected inputs

- Frozen tokenizer revision:
  `0e9e39f249a16976918f6564b8830bc894c89659`; tokenizer-config SHA-256
  `177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424`;
  raw chat-template SHA-256
  `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`.
- WikiText-2 raw test revision:
  `b08601e04326c79dfdd32d625aee71d232d685c3`; 2,891 non-empty documents,
  291,827 stream tokens.
- C4 validation revision:
  `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`; 2,048 documents selected
  without replacement from the declared first-4,096-row partial-viewer frame,
  seed 20260722; 935,832 stream tokens. This is explicitly not a uniform sample
  of all C4 validation.
- LongBench-E dataset revision:
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`; official source commit
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`; all 13 required `_e` tasks,
  3,668 selected and model-length eligible samples, zero exclusions.
- LongBench-v2 revision:
  `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`; all 503 sample IDs are
  frozen. Under the primary no-CoT budget, 321 are eligible and 182 are
  explicitly excluded by the model-length equation. The separately frozen
  128-sample CoT stress subset has 78 eligible and 50 length exclusions.
- LongBench prompts preserve the official no-chat-wrapper exceptions for
  `trec`, `triviaqa`, `samsum`, `lcc`, and `repobench-p`. All other selected
  E tasks and LongBench-v2 use the frozen Llama-3.1 chat template once, with
  generation prompt enabled. Every sample stores prompt-text and final token-ID
  hashes, token length, generation budget, parser, and eligibility.
- The cache-sensitive split is fixed to the final 16 prompt token IDs:
  `prompt_ids = prefill_ids + conditioning_ids`; answer generation follows all
  16 conditioning decodes. The same IDs are used by BF16 and every compressed
  method.

PPL fixtures contain one token stream per dataset rather than copied prefixes.
Fast PPL freezes 128 anchors and 16,384 scored tokens across both datasets;
Full PPL freezes 1,024 anchors and 262,144 scored tokens. Every scored horizon
is kept within one source document after the 32-token boundary-ignore region;
the one burn-in token is never scored.

## Gates and resolved choices

Inherited margins are unchanged: PPL global 1%, length review 2%, and hard
fail 5%; LongBench-E macro 2 score points, category review/hard 3/5 points,
and invalid increase 1 percentage point; LongBench-v2 accuracy 2 percentage
points, category/length hard 5 points, invalid increase 1 point, and BF16-
correct retention at least 95% with CI.

Mechanical choices resolved now, after performance but before quality outputs,
are: 10,000 paired bootstrap draws with seed 20260722; document-clustered PPL
and task/length-stratified LongBench resampling; one approval-gated equal-size
extension on insufficient samples; explicit left-inclusive length buckets;
the bounded C4 frame above; a 128-sample length/difficulty-stratified CoT
subset; 1,024-token CoT plus 8-token answer budgets; and deterministic finalist
tie rules. Compression finalist comparison uses `r_alloc` only at the fixed
B=1, historical-L=131071 Graph geometry and never speed.

## Publication and decision

The 62-object bundle root is
`6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62`
at
`r2://kvbench-artifacts/kvbench/sha256/6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62/`.
Publication was COMPLETE-last and one clean retrieval validated every object,
inventory entry, checksum, and root under the private indefinite Bucket Lock.

Only one action remains: a human must approve this exact contract ID and
SHA-256. That approval is not granted by this report.

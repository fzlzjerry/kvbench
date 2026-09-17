# QP-0 Performance Freeze

Status: **PARTIAL**

The compact performance-freeze bundle is complete and durably published, but
the sole local annotated tag contains a contradictory redundant manifest
SHA-256 field. Its target, freeze ID, and content-addressed bundle root are
correct. The tag was not moved, deleted, force-replaced, or pushed. Under the
fail-closed freeze-marker rule, `PERFORMANCE_DATA_FROZEN` remains absent for
human review.

## Frozen release

- Performance source: `83536c37433875cda98c36e2848e05692e9407d0`.
- Freeze ID: `qp0-freeze-20260917t115200000000z-83536c37-bf28e4`.
- Inventory: 2,670 logical replicate slots; 2,205 accepted observations;
  465 capacity-infeasible slots; 38 replacement links with one accepted
  observation per logical slot.
- Locked tracked paths: 68, with Git blob and SHA-256 identities in
  `locked_hot_path_hashes.json`; external kernels remain bound through their
  existing source, patch, extension, and binary evidence references.
- The selected Phase 17 model remains D. All missed modeling targets,
  23/50 identified knees, macro-label interpretation, BF16 edge-holdout tail,
  and outer-selection limitation remain unchanged.

## Tag and marker

- Local tag: `perf-freeze-20260917-83536c37`.
- Target: `83536c37433875cda98c36e2848e05692e9407d0`.
- Tag object: `c5049cf73c78a96729c4e8a791ae77acf7a1963c`.
- Annotation freeze root: `9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f`
  (correct).
- Annotation manifest SHA-256: `3d14895e...`; actual manifest SHA-256:
  `91db28a...` (mismatch).
- Tag push: no.
- Freeze marker: absent, intentionally withheld pending human disposition of
  the immutable tag annotation inconsistency.

## Images and quality handoff

- Measurement image:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
- Quality derivative image:
  `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`.
- Runtime delta: none. The QP-1 evaluation dependency lock remains pending.
- `configs/quality/quality_contract.yaml` is manifest-bound and has status
  `requires_human_approval`; no approval is recorded.
- Pending QP-1 items: quality dependency lock, chat-template hash, dataset
  revisions, sample IDs, prompt hashes, fixture materialization, and formal
  human approval.

## Publication and scope

The 14-object bundle root
`9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f`
was published with `COMPLETE` last and passed one clean retrieval under the
private indefinite Bucket Lock. No historical bulk bundle was downloaded or
reuploaded. No GPU job, performance run, refit, admission, reproduction, Q0,
PPL, LongBench, invariance, or other quality evaluation ran. Quality remains
**LOCKED** and human approval is not granted.

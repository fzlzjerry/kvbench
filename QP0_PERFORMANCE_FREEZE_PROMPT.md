# QP-0 — Freeze the Completed Performance Release and Prepare the Quality Handoff

Execute the existing QP-0 now. Prepare the manifest-bound quality contract for human review, but do not start Q0, PPL, LongBench, or any other quality experiment.

This is a release handoff, not another admission campaign. Do not rerun performance measurements, model selection, cross-validation, reproduction, profilers, or historical validation suites.

## 1. Read the existing protocol and use its actual paths

Read:
- AGENTS.md and the current status/task entry.
- CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md: document precedence, QP-0, QP-1, configuration identity, cache-sensitive execution, and contract template.
- CODEX_QUALITY_EVALUATION_ADDENDUM.md only where the authoritative post-performance protocol references it. Do not revive superseded rules.
- docs/phase_reports/phase18-reproduction.md.
- docs/evidence/phase18/quality-release-manifest-draft.json.
- The compact Phase 16R, Phase 17, Phase 18, and mechanism-study manifests/receipts referenced by that draft.
- Existing freeze/tag helpers and quality-contract template, if present.

Use the repository's current accepted protocol as authority for exact filenames and freeze-marker timing. Do not invent another numbered stage, a new approval framework, or new quality thresholds.

At entry run only:

```bash
git status --short
git rev-parse HEAD
```

Expected completed performance release HEAD:

```text
83536c37433875cda98c36e2848e05692e9407d0
```

Record a legitimate later documentation-only descendant rather than resetting Git. Do not discard unrelated changes. Check the existing coordinator/process registry once to ensure no performance writer remains active; do not launch CUDA or repeat GPU admission.

Accept existing publication receipts. Fetch only a specifically missing compact manifest or checksum ledger. Do not download historical raw campaigns, prefix catalogs, model weights, Fisher payloads, or profiler archives.

## 2. Carry forward the completed release accurately

Known identities from the Phase 18 report:

```text
Phase 18 package execution HEAD:
a866fa6859e8128cc3ea21d340be39b66d034efe

Phase 18 final HEAD:
83536c37433875cda98c36e2848e05692e9407d0

Phase 18 bundle:
phase18-20260917t024901884906z-a866fa68-6cfb61

Phase 18 root:
cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2

Phase 18 R2 URI:
r2://kvbench-artifacts/kvbench/sha256/cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2/

Phase 17 root:
05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb
```

Read exact remaining roots and identities from the draft and its references; do not guess them.

Retain the scientific outcome unchanged:
- D is the selected predictive model and deployment default.
- Median/P95 relative-error, speedup-sign, and ranking targets were missed.
- Knee-relative-error is not evaluable; 23/50 curves have identified knees.
- Macro metrics average 11 applicable geometry-holdout cells, not pooled quantiles.
- BF16's B=1 short-context batch-holdout failure is retained, not clipped or refitted.
- Outer-selected scores retain their selection limitation.
- Individual fold parameter vectors were not saved. Preserve this disclosed limitation; do not reconstruct them just to freeze the release or claim independent fold-parameter reproducibility.
- Quality remains unvalidated. Freeze completion does not mean acceptable model quality or predictive accuracy.

## 3. Complete the performance inventory from existing evidence

Extend the Phase 18 handoff draft into the protocol's release inventory under:

```text
artifacts/performance_freeze/<freeze_id>/
```

Reuse the existing inventory schema/writer. One small CPU-side assembly script is sufficient if a helper is missing.

The inventory must bind:
- Phase 16R's five replicate segments, outer bundle, and authoritative host-wall supplement.
- Accepted run IDs, original/replacement linkage, and terminal feasibility records.
- Phase 17 predictions, splits, candidate definitions, fitted models, metrics, and limitations.
- Phase 18 reproduction package, reporting audit, predictor documentation, and research report.
- Pilot/densification, Graph A/B/closure, and profiler roots actually used as design or mechanism evidence, with their distinct roles.
- Exact method, calibration, reference, model/tokenizer, container, and recorded hardware identities needed to interpret and later join the results.

For run rows use the protocol's fields: run_id, method, method_config_id, method_config_fingerprint, model/tokenizer revisions, adapter/kernel source and binary hashes, container digest, execution Git SHA, B, L, graph mode, terminal status, raw locator and checksum, quality_status, and claim_eligibility.

Keep separate concepts separate:
- 2,670 logical replicate slots: 2,205 accepted observations and 465 capacity-infeasible slots.
- 441 feasible logical points with five accepted replicates each.
- 38 infrastructure replacements contribute one accepted observation per replaced slot, not additional statistical weight.
- Original failed/replaced attempts remain referenced as excluded provenance.
- Stopped campaigns remain historical evidence, not additional fitting data.
- A host-wall supplement is an authority correction/join, not another observation.
- Timing, profiler, model, and reproduction artifacts have separate roles.

Record the historical source hashes already bound by their ledgers. Hash the newly assembled inventory and release files now. Label inherited digest checks as inherited receipt/ledger verification, not a fresh re-read of every remote payload. Do not mislabel a bundle root as an individual raw-file checksum.

No mass raw-data download or re-upload is required. If a release-critical checksum or reference is genuinely missing, fetch only that manifest/ledger. Do not infer it.

## 4. Freeze the measured implementation, not the entire machine

Create the protocol's locked_paths.txt and a corresponding hash list from the frozen performance source.

Cover the existing adapter, runtime, model-forward, CUDA/Triton kernel, method-config, and model-config paths specified by the protocol. Bind external source patches and kernel binaries through their existing manifests and hashes as well; a Git path list alone does not cover an external extension.

Record:
- Exact protected tracked paths and Git blobs/SHA-256 values.
- External source/extension identities and their evidence references.
- Measurement image digest and recorded package/runtime identity.
- Historical GPU/hardware/driver identity used for the measurements.
- Model/tokenizer revision and configuration hashes.

Do not replace historical hardware identity with the current host's identity. Reuse existing hardware manifests; a changed host is something to record for future quality setup, not grounds to rerun completed performance experiments.

Do not change filesystem permissions recursively, install monitoring services, modify host packages, rebuild the Measurement Container, or rewrite a method fingerprint. A manifest and the existing targeted diff check are enough.

Quality code must later live outside the locked paths. No adapter, kernel, calibration, threshold, sink/residual policy, or cache-layout change is authorized here.

## 5. Create one local performance-freeze tag without circular identifiers

Creating one local annotated performance-freeze tag is authorized by this task. Pushing a tag, Git branch, release, or PR is not authorized.

Prefer the protocol's naming form:

```text
perf-freeze-<YYYYMMDD>-<short_performance_source_sha>
```

Use the actual task date. Bind the tag to the completed performance source commit, not an unrelated quality harness commit. Distinguish:
- performance_source_commit;
- QP-0 metadata/preparation commit;
- historical per-run execution commits.

Keep identification acyclic:
1. Choose the existing performance commit and intended tag name.
2. Finalize the freeze manifest/inventory and calculate their digests.
3. Create the annotated local tag targeting that performance commit, with the manifest identity in its annotation.
4. Record the tag object ID and release-publication receipt in a separate small receipt outside the bundle they name.

Do not put a tag object hash inside a manifest whose hash that same tag must contain. Do not create nested bundles merely to include a receipt referring to its own root.

If the exact tag already exists and matches, reuse it. Do not move, delete, or force-replace any existing freeze tag.

Handle PERFORMANCE_DATA_FROZEN through the repository's existing freeze helper and consumer semantics. If the marker denotes completed performance freezing, create it only after those requirements are met, while retaining Quality LOCKED. If the accepted protocol defers it to human approval, leave it pending and report that fact. The marker must never fabricate human approval or launch quality jobs.

## 6. Prepare the Quality image as required by QP-0

The saved post-performance protocol includes a Quality image in QP-0. Reuse an existing compatible image if already present; otherwise use a thin derivative of the exact frozen Measurement image.

Keep CUDA, PyTorch, Triton, method kernels, and the driver-visible runtime unchanged. Add only the evaluation/dataset/statistics dependencies already specified by the quality protocol or its lock. Do not replace kernel binaries or silently upgrade runtime packages to satisfy a dependency resolver.

Keep model weights, credentials, caches, and quality results outside image layers. Record the base digest, derivative image digest, and dependency delta. A local build or metadata/import check is sufficient here; do not load the model, run CUDA correctness, preflight, PPL, or LongBench.

If a real dependency conflict prevents the required image, retain the completed inventory/tag preparation and report that specific unfinished QP-0 item. Do not change the measured environment or restart the performance workflow.

## 7. Populate the quality contract without inventing new settings

Prepare the manifest-bound configs/quality/quality_contract.yaml using the existing template and accepted protocol. Keep its approval status pending.

Read, rather than guess:
- Exact model/tokenizer revision, chat template, RoPE, weight dtype, and model limit.
- All ten measured configurations and their exact source, kernel, calibration, backend, layout, skip-layer, residual, and sparse/sink policies.
- The intended quality execution mode and its equivalence requirements to the performance implementation.
- Dataset versions, sample-selection rules, seeds, prompts, length budgets, generation settings, quality margins, and statistical decision rules already specified.

Use every exact measured configuration for Q0 and Fast PPL. Do not select initial quality candidates by speed or by the Phase 17 predictor. If existing fingerprints vary with shape or execution mode, preserve that mapping and the protocol's equivalence rules; do not invent one merged fingerprint that hides those differences.

Carry forward the existing cache-sensitive protocols:
- PPL uses compressed-cache prefill, one unscored decode-conditioning token, and teacher-forced one-token scoring whose logits depend on cache reads.
- LongBench preserves the official prompt tokens and uses the prescribed final 16 prompt tokens through decode before answer generation.
- BF16 uses the identical split.
- Evaluation statuses are pass/fail/inconclusive, with the existing paired comparisons and margins.

Do not implement or execute these runners in QP-0.

Record chronology honestly: original protocol commitment, subsequent amendments, and today's manifest binding are different events. Performance results are known at this handoff; do not erase earlier preregistration history or claim all newly resolved choices were fixed before those results.

QP-1 owns final dataset/sample/prompt materialization and formal contract approval. Reuse existing finalized fixtures when present. Any still-unresolved revision, sample-ID set, prompt hash, or execution detail must remain an explicit pending QP-1 item, not a guessed value or an approval-ready claim. Do not start a bulk dataset/tokenization campaign in this task.

## 8. Finish with a compact handoff and a human stop

Use the existing writer to finalize a compact freeze bundle containing the inventory, source/hardware references, locked paths/hash list, release manifest, quality-image identity, and contract handoff state.

Publish only this new compact bundle through the existing R2 tool. Use conditional content-addressed writes and COMPLETE last; verify its new objects once. Reference historical roots rather than uploading or retrieving their complete payloads again. Keep the receipt outside the bundle it identifies. An upload retry does not require redoing the freeze, rebuilding images, or rerunning analysis.

Focused CPU checks only:
- Inventory slot/replacement accounting and explicit source-role separation.
- Exact protected-source identity and tag target.
- Contract-to-manifest identity binding, with pending fields represented honestly.
- Quality remains LOCKED and no evaluation was launched.

Do not run make test, broad make checks, admission, CUDA tests, profiler tests, or historical R2 validation as another entry campaign. If one existing scope check rejects a newly necessary release path, make only the small relevant adjustment.

Use existing credentials host-side without printing them or including .env in any manifest, checksum inventory, image, commit, or upload. Do not delete prefix catalogs or raw artifacts.

Prefer one helper and a few release files, not a new release-management framework. Preserve all existing performance artifacts and reports. Do not refit missing fold vectors or change failed modeling targets.

Return a concise QP-0 REPORT:
- Status and actual starting/final HEAD.
- Frozen performance source commit; local tag name, target, and tag object ID.
- Freeze ID, inventory count, accepted/replaced/infeasible coverage.
- Bound Full Scan/host-wall, modeling, reproduction, and mechanism roots.
- Locked-path/hash-list locations and external kernel identity references.
- Measurement and Quality image identities; any unfinished image item.
- Freeze marker status and what it means in the existing protocol.
- Quality contract status, resolved fields, and remaining QP-1 items.
- New freeze R2 root/URI and verification result.
- Focused checks; confirmation no GPU experiment/refit/quality evaluation ran.
- Quality: LOCKED; human approval: not granted by this task.

Then stop. Do not automatically begin QP-1 execution, Q0, Fast PPL, Full PPL, LongBench, invariance, or a quality/performance join. Present the manifest-bound handoff for human review; unresolved QP-1 items must be completed before approval of formal quality evaluation.

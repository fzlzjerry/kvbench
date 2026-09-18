# QP-0 — Correct the Tag Binding and Finish the Existing Freeze

Continue the existing QP-0 task. This instruction authorizes a metadata-only
correction and completion of the performance freeze. It does NOT approve the
quality contract or authorize Q0, PPL, LongBench, or other quality execution.

Do not introduce another numbered phase, a new approval framework, or a new
validation campaign. No GPU work is needed.

## 1. Use the existing evidence

Read only:
- AGENTS.md and the current QP-0 status entry.
- docs/phase_reports/qp0-performance-freeze.md.
- docs/evidence/qp0/freeze-tag-receipt.json.
- docs/evidence/qp0/r2-publication.json.
- The existing finalized freeze manifest, its checksum-ledger entry, and the
  relevant inventory entry that binds that manifest.
- configs/quality/quality_contract.yaml.
- The QP-0/QP-1 freeze-marker rules and the existing tag/marker helper as needed.

At entry run only `git status --short` and `git rev-parse HEAD`.
Expected current HEAD is dece4fcabd47e78360d6b19af0cee2274d77670b; retain a
legitimate later descendant and unrelated user changes.

Known identities:

    performance_source_commit:
      83536c37433875cda98c36e2848e05692e9407d0

    freeze_id:
      qp0-freeze-20260917t115200000000z-83536c37-bf28e4

    freeze_bundle_root:
      9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f

    original_tag:
      perf-freeze-20260917-83536c37

    original_tag_object:
      c5049cf73c78a96729c4e8a791ae77acf7a1963c

    corrected_tag:
      perf-freeze-20260917-83536c37-r1

The reported defect is a mismatch between the manifest SHA in the original tag
annotation and the SHA of the finalized manifest. The tag's commit target and
bundle-root annotation are reported correct. Confirm only these local facts.
Do not assume why the wrong SHA was produced.

Accept the existing successful R2 retrieval receipt. Fetch only a missing small
manifest/ledger if necessary; do not download historical raw campaigns.

## 2. Resolve the actual finalized manifest identity

Read the exact manifest file named in the finalized bundle's inventory.
Compute its SHA-256 once and compare it with that file's bound ledger entry.
Keep these distinct:

    bundle root digest
    manifest file SHA-256
    Git commit ID
    annotated Git tag object ID

Do not substitute the bundle root for the manifest SHA. Do not hash a newly
serialized JSON object, a draft, or an in-memory manifest instead of the actual
finalized file bytes.

If the file agrees with its finalized ledger, use that SHA for the correction.
If the file and its own ledger disagree, report that specific integrity issue;
that would be different from the annotation-only defect described here.

Do not rebuild, rename, rewrite, or republish the existing freeze bundle.
The inventory, all 2,670 slots, 38 replacement links, and 68 locked paths remain
unchanged.

## 3. Create a corrected tag without moving the original

Preserve the original tag and its object unchanged. Create the distinct local
annotated tag `perf-freeze-20260917-83536c37-r1` targeting the SAME performance
source commit, not the current QP-0 metadata commit.

Generate its annotation from the finalized files. Include only:
- performance source commit;
- existing freeze ID and bundle root;
- manifest relative path and actual finalized-file SHA-256;
- the original tag name/object being superseded for active freeze selection;
- a short explanation: annotation correction only; no data or source change.

Use the existing tag helper or a direct `git tag -a ... -F ...` command.
Do not use force replacement, delete the old tag, push either tag, or backdate
the new tag. If the corrected tag already exists and matches exactly, reuse it.

If the helper caused the mismatch, fix only its annotation-generation ordering:
finalize manifest -> read finalized bytes -> compute SHA -> create tag.
Keep this outside the frozen hot-path files. No generic release manager is needed.

## 4. Record one explicit correction binding

Create one small correction receipt, for example:

    docs/evidence/qp0/freeze-tag-correction.json

Record the old tag/object and its annotated manifest SHA, the corrected tag and
object, the verified final manifest SHA, the unchanged bundle root and source
commit, the correction timestamp, and the metadata-only scope.

The receipt establishes the corrected tag as the active freeze binding. The
original tag remains historical evidence with an erroneous annotation.

If the immutable manifest names the original intended tag, leave that field
unchanged. The correction receipt explicitly supersedes that external tag
binding; it does not supersede the data or alter the original manifest bytes.
Make only the narrow consumer change needed to read this explicit correction.
Do not broadly ignore hash mismatches or choose tags by latest timestamp.

Update the mutable quality handoff to reference the corrected tag and existing
freeze bundle. Preserve the original QP-0 PARTIAL report and receipt; write a
short closure note instead of rewriting historical evidence.

Keep the dependency lock, dataset/sample revisions, prompts, chat-template
hashes, fixture materialization, and formal quality approval as QP-1 pending
items. Do not claim those items are complete or start resolving them here.
Do not rebuild either image for this tag correction.

## 5. Complete the performance freeze, not quality approval

This instruction approves the metadata-only correction and acceptance of the
already completed performance inventory for freezing, conditional on the
focused checks above. Do not request another human disposition for the same
tag-annotation issue.

After the corrected binding passes, complete QP-0 through the existing helper.
Create PERFORMANCE_DATA_FROZEN if it denotes completed performance freezing.
Use the existing marker schema and bind it to the corrected tag and original
bundle, with the correction receipt referenced outside that original bundle.

Maintain separately:

    performance freeze: complete
    QP-1 preparation: ready
    quality contract: requires_human_approval
    Quality execution: LOCKED

The freeze marker must not stand in for quality-contract approval. Do not write
an approved quality status or launch Q0/PPL/LongBench. If an existing consumer
incorrectly equates the marker with quality approval, correct only that
release/quality-authorization check outside the locked hot path.

## 6. Validate and publish only the correction

Run only focused CPU checks:
1. The finalized manifest file SHA matches its existing ledger.
2. The corrected tag targets the original performance source and contains the
   exact finalized manifest SHA and original bundle root.
3. The original tag/object, original freeze bundle, and locked hot paths remain
   unchanged; the current handoff resolves the explicit corrected binding.
4. QP-0 completion does not grant quality approval or launch any evaluation.

No full tests, package audit, historical admission, image rebuild, GPU smoke,
model refit, reproduction run, or bulk R2 verification.

Publish only the small correction receipt/closure note and any new marker
snapshot required by the existing artifact policy. Reference the original
freeze root. Verify this new small bundle once. Keep its R2 publication receipt
outside the bundle it names. Never insert its own root or tag object hash into
an artifact that must determine that same hash.

An upload retry does not require recreating tags or rebuilding the freeze.
Use the existing host-side credential handling without exposing secrets.

Prefer one small metadata/helper commit and any necessary receipt closeout,
not a multi-commit governance project. Do not push tags or branches.

## 7. Return a short QP-0 closure report

Report only:
- Status and final HEAD.
- Original tag preserved.
- Corrected tag, object, and target.
- Actual finalized manifest SHA and unchanged freeze-bundle root.
- Correction receipt and small publication root.
- PERFORMANCE_DATA_FROZEN state.
- QP-1 readiness and outstanding contract items.
- Quality LOCKED; no evaluation executed.

Then stop. Do not start QP-1 or another phase automatically.

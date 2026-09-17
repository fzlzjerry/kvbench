# Phase 18 CPU reproduction plan

Phase 18 packages the completed performance study. It does not refit or tune
models, launch GPU work, rerun measurement/admission/profiling, download
historical campaigns, or execute quality evaluation.

## Frozen authority

- Starting HEAD: `1aac29e408383b3ac265e64b89c7a13abc553814`.
- Phase 17 bundle:
  `phase17-20260916t164055992658z-9e20208a-c944c5`.
- Phase 17 root:
  `05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb`.
- Primary data are the 2,205 accepted Phase 16R host-wall process
  observations and 441 feasible logical points already embedded in the compact
  Phase 17 bundle. The 465 infeasible records remain coverage evidence; 38
  infrastructure replacements retain one accepted observation per slot.
- Scientific and deployment model remain D. Candidate definitions, outer
  scores, targets, split membership, and knee results are immutable inputs.

## Reporting audit

The frozen aggregate labels are the unweighted macro mean across the 11
applicable `(geometry holdout protocol, method family)` cells for selected
model D. Each cell contributes its logical-point median relative error and its
logical-point P95 relative error. BF16 leave-one-config-out is not applicable;
session holdout is reported separately. These are
`macro_mean_of_cell_median_relative_errors` and
`macro_mean_of_cell_p95_relative_errors`, not pooled quantiles.

The BF16 leave-one-batch audit collapses the existing five process predictions
to one median observed and predicted value per `(fold,B,L)` and exports the 20
largest absolute relative errors without clipping or exclusion. It checks
units, full-batch normalization, inverse transform, coordinate convention,
fold joins, BF16 `r_alloc`, and finite positive outputs using existing saved
rows only. Fold-specific fitted parameters were not separately persisted by
Phase 17; the audit binds predictions to the OOF artifact, candidate
specification, code HEAD, and final full-data model without falsely claiming
that the latter generated the OOF rows.

## Reproduction package

The default path reads only the compact Phase 17 bundle, recomputes the audit
tables, regenerates six data-driven SVG figures, and verifies small predictor
examples in a fresh CPU-only directory. It never imports torch or invokes
CUDA, Docker, network, admission, fitting, or experiment commands. An optional
explicit command may rerun frozen Phase 17 fitting/CV, but Phase 18 does not
execute it.

The append-only package contains the compact analysis tables, frozen exported
models, a standalone CPU predictor, audit tables, figures, report,
documentation, reproduction manifest, dependency record, inventory,
checksums, and COMPLETE. Large Phase 14-17 artifacts remain references by
root. Publication is content-addressed and COMPLETE-last, followed by exactly
one clean retrieval; its receipt stays outside the named bundle.

## Quality handoff

Quality remains LOCKED and `PERFORMANCE_DATA_FROZEN` remains absent. Phase 18
creates only a release-manifest draft listing the existing freeze prerequisites.
The next separately authorized task is QP-0: verify no active performance
process, create the complete performance inventory/checksums, record repository
and container/hardware identities, create the performance-freeze tag and
locked hot-path list, populate the quality contract from that manifest, then
stop at the human-approval gate. This phase does none of those actions.


"""Fail-closed GQA evidence taxonomy for Phase 3 remediation."""

from __future__ import annotations

from kvbench.schema import GQAVerdict, RunStatus


def classify_gqa_evidence(
    *,
    materialization_evidence: bool,
    dispatch_verified: bool,
    no_replication_kernel_verified: bool,
    allocation_verified: bool,
    source_verified: bool,
    shape_verified: bool,
) -> GQAVerdict:
    """Classify evidence without treating missing proof as positive evidence.

    Positive materialization evidence takes precedence over every other field.
    Dispatch uncertainty is reported separately. A clean observation remains
    unproven until every preregistered proof layer has passed.
    """

    if materialization_evidence:
        return GQAVerdict.MATERIALIZATION_DETECTED
    if not dispatch_verified:
        return GQAVerdict.DISPATCH_UNVERIFIED
    if all(
        (
            no_replication_kernel_verified,
            allocation_verified,
            source_verified,
            shape_verified,
        )
    ):
        return GQAVerdict.NONMATERIALIZATION_VERIFIED
    return GQAVerdict.NONMATERIALIZATION_UNPROVEN


def gqa_failure_status(verdict: GQAVerdict) -> RunStatus | None:
    """Map a non-passing GQA verdict to its truthful terminal run status."""

    if verdict is GQAVerdict.NONMATERIALIZATION_VERIFIED:
        return None
    return {
        GQAVerdict.MATERIALIZATION_DETECTED: (
            RunStatus.GQA_MATERIALIZATION_DETECTED
        ),
        GQAVerdict.DISPATCH_UNVERIFIED: RunStatus.GQA_DISPATCH_UNVERIFIED,
        GQAVerdict.NONMATERIALIZATION_UNPROVEN: (
            RunStatus.GQA_NONMATERIALIZATION_UNPROVEN
        ),
    }[verdict]

"""Deterministic command reconstruction from saved manifest fields."""

from __future__ import annotations

from kvbench.errors import ProvenanceError
from kvbench.schema import (
    Phase3RunManifest,
    RunManifest,
    parse_run_manifest,
)


def _parsed_manifest(manifest: object) -> object:
    if isinstance(manifest, (RunManifest, Phase3RunManifest)):
        return manifest
    if isinstance(manifest, dict):
        return parse_run_manifest(manifest)
    to_dict = getattr(manifest, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return parse_run_manifest(result)
    raise TypeError("manifest must be a mapping or expose to_dict()")


def reconstruct_command(manifest: object) -> tuple[str, ...]:
    """Rebuild the declared command without consulting ambient state."""

    parsed = _parsed_manifest(manifest)
    if parsed.plan_source.kind.value != "path":
        raise ProvenanceError(
            "only a saved path config source is reconstructable",
        )
    if parsed.plan_source.path is None:
        raise ProvenanceError(
            "manifest.plan_source.path must be a non-empty string",
        )
    return parsed.command.argv

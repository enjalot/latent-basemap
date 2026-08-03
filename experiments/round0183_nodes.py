"""Execute the light CPU-only R0183 baseline synthesis."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_bytes, atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0183_baseline_table import (
    ROUND_ID,
    Round0183Error,
    build_table,
    render_markdown,
)


def _read_bound(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0183Error(f"{label} bytes changed")
    with open(actual["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0183Error(f"{label} is not an object")
    validate_seal(value, label=label)
    return value


def run_job(active: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> None:
    if (
        active.get("manifest", {}).get("round_id") != ROUND_ID
        or job is None
        or job.get("action") != "baseline_table"
    ):
        raise Round0183Error("R0183 handler requires its exact queue/job")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0183 held-out projection table"
    )
    started = time.monotonic()
    aumap = _read_bound(job["aumap_synthesis"], label="R0175 aUMAP synthesis")
    numap_signature = job.get("numap_synthesis")
    numap = (
        _read_bound(numap_signature, label="R0181 NUMAP synthesis")
        if isinstance(numap_signature, Mapping)
        else None
    )
    table = build_table(
        aumap=aumap,
        numap=numap,
        numap_terminal_status=str(job["numap_terminal_status"]),
    )
    markdown_path = os.path.join(output, "heldout-projection-methods.md")
    atomic_write_new_bytes(
        markdown_path, render_markdown(table).encode("utf-8"), immutable=True
    )
    science_identity = table.pop("identity_sha256")
    receipt = seal({
        **table,
        "science_identity_sha256": science_identity,
        "release_sha": active["manifest"]["release_sha"],
        "sources": {
            "aumap_synthesis": dict(job["aumap_synthesis"]),
            "numap_synthesis": (
                dict(numap_signature) if isinstance(numap_signature, Mapping) else None
            ),
            "r0181_result": dict(job["r0181_result"]),
            "r0181_review": dict(job["r0181_review"]),
        },
        "rendered_markdown": expected_input_signature(markdown_path),
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "table.json"), receipt, immutable=True)


__all__ = ["run_job"]

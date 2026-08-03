"""Execute the light CPU-only R0182 universality synthesis."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_bytes, atomic_write_new_json, create_fresh_directory
from basemap.round0142_jina_universality import validate_seal as validate_raw
from basemap.round0146_projection_predictors import validate_seal as validate_predictors
from basemap.round0167_prompted_universality import seal, validate_seal as validate_prompted
from basemap.round0182_universality_packet import (
    CAPABILITY,
    ROUND_ID,
    Round0182Error,
    build_packet,
    render_markdown,
)


def _read_bound(
    expected: Mapping[str, Any], *, label: str, validator: Any
) -> dict[str, Any]:
    signature = expected_input_signature(str(expected.get("canonical_path") or ""))
    if signature != dict(expected):
        raise Round0182Error(f"{label} bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0182Error(f"{label} is not a JSON object")
    validator(value, label=label)
    return value


def run_job(active: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> None:
    if (
        active.get("manifest", {}).get("round_id") != ROUND_ID
        or job is None
        or job.get("action") != "universality_packet"
    ):
        raise Round0182Error("R0182 handler requires its exact queue/job")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0182 universality readout packet"
    )
    started = time.monotonic()
    prompted = _read_bound(
        job["prompted_panel"], label="R0178 prompted panel", validator=validate_prompted
    )
    raw = _read_bound(
        job["raw_panel"], label="R0142 raw panel", validator=validate_raw
    )
    predictors = _read_bound(
        job["raw_predictors"], label="R0146 predictors", validator=validate_predictors
    )
    packet = build_packet(prompted=prompted, raw=raw, predictors=predictors)
    markdown_path = os.path.join(output, "universality-readout.md")
    atomic_write_new_bytes(
        markdown_path, render_markdown(packet).encode("utf-8"), immutable=True
    )
    science_identity = packet.pop("identity_sha256")
    receipt = seal({
        **packet,
        "science_identity_sha256": science_identity,
        "release_sha": active["manifest"]["release_sha"],
        "sources": {
            "prompted_panel": dict(job["prompted_panel"]),
            "raw_panel": dict(job["raw_panel"]),
            "raw_predictors": dict(job["raw_predictors"]),
        },
        "rendered_markdown": expected_input_signature(markdown_path),
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "packet.json"), receipt, immutable=True
    )


__all__ = ["run_job"]

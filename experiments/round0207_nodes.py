"""Execute the CPU-only width-factorial and U12 design synthesis."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_bytes,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0207_width_factorial import (
    FACTORIAL_SCHEMA,
    ROUND_ID,
    RUNGS,
    WIDTHS,
    Round0207Error,
    build_factorial,
    build_u12_design,
    render_factorial_markdown,
    render_u12_markdown,
)


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0207Error(f"{label} bytes changed")
    return actual


def _read_bound(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0207Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


def _read_path_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0207Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


def _diagnostics(
    ladders: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    signatures: dict[str, Any] = {}
    for width in WIDTHS:
        evaluations = ladders[width].get("evaluations") or {}
        if set(evaluations) != set(RUNGS):
            raise Round0207Error(f"{width} evaluation bindings changed")
        output[width] = {}
        signatures[width] = {}
        for rung in RUNGS:
            signature = _signature(
                evaluations[rung], label=f"{width}/{rung} evaluation"
            )
            with open(signature["canonical_path"], encoding="utf-8") as handle:
                evaluation = json.load(handle)
            if (
                not isinstance(evaluation, dict)
                or evaluation.get("rung") != rung
                or int(evaluation.get("seed", -1)) != 42
                or evaluation.get("primary_metrics")
                != ladders[width]["summary"]["cells"][rung]
                or not all((evaluation.get("execution_checks") or {}).values())
            ):
                raise Round0207Error(f"{width}/{rung} evaluation changed")
            output[width][rung] = dict(evaluation.get("diagnostic_metrics") or {})
            signatures[width][rung] = signature
    return output, signatures


def run_factorial(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0207 width factorial"
    )
    started = time.monotonic()
    ladders = {
        width: _read_bound(job["ladders"][width], label=f"accepted {width} ladder")
        for width in WIDTHS
    }
    diagnostics, evaluation_signatures = _diagnostics(ladders)
    r0190 = _read_bound(job["r0190"], label="accepted R0190 synthesis")
    r0191 = _read_bound(job["r0191"], label="accepted R0191 width decision")
    r0201 = _read_bound(job["r0201"], label="accepted R0201 localization")
    factorial = build_factorial(
        ladders=ladders,
        diagnostics=diagnostics,
        r0190=r0190,
        r0191=r0191,
        r0201=r0201,
    )
    markdown_path = os.path.join(output, "width-factorial-and-economics.md")
    atomic_write_new_bytes(
        markdown_path,
        render_factorial_markdown(factorial).encode("utf-8"),
        immutable=True,
    )
    science_identity = factorial.pop("identity_sha256")
    receipt = seal({
        **factorial,
        "science_identity_sha256": science_identity,
        "release_sha": active["manifest"]["release_sha"],
        "sources": {
            "ladders": {
                width: dict(job["ladders"][width]) for width in WIDTHS
            },
            "evaluations": evaluation_signatures,
            "r0190": dict(job["r0190"]),
            "r0191": dict(job["r0191"]),
            "r0201": dict(job["r0201"]),
        },
        "rendered_markdown": expected_input_signature(markdown_path),
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "width-factorial.json"), receipt, immutable=True
    )


def run_u12_memo(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0207 prompted U12 memo"
    )
    started = time.monotonic()
    factorial_path = os.path.join(str(job["factorial_output"]), "width-factorial.json")
    factorial = _read_path_sealed(factorial_path, label="R0207 factorial")
    if factorial.get("schema") != FACTORIAL_SCHEMA:
        raise Round0207Error("R0207 factorial schema changed before memo")
    u12 = _read_bound(job["u12_manifest"], label="accepted R0168 U12 manifest")
    graph = _read_bound(job["graph_precedent"], label="accepted R0171 graph")
    audit = _read_bound(job["ood_audit"], label="accepted R0173 OOD audit")
    memo = build_u12_design(
        factorial=factorial,
        u12_manifest=u12,
        graph_precedent=graph,
        ood_audit=audit,
    )
    markdown_path = os.path.join(output, "prompted-diverse-u12-design.md")
    atomic_write_new_bytes(
        markdown_path, render_u12_markdown(memo).encode("utf-8"), immutable=True
    )
    science_identity = memo.pop("identity_sha256")
    receipt = seal({
        **memo,
        "science_identity_sha256": science_identity,
        "release_sha": active["manifest"]["release_sha"],
        "sources": {
            "factorial": expected_input_signature(factorial_path),
            "u12_manifest": dict(job["u12_manifest"]),
            "graph_precedent": dict(job["graph_precedent"]),
            "ood_audit": dict(job["ood_audit"]),
        },
        "rendered_markdown": expected_input_signature(markdown_path),
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "prompted-diverse-u12-design.json"),
        receipt,
        immutable=True,
    )


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0207Error("R0207 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "factorial":
        return run_factorial(active, job)
    if action == "u12_memo":
        return run_u12_memo(active, job)
    raise Round0207Error(f"unknown R0207 action {action!r}")


__all__ = ["run_job"]

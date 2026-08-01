#!/usr/bin/env python3
"""Prepare, but never launch, the R0146 CPU projection-predictor queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0142_jina_universality import (
    CAPABILITY as R0142_CAPABILITY,
    MAP_ORDER,
    PROBE_ORDER,
    validate_seal as validate_r0142_seal,
)
from basemap.round0146_projection_predictors import (
    BLAS_THREADS,
    CAPABILITY,
    GEOMETRY_SAMPLE_ROWS,
    PREDICTOR_ORDER,
    ROUND_ID,
    TRAINING_SUPPORT_ROWS,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0146"
# CPU-only analysis must not share the mutable checkout pointer used by the
# active GPU queue.  The environment below is still the sealed, read-only run
# environment, but this independent checkout can stay pinned to R0146 while a
# different release remains pinned in latent-basemap-run.
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0146-*.md")
RESULT_FILE_GLOB = os.path.join(LAB_ROOT, "result-0142-*.md")
REVIEW_FILE_GLOB = os.path.join(LAB_ROOT, "review-0142-*.md")
R0142_ROOT = "/data/latent-basemap/runs/round-0142/queue/artifacts"
RETENTION_TABLE = os.path.join(
    R0142_ROOT, R0142_CAPABILITY, "retention-table.json"
)
PANELS = {
    map_key: os.path.join(R0142_ROOT, map_key, "universality-panel.json")
    for map_key in MAP_ORDER
}
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0103/queue/artifacts/"
    "jina-diverse-25m-full768-int8-substrate/"
    "jina-diverse-25m-full768-int8-substrate-v1.json"
)
FULL_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/graph-manifest.json"
)
HALF_SUBSET_MANIFEST = (
    "/data/latent-basemap/runs/round-0132/queue/artifacts/"
    "half-subset/subset-manifest.json"
)
EXPECTED_WALL_SECONDS = 300.0
P90_WALL_SECONDS = 1_800.0


def _status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(8_192)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _unique_file(pattern: str, *, status: str, label: str) -> str:
    paths = [path for path in sorted(glob.glob(pattern)) if _status(path) == status]
    if len(paths) != 1:
        raise RuntimeError(f"R0146 requires one {label}; found {len(paths)}")
    return paths[0]


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _validate_identity(value: dict[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise RuntimeError(f"{label} identity seal changed")


def _r0142_evidence() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    result_path = _unique_file(
        RESULT_FILE_GLOB, status="complete", label="complete R0142 result"
    )
    review_path = _unique_file(
        REVIEW_FILE_GLOB, status="accepted", label="accepted R0142 review"
    )
    with open(review_path, encoding="utf-8") as handle:
        review_text = handle.read()
    if f"capability:{R0142_CAPABILITY}" not in review_text:
        raise RuntimeError("accepted R0142 review does not release B1 capability")

    table_signature = expected_input_signature(RETENTION_TABLE)
    table = _read_json(RETENTION_TABLE)
    validate_r0142_seal(table, label="R0142 retention table")
    if (
        table.get("schema") != R0142_CAPABILITY
        or table.get("round_id") != "0142"
        or table.get("probe_order") != list(PROBE_ORDER)
        or table.get("capability") != R0142_CAPABILITY
    ):
        raise RuntimeError("R0142 retention table semantics changed")
    panels: dict[str, dict[str, Any]] = {}
    for map_key in MAP_ORDER:
        signature = expected_input_signature(PANELS[map_key])
        if table.get("maps", {}).get(map_key) != signature:
            raise RuntimeError(f"R0142 table does not bind {map_key} panel")
        panel = _read_json(PANELS[map_key])
        validate_r0142_seal(panel, label=f"R0142 {map_key} panel")
        if (
            panel.get("schema") != "round0142-jina-universality-map-panel-v1"
            or panel.get("map_key") != map_key
            or panel.get("probe_order") != list(PROBE_ORDER)
        ):
            raise RuntimeError(f"R0142 {map_key} panel semantics changed")
        panels[map_key] = panel
    evidence = [
        expected_input_signature(result_path),
        expected_input_signature(review_path),
        table_signature,
        *(expected_input_signature(PANELS[key]) for key in MAP_ORDER),
    ]
    return evidence, table, panels


def _probe_inputs(panels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for map_key in MAP_ORDER:
        for probe in PROBE_ORDER:
            scored = panels[map_key]["probes"][probe]["probe"]
            signatures.append(dict(scored["coordinates"]))
            inputs = scored.get("inputs") or {}
            if "embeddings" in inputs:
                signatures.append(dict(inputs["embeddings"]))
            else:
                signatures.extend([
                    dict(inputs["corpus_embeddings"]),
                    dict(inputs["query_embeddings"]),
                ])
    for signature in signatures:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError("R0142 probe input bytes changed")
    return _dedupe(signatures)


def _training_support() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    substrate_signature = expected_input_signature(SUBSTRATE_MANIFEST)
    substrate = _read_json(SUBSTRATE_MANIFEST)
    _validate_identity(substrate, label="R0103 substrate")
    if (
        substrate.get("schema")
        != "jina-diverse-25m-full768-int8-substrate-v1"
        or int(substrate.get("row_count", -1)) != 25_000_000
        or int(substrate.get("dimension", -1)) != 768
    ):
        raise RuntimeError("R0103 substrate semantics changed")
    int8 = dict(substrate["outputs"]["int8"])
    scales = dict(substrate["outputs"]["scales"])
    if (
        expected_input_signature(int8["canonical_path"]) != int8
        or expected_input_signature(scales["canonical_path"]) != scales
    ):
        raise RuntimeError("R0103 substrate payload bytes changed")

    full_manifest_signature = expected_input_signature(FULL_GRAPH_MANIFEST)
    full = _read_json(FULL_GRAPH_MANIFEST)
    _validate_identity(full, label="R0106 graph")
    full_mapping = dict(full["compact_mapping"])
    if (
        int(full.get("retained_rows", -1)) != 24_948_663
        or expected_input_signature(full_mapping["canonical_path"]) != full_mapping
    ):
        raise RuntimeError("R0106 full training mapping changed")

    half_manifest_signature = expected_input_signature(HALF_SUBSET_MANIFEST)
    half = _read_json(HALF_SUBSET_MANIFEST)
    _validate_identity(half, label="R0132 half subset")
    half_mapping = dict(half["mapping"])
    if (
        int(half.get("selected_rows", -1)) != 12_474_331
        or expected_input_signature(half_mapping["canonical_path"]) != half_mapping
    ):
        raise RuntimeError("R0132 half training mapping changed")

    common = {"int8": int8, "scales": scales}
    support = {
        MAP_ORDER[0]: {**common, "mapping": full_mapping, "mapping_rows": 24_948_663},
        MAP_ORDER[1]: {**common, "mapping": half_mapping, "mapping_rows": 12_474_331},
    }
    evidence = _dedupe([
        substrate_signature,
        int8,
        scales,
        full_manifest_signature,
        full_mapping,
        half_manifest_signature,
        half_mapping,
    ])
    return support, evidence


def prepare_round0146(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0146 release SHA must be one full commit")
    round_file = _unique_file(
        ROUND_FILE_GLOB, status="issued", label="issued R0146 round"
    )
    round_signature = expected_input_signature(round_file)
    r0142_evidence, table, panels = _r0142_evidence()
    probe_inputs = _probe_inputs(panels)
    support, support_evidence = _training_support()
    queue_root = create_fresh_directory(queue_root, label="R0146 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    expected_inputs = _dedupe([
        round_signature,
        *r0142_evidence,
        *probe_inputs,
        *support_evidence,
    ])
    job = {
        "id": "analyze_projection_loss_predictors",
        "action": "projection_loss_predictors",
        "handler_module": "experiments.round0146_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts, "analyze-projection-loss-predictors.done.json"
        ),
        "expected_inputs": expected_inputs,
        "p90_wall_s": P90_WALL_SECONDS,
        "cpu_threads": BLAS_THREADS,
        "retention_table": expected_input_signature(RETENTION_TABLE),
        "panels": {
            map_key: expected_input_signature(PANELS[map_key])
            for map_key in MAP_ORDER
        },
        "training_support": support,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
        },
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.05,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0146-projection-loss-predictor-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0142"],
        "ordering_dependencies": ["0142"],
        "capability_dependencies": [R0142_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "which preregistered probe properties co-vary with R0142 retention?",
            "maps": list(MAP_ORDER),
            "probes": list(PROBE_ORDER),
            "geometry_sample_rows": GEOMETRY_SAMPLE_ROWS,
            "training_support_rows_per_map": TRAINING_SUPPORT_ROWS,
            "predictors": list(PREDICTOR_ORDER),
            "primary_outcome": "ffr_retention",
            "secondary_outcome": "recall10_retention",
            "analysis": "per-map Spearman rank correlation plus explicitly descriptive pooled rows",
            "cpu_threads": BLAS_THREADS,
            "expected_wall_seconds": EXPECTED_WALL_SECONDS,
            "p90_wall_seconds": P90_WALL_SECONDS,
            "diagnostic_only": True,
            "no_causal_predictor_claim": True,
            "no_universal_map_claim": True,
            "no_quality_gate_change": True,
            "no_training": True,
            "r0142_table_identity_sha256": table["identity_sha256"],
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0146(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

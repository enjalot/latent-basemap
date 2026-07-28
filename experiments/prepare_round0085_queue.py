#!/usr/bin/env python3
"""Prepare, but never launch, the duplicate-controlled density-v2 queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ID = "0085"
ROUND_ROOT = "/data/latent-basemap/runs/round-0085"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0085-2026-07-27.md")
REVIEWS = {
    "0064": os.path.join(LAB_ROOT, "review-0064-2026-07-26.md"),
    "0069": os.path.join(LAB_ROOT, "review-0069-2026-07-27.md"),
    "0074": os.path.join(LAB_ROOT, "review-0074-2026-07-27.md"),
    "0076": os.path.join(LAB_ROOT, "review-0076-2026-07-27.md"),
}
REQUIRED_CAPABILITIES = {
    "0064": "capability:minilm-balanced-30m-60m-scale-geometry-v1",
    "0069": "capability:minilm-balanced-30m-45m-60m-scale-geometry-v1",
    "0074": "capability:minilm-30m-density-anchor-leverage-v1",
    "0076": "capability:minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
}
EXPECTED_REVIEW_SHA256 = {
    "0064": "f6b77feedfbc6dbef2e098edb88e2e8d9414ebe12ed2a67d18ce05676adcf0a7",
    "0069": "84890e5cb699697b07e658bd6f70b257f1aa0c901b34b7b668426f0974297cb9",
    "0074": "cfc231a4a3e8a661d591e4f9be81dcf41ab1d2c46c26625b4a26aa577989cf9d",
    "0076": "d987d15d33bd49119efa279d1e375c73d4edab19c020b21304fe9103415958b5",
}
R64 = "/data/latent-basemap/runs/round-0064/queue/artifacts"
R69 = "/data/latent-basemap/runs/round-0069/queue/artifacts"
R74 = (
    "/data/latent-basemap/runs/round-0074/queue-attempt-2/artifacts/"
    "duplicate-anchor-leverage"
)
R76 = (
    "/data/latent-basemap/runs/round-0076/queue-attempt-2/artifacts"
)
UNIVERSE_PATHS = {
    "balanced-30m": {
        "row_count": 30_000_000,
        "eligibility_path": (
            "/data/latent-basemap/runs/round-0053/queue/artifacts/"
            "balanced-30m-int8-substrate/"
            "minilm-balanced-30m-int8-row-eligibility-v1.npz"
        ),
        "reference_root": os.path.join(R64, "high-d-reference-30m"),
        "eligibility_sha256": "591b45adb006dd6ac923788279bc8c03c3ba7b0eaea48412bb0d5458d3c0c836",
        "reference_sha256": "71c123df83247dbbd6d8d6d5a2ea79ec12b90d012d638faa0f1281d706834ae3",
        "reference_receipt_sha256": "bc0917863e8023e04921afd8318fd33cd303769ec39d7851ba51ef113599f408",
        "anchor_rows_sha256": "daca7475eab6392ad15c5941cdcf17967a45cdad56cb07f8d9c76f598ed1af84",
    },
    "balanced-45m": {
        "row_count": 45_000_000,
        "eligibility_path": (
            "/data/latent-basemap/runs/round-0065/queue/artifacts/"
            "balanced-45m-int8-substrate/"
            "minilm-balanced-45m-row-eligibility-v1.npz"
        ),
        "reference_root": os.path.join(R69, "high-d-reference-45m"),
        "eligibility_sha256": "f737a814d17088d95324b933f1e9c6f05ef9d7c63e77754ac0eac73a64b654a1",
        "reference_sha256": "80b7fb546c81203c406881bd40e517a91d328d9056a6522111bce1393dd5ae1b",
        "reference_receipt_sha256": "f417e1fbaa2b07964950ea3c65c978f3683946307a19b2310a89004def17b645",
        "anchor_rows_sha256": "b594f84864c305b740e26165d2b6d561d154a8af0ea8fbc432ef4e6c92934066",
    },
    "balanced-60m": {
        "row_count": 60_000_000,
        "eligibility_path": (
            "/data/latent-basemap/runs/round-0049/queue/artifacts/"
            "balanced-60m-substrate/"
            "minilm-balanced-60m-row-eligibility-v1.npz"
        ),
        "reference_root": os.path.join(R64, "high-d-reference-60m"),
        "eligibility_sha256": "52395485800cc834d889d533adfa1fcce0d9cbb404f16a680aba6a51c8913a84",
        "reference_sha256": "0a98ab99660346741e45d63c8e7837833ea7726ae3bd60762b872aacb4d88703",
        "reference_receipt_sha256": "56d698128aa08835b03c96d8e10724b62d19b4d79f7f57369fe2cc15b7e1a422",
        "anchor_rows_sha256": "cf4206ee5fc2da96d62206658c81679707e535a00f3952a0a88bdbf0fc073233",
    },
    "balanced-90m": {
        "row_count": 90_000_000,
        "eligibility_path": (
            "/data/latent-basemap/runs/round-0071/queue/artifacts/"
            "balanced-90m-int8-substrate/"
            "minilm-balanced-90m-row-eligibility-v1.npz"
        ),
        "reference_root": os.path.join(R76, "high-d-reference-90m"),
        "eligibility_sha256": "8be881b43120b501e1a534bf0c46ad01aa3c24e221be9d5a798ceb58f831abd1",
        "reference_sha256": "9cf81ea4e9e3f44367e3781f98d70eae0a3e25974c39079d4d546c9126040c18",
        "reference_receipt_sha256": "cc3c749501506e6f06e58221ae3fd0bae39ed0c6b219bb6af03720ec7f5b6642",
        "anchor_rows_sha256": "e23e533beb24827c2fe67690d6fba1ca56a0e869f9b7b27949b0bdbb50af9450",
    },
}
CELL_PATHS = (
    (
        "r0061_30m_on_30m",
        "balanced-30m",
        "r0061-30m-on-30m",
        os.path.join(R64, "coordinates-r0061-30m"),
        "ca3ec46bceeb1c77e2ee13ccb9a25f8a0521807dd32cbe6066500f66ac714dd3",
        "76d640356486ca1d4560664b76e314cbb3d8047aaf84444a89d2e104b3dc5156",
    ),
    (
        "r0068_45m_on_30m",
        "balanced-30m",
        "r0068-45m-on-30m",
        os.path.join(R69, "coordinates-r0068-on-30m"),
        "e969ccd31156ba8a34ef16275d1ab3bca20ae0ee0ce1fbd0624e38efce466955",
        "c4408ee01571d5c8e2f0938495a324df286a26b86a51ba83b0e58599ccc36779",
    ),
    (
        "r0063_60m_on_30m",
        "balanced-30m",
        "r0063-60m-on-30m",
        os.path.join(R64, "coordinates-r0063-on-30m"),
        "06e8714aaca843fc456f72753f6fe956dfb753d19a1d9753a92d11e988206bcf",
        "742eb3c0161c35c927eacf38b2a3d3dd34defbcced5a860a49d8096fac14dff1",
    ),
    (
        "r0075_90m_on_30m",
        "balanced-30m",
        "r0075-90m-on-30m",
        os.path.join(R76, "coordinates-r0075-on-30m"),
        "31b279270f1c3ce5a40be80c41f886f0d24665eeda28920feaf638afa2305218",
        "197af726596c8959c1f0d107ae7430df721452e340d64155ffdc618e3cb855ee",
    ),
    (
        "r0068_45m_on_45m",
        "balanced-45m",
        "r0068-45m-on-45m",
        os.path.join(R69, "coordinates-r0068-45m"),
        "8ce7fb2c001aa43b3e0ec358964165e422bf1155c8f99994f1f12244eb4acdff",
        "c4408ee01571d5c8e2f0938495a324df286a26b86a51ba83b0e58599ccc36779",
    ),
    (
        "r0063_60m_on_60m",
        "balanced-60m",
        "r0063-60m-on-60m",
        os.path.join(R64, "coordinates-r0063-60m"),
        "0254c1ca18c44085afcac7dbcca0513657a0c2bbf2e7d5361531411a01921fab",
        "742eb3c0161c35c927eacf38b2a3d3dd34defbcced5a860a49d8096fac14dff1",
    ),
    (
        "r0075_90m_on_90m",
        "balanced-90m",
        "r0075-90m-on-90m",
        os.path.join(R76, "coordinates-r0075-90m"),
        "abab96a89e45226335d1b87789fec2fe4ec38152fc7f096576fd77cb47bed009",
        "197af726596c8959c1f0d107ae7430df721452e340d64155ffdc618e3cb855ee",
    ),
)
R0074_EXPECTED = {
    "receipt_sha256": "4f2f64a38754791d640b6d101f2cc8bdbe8d17f2fcf5723290ab48dd663bd97a",
    "radii_sha256": "7452f7c6bd847dbe09f9fba763aba3a2abc581ba5166ab75e03731786df5d992",
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> None:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0085 remains draft; refuse queue materialization")


def _require_review(round_id: str) -> dict[str, Any]:
    path = REVIEWS[round_id]
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError(f"R{round_id} does not release reviewed evidence")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    capability = REQUIRED_CAPABILITIES[round_id]
    if capability not in text:
        raise RuntimeError(f"R{round_id} does not release {capability}")
    signature = expected_input_signature(path)
    if signature["sha256"] != EXPECTED_REVIEW_SHA256[round_id]:
        raise RuntimeError(f"R{round_id} review bytes changed")
    return signature


def _coordinate_inputs(root: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(root, "chunk-*", "coordinates.npy")))
    if not paths:
        raise RuntimeError(f"coordinate stream has no chunks: {root}")
    return [os.path.join(root, "actual-transform.json"), *paths]


def _model_signature(
    transform_root: str,
) -> tuple[str, dict[str, Any]]:
    receipt_path = os.path.join(transform_root, "actual-transform.json")
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    model_sha = (receipt.get("model") or {}).get("sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(model_sha)):
        raise RuntimeError(f"coordinate model is not content-bound: {receipt_path}")
    return str(model_sha), expected_input_signature(receipt_path)


def prepare_round0085(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0085 release SHA must be one full commit")
    reviews = {round_id: _require_review(round_id) for round_id in REVIEWS}
    universes: list[dict[str, Any]] = []
    validated_inputs = list(reviews.values())
    input_paths = [ROUND_FILE]
    for label, raw in UNIVERSE_PATHS.items():
        reference_root = str(raw["reference_root"])
        eligibility = expected_input_signature(str(raw["eligibility_path"]))
        reference = expected_input_signature(
            os.path.join(reference_root, "reference.npz")
        )
        receipt = expected_input_signature(
            os.path.join(reference_root, "reference-receipt.json")
        )
        anchors = expected_input_signature(
            os.path.join(reference_root, "anchor-substrate-rows.npy")
        )
        observed = {
            "eligibility_sha256": eligibility["sha256"],
            "reference_sha256": reference["sha256"],
            "reference_receipt_sha256": receipt["sha256"],
            "anchor_rows_sha256": anchors["sha256"],
        }
        expected = {
            key: str(raw[key])
            for key in observed
        }
        if observed != expected:
            raise RuntimeError(f"{label} issued reference bytes changed")
        universes.append({
            "label": label,
            "row_count": int(raw["row_count"]),
            "eligibility_path": eligibility["canonical_path"],
            "eligibility_sha256": eligibility["sha256"],
            "reference_path": reference["canonical_path"],
            "reference_sha256": reference["sha256"],
            "reference_receipt_path": receipt["canonical_path"],
            "reference_receipt_sha256": receipt["sha256"],
            "anchor_rows_path": anchors["canonical_path"],
            "anchor_rows_sha256": anchors["sha256"],
        })
        validated_inputs.extend([
            eligibility,
            reference,
            receipt,
            anchors,
        ])

    cells: list[dict[str, Any]] = []
    for (
        key,
        universe,
        map_key,
        coordinates,
        expected_receipt_sha,
        expected_model_sha,
    ) in CELL_PATHS:
        model_sha, receipt_signature = _model_signature(coordinates)
        receipt_sha = receipt_signature["sha256"]
        if (
            receipt_sha != expected_receipt_sha
            or model_sha != expected_model_sha
        ):
            raise RuntimeError(f"{key} issued coordinate/model bytes changed")
        cells.append({
            "key": key,
            "universe": universe,
            "map_key": map_key,
            "model_sha256": model_sha,
            "coordinates_path": coordinates,
            "coordinate_receipt_sha256": receipt_sha,
        })
        coordinate_inputs = _coordinate_inputs(coordinates)
        if coordinate_inputs[0] != receipt_signature["canonical_path"]:
            raise RuntimeError(
                f"coordinate receipt path changed: {coordinates}"
            )
        validated_inputs.append(receipt_signature)
        input_paths.extend(coordinate_inputs[1:])

    r0074_receipt = expected_input_signature(
        os.path.join(R74, "duplicate-anchor-leverage.json")
    )
    r0074_radii = expected_input_signature(
        os.path.join(R74, "anchor-leverage-radii.npz")
    )
    if (
        r0074_receipt["sha256"] != R0074_EXPECTED["receipt_sha256"]
        or r0074_radii["sha256"] != R0074_EXPECTED["radii_sha256"]
    ):
        raise RuntimeError("issued R0074 replay bytes changed")
    validated_inputs.extend([
        r0074_receipt,
        r0074_radii,
    ])
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0085 density-v2 queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "density-v2-calibration")
    job = {
        "id": "density_v2_calibration",
        "action": "density_v2",
        "handler_module": "experiments.round0085_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "density_v2_calibration.done.json",
        ),
        "expected_inputs": _dedupe([
            *validated_inputs,
            *_file_inputs(input_paths),
        ]),
        "p90_wall_s": 1_200.0,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
        "r0074_receipt_path": r0074_receipt["canonical_path"],
        "r0074_receipt_sha256": r0074_receipt["sha256"],
        "r0074_radii_path": r0074_radii["canonical_path"],
        "r0074_radii_sha256": r0074_radii["sha256"],
        "universes": universes,
        "cells": cells,
    }
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.60,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0085-density-v2-calibration-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = list(REVIEWS)
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-60m-scale-geometry-v1",
        "minilm-balanced-30m-45m-60m-scale-geometry-v1",
        "minilm-30m-density-anchor-leverage-v1",
        "minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-density-v2-calibration-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "r0074_receipt": r0074_receipt,
        "r0074_radii": r0074_radii,
    }
    manifest["scientific_contract"] = {
        "primary_metric": (
            "Pearson correlation of log exact high-/low-D mean-k15 radii "
            "on reviewed representative-reference anchors with exact-family "
            "size <16"
        ),
        "matched_cells": [
            "r0061_30m_on_30m",
            "r0068_45m_on_30m",
            "r0063_60m_on_30m",
            "r0075_90m_on_30m",
        ],
        "bootstrap": {"draws": 1_000, "seed": 85_001},
        "permuted_radius_null": {"draws": 1_000, "seed": 85_002},
        "floor_rule": (
            "min matched density_v2 - 3 * max matched bootstrap SD"
        ),
        "floor_guards": [
            "proposed floor must be positive",
            "proposed floor must exceed every matched cell's absolute "
            "permuted-null 99.9th percentile",
        ],
        "training_performed": False,
    }
    manifest["jobs"] = [job]
    manifest["p90_gpu_seconds"] = {
        "density_v2_calibration": 1_200.0,
        "total": 1_200.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(prepare_round0085(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

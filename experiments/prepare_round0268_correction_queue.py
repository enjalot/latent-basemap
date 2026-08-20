#!/usr/bin/env python3
"""Prepare (never launch) R0268 Queue-A — the seed-42 TRANSFORM-CORRECTION queue (R10).

Background: attempt-4 (R9) trained seed42's full clean dose and SEALED its train-receipt +
model BEFORE the transform (the R9 phase-split working as designed), but the R9 transform poll
called `os.path.exists()` on the `_start_node()` DICT → TypeError, so the 100M transform never
produced its transform-receipt / seed-1 tripwire. R10 fixes the poll and adds a
`transform_correct_minilm_fneg_100m_x2_hostint8` node that re-runs ONLY the fixed, unguarded
transform from seed42's sealed read-only train-receipt + model, sealing a transform-receipt
(+ failed-marker provenance) + its own done-marker. No re-train.

This is Queue-A of a TWO-QUEUE hard-stop: it holds ONLY the seed-42 correction node. After it
runs, a driver reads the seed-1 tripwire and takes it to the owner-delegate; Queue-B (seed43
train → seed44 train → panel → gate) is built + launched only after that checkpoint reply, so
seed43 structurally cannot auto-start.

REFUSES until the round is issued and every bound input exists on disk — in particular seed42's
SEALED train-receipt.json + model.pt (they exist only after attempt-4's ~16:10Z seal) and the
original R9 failed marker (the record of the defect this node corrects, bound read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.output_safety import atomic_write_new_json
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    assert_derived_entries_install,
    dispatch_census,
    entry_tuples,
    gate_census,
    scope_residual,
)
from basemap import round0268_int8_treatment as T
from basemap.round0268_int8_treatment import ROUND_ID, ROWS, capability_for_seed
from experiments.prepare_round0020_0022_queues import _base_manifest, _dedupe
from experiments.prepare_round0268_queue import (
    R0238_SUBSTRATE_MANIFEST,
    RELEASE_ROOT,
    ROUND_FILE,
    _issued_round,
    _signature,
    file_sha256_manifest,
)
from experiments.round0268_nodes import TRANSFORM_CORRECTION_ACTION

#: The seed42 artifacts from the ORIGINAL R9 flagship queue (attempt-4): the sealed train-receipt
#: + model are the read-only correction inputs; the failed marker is the provenance record.
R9_QUEUE_ARTIFACTS = "/data/latent-basemap/runs/round-0268/queue/artifacts"
SEED = 42
CORRECTION_CAPABILITY = (
    "minilm-mixed-100000k-fneg-x2-md000-hostint8-seed42-transform-correction-r0268-v1"
)
QUEUE_ROOT = "/data/latent-basemap/runs/round-0268/correction-queue-A"
CORRECTION_P90_WALL_S = 3600.0  # ~35-40 min transform + verify; generous.


def prepare_correction_queue(*, release_sha: str, queue_root: str,
                             r9_artifacts: str = R9_QUEUE_ARTIFACTS) -> str:
    round_doc, required_reviews = _issued_round(release_sha)  # refuses unless issued + ancestor

    capability = capability_for_seed(SEED)
    train_receipt_path = os.path.join(r9_artifacts, capability, "train-receipt.json")
    model_path = os.path.join(r9_artifacts, capability, "model.pt")
    failed_marker_path = os.path.join(
        r9_artifacts, f"train_minilm_fneg_100m_x2_hostint8_seed{SEED}.failed.json"
    )
    # every bound input must exist on disk now (post-seal). _signature raises otherwise.
    train_receipt_sig = _signature(train_receipt_path, "R0268 seed42 sealed train-receipt")
    model_sig = _signature(model_path, "R0268 seed42 sealed model")
    failed_marker_sig = _signature(failed_marker_path, "R0268 seed42 original R9 failed marker")
    substrate_manifest_signature = _signature(
        R0238_SUBSTRATE_MANIFEST, "R0238 100M substrate manifest"
    )

    census = dispatch_census()
    guard = assert_derived_entries_install(SCOPE_MODULES, census)
    gates = gate_census(entry_tuples(guard["derived"]))
    residual = scope_residual(census, SCOPE_MODULES)

    os.makedirs(queue_root, exist_ok=True)
    artifacts = os.path.join(queue_root, "artifacts")
    os.makedirs(artifacts, exist_ok=True)

    node_id = f"{TRANSFORM_CORRECTION_ACTION}_seed{SEED}"
    output = os.path.join(artifacts, CORRECTION_CAPABILITY)
    jobs = [{
        "id": node_id,
        "action": TRANSFORM_CORRECTION_ACTION,
        "handler_module": "experiments.round0268_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": _dedupe([
            substrate_manifest_signature, train_receipt_sig, model_sig, failed_marker_sig,
        ]),
        "p90_wall_s": CORRECTION_P90_WALL_S,
        "training_seed": SEED,
        "capability": capability,
        "substrate_manifest_signature": substrate_manifest_signature,
        # the read-only correction inputs (full signatures; the node re-verifies on load).
        "train_receipt": train_receipt_sig,
        "model": model_sig,
        "original_failed_marker": failed_marker_sig,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": False},
    }]
    p90 = {node_id: CORRECTION_P90_WALL_S, "total": CORRECTION_P90_WALL_S}

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=2.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0268-fneg-100m-x2-transform-correction-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-correction",
        "required_reviews": list(required_reviews),
        "jobs": jobs,
        "p90_wall_s": p90,
        "scope_modules": list(SCOPE_MODULES),
        "stop_hook_install_guard": {
            "derived_entries": guard["derived"],
            "every_derived_entry_installs": guard["audit"]["every_entry_installs_effectively"],
            "gate_census": gates,
            "scope_residual": residual,
        },
        "registered": {
            "what_this_queue_is": (
                "Queue-A of the R10 two-queue hard-stop: re-run ONLY the fixed, unguarded 100M "
                "transform for seed42 from its SEALED read-only train-receipt + model (attempt-4, "
                "R9), sealing the transform-receipt + coordinates + seed-1 tripwire the R9 poll "
                "defect prevented. NO re-train. Seeds 43/44 + panel + gate are Queue-B, built + "
                "launched only after the seed-1 tripwire checkpoint."
            ),
            "corrects_defect": (
                "R9 round0268_nodes._transform_poll called os.path.exists() on the _start_node() "
                "dict -> TypeError in the transform; R10 checks abort_flag['abort_flag_path']."
            ),
            "reuses_sealed_read_only": {
                "train_receipt_sha256": train_receipt_sig["sha256"],
                "model_sha256": model_sig["sha256"],
                "original_r9_failed_marker_sha256": failed_marker_sig["sha256"],
            },
            "consumes_sealed_100m_inputs": {"substrate": T.R0238_SUBSTRATE_CAPABILITY},
            "gate_registerable_here": False,
            "acceptance_rule": (
                "produce seed42's corrected transform-receipt + seed-1 tripwire; NO numerical "
                "outcome makes it a failure — the tripwire is a MEASUREMENT taken to the owner."
            ),
        },
    })
    manifest_path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(manifest_path, queue, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R0268 seed-42 correction queue (Queue-A)")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    parser.add_argument("--r9-artifacts", default=R9_QUEUE_ARTIFACTS)
    args = parser.parse_args(argv)
    path = prepare_correction_queue(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
        r9_artifacts=args.r9_artifacts,
    )
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

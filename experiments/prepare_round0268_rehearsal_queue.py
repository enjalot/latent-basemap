#!/usr/bin/env python3
"""Prepare (never launch) the R11 REHEARSAL queue — a single-node, NON-EVIDENCE plumbing rehearsal.

Reuses the REAL R0268 seed-42 train node (so every bound input — substrate, R0243 graph members,
R0262 int8 full file, treatment closure — is signature-identical to attempt-5) and rewrites ONLY:
  * handler_module/handler_callable → the rehearsal handler (stubs model.fit to load attempt-4's
    byte-identical preserved model; everything downstream runs for real),
  * outputs + done_marker → a throwaway `.../round-0268/rehearsal/...` dir (NON-EVIDENCE),
  * action → the rehearsal action,
  * + `preserved_model_path` bound read-only.

The runner then executes the whole post-training pipe — receipt assembly, seal, the fixed transform
poll, the 100M transform, the transform-receipt, the done-marker, and the runner's own validation —
exactly as attempt-5 will. Cost: ~35-40 min transform on an idle GPU. On PASS, attempt-5 launches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json
from experiments.prepare_round0268_queue import (
    SEEDS,
    capability_for_seed,
    file_sha256_manifest,
    prepare_round0268,
)

REHEARSAL_ROOT = "/data/latent-basemap/runs/round-0268/rehearsal"
PRESERVED_MODEL = "/data/latent-basemap/runs/round-0268/salvage/attempt4-preserved/model.pt"
REHEARSAL_ACTION = "rehearse_transform_pipe_minilm_fneg_100m_x2_hostint8"
REHEARSAL_SEED = 42


def _rehearsal_manifest_from_real(
    real: dict[str, Any], *, queue_root: str, preserved_model_path: str,
    preserved_model_sig: dict[str, Any],
) -> dict[str, Any]:
    """PURE transform: take a REAL R0268 queue manifest and return the single-node rehearsal
    manifest. Rewrites ONLY the seed-42 train node's handler/outputs/action + binds the preserved
    model; every bound INPUT signature is carried through verbatim (so the rehearsal exercises the
    exact inputs attempt-5 will). Unit-testable without building the real queue."""
    capability = capability_for_seed(REHEARSAL_SEED)
    train_id = f"train_minilm_fneg_100m_x2_hostint8_seed{REHEARSAL_SEED}"
    seed42 = next(j for j in real["jobs"] if j["id"] == train_id)

    artifacts = os.path.join(queue_root, "artifacts")
    rehearsal_output = os.path.join(artifacts, f"rehearsal-{capability}")
    node = dict(seed42)
    node["id"] = f"{REHEARSAL_ACTION}_seed{REHEARSAL_SEED}"
    node["action"] = REHEARSAL_ACTION
    node["handler_module"] = "experiments.round0268_rehearsal"
    node["handler_callable"] = "run_rehearsal_job"
    node["outputs"] = [rehearsal_output]
    node["done_marker"] = os.path.join(artifacts, f"{node['id']}.done.json")
    node["preserved_model_path"] = preserved_model_path
    node["expected_inputs"] = list(seed42["expected_inputs"]) + [preserved_model_sig]
    node["is_non_evidence_rehearsal"] = True

    manifest = dict(real)
    manifest["schema"] = "round0268-r11-transform-pipe-rehearsal-queue-v1"
    manifest["queue_class"] = "gpu-rehearsal-non-evidence"
    manifest["jobs"] = [node]
    manifest["p90_wall_s"] = {node["id"]: 3600.0, "total": 3600.0}
    manifest["gpu_hours_cap"] = 2.0
    manifest["capabilities_produced"] = []
    manifest["registered"] = {
        "what_this_queue_is": (
            "R11 NON-EVIDENCE plumbing rehearsal: run the REAL seed-42 train node via the real "
            "runner with ONLY model.fit stubbed to load attempt-4's byte-identical preserved model, "
            "so the whole post-training pipe (receipt→seal→fixed transform poll→100M transform→"
            "transform-receipt→done-marker→runner validation) executes before attempt-5. Outputs are "
            "NON-EVIDENCE; attempt-5 produces the round evidence."
        ),
        "preserved_model": preserved_model_sig,
        "reuses_real_seed42_bound_inputs": True,
        "gate_registerable_here": False,
    }
    return manifest


def _rmtree_force(path: str) -> None:
    if not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                os.chmod(os.path.join(root, name), 0o644)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def prepare_rehearsal_queue(*, release_sha: str, queue_root: str = REHEARSAL_ROOT,
                            preserved_model_path: str = PRESERVED_MODEL) -> str:
    if not os.path.exists(preserved_model_path):
        raise RuntimeError(f"R11 rehearsal: preserved model absent at {preserved_model_path}")

    # 1. build the REAL R0268 queue into a throwaway source root (all guards + real signatures).
    source_root = os.path.join(queue_root, "_source_real_queue")
    _rmtree_force(source_root)
    real_manifest_path = prepare_round0268(release_sha=release_sha, queue_root=source_root)
    with open(real_manifest_path) as handle:
        real = json.load(handle)

    # 2. pure transform → single-node rehearsal manifest.
    os.makedirs(os.path.join(queue_root, "artifacts"), exist_ok=True)
    manifest = _rehearsal_manifest_from_real(
        real,
        queue_root=queue_root,
        preserved_model_path=preserved_model_path,
        preserved_model_sig=expected_input_signature(preserved_model_path),
    )
    manifest_path = os.path.join(queue_root, "queue.json")
    if os.path.exists(manifest_path):
        os.chmod(manifest_path, 0o644)
        os.remove(manifest_path)
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare the R11 rehearsal queue (NON-EVIDENCE)")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=REHEARSAL_ROOT)
    parser.add_argument("--preserved-model", default=PRESERVED_MODEL)
    args = parser.parse_args(argv)
    path = prepare_rehearsal_queue(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
        preserved_model_path=args.preserved_model,
    )
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

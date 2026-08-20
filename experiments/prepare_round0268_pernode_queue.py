#!/usr/bin/env python3
"""Prepare (never launch) a PER-NODE R0268 attempt-5 queue — one queue per seed train.

The owner's interruptible-sequence directive (2026-08-20): Queue-B is split into per-node queues so
every node boundary is a clean interrupt point for other GPU work — B1 = seed43 alone, then HOLD,
then B2 = seed44, B3 = seed42 retrain, B4 = panel+gate. The runner enforces no in-queue deps (it runs
listed jobs in order, skipping via done-markers), so each per-node queue lists ONLY its own train node
and ALL of them write into ONE shared canonical artifacts root, so B4's panel can later bind every
seed's sealed train-receipt from that root.

This builder reuses the REAL `prepare_round0268` (so every bound input — substrate, R0243 graph,
R0262 int8, treatment closure — is signature-identical to the flagship) and extracts a single seed's
train node, rewriting ONLY its output + done_marker to the shared artifacts root. It is a build-time
script; it does NOT touch round0268_nodes.py or anything the evidence path imports. It produces REAL
evidence (a genuine 24h train), so it carries the same review bar as every queue builder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.output_safety import atomic_write_new_json
from experiments.prepare_round0268_queue import (
    SEEDS,
    capability_for_seed,
    file_sha256_manifest,
    prepare_round0268,
)

#: The SHARED canonical artifacts root for attempt-5. Every per-node train queue (B1/B2/B3) writes its
#: train output + done-marker here, so B4's panel binds `<ARTIFACTS>/<capability>/train-receipt.json`
#: for all three seeds from one place.
ATTEMPT5_ROOT = "/data/latent-basemap/runs/round-0268/attempt5"
SHARED_ARTIFACTS = os.path.join(ATTEMPT5_ROOT, "artifacts")


def _rmtree_force(path: str) -> None:
    if not os.path.isdir(path):
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                os.chmod(os.path.join(root, name), 0o644)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _single_train_node_manifest(
    real: dict[str, Any], *, seed: int, shared_artifacts: str,
) -> dict[str, Any]:
    """PURE transform: real full manifest → single-node manifest for `seed`'s train, with the node's
    output + done_marker rehomed to the SHARED artifacts root. Every bound-input signature and every
    train field is carried through verbatim (the train is byte-for-byte what the flagship queue runs).
    """
    capability = capability_for_seed(seed)
    train_id = f"train_minilm_fneg_100m_x2_hostint8_seed{seed}"
    node = dict(next(j for j in real["jobs"] if j["id"] == train_id))
    # rehome output + done_marker to the shared root; DO NOT touch any bound-input signature.
    node["outputs"] = [os.path.join(shared_artifacts, capability)]
    node["done_marker"] = os.path.join(shared_artifacts, f"{train_id}.done.json")

    manifest = dict(real)
    manifest["schema"] = "round0268-fneg-100m-x2-pernode-train-queue-v1"
    manifest["queue_class"] = "gpu-training"
    manifest["jobs"] = [node]
    manifest["p90_wall_s"] = {node["id"]: real["p90_wall_s"].get(node["id"], 86400.0), "total":
                              real["p90_wall_s"].get(node["id"], 86400.0)}
    manifest["capabilities_produced"] = [capability]
    manifest["registered"] = {
        "what_this_queue_is": (
            f"attempt-5 PER-NODE train: seed {seed} ALONE (owner interruptible-sequence directive). "
            f"Byte-for-byte the flagship seed-{seed} train; output + done-marker land in the shared "
            f"artifacts root {shared_artifacts} so the terminal panel+gate queue binds all three "
            f"seeds' sealed train-receipts. A node boundary = a clean interrupt point for other GPU "
            f"work; the next queue never auto-starts."
        ),
        "seed": seed,
        "shared_artifacts_root": shared_artifacts,
        "gate_registerable_here": False,
    }
    return manifest


def prepare_pernode_train_queue(
    *, release_sha: str, seed: int, queue_root: str | None = None,
    shared_artifacts: str = SHARED_ARTIFACTS,
) -> str:
    if int(seed) not in SEEDS:
        raise RuntimeError(f"attempt-5 per-node: seed {seed!r} is not a flagship seed {SEEDS}")
    queue_root = queue_root or os.path.join(ATTEMPT5_ROOT, f"queue-seed{seed}")

    # build the REAL full queue into a throwaway source root (all guards + real signatures).
    source_root = os.path.join(queue_root, "_source_real_queue")
    _rmtree_force(source_root)
    real_manifest_path = prepare_round0268(release_sha=release_sha, queue_root=source_root)
    with open(real_manifest_path) as handle:
        real = json.load(handle)

    os.makedirs(shared_artifacts, exist_ok=True)
    os.makedirs(queue_root, exist_ok=True)
    manifest = _single_train_node_manifest(real, seed=seed, shared_artifacts=shared_artifacts)
    manifest_path = os.path.join(queue_root, "queue.json")
    if os.path.exists(manifest_path):
        os.chmod(manifest_path, 0o644)
        os.remove(manifest_path)
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare a per-node R0268 attempt-5 train queue")
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--queue-root", default=None)
    parser.add_argument("--shared-artifacts", default=SHARED_ARTIFACTS)
    args = parser.parse_args(argv)
    path = prepare_pernode_train_queue(
        release_sha=args.release_sha, seed=args.seed,
        queue_root=args.queue_root, shared_artifacts=args.shared_artifacts,
    )
    print(json.dumps({"queue": path, "sha256": file_sha256_manifest(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

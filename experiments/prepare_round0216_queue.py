#!/usr/bin/env python3
"""Prepare, but never launch, the R0216 MiniLM 2M substrate + exact k15 graph."""
from __future__ import annotations

import argparse, glob, json, os, re, subprocess, sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json, create_fresh_directory, ensure_data_directory)
from basemap.round0216_minilm_2m_substrate import (
    CAPABILITY, COMPOSITION, DIMENSION, GRAPH_K, MAX_ZERO_DEGREE_ROWS,
    MEAN_RECALL_FLOOR, P10_RECALL_FLOOR, RAW_FORMAT, ROUND_ID, ROWS,
    ROW_POLICY, SELECTION_SEED, TRAILING_FRAGMENT_POLICY, resolve_shard_rows)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter

EMB = "/data/embeddings"
ROUND_ROOT = "/data/latent-basemap/runs/round-0216"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0216-2026-08-08.md")
GPU_HOURS_CAP = 2.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    fm = _frontmatter(ROUND_FILE)
    ok = subprocess.run(["git","-C",RELEASE_ROOT,"merge-base","--is-ancestor",
                         str(fm.get("base_commit") or ""), release_sha],
                        check=False, timeout=10).returncode == 0
    if fm.get("round_id") != ROUND_ID or fm.get("status") != "issued" or not ok:
        raise RuntimeError("R0216 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def prepare_round0216(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0216 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    inputs, corpora = [round_signature], {}
    for corpus, want in COMPOSITION:
        shards = sorted(p for p in glob.glob(os.path.join(EMB, corpus, "train", "*.npy"))
                        if not p.endswith(".tmp.npy"))
        if not shards:
            raise RuntimeError(f"R0216 found no shards for {corpus}")
        total = 0
        for p in shards:
            with open(p, "rb") as h:
                real = h.read(6) == b"\x93NUMPY"
            if real:
                import numpy as np
                total += int(np.load(p, mmap_mode="r").shape[0])
            else:
                total += resolve_shard_rows(relative_path=os.path.relpath(p, EMB),
                                            size_bytes=os.path.getsize(p))
        if total < want:
            raise RuntimeError(f"R0216 {corpus}: need {want}, corpus has {total}")
        corpora[corpus] = {
            "shards": len(shards), "rows": total, "selected": want,
            "shard_sizes": {os.path.relpath(p, EMB): os.path.getsize(p) for p in shards},
        }
        # Binding 581 GB of base-corpus shards by SHA-256 costs ~10 minutes per
        # prepare. Protocol v2.1 allows size-check plus declared-hash lineage for
        # large inputs a prior reviewed round already verified at creation: R0025
        # bound these exact files by hash when it built the int8 corpus. So the
        # large corpora are bound by an explicit size manifest (hashed itself,
        # and re-verified inside the node), while the new code corpus — small
        # enough to hash directly — is bound by hash.
        if corpus.startswith("starcoderdata"):
            inputs.extend(expected_input_signature(p) for p in shards)
            corpora[corpus]["binding"] = "sha256 per shard"
        else:
            corpora[corpus]["binding"] = (
                "size manifest + R0025 declared-hash lineage (protocol v2.1)"
            )
    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0216 GPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    manifest_path = os.path.join(queue_root, "source-size-manifest.json")
    atomic_write_new_json(manifest_path, {
        "schema": "round0216-source-size-manifest-v1",
        "embeddings_root": EMB, "corpora": corpora,
        "binding_policy": (
            "large base corpora bound by exact byte size here and re-verified in "
            "the node; hash lineage carried by R0025 which bound these files at "
            "creation. The new code corpus is bound by SHA-256 directly."
        ),
    }, immutable=True)
    inputs.append(expected_input_signature(manifest_path))
    r0025 = ("/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
             "int8-shards-v1.json")
    if os.path.exists(r0025):
        inputs.append(expected_input_signature(r0025))
    job_extra = {"source_size_manifest": expected_input_signature(manifest_path)}
    job = {
        "id": "assemble_2m_substrate_and_graph",
        "action": "assemble_2m_substrate_and_graph",
        "handler_module": "experiments.round0216_nodes",
        "handler_callable": "run_job",
        "deps": [], "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, "substrate-graph.done.json"),
        "expected_inputs": _dedupe(inputs), "p90_wall_s": 5_400.0,
        **job_extra,
        "node_policy": {"gpu_required": True, "training_performed": False, "cpu_heavy": True},
    }
    queue = _base_manifest(round_id=ROUND_ID, release_sha=release_sha, round_file=ROUND_FILE,
                           queue_root=queue_root, gpu_hours_cap=GPU_HOURS_CAP,
                           execution_authority="autonomous-gpu", gpu=True)
    queue.update({
        "schema": "round0216-minilm-2m-substrate-queue-v1", "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-substrate", "required_reviews": [],
        "capability_dependencies": [], "capabilities_produced": [CAPABILITY],
        "training_performed": False, "jobs": [job],
        "p90_gpu_seconds": {"assemble_2m_substrate_and_graph": 5_400.0, "total": 5_400.0},
        "scientific_contract": {
            "rows": ROWS, "dimension": DIMENSION, "k": GRAPH_K,
            "composition": {n: c for n, c in COMPOSITION},
            "corpora": corpora, "selection_seed": SELECTION_SEED,
            "graph": "exact brute-force fp32 cosine top-k, never quantized",
            "mean_recall_floor": MEAN_RECALL_FLOOR, "p10_recall_floor": P10_RECALL_FLOOR,
            "max_zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
            "loading_contract": {"raw_format": RAW_FORMAT, "row_policy": ROW_POLICY,
                                 "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY},
            "training_performed": False, "production_or_publishing": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-sha", required=True)
    ap.add_argument("--queue-root", default=QUEUE_ROOT)
    a = ap.parse_args(argv)
    print(json.dumps({"queue_manifest": prepare_round0216(
        release_sha=a.release_sha, queue_root=a.queue_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

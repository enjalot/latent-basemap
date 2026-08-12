#!/usr/bin/env python3
"""Prepare, but never launch, the R0261 4M substrate + exact k15 graph queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json, create_fresh_directory, ensure_data_directory)
from basemap.round0261_four_m_graph import (
    BACK_CHECK_REL_TOL, CAPABILITY, COMPOSITION, CPU_PROBE_ROWS, DIMENSION,
    GPU_PROBE_ROWS, GRAPH_K, MEAN_RECALL_FLOOR, MAX_ZERO_DEGREE_ROWS,
    P10_RECALL_FLOOR, PROGRAM_RECALL_FLOOR, QUERY_BLOCK, RAW_FORMAT, ROUND_ID,
    ROWS, ROW_POLICY, SEARCH_BLOCK, SELECTION_SEED, TRAILING_FRAGMENT_POLICY,
    cost_prediction, other_predictions, resolve_shard_rows,
)
from experiments.round0261_nodes import (
    BUILD_ACTION, PREDICT_ACTION, PREDICT_CAPABILITY)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter

EMB = "/data/embeddings"
ROUND_ROOT = "/data/latent-basemap/runs/round-0261"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0261-2026-08-12.md")
GPU_HOURS_CAP = 3.0

#: The two sealed prior walls the prediction is built from. Both are bound as
#: queue inputs and re-read inside the prediction node.
R0216_GRAPH_RECEIPT = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1/substrate-graph.json"
)
R0233_TRUTH_RECEIPT = (
    "/data/latent-basemap/runs/round-0233/queue-correction-1/artifacts/"
    "minilm-mixed-6250k-exact-k15-truth-v1/exact-k15-truth.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    fm = _frontmatter(ROUND_FILE)
    ok = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
         str(fm.get("base_commit") or ""), release_sha],
        check=False, timeout=10).returncode == 0
    if fm.get("round_id") != ROUND_ID or fm.get("status") != "issued" or not ok:
        raise RuntimeError("R0261 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def prepare_round0261(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0261 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    for path in (R0216_GRAPH_RECEIPT, R0233_TRUTH_RECEIPT):
        if not os.path.exists(path):
            raise RuntimeError(f"R0261 needs the sealed prior wall at {path}")
    r0216_sig = expected_input_signature(R0216_GRAPH_RECEIPT)
    r0233_sig = expected_input_signature(R0233_TRUTH_RECEIPT)

    inputs: list[dict[str, Any]] = [round_signature, r0216_sig, r0233_sig]
    corpora: dict[str, Any] = {}
    for corpus, want in COMPOSITION:
        shards = sorted(p for p in glob.glob(os.path.join(EMB, corpus, "train", "*.npy"))
                        if not p.endswith(".tmp.npy"))
        if not shards:
            raise RuntimeError(f"R0261 found no shards for {corpus}")
        total = 0
        for p in shards:
            with open(p, "rb") as handle:
                real = handle.read(6) == b"\x93NUMPY"
            if real:
                import numpy as np
                total += int(np.load(p, mmap_mode="r").shape[0])
            else:
                total += resolve_shard_rows(relative_path=os.path.relpath(p, EMB),
                                            size_bytes=os.path.getsize(p))
        if total < want:
            raise RuntimeError(f"R0261 {corpus}: need {want}, corpus has {total}")
        corpora[corpus] = {
            "shards": len(shards), "rows": total, "selected": want,
            "shard_sizes": {os.path.relpath(p, EMB): os.path.getsize(p) for p in shards},
        }
        # Protocol v2.1 binding, identical to R0216: the three large base corpora
        # are bound by an explicit size manifest (itself hashed, and re-verified
        # inside the node) on R0025's declared-hash lineage; the small code
        # corpus is bound by SHA-256 per shard.
        if corpus.startswith("starcoderdata"):
            inputs.extend(expected_input_signature(p) for p in shards)
            corpora[corpus]["binding"] = "sha256 per shard"
        else:
            corpora[corpus]["binding"] = (
                "size manifest + R0025 declared-hash lineage (protocol v2.1)")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0261 GPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    manifest_path = os.path.join(queue_root, "source-size-manifest.json")
    atomic_write_new_json(manifest_path, {
        "schema": "round0261-source-size-manifest-v1",
        "embeddings_root": EMB, "corpora": corpora,
        "binding_policy": (
            "large base corpora bound by exact byte size here and re-verified in "
            "the node; hash lineage carried by R0025 which bound these files at "
            "creation. The code corpus is bound by SHA-256 directly."),
    }, immutable=True)
    manifest_sig = expected_input_signature(manifest_path)
    inputs.append(manifest_sig)

    predict_out = os.path.join(artifacts, PREDICT_CAPABILITY)
    build_out = os.path.join(artifacts, CAPABILITY)
    prediction_path = os.path.join(predict_out, "predict_0261-price-prediction.json")

    predict_job = {
        "id": "predict_0261",
        "action": PREDICT_ACTION,
        "handler_module": "experiments.round0261_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [predict_out],
        "done_marker": os.path.join(artifacts, "predict_0261.done.json"),
        "expected_inputs": _dedupe([round_signature, r0216_sig, r0233_sig]),
        "r0216_graph_receipt": r0216_sig,
        "r0233_truth_receipt": r0233_sig,
        "p90_wall_s": 300.0,
        "node_policy": {"gpu_required": False, "training_performed": False,
                        "cpu_heavy": False},
    }
    build_job = {
        "id": "build_0261",
        "action": BUILD_ACTION,
        "handler_module": "experiments.round0261_nodes",
        "handler_callable": "run_job",
        "deps": ["predict_0261"],
        "outputs": [build_out],
        "done_marker": os.path.join(artifacts, "build_0261.done.json"),
        "expected_inputs": _dedupe(inputs),
        "source_size_manifest": manifest_sig,
        "price_prediction_path": prediction_path,
        "p90_wall_s": 5_400.0,
        "node_policy": {"gpu_required": True, "training_performed": False,
                        "cpu_heavy": True},
    }

    queue = _base_manifest(round_id=ROUND_ID, release_sha=release_sha,
                           round_file=ROUND_FILE, queue_root=queue_root,
                           gpu_hours_cap=GPU_HOURS_CAP,
                           execution_authority="autonomous-gpu", gpu=True)
    prediction = cost_prediction()
    queue.update({
        "schema": "round0261-four-m-exact-graph-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-substrate",
        "required_reviews": ["0215", "0216", "0220", "0233", "0243", "0257", "0260"],
        "capability_dependencies": [],
        "capabilities_produced": [PREDICT_CAPABILITY, CAPABILITY],
        "training_performed": False,
        "jobs": [predict_job, build_job],
        "p90_gpu_seconds": {"predict_0261": 0.0, "build_0261": 5_400.0,
                            "total": 5_400.0},
        "scientific_contract": {
            "question": (
                "what does the 4,000,000-row exact k15 fuzzy graph that "
                "design-0260 §5 E3 needs actually cost, on the same universe as "
                "the sealed n = 29 family?"),
            "choice": (
                "BUILD it rather than extrapolate. Two sealed walls (R0216 at "
                "2,000,000 rows, R0233 at 6,250,000) bound the answer between "
                "336.57 s and 448.76 s of exact search, which is 0.09-0.12 "
                "GPU-h against a 3.0 GPU-h cap, so the measurement is cheaper "
                "than the argument about the extrapolation -- and it leaves the "
                "artifact E3 needs on disk."),
            "rows": ROWS, "dimension": DIMENSION, "k": GRAPH_K,
            "composition": {name: n for name, n in COMPOSITION},
            "corpora": corpora,
            "selection_seed": SELECTION_SEED,
            "nested_in_r0216_2m": False,
            "graph": "exact brute-force fp32 cosine top-k, never quantized",
            "query_block": QUERY_BLOCK, "search_block": SEARCH_BLOCK,
            "block_geometry_is_r0216s": True,
            "gating_probe": "independent plain-NumPy CPU brute-force pass",
            "gating_probe_rows": CPU_PROBE_ROWS,
            "builder_probe_rows": GPU_PROBE_ROWS,
            "builder_probe_gates": False,
            "mean_recall_floor": MEAN_RECALL_FLOOR,
            "p10_recall_floor": P10_RECALL_FLOOR,
            "program_recall_floor": PROGRAM_RECALL_FLOOR,
            "max_zero_degree_rows": MAX_ZERO_DEGREE_ROWS,
            "registered_cost_prediction": prediction,
            "registered_other_predictions": other_predictions(),
            "back_check_rel_tol": BACK_CHECK_REL_TOL,
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
    print(json.dumps({"queue_manifest": prepare_round0261(
        release_sha=a.release_sha, queue_root=a.queue_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

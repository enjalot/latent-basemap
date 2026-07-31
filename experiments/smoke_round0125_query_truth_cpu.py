#!/usr/bin/env python3
"""CUDA-hidden R0125 accepted-query-truth load -> score -> seal smoke."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.panel_v2 import ffr_from_neighbors, recall_at_k_from_neighbors
from basemap.round0125_runtime_bridge import (
    R0104_QUERY_TRUTH_KEY,
    R0104_QUERY_TRUTH_PATH,
    R0104_QUERY_TRUTH_PRODUCER_BACKEND,
    R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256,
    R0104_QUERY_TRUTH_SHA256,
    seal,
    validate_seal,
)
from experiments.round0125_nodes import _load_accepted_r0104_query_truth


def _source_files() -> list[dict[str, object]]:
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    return [
        expected_input_signature(os.path.join(root, relative))
        for relative in (
            "basemap/panel_v2.py",
            "basemap/round0125_runtime_bridge.py",
            "experiments/round0125_nodes.py",
            "experiments/smoke_round0125_query_truth_cpu.py",
        )
    ]


def run_smoke(
    *, release_sha: str, output_root: str, enforce_checkout: bool = False
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("R0125 query-truth smoke release SHA must be one commit")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R0125 query-truth smoke requires CUDA_VISIBLE_DEVICES=''")
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    if enforce_checkout:
        observed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
        if observed != release_sha or dirty:
            raise RuntimeError(
                "query-truth smoke requires the exact clean release checkout"
            )

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA-hidden query-truth smoke discovered CUDA")
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = create_fresh_directory(
        output_root, label="R0125 accepted query-truth CPU smoke"
    )
    try:
        signature = expected_input_signature(R0104_QUERY_TRUTH_PATH)
        if signature["sha256"] != R0104_QUERY_TRUTH_SHA256:
            raise RuntimeError("accepted R0104 query-truth bytes changed")
        truth = _load_accepted_r0104_query_truth({
            "query_truth": signature,
            "query_truth_key": R0104_QUERY_TRUTH_KEY,
        })
        neighbors = np.asarray(truth["neighbors"], dtype=np.int64)
        representative = np.array(neighbors[:32], copy=True)
        if representative.shape != (32, 10):
            raise RuntimeError("accepted truth lacks the representative score rows")
        ffr = ffr_from_neighbors(representative, representative, 10)
        recall = recall_at_k_from_neighbors(representative, representative, 10)
        policy = truth["key_parts"]["policy"]
        checks = {
            "cuda_hidden": not torch.cuda.is_available(),
            "one_torch_thread": torch.get_num_threads() == 1,
            "exact_accepted_archive": signature["sha256"]
            == R0104_QUERY_TRUTH_SHA256,
            "exact_accepted_key": truth["key"] == R0104_QUERY_TRUTH_KEY,
            "historical_producer_implementation_authenticated": (
                policy["implementation_sha256"]
                == R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256
            ),
            "historical_producer_backend_authenticated": (
                policy["candidate_compute_backend"]
                == R0104_QUERY_TRUTH_PRODUCER_BACKEND
            ),
            "representative_truth_scored": ffr == 1.0 and recall == 1.0,
        }
        receipt = seal({
            "schema": "round0125-accepted-query-truth-cpu-smoke-v1",
            "release_sha": release_sha,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_threads": torch.get_num_threads(),
            "query_truth": signature,
            "query_truth_key": truth["key"],
            "query_truth_payload_sha256": truth["payload_sha256"],
            "query_truth_producer_policy": policy,
            "representative_rows": len(representative),
            "representative_neighbors_sha256": ordered_array_sha256(
                representative
            ),
            "metrics": {"ffr": ffr, "recall_at_10": recall},
            "source_files": _source_files(),
            "checks": checks,
            "outcome": "passed" if all(checks.values()) else "failed",
        })
        receipt_path = os.path.join(output, "query-truth-smoke.json")
        atomic_write_new_json(receipt_path, receipt, immutable=True)
        with open(receipt_path, encoding="utf-8") as handle:
            validate_seal(json.load(handle), label="R0125 query-truth smoke")
        if receipt["outcome"] != "passed":
            raise RuntimeError(f"R0125 query-truth smoke failed: {checks}")
        return receipt_path
    finally:
        torch.set_num_threads(prior_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = run_smoke(
        release_sha=args.release_sha,
        output_root=args.output,
        enforce_checkout=True,
    )
    print(json.dumps({"query_truth_smoke": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

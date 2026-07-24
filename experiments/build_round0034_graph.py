#!/usr/bin/env python3
"""Build the pre-issuance canonical graph capability for Round 0034."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.round0034_pipeline import build_canonical_graph


DEFAULT_GRAPH = "/data/checkpoints/pumap/edges_150m_k15.npz"
DEFAULT_GRAPH_SHA256 = (
    "4cf448a05bfdc230f3a538dffaa641a1ab4969b075c7b0628a41fc2ee80d990a"
)


def load_released_eligibility(path: str, expected_sha256: str, row_count: int):
    """Import R0033 lazily so this branch can merge cleanly after its release."""
    try:
        from basemap.int8_eligibility import load_int8_eligibility
    except ImportError as error:
        raise RuntimeError(
            "R0033 implementation is not in this checkout; merge its accepted "
            "release before building the R0034 graph"
        ) from error
    return load_int8_eligibility(
        path, expected_sha256=expected_sha256, row_count=row_count
    )


def run(
    *,
    graph_path: str,
    graph_sha256: str,
    eligibility_path: str,
    eligibility_sha256: str,
    output_root: str,
    row_count: int = 150_000_000,
) -> dict[str, Any]:
    eligibility = load_released_eligibility(
        eligibility_path, eligibility_sha256, row_count
    )
    return build_canonical_graph(
        graph_path=graph_path,
        expected_graph_sha256=graph_sha256,
        eligibility=eligibility,
        output_root=output_root,
        row_count=row_count,
        k=15,
    )


def run_job(active: dict[str, Any]) -> dict[str, Any]:
    job = active.get("job") or {}
    if active.get("manifest", {}).get("round_id") != "0034":
        raise RuntimeError("R0034 graph builder received another queue")
    if job.get("id") != "canonical_graph_150m" or len(job.get("outputs") or []) != 1:
        raise RuntimeError("R0034 canonical graph job contract changed")
    return run(
        graph_path=job["graph_path"],
        graph_sha256=job["graph_sha256"],
        eligibility_path=job["eligibility_path"],
        eligibility_sha256=job["eligibility_sha256"],
        output_root=job["outputs"][0],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--graph-sha256", default=DEFAULT_GRAPH_SHA256)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--eligibility-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(
        graph_path=args.graph,
        graph_sha256=args.graph_sha256,
        eligibility_path=args.eligibility,
        eligibility_sha256=args.eligibility_sha256,
        output_root=args.output,
    )
    print(json.dumps(result, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

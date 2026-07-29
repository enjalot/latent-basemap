#!/usr/bin/env python3
"""Materialize the one-node CPU-only R0033 queue."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from experiments.census_round0033 import INT8, MANIFEST, SCALES


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"


def prepare(release_sha: str, *, date: str = "2026-07-22") -> str:
    root = ensure_data_directory("/data/latent-basemap/runs/round-0033")
    queue_root = create_fresh_directory(os.path.join(root, "queue"), label="R0033 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = [expected_input_signature(path) for path in (MANIFEST, INT8, SCALES)]
    review = os.path.join(LAB_ROOT, "review-0025-2026-07-20.md")
    inputs.append(expected_input_signature(review))
    manifest = {
        "schema_version": 1,
        "program": "basemap-100m-round-0033",
        "round_id": "0033",
        "round_sha256": sha256_file(os.path.join(LAB_ROOT, f"round-0033-{date}.md")),
        "release_sha": release_sha,
        "execution_authority": "autonomous-cpu",
        "required_reviews": ["0025"],
        "gpu_hours_cap": 0.0,
        "queue_class": "cpu-preparation",
        "training_performed": False,
        "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)).isoformat(timespec="seconds"),
        "repo_root": RUN_ROOT,
        "lease_path": "/data/latent-basemap/.gpu_lease",
        "child_environment": {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPYCACHEPREFIX": os.path.join(queue_root, "cache", "pythonpycacheprefix"),
            "NUMBA_CACHE_DIR": os.path.join(queue_root, "cache", "numba"),
        },
        "jobs": [
            {
                "id": "eligibility_census_150m",
                "handler_module": "experiments.census_round0033",
                "handler_callable": "run_job",
                "deps": [],
                "done_marker": os.path.join(artifacts, "eligibility_census_150m.done.json"),
                "outputs": [os.path.join(artifacts, "eligibility")],
                "expected_inputs": inputs,
                "p90_wall_s": 7200,
                "node_policy": {"gpu_required": False},
            }
        ],
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--date", default="2026-07-22")
    args = parser.parse_args(argv)
    print(prepare(args.release_sha, date=args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

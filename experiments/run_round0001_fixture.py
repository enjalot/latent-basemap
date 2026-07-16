"""Run the Round 0001 all-node fixture and persist a content-bound report."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.artifact_identity import git_checkout_state, path_signature


TESTS = [
    "tests/test_round0001_admission.py::test_no_model_does_not_require_sample_indices_and_model_path_does",
    "tests/test_panel_v2.py::test_projection_scores_when_held_out",
    "tests/test_o2_sparse_anchors.py::test_sparse_artifacts_save_reload_roundtrip",
    "tests/test_round0001_admission.py::test_selector_cohorts_self_exclusion_retention_and_true_jaccard",
    "tests/test_round0001_admission.py::test_status_registry_accepts_only_reviewed_evidence_by_default",
    "tests/test_round0001_admission.py::test_contract_fixture_consumes_query_and_enforces_allowlist",
    "tests/test_round0001_admission.py::test_controller_boundary_rejection_stops_sentinel",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    out = os.path.realpath(args.out)
    if not out.startswith("/data/"):
        raise ValueError("fixture report must be written under /data")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [args.python, "-m", "pytest", *TESTS, "-q"]
    started = time.time()
    proc = subprocess.run(command, cwd=root, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = {
        "schema": "round0001_all_node_fixture.v1",
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "wall_s": round(time.time() - started, 3),
        "cuda_visible_devices": "",
        "tests": TESTS,
        "command": command,
        "checkout": git_checkout_state(root),
        "source_signatures": [
            path_signature(os.path.join(root, "basemap", "experiment_contract.py")),
            path_signature(os.path.join(root, "basemap", "evidence_status.py")),
            path_signature(os.path.join(root, "basemap", "cohort_metrics.py")),
            path_signature(os.path.join(root, "basemap", "run_controller.py")),
        ],
        "output": proc.stdout,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(proc.stdout, end="")
    print(out)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

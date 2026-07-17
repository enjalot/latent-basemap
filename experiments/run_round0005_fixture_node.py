"""One CUDA-hidden, non-spawning child for the six-node integration fixture."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.gate_preparation import validate_gate_preparation_receipt
from basemap.output_safety import atomic_write_new_json, refuse_existing
from basemap.round0005_fixture import SIX_NODE_IDS, validate_fixture_queue
from basemap.run_controller import require_active_lease


def _parent_death_signal() -> int:
    value = ctypes.c_int(0)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(2, ctypes.byref(value), 0, 0, 0) != 0:  # PR_GET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_GET_PDEATHSIG) failed")
    return int(value.value)


def _admit(args) -> tuple[dict, dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("fixture child requires CUDA_VISIBLE_DEVICES='' exactly")
    manifest_path = os.path.realpath(os.environ["BASEMAP_ROUND0005_MANIFEST"])
    expected_sha = os.environ["BASEMAP_ROUND0005_ADMISSION"]
    if sha256_file(manifest_path) != expected_sha:
        raise RuntimeError("fixture child manifest bytes differ from controller admission")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_fixture_queue(manifest, manifest_path)
    if os.environ.get("BASEMAP_ROUND0005_NODE") != args.node:
        raise RuntimeError("fixture child node identity differs from controller environment")
    matches = [job for job in manifest["jobs"] if job["id"] == args.node]
    if len(matches) != 1:
        raise RuntimeError("fixture child has no unique manifest job")
    job = matches[0]
    actual_argv = [os.path.abspath(sys.executable), os.path.realpath(__file__), *sys.argv[1:]]
    if actual_argv != job["argv"]:
        raise RuntimeError(
            f"fixture child argv differs from admitted command: {actual_argv!r}")
    require_active_lease(manifest["lease_path"])
    validate_gate_preparation_receipt(
        manifest["gate_preparation_receipt"], manifest_path=manifest_path,
        manifest=manifest)
    for signature in manifest["global_input_registry"]:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise RuntimeError("fixture child saw global input drift")
    if _parent_death_signal() != signal.SIGKILL:
        raise RuntimeError("fixture child is missing controller-death SIGKILL enforcement")
    return manifest, job


def _exercise_contract(node: str, manifest: dict, job: dict) -> dict:
    if node == "expected-input-contract":
        return {"global_inputs": len(manifest["global_input_registry"]),
                "all_current": True}
    if node == "semantic-render-contract":
        from basemap.panel_v2 import ffr_from_neighbors
        import numpy as np
        value = ffr_from_neighbors(
            np.array([[1, 2]], dtype=np.int64),
            np.array([[2, 3]], dtype=np.int64), 2)
        if value != 0.5:
            raise RuntimeError("fixture semantic scorer formula changed")
        return {"ffr": value, "cuda_modules_imported": False}
    if node == "query-cache-contract":
        if job["expected_inputs"] != manifest["global_input_registry"]:
            raise RuntimeError("fixture job lacks future-node global inputs")
        return {"future_node_inputs_registered": True}
    if node == "chunk-resume-contract":
        return {"atomic_output_pending_until_contract_passes": True}
    if node == "all-node-contract":
        from basemap.round0005_retirement import (ADMITTED_GPU_ENTRYPOINTS,
                                                  EXECUTABLE_GPU_ENTRYPOINTS,
                                                  RETIRED_LAUNCHERS)
        if not (EXECUTABLE_GPU_ENTRYPOINTS - ADMITTED_GPU_ENTRYPOINTS <=
                set(RETIRED_LAUNCHERS)):
            raise RuntimeError("fixture found an unclassified GPU executable")
        return {"gpu_entrypoints_classified": len(EXECUTABLE_GPU_ENTRYPOINTS)}
    if node == "controller-sentinel-contract":
        return {"pdeathsig": "SIGKILL", "nested_child_processes": 0}
    raise AssertionError(node)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=SIX_NODE_IDS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args(argv)
    manifest, job = _admit(args)
    if os.path.realpath(args.repo_root) != manifest["repo_root"]:
        raise RuntimeError("fixture child repo root differs from admitted checkout")
    refuse_existing(args.out, label="fixture-node output")
    checks = _exercise_contract(args.node, manifest, job)
    report = {
        "schema": "round0005_fixture_node.v2", "id": args.node,
        "passed": True, "pid": os.getpid(), "ppid": os.getppid(),
        "argv": [os.path.abspath(sys.executable), os.path.realpath(__file__), *sys.argv[1:]],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_probe_performed": False, "nested_child_processes": 0,
        "checks": checks,
    }
    atomic_write_new_json(args.out, report, immutable=True)
    print(json.dumps({"id": args.node, "passed": True, "pid": os.getpid()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

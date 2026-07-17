"""The one admitted basemap-100m Round 0005 production program.

This is intentionally not a general job schema.  The issued round authorizes a
single ordered six-node DAG, six canonical scripts, fixed argv layouts, and
derived resource/scientific policy.  Queue JSON records the derived policy for
auditability, but validation always reopens the canonical inputs and recomputes
it rather than trusting the JSON values.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .release_preflight import validate_release_preflight_receipt
from .source_closure import validate_source_closure_receipt

ROUND0005_ROUND_SHA256 = "a99d88b8d7d3631d1381075a541e4974767e5f954a08308e268749e27368214c"
ROUND0005_ROUND_FILE = (
    "/home/enjalot/code/latent-labs/basemap-100m/round-0005-2026-07-16.md")
ROUND0005_PROGRAM_INPUT_ROLES = (
    "calibration_inventory",
    "environment_manifest",
    "fixture",
    "maps_seal",
    "model_root",
    "model_seal",
    "query_artifact",
    "query_expectation",
    "release_preflight_receipt",
    "round_file",
    "scorer_fixture",
    "testbed_seal",
)
HASH64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    script: str
    dependency: str | None
    required_free_gb: float
    gpu_memory_cap_mb: int
    predicted_wall_s: float
    p90_wall_s: float
    row_kind: str


ROUND0005_NODES = (
    NodeSpec("fresh_uncached_2m", "experiments/score_complete_panel.py", None,
             28.0, 26 * 1024, 540.0, 650.0, "testbed"),
    NodeSpec("cached_nine_map", "experiments/score_complete_panel.py",
             "fresh_uncached_2m", 28.0, 26 * 1024, 120.0, 180.0, "testbed"),
    NodeSpec("scalar_equivalence", "experiments/compare_panel_cache.py",
             "cached_nine_map", 28.0, 26 * 1024, 660.0, 850.0, "testbed"),
    NodeSpec("synthetic_4x_regression", "experiments/round0005_performance_gate.py",
             "scalar_equivalence", 2.0, 4 * 1024, 30.0, 45.0, "scorer_fixture"),
    NodeSpec("embedding_calibration", "experiments/calibrate_jina_embedding.py",
             "synthetic_4x_regression", 12.0, 16 * 1024, 240.0, 300.0,
             "calibration_inventory"),
    NodeSpec("no_training_seal_canary", "experiments/run_round0005_seal_canary.py",
             "embedding_calibration", 4.0, 8 * 1024, 30.0, 45.0,
             "canonical_seal_rows"),
)
ROUND0005_NODE_BY_ID = {node.node_id: node for node in ROUND0005_NODES}
ROUND0005_JOB_FIELDS = {
    "id", "argv", "inputs", "expected_inputs", "outputs", "done_marker", "log",
    "manifest", "cwd", "predicted_wall_s", "p90_wall_s", "deps", "node_policy",
}


def _role_map(manifest: dict) -> dict[str, dict]:
    values = manifest.get("program_inputs")
    if not isinstance(values, list):
        raise ValueError("queue program_inputs must be an ordered role/signature list")
    roles: dict[str, dict] = {}
    for entry in values:
        if not isinstance(entry, dict) or set(entry) != {"role", "signature"}:
            raise ValueError("queue program input entries have invalid fields")
        role = entry["role"]
        if role in roles:
            raise ValueError(f"queue program input role is duplicated: {role}")
        roles[role] = entry["signature"]
    if tuple(sorted(roles)) != ROUND0005_PROGRAM_INPUT_ROLES:
        raise ValueError(
            f"queue program input roles mismatch: expected={ROUND0005_PROGRAM_INPUT_ROLES} "
            f"observed={tuple(sorted(roles))}")
    return roles


def _validate_bound_release(manifest: dict) -> dict[str, Any]:
    signature = _role_map(manifest)["release_preflight_receipt"]
    return validate_release_preflight_receipt(
        signature["canonical_path"],
        expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=signature)


def _validate_program_input_signatures(manifest: dict) -> dict[str, str]:
    roles = _role_map(manifest)
    paths: dict[str, str] = {}
    for role in ROUND0005_PROGRAM_INPUT_ROLES:
        signature = roles[role]
        if not isinstance(signature, dict) or "canonical_path" not in signature:
            raise ValueError(f"program input {role} lacks a content signature")
        path = signature["canonical_path"]
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ValueError(f"program input {role} path must be absolute")
        if os.path.realpath(path) != path:
            raise ValueError(f"program input {role} traverses a symlink")
        observed = expected_input_signature(path)
        if observed != signature:
            raise ValueError(
                f"program input {role} changed: expected={signature!r} observed={observed!r}")
        paths[role] = path
    if paths["round_file"] != ROUND0005_ROUND_FILE:
        raise ValueError("queue does not bind the canonical full Round 0005 file")
    if roles["round_file"].get("sha256") != ROUND0005_ROUND_SHA256:
        raise ValueError("canonical full Round 0005 file hash changed")
    if paths["environment_manifest"] != os.path.realpath(manifest["environment_manifest"]):
        raise ValueError("program environment manifest differs from queue binding")
    _validate_bound_release(manifest)
    return paths


def _derive_rows(spec: NodeSpec, context: dict[str, Any]) -> int:
    if spec.row_kind == "testbed":
        import numpy as np

        rows = int(np.load(
            os.path.join(context["testbed"], "sample_indices.npy"),
            mmap_mode="r", allow_pickle=False).shape[0])
        if rows != 2_000_000:
            raise ValueError(
                f"{spec.node_id} requires reopened 2,000,000-row testbed, observed {rows}")
        return rows
    if spec.row_kind == "scorer_fixture":
        from experiments.compare_panel_cache import load_fixture

        fixture = load_fixture(context["paths"]["scorer_fixture"])
        rows = int(fixture["X"].shape[0])
        if rows <= 0:
            raise ValueError("synthetic regression fixture has no scientific rows")
        return rows
    if spec.row_kind == "calibration_inventory":
        import pyarrow.parquet as pq

        rows = int(pq.ParquetFile(context["inventory_parquet"]).metadata.num_rows)
        if rows != 50_000:
            raise ValueError(
                f"embedding calibration requires reopened 50,000 rows, observed {rows}")
        return rows
    if spec.row_kind == "canonical_seal_rows":
        return 4096
    raise AssertionError(f"unknown canonical row derivation {spec.row_kind}")


def derive_program_context(manifest: dict, *, repo_root: str) -> dict[str, Any]:
    """Reopen all seals and derive paths/rows needed by the exact program."""
    paths = _validate_program_input_signatures(manifest)
    release = _validate_bound_release(manifest)
    if release["release_sha"] != manifest["release_sha"]:
        raise ValueError("release preflight receipt release differs from queue release")
    if release["run_checkout_path"] != os.path.realpath(repo_root):
        raise ValueError("release preflight run checkout differs from queue repo_root")

    validate_source_closure_receipt(manifest["source_closure"], repo_root=repo_root)

    # Delayed imports avoid a queue-admission/fixture import cycle.
    from .query_artifact import load_query_artifact, validate_convention
    from .round0005_fixture import validate_round0005_fixture
    from .round0005_staging import (
        MAP_EXPECTATIONS, ROUND0005_MODEL_ID, ROUND0005_MODEL_REVISION,
        cross_check_round0005_data_identity, validate_round0005_testbed_seal,
        validate_staged_map_seal, validate_staged_model_seal,
    )
    from experiments.calibrate_jina_embedding import validate_inventory

    fixture = validate_round0005_fixture(
        paths["fixture"], repo_root=repo_root, release_sha=manifest["release_sha"],
        environment_manifest=paths["environment_manifest"])
    testbed_seal = validate_round0005_testbed_seal(
        paths["testbed_seal"], require_round0005=True)
    testbed = testbed_seal["testbed_root"]
    maps = validate_staged_map_seal(
        paths["maps_seal"], expected_testbed_seal=paths["testbed_seal"],
        require_round0005=True)
    if maps.get("expected_rows") != 2_000_000:
        raise ValueError("Round 0005 map seal does not contain exact 2M maps")
    by_label = {entry["label"]: entry["staged_dir"] for entry in maps["maps"]}
    if set(by_label) != set(MAP_EXPECTATIONS) or len(by_label) != 9:
        raise ValueError("Round 0005 map seal is not the exact nine-map corpus")
    model = validate_staged_model_seal(
        paths["model_seal"], expected_root=paths["model_root"],
        expected_model_id=ROUND0005_MODEL_ID,
        expected_revision=ROUND0005_MODEL_REVISION,
        expected_testbed_seal=paths["testbed_seal"], require_round0005=True)
    with open(paths["query_expectation"], encoding="utf-8") as handle:
        query_convention = validate_convention(json.load(handle))
    query = load_query_artifact(
        paths["query_artifact"], testbed=testbed,
        expected_convention=query_convention,
        expected_testbed_seal=paths["testbed_seal"], require_round0005=True)
    inventory, _ = validate_inventory(
        paths["calibration_inventory"],
        expected_testbed_seal=paths["testbed_seal"], require_round0005=True)
    inventory_parquet = inventory["inventory"]["canonical_path"]
    data_closure = cross_check_round0005_data_identity(
        testbed_seal_path=paths["testbed_seal"], maps_seal_path=paths["maps_seal"],
        model_seal_path=paths["model_seal"],
        query_manifest_path=paths["query_artifact"],
        calibration_manifest_path=paths["calibration_inventory"],
        maps_root=maps["destination_root"], model_root=paths["model_root"])
    expected_data_identity = manifest["input_staging"]["data_closure_identity_sha256"]
    if data_closure["identity_sha256"] != expected_data_identity:
        raise ValueError("reopened Round 0005 data closure identity changed")
    if model["model_revision"] != manifest["input_staging"]["model_revision"]:
        raise ValueError("reopened staged model revision changed")
    context = {
        "paths": paths,
        "fixture": fixture,
        "testbed": testbed,
        "maps": {label: by_label[label] for label in sorted(by_label)},
        "model": model,
        "query": query,
        "inventory": inventory,
        "inventory_parquet": inventory_parquet,
    }
    context["rows"] = {
        node.node_id: _derive_rows(node, context) for node in ROUND0005_NODES
    }
    return context


def derived_node_policy(spec: NodeSpec, *, scientific_rows: int) -> dict[str, Any]:
    return {
        "schema": "round0005_derived_node_policy.v2",
        "node_id": spec.node_id,
        "canonical_script": spec.script,
        "training_performed": False,
        "gpu_required": True,
        "cuda_device_count": 1,
        "scientific_rows": int(scientific_rows),
        "required_free_gb": spec.required_free_gb,
        "gpu_memory_cap_mb": spec.gpu_memory_cap_mb,
        "scale_certificate_required": int(scientific_rows) >= 8_000_000,
    }


def _run_pairs(context: dict[str, Any]) -> list[str]:
    return [f"{label}={context['maps'][label]}" for label in sorted(context["maps"])]


def expected_argv(spec: NodeSpec, *, manifest: dict, context: dict[str, Any],
                  queue_root: str) -> list[str]:
    python = _validate_bound_release(manifest)["python_invocation_path"]
    script = os.path.join(manifest["repo_root"], spec.script)
    artifacts = os.path.join(queue_root, "artifacts")
    runs = _run_pairs(context)
    paths = context["paths"]
    if spec.node_id == "fresh_uncached_2m":
        return [
            python, script, "--runs", *runs, "--testbed", context["testbed"],
            "--dim", "768", "--query-artifact", paths["query_artifact"],
            "--query-expectation", paths["query_expectation"],
            "--query-cache-mode", "off", "--require-round0005-nine-maps",
            "--expected-highd-builds", "1", "--wall-max", "720", "--peak-max", "26",
            "--out-root", os.path.join(artifacts, "uncached-nine-map"),
        ]
    if spec.node_id == "cached_nine_map":
        return [
            python, script, "--runs", *runs, "--testbed", context["testbed"],
            "--dim", "768", "--query-artifact", paths["query_artifact"],
            "--query-expectation", paths["query_expectation"],
            "--query-cache-mode", "on", "--require-round0005-nine-maps",
            "--expected-highd-builds", "1", "--wall-max", "120", "--peak-max", "26",
            "--out-root", os.path.join(artifacts, "cached-nine-map"),
        ]
    if spec.node_id == "scalar_equivalence":
        return [
            python, script, "--runs", *runs, "--testbed", context["testbed"],
            "--query-artifact", paths["query_artifact"],
            "--query-expectation", paths["query_expectation"], "--dim", "768",
            "--frac", "0.001", "--n-anchors", "10000", "--seed", "123",
            "--require-round0005-nine-maps", "--out-root",
            os.path.join(artifacts, "scalar-equivalence"),
        ]
    if spec.node_id == "synthetic_4x_regression":
        return [python, script, "--fixture", paths["scorer_fixture"],
                "--out-root", os.path.join(artifacts, "synthetic-4x-regression")]
    if spec.node_id == "embedding_calibration":
        return [
            python, script, "--inventory-manifest", paths["calibration_inventory"],
            "--model-path", paths["model_root"], "--model-seal", paths["model_seal"],
            "--testbed-seal", paths["testbed_seal"], "--out-dir",
            os.path.join(artifacts, "embedding-calibration"), "--batch-size", "256",
            "--dtype", "float32", "--max-prediction-error", "0.15",
        ]
    if spec.node_id == "no_training_seal_canary":
        return [
            python, script, "--fixture", paths["fixture"], "--scorer-fixture",
            paths["scorer_fixture"], "--out",
            os.path.join(artifacts, "no-training-seal-canary.json"),
            "--rows", "4096", "--dimensions", "32", "--seed", "20260716",
        ]
    raise AssertionError(spec.node_id)


def expected_outputs(spec: NodeSpec, *, queue_root: str) -> list[str]:
    root = os.path.join(queue_root, "artifacts")
    if spec.node_id == "fresh_uncached_2m":
        base = os.path.join(root, "uncached-nine-map")
        return [os.path.join(base, value) for value in
                ("report.json", "hiD-reference.npz", "hiD-reference-receipt.json")]
    if spec.node_id == "cached_nine_map":
        base = os.path.join(root, "cached-nine-map")
        return [os.path.join(base, value) for value in
                ("report.json", "hiD-reference.npz", "hiD-reference-receipt.json",
                 "query-truth-cache")]
    if spec.node_id == "scalar_equivalence":
        base = os.path.join(root, "scalar-equivalence")
        outputs = [os.path.join(base, "equivalence.json")]
        for mode in ("uncached", "cached"):
            for name in ("report.json", "hiD-reference.npz",
                         "hiD-reference-receipt.json", "child-process.json"):
                outputs.append(os.path.join(base, mode, name))
        outputs.append(os.path.join(base, "cached", "query-truth-cache"))
        return outputs
    if spec.node_id == "synthetic_4x_regression":
        base = os.path.join(root, "synthetic-4x-regression")
        return [os.path.join(base, name) for name in
                ("baseline.json", "slowed.json", "regression.json")]
    if spec.node_id == "embedding_calibration":
        return [os.path.join(root, "embedding-calibration")]
    if spec.node_id == "no_training_seal_canary":
        return [os.path.join(root, "no-training-seal-canary.json")]
    raise AssertionError(spec.node_id)


def validate_exact_program(manifest: dict, *, manifest_path: str,
                           repo_root: str) -> dict[str, Any]:
    """Reject anything except the issued six-node program and derived policy."""
    context = derive_program_context(manifest, repo_root=repo_root)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or [job.get("id") for job in jobs
                                     if isinstance(job, dict)] != [
            node.node_id for node in ROUND0005_NODES]:
        raise ValueError("queue must contain the exact six issued node IDs in order")
    queue_root = os.path.dirname(os.path.realpath(manifest_path))
    registry = manifest.get("global_input_registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("queue needs a nonempty manifest-global input registry")
    registry_paths = [entry.get("canonical_path") if isinstance(entry, dict) else None
                      for entry in registry]
    if registry_paths != sorted(registry_paths) or len(registry_paths) != len(set(registry_paths)):
        raise ValueError("manifest-global input registry must be sorted and unique")
    for signature in registry:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise ValueError(
                f"manifest-global queue input changed: {signature['canonical_path']}")
    required_global_paths = {
        entry["signature"]["canonical_path"] for entry in manifest["program_inputs"]
    } | {
        entry["signature"]["canonical_path"]
        for entry in manifest["source_closure"]["members"]
    }
    if not required_global_paths.issubset(set(registry_paths)):
        missing = sorted(required_global_paths - set(registry_paths))
        raise ValueError(f"manifest-global input registry omits required inputs: {missing}")

    for position, (job, spec) in enumerate(zip(jobs, ROUND0005_NODES)):
        if set(job) != ROUND0005_JOB_FIELDS:
            raise ValueError(
                f"queue job {spec.node_id} fields mismatch: "
                f"expected={sorted(ROUND0005_JOB_FIELDS)} observed={sorted(job)}")
        expected_deps = [] if spec.dependency is None else [spec.dependency]
        if job["deps"] != expected_deps:
            raise ValueError(f"queue job {spec.node_id} has a noncanonical DAG edge")
        expected_command = expected_argv(
            spec, manifest=manifest, context=context, queue_root=queue_root)
        if job["argv"] != expected_command:
            raise ValueError(
                f"queue job {spec.node_id} argv is not canonical: "
                f"expected={expected_command!r} observed={job['argv']!r}")
        if "-c" in job["argv"] or job["argv"][1] != os.path.join(repo_root, spec.script):
            raise ValueError(f"queue job {spec.node_id} uses an arbitrary Python program")
        if job["inputs"] != registry_paths or job["expected_inputs"] != registry:
            raise ValueError(
                f"queue job {spec.node_id} must bind the complete global input registry, "
                "including fixture and future-node-only inputs")
        outputs = expected_outputs(spec, queue_root=queue_root)
        if job["outputs"] != outputs:
            raise ValueError(f"queue job {spec.node_id} output contract is not canonical")
        controls = {
            "done_marker": os.path.join(queue_root, "artifacts", f"{spec.node_id}.done.json"),
            "log": os.path.join(queue_root, "artifacts", f"{spec.node_id}.log"),
            "manifest": os.path.join(
                queue_root, "artifacts", f"{spec.node_id}.controller.json"),
        }
        for field, value in controls.items():
            if job[field] != value:
                raise ValueError(f"queue job {spec.node_id} {field} is not canonical")
        if job["cwd"] != repo_root:
            raise ValueError(f"queue job {spec.node_id} cwd differs from release checkout")
        if (float(job["predicted_wall_s"]) != spec.predicted_wall_s or
                float(job["p90_wall_s"]) != spec.p90_wall_s):
            raise ValueError(f"queue job {spec.node_id} timing registry is not canonical")
        policy = derived_node_policy(
            spec, scientific_rows=context["rows"][spec.node_id])
        if job["node_policy"] != policy:
            raise ValueError(
                f"queue job {spec.node_id} self-declared row/GPU/training policy rejected: "
                f"expected={policy!r} observed={job['node_policy']!r}")
        if policy["scale_certificate_required"]:
            raise ValueError(
                f"the issued Round 0005 program contains no >=8M node: {spec.node_id}")
        if position and spec.dependency != jobs[position - 1]["id"]:
            raise ValueError("Round 0005 exact DAG must be one ordered fail-stop chain")
    policy_body = {
        "schema": "round0005_program_policy.v2",
        "node_ids": [node.node_id for node in ROUND0005_NODES],
        "training_performed": False,
        "all_nodes_require_fixture": True,
        "all_nodes_require_one_gpu": True,
        "exact_ordered_fail_stop_dag": True,
    }
    expected_program_policy = {
        **policy_body, "identity_sha256": sha256_bytes(canonical_json(policy_body))}
    if manifest.get("program_policy") != expected_program_policy:
        raise ValueError("queue program policy is not the canonical derived policy")
    return context

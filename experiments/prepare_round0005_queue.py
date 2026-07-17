"""Build the one immutable, fully signed Round 0005 no-training queue."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       sha256_bytes)
from basemap.output_safety import (atomic_write_new_json, canonical_data_path,
                                   create_fresh_directory, refuse_existing)
from basemap.query_artifact import load_query_artifact, validate_convention
from basemap.queue_admission import (CACHE_KEYS, NVIDIA_SMI_EXECUTABLE,
                                     _emit_component_rejection_receipt,
                                     validate_canonical_gpu_environment,
                                     validate_queue_manifest)
from basemap.release_preflight import (issue_release_preflight_receipt,
                                       release_reports_equivalent,
                                       verify_release)
from basemap.round0005_fixture import validate_round0005_fixture
from basemap.round0005_program import (
    ROUND0005_NODES, ROUND0005_PROGRAM_INPUT_ROLES, ROUND0005_ROUND_FILE,
    ROUND0005_ROUND_SHA256, derive_program_context, derived_node_policy,
    expected_argv, expected_outputs,
)
from basemap.round0005_staging import (
    ROUND0005_MODEL_ID, ROUND0005_MODEL_REVISION,
    cross_check_round0005_data_identity, validate_round0005_testbed_seal,
    validate_staged_map_seal, validate_staged_model_seal,
)
from basemap.roundwatch_gate import canonical_roundwatch_binding
from basemap.run_controller import process_identity
from basemap.source_closure import source_closure_receipt
from experiments.calibrate_jina_embedding import validate_inventory
from experiments.compare_panel_cache import load_fixture as load_scorer_fixture

ROUND_ID = "0005"
PROGRAM = "basemap-100m"
GPU_HOURS_CAP = 0.75
ROUND_SHA256 = ROUND0005_ROUND_SHA256
_FIXTURE_BUILDER_CAPABILITY = object()


def _manifest_sha(value: dict) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
               + "\n").encode("utf-8")
    return sha256_bytes(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--implementation-commit", action="append", required=True)
    parser.add_argument("--integration-repo", required=True)
    parser.add_argument("--pushed-ref", required=True,
                        help="explicit refs/remotes/<remote>/<branch> ref")
    parser.add_argument("--round-sha256", required=True)
    parser.add_argument("--round-file", default=ROUND0005_ROUND_FILE)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--scorer-fixture", required=True)
    parser.add_argument("--query-artifact", required=True)
    parser.add_argument("--query-expectation", required=True)
    parser.add_argument("--maps-seal", required=True)
    parser.add_argument("--calibration-inventory", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-seal", required=True)
    parser.add_argument("--testbed-seal", required=True)
    parser.add_argument("--environment-manifest",
                        default="/data/latent-basemap/envs/run-env.json")
    parser.add_argument(
        "--allowed-service", action="append", default=[], metavar="PID:MARKER:VRAM_MB",
        help="pre-gate service identity; raw PID-only allow lists are forbidden")
    parser.add_argument("--deadline-utc")
    parser.add_argument("--out", required=True)
    return parser


def build_queue(args, *, _fixture_phase_hook=None, _fixture_capability=None) -> dict:
    """Build and publish a queue; the hook is private to CPU integration tests."""
    if (_fixture_phase_hook is not None and
            _fixture_capability is not _FIXTURE_BUILDER_CAPABILITY):
        raise RuntimeError("builder adversarial hook requires fixture-only capability")
    if args.round_sha256 != ROUND_SHA256:
        raise ValueError("Round 0005 queue uses the wrong issued-round SHA-256")
    if os.path.realpath(args.round_file) != ROUND0005_ROUND_FILE:
        raise ValueError("Round 0005 queue must bind the canonical full round file")
    if expected_input_signature(args.round_file)["sha256"] != ROUND_SHA256:
        raise ValueError("canonical full Round 0005 file bytes changed")

    run_root = os.path.realpath(args.run_root)
    round_root = canonical_data_path(args.round_root, label="round root")
    queue_root = canonical_data_path(args.queue_root, label="queue root")
    out = canonical_data_path(args.out, label="queue manifest")
    if out != os.path.join(queue_root, "queue.json"):
        raise ValueError("Round 0005 queue manifest must be queue-root/queue.json")
    refuse_existing(queue_root, label="Round 0005 queue root")
    refuse_existing(out, label="Round 0005 queue manifest")

    with open(args.environment_manifest, encoding="utf-8") as handle:
        environment = json.load(handle)
    validate_canonical_gpu_environment(environment)
    venv = os.path.realpath(environment["venv_path"])
    python = os.path.realpath(os.path.join(venv, "bin", "python"))
    if not os.path.isfile(python):
        raise FileNotFoundError(f"sealed venv Python is missing: {python}")

    cache_root = os.path.join(queue_root, "cache")
    cache_environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    for key in CACHE_KEYS:
        cache_environment[key] = os.path.join(cache_root, key.lower())
    child_environment = {
        **cache_environment,
        "CUDA_VISIBLE_DEVICES": environment["gpu_uuid"],
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "TOKENIZERS_PARALLELISM": "false",
    }

    # Fail before creating queue/output state if the release is merely local,
    # dirty, has changed implementation ancestry, or does not match the venv.
    preflight = verify_release(
        integration_repo=args.integration_repo, release_sha=args.release_sha,
        implementation_commits=args.implementation_commit, pushed_ref=args.pushed_ref,
        run_checkout=run_root, environment_manifest=args.environment_manifest,
        cache_environment=cache_environment)
    if not preflight["passed"]:
        raise RuntimeError("release preflight rejected: " + "; ".join(preflight["errors"]))

    fixture = validate_round0005_fixture(
        args.fixture, repo_root=run_root, release_sha=args.release_sha,
        environment_manifest=args.environment_manifest)
    testbed_seal = validate_round0005_testbed_seal(
        args.testbed_seal, require_round0005=True)
    testbed = testbed_seal["testbed_root"]
    maps = validate_staged_map_seal(
        args.maps_seal, expected_testbed_seal=args.testbed_seal,
        require_round0005=True)
    if maps.get("expected_rows") != 2_000_000 or len(maps.get("maps") or []) != 9:
        raise ValueError("queue requires the exact nine production 2,000,000-row maps")
    if maps["destination_root"] != os.path.join(round_root, "inputs", "maps"):
        raise ValueError("maps are not staged at the exact Round 0005 runtime root")
    model = validate_staged_model_seal(
        args.model_seal, expected_root=args.model_path,
        expected_model_id=ROUND0005_MODEL_ID,
        expected_revision=ROUND0005_MODEL_REVISION,
        expected_testbed_seal=args.testbed_seal, require_round0005=True)
    with open(args.query_expectation, encoding="utf-8") as handle:
        query_convention = validate_convention(json.load(handle))
    query = load_query_artifact(
        args.query_artifact, testbed=testbed, expected_convention=query_convention,
        expected_testbed_seal=args.testbed_seal, require_round0005=True)
    inventory, _ = validate_inventory(
        args.calibration_inventory, expected_testbed_seal=args.testbed_seal,
        require_round0005=True)
    load_scorer_fixture(args.scorer_fixture)
    data_closure = cross_check_round0005_data_identity(
        testbed_seal_path=args.testbed_seal, maps_seal_path=args.maps_seal,
        model_seal_path=args.model_seal, query_manifest_path=args.query_artifact,
        calibration_manifest_path=args.calibration_inventory,
        maps_root=maps["destination_root"], model_root=args.model_path)
    sources = source_closure_receipt(run_root, require_tracked=True)
    roundwatch = canonical_roundwatch_binding()

    allowed_processes = []
    for value in args.allowed_service:
        try:
            raw_pid, marker, raw_budget = value.split(":", 2)
            allowed_processes.append(process_identity(
                int(raw_pid), marker=marker, gpu_memory_budget_mb=int(raw_budget)))
        except Exception as exc:
            raise ValueError(
                f"invalid --allowed-service {value!r}; expected PID:MARKER:VRAM_MB") from exc
    if len({entry["pid"] for entry in allowed_processes}) != len(allowed_processes):
        raise ValueError("duplicate allowed service PID identity")

    # Only after all read-only release/data/source validation passes do we create
    # fresh queue scaffolding and publish the replayable release receipt.
    create_fresh_directory(queue_root, label="Round 0005 queue root")
    artifacts = create_fresh_directory(
        os.path.join(queue_root, "artifacts"), label="queue artifact parent")
    receipts = create_fresh_directory(
        os.path.join(queue_root, "gate-receipts"), label="queue receipt root")
    checkpoints = create_fresh_directory(
        os.path.join(queue_root, "controller-checkpoints"),
        label="queue controller checkpoint root")
    create_fresh_directory(cache_root, label="queue cache parent")
    for key in CACHE_KEYS:
        create_fresh_directory(cache_environment[key], label=f"queue cache {key}")
    for name in ("uncached-nine-map", "cached-nine-map", "scalar-equivalence"):
        create_fresh_directory(os.path.join(artifacts, name),
                               label=f"{name} private output parent")
    terminal = os.path.join(queue_root, "controller-terminal.json")
    refuse_existing(terminal, label="queue controller terminal summary")
    gate_preparation_receipt = os.path.join(queue_root, "gate-preparation.json")
    refuse_existing(gate_preparation_receipt, label="queue gate preparation receipt")
    release_receipt_path = os.path.join(queue_root, "release-preflight.json")
    issued_preflight = issue_release_preflight_receipt(
        release_receipt_path, integration_repo=args.integration_repo,
        release_sha=args.release_sha,
        implementation_commits=args.implementation_commit,
        pushed_ref=args.pushed_ref, run_checkout=run_root,
        environment_manifest=args.environment_manifest,
        cache_environment=cache_environment)
    if not release_reports_equivalent(issued_preflight, preflight):
        raise RuntimeError("release state changed while queue scaffolding was created")

    signature_cache: dict[str, dict] = {}

    def signature(value: str) -> dict:
        canonical = os.path.realpath(value)
        observed = expected_input_signature(canonical)
        prior = signature_cache.get(canonical)
        if prior is not None and prior != observed:
            raise RuntimeError(
                f"input changed during queue signature capture: {canonical}")
        signature_cache[canonical] = observed
        return observed

    query_manifest = query["manifest"]
    inventory_paths = [
        inventory["inventory"]["canonical_path"],
        inventory["source_sample_indices"]["canonical_path"],
        *[entry["canonical_path"] for entry in inventory["source_text_shards"]],
        *[entry["canonical_path"] for entry in inventory["source_embedding_shards"]],
    ]
    data_paths = [
        args.fixture, args.scorer_fixture, args.query_artifact, args.query_expectation,
        query_manifest["embeddings"]["canonical_path"],
        query_manifest["ids"]["canonical_path"],
        query_manifest["source_embeddings"]["canonical_path"],
        args.maps_seal, maps["destination_root"],
        *[entry["staged_dir"] for entry in maps["maps"]],
        args.model_path, args.model_seal, args.testbed_seal,
        os.path.join(testbed, "train"), os.path.join(testbed, "sample_indices.npy"),
        os.path.join(testbed, "centroids_k256.npy"),
        os.path.join(testbed, "centroids_k1024.npy"),
        args.calibration_inventory, *inventory_paths,
        args.environment_manifest, environment["freeze_file"], python,
        args.round_file, release_receipt_path,
        roundwatch["cli"]["path"],
        roundwatch["interpreter"]["canonical_path"],
        roundwatch["git"]["canonical_path"],
        NVIDIA_SMI_EXECUTABLE,
        *[entry["canonical_path"] for entry in roundwatch["import_closure"]],
        *[entry["signature"]["canonical_path"] for entry in sources["members"]],
    ]
    for value in data_paths:
        signature(value)
    global_registry = [signature_cache[path] for path in sorted(signature_cache)]

    role_paths = {
        "calibration_inventory": os.path.realpath(args.calibration_inventory),
        "environment_manifest": os.path.realpath(args.environment_manifest),
        "fixture": os.path.realpath(args.fixture),
        "maps_seal": os.path.realpath(args.maps_seal),
        "model_root": os.path.realpath(args.model_path),
        "model_seal": os.path.realpath(args.model_seal),
        "query_artifact": os.path.realpath(args.query_artifact),
        "query_expectation": os.path.realpath(args.query_expectation),
        "release_preflight_receipt": os.path.realpath(release_receipt_path),
        "round_file": os.path.realpath(args.round_file),
        "scorer_fixture": os.path.realpath(args.scorer_fixture),
        "testbed_seal": os.path.realpath(args.testbed_seal),
    }
    if tuple(sorted(role_paths)) != ROUND0005_PROGRAM_INPUT_ROLES:
        raise AssertionError("program input role implementation drift")
    program_inputs = [
        {"role": role, "signature": signature_cache[role_paths[role]]}
        for role in sorted(role_paths)
    ]
    program_policy_body = {
        "schema": "round0005_program_policy.v2",
        "node_ids": [node.node_id for node in ROUND0005_NODES],
        "training_performed": False,
        "all_nodes_require_fixture": True,
        "all_nodes_require_one_gpu": True,
        "exact_ordered_fail_stop_dag": True,
    }
    deadline = (args.deadline_utc or
                (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(
                    timespec="seconds"))
    manifest = {
        "schema_version": 1, "program": PROGRAM, "round_id": ROUND_ID,
        "execution_authority": "planner-gpu", "required_reviews": [],
        "round_sha256": args.round_sha256, "release_sha": args.release_sha,
        "environment_freeze_sha": environment["freeze_sha256"],
        "environment_identity_sha": environment["identity_sha256"],
        "gpu_hours_cap": GPU_HOURS_CAP, "queue_class": "research",
        "training_performed": False, "deadline_utc": deadline,
        "environment_manifest": os.path.realpath(args.environment_manifest),
        "cache_environment": cache_environment,
        "child_environment": child_environment,
        "gate_receipts_dir": receipts,
        "controller_checkpoints_dir": checkpoints,
        "controller_terminal_summary": terminal,
        "gate_preparation_receipt": gate_preparation_receipt,
        "repo_root": run_root,
        "lease_path": "/data/latent-basemap/.gpu_lease",
        "allowed_processes": allowed_processes,
        "jobs": [],
        "input_staging": {
            "maps_seal": signature_cache[os.path.realpath(args.maps_seal)],
            "model_seal": signature_cache[os.path.realpath(args.model_seal)],
            "testbed_seal": signature_cache[os.path.realpath(args.testbed_seal)],
            "data_closure_identity_sha256": data_closure["identity_sha256"],
            "model_revision": model["model_revision"],
        },
        "fixture_identity": {
            "schema": fixture["schema"],
            "canonical_path": os.path.realpath(args.fixture),
            "sha256": signature_cache[os.path.realpath(args.fixture)]["sha256"],
            "identity_sha256": fixture["identity_sha256"],
        },
        "program_policy": {
            **program_policy_body,
            "identity_sha256": sha256_bytes(canonical_json(program_policy_body)),
        },
        "program_inputs": program_inputs,
        "global_input_registry": global_registry,
        "source_closure": sources,
        "roundwatch_binding": roundwatch,
        "release_preflight_identity": issued_preflight["identity_sha256"],
    }
    context = derive_program_context(manifest, repo_root=run_root)
    for spec in ROUND0005_NODES:
        outputs = expected_outputs(spec, queue_root=queue_root)
        controls = {
            "done_marker": os.path.join(artifacts, f"{spec.node_id}.done.json"),
            "log": os.path.join(artifacts, f"{spec.node_id}.log"),
            "manifest": os.path.join(artifacts, f"{spec.node_id}.controller.json"),
        }
        for value in [*outputs, *controls.values()]:
            refuse_existing(value, label=f"queue job {spec.node_id} output/control")
        manifest["jobs"].append({
            "id": spec.node_id,
            "argv": expected_argv(
                spec, manifest=manifest, context=context, queue_root=queue_root),
            "inputs": [item["canonical_path"] for item in global_registry],
            "expected_inputs": global_registry,
            "outputs": outputs,
            **controls,
            "cwd": run_root,
            "predicted_wall_s": spec.predicted_wall_s,
            "p90_wall_s": spec.p90_wall_s,
            "deps": [] if spec.dependency is None else [spec.dependency],
            "node_policy": derived_node_policy(
                spec, scientific_rows=context["rows"][spec.node_id]),
        })

    original_manifest_sha = _manifest_sha(manifest)
    if _fixture_phase_hook is not None:
        _fixture_phase_hook("capture-to-manifest-publication", manifest)
    observed_registry = []
    for expected in global_registry:
        try:
            observed_registry.append(expected_input_signature(expected["canonical_path"]))
        except Exception as exc:
            observed_registry.append({"canonical_path": expected["canonical_path"],
                                      "error": f"{type(exc).__name__}: {exc}"})
    if observed_registry != global_registry:
        receipt = _emit_component_rejection_receipt(
            receipts_dir=receipts, phase="capture-to-manifest-publication",
            manifest_path=out, original_manifest_sha256=original_manifest_sha,
            job=manifest["jobs"][0], expected=global_registry,
            observed=observed_registry,
            error="builder detected input drift after signature capture")
        raise RuntimeError(
            f"queue input changed before manifest publication; receipt={receipt}")

    try:
        validate_queue_manifest(manifest, out)
    except Exception as exc:
        observed = []
        for expected in global_registry:
            try:
                observed.append(expected_input_signature(expected["canonical_path"]))
            except Exception as observed_exc:
                observed.append({"canonical_path": expected["canonical_path"],
                                 "error": f"{type(observed_exc).__name__}: {observed_exc}"})
        if observed != global_registry:
            receipt = _emit_component_rejection_receipt(
                receipts_dir=receipts, phase="capture-to-manifest-publication",
                manifest_path=out, original_manifest_sha256=original_manifest_sha,
                job=manifest["jobs"][0], expected=global_registry, observed=observed,
                error=f"{type(exc).__name__}: {exc}")
            try:
                exc.add_note(f"automatic builder rejection receipt: {receipt}")
            except AttributeError:
                pass
        raise
    atomic_write_new_json(out, manifest, immutable=True)
    return manifest


def _publish_fixture_queue(*, manifest: dict, out: str, validator,
                           phase_hook=None, capability=None) -> dict:
    """Exercise the real capture/publication boundary with a fixture program."""
    if capability is not _FIXTURE_BUILDER_CAPABILITY:
        raise RuntimeError("fixture queue publication capability is invalid")
    captured_registry = json.loads(json.dumps(manifest["global_input_registry"]))
    original_manifest_sha = _manifest_sha(manifest)
    if phase_hook is not None:
        phase_hook("capture-to-manifest-publication", manifest)
    observed = []
    for signature in captured_registry:
        try:
            observed.append(expected_input_signature(signature["canonical_path"]))
        except Exception as exc:
            observed.append({"canonical_path": signature["canonical_path"],
                             "error": f"{type(exc).__name__}: {exc}"})
    if observed != captured_registry:
        expected_state = {"manifest_sha256": original_manifest_sha,
                          "global_inputs": captured_registry}
        observed_state = {"manifest_sha256": _manifest_sha(manifest),
                          "global_inputs": observed}
        receipt = _emit_component_rejection_receipt(
            receipts_dir=manifest["gate_receipts_dir"],
            phase="capture-to-manifest-publication", manifest_path=out,
            original_manifest_sha256=original_manifest_sha,
            job=manifest["jobs"][0], expected=expected_state,
            observed=observed_state,
            error="fixture builder detected input drift after signature capture")
        raise RuntimeError(
            f"fixture queue input changed before publication; receipt={receipt}")
    validator(manifest, out)
    atomic_write_new_json(out, manifest, immutable=True)
    return manifest


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_queue(args)
    print(json.dumps({
        "queue_manifest": os.path.realpath(args.out),
        "jobs": [job["id"] for job in manifest["jobs"]],
        "gpu_hours_cap": manifest["gpu_hours_cap"],
        "deadline_utc": manifest["deadline_utc"],
        "signed_unique_inputs": len(manifest["global_input_registry"]),
        "post_gate_output_consumption": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

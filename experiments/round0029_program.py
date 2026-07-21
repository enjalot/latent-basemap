"""Round 0029 weighted-graph artifact qualification nodes.

The round is intentionally bounded: build one clean 30M artifact, run the
corrected CPU checks, and pull one batch through the real capped hybrid sampler.
It never allocates a model or performs an optimizer update.
"""
from __future__ import annotations

from argparse import Namespace
import json
import os
from typing import Any

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import atomic_write_new_json, create_fresh_directory


INPUT_PACK = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
UNIFORM_GRAPH = "/data/checkpoints/pumap/edges_30m_k15.npz"
DUPLICATE_CAP = (
    "/data/latent-basemap/runs/round-0020/queue/artifacts/"
    "duplicate-census/global-cap-v1.npz"
)
DUPLICATE_CAP_SHA256 = (
    "9511ceca802da603bfbfe9164f8c6ffd7006df82df17b9499d4ed33288fde7cb"
)
EXCLUDED_ROWS = 218_242
RETAINED_ROWS = 29_781_758
ARTIFACT_NAME = "edges_30m_k15_fuzzy-v2.npz"


def ordered_embedding_paths() -> list[str]:
    with open(INPUT_PACK, encoding="utf-8") as handle:
        pack = json.load(handle)
    materialized = pack["capability_payload"]["materialized_fp16"]
    members = materialized["ordered_members"]
    paths = [os.path.realpath(item["path"]) for item in members]
    if (len(paths) != 30
            or [int(item["chunk_index"]) for item in members] != list(range(30))
            or any(item.get("dtype") != "<f2" for item in members)
            or any(item.get("shape") != [1_000_000, 384] for item in members)):
        raise RuntimeError("Round 0029 requires the accepted ordered 30M fp16 pack")
    return paths


def artifact_path(build_root: str) -> str:
    return os.path.join(build_root, ARTIFACT_NAME)


def run_build(*, output_root: str, workdir: str) -> dict[str, Any]:
    from experiments.build_weighted_graph import cmd_build

    output_root = create_fresh_directory(
        output_root, label="Round 0029 weighted graph output")
    output = artifact_path(output_root)
    args = Namespace(
        edges=UNIFORM_GRAPH,
        embeddings_list=ordered_embedding_paths(),
        embeddings_dir=None,
        corpus=None,
        raw_dtype=None,
        out=output,
        workdir=workdir,
        dim=384,
        chunk_size=150_000,
        partitions=64,
        phase_c_workers=1,
        target_neighbors=16,
        device="cuda",
        sharded=False,
        no_sort=False,
        skip_gpu=False,
        force_gpu=False,
        yield_seconds=600.0,
    )
    cmd_build(args)

    manifest_path = output + ".manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    artifact = expected_input_signature(output)
    manifest_receipt = expected_input_signature(manifest_path)
    if (
        manifest.get("schema") != "graph_manifest.v2"
        or manifest.get("production_trainer_ready") is not True
        or manifest.get("builder_dirty") is not False
        or manifest.get("graph_sha256") != artifact["sha256"]
        or manifest.get("n_nodes") != 30_000_000
        or manifest.get("k") != 15
        or manifest.get("n_neighbors_param") != 16
        or manifest.get("distance_compute_dtype") != "float32"
        or len(manifest.get("data_shard_records") or []) != 30
    ):
        raise RuntimeError("Round 0029 clean weighted artifact contract failed")
    receipt = {
        "schema": "round0029-weighted-graph-build-v1",
        "training_performed": False,
        "optimizer_updates": 0,
        "artifact": artifact,
        "manifest": manifest_receipt,
        "graph_sha256": manifest["graph_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "build_contract_sha256": manifest["build_contract_sha256"],
        "n_nodes": manifest["n_nodes"],
        "n_edges": manifest["n_edges"],
        "weight_summary": manifest["weight_summary"],
        "input_topology_validation": manifest["input_topology_validation"],
        "resources": manifest["resources"],
    }
    atomic_write_new_json(
        os.path.join(output_root, "build-receipt.json"), receipt, immutable=True)
    return receipt


def _require_build_receipt(build_root: str) -> tuple[str, dict[str, Any]]:
    receipt_path = os.path.join(build_root, "build-receipt.json")
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    output = artifact_path(build_root)
    if (
        receipt.get("schema") != "round0029-weighted-graph-build-v1"
        or receipt.get("artifact") != expected_input_signature(output)
        or receipt.get("manifest") != expected_input_signature(output + ".manifest.json")
    ):
        raise RuntimeError("Round 0029 build receipt or artifact changed")
    return output, receipt


def run_cpu_validation(*, output_root: str, build_root: str) -> dict[str, Any]:
    import gc
    from experiments.weighted_graph_validate import v3, v4

    output_root = create_fresh_directory(
        output_root, label="Round 0029 CPU validation output")
    artifact, build = _require_build_receipt(build_root)
    common = {
        "embeddings_list": ordered_embedding_paths(),
        "embeddings_dir": None,
        "dim": 384,
        "device": "cpu",
        "target_neighbors": 16,
    }
    v3_path = os.path.join(output_root, "v3-consumer-contract.json")
    result_v3 = v3(Namespace(
        **common,
        artifact=artifact,
        sampler_edges=1_000_000,
        n_draw=10_000_000,
        json_out=v3_path,
    ))
    gc.collect()
    v4_path = os.path.join(output_root, "v4-physical-check.json")
    result_v4 = v4(Namespace(
        **common,
        edges=UNIFORM_GRAPH,
        artifact=artifact,
        n_nodes=20,
        k=15,
        seed=0,
        json_out=v4_path,
    ))
    if result_v3.get("PASS") is not True or result_v4.get("PASS") is not True:
        raise RuntimeError("Round 0029 corrected V3/V4 validation failed")
    result = {
        "schema": "round0029-cpu-validation-v1",
        "training_performed": False,
        "optimizer_updates": 0,
        "graph_sha256": build["graph_sha256"],
        "v3": expected_input_signature(v3_path),
        "v4": expected_input_signature(v4_path),
        "passed": True,
    }
    atomic_write_new_json(
        os.path.join(output_root, "validation-receipt.json"), result,
        immutable=True)
    return result


def run_production_canary(*, output_root: str, build_root: str,
                          validation_root: str) -> dict[str, Any]:
    from experiments.weighted_graph_canary import run_canary

    output_root = create_fresh_directory(
        output_root, label="Round 0029 production canary output")
    artifact, build = _require_build_receipt(build_root)
    validation_path = os.path.join(validation_root, "validation-receipt.json")
    with open(validation_path, encoding="utf-8") as handle:
        validation = json.load(handle)
    if (validation.get("passed") is not True
            or validation.get("graph_sha256") != build["graph_sha256"]):
        raise RuntimeError("Round 0029 canary requires its passing CPU validation")
    args = Namespace(
        artifact=artifact,
        expected_graph_sha256=build["graph_sha256"],
        expected_manifest_sha256=build["manifest_sha256"],
        json_out=os.path.join(output_root, "production-canary.json"),
        batch_size=8192,
        pos_ratio=0.2,
        seed=42,
        duplicate_cap=DUPLICATE_CAP,
        expected_duplicate_cap_sha256=DUPLICATE_CAP_SHA256,
        fixed_edges_per_source=15,
        expected_excluded_rows=EXCLUDED_ROWS,
        expected_retained_rows=RETAINED_ROWS,
    )
    result = run_canary(args)
    if result.get("training_performed") is not False:
        raise RuntimeError("Round 0029 canary unexpectedly trained")
    return result

"""Exact CPU-visible contract for the one Round 0014 production queue.

This is deliberately a closed program, not a reusable scale-run framework.  It
binds the accepted Round-0013 30M MiniLM pack, the five registered scorer
artifacts, one fixed treatment, and one ordered six-node fail-stop chain.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes, sha256_file)
from .minilm_input_pack import (DIMENSION, MATERIALIZED_DTYPE, PACK_SCHEMA,
                                RAW_DTYPE, TOTAL_ROWS, NpyShardMap, PackError,
                                RawMapMember, RawSourceMap, canonical_sha256,
                                read_json, verify_sealed_record)
from .release_preflight import validate_release_preflight_receipt
from .source_closure import validate_round0014_source_closure_receipt


ROUND_ID = "0014"
PROGRAM = "basemap-100m"
ISSUED_BASE = "7dd55bf79d73a894f7e1354c803ec725fc9a7579"
ROUND_FILE = "/home/enjalot/code/latent-labs/basemap-100m/round-0014-2026-07-17.md"
ROUND_SHA256 = "c724f0b478ad7400430a151654572698b9a725883977c4a8dd5203889172493b"
SEQUENCED_REVIEW_FILE = (
    "/home/enjalot/code/latent-labs/basemap-100m/review-0013-2026-07-17-01.md")
SEQUENCED_REVIEW_SHA256 = (
    "9c0cae77ca0800f0d77539ebda0289ae1245a81ce6a1ee0ed8baf1a07beac5e5")
ACCEPTED_MANIFEST = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
ACCEPTED_MANIFEST_FILE_SHA256 = (
    "bf6537b83559c1eb5b52b8e73c4a47315191dfcf1b1128c141d32308b3b167f8")
ACCEPTED_MANIFEST_RECEIPT_SHA256 = (
    "2de6dfd0dcc652c20b1b657aaf0a45bdf40e7f02b5ffada9bcb3f5404c643ca5")
ACCEPTED_CAPABILITY_SHA256 = (
    "8f5a6ba8203aa583bbbdca3383f050e29c443ca0e25d628735bba873075bf7f2")

GRAPH_PATH = "/data/checkpoints/pumap/edges_30m_k15.npz"
GRAPH_SHA256 = "2fc30fc27ced442c5b69fde084ab41c054fcc1bf5e7913a5cee9d20f59baadca"
INDEX_PATH = "/data/checkpoints/pumap/faiss_ivf_pq_30m.index"
INDEX_SHA256 = "1e8a43b358a5174e1e6469b834e12a83a1b07e19316b1da003415711bbcaf567"
CENTROIDS_K256_PATH = "/data/latent-basemap/track1/centroids_minilm_k256.npy"
CENTROIDS_K256_SHA256 = (
    "184de3adb34803a482ba374fcc44d0704f706a7a6c3e963d82723b1f14797b5c")
CENTROIDS_K1024_PATH = "/data/latent-basemap/track1/centroids_minilm_k1024.npy"
CENTROIDS_K1024_SHA256 = (
    "271ec84de29aa659c3b94246ef9bf6260ae8556098e1d0c24d4ae96719376c75")
QUERIES_PATH = "/data/latent-basemap/track1/minilm_queries.npy"
QUERIES_SHA256 = "74e459785c0496904b385a1d11eb229b6164580d3877b9e00f1eed5679dee1b4"
QUERY_PROVENANCE_PATH = "/data/latent-basemap/track1/minilm_queries_prov.json"
QUERY_PROVENANCE_SHA256 = (
    "d622b51fad4f31d5f1771f05b048164b5b82acb5e49825fd2c11aaf38607902d")
GPU_HOURS_CAP = 5.5
GPU_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
HASH64 = re.compile(r"[0-9a-f]{64}")
FULL_SHA = re.compile(r"[0-9a-f]{40}")

PROGRAM_INPUT_ROLES = (
    "accepted_pack_manifest",
    "canary_derivation",
    "environment_manifest",
    "input_reference_manifest",
    "production_config",
    "release_preflight_receipt",
    "round_file",
    "sequenced_review0013",
    "transform_spec_template",
)
JOB_FIELDS = {
    "id", "argv", "inputs", "expected_inputs", "outputs", "done_marker", "log",
    "manifest", "cwd", "predicted_wall_s", "p90_wall_s", "deps", "node_policy",
}


TRAIN_CONFIG: dict[str, Any] = {
    "schema": "round0014-production-config-v1",
    "phrase": "one 30M uniform-k15 MiniLM rung on one GSV RTX 5090",
    "row_universe": {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows": TOTAL_ROWS,
        "input_dimension": DIMENSION,
        "materialized_dtype": np.dtype(MATERIALIZED_DTYPE).str,
    },
    "graph": {
        "path": GRAPH_PATH,
        "sha256": GRAPH_SHA256,
        "k": 15,
        "directed_edges": 450_000_000,
        "sampling": "uniform-over-directed-edges",
        "with_replacement": True,
        "weights_consumed": False,
    },
    "model": {
        "architecture": "residual_bottleneck",
        "input_dimension": DIMENSION,
        "hidden_dimension": 2048,
        "hidden_layers": 3,
        "output_dimension": 2,
        "use_batchnorm": False,
        "use_dropout": False,
        "low_dim_kernel": "legacy_lp",
        "a": 1.0,
        "b": 1.0,
    },
    "optimizer": {
        "seed": 42,
        "learning_rate": 0.001,
        "batch_size": 8192,
        "positive_ratio": 0.05,
        "positive_target_mode": "binary",
        "weighted_edge_sampling": False,
        "correlation_weight": 0.0,
        "clip_grad_norm": 1.0,
        "use_amp": True,
        "schedule": "cosine-v3-positive-budget",
        "warmup_successful_updates": 200,
        "successful_positive_lr_updates": 500_000,
        "reject_neighbors": False,
        "anchored_init": "none",
        "midnear_enabled": False,
    },
    "execution": {
        "device_count": 1,
        "required_pipeline": "device_uniform",
        "residency": "device_fp16",
        "canary_optimizer_updates": 0,
        "minimum_post_setup_headroom_gib": 1.5,
        "subfloor_updates_per_second": 45.0,
        "subfloor_consecutive_windows": 2,
        "warning_updates_per_second": 55.0,
        "full_run_retry_count": 0,
        "canary_retry_count": 0,
    },
    "transform": {
        "input_dtype": np.dtype(RAW_DTYPE).str,
        "model_weight_dtype": "float32",
        "output_dtype": "<f4",
        "output_dimension": 2,
        "model_batch_rows": 4096,
        "read_block_rows": 32768,
        "rows_per_output_chunk": 1_000_000,
        "normalization": "none",
        "centering": "none",
        "stochastic_options": [],
    },
    "scorer": {
        "fraction": 0.001,
        "anchors": 10_000,
        "seed": 123,
        "rerank_dtype": "float32",
        "ffr_threshold": 0.001,
        "registered_index": {"path": INDEX_PATH, "sha256": INDEX_SHA256},
        "centroids": {
            "k256": {"path": CENTROIDS_K256_PATH, "sha256": CENTROIDS_K256_SHA256},
            "k1024": {"path": CENTROIDS_K1024_PATH, "sha256": CENTROIDS_K1024_SHA256},
        },
        "queries": {"path": QUERIES_PATH, "sha256": QUERIES_SHA256},
        "query_provenance": {
            "path": QUERY_PROVENANCE_PATH, "sha256": QUERY_PROVENANCE_SHA256},
        "cache_scalar_equivalence_required": True,
    },
    "decision_thresholds": {
        "ffr_at_0_1_percent_min": 0.40,
        "density_min": 0.55,
        "purity_k256_ratio_min": 0.50,
        "purity_k1024_ratio_min": 0.50,
        "projection_over_untrained_floor_min": 100.0,
        "recall_50_strictly_greater_than_recall_10": True,
    },
}
TRAIN_CONFIG_SHA256 = sha256_bytes(canonical_json(TRAIN_CONFIG))


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    dependency: str | None
    predicted_wall_s: float
    p90_wall_s: float
    training_performed: bool
    output_name: str


NODES = (
    NodeSpec("no_training_seal_canary", None, 180.0, 300.0, False,
             "canary"),
    NodeSpec("train_seed42_30m", "no_training_seal_canary", 8460.0, 10800.0,
             True, "train"),
    NodeSpec("transform_30m", "train_seed42_30m", 900.0, 1800.0, False,
             "coordinates"),
    NodeSpec("high_d_reference", "transform_30m", 600.0, 900.0, False,
             "high-d-reference"),
    NodeSpec("registered_panel", "high_d_reference", 1800.0, 1800.0, False,
             "panel"),
    NodeSpec("semantic_renders", "registered_panel", 300.0, 300.0, False,
             "semantic-renders"),
)
NODE_BY_ID = {item.node_id: item for item in NODES}


def _file_record(role: str, path: str, digest: str, size: int) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != path or not os.path.isabs(path) or os.path.islink(path):
        raise PackError(f"{role} path is not canonical/non-symlinked: {path}")
    if not HASH64.fullmatch(str(digest)) or int(size) <= 0:
        raise PackError(f"{role} has a malformed immutable identity")
    return {"role": role, "canonical_path": path, "kind": "file",
            "bytes": int(size), "sha256": digest}


def _accepted_manifest() -> dict[str, Any]:
    if sha256_file(ACCEPTED_MANIFEST) != ACCEPTED_MANIFEST_FILE_SHA256:
        raise PackError("accepted Round-0013 manifest file hash changed")
    value = read_json(ACCEPTED_MANIFEST)
    verify_sealed_record(value, field="manifest_receipt_sha256")
    payload = value.get("capability_payload")
    if (
        value.get("schema") != PACK_SCHEMA
        or value.get("round_id") != "0013"
        or value.get("manifest_receipt_sha256") != ACCEPTED_MANIFEST_RECEIPT_SHA256
        or value.get("capability_hash_sha256") != ACCEPTED_CAPABILITY_SHA256
        or value.get("implementation_release_commit") != ISSUED_BASE
        or not isinstance(payload, dict)
        or canonical_sha256(payload) != ACCEPTED_CAPABILITY_SHA256
        or payload.get("implementation_release_commit") != ISSUED_BASE
    ):
        raise PackError("accepted Round-0013 manifest/capability tuple changed")
    universe = payload.get("row_universe")
    if universe != {
        "blocked_corpus_order": ["fineweb", "redpajama", "pile"],
        "dimension": DIMENSION,
        "rows_per_corpus": 10_000_000,
        "total_rows": TOTAL_ROWS,
    }:
        raise PackError("accepted 30M row-universe contract changed")
    graph = payload.get("graph")
    weights = graph.get("weights") if isinstance(graph, dict) else None
    if (
        not isinstance(graph, dict)
        or graph.get("path") != GRAPH_PATH
        or graph.get("sha256") != GRAPH_SHA256
        or graph.get("n_nodes") != TOTAL_ROWS
        or graph.get("edge_count") != 450_000_000
        or graph.get("k") != 15
        or not isinstance(weights, dict)
        or weights.get("all_values_bitwise_equal") is not True
        or weights.get("cdf_required") is not False
        or weights.get("constant_value_bits_hex") != "0x3d888889"
        or weights.get("sampling_semantics") != "uniform-over-directed-edges"
    ):
        raise PackError("accepted uniform-k15 graph contract changed")
    return value


def accepted_reference_records(*, full_hash: bool) -> list[dict[str, Any]]:
    """Return the exact 71 pack members plus five registered scorer inputs."""
    manifest = _accepted_manifest()
    payload = manifest["capability_payload"]
    records: list[dict[str, Any]] = []
    for position, member in enumerate(payload["raw_source"]["ordered_members"]):
        records.append(_file_record(
            f"accepted_raw_source_{position:03d}", member["path"],
            member["sha256_full_file"], member["identity"]["size_bytes"]))
    for position, member in enumerate(payload["materialized_fp16"]["ordered_members"]):
        records.append(_file_record(
            f"accepted_materialized_fp16_{position:03d}", member["path"],
            member["sha256"], member["size_bytes"]))
    graph = payload["graph"]
    records.append(_file_record(
        "accepted_graph", graph["path"], graph["sha256"],
        os.stat(graph["path"], follow_symlinks=False).st_size))
    for name in ("sources", "targets"):
        item = graph["ordered_int32_endpoints"][name]
        records.append(_file_record(
            f"accepted_endpoint_{name}", item["path"], item["sha256"],
            item["size_bytes"]))
    upstream = manifest["upstream_manifest"]
    records.append(_file_record(
        "accepted_upstream_round0010_manifest", upstream["path"], upstream["sha256"],
        upstream["size_bytes"]))
    diagnostic = payload["lineage"]["round0012_diagnostic_manifest"]
    records.append(_file_record(
        "accepted_diagnostic_round0012_manifest", diagnostic["path"],
        diagnostic["file_sha256"], diagnostic["size_bytes"]))
    scorer = (
        ("registered_scorer_index", INDEX_PATH, INDEX_SHA256),
        ("registered_centroids_k256", CENTROIDS_K256_PATH, CENTROIDS_K256_SHA256),
        ("registered_centroids_k1024", CENTROIDS_K1024_PATH, CENTROIDS_K1024_SHA256),
        ("registered_queries", QUERIES_PATH, QUERIES_SHA256),
        ("registered_query_provenance", QUERY_PROVENANCE_PATH,
         QUERY_PROVENANCE_SHA256),
    )
    for role, path, digest in scorer:
        records.append(_file_record(
            role, path, digest, os.stat(path, follow_symlinks=False).st_size))
    paths = [item["canonical_path"] for item in records]
    if len(records) != 76 or len(paths) != len(set(paths)):
        raise PackError(
            f"accepted reference closure must contain 76 unique files, got {len(records)}")
    if full_hash:
        for record in records:
            observed = expected_input_signature(record["canonical_path"])
            expected = {key: record[key] for key in
                        ("canonical_path", "kind", "bytes", "sha256")}
            if observed != expected:
                raise PackError(
                    f"immutable input changed: expected={expected!r} observed={observed!r}")
    return records


class Round0014MaterializedArray:
    """Lazy array view over the exact accepted thirty fp16 NPY shards."""

    def __init__(self) -> None:
        manifest = _accepted_manifest()
        members = manifest["capability_payload"]["materialized_fp16"]["ordered_members"]
        if len(members) != 30:
            raise PackError("accepted materialized pack is not exactly thirty shards")
        self._members = tuple(dict(item) for item in members)
        self._map = NpyShardMap(
            self._members, total_rows=TOTAL_ROWS, dimension=DIMENSION,
            dtype=np.dtype(MATERIALIZED_DTYPE))
        self.shape = (TOTAL_ROWS, DIMENSION)
        self.dtype = np.dtype(MATERIALIZED_DTYPE)
        self.loaded_shard_paths = [item["path"] for item in self._members]
        self.shard_paths = list(self.loaded_shard_paths)
        self.round0014_pack_seal = {
            "accepted_manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
            "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
            "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
            "materialized_members_sha256": sha256_bytes(canonical_json([
                {key: item[key] for key in (
                    "chunk_index", "global_row_start", "global_row_stop", "path",
                    "sha256", "shape", "size_bytes", "dtype")}
                for item in self._members
            ])),
        }

    def __len__(self) -> int:
        return TOTAL_ROWS

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Round0014MaterializedArray accepts two-dimensional keys")
            rows, columns = key
            return self.__getitem__(rows)[..., columns]
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += TOTAL_ROWS
            if index < 0 or index >= TOTAL_ROWS:
                raise IndexError(index)
            return self._map.read(index, index + 1)[0]
        if isinstance(key, slice):
            start, stop, step = key.indices(TOTAL_ROWS)
            if step == 1:
                return self._map.read(start, stop)
            return self[np.arange(start, stop, step, dtype=np.int64)]
        indices = np.asarray(key)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise IndexError("row selection must be an integer, slice, or 1-D integers")
        normalized = indices.astype(np.int64, copy=True)
        normalized[normalized < 0] += TOTAL_ROWS
        if np.any(normalized < 0) or np.any(normalized >= TOTAL_ROWS):
            raise IndexError("row selection outside accepted 30M universe")
        output = np.empty((len(normalized), DIMENSION), dtype=self.dtype)
        for member in self._members:
            lo = int(member["global_row_start"])
            hi = int(member["global_row_stop"])
            selected = np.flatnonzero((normalized >= lo) & (normalized < hi))
            if not len(selected):
                continue
            array = np.load(member["path"], mmap_mode="r", allow_pickle=False)
            output[selected] = array[normalized[selected] - lo]
            del array
        return output


def raw_source_map() -> RawSourceMap:
    manifest = _accepted_manifest()
    members = []
    for item in manifest["capability_payload"]["raw_source"]["ordered_members"]:
        members.append(RawMapMember(
            path=Path(item["path"]), corpus=item["corpus"],
            global_start=int(item["output_global_row_start"]),
            global_stop=int(item["output_global_row_stop"]),
            local_start=int(item["selected_local_row_start"]),
            full_rows=int(item["full_rows"]), identity=item["identity"],
            sha256=item["sha256_full_file"]))
    return RawSourceMap(members, total_rows=TOTAL_ROWS, dimension=DIMENSION)


def validate_device_uniform_pack(X: Any, edges_path: str) -> dict[str, Any]:
    """Admit the accepted capability before model allocation in the trainer."""
    if type(X) is not Round0014MaterializedArray:
        raise PackError("device_uniform requires the exact Round0014MaterializedArray type")
    canonical_graph = os.path.realpath(edges_path)
    if canonical_graph != GRAPH_PATH or os.path.islink(edges_path):
        raise PackError("device_uniform graph path differs from the accepted graph")
    graph_sig = expected_input_signature(GRAPH_PATH)
    if graph_sig["sha256"] != GRAPH_SHA256:
        raise PackError("device_uniform graph bytes changed")
    accepted = _accepted_manifest()
    expected_members = accepted["capability_payload"]["materialized_fp16"]["ordered_members"]
    if list(X._members) != expected_members:
        raise PackError("device_uniform materialized member order changed")
    for item in expected_members:
        array = np.load(item["path"], mmap_mode="r", allow_pickle=False)
        if (list(array.shape) != item["shape"] or array.dtype.str != item["dtype"] or
                not array.flags.c_contiguous):
            raise PackError(f"device_uniform shard header changed: {item['path']}")
        del array
    return {
        "schema": "round0014-device-uniform-admission-v1",
        "accepted_manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
        "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
        "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "graph": graph_sig,
        "materialized_members_sha256": X.round0014_pack_seal[
            "materialized_members_sha256"],
        "pipeline": "device_uniform",
        "graph_sampling_semantics": "uniform-over-directed-edges",
        "with_replacement": True,
        "weights_consumed": False,
    }


def derived_node_policy(spec: NodeSpec) -> dict[str, Any]:
    body = {
        "schema": "round0014-derived-node-policy-v1",
        "node_id": spec.node_id,
        "canonical_script": "experiments/run_round0014_node.py",
        "training_performed": spec.training_performed,
        "gpu_required": True,
        "cuda_device_count": 1,
        "scientific_rows": TOTAL_ROWS,
        "required_free_gb": 29.0,
        "gpu_memory_cap_mb": 31 * 1024,
        "scale_certificate_required": False,
        "canary_predecessor": (None if spec.dependency is None else
                               NODES[0].node_id),
        "one_use": True,
        "retry_count": 0,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def expected_argv(spec: NodeSpec, *, manifest: dict[str, Any],
                  manifest_path: str) -> list[str]:
    release = _validate_bound_release(manifest)
    return [
        release["python_invocation_path"],
        os.path.join(manifest["repo_root"], "experiments/run_round0014_node.py"),
        "--queue-manifest", os.path.realpath(manifest_path),
        "--node", spec.node_id,
    ]


def expected_outputs(spec: NodeSpec, *, queue_root: str) -> list[str]:
    return [os.path.join(queue_root, "artifacts", spec.output_name)]


def _role_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("program_inputs")
    if not isinstance(entries, list):
        raise ValueError("Round 0014 program inputs must be an ordered list")
    roles: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"role", "signature"}:
            raise ValueError("Round 0014 program input entry fields changed")
        if entry["role"] in roles:
            raise ValueError("Round 0014 program input role is duplicated")
        roles[entry["role"]] = entry["signature"]
    if tuple(sorted(roles)) != PROGRAM_INPUT_ROLES:
        raise ValueError("Round 0014 program input roles changed")
    return roles


def _validate_bound_release(manifest: dict[str, Any]) -> dict[str, Any]:
    signature = _role_map(manifest)["release_preflight_receipt"]
    return validate_release_preflight_receipt(
        signature["canonical_path"],
        expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=signature)


def derive_program_context(manifest: dict[str, Any], *, repo_root: str) -> dict[str, Any]:
    roles = _role_map(manifest)
    paths: dict[str, str] = {}
    for role in PROGRAM_INPUT_ROLES:
        signature = roles[role]
        path = signature.get("canonical_path") if isinstance(signature, dict) else None
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ValueError(f"Round 0014 program input {role} lacks an absolute path")
        if expected_input_signature(path) != signature:
            raise ValueError(f"Round 0014 program input changed: {role}")
        paths[role] = path
    fixed = {
        "round_file": (ROUND_FILE, ROUND_SHA256),
        "sequenced_review0013": (SEQUENCED_REVIEW_FILE, SEQUENCED_REVIEW_SHA256),
        "accepted_pack_manifest": (ACCEPTED_MANIFEST, ACCEPTED_MANIFEST_FILE_SHA256),
    }
    for role, (path, digest) in fixed.items():
        if paths[role] != path or roles[role]["sha256"] != digest:
            raise ValueError(f"Round 0014 fixed {role} binding changed")
    accepted = _accepted_manifest()
    release = _validate_bound_release(manifest)
    if (release["release_sha"] != manifest["release_sha"] or
            release["run_checkout_path"] != os.path.realpath(repo_root)):
        raise ValueError("Round 0014 release receipt differs from queue checkout")
    validate_round0014_source_closure_receipt(
        manifest["source_closure"], repo_root=repo_root)
    with open(paths["production_config"], encoding="utf-8") as handle:
        config_record = json.load(handle)
    if config_record != {"config": TRAIN_CONFIG, "config_sha256": TRAIN_CONFIG_SHA256,
                         "schema": "round0014-production-config-receipt-v1"}:
        raise ValueError("Round 0014 production config changed")
    with open(paths["input_reference_manifest"], encoding="utf-8") as handle:
        reference_manifest = json.load(handle)
    from .round0014_staging import validate_input_reference_manifest
    validate_input_reference_manifest(reference_manifest, full_hash=False)
    from .round0014_transform import validate_transform_template
    transform_template = validate_transform_template(
        paths["transform_spec_template"], release_root=repo_root,
        release_sha=manifest["release_sha"])
    return {
        "paths": paths,
        "accepted_manifest": accepted,
        "release": release,
        "reference_manifest": reference_manifest,
        "transform_template": transform_template,
    }


def validate_exact_program(manifest: dict[str, Any], *, manifest_path: str,
                           repo_root: str) -> dict[str, Any]:
    context = derive_program_context(manifest, repo_root=repo_root)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or [item.get("id") for item in jobs] != [
            spec.node_id for spec in NODES]:
        raise ValueError("Round 0014 queue must contain its exact six nodes in order")
    registry = manifest.get("global_input_registry")
    paths = [item.get("canonical_path") for item in registry] \
        if isinstance(registry, list) else []
    if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Round 0014 global input registry is not sorted/unique")
    required = {
        entry["signature"]["canonical_path"] for entry in manifest["program_inputs"]
    } | {
        entry["signature"]["canonical_path"] for entry in manifest["source_closure"]["members"]
    } | {
        item["canonical_path"] for item in context["reference_manifest"]["references"]
    }
    if not required.issubset(set(paths)):
        raise ValueError(f"Round 0014 global registry omits {sorted(required - set(paths))}")
    queue_root = os.path.dirname(os.path.realpath(manifest_path))
    for position, (job, spec) in enumerate(zip(jobs, NODES)):
        if set(job) != JOB_FIELDS:
            raise ValueError(f"Round 0014 job fields changed: {spec.node_id}")
        dependency = [] if spec.dependency is None else [spec.dependency]
        if job["deps"] != dependency or (position and spec.dependency != jobs[position - 1]["id"]):
            raise ValueError(f"Round 0014 DAG edge changed: {spec.node_id}")
        if job["argv"] != expected_argv(spec, manifest=manifest,
                                         manifest_path=manifest_path):
            raise ValueError(f"Round 0014 argv changed: {spec.node_id}")
        if job["inputs"] != paths or job["expected_inputs"] != registry:
            raise ValueError(f"Round 0014 job does not bind every pre-gate input: {spec.node_id}")
        if job["outputs"] != expected_outputs(spec, queue_root=queue_root):
            raise ValueError(f"Round 0014 output root changed: {spec.node_id}")
        controls = {
            "done_marker": os.path.join(queue_root, "artifacts", f"{spec.node_id}.done.json"),
            "log": os.path.join(queue_root, "artifacts", f"{spec.node_id}.log"),
            "manifest": os.path.join(queue_root, "artifacts", f"{spec.node_id}.controller.json"),
        }
        if any(job[key] != value for key, value in controls.items()):
            raise ValueError(f"Round 0014 controller path changed: {spec.node_id}")
        if (job["cwd"] != repo_root or
                float(job["predicted_wall_s"]) != spec.predicted_wall_s or
                float(job["p90_wall_s"]) != spec.p90_wall_s or
                job["node_policy"] != derived_node_policy(spec)):
            raise ValueError(f"Round 0014 derived node policy changed: {spec.node_id}")
    aggregate = sum(item.p90_wall_s * 1.15 for item in NODES)
    if aggregate > GPU_HOURS_CAP * 3600:
        raise AssertionError("Round 0014 registered p90+15% exceeds 5.5 hours")
    policy_body = {
        "schema": "round0014-program-policy-v1",
        "node_ids": [item.node_id for item in NODES],
        "training_nodes": [item.node_id for item in NODES if item.training_performed],
        "one_no_training_canary": True,
        "one_seed42_treatment": True,
        "exact_ordered_fail_stop_dag": True,
        "retry_count": 0,
        "registered_p90_plus_margin_seconds": aggregate,
    }
    expected_policy = {**policy_body,
                       "identity_sha256": sha256_bytes(canonical_json(policy_body))}
    if manifest.get("program_policy") != expected_policy:
        raise ValueError("Round 0014 program policy changed")
    return context


def program_policy() -> dict[str, Any]:
    body = {
        "schema": "round0014-program-policy-v1",
        "node_ids": [item.node_id for item in NODES],
        "training_nodes": [item.node_id for item in NODES if item.training_performed],
        "one_no_training_canary": True,
        "one_seed42_treatment": True,
        "exact_ordered_fail_stop_dag": True,
        "retry_count": 0,
        "registered_p90_plus_margin_seconds": sum(item.p90_wall_s * 1.15 for item in NODES),
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}

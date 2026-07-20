"""The one controller-admitted executable for all six Round 0014 nodes."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       ordered_array_sha256, sha256_bytes)
from basemap.output_safety import (atomic_build_new_file, atomic_save_new_npy,
                                   atomic_write_new_json, create_fresh_directory)
from basemap.round0014_program import (
    ACCEPTED_CAPABILITY_SHA256, CENTROIDS_K1024_PATH, CENTROIDS_K256_PATH,
    GRAPH_PATH, INDEX_PATH, NODES, QUERIES_PATH, QUERY_PROVENANCE_PATH,
    Round0014MaterializedArray, TRAIN_CONFIG, TRAIN_CONFIG_SHA256,
)


ROUND_ID = "0014"
ROUND_LABEL = "Round 0014"
SCHEMA_PREFIX = "round0014"
RUNTIME_SCRIPT = "experiments/run_round0014_node.py"
RoundMaterializedArray = Round0014MaterializedArray

MINILM_UNTRAINED_FLOOR_PATH = (
    "/data/latent-basemap/runs/round-0018/posthoc-untrained-floor.json"
)
MINILM_UNTRAINED_FLOOR_SHA256 = (
    "d0bf82364a004850fb766e7ff647c01b4cab7437567c2dc2b1b634f6795fdc1c"
)


def configure_round0015() -> None:
    """Select the one additive Round-0015 wrapper before child admission."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global RoundMaterializedArray, TRAIN_CONFIG, TRAIN_CONFIG_SHA256, NODES
    from basemap.round0015_program import (
        NODES as round0015_nodes,
        Round0015MaterializedArray,
        TRAIN_CONFIG as round0015_config,
        TRAIN_CONFIG_SHA256 as round0015_config_sha256,
    )
    ROUND_ID = "0015"
    ROUND_LABEL = "Round 0015"
    SCHEMA_PREFIX = "round0015"
    RUNTIME_SCRIPT = "experiments/run_round0015_node.py"
    RoundMaterializedArray = Round0015MaterializedArray
    TRAIN_CONFIG = round0015_config
    TRAIN_CONFIG_SHA256 = round0015_config_sha256
    NODES = round0015_nodes


def configure_round0016() -> None:
    """Select the path-bound Round-0016 wrapper before child admission."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global RoundMaterializedArray, TRAIN_CONFIG, TRAIN_CONFIG_SHA256, NODES
    program = importlib.import_module("basemap.round0016_program")
    ROUND_ID = "0016"
    ROUND_LABEL = "Round 0016"
    SCHEMA_PREFIX = "round0016"
    RUNTIME_SCRIPT = "experiments/run_round0016_node.py"
    RoundMaterializedArray = program.Round0016MaterializedArray
    TRAIN_CONFIG = program.TRAIN_CONFIG
    TRAIN_CONFIG_SHA256 = program.TRAIN_CONFIG_SHA256
    NODES = program.NODES


def configure_round0017() -> None:
    """Select the path-bound Round-0017 metadata adapter before admission."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global RoundMaterializedArray, TRAIN_CONFIG, TRAIN_CONFIG_SHA256, NODES
    program = importlib.import_module("basemap.round0017_program")
    ROUND_ID = "0017"
    ROUND_LABEL = "Round 0017"
    SCHEMA_PREFIX = "round0017"
    RUNTIME_SCRIPT = "experiments/run_round0017_node.py"
    RoundMaterializedArray = program.Round0017MaterializedArray
    TRAIN_CONFIG = program.TRAIN_CONFIG
    TRAIN_CONFIG_SHA256 = program.TRAIN_CONFIG_SHA256
    NODES = program.NODES


def configure_round0018() -> None:
    """Select the bf16-autocast Round-0018 metadata adapter before admission."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global RoundMaterializedArray, TRAIN_CONFIG, TRAIN_CONFIG_SHA256, NODES
    program = importlib.import_module("basemap.round0018_program")
    ROUND_ID = "0018"
    ROUND_LABEL = "Round 0018"
    SCHEMA_PREFIX = "round0018"
    RUNTIME_SCRIPT = "experiments/run_round0018_node.py"
    RoundMaterializedArray = program.Round0018MaterializedArray
    TRAIN_CONFIG = program.TRAIN_CONFIG
    TRAIN_CONFIG_SHA256 = program.TRAIN_CONFIG_SHA256
    NODES = program.NODES


def configure_round0019() -> None:
    """Select the exact-duplicate multiplicity treatment before admission."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global RoundMaterializedArray, TRAIN_CONFIG, TRAIN_CONFIG_SHA256, NODES
    program = importlib.import_module("basemap.round0019_program")
    ROUND_ID = "0019"
    ROUND_LABEL = "Round 0019"
    SCHEMA_PREFIX = "round0019"
    RUNTIME_SCRIPT = "experiments/run_round0019_node.py"
    RoundMaterializedArray = program.Round0019MaterializedArray
    TRAIN_CONFIG = program.TRAIN_CONFIG
    TRAIN_CONFIG_SHA256 = program.TRAIN_CONFIG_SHA256
    NODES = program.NODES


def configure_round0020() -> None:
    """Select the CPU-only global duplicate-census node."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global _run_canary
    ROUND_ID = "0020"
    ROUND_LABEL = "Round 0020"
    SCHEMA_PREFIX = "round0020"
    RUNTIME_SCRIPT = "experiments/duplicate_census.py"
    _run_canary = _run_round0020_duplicate_census


def configure_round0022() -> None:
    """Select the MiniLM universality-panel nodes."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global _run_canary, _run_panel, _run_semantic_renders
    ROUND_ID = "0022"
    ROUND_LABEL = "Round 0022"
    SCHEMA_PREFIX = "round0022"
    RUNTIME_SCRIPT = "experiments/universality_panel.py"
    _run_canary = _run_round0022_canary
    _run_panel = _run_round0022_panel
    _run_semantic_renders = _run_round0022_renders


def configure_round0028() -> None:
    """Select the corrected fp32 MiniLM universality-panel nodes."""
    global ROUND_ID, ROUND_LABEL, SCHEMA_PREFIX, RUNTIME_SCRIPT
    global _run_canary, _run_panel, _run_semantic_renders
    ROUND_ID = "0028"
    ROUND_LABEL = "Round 0028"
    SCHEMA_PREFIX = "round0028"
    RUNTIME_SCRIPT = "experiments/universality_panel.py"
    _run_canary = _run_round0028_canary
    _run_panel = _run_round0028_panel
    _run_semantic_renders = _run_round0028_renders


def _run_round0020_duplicate_census(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = job["outputs"][0]
    configure_round0019()
    coordinates = StreamedCoordinateArray(
        "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates"
    )
    from basemap.duplicate_census import build_duplicate_census

    build_duplicate_census(
        pack_manifest="/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json",
        output_root=output,
        coordinates=coordinates,
        coordinate_receipt_path=(
            "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates/"
            "actual-transform.json"
        ),
    )


def _run_round0022_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    from experiments.universality_panel import run_canary

    run_canary(output_root=job["outputs"][0])


def _run_round0022_panel(active: dict[str, Any], job: dict[str, Any]) -> None:
    from experiments.universality_panel import run_panel

    canary_path = os.path.join(active["manifest"]["jobs"][0]["outputs"][0], "verdict.json")
    run_panel(canary_path=canary_path, output_root=job["outputs"][0])


def _run_round0022_renders(active: dict[str, Any], job: dict[str, Any]) -> None:
    from experiments.universality_panel import run_renders

    panel_path = os.path.join(
        active["manifest"]["jobs"][1]["outputs"][0], "universality-panel-v1.json"
    )
    run_renders(panel_path=panel_path, output_root=job["outputs"][0])


def _run_round0028_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    from experiments import universality_panel

    universality_panel.configure_round0028()
    universality_panel.run_canary(output_root=job["outputs"][0])


def _run_round0028_panel(active: dict[str, Any], job: dict[str, Any]) -> None:
    from experiments import universality_panel

    universality_panel.configure_round0028()
    canary_path = os.path.join(active["manifest"]["jobs"][0]["outputs"][0], "verdict.json")
    universality_panel.run_panel(canary_path=canary_path, output_root=job["outputs"][0])


def _run_round0028_renders(active: dict[str, Any], job: dict[str, Any]) -> None:
    from experiments import universality_panel

    universality_panel.configure_round0028()
    panel_path = os.path.join(
        active["manifest"]["jobs"][1]["outputs"][0], "universality-panel-v1.json"
    )
    universality_panel.run_renders(panel_path=panel_path, output_root=job["outputs"][0])


def _schema(name: str) -> str:
    return f"{SCHEMA_PREFIX}-{name}-v1"


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _multiplicity_treatment() -> dict[str, Any] | None:
    return TRAIN_CONFIG["execution"].get("duplicate_multiplicity")


def _node_job(active: dict[str, Any], node_id: str) -> dict[str, Any]:
    if active.get("node_id") != node_id or active.get("manifest", {}).get(
            "round_id") != ROUND_ID:
        raise RuntimeError(f"{ROUND_LABEL} executable received another capability")
    job = active["job"]
    if job.get("id") != node_id or len(job.get("outputs") or []) != 1:
        raise RuntimeError(f"{ROUND_LABEL} executable output contract changed")
    return job


def _new_exact_model():
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = TRAIN_CONFIG["model"]
    train = TRAIN_CONFIG["optimizer"]
    instance = ParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"], n_layers=model["hidden_layers"],
        n_neighbors=15, a=model["a"], b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=train["correlation_weight"],
        learning_rate=train["learning_rate"], n_epochs=1,
        batch_size=train["batch_size"], device="cuda",
        use_batchnorm=model["use_batchnorm"], use_dropout=model["use_dropout"],
        clip_grad_norm=train["clip_grad_norm"], clip_grad_value=None,
        pos_ratio=train["positive_ratio"], architecture=model["architecture"],
        correlation_distance_transform="raw", lr_schedule="cosine",
        warmup_steps=train["warmup_successful_updates"],
        total_steps_estimate=train["successful_positive_lr_updates"],
        require_full_budget=True, require_graph_manifest=True,
        required_input_pipeline="device_uniform", use_amp=train["use_amp"],
        positive_target_mode=train["positive_target_mode"],
        reject_neighbors=train["reject_neighbors"], anchored_init="none",
        anchor_hold_weight=0.0, midnear_enabled=False, mn_pairs_per_batch=0,
        weighted_edge_sampling=False, gpu_resident_data="auto",
        gpu_resident_vram_budget_gb=31.0)
    multiplicity = TRAIN_CONFIG["execution"].get("duplicate_multiplicity")
    if multiplicity:
        instance.duplicate_multiplicity_cap_path = multiplicity["artifact_path"]
        instance.duplicate_multiplicity_cap_sha256 = multiplicity["artifact_sha256"]
        instance.duplicate_multiplicity_fixed_k = multiplicity[
            "fixed_edges_per_source"
        ]
    return instance


def _fixture_scalar_equivalence(root: str) -> dict[str, Any]:
    # Exercise the imported registered scorer fixture itself: two independent
    # cache modes, persisted reports, the one-build k15 private truth, all four
    # query consumers, and every persisted scientific scalar leaf.
    from experiments.build_round0005_scorer_fixture import build_fixture
    from experiments.compare_panel_cache import run_equivalence

    fixture_path = os.path.join(root, "registered-scorer-fixture.npz")
    fixture = build_fixture(
        fixture_path, rows=128, query_rows=12, dimensions=8, seed=140014)
    report = run_equivalence(
        fixture_path=fixture_path,
        out_root=os.path.join(root, "registered-scorer-equivalence"))
    if (report.get("passed") is not True or
            report.get("checks", {}).get("persisted_scalars_identical") is not True or
            report.get("checks", {}).get("cached_real_build_once") is not True or
            report.get("checks", {}).get("maximum_k_15") is not True or
            report.get("checks", {}).get("cached_truth_shared") is not True):
        raise RuntimeError(f"{ROUND_LABEL} registered scorer scalar equivalence failed")
    return {
        "passed": True,
        "fixture": expected_input_signature(fixture_path),
        "fixture_payload_sha256": fixture["payload_sha256"],
        "equivalence": expected_input_signature(
            os.path.join(root, "registered-scorer-equivalence", "equivalence.json")),
        "report": report,
    }


def _fixture_semantic_render(root: str) -> dict[str, Any]:
    rng = np.random.RandomState(140014)
    coordinates = rng.normal(size=(128, 2)).astype("float32")
    semantic_ids = np.arange(len(coordinates), dtype=np.int64)
    permutation = rng.permutation(len(coordinates))
    permuted_ids = semantic_ids[permutation]
    permuted_coordinates = coordinates[permutation]
    sample_ids = np.sort(rng.choice(len(coordinates), 64, replace=False)).astype(np.int64)
    order = np.argsort(permuted_ids, kind="mergesort")
    positions = np.searchsorted(permuted_ids[order], sample_ids)
    gathered = permuted_coordinates[order[positions]]
    if not np.array_equal(gathered, coordinates[sample_ids]):
        raise RuntimeError(f"{ROUND_LABEL} canary semantic-ID gathering is misaligned")
    sample_path = os.path.join(root, "semantic-sample-ids.npy")
    image_path = os.path.join(root, "semantic-alignment.png")
    atomic_save_new_npy(sample_path, sample_ids, immutable=True)

    def draw(path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.scatter(gathered[:, 0], gathered[:, 1], s=5, linewidths=0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([]); axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=120)
        plt.close(figure)

    atomic_build_new_file(image_path, draw, immutable=True)
    return {
        "passed": True,
        "semantic_universe_sha256": ordered_array_sha256(semantic_ids),
        "sample_ids": expected_input_signature(sample_path),
        "gathered_rows_sha256": ordered_array_sha256(order[positions].astype(np.int64)),
        "expected_coordinate_sha256": ordered_array_sha256(coordinates[sample_ids]),
        "gathered_coordinate_sha256": ordered_array_sha256(gathered),
        "image": expected_input_signature(image_path),
    }


def _run_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = create_fresh_directory(job["outputs"][0], label=f"{ROUND_LABEL} canary output")
    import torch

    X = RoundMaterializedArray()
    pumap = _new_exact_model()
    dataset, loader, edges = pumap._prepare_edge_list_training(
        X, GRAPH_PATH, len(X), False, 42)
    src_batch, dst_batch, batch_targets = next(iter(loader))
    torch.cuda.synchronize("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    headroom_gib = free_bytes / 1024**3
    multiplicity = _multiplicity_treatment()
    expected_edges = (
        multiplicity["effective_positive_edges"] if multiplicity else 450_000_000
    )
    multiplicity_ok = (
        pumap._pipeline_info.get("multiplicity_policy")
        == ("exact_duplicate_cap_one" if multiplicity else "row_multiplicity_uncapped")
    )
    if multiplicity:
        multiplicity_ok = bool(
            multiplicity_ok
            and pumap._pipeline_info.get("multiplicity_cap_artifact_sha256")
            == multiplicity["artifact_sha256"]
            and pumap._pipeline_info.get("multiplicity_excluded_source_rows")
            == multiplicity["excluded_rows"]
            and pumap._pipeline_info.get("multiplicity_retained_rows")
            == multiplicity["retained_rows"]
            and loader.positive_source_rows_t is not None
            and len(loader.positive_source_rows_t) == multiplicity["retained_rows"]
            and loader.fixed_edges_per_source == multiplicity["fixed_edges_per_source"]
        )
    if (pumap.model is not None or hasattr(pumap, "_train_stats") or
            pumap._pipeline_info.get("pipeline") != "device_uniform" or
            pumap._pipeline_info.get("positive_sampling") != "uniform" or
            pumap._pipeline_info.get("uniform_with_replacement") is not True or
            pumap._pipeline_info.get("x_residency") != "device_fp16" or
            loader.uniform_with_replacement is not True or loader.perm is not None or
            tuple(src_batch.shape) != (8192, 384) or
            tuple(dst_batch.shape) != (8192, 384) or
            int((batch_targets == 1.0).sum().item()) != 409 or
            int((batch_targets == 0.0).sum().item()) != 7783 or
            edges != expected_edges or not multiplicity_ok or headroom_gib < 1.5):
        raise RuntimeError(f"{ROUND_LABEL} no-training setup/headroom contract failed")
    scalar = _fixture_scalar_equivalence(output)
    semantic = _fixture_semantic_render(output)
    aggregate = sum(float(item["p90_wall_s"]) * 1.15
                    for item in active["manifest"]["jobs"])
    if aggregate > 5.5 * 3600:
        raise RuntimeError(f"{ROUND_LABEL} full-queue p90+15% exceeds 5.5 hours")
    evidence = {
        "schema": _schema("canary-evidence"),
        "accepted_capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "production_config_sha256": TRAIN_CONFIG_SHA256,
        "optimizer_updates": 0,
        "model_unallocated": pumap.model is None,
        "pipeline": pumap._pipeline_info,
        "positive_edges": edges,
        "sampler_probe": {
            "uniform_with_replacement": loader.uniform_with_replacement,
            "permutation_materialized": loader.perm is not None,
            "source_shape": list(src_batch.shape),
            "target_shape": list(dst_batch.shape),
            "positive_labels": int((batch_targets == 1.0).sum().item()),
            "negative_labels": int((batch_targets == 0.0).sum().item()),
        },
        "post_setup_memory": {
            "free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
            "headroom_gib": headroom_gib, "minimum_headroom_gib": 1.5,
        },
        "scorer_scalar_equivalence": scalar,
        "semantic_render_alignment": semantic,
        "registered_p90_plus_margin_seconds": aggregate,
        "gpu_hours_cap": 5.5,
    }
    evidence_path = os.path.join(output, "evidence.json")
    atomic_write_new_json(evidence_path, _seal(evidence), immutable=True)
    verdict_body = {
        "schema": _schema("canary-verdict"), "passed": True,
        "optimizer_updates": 0, "pipeline": "device_uniform",
        "sampling": TRAIN_CONFIG["graph"]["sampling"],
        "headroom_gib": headroom_gib,
        "scorer_scalar_equivalence": True,
        "semantic_render_alignment": True,
        "registered_p90_plus_margin_seconds": aggregate,
        "evidence": expected_input_signature(evidence_path),
        "retry_count": 0,
    }
    atomic_write_new_json(
        os.path.join(output, "verdict.json"), _seal(verdict_body), immutable=True)
    del loader, dataset, pumap, X


def _publish_model(model, path: str) -> None:
    temporary = os.path.join(os.path.dirname(path), f".model.partial-{os.getpid()}")
    if os.path.lexists(temporary) or os.path.lexists(path):
        raise FileExistsError(f"{ROUND_LABEL} model output/temporary already exists")
    model.save(temporary)
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def _run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    canary = active["manifest"]["jobs"][0]["outputs"][0]
    with open(os.path.join(canary, "verdict.json"), encoding="utf-8") as handle:
        verdict = json.load(handle)
    verdict_body = {key: verdict[key] for key in verdict if key != "identity_sha256"}
    evidence = verdict.get("evidence")
    if (verdict.get("identity_sha256") != sha256_bytes(canonical_json(verdict_body)) or
            verdict.get("passed") is not True or
            verdict.get("optimizer_updates") != 0 or
            verdict.get("pipeline") != "device_uniform" or
            verdict.get("sampling") != TRAIN_CONFIG["graph"]["sampling"] or
            not isinstance(evidence, dict) or
            expected_input_signature(evidence.get("canonical_path", "")) != evidence):
        raise RuntimeError(f"{ROUND_LABEL} training lacks its passing no-update canary")
    output = create_fresh_directory(job["outputs"][0], label=f"{ROUND_LABEL} train output")
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {"schema": _schema("production-config-receipt"),
         "config": TRAIN_CONFIG, "config_sha256": TRAIN_CONFIG_SHA256},
        immutable=True)
    import torch
    np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    X = RoundMaterializedArray()
    pumap = _new_exact_model()
    pumap._max_train_steps = 500_000
    pumap._bench_warmup = 200
    pumap._perf_profile = True
    pumap._perf_floor = 45.0
    pumap._perf_warn_rate = 55.0
    pumap._perf_subfloor_patience = 2
    pumap._abort_on_first_nonfinite = True
    pumap._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    pumap.fit(
        X, low_memory=False, verbose=False, n_processes=6, random_state=42,
        resample_negatives=False, precomputed_edges_path=GRAPH_PATH,
        use_wandb=False)
    wall = time.monotonic() - started
    stats = dict(pumap._train_stats)
    # Transient AMP scaler overflows are designed-for (counted separately,
    # bounded by the trainer's consecutive/fraction guards) and offset the
    # attempted/finite-loss batch counts; genuine non-finites stay at zero.
    amp_skips = int(stats.get("amp_overflow_skips") or 0)
    requested_amp = TRAIN_CONFIG["optimizer"]["use_amp"]
    exact = {
        "amp_dtype": ("bfloat16" if requested_amp == "bf16" else
                      ("float16" if requested_amp else None)),
        "schedule_version": "cosine-v3-positive-budget",
        "lr_horizon": 500_000,
        "positive_lr_optimizer_steps": 500_000,
        "scheduler_steps": 500_000,
        "attempted_batches": 500_000 + amp_skips,
        "finite_loss_batches": 500_000 + amp_skips,
        "optimizer_steps_succeeded": 500_000,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "pipeline_pipeline": "device_uniform",
        "pipeline_positive_sampling": "uniform",
        "pipeline_uniform_with_replacement": True,
        "pipeline_x_residency": "device_fp16",
        "pipeline_weighted_requested": False,
        "pipeline_weighted_effective": False,
    }
    multiplicity = _multiplicity_treatment()
    if multiplicity:
        exact.update({
            "n_pos_edges": multiplicity["effective_positive_edges"],
            "pipeline_multiplicity_policy": "exact_duplicate_cap_one",
            "pipeline_multiplicity_cap_artifact_sha256": multiplicity[
                "artifact_sha256"
            ],
            "pipeline_multiplicity_excluded_source_rows": multiplicity[
                "excluded_rows"
            ],
            "pipeline_multiplicity_retained_rows": multiplicity["retained_rows"],
            "pipeline_multiplicity_positive_edges_effective": multiplicity[
                "effective_positive_edges"
            ],
            "pipeline_multiplicity_positive_source_sampling": (
                "uniform_retained_rows_then_fixed_k_slot_with_replacement"
            ),
            "pipeline_multiplicity_negative_sampling": (
                "uniform_retained_rows_nonself"
            ),
            "pipeline_multiplicity_positive_destinations": "original_graph_rows",
        })
    mismatches = {key: {"expected": value, "observed": stats.get(key)}
                  for key, value in exact.items() if stats.get(key) != value}
    for key in ("nonfinite_loss_skips", "nonfinite_gradient_skips"):
        if stats.get(key) != 0:
            mismatches[key] = {"expected": 0, "observed": stats.get(key)}
    attempted_steps = stats.get("optimizer_steps_attempted")
    if (not isinstance(attempted_steps, int) or
            not 500_000 <= attempted_steps <= 500_000 + amp_skips):
        mismatches["optimizer_steps_attempted"] = {
            "expected": f"500000..{500_000 + amp_skips}",
            "observed": attempted_steps}
    for key in ("lr_used_first", "lr_used_last", "first_lr", "final_lr"):
        if not isinstance(stats.get(key), (int, float)) or stats[key] <= 0:
            mismatches[key] = {
                "expected": "finite positive LR", "observed": stats.get(key)}
    if mismatches:
        raise RuntimeError(f"{ROUND_LABEL} exact train accounting failed: {mismatches}")
    model_path = os.path.join(output, "model.pt")
    _publish_model(pumap, model_path)
    profiler = pumap._canary_profiler.finalize(
        bench_seconds=pumap._bench_seconds,
        setup_seconds=getattr(pumap, "_setup_seconds", None))
    receipt = {
        "schema": _schema("train-receipt"),
        "production_config_sha256": TRAIN_CONFIG_SHA256,
        "accepted_capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "model": expected_input_signature(model_path),
        "train_stats": stats, "performance_profile": profiler,
        "train_wall_seconds": wall, "seed": 42,
        "retry_count": 0,
    }
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), _seal(receipt), immutable=True)


class StreamedCoordinateArray:
    """Lazy exact view over the thirty authenticated coordinate chunks."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        if self.root != root or not os.path.isdir(self.root) or os.path.islink(root):
            raise ValueError(f"{ROUND_LABEL} coordinate root is not canonical")
        with open(os.path.join(self.root, "actual-transform.json"), encoding="utf-8") as handle:
            self.record = json.load(handle)
        record_body = {key: self.record[key] for key in self.record
                       if key != "identity_sha256"}
        if (self.record.get("schema") != _schema("transform-capability") or
                self.record.get("identity_sha256") !=
                sha256_bytes(canonical_json(record_body)) or
                self.record.get("actual_transform", {}).get(
                    "transform_config_sha256") !=
                sha256_bytes(canonical_json(
                    self.record["actual_transform"]["transform_config"]))):
            raise ValueError(f"{ROUND_LABEL} actual transform record is not sealed")
        stream = self.record.get("stream_capability")
        capability = stream.get("capability_payload") if isinstance(stream, dict) else None
        if (not isinstance(capability, dict) or
                stream.get("capability_sha256") !=
                sha256_bytes(canonical_json(capability))):
            raise ValueError(f"{ROUND_LABEL} stream capability seal changed")
        plan = capability["plan"]
        if (capability.get("schema") != "round0013-stream-output-v1" or
                plan["output"]["shape"] != [30_000_000, 2] or
                plan["output"]["dtype"] != "<f4"):
            raise ValueError(f"{ROUND_LABEL} coordinate stream capability changed")
        ordered = capability.get("ordered_chunks")
        if not isinstance(ordered, list) or len(ordered) != 30:
            raise ValueError(f"{ROUND_LABEL} coordinate stream is not thirty chunks")
        self._members = []
        cursor = 0
        for position, item in enumerate(ordered):
            path = os.path.join(
                self.root, f"chunk-{item['chunk_index']:05d}", "coordinates.npy")
            signature = expected_input_signature(path)
            if (item.get("chunk_index") != position or
                    item.get("global_row_start") != cursor or
                    item.get("global_row_stop") != min(cursor + 1_000_000, 30_000_000) or
                    signature.get("sha256") != item.get("sha256") or
                    signature.get("bytes") != item.get("size_bytes")):
                raise ValueError(f"{ROUND_LABEL} coordinate chunk identity/order changed")
            self._members.append({**item, "path": path, "signature": signature})
            cursor = int(item["global_row_stop"])
        if cursor != 30_000_000:
            raise ValueError(f"{ROUND_LABEL} coordinate stream row coverage changed")
        self.shape = (30_000_000, 2); self.dtype = np.dtype("<f4")
        self.shard_paths = [item["path"] for item in self._members]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, columns = key
            return self[rows][..., columns]
        if isinstance(key, (int, np.integer)):
            value = self[np.array([int(key)], dtype=np.int64)]
            return value[0]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return self[np.arange(start, stop, step, dtype=np.int64)]
            indices = np.arange(start, stop, dtype=np.int64)
        else:
            indices = np.asarray(key, dtype=np.int64)
        # N-D fancy indexing (e.g. panel kNN gather Z[nb] with nb shape (nq,k))
        # mirrors numpy: gather on the flattened row ids, restore the index
        # shape with the trailing coordinate axis. Bounds are still enforced.
        if indices.ndim >= 2:
            flat = self[indices.reshape(-1)]
            return flat.reshape(indices.shape + (2,))
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError(f"{ROUND_LABEL} coordinate row selection is invalid")
        output = np.empty((len(indices), 2), dtype="<f4")
        for member in self._members:
            lo, hi = int(member["global_row_start"]), int(member["global_row_stop"])
            selected = np.flatnonzero((indices >= lo) & (indices < hi))
            if not len(selected):
                continue
            array = np.load(member["path"], mmap_mode="r", allow_pickle=False)
            if array.shape != (hi - lo, 2) or array.dtype.str != "<f4":
                raise ValueError(f"{ROUND_LABEL} coordinate chunk changed")
            output[selected] = array[indices[selected] - lo]
            del array
        return output

    def _reduce_axis0(self, ufunc, seed):
        acc = np.full(2, seed, dtype="<f4")
        for member in self._members:
            array = np.load(member["path"], mmap_mode="r", allow_pickle=False)
            acc = ufunc(acc, ufunc.reduce(np.asarray(array, dtype="<f4"), axis=0))
            del array
        return acc

    def min(self, axis=None):
        if axis not in (0, None):
            raise ValueError(f"{ROUND_LABEL} coordinate min supports axis in {{0, None}}")
        column = self._reduce_axis0(np.minimum, np.inf)
        return column if axis == 0 else column.min()

    def max(self, axis=None):
        if axis not in (0, None):
            raise ValueError(f"{ROUND_LABEL} coordinate max supports axis in {{0, None}}")
        column = self._reduce_axis0(np.maximum, -np.inf)
        return column if axis == 0 else column.max()


def _run_transform(active: dict[str, Any], job: dict[str, Any]) -> None:
    train = active["manifest"]["jobs"][1]["outputs"][0]
    model_path = os.path.join(train, "model.pt")
    template = [item["signature"]["canonical_path"]
                for item in active["manifest"]["program_inputs"]
                if item["role"] == "transform_spec_template"][0]
    from basemap.round0014_transform import (production_transform,
                                             stream_production_coordinates)
    result = stream_production_coordinates(
        model_path=model_path, template_path=template,
        release_root=active["manifest"]["repo_root"],
        release_sha=active["manifest"]["release_sha"],
        output_root=job["outputs"][0])
    queries = np.load(QUERIES_PATH, mmap_mode="r", allow_pickle=False)
    query_coordinates = production_transform(np.asarray(queries, dtype="<f4"))
    query_path = os.path.join(job["outputs"][0], "heldout-query-coordinates.npy")
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    body = {
        "schema": _schema("transform-capability"),
        **result,
        "heldout_queries": expected_input_signature(QUERIES_PATH),
        "heldout_query_provenance": expected_input_signature(QUERY_PROVENANCE_PATH),
        "heldout_query_coordinates": expected_input_signature(query_path),
    }
    atomic_write_new_json(
        os.path.join(job["outputs"][0], "actual-transform.json"),
        _seal(body), immutable=True)


def _panel_config():
    from basemap.panel_v2 import PanelV2Config
    return PanelV2Config(
        frac=0.001, k_clust=(256, 1024), k_density=15, k_hit=10,
        n_anchors=10_000, anchor_seed=123, corpus_chunk=500_000,
        overselect=8, block_elems=500_000_000,
        rerank_byte_cap=2_000_000_000, rerank_scratch=3.0,
        peak_byte_cap=26_000_000_000)


def _load_minilm_untrained_floor() -> dict[str, Any]:
    signature = expected_input_signature(MINILM_UNTRAINED_FLOOR_PATH)
    if signature["sha256"] != MINILM_UNTRAINED_FLOOR_SHA256:
        raise ValueError(f"{ROUND_LABEL} registered untrained floor changed")
    with open(MINILM_UNTRAINED_FLOOR_PATH, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    controls = receipt.get("controls")
    floor = receipt.get("untrained_projection_floor_ffr")
    architecture = receipt.get("architecture")
    expected_architecture = {
        "name": TRAIN_CONFIG["model"]["architecture"],
        "input_dimension": TRAIN_CONFIG["model"]["input_dimension"],
        "hidden_dimension": TRAIN_CONFIG["model"]["hidden_dimension"],
        "hidden_layers": TRAIN_CONFIG["model"]["hidden_layers"],
        "output_dimension": TRAIN_CONFIG["model"]["output_dimension"],
        "use_batchnorm": TRAIN_CONFIG["model"]["use_batchnorm"],
        "use_dropout": TRAIN_CONFIG["model"]["use_dropout"],
    }
    if (
        receipt.get("schema") != "round0018-minilm-untrained-projection-floor.v1"
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or receipt.get("training_performed") is not False
        or architecture != expected_architecture
        or not isinstance(controls, list)
        or len(controls) < 3
        or len({cell.get("seed") for cell in controls}) != len(controls)
        or not isinstance(floor, (int, float))
        or isinstance(floor, bool)
        or not np.isfinite(floor)
        or floor <= 0
        or floor != max(float(cell["projection_ffr"]) for cell in controls)
    ):
        raise ValueError(f"{ROUND_LABEL} registered untrained floor is invalid")
    return {"receipt": signature, "floor_ffr": float(floor), "controls": controls}


def _load_duplicate_baseline(treatment: dict[str, Any]) -> dict[str, Any]:
    path = treatment["baseline_diagnostic_path"]
    signature = expected_input_signature(path)
    if signature["sha256"] != treatment["baseline_diagnostic_sha256"]:
        raise ValueError(f"{ROUND_LABEL} duplicate-component baseline changed")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    diagnostics = receipt.get("diagnostics")
    if (
        receipt.get("schema") != "round0019-duplicate-component-baseline.v1"
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or not isinstance(diagnostics, dict)
        or diagnostics.get("method")
        != "fixed-sample-mahalanobis-excluding-selected-component"
        or diagnostics.get("maximum_representative_mahalanobis")
        != treatment["baseline_maximum_mahalanobis"]
    ):
        raise ValueError(f"{ROUND_LABEL} duplicate-component baseline is invalid")
    return {"receipt": signature, "diagnostics": diagnostics}


def _registered_panel_decision(
        panel: dict[str, Any], projection: dict[str, Any], recall50: float,
        canary_equivalence: dict[str, Any], untrained_floor_ffr: float,
) -> tuple[float, dict[str, bool]]:
    if not np.isfinite(untrained_floor_ffr) or untrained_floor_ffr <= 0:
        raise ValueError(f"{ROUND_LABEL} untrained projection floor must be positive")
    projection_ratio = projection["proj_ffr"] / untrained_floor_ffr
    guards = panel.get("guards")
    numerical_guards = bool(
        isinstance(guards, dict)
        and guards.get("coords_finite") is True
        and guards.get("coords_collapsed") is False
        and guards.get("emb_finite") is True
        and guards.get("emb_zero_rows") == 0
    )
    checks = {
        "ffr": panel["ffr"] >= 0.40,
        "density": panel["density"] >= 0.55,
        "purity_k256": panel["purity"]["k256"] >= 0.50,
        "purity_k1024": panel["purity"]["k1024"] >= 0.50,
        "projection_over_untrained_floor": projection_ratio >= 100.0,
        "recall50_gt_recall10": recall50 > panel["recall@k"],
        "numerical_guards": numerical_guards,
        "canary_cache_scalar_equivalence": canary_equivalence["passed"] is True,
    }
    return projection_ratio, checks


def _run_high_d_reference(active: dict[str, Any], job: dict[str, Any]) -> None:
    from basemap.panel_v2 import (build_hiD_reference, load_hiD_reference,
                                  sample_anchors, save_hiD_reference, _self_knn)
    output = create_fresh_directory(
        job["outputs"][0], label=f"{ROUND_LABEL} high-D reference output")
    X = RoundMaterializedArray(); cfg = _panel_config()
    centroids = {
        256: np.load(CENTROIDS_K256_PATH, mmap_mode="r", allow_pickle=False),
        1024: np.load(CENTROIDS_K1024_PATH, mmap_mode="r", allow_pickle=False),
    }
    anchors = sample_anchors(len(X), cfg)
    reference = build_hiD_reference(X, anchors, cfg, centroids)
    reference_path = os.path.join(output, "reference.npz")
    save_hiD_reference(reference, reference_path)
    reopened = load_hiD_reference(
        reference_path, expected_key=reference["key"],
        expected_key_parts=reference["key_parts"])
    hi50, _, guard50 = _self_knn(X, anchors, 50, cfg, hi_dim=True, exact=True)
    hi50_path = os.path.join(output, "recall50-truth.npy")
    atomic_save_new_npy(hi50_path, np.asarray(hi50, dtype=np.int64), immutable=True)
    bundle = {
        "schema": _schema("high-d-reference"),
        "accepted_capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "base_reference": expected_input_signature(reference_path),
        "base_identity_key": reopened["key"],
        "base_content_sha256": reopened["content_sha256"],
        "recall50_truth": expected_input_signature(hi50_path),
        "recall50_guard": guard50,
        "anchors_sha256": ordered_array_sha256(anchors),
        "registered_index": expected_input_signature(INDEX_PATH),
        "centroids": {
            "k256": expected_input_signature(CENTROIDS_K256_PATH),
            "k1024": expected_input_signature(CENTROIDS_K1024_PATH),
        },
        "single_content_addressed_reference_bundle": True,
    }
    atomic_write_new_json(
        os.path.join(output, "reference-receipt.json"), _seal(bundle), immutable=True)


def _run_panel(active: dict[str, Any], job: dict[str, Any]) -> None:
    from basemap.panel_v2 import (QueryTruthCache, _self_knn,
                                  load_hiD_reference, recall_at_k_from_neighbors,
                                  score_panel)
    from experiments.score_complete_panel import score_query_bundle
    output = create_fresh_directory(job["outputs"][0], label=f"{ROUND_LABEL} panel output")
    transform = active["manifest"]["jobs"][2]["outputs"][0]
    reference_root = active["manifest"]["jobs"][3]["outputs"][0]
    train_root = active["manifest"]["jobs"][1]["outputs"][0]
    X = RoundMaterializedArray(); Z = StreamedCoordinateArray(transform)
    cfg = _panel_config()
    centroids = {
        256: np.load(CENTROIDS_K256_PATH, mmap_mode="r", allow_pickle=False),
        1024: np.load(CENTROIDS_K1024_PATH, mmap_mode="r", allow_pickle=False),
    }
    reference = load_hiD_reference(os.path.join(reference_root, "reference.npz"))
    panel = score_panel(
        X, Z, config=cfg, centroids_by_k=centroids,
        hiD_reference=reference, scale_admission=None,
        provenance={
            "round_id": ROUND_ID, "release_sha": active["manifest"]["release_sha"],
            "accepted_capability_sha256": ACCEPTED_CAPABILITY_SHA256,
            "coordinate_capability": expected_input_signature(
                os.path.join(transform, "actual-transform.json")),
            "high_d_reference": expected_input_signature(
                os.path.join(reference_root, "reference-receipt.json")),
        })
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    hi50 = np.load(os.path.join(reference_root, "recall50-truth.npy"),
                   mmap_mode="r", allow_pickle=False)
    lo50, _, guard50 = _self_knn(Z, anchors, 50, cfg, hi_dim=False, exact=True)
    recall50 = round(recall_at_k_from_neighbors(hi50, lo50, 50), 5)
    queries = np.load(QUERIES_PATH, mmap_mode="r", allow_pickle=False)
    query_coords = np.load(
        os.path.join(transform, "heldout-query-coordinates.npy"),
        mmap_mode="r", allow_pickle=False)
    cache = QueryTruthCache(
        cache_dir=os.path.join(output, "query-truth-cache"), enabled=True)
    cache.get_or_build(
        queries, X, cfg=cfg,
        corpus_identity=reference["key_parts"]["data"],
        query_identity={
            "query": expected_input_signature(QUERIES_PATH),
            "provenance": expected_input_signature(QUERY_PROVENANCE_PATH),
        }, k=15)
    projection = score_query_bundle(
        X=X, Z=Z, Xq=queries, Zq=query_coords, cfg=cfg,
        truth_cache=cache, label=f"{SCHEMA_PREFIX}-seed42", random_seed=123)
    cache_telemetry = cache.telemetry()
    if (cache_telemetry["build_count"] != 1 or cache_telemetry["hit_count"] != 3 or
            cache_telemetry["maximum_k"] != 15):
        raise RuntimeError(f"{ROUND_LABEL} query truth cache contract changed")
    canary_root = active["manifest"]["jobs"][0]["outputs"][0]
    with open(os.path.join(canary_root, "evidence.json"), encoding="utf-8") as handle:
        canary_equivalence = json.load(handle)["scorer_scalar_equivalence"]
    untrained_floor = _load_minilm_untrained_floor()
    projection_ratio, decision_checks = _registered_panel_decision(
        panel, projection, recall50, canary_equivalence,
        untrained_floor["floor_ffr"])
    random_floor_ratio = (
        projection["proj_ffr"] / projection["proj_random_floor_ffr"]
        if projection["proj_random_floor_ffr"] > 0 else float("inf"))
    duplicate_diagnostics = None
    multiplicity = _multiplicity_treatment()
    if multiplicity:
        from basemap.duplicate_diagnostics import duplicate_component_diagnostics
        from basemap.duplicate_multiplicity import load_duplicate_cap

        cap = load_duplicate_cap(
            multiplicity["artifact_path"],
            expected_sha256=multiplicity["artifact_sha256"],
            row_count=len(Z),
            fixed_edges_per_source=multiplicity["fixed_edges_per_source"],
        )
        baseline = _load_duplicate_baseline(multiplicity)
        observed = duplicate_component_diagnostics(
            Z,
            excluded_rows=cap["excluded_rows"],
            representative_rows=cap["representative_rows"],
        )
        if (
            observed["sample_ids_sha256"]
            != baseline["diagnostics"]["sample_ids_sha256"]
        ):
            raise RuntimeError(f"{ROUND_LABEL} duplicate diagnostic sample changed")
        observed_max = observed["maximum_representative_mahalanobis"]
        baseline_max = baseline["diagnostics"]["maximum_representative_mahalanobis"]
        decision_checks["duplicate_component_offset_reduced_50pct"] = (
            observed_max <= multiplicity["required_maximum_mahalanobis"]
        )
        duplicate_diagnostics = {
            "baseline": baseline,
            "observed": observed,
            "observed_over_baseline": observed_max / baseline_max,
            "required_maximum_mahalanobis": multiplicity[
                "required_maximum_mahalanobis"
            ],
        }
    report = {
        "schema": _schema("registered-panel"),
        "production_config_sha256": TRAIN_CONFIG_SHA256,
        "accepted_capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "registered_inputs": {
            "graph": expected_input_signature(GRAPH_PATH),
            "index": expected_input_signature(INDEX_PATH),
            "centroids_k256": expected_input_signature(CENTROIDS_K256_PATH),
            "centroids_k1024": expected_input_signature(CENTROIDS_K1024_PATH),
            "queries": expected_input_signature(QUERIES_PATH),
            "query_provenance": expected_input_signature(QUERY_PROVENANCE_PATH),
        },
        "panel": panel,
        "recall@10": panel["recall@k"], "recall@50": recall50,
        "recall50_guard": guard50,
        "projection": projection,
        "registered_untrained_projection_floor": untrained_floor,
        "projection_over_untrained_floor": projection_ratio,
        "projection_over_random_bbox_floor": random_floor_ratio,
        "duplicate_component": duplicate_diagnostics,
        "query_truth_cache": cache_telemetry,
        "canary_scalar_equivalence": canary_equivalence,
        "decision_checks": decision_checks,
        "selector_passed": all(decision_checks.values()),
        "valid_threshold_miss_is_terminal_negative": True,
    }
    atomic_write_new_json(
        os.path.join(output, "panel.json"), _seal(report), immutable=True)


def _streamed_arange_identity(rows: int, block: int = 1_000_000) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": (rows,), "dtype": np.dtype("int64").str}))
    for start in range(0, rows, block):
        values = np.arange(start, min(start + block, rows), dtype=np.int64)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _run_semantic_renders(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = create_fresh_directory(
        job["outputs"][0], label=f"{ROUND_LABEL} semantic render output")
    transform = active["manifest"]["jobs"][2]["outputs"][0]
    panel_root = active["manifest"]["jobs"][4]["outputs"][0]
    Z = StreamedCoordinateArray(transform)
    rng = np.random.RandomState(20260717)
    sample_ids = np.sort(rng.choice(len(Z), 50_000, replace=False)).astype(np.int64)
    points = Z[sample_ids]
    if not np.isfinite(points).all():
        raise RuntimeError(f"{ROUND_LABEL} semantic render has non-finite points")
    sample_path = os.path.join(output, "sample-semantic-ids.npy")
    rows_path = os.path.join(output, "gathered-row-positions.npy")
    image_path = os.path.join(output, "seed42-map.png")
    atomic_save_new_npy(sample_path, sample_ids, immutable=True)
    atomic_save_new_npy(rows_path, sample_ids.copy(), immutable=True)

    def draw(path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 10))
        axis.scatter(points[:, 0], points[:, 1], s=0.15, alpha=0.35,
                     linewidths=0, rasterized=True)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{ROUND_LABEL} seed-42 30M MiniLM map")
        axis.set_xticks([]); axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    atomic_build_new_file(image_path, draw, immutable=True)
    universe = _streamed_arange_identity(len(Z))
    namespace = {
        "schema": "basemap_semantic_id_namespace.v1",
        "name": f"{SCHEMA_PREFIX}:30m-minilm-row-position",
        "kind": "row_position",
        "corpus_identity_sha256": ACCEPTED_CAPABILITY_SHA256,
        "universe_sha256": universe, "row_count": len(Z),
    }
    from basemap.round0005_staging import validate_semantic_namespace
    validate_semantic_namespace(namespace)
    diagnostics = {
        "finite_fraction": float(np.isfinite(points).all(axis=1).mean()),
        "axis_std": [float(value) for value in points.std(axis=0)],
        "axis_span": [float(value) for value in np.ptp(points, axis=0)],
        "collapsed": bool(np.any(points.std(axis=0) <= 1e-8)),
    }
    render = {
        "schema": _schema("semantic-render"),
        "semantic_namespace": namespace,
        "sample_seed": 20260717, "sample_size": 50_000,
        "sample_semantic_ids": expected_input_signature(sample_path),
        "sample_semantic_ids_sha256": ordered_array_sha256(sample_ids),
        "gathered_row_positions": expected_input_signature(rows_path),
        "gathered_rows_match_semantic_ids": True,
        "coordinate_stream": expected_input_signature(
            os.path.join(transform, "actual-transform.json")),
        "panel": expected_input_signature(os.path.join(panel_root, "panel.json")),
        "image": expected_input_signature(image_path),
        "diagnostics": diagnostics,
        "fixed_axis_policy": "one map; exact sampled union extent; no normalization",
    }
    atomic_write_new_json(
        os.path.join(output, "render-manifest.json"), _seal(render), immutable=True)


def main(argv=None) -> int:
    # This call precedes argument parsing, output creation, Torch import, CUDA
    # discovery, or any scientific read.  The process blocks until the live
    # controller releases its one exact child capability.
    from basemap.run_controller import require_round0005_child_admission
    active = require_round0005_child_admission(RUNTIME_SCRIPT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--node", choices=[item.node_id for item in NODES], required=True)
    args = parser.parse_args(argv)
    if (os.path.realpath(args.queue_manifest) !=
            os.path.realpath(os.environ["BASEMAP_ROUND0005_MANIFEST"])):
        raise RuntimeError(f"{ROUND_LABEL} argv manifest differs from child capability")
    job = _node_job(active, args.node)
    handlers = {
        "no_training_seal_canary": _run_canary,
        "train_seed42_30m": _run_train,
        "transform_30m": _run_transform,
        "high_d_reference": _run_high_d_reference,
        "registered_panel": _run_panel,
        "semantic_renders": _run_semantic_renders,
    }
    handlers[args.node](active, job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

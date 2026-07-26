"""Fresh-process nodes for the matched balanced-30M/60M evaluation."""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0036_pipeline import (
    COORDINATE_SCHEMA,
    TRANSFORM_SCHEMA,
    CoordinateStream,
    panel_config_identity,
)
from basemap.round0040_program import RepresentativeArrayView
from basemap.round0064_evaluation import (
    ROUND_ID,
    Round0064Error,
    load_substrate,
    load_train_model,
    retained_identity,
    seal,
    validate_seal,
    validate_train_bundle,
)
from experiments.run_round0036_node import (
    CENTROIDS,
    MINILM_QUERIES,
    MINILM_QUERY_PROVENANCE,
    _project_encoded_block,
    _project_float,
    _score_untrained_floor,
)


MAP_LABELS = {
    "r0061-30m-on-30m": "r0061-balanced-30m-seed42",
    "r0063-60m-on-30m": "r0063-balanced-60m-seed42-on-matched-30m",
    "r0063-60m-on-60m": "r0063-balanced-60m-seed42",
}
MATCHED_NONINFERIORITY_MARGINS = {
    "ffr": 0.02,
    "density": 0.05,
    "purity_k256": 0.05,
    "purity_k1024": 0.05,
    "projection_ffr": 0.02,
}


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0064Error(f"{path} is not a JSON object")
    return value


def _bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    return validate_train_bundle(
        label=str(job["model_label"]),
        model_path=str(job["model_path"]),
        model_sha256=str(job["model_sha256"]),
        train_receipt_path=str(job["train_receipt_path"]),
        train_receipt_sha256=str(job["train_receipt_sha256"]),
    )


def _substrate(job: Mapping[str, Any]):
    return load_substrate(
        int8_path=str(job["int8_path"]),
        int8_sha256=str(job["int8_sha256"]),
        scales_path=str(job["scales_path"]),
        scales_sha256=str(job["scales_sha256"]),
        eligibility_path=str(job["eligibility_path"]),
        eligibility_sha256=str(job["eligibility_sha256"]),
        row_count=int(job["row_count"]),
    )


def _panel_config():
    from basemap.panel_v2 import PanelV2Config

    return PanelV2Config(**{
        key: tuple(value) if key == "k_clust" else value
        for key, value in panel_config_identity().items()
        if key != "formula_version"
    })


def run_transform(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0064 {job['map_key']} coordinates",
    )
    started = time.monotonic()
    bundle = _bundle(job)
    encoded, selector, _, eligibility = _substrate(job)
    model = load_train_model(bundle, device="cuda")
    chunk_rows = int(job.get("coordinate_chunk_rows", 5_000_000))
    batch_rows = int(job.get("model_batch_rows", 65_536))
    if chunk_rows <= 0 or batch_rows != 65_536:
        raise Round0064Error("R0064 transform geometry changed")
    members: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(encoded), chunk_rows)):
        stop = min(start + chunk_rows, len(encoded))
        root = create_fresh_directory(
            os.path.join(output, f"chunk-{index:05d}"),
            label="R0064 coordinate chunk",
        )
        path = os.path.join(root, "coordinates.npy")
        coordinates = _project_encoded_block(
            model,
            encoded,
            start,
            stop,
            batch_rows=batch_rows,
        )
        atomic_save_new_npy(path, coordinates, immutable=True)
        signature = expected_input_signature(path)
        members.append({
            "chunk_index": index,
            "global_row_start": start,
            "global_row_stop": stop,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
            "path": path,
        })
        del coordinates
    queries = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)
    if queries.shape != (10_002, 384) or queries.dtype.str != "<f4":
        raise Round0064Error("held-out MiniLM query artifact changed")
    query_coordinates = _project_float(model, queries)
    query_path = os.path.join(output, "heldout-query-coordinates.npy")
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    body = {
        "schema": TRANSFORM_SCHEMA,
        "round_id": ROUND_ID,
        "map_key": job["map_key"],
        "map_label": MAP_LABELS[str(job["map_key"])],
        "model": bundle["model"],
        "train_receipt": bundle["train_receipt"],
        "production_config_sha256": bundle["production_config_sha256"],
        "input": {
            "int8": encoded.signatures["int8"],
            "scales": encoded.signatures["scales"],
            "dequantization": "fp32(int8) * fp32(exact stored fp16 row scale)",
        },
        "inference": {
            "batch_rows": batch_rows,
            "short_tail_policy": "zero-pad-to-fixed-batch-then-discard-padding",
            "all_real_rows_projected": True,
        },
        "eligibility": eligibility["signature"],
        "row_accounting": {
            "all_rows": len(encoded),
            "retained_representatives": selector.retained_count,
            "excluded_rows": int(len(selector.excluded_rows)),
        },
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": len(encoded),
            "dimension": 2,
            "dtype": "<f4",
            "row_order": str(job["row_order"]),
            "ordered_chunks": [
                {
                    key: value
                    for key, value in member.items()
                    if key != "path"
                }
                for member in members
            ],
        },
        "heldout_queries": expected_input_signature(MINILM_QUERIES),
        "heldout_query_provenance": expected_input_signature(
            MINILM_QUERY_PROVENANCE
        ),
        "heldout_query_coordinates": expected_input_signature(query_path),
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_high_d_reference(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        _self_knn,
        build_hiD_reference,
        load_hiD_reference,
        sample_anchors,
        save_hiD_reference,
    )

    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0064 {job['substrate_label']} high-D reference",
    )
    started = time.monotonic()
    encoded, selector, retained, eligibility = _substrate(job)
    config = _panel_config()
    centroids = {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in CENTROIDS.items()
    }
    anchors = sample_anchors(len(retained), config).astype(np.int64)
    identity = retained_identity(
        encoded,
        selector,
        eligibility,
        label=str(job["substrate_label"]),
    )
    reference = build_hiD_reference(
        retained,
        anchors,
        config,
        centroids,
        **identity,
    )
    reference_path = os.path.join(output, "reference.npz")
    save_hiD_reference(reference, reference_path)
    reopened = load_hiD_reference(
        reference_path,
        expected_key=reference["key"],
        expected_key_parts=reference["key_parts"],
    )
    hi50, _, guard50 = _self_knn(
        retained,
        anchors,
        50,
        config,
        hi_dim=True,
        exact=True,
    )
    hi50_path = os.path.join(output, "recall50-truth.npy")
    atomic_save_new_npy(
        hi50_path,
        hi50.astype(np.int64),
        immutable=True,
    )
    anchor_global = selector.compact_to_global(anchors)
    anchors_path = os.path.join(output, "anchor-substrate-rows.npy")
    atomic_save_new_npy(anchors_path, anchor_global, immutable=True)
    body = {
        "schema": "round0064-high-d-reference-v1",
        "round_id": ROUND_ID,
        "substrate_label": job["substrate_label"],
        "eligibility": eligibility["signature"],
        "selector": selector.identity(),
        "input": encoded.scientific_identity(),
        "reference": expected_input_signature(reference_path),
        "reference_key": reopened["key"],
        "reference_content_sha256": reopened["content_sha256"],
        "reference_identity": identity,
        "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
        "anchor_substrate_rows": expected_input_signature(anchors_path),
        "anchor_substrate_rows_sha256": ordered_array_sha256(anchor_global),
        "recall50_truth": expected_input_signature(hi50_path),
        "recall50_guard": guard50,
        "centroids": {
            f"k{key}": expected_input_signature(path)
            for key, path in CENTROIDS.items()
        },
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "reference-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_panel(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        QueryTruthCache,
        _self_knn,
        load_hiD_reference,
        recall_at_k_from_neighbors,
        score_panel,
    )
    from experiments.score_complete_panel import score_query_bundle

    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0064 {job['map_key']} panel",
    )
    started = time.monotonic()
    bundle = _bundle(job)
    encoded, selector, retained, eligibility = _substrate(job)
    full_coordinates = CoordinateStream(str(job["transform_output"]))
    coordinates = RepresentativeArrayView(full_coordinates, selector)
    config = _panel_config()
    centroids = {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in CENTROIDS.items()
    }
    reference_path = os.path.join(
        str(job["reference_output"]),
        "reference.npz",
    )
    reference = load_hiD_reference(reference_path)
    identity = retained_identity(
        encoded,
        selector,
        eligibility,
        label=str(job["substrate_label"]),
    )
    transform_signature = expected_input_signature(
        os.path.join(
            str(job["transform_output"]),
            "actual-transform.json",
        )
    )
    panel = score_panel(
        retained,
        coordinates,
        config=config,
        centroids_by_k=centroids,
        hiD_reference=reference,
        reference_identity=identity,
        provenance={
            "round_id": ROUND_ID,
            "map_key": job["map_key"],
            "map_label": MAP_LABELS[str(job["map_key"])],
            "model": bundle["model"],
            "train_receipt": bundle["train_receipt"],
            "coordinate_capability": transform_signature,
            "eligibility": eligibility["signature"],
            "excluded_rows_entered_scientific_universe": False,
        },
    )
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    hi50 = np.load(
        os.path.join(str(job["reference_output"]), "recall50-truth.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    lo50, _, guard50 = _self_knn(
        coordinates,
        anchors,
        50,
        config,
        hi_dim=False,
        exact=True,
    )
    recall50 = round(
        recall_at_k_from_neighbors(hi50, lo50, 50),
        5,
    )
    queries = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)
    query_coordinates = np.load(
        os.path.join(
            str(job["transform_output"]),
            "heldout-query-coordinates.npy",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    cache = QueryTruthCache(
        cache_dir=os.path.join(output, "query-truth-cache"),
        enabled=True,
    )
    truth = cache.get_or_build(
        queries,
        retained,
        cfg=config,
        corpus_identity=reference["key_parts"]["data"],
        query_identity={
            "query": expected_input_signature(MINILM_QUERIES),
            "provenance": expected_input_signature(
                MINILM_QUERY_PROVENANCE
            ),
            "excluded_from_training_blocks": True,
        },
        k=15,
    )
    projection = score_query_bundle(
        X=retained,
        Z=coordinates,
        Xq=queries,
        Zq=query_coordinates,
        cfg=config,
        truth_cache=cache,
        label=MAP_LABELS[str(job["map_key"])],
        random_seed=123,
    )
    untrained = _score_untrained_floor(
        bundle=bundle,
        queries=queries,
        coordinates=coordinates,
        truth=truth["neighbors"],
        config=config,
    )
    guards = panel.get("guards") or {}
    purity = panel.get("purity") or {}
    checks = {
        "ffr_at_least_0_40": panel.get("ffr", -math.inf) >= 0.40,
        "density_at_least_0_60": (
            panel.get("density", -math.inf) >= 0.60
        ),
        "purity_k256_at_least_0_50": (
            purity.get("k256", -math.inf) >= 0.50
        ),
        "purity_k1024_at_least_0_50": (
            purity.get("k1024", -math.inf) >= 0.50
        ),
        "heldout_projection_beats_untrained_floor": (
            projection["proj_ffr"] > untrained["floor_ffr"]
        ),
        "recall_at_50_exceeds_recall_at_10": (
            recall50 > panel["recall@k"]
        ),
        "coords_finite": guards.get("coords_finite") is True,
        "coords_not_collapsed": guards.get("coords_collapsed") is False,
        "embeddings_finite": guards.get("emb_finite") is True,
        "eligible_embeddings_nonzero": guards.get("emb_zero_rows") == 0,
    }
    body = {
        "schema": "round0064-registered-panel-v1",
        "round_id": ROUND_ID,
        "map_key": job["map_key"],
        "map": {
            "label": MAP_LABELS[str(job["map_key"])],
            "model": bundle["model"],
            "coordinate_receipt": transform_signature,
        },
        "eligibility": eligibility["signature"],
        "scientific_universe": {
            "rows": len(retained),
            "substrate": job["substrate_label"],
            "row_namespace": (
                f"compact ascending {job['substrate_label']} retained rows"
            ),
            "excluded_rows_in_scoring": False,
        },
        "panel": panel,
        "recall_at_10": panel["recall@k"],
        "recall_at_50": recall50,
        "recall50_guard": guard50,
        "projection": projection,
        "untrained_projection_floor": untrained,
        "query_truth_cache": cache.telemetry(),
        "decision_checks": checks,
        "absolute_selector_passed": all(checks.values()),
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "panel.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _metrics(panel: Mapping[str, Any]) -> dict[str, float]:
    scientific = panel["panel"]
    purity = scientific["purity"]
    return {
        "ffr": float(scientific["ffr"]),
        "density": float(scientific["density"]),
        "purity_k256": float(purity["k256"]),
        "purity_k1024": float(purity["k1024"]),
        "projection_ffr": float(panel["projection"]["proj_ffr"]),
        "recall_at_10": float(panel["recall_at_10"]),
        "recall_at_50": float(panel["recall_at_50"]),
    }


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0064 scale comparison",
    )
    control_path = os.path.join(
        str(job["matched_control_panel"]),
        "panel.json",
    )
    scaled_matched_path = os.path.join(
        str(job["scaled_matched_panel"]),
        "panel.json",
    )
    scaled_full_path = os.path.join(
        str(job["scaled_full_panel"]),
        "panel.json",
    )
    control = _read_json(control_path)
    scaled_matched = _read_json(scaled_matched_path)
    scaled_full = _read_json(scaled_full_path)
    for label, value, expected_key in (
        ("control", control, "r0061-30m-on-30m"),
        ("scaled matched", scaled_matched, "r0063-60m-on-30m"),
        ("scaled full", scaled_full, "r0063-60m-on-60m"),
    ):
        validate_seal(value, label=f"R0064 {label} panel")
        if (
            value.get("schema") != "round0064-registered-panel-v1"
            or value.get("map_key") != expected_key
        ):
            raise Round0064Error(f"{label} panel identity changed")
    control_scientific = control["panel"]
    scaled_scientific = scaled_matched["panel"]
    if (
        control.get("eligibility") != scaled_matched.get("eligibility")
        or control.get("scientific_universe")
        != scaled_matched.get("scientific_universe")
        or control_scientific.get("n") != scaled_scientific.get("n")
        or control_scientific.get("anchor_hash")
        != scaled_scientific.get("anchor_hash")
        or control_scientific.get("provenance", {}).get(
            "hiD_reference_key"
        )
        != scaled_scientific.get("provenance", {}).get(
            "hiD_reference_key"
        )
    ):
        raise Round0064Error(
            "matched panels do not share one exact row universe/reference"
        )
    baseline = _metrics(control)
    treatment = _metrics(scaled_matched)
    comparisons: dict[str, Any] = {}
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        delta = treatment[metric] - baseline[metric]
        comparisons[metric] = {
            "control": baseline[metric],
            "scaled_model_on_same_30m_rows": treatment[metric],
            "delta": round(delta, 6),
            "noninferiority_margin": margin,
            "passed": delta >= -margin,
        }
    full_absolute = bool(scaled_full.get("absolute_selector_passed"))
    matched_noninferior = all(
        item["passed"] for item in comparisons.values()
    )
    body = {
        "schema": "round0064-scale-geometry-comparison-v1",
        "round_id": ROUND_ID,
        "panels": {
            "matched_30m_control": expected_input_signature(control_path),
            "scaled_model_on_matched_30m": expected_input_signature(
                scaled_matched_path
            ),
            "scaled_model_on_full_60m": expected_input_signature(
                scaled_full_path
            ),
        },
        "same_row_comparison": {
            "universe": (
                "exact R0053 balanced-30M retained representatives, "
                "identical high-D reference and anchors"
            ),
            "metrics": comparisons,
            "passed": matched_noninferior,
        },
        "full_60m_metrics": _metrics(scaled_full),
        "full_60m_absolute_selector_passed": full_absolute,
        "decision": {
            "advance_to_120m_scale_rung": (
                full_absolute and matched_noninferior
            ),
            "bisect_at_45m_if_false": not (
                full_absolute and matched_noninferior
            ),
            "reason": (
                "advance requires both a valid full-60M map and no material "
                "regression when both models are scored on the same 30M rows"
            ),
        },
        "ood_is_reported_separately_and_non_gating": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "scale-comparison.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_ood(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0064 {job['map_key']} OOD panels",
    )
    bundle = _bundle(job)
    coordinate_path = os.path.join(
        str(job["transform_output"]),
        "actual-transform.json",
    )
    coordinate = expected_input_signature(coordinate_path)

    def loader() -> Any:
        return load_train_model(bundle, device="cuda")

    common = {
        "round_id": ROUND_ID,
        "map_label": MAP_LABELS[str(job["map_key"])],
        "model_path": bundle["model"]["canonical_path"],
        "model_sha256": bundle["model"]["sha256"],
        "model_loader": loader,
    }
    from experiments import universality_panel

    universality_panel.configure_map(
        **common,
        coordinates_root=str(job["transform_output"]),
        coordinate_receipt_sha256=coordinate["sha256"],
    )
    ucanary = universality_panel.run_canary(
        output_root=os.path.join(output, "universality", "canary")
    )
    upanel = universality_panel.run_panel(
        canary_path=ucanary["verdict"]["canonical_path"],
        output_root=os.path.join(output, "universality", "panel"),
    )

    from experiments import common_corpus_ood_round0035 as common_corpus

    common_corpus.configure_map(
        **common,
        coordinate_receipt=coordinate_path,
        coordinate_receipt_sha256=coordinate["sha256"],
    )
    ccanary = common_corpus.run_canary(
        output_root=os.path.join(output, "common-corpus", "canary")
    )
    cpanel = common_corpus.run_panel(
        canary_path=ccanary["verdict"]["canonical_path"],
        output_root=os.path.join(output, "common-corpus", "panel"),
    )
    body = {
        "schema": "round0064-ood-bundle-v1",
        "round_id": ROUND_ID,
        "map_key": job["map_key"],
        "map_label": MAP_LABELS[str(job["map_key"])],
        "universality_canary": ucanary["verdict"],
        "universality_panel": upanel["panel"],
        "common_corpus_canary": ccanary["verdict"],
        "common_corpus_panel": cpanel["panel"],
        "probe_names": [
            "dadabase",
            "trec-covid",
            "code",
            "science",
            "latin",
        ],
        "retention_is_non_gating_map_card_evidence": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "ood-bundle.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_renders(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0064 fixed scale renders",
    )
    encoded, selector, _, eligibility = _substrate(job)
    del encoded
    rng = np.random.RandomState(20260726)
    compact = np.sort(
        rng.choice(selector.retained_count, 50_000, replace=False)
    ).astype(np.int64)
    substrate_rows = selector.compact_to_global(compact)
    ids_path = os.path.join(output, "matched-30m-sample-rows.npy")
    atomic_save_new_npy(ids_path, substrate_rows, immutable=True)
    renders: dict[str, Any] = {}
    for key, root in (
        ("r0061-30m-on-30m", job["control_transform"]),
        ("r0063-60m-on-30m", job["scaled_matched_transform"]),
    ):
        coordinates = CoordinateStream(str(root))
        points = coordinates[substrate_rows]
        if (
            not np.isfinite(points).all()
            or np.any(np.std(points, axis=0) <= 1e-8)
        ):
            raise Round0064Error(f"{key} render coordinates collapsed")
        image_path = os.path.join(output, f"{key}.png")

        def draw(path: str, *, values=points, title=MAP_LABELS[key]) -> None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(10, 10))
            axis.scatter(
                values[:, 0],
                values[:, 1],
                s=0.15,
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.tight_layout()
            figure.savefig(
                path,
                format="png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)

        atomic_build_new_file(image_path, draw, immutable=True)
        renders[key] = {
            "image": expected_input_signature(image_path),
            "axis_std": points.std(axis=0).astype(float).tolist(),
            "axis_span": np.ptp(points, axis=0).astype(float).tolist(),
            "sample_rows": expected_input_signature(ids_path),
            "sample_rows_sha256": ordered_array_sha256(substrate_rows),
            "sample_universe": "balanced-30m retained representatives",
        }
    _, full_selector, _, full_eligibility = load_substrate(
        int8_path=str(job["full_int8_path"]),
        int8_sha256=str(job["full_int8_sha256"]),
        scales_path=str(job["full_scales_path"]),
        scales_sha256=str(job["full_scales_sha256"]),
        eligibility_path=str(job["full_eligibility_path"]),
        eligibility_sha256=str(job["full_eligibility_sha256"]),
        row_count=60_000_000,
    )
    full_rng = np.random.RandomState(20260726)
    full_compact = np.sort(
        full_rng.choice(
            full_selector.retained_count,
            50_000,
            replace=False,
        )
    ).astype(np.int64)
    full_rows = full_selector.compact_to_global(full_compact)
    full_ids_path = os.path.join(output, "full-60m-sample-rows.npy")
    atomic_save_new_npy(full_ids_path, full_rows, immutable=True)
    full_coordinates = CoordinateStream(str(job["scaled_full_transform"]))
    full_points = full_coordinates[full_rows]
    if (
        not np.isfinite(full_points).all()
        or np.any(np.std(full_points, axis=0) <= 1e-8)
    ):
        raise Round0064Error("full 60M render coordinates collapsed")
    full_image_path = os.path.join(output, "r0063-60m-on-60m.png")

    def draw_full(path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 10))
        axis.scatter(
            full_points[:, 0],
            full_points[:, 1],
            s=0.15,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(MAP_LABELS["r0063-60m-on-60m"])
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(
            path,
            format="png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    atomic_build_new_file(full_image_path, draw_full, immutable=True)
    renders["r0063-60m-on-60m"] = {
        "image": expected_input_signature(full_image_path),
        "axis_std": full_points.std(axis=0).astype(float).tolist(),
        "axis_span": np.ptp(full_points, axis=0).astype(float).tolist(),
        "sample_rows": expected_input_signature(full_ids_path),
        "sample_rows_sha256": ordered_array_sha256(full_rows),
        "sample_universe": "balanced-60m retained representatives",
        "eligibility": full_eligibility["signature"],
    }
    body = {
        "schema": "round0064-matched-render-v1",
        "round_id": ROUND_ID,
        "eligibility": eligibility["signature"],
        "sample_seed": 20260726,
        "sample_size": 50_000,
        "sample_substrate_rows": expected_input_signature(ids_path),
        "sample_substrate_rows_sha256": ordered_array_sha256(
            substrate_rows
        ),
        "identical_semantic_rows_across_maps": True,
        "renders": renders,
    }
    receipt = seal(body)
    path = os.path.join(output, "render-manifest.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_registry(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0064 map registry receipt",
    )
    from experiments import map_registry

    registry = map_registry.scan()
    entries = [
        item
        for item in registry["maps"]
        if item.get("round_id") == ROUND_ID
    ]
    projection_entries = [
        item for item in entries if item.get("kind") == "projection-map"
    ]
    map_entries = [
        item for item in entries if item.get("kind") == "round-map"
    ]
    required_maps = {
        MAP_LABELS["r0061-30m-on-30m"],
        MAP_LABELS["r0063-60m-on-60m"],
    }
    observed_maps = {
        item.get("map_label") for item in map_entries
    }
    required_projection_pairs = {
        (label, probe)
        for label in required_maps
        for probe in ("dadabase", "trec-covid", "code", "science", "latin")
    }
    observed_projection_pairs = {
        (
            item.get("base_map"),
            item.get("projection", {}).get("probe"),
        )
        for item in projection_entries
    }
    if (
        not required_maps.issubset(observed_maps)
        or not required_projection_pairs.issubset(
            observed_projection_pairs
        )
    ):
        raise Round0064Error(
            "registry did not discover both base maps and ten projections"
        )
    map_registry.REGISTRY_PATH.write_text(json.dumps(registry, indent=1))
    map_registry.publish(registry)
    body = {
        "schema": "round0064-map-registry-publication-v1",
        "round_id": ROUND_ID,
        "registry": expected_input_signature(
            str(map_registry.REGISTRY_PATH)
        ),
        "map_ids": sorted(item["map_id"] for item in entries),
        "base_maps": sorted(required_maps),
        "projection_pairs": sorted(
            [list(value) for value in required_projection_pairs]
        ),
        "local_site_url": map_registry.SITE_URL,
    }
    receipt = seal(body)
    path = os.path.join(output, "registry-publication.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0064 handler received another queue")
    if job is None:
        raise RuntimeError("R0064 handler requires the exact job")
    handlers = {
        "transform": run_transform,
        "high_d_reference": run_high_d_reference,
        "panel": run_panel,
        "comparison": run_comparison,
        "ood": run_ood,
        "renders": run_renders,
        "registry": run_registry,
    }
    try:
        handler = handlers[str(job["action"])]
    except KeyError as exc:
        raise RuntimeError(
            f"unknown R0064 action {job.get('action')!r}"
        ) from exc
    return handler(active, job)

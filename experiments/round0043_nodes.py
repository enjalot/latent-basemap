"""Fresh-process handlers for the R0043 nested scale-geometry diagnostic."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import (
    PanelV2Config,
    _self_knn,
    estimate_panel_peak_bytes,
    ffr_from_neighbors,
    process_cuda_peak,
    recall_at_k_from_neighbors,
    reset_process_cuda_peak,
    sample_anchors,
)
from basemap.round0036_pipeline import (
    CoordinateStream,
    EncodedInt8Array,
    load_released_selector,
    validate_seal,
)
from basemap.round0043_program import (
    CORE_WIDTH,
    COORDINATE_RECEIPT_SHA256,
    COORDINATE_ROOT,
    ELIGIBILITY_SHA256,
    ELIGIBILITY_PATH,
    PANEL_CONFIG,
    R0036_PANEL,
    R0036_PANEL_SHA256,
    ROUND_ID,
    RUNG_WIDTHS,
    BalancedRungSelector,
    BalancedRungView,
    Round0043Error,
    rung_label,
)


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {
        **payload,
        "identity_sha256": sha256_bytes(canonical_json(payload)),
    }


def _panel_config() -> PanelV2Config:
    return PanelV2Config(**PANEL_CONFIG)


def _short_anchor_hash(compact_rows: np.ndarray) -> str:
    return hashlib.sha1(
        np.ascontiguousarray(
            np.asarray(compact_rows, dtype=np.int64)
        ).tobytes()
    ).hexdigest()[:12]


def _rung_guards(
    *,
    core_anchors_are_members: bool,
    rung_anchors_are_members: bool,
    coordinates_and_embeddings_row_aligned: bool,
    training_performed: bool,
) -> dict[str, bool]:
    """Return affirmative predicates only, so all(values) is meaningful."""
    return {
        "core_anchors_are_members": core_anchors_are_members,
        "rung_anchors_are_members": rung_anchors_are_members,
        "coordinates_and_embeddings_row_aligned": (
            coordinates_and_embeddings_row_aligned
        ),
        "no_training_performed": not training_performed,
    }


def _score_anchor_view(
    *,
    embeddings: BalancedRungView,
    coordinates: BalancedRungView,
    anchors: np.ndarray,
    config: PanelV2Config,
) -> dict[str, Any]:
    """Score only the metrics that identify candidate-universe geometry."""
    anchors = np.ascontiguousarray(
        np.asarray(anchors, dtype=np.int64)
    )
    if (
        anchors.ndim != 1
        or len(anchors) != config.n_anchors
        or np.any(anchors < 0)
        or np.any(anchors >= len(embeddings))
        or not np.array_equal(anchors, np.unique(anchors))
    ):
        raise Round0043Error("R0043 anchor view is malformed")
    k_fraction = max(
        config.k_hit, int(math.ceil(config.frac * len(embeddings)))
    )
    peak = estimate_panel_peak_bytes(
        config, n_dims=embeddings.shape[1], k_frac=k_fraction
    )
    if peak["dominant_bytes"] > config.peak_byte_cap:
        raise Round0043Error("R0043 scorer peak exceeds registered cap")

    # Preserve the registered panel's separate shortlist widths. Deriving the
    # top-10 truth from a k15 shortlist would subtly widen the approximate
    # candidate pool before exact reranking and could fail to reproduce R0036.
    high_hit, _, high_hit_guard = _self_knn(
        embeddings,
        anchors,
        config.k_hit,
        config,
        hi_dim=True,
        exact=True,
    )
    _, high_distances, high_density_guard = _self_knn(
        embeddings,
        anchors,
        config.k_density,
        config,
        hi_dim=True,
        want_dist=True,
        exact=True,
    )
    low_fraction, _, _ = _self_knn(
        coordinates,
        anchors,
        k_fraction,
        config,
        hi_dim=False,
    )
    _, low_distances, _ = _self_knn(
        coordinates,
        anchors,
        config.k_density,
        config,
        hi_dim=False,
        want_dist=True,
    )
    if high_distances is None or low_distances is None:
        raise Round0043Error("R0043 exact density radii are absent")

    ffr = ffr_from_neighbors(
        high_hit, low_fraction, config.k_hit
    )
    recall = recall_at_k_from_neighbors(
        high_hit, low_fraction[:, : config.k_hit], config.k_hit
    )
    high_radius = high_distances.mean(axis=1)
    low_radius = low_distances.mean(axis=1)
    density = float(np.corrcoef(
        np.log(high_radius + 1e-12),
        np.log(low_radius + 1e-12),
    )[0, 1])
    if not all(math.isfinite(value) for value in (ffr, recall, density)):
        raise Round0043Error("R0043 produced a non-finite metric")
    result = {
        "n": len(embeddings),
        "n_anchors": len(anchors),
        "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
        "anchor_hash": _short_anchor_hash(anchors),
        "frac": config.frac,
        "k_fraction": k_fraction,
        "k_hit": config.k_hit,
        "k_density": config.k_density,
        "ffr": round(ffr, 4),
        "recall_at_10": round(recall, 5),
        "density": round(density, 4),
        "high_d_hit_guard": high_hit_guard,
        "high_d_density_guard": high_density_guard,
        "peak_byte_model": peak,
    }
    del (
        high_distances,
        high_hit,
        low_fraction,
        low_distances,
        high_radius,
        low_radius,
    )
    return result


def _load_r0036_panel() -> dict[str, Any]:
    signature = expected_input_signature(R0036_PANEL)
    if signature["sha256"] != R0036_PANEL_SHA256:
        raise Round0043Error("accepted R0036 panel bytes changed")
    with open(R0036_PANEL, encoding="utf-8") as handle:
        panel = json.load(handle)
    validate_seal(panel, label="R0036 panel")
    if (
        panel.get("schema") != "round0036-registered-panel-v1"
        or panel.get("round_id") != "0036"
        or panel.get("scientific_universe", {}).get("rows")
        != 147_221_757
    ):
        raise Round0043Error("accepted R0036 panel contract changed")
    return panel


def _reproduce_r0036(
    *,
    per_corpus_rows: int,
    rung_metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    if per_corpus_rows != 50_000_000:
        return None
    baseline = _load_r0036_panel()["panel"]
    observed = {
        "anchor_hash": rung_metrics["anchor_hash"],
        "ffr": rung_metrics["ffr"],
        "recall@k": rung_metrics["recall_at_10"],
        "density": rung_metrics["density"],
    }
    expected = {
        "anchor_hash": baseline["anchor_hash"],
        "ffr": baseline["ffr"],
        "recall@k": baseline["recall@k"],
        "density": baseline["density"],
    }
    checks = {
        key: observed[key] == expected[key] for key in expected
    }
    if not all(checks.values()):
        raise Round0043Error(
            "R0043 full-rung scorer does not reproduce accepted R0036"
        )
    return {
        "baseline_panel": expected_input_signature(R0036_PANEL),
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "passed": True,
    }


def run_score_rung(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    per_corpus_rows = int(job["per_corpus_rows"])
    if per_corpus_rows not in RUNG_WIDTHS:
        raise Round0043Error("queue requested an unregistered R0043 rung")
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"Round 0043 {rung_label(per_corpus_rows)} output",
    )
    started = time.monotonic()
    reset_process_cuda_peak()

    released, eligibility = load_released_selector(
        ELIGIBILITY_PATH,
        eligibility_sha256=ELIGIBILITY_SHA256,
    )
    rung_selector = BalancedRungSelector(
        released.excluded_rows,
        per_corpus_rows=per_corpus_rows,
    )
    core_selector = (
        rung_selector
        if per_corpus_rows == CORE_WIDTH
        else BalancedRungSelector(
            released.excluded_rows,
            per_corpus_rows=CORE_WIDTH,
        )
    )
    embeddings_full = EncodedInt8Array.from_files()
    coordinates_full = CoordinateStream(
        COORDINATE_ROOT,
        expected_receipt_sha256=COORDINATE_RECEIPT_SHA256,
    )
    embeddings = BalancedRungView(embeddings_full, rung_selector)
    coordinates = BalancedRungView(coordinates_full, rung_selector)
    config = _panel_config()

    core_compact_30m = sample_anchors(
        len(core_selector), config
    ).astype(np.int64)
    core_global = core_selector.compact_to_global(core_compact_30m)
    core_compact = rung_selector.global_to_compact(core_global)
    rung_compact = sample_anchors(
        len(rung_selector), config
    ).astype(np.int64)
    rung_global = rung_selector.compact_to_global(rung_compact)
    if not np.all(rung_selector.is_member(core_global)):
        raise Round0043Error("fixed core anchors escaped a larger rung")

    core_path = os.path.join(output, "core-anchor-global-rows.npy")
    rung_path = os.path.join(output, "rung-anchor-global-rows.npy")
    atomic_save_new_npy(core_path, core_global, immutable=True)
    atomic_save_new_npy(rung_path, rung_global, immutable=True)

    core_metrics = _score_anchor_view(
        embeddings=embeddings,
        coordinates=coordinates,
        anchors=core_compact,
        config=config,
    )
    if np.array_equal(core_compact, rung_compact):
        rung_metrics = dict(core_metrics)
        rung_reused_core = True
    else:
        rung_metrics = _score_anchor_view(
            embeddings=embeddings,
            coordinates=coordinates,
            anchors=rung_compact,
            config=config,
        )
        rung_reused_core = False
    reproduction = _reproduce_r0036(
        per_corpus_rows=per_corpus_rows,
        rung_metrics=rung_metrics,
    )
    selector_identity = rung_selector.identity()
    cuda_peak = process_cuda_peak()
    body = {
        "schema": "round0043-nested-rung-score-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "rung": {
            "label": rung_label(per_corpus_rows),
            "per_corpus_rows": per_corpus_rows,
            "selector": selector_identity,
        },
        "eligibility": eligibility["signature"],
        "inputs": {
            "embeddings": embeddings_full.scientific_identity(),
            "coordinates": coordinates_full.scientific_identity(),
            "coordinate_receipt": expected_input_signature(
                os.path.join(COORDINATE_ROOT, "actual-transform.json")
            ),
        },
        "anchor_views": {
            "fixed_30m_core": {
                "global_rows": expected_input_signature(core_path),
                "global_rows_sha256": ordered_array_sha256(core_global),
                "metrics": core_metrics,
            },
            "rung_wide": {
                "global_rows": expected_input_signature(rung_path),
                "global_rows_sha256": ordered_array_sha256(rung_global),
                "metrics": rung_metrics,
                "reused_fixed_core_computation": rung_reused_core,
            },
        },
        "r0036_full_rung_reproduction": reproduction,
        "guards": _rung_guards(
            core_anchors_are_members=True,
            rung_anchors_are_members=bool(
                np.all(rung_selector.is_member(rung_global))
            ),
            coordinates_and_embeddings_row_aligned=(
                len(embeddings) == len(coordinates)
            ),
            training_performed=False,
        ),
        "gpu_peak": cuda_peak,
        "wall_seconds": time.monotonic() - started,
        "training_performed": False,
    }
    if not all(body["guards"].values()):
        raise Round0043Error("R0043 rung guard failed")
    receipt = _seal(body)
    path = os.path.join(
        output, f"rung-{rung_label(per_corpus_rows)}.json"
    )
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _read_rung(path: str) -> dict[str, Any]:
    files = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.startswith("rung-") and name.endswith(".json")
    ]
    if len(files) != 1:
        raise Round0043Error("R0043 rung output is incomplete")
    with open(files[0], encoding="utf-8") as handle:
        value = json.load(handle)
    validate_seal(value, label="R0043 rung")
    return value


def run_aggregate(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0043 aggregate output"
    )
    started = time.monotonic()
    rungs = {
        label: _read_rung(path)
        for label, path in job["rung_outputs"].items()
    }
    ordered = ["030m", "060m", "120m", "150m"]
    if list(job["rung_outputs"]) != ordered or set(rungs) != set(ordered):
        raise Round0043Error("R0043 aggregate rung set changed")
    release_sha = active["manifest"]["release_sha"]
    if any(value.get("release_sha") != release_sha for value in rungs.values()):
        raise Round0043Error("R0043 rung releases differ")

    core_hashes = {
        value["anchor_views"]["fixed_30m_core"][
            "global_rows_sha256"
        ]
        for value in rungs.values()
    }
    widths = [
        rungs[label]["rung"]["per_corpus_rows"] for label in ordered
    ]
    intervals = [
        rungs[label]["rung"]["selector"]["intervals"]
        for label in ordered
    ]
    nesting = (
        widths == sorted(widths)
        and len(core_hashes) == 1
        and all(
            intervals[index][corpus][0]
            == intervals[index + 1][corpus][0]
            and intervals[index][corpus][1]
            < intervals[index + 1][corpus][1]
            for index in range(len(intervals) - 1)
            for corpus in range(3)
        )
    )
    baseline = rungs["030m"]["anchor_views"]["fixed_30m_core"][
        "metrics"
    ]
    metric_names = ("ffr", "recall_at_10", "density")
    core = {
        label: rungs[label]["anchor_views"]["fixed_30m_core"][
            "metrics"
        ]
        for label in ordered
    }
    rung_wide = {
        label: rungs[label]["anchor_views"]["rung_wide"]["metrics"]
        for label in ordered
    }
    deltas = {
        "fixed_core_from_030m": {
            label: {
                metric: round(
                    float(core[label][metric])
                    - float(baseline[metric]),
                    5,
                )
                for metric in metric_names
            }
            for label in ordered
        },
        "rung_wide_from_030m": {
            label: {
                metric: round(
                    float(rung_wide[label][metric])
                    - float(baseline[metric]),
                    5,
                )
                for metric in metric_names
            }
            for label in ordered
        },
    }
    checks = {
        "balanced_intervals_strictly_nested": nesting,
        "fixed_core_anchor_bytes_identical": len(core_hashes) == 1,
        "030m_core_and_rung_views_identical": (
            core["030m"] == rung_wide["030m"]
        ),
        "150m_reproduces_accepted_R0036": (
            rungs["150m"].get("r0036_full_rung_reproduction", {}).get(
                "passed"
            )
            is True
        ),
        "all_rung_guards_passed": all(
            all(value["guards"].values()) for value in rungs.values()
        ),
    }
    if not all(checks.values()):
        raise Round0043Error("R0043 aggregate correctness check failed")
    body = {
        "schema": "round0043-nested-scale-geometry-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "candidate_universes": {
            label: rungs[label]["rung"] for label in ordered
        },
        "fixed_core_metrics": core,
        "rung_wide_metrics": rung_wide,
        "metric_deltas": deltas,
        "checks": checks,
        "interpretation_contract": {
            "model_and_coordinates_fixed": True,
            "only_candidate_universe_and_anchor_view_change": True,
            "large_fixed_core_density_change_implicates_evaluator_geometry": (
                True
            ),
            "low_030m_density_implicates_training_or_graph_semantics": True,
            "does_not_select_R0042_or_a_future_sampler": True,
            "does_not_claim_a_training_scale_law": True,
            "purity_omitted_because_it_requires_an_extra_full_corpus_"
            "centroid_and_high_fraction_pass": True,
        },
        "rung_receipts": {
            label: expected_input_signature(
                next(
                    os.path.join(job["rung_outputs"][label], name)
                    for name in os.listdir(job["rung_outputs"][label])
                    if name.startswith("rung-") and name.endswith(".json")
                )
            )
            for label in ordered
        },
        "wall_seconds": time.monotonic() - started,
        "training_performed": False,
    }
    receipt = _seal(body)
    path = os.path.join(output, "nested-scale-geometry-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0043 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if len(selected.get("outputs") or []) != 1:
        raise RuntimeError("R0043 job output contract changed")
    action = selected.get("action")
    if action == "score_rung":
        return run_score_rung(active, selected)
    if action == "aggregate":
        return run_aggregate(active, selected)
    raise RuntimeError(f"unknown R0043 action: {action!r}")

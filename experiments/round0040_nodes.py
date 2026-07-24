"""Fresh-process handlers for the Round 0040 representative-only rescore."""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.duplicate_multiplicity import load_duplicate_cap
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0040_program import (
    CachedShardedArray,
    JINA_ROWS,
    MINILM_CAP_PATH,
    MINILM_CAP_SHA256,
    MINILM_DIMENSION,
    MINILM_CAP_RETAINED_ROWS,
    MINILM_ROWS,
    ROUND_ID,
    RepresentativeArrayView,
    RepresentativeRowSelector,
    build_jina_census,
    load_jina_census,
    panel_config,
    seal,
    validate_seal,
)
from basemap import round0027_program as jina
from basemap import round0014_program as minilm


JINA_CENSUS_SOURCE = jina.TRAIN_PATH
JINA_CENSUS_SOURCE_SHA256 = jina.TRAIN_SHA256
JINA_SHARED_ROOT = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/shared-reference"
)
JINA_QUERY_EMBEDDINGS = os.path.join(
    JINA_SHARED_ROOT, "oos-query-embeddings-768.npy"
)
JINA_DECISION = (
    "/data/latent-basemap/runs/round-0038/queue/artifacts/decision/"
    "mrl-seed43-completion-v1.json"
)
JINA_CELLS = {
    "d768_s42": {
        "coordinates": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d768_s42/transform/coordinates.npy"
        ),
        "query_coordinates": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d768_s42/transform/oos-query-coordinates.npy"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d768_s42/panel/panel.json"
        ),
    },
    "d384_s42": {
        "coordinates": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d384_s42/transform/coordinates.npy"
        ),
        "query_coordinates": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d384_s42/transform/oos-query-coordinates.npy"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0037/queue/artifacts/"
            "d384_s42/panel/panel.json"
        ),
    },
    "d768_s43": {
        "coordinates": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d768_s43/transform/coordinates.npy"
        ),
        "query_coordinates": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d768_s43/transform/oos-query-coordinates.npy"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d768_s43/panel/panel.json"
        ),
    },
    "d384_s43": {
        "coordinates": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d384_s43/transform/coordinates.npy"
        ),
        "query_coordinates": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d384_s43/transform/oos-query-coordinates.npy"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0038/queue/artifacts/"
            "d384_s43/panel/panel.json"
        ),
    },
}

MINILM_QUERIES = minilm.QUERIES_PATH
MINILM_QUERY_PROVENANCE = minilm.QUERY_PROVENANCE_PATH
MINILM_CELLS = {
    "u250k": {
        "round_id": "0039",
        "arm": "u250k",
        "coordinates": (
            "/data/latent-basemap/runs/round-0039/queue/artifacts/"
            "u250k/coordinates"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0039/queue/artifacts/"
            "u250k/panel/panel.json"
        ),
    },
    "u500k": {
        "round_id": "0030",
        "arm": "uniform",
        "coordinates": (
            "/data/latent-basemap/runs/round-0030/queue/artifacts/"
            "uniform/coordinates"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0030/queue/artifacts/"
            "uniform/panel/panel.json"
        ),
    },
    "u1000k": {
        "round_id": "0039",
        "arm": "u1000k",
        "coordinates": (
            "/data/latent-basemap/runs/round-0039/queue/artifacts/"
            "u1000k/coordinates"
        ),
        "prior_panel": (
            "/data/latent-basemap/runs/round-0039/queue/artifacts/"
            "u1000k/panel/panel.json"
        ),
    },
}
R0036_PANEL = (
    "/data/latent-basemap/runs/round-0036/queue/artifacts/panel/panel.json"
)


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    validate_seal(value, label=label)
    return value


def _reference_identity(
    *,
    shape: tuple[int, int],
    dtype: str,
    base_shards: list[dict[str, Any]],
    selector: RepresentativeRowSelector,
) -> dict[str, Any]:
    selector_source = selector.source
    shards = [
        {
            "position": position,
            "name": str(item["name"]),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        for position, item in enumerate(base_shards)
    ]
    shards.append({
        "position": len(shards),
        "name": "representative-selector",
        "bytes": int(selector_source["bytes"]),
        "sha256": str(selector_source["sha256"]),
    })
    return {
        "data_identity": {
            "kind": "ordered_shards",
            "shape": [int(shape[0]), int(shape[1])],
            "dtype": np.dtype(dtype).str,
            "shards": shards,
        },
        "convention": {
            "row_order": "compact ascending representative global row IDs",
            "selector": selector.identity(),
            "distance": "squared L2 with panel-v2 exact fp32 rerank",
            "self_exclusion": True,
            "anchor_namespace": "compact representative-row positions",
        },
    }


def _panel_scalars(panel: Mapping[str, Any]) -> dict[str, float]:
    purity = panel.get("purity") or {}
    values = {
        "ffr": float(panel["ffr"]),
        "recall_at_10": float(panel["recall@k"]),
        "density": float(panel["density"]),
        "purity_k256": float(purity["k256"]),
        "purity_k1024": float(purity["k1024"]),
    }
    if any(not np.isfinite(value) for value in values.values()):
        raise RuntimeError(f"non-finite registered panel value: {values}")
    return values


def _projection(
    *,
    query_coordinates: np.ndarray,
    coordinates: RepresentativeArrayView,
    truth: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        cross_knn,
        ffr_from_neighbors,
        recall_at_k_from_neighbors,
    )

    k_fraction = max(config.k_hit, int(math.ceil(config.frac * len(coordinates))))
    low = cross_knn(
        query_coordinates,
        coordinates,
        k_fraction,
        config,
        hi_dim=False,
    )
    high = np.asarray(truth["neighbors"], dtype=np.int64)[:, :config.k_hit]
    return {
        "proj_ffr": round(
            ffr_from_neighbors(high, low, config.k_hit), 4
        ),
        "proj_recall_at_10": round(
            recall_at_k_from_neighbors(high, low, config.k_hit), 5
        ),
        "queries": int(len(query_coordinates)),
        "k_fraction": int(k_fraction),
        "representative_candidates_only": True,
    }


def _tail_receipt(
    *,
    coordinates: np.ndarray,
    selector: RepresentativeRowSelector,
    census: Mapping[str, Any],
    bounds: Mapping[str, Any],
) -> dict[str, Any]:
    xlim = bounds["xlim"]
    ylim = bounds["ylim"]
    outside = np.flatnonzero(
        (coordinates[:, 0] < xlim[0])
        | (coordinates[:, 0] > xlim[1])
        | (coordinates[:, 1] < ylim[0])
        | (coordinates[:, 1] > ylim[1])
    ).astype(np.int64)
    retained = selector.is_retained(outside)
    source = np.load(
        JINA_CENSUS_SOURCE, mmap_mode="r", allow_pickle=False
    )
    outside_vectors = np.asarray(source[outside], dtype=np.float16)
    unique_vectors = (
        int(len(np.unique(outside_vectors, axis=0))) if len(outside) else 0
    )
    arrays = census["arrays"]
    member_rows = arrays["member_rows"]
    offsets = arrays["family_offsets"]
    family_counts = arrays["family_counts"]
    outside_set = set(int(row) for row in outside.tolist())
    contributing = []
    for index in range(len(family_counts)):
        rows = member_rows[offsets[index]:offsets[index + 1]]
        count = sum(int(row) in outside_set for row in rows.tolist())
        if count:
            contributing.append({
                "representative_row": int(rows[0]),
                "family_count": int(family_counts[index]),
                "outside_rows": int(count),
            })
    contributing.sort(
        key=lambda item: (-item["outside_rows"], item["representative_row"])
    )
    return {
        "registered_fixed_axis_bounds": {
            "xlim": [float(value) for value in xlim],
            "ylim": [float(value) for value in ylim],
        },
        "full_rows_outside": int(len(outside)),
        "full_outside_row_ids_sha256": ordered_array_sha256(outside),
        "representatives_outside": int(np.count_nonzero(retained)),
        "excluded_duplicate_or_invalid_rows_outside": int(
            np.count_nonzero(~retained)
        ),
        "unique_exact_vectors_outside": unique_vectors,
        "contributing_exact_families": contributing,
        "largest_family_outside_rows": (
            int(contributing[0]["outside_rows"]) if contributing else 0
        ),
    }


def run_jina_census(
    _active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0040 Jina census output"
    )
    return build_jina_census(
        source_path=JINA_CENSUS_SOURCE,
        output_root=output,
        expected_source_sha256=JINA_CENSUS_SOURCE_SHA256,
    )


def run_jina_rescore(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from basemap.data_loader import PrefixL2NormalizedArray
    from basemap.panel_v2 import (
        build_hiD_reference,
        build_query_truth,
        sample_anchors,
        save_hiD_reference,
        save_query_truth,
        score_panel,
    )

    output = create_fresh_directory(
        job["outputs"][0], label="Round 0040 Jina rescore output"
    )
    started = time.monotonic()
    census_path = os.path.join(job["census_output"], "receipt.json")
    census = load_jina_census(census_path)
    selector = RepresentativeRowSelector(
        census["arrays"]["excluded_rows"],
        row_count=JINA_ROWS,
        source=census["signature"],
        policy=(
            "one-exact-nonzero-fp16-vector; exclude duplicate copies and "
            "zero/nonfinite rows"
        ),
    )
    X_full = jina.input_array(768)
    X = RepresentativeArrayView(X_full, selector)
    config = panel_config()
    centroids = {
        key: np.load(item["path"], mmap_mode="r", allow_pickle=False)
        for key, item in jina.CENTROIDS.items()
    }
    source_signature = expected_input_signature(JINA_CENSUS_SOURCE)
    identity = _reference_identity(
        shape=X.shape,
        dtype=X.dtype.str,
        base_shards=[{
            "name": os.path.basename(JINA_CENSUS_SOURCE),
            "bytes": source_signature["bytes"],
            "sha256": source_signature["sha256"],
        }],
        selector=selector,
    )
    anchors = sample_anchors(len(X), config).astype(np.int64)
    reference = build_hiD_reference(
        X, anchors, config, centroids, **identity
    )
    reference_path = os.path.join(
        output, "representative-high-d-reference.npz"
    )
    save_hiD_reference(reference, reference_path)

    raw_queries = np.load(
        JINA_QUERY_EMBEDDINGS, mmap_mode="r", allow_pickle=False
    )
    Xq_cosine = PrefixL2NormalizedArray(
        raw_queries,
        source_dimension=768,
        output_dimension=768,
        normalize=True,
        source_paths=[JINA_QUERY_EMBEDDINGS],
    )
    X_cosine = RepresentativeArrayView(
        jina.cosine_truth_array(), selector
    )
    cosine_corpus_identity = {
        "representative_data": identity["data_identity"],
        "preprocessing": X_cosine.base.preprocessing,
    }
    query_identity = {
        "query_embeddings": expected_input_signature(JINA_QUERY_EMBEDDINGS),
        "preprocessing": Xq_cosine.preprocessing,
    }
    truth = build_query_truth(
        Xq_cosine,
        X_cosine,
        cfg=config,
        corpus_identity=cosine_corpus_identity,
        query_identity=query_identity,
        k=10,
    )
    truth_path = os.path.join(
        output, "representative-oos-query-truth-k10.npz"
    )
    save_query_truth(truth, truth_path)

    decision = _read_sealed(
        JINA_DECISION, label="R0038 fixed-axis decision"
    )
    bounds = decision["fixed_axis_render"]
    cells: dict[str, Any] = {}
    for label, paths in JINA_CELLS.items():
        coordinates_full = np.load(
            paths["coordinates"], mmap_mode="r", allow_pickle=False
        )
        query_coordinates = np.load(
            paths["query_coordinates"], mmap_mode="r", allow_pickle=False
        )
        if (
            coordinates_full.shape != (JINA_ROWS, 2)
            or coordinates_full.dtype != np.dtype("<f4")
            or query_coordinates.ndim != 2
            or query_coordinates.shape[1] != 2
        ):
            raise RuntimeError(f"Round 0040 Jina {label} coordinate shape changed")
        coordinates = RepresentativeArrayView(
            coordinates_full, selector
        )
        panel = score_panel(
            X,
            coordinates,
            config=config,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=identity,
            provenance={
                "round_id": ROUND_ID,
                "release_sha": active["manifest"]["release_sha"],
                "map_label": label,
                "coordinate": expected_input_signature(
                    paths["coordinates"]
                ),
                "census": census["signature"],
                "scientific_universe": "exact-nonzero-representatives",
            },
        )
        projection = _projection(
            query_coordinates=query_coordinates,
            coordinates=coordinates,
            truth=truth,
            config=config,
        )
        prior = _read_sealed(
            paths["prior_panel"], label=f"{label} prior panel"
        )
        current_scalars = _panel_scalars(panel)
        prior_scalars = _panel_scalars(prior["panel"])
        cell = {
            "schema": "round0040-jina-representative-cell-v1",
            "round_id": ROUND_ID,
            "cell": label,
            "scientific_rows": len(coordinates),
            "full_rows": JINA_ROWS,
            "panel": panel,
            "projection": projection,
            "prior_all_row_panel": expected_input_signature(
                paths["prior_panel"]
            ),
            "prior_all_row_scalars": prior_scalars,
            "representative_scalars": current_scalars,
            "representative_minus_all_row": {
                key: current_scalars[key] - prior_scalars[key]
                for key in current_scalars
            },
            "tail": _tail_receipt(
                coordinates=coordinates_full,
                selector=selector,
                census=census,
                bounds=bounds,
            ),
        }
        path = os.path.join(output, f"{label}-panel.json")
        atomic_write_new_json(path, seal(cell), immutable=True)
        cells[label] = {
            "receipt": expected_input_signature(path),
            "representative_scalars": current_scalars,
            "projection": projection,
            "tail": cell["tail"],
        }
    body = {
        "schema": "round0040-jina-representative-rescore-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "census": census["signature"],
        "census_summary": census["metadata"]["summary"],
        "selector": selector.identity(),
        "reference": expected_input_signature(reference_path),
        "reference_key": reference["key"],
        "query_truth": expected_input_signature(truth_path),
        "query_truth_payload_sha256": truth["payload_sha256"],
        "cells": cells,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _minilm_base() -> CachedShardedArray:
    source = minilm.Round0014MaterializedArray()
    members = []
    for item in source._members:
        members.append({
            "path": item["path"],
            "global_row_start": int(item["global_row_start"]),
            "global_row_stop": int(item["global_row_stop"]),
            "signature": {
                "canonical_path": os.path.realpath(item["path"]),
                "kind": "file",
                "bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            },
        })
    return CachedShardedArray(
        members,
        row_count=MINILM_ROWS,
        dimension=MINILM_DIMENSION,
        dtype="<f2",
    )


def _scan_minilm_invalid_rows(
    base: CachedShardedArray,
) -> tuple[np.ndarray, np.ndarray]:
    zero_parts: list[np.ndarray] = []
    nonfinite_parts: list[np.ndarray] = []
    for member, array in zip(base._members, base._arrays):
        start = int(member["global_row_start"])
        for local_start in range(0, len(array), 65_536):
            block = np.asarray(
                array[local_start:local_start + 65_536]
            )
            zero = np.flatnonzero(np.all(block == 0, axis=1))
            nonfinite = np.flatnonzero(~np.isfinite(block).all(axis=1))
            if len(zero):
                zero_parts.append(
                    (zero + start + local_start).astype(np.int64)
                )
            if len(nonfinite):
                nonfinite_parts.append(
                    (nonfinite + start + local_start).astype(np.int64)
                )
    return (
        np.concatenate(zero_parts)
        if zero_parts else np.empty(0, dtype=np.int64),
        np.concatenate(nonfinite_parts)
        if nonfinite_parts else np.empty(0, dtype=np.int64),
    )


def _build_minilm_selector(
    base: CachedShardedArray,
    *,
    output: str,
) -> tuple[RepresentativeRowSelector, dict[str, Any], dict[str, Any]]:
    cap = load_duplicate_cap(
        MINILM_CAP_PATH,
        expected_sha256=MINILM_CAP_SHA256,
        row_count=MINILM_ROWS,
        fixed_edges_per_source=15,
    )
    zero_rows, nonfinite_rows = _scan_minilm_invalid_rows(base)
    invalid_rows = np.union1d(zero_rows, nonfinite_rows).astype(np.int64)
    excluded_rows = np.union1d(
        cap["excluded_rows"], invalid_rows
    ).astype(np.int64)
    arrays = {
        "excluded_rows": excluded_rows,
        "duplicate_copy_rows": cap["excluded_rows"],
        "zero_rows": zero_rows,
        "nonfinite_rows": nonfinite_rows,
    }
    metadata = seal({
        "schema": "round0040-minilm-representative-selector-v1",
        "round_id": ROUND_ID,
        "row_count": MINILM_ROWS,
        "cap": cap["signature"],
        "selection": (
            "R0020 exact-fp16 duplicate copies plus every exact zero or "
            "nonfinite fp16 row"
        ),
        "cap_retained_rows": MINILM_CAP_RETAINED_ROWS,
        "representative_rows": MINILM_ROWS - len(excluded_rows),
        "zero_rows": int(len(zero_rows)),
        "nonfinite_rows": int(len(nonfinite_rows)),
        "array_sha256": {
            name: ordered_array_sha256(value)
            for name, value in arrays.items()
        },
    })
    path = os.path.join(output, "representative-selector.npz")
    atomic_save_new_npz(
        path,
        immutable=True,
        metadata=np.asarray(
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
        **arrays,
    )
    signature = expected_input_signature(path)
    selector = RepresentativeRowSelector(
        excluded_rows,
        row_count=MINILM_ROWS,
        source=signature,
        policy=(
            "one-exact-nonzero-finite-fp16-vector from R0020 census plus "
            "R0040 full invalid-row scan"
        ),
    )
    return selector, cap, {
        "signature": signature,
        "metadata": metadata,
        "arrays": arrays,
    }


def _load_minilm_selector(
    signature: Mapping[str, Any],
) -> tuple[RepresentativeRowSelector, dict[str, Any]]:
    if (
        not isinstance(signature, Mapping)
        or expected_input_signature(signature.get("canonical_path", ""))
        != dict(signature)
    ):
        raise RuntimeError("Round 0040 MiniLM selector bytes changed")
    with np.load(
        signature["canonical_path"], allow_pickle=False
    ) as archive:
        raw = archive["metadata"].item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(str(raw))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files if name != "metadata"
        }
    validate_seal(metadata, label="Round 0040 MiniLM selector")
    hashes = {
        name: ordered_array_sha256(value)
        for name, value in arrays.items()
    }
    excluded = arrays.get("excluded_rows")
    if (
        metadata.get("schema")
        != "round0040-minilm-representative-selector-v1"
        or metadata.get("array_sha256") != hashes
        or not isinstance(excluded, np.ndarray)
        or excluded.dtype != np.dtype("int64")
        or not np.array_equal(excluded, np.unique(excluded))
        or metadata.get("representative_rows")
        != MINILM_ROWS - len(excluded)
    ):
        raise RuntimeError("Round 0040 MiniLM selector content changed")
    selector = RepresentativeRowSelector(
        excluded,
        row_count=MINILM_ROWS,
        source=dict(signature),
        policy=(
            "one-exact-nonzero-finite-fp16-vector from R0020 census plus "
            "R0040 full invalid-row scan"
        ),
    )
    return selector, {
        "signature": dict(signature),
        "metadata": metadata,
        "arrays": arrays,
    }


def run_minilm_reference(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        build_hiD_reference,
        build_query_truth,
        sample_anchors,
        save_hiD_reference,
        save_query_truth,
    )

    output = create_fresh_directory(
        job["outputs"][0], label="Round 0040 MiniLM reference output"
    )
    started = time.monotonic()
    base = _minilm_base()
    selector, cap, selector_artifact = _build_minilm_selector(
        base, output=output
    )
    X = RepresentativeArrayView(base, selector)
    config = panel_config()
    centroids = {
        256: np.load(
            minilm.CENTROIDS_K256_PATH, mmap_mode="r", allow_pickle=False
        ),
        1024: np.load(
            minilm.CENTROIDS_K1024_PATH, mmap_mode="r", allow_pickle=False
        ),
    }
    identity = _reference_identity(
        shape=X.shape,
        dtype=X.dtype.str,
        base_shards=base.scientific_identity()["shards"],
        selector=selector,
    )
    anchors = sample_anchors(len(X), config).astype(np.int64)
    reference = build_hiD_reference(
        X, anchors, config, centroids, **identity
    )
    reference_path = os.path.join(
        output, "representative-high-d-reference.npz"
    )
    save_hiD_reference(reference, reference_path)
    queries = np.load(
        MINILM_QUERIES, mmap_mode="r", allow_pickle=False
    )
    truth = build_query_truth(
        queries,
        X,
        cfg=config,
        corpus_identity=identity["data_identity"],
        query_identity={
            "queries": expected_input_signature(MINILM_QUERIES),
            "provenance": expected_input_signature(
                MINILM_QUERY_PROVENANCE
            ),
        },
        k=10,
    )
    truth_path = os.path.join(
        output, "representative-oos-query-truth-k10.npz"
    )
    save_query_truth(truth, truth_path)
    body = {
        "schema": "round0040-minilm-representative-reference-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "selector": selector.identity(),
        "selector_artifact": selector_artifact["signature"],
        "selector_summary": {
            "cap_retained_rows": MINILM_CAP_RETAINED_ROWS,
            "representative_rows": selector.retained_count,
            "zero_rows": selector_artifact["metadata"]["zero_rows"],
            "nonfinite_rows": selector_artifact["metadata"][
                "nonfinite_rows"
            ],
        },
        "cap": cap["signature"],
        "reference": expected_input_signature(reference_path),
        "reference_key": reference["key"],
        "reference_content_sha256": reference["content_sha256"],
        "query_truth": expected_input_signature(truth_path),
        "query_truth_key": truth["key"],
        "query_truth_payload_sha256": truth["payload_sha256"],
        "identity": identity,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def _cached_minilm_coordinates(
    *,
    root: str,
    round_id: str,
    arm: str,
) -> tuple[CachedShardedArray, Any]:
    from experiments import run_round0014_node as inherited

    if round_id == "0030":
        inherited.configure_round0030(job={"arm": arm})
    elif round_id == "0039":
        inherited.configure_round0039(job={"arm": arm})
    else:
        raise ValueError(f"unsupported coordinate producer: {round_id}")
    validated = inherited.StreamedCoordinateArray(root)
    members = [{
        "path": item["path"],
        "global_row_start": int(item["global_row_start"]),
        "global_row_stop": int(item["global_row_stop"]),
        "signature": item["signature"],
    } for item in validated._members]
    cached = CachedShardedArray(
        members,
        row_count=MINILM_ROWS,
        dimension=2,
        dtype="<f4",
    )
    return cached, validated


def run_minilm_rescore(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import (
        load_hiD_reference,
        load_query_truth,
        score_panel,
    )

    output = create_fresh_directory(
        job["outputs"][0], label="Round 0040 MiniLM rescore output"
    )
    started = time.monotonic()
    reference_receipt_path = os.path.join(
        job["reference_output"], "receipt.json"
    )
    reference_receipt = _read_sealed(
        reference_receipt_path, label="Round 0040 MiniLM reference"
    )
    reference = load_hiD_reference(
        reference_receipt["reference"]["canonical_path"],
        expected_key=reference_receipt["reference_key"],
    )
    truth = load_query_truth(
        reference_receipt["query_truth"]["canonical_path"],
        expected_key=reference_receipt["query_truth_key"],
        expected_candidate_compute_backend="cuda",
    )
    selector, selector_artifact = _load_minilm_selector(
        reference_receipt["selector_artifact"]
    )
    cap = load_duplicate_cap(
        MINILM_CAP_PATH,
        expected_sha256=MINILM_CAP_SHA256,
        row_count=MINILM_ROWS,
        fixed_edges_per_source=15,
    )
    base = _minilm_base()
    X = RepresentativeArrayView(base, selector)
    config = panel_config()
    centroids = {
        256: np.load(
            minilm.CENTROIDS_K256_PATH, mmap_mode="r", allow_pickle=False
        ),
        1024: np.load(
            minilm.CENTROIDS_K1024_PATH, mmap_mode="r", allow_pickle=False
        ),
    }
    identity = reference_receipt["identity"]
    cells: dict[str, Any] = {}
    for label, spec in MINILM_CELLS.items():
        coordinates_full, validated = _cached_minilm_coordinates(
            root=spec["coordinates"],
            round_id=spec["round_id"],
            arm=spec["arm"],
        )
        coordinates = RepresentativeArrayView(
            coordinates_full, selector
        )
        panel = score_panel(
            X,
            coordinates,
            config=config,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=identity,
            provenance={
                "round_id": ROUND_ID,
                "release_sha": active["manifest"]["release_sha"],
                "map_label": label,
                "coordinate_capability": expected_input_signature(
                    os.path.join(
                        spec["coordinates"], "actual-transform.json"
                    )
                ),
                "cap": cap["signature"],
                "scientific_universe": "exact-fp16-representatives",
            },
        )
        query_coordinates = np.load(
            os.path.join(
                spec["coordinates"], "heldout-query-coordinates.npy"
            ),
            mmap_mode="r",
            allow_pickle=False,
        )
        projection = _projection(
            query_coordinates=query_coordinates,
            coordinates=coordinates,
            truth=truth,
            config=config,
        )
        prior = _read_sealed(
            spec["prior_panel"], label=f"{label} prior all-row panel"
        )
        current_scalars = _panel_scalars(panel)
        prior_scalars = _panel_scalars(prior["panel"])
        cell = {
            "schema": "round0040-minilm-representative-cell-v1",
            "round_id": ROUND_ID,
            "cell": label,
            "source_round_id": spec["round_id"],
            "scientific_rows": len(coordinates),
            "full_rows": MINILM_ROWS,
            "panel": panel,
            "projection": projection,
            "prior_all_row_panel": expected_input_signature(
                spec["prior_panel"]
            ),
            "prior_all_row_scalars": prior_scalars,
            "representative_scalars": current_scalars,
            "representative_minus_all_row": {
                key: current_scalars[key] - prior_scalars[key]
                for key in current_scalars
            },
            "coordinate_stream_record": expected_input_signature(
                os.path.join(
                    validated.root, "actual-transform.json"
                )
            ),
        }
        path = os.path.join(output, f"{label}-panel.json")
        atomic_write_new_json(path, seal(cell), immutable=True)
        cells[label] = {
            "receipt": expected_input_signature(path),
            "representative_scalars": current_scalars,
            "projection": projection,
        }
    body = {
        "schema": "round0040-minilm-representative-rescore-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "selector": selector.identity(),
        "selector_summary": reference_receipt["selector_summary"],
        "reference_receipt": expected_input_signature(
            reference_receipt_path
        ),
        "cells": cells,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = seal(body)
    path = os.path.join(output, "receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_comparison(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0040 comparison output"
    )
    jina_receipt_path = os.path.join(
        job["jina_output"], "receipt.json"
    )
    minilm_receipt_path = os.path.join(
        job["minilm_output"], "receipt.json"
    )
    jina_result = _read_sealed(
        jina_receipt_path, label="Round 0040 Jina rescore"
    )
    minilm_result = _read_sealed(
        minilm_receipt_path, label="Round 0040 MiniLM rescore"
    )
    scale = _read_sealed(R0036_PANEL, label="R0036 representative panel")
    jina_tail = {
        label: cell["tail"] for label, cell in jina_result["cells"].items()
    }
    guards = []
    for result in (jina_result, minilm_result):
        for cell in result["cells"].values():
            panel_path = cell["receipt"]["canonical_path"]
            panel = _read_sealed(panel_path, label="Round 0040 cell panel")
            panel_guards = panel["panel"].get("guards") or {}
            guards.append(
                panel_guards.get("coords_finite") is True
                and panel_guards.get("coords_collapsed") is False
                and panel_guards.get("emb_finite") is True
                and panel_guards.get("emb_zero_rows") == 0
            )
    body = {
        "schema": "round0040-duplicate-controlled-comparison-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "jina": {
            "receipt": expected_input_signature(jina_receipt_path),
            "census_summary": jina_result["census_summary"],
            "cells": jina_result["cells"],
            "tail": jina_tail,
        },
        "minilm_30m": {
            "receipt": expected_input_signature(minilm_receipt_path),
            "cells": minilm_result["cells"],
        },
        "minilm_150m_context": {
            "panel": expected_input_signature(R0036_PANEL),
            "scientific_universe": scale["scientific_universe"],
            "scalars": _panel_scalars(scale["panel"]),
            "comparison_rule": (
                "context only: row count, graph, sampler, representation, "
                "and training pipeline differ; do not interpret as a matched "
                "30M-to-150M scaling cell"
            ),
        },
        "acceptance": {
            "all_seven_panels_expected": (
                len(jina_result["cells"]) == 4
                and len(minilm_result["cells"]) == 3
            ),
            "all_numerical_guards_passed": all(guards) and len(guards) == 7,
            "jina_tail_reports_representative_mass": all(
                item["representatives_outside"]
                <= item["full_rows_outside"]
                for item in jina_tail.values()
            ),
            "diagnostic_only_no_prior_decision_reopened": True,
        },
        "interpretation_contract": {
            "exact_copy_multiplicity": (
                "excluded from primary graph/training/evaluation geometry"
            ),
            "all_row_metrics": (
                "retained only as labeled product/multiplicity diagnostics"
            ),
            "near_or_text_duplicates": "not treated by this round",
            "prior_r0038_selector": (
                "not reopened; representative panels adjudicate tail "
                "interpretation only"
            ),
        },
    }
    body["selector_passed"] = all(body["acceptance"].values())
    receipt = seal(body)
    path = os.path.join(
        output, "duplicate-controlled-comparison-v1.json"
    )
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> Any:
    selected = job or active["job"]
    handlers = {
        "jina_census": run_jina_census,
        "jina_representative_rescore": run_jina_rescore,
        "minilm_representative_reference": run_minilm_reference,
        "minilm_representative_rescore": run_minilm_rescore,
        "duplicate_controlled_comparison": run_comparison,
    }
    handler = handlers.get(str(selected.get("handler") or ""))
    if handler is None:
        raise ValueError(f"unknown Round 0040 handler: {selected.get('handler')}")
    return handler(active, selected)

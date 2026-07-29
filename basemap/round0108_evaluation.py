"""Preregistered evaluation helpers for the 25M diverse-Jina atlas.

Round 0108 deliberately separates three questions:

* a Jina-specific density-v2 floor calibrated only from the accepted R0037
  and R0038 full-768 maps;
* transductive geometry on the retained R0106 universe; and
* projection quality on sources that were not used to train the map.

Projection FFR is retained as a diagnostic.  The only headline OOD decision
is the held-out ``pol_Latn`` recall@50 comparison registered in the round.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .round0104_training import validate_substrate_manifest
from .round0105_search import DIMENSION, GROUPS, K, RETAINED_ROWS, ROW_COUNT
from .round0106_graph import GRAPH_SCHEMA
from .round0107_training import (
    TRAIN_RECEIPT_SCHEMA,
    validate_seal as validate_training_seal,
)


ROUND_ID = "0108"
MAP_KEY = "r0107-diverse-jina-25m-seed42"
MAP_LABEL = "r0107-diverse-jina-25m-seed42"

PANEL_SEED = 108
ANCHORS_PER_GROUP = 256
PANEL_ANCHORS = ANCHORS_PER_GROUP * len(GROUPS)
FRACTION = 0.001
K_HIT = 10
K_LOW_MAX = 50
K_DENSITY = 15
FAMILY_SIZE_CUTOFF = 16
LOG_EPSILON = 1e-12

CALIBRATION_BOOTSTRAP_DRAWS = 1_000
CALIBRATION_BOOTSTRAP_SEED = 10_801
CALIBRATION_NULL_DRAWS = 1_000
CALIBRATION_NULL_SEED = 10_802

HELDOUT_TOTAL_ROWS = 50_000
HELDOUT_CORPUS_ROWS = 49_500
HELDOUT_QUERY_ROWS = 500
HELDOUT_SEED = 108
CROSS_ATLAS_CONTROL_SEED = 10_805
POLISH = "pol_Latn"
IN_MIX_LANGUAGES = tuple(GROUPS[3:])

GLOBAL_FFR_FLOOR = 0.40
LANGUAGE_TO_POOLED_ENGLISH_RATIO = 0.40
POLISH_TO_IN_MIX_MEDIAN_RATIO = 0.50

TRANSFORM_CHUNK_ROWS = 5_000_000
TRANSFORM_BATCH_ROWS = 8_192

CALIBRATION_SCHEMA = "round0108-jina-density-v2-calibration-v1"
CORE_SCHEMA = "round0108-diverse-jina-core-geometry-v1"
OOD_SCHEMA = "round0108-diverse-jina-ood-evaluation-v1"
DECISION_SCHEMA = "round0108-diverse-jina-atlas-decision-v1"


class Round0108Error(RuntimeError):
    """The preregistered R0108 evaluation contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0108Error(f"{label} identity seal is invalid")


def verify_signature(signature: Any, *, label: str) -> str:
    if not isinstance(signature, Mapping):
        raise Round0108Error(f"{label} signature missing")
    path = str(signature.get("canonical_path") or "")
    if not path or expected_input_signature(path) != dict(signature):
        raise Round0108Error(f"{label} bytes changed")
    return path


def read_sealed(path: str, *, label: str, schema: str | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0108Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    if schema is not None and value.get("schema") != schema:
        raise Round0108Error(f"{label} schema changed")
    return value


def pearson_log_radius(high_radius: np.ndarray, low_radius: np.ndarray) -> float:
    high = np.asarray(high_radius, dtype=np.float64)
    low = np.asarray(low_radius, dtype=np.float64)
    if (
        high.ndim != 1
        or low.shape != high.shape
        or len(high) < 100
        or np.any(high < 0)
        or np.any(low < 0)
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        raise Round0108Error("density-v2 radii are malformed")
    high = np.log(high + LOG_EPSILON)
    low = np.log(low + LOG_EPSILON)
    high -= high.mean()
    low -= low.mean()
    denominator = math.sqrt(
        float(np.dot(high, high)) * float(np.dot(low, low))
    )
    if not denominator > 0 or not math.isfinite(denominator):
        raise Round0108Error("density-v2 radius variance collapsed")
    value = float(np.dot(high, low) / denominator)
    if not math.isfinite(value):
        raise Round0108Error("density-v2 correlation is nonfinite")
    return value


def jina_density_floor(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the two-cell floor registered before the R0107 map existed."""
    expected = {"seed42", "seed43"}
    if set(cells) != expected:
        raise Round0108Error("Jina density calibration cells are incomplete")
    points = [
        float(cells[key]["density_v2"]["correlation"])
        for key in sorted(expected)
    ]
    deviations = [
        float(cells[key]["density_v2"]["bootstrap"]["standard_deviation"])
        for key in sorted(expected)
    ]
    nulls = [
        float(
            cells[key]["density_v2"]["permuted_radius_null"][
                "absolute_99_9_percentile"
            ]
        )
        for key in sorted(expected)
    ]
    minimum = min(points)
    maximum_sd = max(deviations)
    proposed = minimum - 3.0 * maximum_sd
    maximum_null = max(nulls)
    finite = all(
        math.isfinite(value)
        for value in (*points, *deviations, *nulls, proposed)
    )
    registered = finite and proposed > 0.0 and proposed > maximum_null
    return {
        "rule": (
            "min(seed42 density_v2, seed43 density_v2) minus three times "
            "max(seed42 bootstrap SD, seed43 bootstrap SD)"
        ),
        "minimum_density_v2": minimum,
        "maximum_bootstrap_standard_deviation": maximum_sd,
        "proposed_floor": proposed,
        "maximum_absolute_null_99_9_percentile": maximum_null,
        "positive": finite and proposed > 0.0,
        "separated_from_permuted_radius_null": (
            finite and proposed > maximum_null
        ),
        "gating_floor_registered": registered,
        "registered_floor": proposed if registered else None,
        "status": (
            "registered"
            if registered
            else "diagnostic-only; positive/null-separation guard failed"
        ),
    }


def fixed_probe_split(
    *,
    row_start: int,
    row_stop: int,
    seed: int,
    total: int = HELDOUT_TOTAL_ROWS,
    corpus: int = HELDOUT_CORPUS_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic, disjoint corpus/query split."""
    if (
        row_start < 0
        or row_stop <= row_start
        or total <= 0
        or corpus <= 0
        or corpus >= total
        or row_stop - row_start < total
    ):
        raise Round0108Error("held-out selection interval is invalid")
    rng = np.random.RandomState(seed)
    selected = rng.choice(
        row_stop - row_start, size=total, replace=False
    ).astype(np.int64)
    selected += row_start
    corpus_rows = np.sort(selected[:corpus])
    query_rows = np.sort(selected[corpus:])
    if (
        len(corpus_rows) != corpus
        or len(query_rows) != total - corpus
        or len(np.intersect1d(corpus_rows, query_rows)) != 0
    ):
        raise Round0108Error("held-out split does not close")
    return corpus_rows, query_rows


def exact_split_duplicate_diagnostics(
    corpus: np.ndarray,
    queries: np.ndarray,
) -> dict[str, Any]:
    """Byte-verify exact families and whether any cross the probe split."""
    left = np.asarray(corpus)
    right = np.asarray(queries)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1:] != right.shape[1:]
        or left.shape[1] <= 0
        or left.dtype != right.dtype
        or left.dtype.kind != "f"
        or left.dtype.itemsize not in {2, 4, 8}
    ):
        raise Round0108Error("probe duplicate-audit inputs are malformed")

    # Identical rows must match at these positions.  Grouping on this compact
    # necessary condition keeps the audit cheap; every candidate group is
    # then split by complete row bytes, so projection collisions cannot create
    # false exact families and exact families cannot be missed.
    positions = np.unique(
        np.linspace(0, left.shape[1] - 1, 32, dtype=np.int64)
    )
    uint_dtype = np.dtype(f"u{left.dtype.itemsize}")
    fingerprints = np.concatenate(
        (
            np.ascontiguousarray(left[:, positions]).view(uint_dtype),
            np.ascontiguousarray(right[:, positions]).view(uint_dtype),
        ),
        axis=0,
    )
    _keys, inverse, counts = np.unique(
        fingerprints, axis=0, return_inverse=True, return_counts=True
    )
    candidate_groups = np.flatnonzero(counts > 1)
    exact_family_count = 0
    exact_family_rows = 0
    maximum_family_size = 1
    cross_split_family_count = 0
    query_rows_with_exact_corpus_copy = 0
    collision_splits = 0
    corpus_rows = len(left)

    def row_bytes(index: int) -> bytes:
        if index < corpus_rows:
            return np.asarray(left[index]).tobytes(order="C")
        return np.asarray(right[index - corpus_rows]).tobytes(order="C")

    for candidate_id in candidate_groups:
        members = np.flatnonzero(inverse == candidate_id)
        exact: dict[bytes, list[int]] = {}
        for member in members.tolist():
            exact.setdefault(row_bytes(member), []).append(member)
        collision_splits += max(len(exact) - 1, 0)
        for family in exact.values():
            if len(family) < 2:
                continue
            exact_family_count += 1
            exact_family_rows += len(family)
            maximum_family_size = max(maximum_family_size, len(family))
            left_count = sum(index < corpus_rows for index in family)
            right_count = len(family) - left_count
            if left_count and right_count:
                cross_split_family_count += 1
                query_rows_with_exact_corpus_copy += right_count

    disjoint = cross_split_family_count == 0
    return {
        "identity": "complete stored-row bytes",
        "candidate_projection_positions": positions.tolist(),
        "candidate_repeated_groups": len(candidate_groups),
        "candidate_collision_splits": collision_splits,
        "exact_nontrivial_family_count": exact_family_count,
        "rows_in_exact_nontrivial_families": exact_family_rows,
        "maximum_exact_family_size": maximum_family_size,
        "cross_split_exact_family_count": cross_split_family_count,
        "query_rows_with_exact_corpus_copy": (
            query_rows_with_exact_corpus_copy
        ),
        "corpus_query_exact_family_disjoint": disjoint,
    }


def recall_from_neighbors(
    truth: np.ndarray,
    observed: np.ndarray,
    *,
    truth_k: int = K_HIT,
) -> float:
    high = np.asarray(truth, dtype=np.int64)
    low = np.asarray(observed, dtype=np.int64)
    if (
        high.ndim != 2
        or low.ndim != 2
        or high.shape[0] != low.shape[0]
        or high.shape[1] < truth_k
        or low.shape[1] < truth_k
    ):
        raise Round0108Error("neighbor recall inputs are malformed")
    return float(np.mean([
        np.isin(high[index, :truth_k], low[index]).sum() / truth_k
        for index in range(len(high))
    ]))


def projection_metrics(
    truth: np.ndarray,
    low_neighbors: np.ndarray,
    *,
    fraction_k: int,
) -> dict[str, float]:
    """Compute the registered projection metrics from exact neighbor IDs."""
    high = np.asarray(truth, dtype=np.int64)
    low = np.asarray(low_neighbors, dtype=np.int64)
    if fraction_k < K_HIT or low.shape[1] < max(K_LOW_MAX, fraction_k):
        raise Round0108Error("projection neighbor width is incomplete")
    return {
        "ffr_diagnostic": recall_from_neighbors(
            high, low[:, :fraction_k]
        ),
        "recall_at_10": recall_from_neighbors(high, low[:, :K_HIT]),
        "recall_at_50_of_high10": recall_from_neighbors(
            high, low[:, :K_LOW_MAX]
        ),
    }


def headline_ood_decision(
    language_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = set(IN_MIX_LANGUAGES) | {POLISH}
    if set(language_metrics) != expected:
        raise Round0108Error("held-out language cells are incomplete")
    in_mix = np.asarray([
        float(language_metrics[name]["recall_at_50_of_high10"])
        for name in IN_MIX_LANGUAGES
    ])
    polish50 = float(language_metrics[POLISH]["recall_at_50_of_high10"])
    polish10 = float(language_metrics[POLISH]["recall_at_10"])
    if (
        not np.isfinite(in_mix).all()
        or not math.isfinite(polish50)
        or not math.isfinite(polish10)
    ):
        raise Round0108Error("held-out language metrics are nonfinite")
    median = float(np.median(in_mix))
    ratio = polish50 / median if median > 0 else 0.0
    checks = {
        "polish_recall50_at_least_half_in_mix_median": (
            polish50 >= POLISH_TO_IN_MIX_MEDIAN_RATIO * median
        ),
        "polish_recall50_strictly_exceeds_recall10": polish50 > polish10,
        "all_twenty_language_cells_complete": len(language_metrics) == 20,
    }
    return {
        "headline_probe": POLISH,
        "in_mix_language_count": len(IN_MIX_LANGUAGES),
        "in_mix_median_recall_at_50_of_high10": median,
        "polish_recall_at_10": polish10,
        "polish_recall_at_50_of_high10": polish50,
        "polish_to_in_mix_median_ratio": ratio,
        "required_ratio": POLISH_TO_IN_MIX_MEDIAN_RATIO,
        "checks": checks,
        "passed": all(checks.values()),
        "projection_ffr_used_for_decision": False,
    }


def core_geometry_decision(
    *,
    density_value: float,
    density_floor: float | None,
    global_ffr: float,
    group_ffr: Mapping[str, float],
    recall_at_10: float,
    recall_at_50: float,
    finite_noncollapsed: bool,
) -> dict[str, Any]:
    if set(group_ffr) != set(GROUPS):
        raise Round0108Error("core group FFR cells are incomplete")
    pooled_english = float(np.mean([group_ffr[name] for name in GROUPS[:3]]))
    language_floor = LANGUAGE_TO_POOLED_ENGLISH_RATIO * pooled_english
    language_checks = {
        name: float(group_ffr[name]) >= language_floor
        for name in IN_MIX_LANGUAGES
    }
    checks = {
        "density_v2_clears_registered_jina_floor": (
            density_floor is not None
            and math.isfinite(density_value)
            and density_value >= density_floor
        ),
        "global_ffr_at_least_0_40": global_ffr >= GLOBAL_FFR_FLOOR,
        "every_language_ffr_at_least_0_40_of_pooled_english": all(
            language_checks.values()
        ),
        "global_recall50_strictly_exceeds_recall10": (
            recall_at_50 > recall_at_10
        ),
        "coordinates_finite_and_noncollapsed": bool(finite_noncollapsed),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "pooled_english_ffr": pooled_english,
        "language_relative_floor": language_floor,
        "language_checks": language_checks,
        "projection_ffr_used_for_decision": False,
    }


def map_family_sizes(
    rows: np.ndarray,
    representative_rows: np.ndarray,
    family_counts: np.ndarray,
) -> np.ndarray:
    query = np.asarray(rows, dtype=np.int64)
    representatives = np.asarray(representative_rows, dtype=np.int64)
    counts = np.asarray(family_counts, dtype=np.int64)
    if (
        representatives.ndim != 1
        or counts.shape != representatives.shape
        or len(representatives)
        and np.any(representatives[1:] <= representatives[:-1])
    ):
        raise Round0108Error("duplicate family arrays are malformed")
    result = np.ones(query.shape, dtype=np.int64)
    positions = np.searchsorted(representatives, query)
    present = positions < len(representatives)
    bounded = np.flatnonzero(present)
    present[bounded] &= representatives[positions[bounded]] == query[bounded]
    result[present] = counts[positions[present]]
    return result


class CompactInt8DequantizedArray:
    """Lazy fp32 dequantization of R0103 rows in R0106 compact order."""

    def __init__(self, mapping: np.ndarray):
        substrate = validate_substrate_manifest(verify_payloads=True)
        payloads = substrate["payloads"]
        self.encoded = np.memmap(
            payloads["int8"]["canonical_path"],
            dtype=np.int8,
            mode="r",
            shape=(ROW_COUNT, DIMENSION),
        )
        self.scales = np.memmap(
            payloads["scales"]["canonical_path"],
            dtype="<f2",
            mode="r",
            shape=(ROW_COUNT,),
        )
        self.mapping = mapping
        if (
            mapping.shape != (RETAINED_ROWS,)
            or mapping.dtype != np.int64
            or int(mapping[0]) < 0
            or int(mapping[-1]) >= ROW_COUNT
            or np.any(mapping[1:] <= mapping[:-1])
        ):
            raise Round0108Error("R0106 compact mapping is malformed")
        self.shape = (RETAINED_ROWS, DIMENSION)
        self.dtype = np.dtype("float32")
        self.substrate = substrate

    def __len__(self) -> int:
        return RETAINED_ROWS

    def __getitem__(self, key: Any) -> np.ndarray:
        scalar = isinstance(key, (int, np.integer))
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            compact = np.arange(start, stop, step, dtype=np.int64)
        else:
            compact = np.asarray([int(key)] if scalar else key, dtype=np.int64)
        shape = compact.shape
        flat = compact.reshape(-1)
        flat = np.where(flat < 0, flat + len(self), flat)
        if np.any(flat < 0) or np.any(flat >= len(self)):
            raise IndexError("compact feature row is out of range")
        global_rows = np.asarray(self.mapping[flat], dtype=np.int64)
        values = np.asarray(self.encoded[global_rows], dtype=np.float32)
        values *= np.asarray(
            self.scales[global_rows], dtype=np.float32
        )[:, None]
        shaped = values.reshape(shape + (DIMENSION,))
        return shaped[0] if scalar else shaped


def load_reviewed_model(
    *,
    train_output: str,
    graph_manifest_path: str,
    graph_manifest_sha256: str,
):
    """Authenticate the R0107 train bundle and load its exact generic model."""
    train_path = os.path.join(train_output, "train-receipt.json")
    config_path = os.path.join(train_output, "production-config.json")
    train_signature = expected_input_signature(train_path)
    config_signature = expected_input_signature(config_path)
    with open(train_path, encoding="utf-8") as handle:
        train = json.load(handle)
    validate_training_seal(train, label="R0107 train receipt")
    with open(config_path, encoding="utf-8") as handle:
        config_receipt = json.load(handle)
    config = config_receipt.get("config")
    if (
        config_receipt.get("schema") != "round0107-production-config-v1"
        or config_receipt.get("round_id") != "0107"
        or not isinstance(config, dict)
    ):
        raise Round0108Error("R0107 production config is missing")
    graph_signature = expected_input_signature(graph_manifest_path)
    with open(graph_manifest_path, encoding="utf-8") as handle:
        graph = json.load(handle)
    validate_seal(graph, label="R0106 graph manifest")
    required_train_checks = {
        "exact_update_closure",
        "zero_numerical_skips",
        "no_pipeline_stamp_drift",
        "endpoint_rows_match_updates",
        "weighted_rejection_accounting_closes",
    }
    train_checks = train.get("train_checks")
    if (
        train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != "0107"
        or train.get("graph_manifest") != graph_signature
        or graph_signature["sha256"] != graph_manifest_sha256
        or graph.get("schema") != GRAPH_SCHEMA
        or train.get("production_config_sha256")
        != sha256_bytes(canonical_json(config))
        or config_receipt.get("config_sha256")
        != train.get("production_config_sha256")
        or not isinstance(train_checks, dict)
        or set(train_checks) != required_train_checks
        or not all(train_checks.values())
    ):
        raise Round0108Error("R0107 reviewed train/model bundle changed")
    model_path = verify_signature(train.get("model"), label="R0107 model")
    mapping_path = verify_signature(
        graph.get("compact_mapping"), label="R0106 compact mapping"
    )
    from .pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected = config["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    if observed != expected:
        raise Round0108Error("R0107 model architecture changed")
    mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
    return {
        "model": model,
        "train": train,
        "train_signature": train_signature,
        "config": config,
        "config_signature": config_signature,
        "graph": graph,
        "graph_signature": graph_signature,
        "mapping": mapping,
    }


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DIMENSION:
        raise Round0108Error("Jina probe geometry changed")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (
        not np.isfinite(array).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise Round0108Error("Jina probe contains zero/nonfinite rows")
    return np.ascontiguousarray(array / norms)


def exact_cosine_topk(
    queries: np.ndarray,
    corpus: np.ndarray,
    *,
    k: int = K_HIT,
    candidate_block_rows: int = 50_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact fp32 cosine truth with bounded candidate blocks on CUDA."""
    import torch

    q = normalize_rows(queries)
    if (
        len(corpus) < k
        or not torch.cuda.is_available()
        or candidate_block_rows <= 0
    ):
        raise Round0108Error("exact OOD truth requires a valid CUDA corpus")
    device = torch.device("cuda")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    query = torch.from_numpy(q).to(device)
    best_values = torch.full(
        (len(q), k), -torch.inf, dtype=torch.float32, device=device
    )
    best_ids = torch.full(
        (len(q), k), -1, dtype=torch.int64, device=device
    )
    blocks = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(corpus), candidate_block_rows):
                stop = min(start + candidate_block_rows, len(corpus))
                candidates = torch.from_numpy(
                    normalize_rows(corpus[start:stop])
                ).to(device)
                similarity = query @ candidates.T
                local_values, local_positions = torch.topk(
                    similarity,
                    min(k, stop - start),
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                local_ids = local_positions.to(torch.int64) + start
                values = torch.cat((best_values, local_values), dim=1)
                ids = torch.cat((best_ids, local_ids), dim=1)
                best_values, order = torch.topk(
                    values, k, dim=1, largest=True, sorted=True
                )
                best_ids = torch.gather(ids, 1, order)
                blocks += 1
                del (
                    candidates,
                    similarity,
                    local_values,
                    local_positions,
                    local_ids,
                    values,
                    ids,
                    order,
                )
        result = best_ids.cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        del query, best_values, best_ids
        torch.cuda.empty_cache()
    if (
        result.shape != (len(q), k)
        or np.any(result < 0)
        or np.any(result >= len(corpus))
        or np.any(np.diff(np.sort(result, axis=1), axis=1) == 0)
    ):
        raise Round0108Error("exact OOD truth is malformed")
    return result, {
        "backend": "cuda-exact-blockwise-fp32-cosine",
        "tf32": False,
        "queries": len(q),
        "corpus_rows": len(corpus),
        "candidate_block_rows": candidate_block_rows,
        "candidate_blocks": blocks,
        "k": k,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        ),
    }


def identity_for_rows(rows: np.ndarray) -> dict[str, Any]:
    values = np.asarray(rows, dtype=np.int64)
    return {
        "rows": len(values),
        "minimum": int(values.min()) if len(values) else None,
        "maximum": int(values.max()) if len(values) else None,
        "ordered_sha256": ordered_array_sha256(values),
    }

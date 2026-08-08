"""Frozen contract for the MiniLM 2M four-seed map family (R0217).

Phase 1 of `guides/plan-minilm-100m-v2.md`. R0216 sealed the 2,000,000-row mixed
MiniLM substrate and its **exact** k15 fuzzy graph — mean recall `1.000000`
against brute-force truth, and `0` zero-degree rows, the R0215 tripwire. This
round trains four maps on exactly those bytes with seeds 42/43/44/45.

Everything except the seed is byte-identical across the four cells. That is the
whole point: a seed family is only interpretable when the seed is the *only*
free variable, so `assert_family_differs_only_by_seed` is a registered,
fail-closed check rather than a convention, and the prepare script stamps the
seed-invariant config digest into the queue so each node can re-derive it and
refuse to run if its own config drifted.

The recipe is the **MiniLM** precedent (R0034's production config shape), not
the 768-dim jina one: `residual_bottleneck`, `input_dimension` 384,
`hidden_dimension` 2048, `hidden_layers` 3, `output_dimension` 2,
`low_dim_kernel` `legacy_lp`, no batchnorm, no dropout, binary positive targets.
Precision is **bf16**: fp16 diverges at this scale (2026-07-18, `logs/topics/basemap.md`).

The dose is not hardcoded. It is derived in code from the *sealed* directed-edge
count with the registered R0184/R0202 low-dose rule::

    updates = ceil(1,000,000 * active_directed_edges / 603,086,368)

Both constants are the accepted R0184 full-rung values and are imported, never
restated. One consequence has to be stated explicitly, because it is a real
property of the rule at this graph size and not a slackened check: the rule is a
`ceil`, so the achieved draws-per-edge can only land on a lattice of spacing
`POSITIVE_ROWS_PER_UPDATE / active_edges`. At the sealed `48,303,258` edges that
spacing is `8.467e-06`, so **no** integer horizon reproduces the target
`0.6781781544098838` to within `1e-06`; the tightest bound the registered rule
admits is one lattice step. The dose check therefore pins two things exactly —
the horizon equals the exact `ceil`, and the achieved dose is inside one
quantisation step of the target — which is strictly stronger than an arbitrary
tolerance because it leaves the horizon no freedom at all.

This round registers **no gate**. Four cells make a family; turning that family
into a mean - 2 sigma gate (the R0161/R0193 method) is a separate later round.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0107_training import performance_windows as _performance_windows
from .round0113_prompt_contrast import (
    BATCH_SIZE,
    NEGATIVE_RNG_SEED_OFFSET,
    POSITIVE_RATIO,
    POSITIVE_ROWS_PER_UPDATE,
    TRAIN_MINIMUM_UPDATES_PER_S,
    TRAIN_WARNING_UPDATES_PER_S,
)
from .round0202_h4096_nested_dose_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
)


ROUND_ID = "0217"

#: The four cells of the family. One round, four train nodes.
SEEDS: tuple[int, ...] = (42, 43, 44, 45)
CANONICAL_SEED = 42

ROWS = 2_000_000
DIMENSION = 384
GRAPH_K = 15

#: The sealed R0216 substrate + exact graph this round consumes.
GRAPH_CAPABILITY = "minilm-mixed-2m-substrate-and-exact-k15-graph-v1"
GRAPH_SCHEMA = "round0216-minilm-mixed-2m-exact-k15-graph-v1"
GRAPH_SOURCE_ROUND_ID = "0216"
#: R0216's sealed receipt reports exactly this many directed edges. The graph is
#: an immutable published artifact, so this is an equality check, not a
#: plausibility band: if the number moves, the input is not the sealed graph.
SEALED_DIRECTED_EDGES = 48_303_258

CAPABILITY_TEMPLATE = "minilm-mixed-2m-map-seed{seed}-low-dose-v1"
TRAIN_SCHEMA = "round0217-minilm-mixed-2m-seed-family-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0217-minilm-mixed-2m-seed-family-production-config-v1"
)
TRAIN_CONFIG_SCHEMA = "round0217-minilm-mixed-2m-seed-family-train-config-v1"

#: The MiniLM recipe (R0034 production-config shape).
ARCHITECTURE = "residual_bottleneck"
HIDDEN_DIMENSION = 2048
HIDDEN_LAYERS = 3
OUTPUT_DIMENSION = 2
LOW_DIM_KERNEL = "legacy_lp"
USE_BATCHNORM = False
USE_DROPOUT = False
POSITIVE_TARGET_MODE = "binary"
#: bf16, never fp16 — fp16 diverges at this scale (2026-07-18 finding).
USE_AMP = "bf16"
LEARNING_RATE = 0.001
CLIP_GRAD_NORM = 1.0
WARMUP_SUCCESSFUL_UPDATES = 200
SCHEDULE = "cosine-v3-positive-budget"

PIPELINE = "host_weighted_minilm_mixed_2m"
PIPELINE_SCHEMA = "round0217-host-weighted-minilm-mixed-2m-pipeline-v1"
SAMPLER_CLASS = "MiniLMMixedWeightedSampler"
POSITIVE_DESTINATION_POLICY = "R0216-exact-k15-fuzzy-tconorm-graph"
#: The 2M universe is a uniform sample of complete rows; no duplicate
#: canonicalization was applied, so no row stands in for any other.
MULTIPLICITY_POLICY = "representative-universe-no-duplicate-canonicalization"
FEATURE_RESIDENCY = "host-mmap-l2-normalized-fp32-substrate"

#: 2M x 384 fp32 is 3.07 GB and is served lazily from a memmap, so resident host
#: memory is page cache plus two pinned 8192-row slots.
HOST_RSS_LIMIT_GIB = 32.0

#: This round registers no gate. Four cells make a family; the mean - 2 sigma
#: registration (R0161/R0193) is a separate later round.
GATE_REGISTERABLE_HERE = False
SEEDS_REQUIRED_FOR_FAMILY_GATE = 3

#: Published-checkpoint check. The saved model is reloaded from disk in the same
#: node and used to project a fixed probe of substrate rows. A checkpoint that
#: cannot be reloaded, or that projects everything onto a point or a line, is
#: not a map — and both failures have historically appeared only after the run.
RELOAD_PROBE_ROWS = 4_096
RELOAD_PROBE_SEED = 217
#: Per-axis spread floor on the probe. A healthy 2-D map spans O(1)-O(10) units;
#: 1e-3 separates "a map" from "a collapsed point cloud" without pretending to
#: be a quality threshold.
MIN_PROBE_COORD_STD = 1.0e-3

#: Config keys whose value is a function of the seed. Everything else must be
#: byte-identical across the four cells.
SEED_BEARING_PATHS: tuple[tuple[str, ...], ...] = (
    ("seed",),
    ("capability",),
    ("optimizer", "seed"),
    ("optimizer", "positive_rng_seed"),
    ("optimizer", "negative_rng_seed"),
    ("execution", "expected_pipeline_stamp", "positive_rng_seed"),
    ("execution", "expected_pipeline_stamp", "negative_rng_seed"),
    ("seed_family", "this_seed"),
    ("seed_family", "this_capability"),
)
SEED_PLACEHOLDER = "<seed-bearing>"


class Round0217Error(RuntimeError):
    """The registered MiniLM 2M seed-family contract changed."""


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0217Error(f"R0217 seed {seed!r} is not a registered cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(
    CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def successful_updates_for_edges(edge_count: int) -> int:
    """The accepted R0184/R0202 low-dose horizon for an active edge count."""
    if int(edge_count) <= 0:
        raise Round0217Error("R0217 active directed-edge count must be positive")
    numerator = FULL_SUCCESSFUL_UPDATES * int(edge_count)
    return (numerator + FULL_GRAPH_EDGES - 1) // FULL_GRAPH_EDGES


def achieved_draws_per_edge(*, updates: int, edge_count: int) -> float:
    if int(edge_count) <= 0 or int(updates) <= 0:
        raise Round0217Error("R0217 dose arithmetic requires positive inputs")
    return int(updates) * POSITIVE_ROWS_PER_UPDATE / int(edge_count)


def dose_quantum(edge_count: int) -> float:
    """One horizon step, in draws-per-edge. The `ceil` rule cannot beat this."""
    if int(edge_count) <= 0:
        raise Round0217Error("R0217 dose arithmetic requires positive inputs")
    return POSITIVE_ROWS_PER_UPDATE / int(edge_count)


def validate_dose(*, updates: int, edge_count: int) -> dict[str, Any]:
    """Pin the horizon exactly and bound the achieved dose by one lattice step.

    Fails closed on either half. The horizon has no freedom at all — it must be
    the exact `ceil` of the registered rule — and the achieved draws-per-edge
    must lie within the single quantisation step that the rule's integer
    rounding necessarily costs.
    """
    exact = successful_updates_for_edges(edge_count)
    if int(updates) != exact:
        raise Round0217Error(
            f"R0217 update horizon {int(updates)} is not the registered "
            f"ceil({FULL_SUCCESSFUL_UPDATES} * {int(edge_count)} / "
            f"{FULL_GRAPH_EDGES}) = {exact}"
        )
    achieved = achieved_draws_per_edge(updates=exact, edge_count=edge_count)
    quantum = dose_quantum(edge_count)
    deviation = abs(achieved - TARGET_POSITIVE_DRAWS_PER_EDGE)
    if not math.isfinite(achieved) or deviation > quantum:
        raise Round0217Error(
            f"R0217 achieved dose {achieved!r} is {deviation:.3e} from the "
            f"registered {TARGET_POSITIVE_DRAWS_PER_EDGE!r}, beyond the "
            f"{quantum:.3e} one-update quantisation bound"
        )
    return {
        "source_round": "0184",
        "source_graph_edges": FULL_GRAPH_EDGES,
        "source_successful_updates": FULL_SUCCESSFUL_UPDATES,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": int(edge_count),
        "successful_updates": exact,
        "dose_rule": (
            "ceil(R0184_successful_updates * active_edges / R0184_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "dose_quantum_draws_per_edge": quantum,
        "achieved_minus_target": achieved - TARGET_POSITIVE_DRAWS_PER_EDGE,
        "tolerance_basis": (
            "one successful-update quantisation step; the ceil rule admits no "
            "tighter bound at this edge count"
        ),
    }


def performance_windows(successful_updates: int) -> int:
    return _performance_windows(int(successful_updates))


def train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str]:
    """The one config every cell uses; only `seed` may differ between cells."""
    if int(seed) not in SEEDS:
        raise Round0217Error(f"R0217 seed {seed!r} is not a registered cell")
    if int(rows) != ROWS:
        raise Round0217Error("R0217 population cardinality changed")
    if int(graph_edges) != SEALED_DIRECTED_EDGES:
        raise Round0217Error(
            f"R0217 graph has {int(graph_edges)} directed edges, sealed R0216 "
            f"reports {SEALED_DIRECTED_EDGES}"
        )
    seed = int(seed)
    updates = successful_updates_for_edges(graph_edges)
    dose = validate_dose(updates=updates, edge_count=graph_edges)
    expected_pipeline = {
        "schema": PIPELINE_SCHEMA,
        "pipeline": PIPELINE,
        "sampler_class": SAMPLER_CLASS,
        "positive_sampling": (
            "fuzzy_weight_proportional_with_replacement_via_exact_"
            "uniform_envelope_rejection"
        ),
        "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
        "negative_sampling": f"uniform-{int(rows)}-substrate-rows-nonself",
        "rng_stream_policy": (
            "separate-positive-rejection-and-negative-pair-streams"
        ),
        "positive_rng_seed": seed,
        "negative_rng_seed": seed + NEGATIVE_RNG_SEED_OFFSET,
        "graph_degree": f"variable-symmetric-fuzzy-k{GRAPH_K}-topology",
        "graph_search_neighbors_including_self": GRAPH_K + 1,
        "graph_nonself_degree": GRAPH_K,
        "host_prefetch": "single-producer-two-pinned-slot",
        "endpoint_forward": "fused-source-destination",
        "weighted_requested": True,
        "weighted_effective": True,
        "uniform_with_replacement": False,
        "positive_with_replacement": True,
        "weight_sampler": "uniform-envelope-rejection-max-weight-one",
        "weight_uniform_dtype": np.dtype("float64").str,
        "valid_canonical_edge_count": int(graph_edges),
        "compact_retained_rows": int(rows),
        "multiplicity_policy": MULTIPLICITY_POLICY,
        "source_representation": "minilm-mixed-2m-fp32",
        "feature_residency": FEATURE_RESIDENCY,
        "device_conversion": "device-fp32-from-exact-fp32",
    }
    config = {
        "schema": TRAIN_CONFIG_SCHEMA,
        "round_id": ROUND_ID,
        "seed": seed,
        "capability": capability_for_seed(seed),
        "family_invariant": {
            "rows": int(rows),
            "dimension": DIMENSION,
            "graph_policy": (
                "byte-identical sealed R0216 exact k15 fuzzy graph in every cell"
            ),
            "sampler": SAMPLER_CLASS,
            "only_treatment_between_cells": "the training seed",
        },
        "input": {
            "rows": int(rows),
            "dimension": DIMENSION,
            "representation": "sealed-R0216-l2-normalized-fp32-substrate",
            "substrate_path": str(substrate_signature["canonical_path"]),
            "substrate_sha256": str(substrate_signature["sha256"]),
            "multiplicity_policy": MULTIPLICITY_POLICY,
        },
        "graph": {
            "capability": GRAPH_CAPABILITY,
            "source_round": GRAPH_SOURCE_ROUND_ID,
            "path": str(graph_signature["canonical_path"]),
            "sha256": str(graph_signature["sha256"]),
            "manifest_path": str(graph_manifest_signature["canonical_path"]),
            "manifest_sha256": str(graph_manifest_signature["sha256"]),
            "k": GRAPH_K,
            "directed_edges": int(graph_edges),
            "exactness": "brute-force fp32 cosine top-k; never quantized",
            "zero_degree_rows": 0,
            "sampling": "fuzzy-weight-proportional-with-replacement",
            "positive_target_mode": POSITIVE_TARGET_MODE,
        },
        "model": {
            "architecture": ARCHITECTURE,
            "input_dimension": DIMENSION,
            "hidden_dimension": HIDDEN_DIMENSION,
            "hidden_layers": HIDDEN_LAYERS,
            "output_dimension": OUTPUT_DIMENSION,
            "use_batchnorm": USE_BATCHNORM,
            "use_dropout": USE_DROPOUT,
            "low_dim_kernel": LOW_DIM_KERNEL,
            "a": 1.0,
            "b": 1.0,
        },
        "optimizer": {
            "seed": seed,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "positive_ratio": POSITIVE_RATIO,
            "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
            "positive_rng_seed": seed,
            "negative_rng_seed": seed + NEGATIVE_RNG_SEED_OFFSET,
            "positive_target_mode": POSITIVE_TARGET_MODE,
            "weighted_edge_sampling": True,
            "correlation_weight": 0.0,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "use_amp": USE_AMP,
            "schedule": SCHEDULE,
            "warmup_successful_updates": WARMUP_SUCCESSFUL_UPDATES,
            "successful_positive_lr_updates": updates,
            "reject_neighbors": False,
        },
        "execution": {
            "device_count": 1,
            "required_pipeline": PIPELINE,
            "gpu_resident_data": False,
            "gpu_resident_vram_budget_gb": 0.0,
            "minimum_train_upd_s": TRAIN_MINIMUM_UPDATES_PER_S,
            "warning_train_upd_s": TRAIN_WARNING_UPDATES_PER_S,
            "performance_subfloor_patience": 2,
            "performance_windows": performance_windows(updates),
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "expected_pipeline_stamp": expected_pipeline,
            "scale_change": (
                "first MiniLM rung of the v2 program: sealed R0216 2M mixed "
                "substrate and its exact k15 graph at h2048 and the accepted "
                "R0184 low dose; recipe, graph, precision, sampler, optimizer "
                "and residency frozen across all four cells"
            ),
            "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
            "achieved_positive_draws_per_edge": dose[
                "achieved_positive_draws_per_edge"
            ],
        },
        "dose_registration": dose,
        "seed_family": {
            "seeds": list(SEEDS),
            "cells": len(SEEDS),
            "this_seed": seed,
            "this_capability": capability_for_seed(seed),
            "canonical_seed": CANONICAL_SEED,
            "cells_required_for_gate": SEEDS_REQUIRED_FOR_FAMILY_GATE,
            "gate_registerable_here": GATE_REGISTERABLE_HERE,
            "gate_note": (
                "four cells make a family; registering the mean - 2 sigma gate "
                "is a separate later round"
            ),
        },
    }
    return config, sha256_bytes(canonical_json(config))


def validate_published_map(coordinates: Any) -> dict[str, Any]:
    """Fail closed unless the reloaded checkpoint produced a real 2-D map."""
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != OUTPUT_DIMENSION:
        raise Round0217Error(
            f"R0217 reloaded checkpoint produced shape {array.shape}, expected "
            f"(n, {OUTPUT_DIMENSION})"
        )
    if array.shape[0] <= 0:
        raise Round0217Error("R0217 reload probe is empty")
    finite = bool(np.isfinite(array).all())
    if not finite:
        raise Round0217Error("R0217 reloaded checkpoint projects nonfinite coordinates")
    std = array.std(axis=0)
    if float(std.min()) < MIN_PROBE_COORD_STD:
        raise Round0217Error(
            f"R0217 reloaded checkpoint is collapsed: per-axis std {std.tolist()} "
            f"below the registered {MIN_PROBE_COORD_STD} floor"
        )
    return {
        "probe_rows": int(array.shape[0]),
        "probe_seed": RELOAD_PROBE_SEED,
        "coordinates_finite": finite,
        "per_axis_std": [float(value) for value in std],
        "minimum_per_axis_std": MIN_PROBE_COORD_STD,
        "collapsed": False,
        "reloaded_from_published_checkpoint": True,
    }


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0217Error(f"R0217 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0217Error(f"R0217 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0217Error(f"R0217 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def seed_invariant_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config with every seed-derived value masked out."""
    projected = copy.deepcopy(dict(config))
    for path in SEED_BEARING_PATHS:
        _set_path(projected, path, SEED_PLACEHOLDER)
    return projected


def seed_invariant_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(seed_invariant_projection(config)))


def expected_seed_bearing_values(seed: int) -> dict[str, Any]:
    """What each seed-bearing field must be, given the seed."""
    seed = int(seed)
    capability = capability_for_seed(seed)
    return {
        ("seed",): seed,
        ("capability",): capability,
        ("optimizer", "seed"): seed,
        ("optimizer", "positive_rng_seed"): seed,
        ("optimizer", "negative_rng_seed"): seed + NEGATIVE_RNG_SEED_OFFSET,
        ("execution", "expected_pipeline_stamp", "positive_rng_seed"): seed,
        ("execution", "expected_pipeline_stamp", "negative_rng_seed"): (
            seed + NEGATIVE_RNG_SEED_OFFSET
        ),
        ("seed_family", "this_seed"): seed,
        ("seed_family", "this_capability"): capability,
    }


def assert_family_differs_only_by_seed(
    configs: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Fail closed unless the four cells differ in the seed and nothing else."""
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0217Error(
            f"R0217 family must be exactly seeds {list(SEEDS)}, got "
            f"{sorted(observed)}"
        )
    digests: dict[int, str] = {}
    for seed in SEEDS:
        config = configs[seed]
        for path, want in expected_seed_bearing_values(seed).items():
            got = _get_path(config, path)
            if got != want:
                raise Round0217Error(
                    f"R0217 seed {seed} cell has {'.'.join(path)}={got!r}, "
                    f"expected {want!r}"
                )
        digests[seed] = seed_invariant_sha256(config)
    unique = sorted(set(digests.values()))
    if len(unique) != 1:
        raise Round0217Error(
            "R0217 cells differ outside the seed; seed-invariant digests are "
            f"{ {seed: digests[seed] for seed in SEEDS} }"
        )
    return {
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": unique[0],
        "per_seed_config_sha256": {
            str(seed): sha256_bytes(canonical_json(dict(configs[seed])))
            for seed in SEEDS
        },
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    }


__all__ = [
    "ARCHITECTURE",
    "BATCH_SIZE",
    "CANONICAL_SEED",
    "CAPABILITIES",
    "CAPABILITY_TEMPLATE",
    "DIMENSION",
    "FEATURE_RESIDENCY",
    "FULL_GRAPH_EDGES",
    "FULL_SUCCESSFUL_UPDATES",
    "GATE_REGISTERABLE_HERE",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GRAPH_SOURCE_ROUND_ID",
    "HIDDEN_DIMENSION",
    "HIDDEN_LAYERS",
    "HOST_RSS_LIMIT_GIB",
    "MIN_PROBE_COORD_STD",
    "MULTIPLICITY_POLICY",
    "OUTPUT_DIMENSION",
    "POSITIVE_TARGET_MODE",
    "RELOAD_PROBE_ROWS",
    "RELOAD_PROBE_SEED",
    "NEGATIVE_RNG_SEED_OFFSET",
    "PIPELINE",
    "PIPELINE_SCHEMA",
    "POSITIVE_DESTINATION_POLICY",
    "POSITIVE_ROWS_PER_UPDATE",
    "PRODUCTION_CONFIG_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0217Error",
    "SAMPLER_CLASS",
    "SEALED_DIRECTED_EDGES",
    "SEEDS",
    "SEEDS_REQUIRED_FOR_FAMILY_GATE",
    "SEED_BEARING_PATHS",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "TRAIN_CONFIG_SCHEMA",
    "TRAIN_SCHEMA",
    "USE_AMP",
    "achieved_draws_per_edge",
    "assert_family_differs_only_by_seed",
    "capability_for_seed",
    "dose_quantum",
    "expected_seed_bearing_values",
    "performance_windows",
    "seed_invariant_projection",
    "seed_invariant_sha256",
    "successful_updates_for_edges",
    "train_config",
    "validate_dose",
    "validate_published_map",
]

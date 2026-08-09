"""R0229 — R0217's treatment with a spill-lifted graph moved in.

R0228's `train_config` is the right construction and the wrong gate: it refuses
any cell outside its own registered `{4, 8, 16} x {42, 43, 44}` set, and its
`graph_exactness` string hardcodes `graph_degree 32`, `intermediate 48`,
`max_iterations 20` and `spill 2` — none of which describe this round's arm.
Publishing R0228's string beside a `spill = 8`, `igd = 256` graph would be a
false description in a sealed artifact.

So the construction is repeated here with the arm's real parameters, and
**nothing about the masking is repeated**: `SEED_BEARING_PATHS`,
`GRAPH_BEARING_PATHS` and `treatment_invariant_sha256` are imported from the
rounds that registered them, the path sets are asserted identical, and the
resulting config must reproduce the cross-round treatment digest
`c28cfd61...` that R0217, R0221, R0223 and R0228 all carry. That digest is the
guarantee; the cell list was only ever a convenience.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from basemap.round0217_minilm_2m_seed_family import (
    SEALED_DIRECTED_EDGES as R0216_SEALED_DIRECTED_EDGES,
    SEED_BEARING_PATHS,
    train_config as r0217_train_config,
)
from basemap.round0223_cuvs_graph_map import (
    GRAPH_BEARING_PATHS,
    REGISTERED_UPDATE_BOUND,
    treatment_invariant_sha256,
)
from basemap.round0228_low_c_map import (
    ROWS,
    TEMPLATE_SEED,
    performance_windows,
    successful_updates_for_edges,
    validate_dose,
)  # R0217's own helpers, re-exported by R0228 and imported read-only here
from basemap.round0226_graph_builders import (
    A_KMEANS_ITERATIONS,
    A_KMEANS_SUBSAMPLE_ROWS,
    A_METRIC,
    A_SEED,
)
from basemap.round0229_phase2_contract import (
    ARM_NAME,
    GRAPH_CAPABILITY,
    TREATMENT_INVARIANT_SHA256,
    map_capability,
)
from basemap.round0229_quality_contract import Round0229Error

BUILDER = "cluster-spill-nnd"


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0229Error(f"R0229 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0229Error(f"R0229 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def graph_exactness(
    *, clusters: int, spill: int, nn_descent: Mapping[str, Any]
) -> str:
    """The arm's real parameters, not R0226's module constants."""
    return (
        f"approximate {BUILDER} at c={int(clusters)} "
        f"(spill {int(spill)}, k-means seed {A_SEED}, {A_KMEANS_ITERATIONS} Lloyd "
        f"iterations on a {A_KMEANS_SUBSAMPLE_ROWS}-row subsample; per-cluster "
        f"cuVS nn-descent graph_degree {int(nn_descent['graph_degree'])}, "
        f"intermediate {int(nn_descent['intermediate_graph_degree'])}, "
        f"max_iterations {int(nn_descent['max_iterations'])}, metric {A_METRIC}; "
        "exact fp32 cosine recompute and exact global top-k merge); never "
        "quantized, never brute force"
    )


def seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    seed = int(seed)
    capability = map_capability(seed)
    values: dict[tuple[str, ...], Any] = {
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
    if set(values) != set(SEED_BEARING_PATHS):
        raise Round0229Error(
            "R0229 seed-bearing path set differs from R0217's registered set"
        )
    return values


def graph_bearing_values(
    *,
    clusters: int,
    spill: int,
    nn_descent: Mapping[str, Any],
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    dose: Mapping[str, Any],
    updates: int,
) -> dict[tuple[str, ...], Any]:
    values: dict[tuple[str, ...], Any] = {
        ("graph", "capability"): GRAPH_CAPABILITY,
        ("graph", "source_round"): "0229",
        ("graph", "path"): str(graph_signature["canonical_path"]),
        ("graph", "sha256"): str(graph_signature["sha256"]),
        ("graph", "manifest_path"): str(graph_manifest_signature["canonical_path"]),
        ("graph", "manifest_sha256"): str(graph_manifest_signature["sha256"]),
        ("graph", "directed_edges"): int(graph_edges),
        ("graph", "exactness"): graph_exactness(
            clusters=clusters, spill=spill, nn_descent=nn_descent
        ),
        ("family_invariant", "graph_policy"): (
            f"byte-identical R0229 {ARM_NAME} c={int(clusters)} s={int(spill)} "
            "k15 fuzzy graph in every cell of this configuration"
        ),
        ("optimizer", "successful_positive_lr_updates"): int(updates),
        ("execution", "expected_pipeline_stamp", "valid_canonical_edge_count"): (
            int(graph_edges)
        ),
        ("execution", "performance_windows"): performance_windows(int(updates)),
        ("execution", "achieved_positive_draws_per_edge"): float(
            dose["achieved_positive_draws_per_edge"]
        ),
        ("execution", "scale_change"): (
            "R0217's treatment on the R0216 queue-correction-3 2M mixed "
            f"substrate with the graph replaced by the {BUILDER} c={int(clusters)} "
            f"s={int(spill)} k15 topology; recipe, precision, sampler, optimizer, "
            "residency and dose rule unchanged"
        ),
        ("dose_registration",): dict(dose),
    }
    if set(values) != set(GRAPH_BEARING_PATHS):
        raise Round0229Error(
            "R0229 graph-bearing path set drifted from R0223's registered set"
        )
    return values


def train_config(
    *,
    clusters: int,
    spill: int,
    nn_descent: Mapping[str, Any],
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    r0216_graph_signature: Mapping[str, Any],
    r0216_graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str, str]:
    """R0217's config with the spill-lifted graph swapped in, nothing else moved."""
    if int(rows) != ROWS:
        raise Round0229Error("R0229 population cardinality changed")
    if int(graph_edges) <= 0:
        raise Round0229Error("R0229 graph has no directed edges")
    updates = successful_updates_for_edges(int(graph_edges))
    if updates > REGISTERED_UPDATE_BOUND:
        raise Round0229Error(
            f"R0229 derived horizon {updates} exceeds the registered bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_dose(updates=updates, edge_count=int(graph_edges))

    template, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=r0216_graph_signature,
        graph_manifest_signature=r0216_graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=R0216_SEALED_DIRECTED_EDGES,
        rows=int(rows),
    )
    config = copy.deepcopy(template)
    for path, replacement in seed_bearing_values(int(seed)).items():
        _set_path(config, path, replacement)
    for path, replacement in graph_bearing_values(
        clusters=int(clusters),
        spill=int(spill),
        nn_descent=nn_descent,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=int(graph_edges),
        dose=dose,
        updates=updates,
    ).items():
        _set_path(config, path, replacement)

    invariant = treatment_invariant_sha256(config)
    template_invariant = treatment_invariant_sha256(template)
    if invariant != template_invariant:
        raise Round0229Error(
            "R0229 cell is not R0217's treatment outside the seed and the graph: "
            f"{invariant} != {template_invariant}"
        )
    if invariant != TREATMENT_INVARIANT_SHA256:
        raise Round0229Error(
            "R0229 cell does not reproduce the cross-round treatment digest "
            f"R0217/R0221/R0223/R0228 carry: {invariant} != "
            f"{TREATMENT_INVARIANT_SHA256}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0229Error("R0229 horizon did not reach the train config")
    return config, sha256_bytes(canonical_json(config)), invariant


__all__ = [
    "BUILDER",
    "graph_bearing_values",
    "graph_exactness",
    "seed_bearing_values",
    "train_config",
]

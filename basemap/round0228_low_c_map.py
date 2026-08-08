"""Frozen contract for R0228 — does a `cluster-spill-nnd` graph move the map?

Review-0227-01 made this the binding question and said why recall can no longer
answer it:

* The R0227 16M `c = 4` headline `0.9939` was scored on `5,000` seeds **unioned
  with their `74,513` exact neighbours** — a size-biased population whose mean
  15th-NN cosine is `0.652` against the substrate's `0.604`. **Uniform it is
  `0.98897`**, so that graph is `1.89x` further from truth than R0223's
  monolithic cuVS graph, which already produced a measurable map difference
  (`density_v2`, exact permutation `p = 0.0121`).
* The residual loss is **not** the partition cut (at `c = 4` only `0.29%` of it
  is), so it cannot be bought back by lowering `c`. It is nn-descent failing
  inside multi-million-row clusters.
* It is `99.6-99.9%` **mutual** and regionally clustered (autocorrelation
  `0.8327`), which is exactly the geometry R0215 showed produces clumps.

So this round trains maps on these graphs at 2M, where exact truth exists, and
compares them to the exact-graph family. Recall is still reported — **always
over the uniform population of all 2,000,000 rows, never a hub-biased
subpopulation**, which is review-0227-01's central methodological finding — but
it is not the evidence. The maps are.

## The scoring population, stated before anything is measured

Every recall figure in this round is over **all 2,000,000 substrate rows**.
There is no seed set, no neighbour union, no sampled query set. R0220's sealed
exact truth covers every row, so a uniform population costs nothing and the
biased alternative has no justification here.

## The configurations, and why `c = 4` is the floor rather than the middle

R0227's per-rung table selects `c = 4` at the `6.25M` and `12.5M` rungs, `c = 5`
at `25M`, `c = 10` at `50M` and `c = 22` at `100M` (review-0227-01 corrects the
last to `c = 24` under measured imbalance). At the 2M substrate the same rule —
the smallest `c` whose largest realised cluster fits the device budget — selects
`c = 4`, and `C_MIN = 4` is a **structural floor**: at spill `s = 2`, `c = 2`
puts every row in every cluster and nothing is partitioned at all.

So the mandate's "one lower-`c` bracket" does not exist inside this builder. The
no-partition limit is `c = 1`, i.e. a monolithic build, and that arm is already
trained: **R0223's three cuVS `igd48` cells at tie-aware `0.99416` over all 2M
rows**. This round therefore brackets upward instead, and says so:

| arm | `c` | tie-aware recall, all 2M rows | provenance |
| --- | --- | --- | --- |
| exact | -- | `1.0` by construction | R0217/R0221, 8 cells |
| monolithic cuVS | (1) | `0.994164` | R0223, 3 cells |
| **cluster-spill** | **4** | measured here | **built here** |
| **cluster-spill** | **8** | `0.970770` | R0227 sealed, re-measured here |
| **cluster-spill** | **16** | `0.951162` | R0227 sealed, re-measured here |

`c = 16` is the bracket that matters most: `0.951` straddles the `0.9447`
R0227 projects for the `100M` rung, so a Phase 2 100M map sits between the
`c = 16` and `c = 8` arms measured here, not near `c = 4`.

## What is held fixed

The treatment is R0217's in every respect except the seed and the graph, by the
**same registered mechanism R0223 used and review-0223-01 verified by its own
independent masking**: `GRAPH_BEARING_PATHS` and `treatment_invariant_projection`
are imported from R0223 rather than restated, and every cell must reproduce the
digest `c28cfd61e744a2e19e136940a13ae0ad26bd9b9d8b9525906df57f0e7a56e784` that
R0217, R0221 and R0223 all carry. A cell that cannot is refused before it trains.

The dose is the registered rule applied to this graph's edge count:
`ceil(1e6 * active_directed_edges / 603,086,368)`. Review-0223-01 verified that a
different edge count legitimately yields a different horizon and that this is
`ceil` quantisation rather than a deviation; the same arithmetic is published
here for the same reason.

## What is gated, and what is only described

Per review-0225-01, `density_v2` is **descriptive only**: one anchor of 4,000
(substrate row `1449227`, 1,377 duplicate coordinates, `r_hd == 0`) supplies
about two thirds of its value, and dropping it leaves cell ranking uncorrelated.
Gating is on `ffr` (one-sided `95/95` floor) and the two purity metrics (two-sided
`95/95` bands on the **unfolded** log ratio). `density_v2` is still published,
still z-scored and still permutation-tested — it is the metric R0223's signal
appeared on and dropping it would be exactly the kind of convenient excision this
program has already had to correct twice.

Review-0225-01's identity bound is registered here too: with `n < 13` no defining
cell can fail a two-sided band and with `n < 11` none can fail a one-sided floor,
because `max|z| <= (n-1)/sqrt(n)`. A pass count on the eight defining cells is
therefore not evidence, and this round says so beside every table.

## What could make this round wrong, and how it would show

`n = 3` per configuration has little power against a shift below ~`1.5` sigma
(review-0223-01's TOST calculation). So an equivalence claim is not available and
is not made. What *is* available at this `n`, and what caught R0223's signal, is
an **exact permutation test**, and this round runs three of them: per
configuration against the eight-cell exact family, pooled across configurations,
and a trend test in `log2(c)` across the nine candidate cells alone. The trend
test is the sharpest instrument here — if the graph is what moves the map, the
metric should move with the amount of missing edge mass, and that is a
distribution-free statement over `9!/(3!3!3!) = 1,680` relabellings that needs no
comparison family at all.
"""
from __future__ import annotations

import copy
import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0216_minilm_2m_substrate import MAX_ZERO_DEGREE_ROWS
from .round0217_minilm_2m_seed_family import (
    BATCH_SIZE,
    DIMENSION,
    GRAPH_K,
    HOST_RSS_LIMIT_GIB,
    OUTPUT_DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROWS,
    SEALED_DIRECTED_EDGES as R0216_SEALED_DIRECTED_EDGES,
    SEED_BEARING_PATHS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    USE_AMP,
    performance_windows,
    seed_invariant_projection,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config as r0217_train_config,
    validate_dose,
    validate_published_map,
)
from .round0218_minilm_2m_panel import PANEL_METRICS
from .round0223_cuvs_graph_map import (
    GRAPH_BEARING_PATHS,
    GRAPH_PLACEHOLDER,
    MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
    R0216_EXACT_KERNEL_MIN_DISTANCE,
    R0216_EXACT_KERNEL_NEGATIVE_ENTRIES,
    R0222_POOLED_SEEDS as EXACT_FAMILY_SEEDS,
    FUZZY_LAW,
    FUZZY_RANDOM_STATE_SEED,
    REGISTERED_UPDATE_BOUND,
    FULL_TRANSFORM_BATCH,
    treatment_invariant_sha256,
)
from .round0226_graph_builders import (
    A_GRAPH_DEGREE,
    A_INTERMEDIATE_DEGREE,
    A_KMEANS_ITERATIONS,
    A_KMEANS_SUBSAMPLE_ROWS,
    A_MAX_ITERATIONS,
    A_METRIC,
    A_SEED,
    A_SPILL,
    SUBSTRATE_2M_PATH,
    SUBSTRATE_2M_ROWS,
)
from .round0227_low_c_contract import C_MIN
from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET


ROUND_ID = "0228"

# --------------------------------------------------------------------------- #
# the arms
# --------------------------------------------------------------------------- #
#: The cluster counts trained here. `4` is what R0227's per-rung table selects at
#: the two smallest Phase 2 rungs AND the structural floor of the builder; `8`
#: and `16` bracket upward, with `16` straddling the `0.9447` the table projects
#: for `100M`. There is no admissible lower bracket (see the module docstring).
CLUSTER_COUNTS: tuple[int, ...] = (4, 8, 16)
#: Three seeds per configuration. Review-0223-01 established that fewer cannot be
#: compared to the eight-cell family at all, and that a real equivalence test
#: needs about `n = 18` per arm — which this round does not have and does not
#: claim.
SEEDS: tuple[int, ...] = (42, 43, 44)
CELLS: tuple[tuple[int, int], ...] = tuple(
    (clusters, seed) for clusters in CLUSTER_COUNTS for seed in SEEDS
)
TEMPLATE_SEED = 42

#: `c = 4` at 2M has no sealed R0227 build (R0227's 2M cells were `c = 64/32/16/8`),
#: so it is built here with R0227's builder script, unmodified and imported.
#: Everything else is R0227's sealed bytes, re-measured rather than trusted.
CLUSTERS_BUILT_HERE: tuple[int, ...] = (4,)
CLUSTERS_FROM_R0227: tuple[int, ...] = (8, 16)

#: R0227's published tie-aware recall over all 2,000,000 rows for the two sealed
#: graphs, which review-0227-01 independently reproduced to six decimals. The
#: node re-measures and must land on these, so a swapped or silently rebuilt
#: graph is caught before a single optimizer step.
R0227_TIE_AWARE_RECALL_BY_C: dict[int, float] = {
    8: 0.9707696000000001,
    16: 0.951162,
}
R0227_STRICT_RECALL_BY_C: dict[int, float] = {
    8: 0.9696856141090393,
    16: 0.9338752,
}
RECALL_CROSS_CHECK_TOLERANCE = 1.0e-6
#: Only the tie-aware value is cross-checked against R0227's publication. Its
#: strict counterpart at `c = 16` is quoted in review-0227-01 via the ladder
#: rather than in the result's headline table, so it is reported, not asserted.
CROSS_CHECKED_CLUSTERS: tuple[int, ...] = (8,)

#: **Uniform.** Every recall figure in this round is over all rows.
RECALL_POPULATION = "all 2,000,000 substrate rows, uniform; no seed set, no neighbour union"
RECALL_POPULATION_NOTE = (
    "review-0227-01 found R0227's 16M headline scored over 5,000 seeds unioned "
    "with their 74,513 exact neighbours, a size-biased set whose mean 15th-NN "
    "cosine is 0.652 against the substrate's 0.604, and that the uniform value "
    "is 0.98897 rather than 0.9939. This round scores every graph over the "
    "complete population, which at 2M costs nothing because R0220's sealed exact "
    "truth covers every row."
)

CLUSTER_SPILL_BUILDER = "cluster-spill-nnd"
BUILDER_SOURCE_ROUNDS = ("0226", "0227")

GRAPH_CAPABILITY_TEMPLATE = "minilm-mixed-2m-cluster-spill-c{clusters}-k15-fuzzy-graph-v1"
MAP_CAPABILITY_TEMPLATE = (
    "minilm-mixed-2m-cluster-spill-c{clusters}-map-seed{seed}-low-dose-v1"
)
COMPARISON_CAPABILITY = "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1"
GEOMETRY_CAPABILITY = "minilm-mixed-2m-cluster-spill-map-geometry-v1"

BUILD_SCHEMA = "round0228-minilm-mixed-2m-cluster-spill-k15-fuzzy-graph-v1"
TRAIN_SCHEMA = "round0228-minilm-mixed-2m-cluster-spill-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = "round0228-minilm-mixed-2m-cluster-spill-production-config-v1"
COMPARISON_SCHEMA = "round0228-minilm-mixed-2m-cluster-spill-graph-map-comparison-v1"
GEOMETRY_SCHEMA = "round0228-minilm-mixed-2m-cluster-spill-map-geometry-v1"

# --------------------------------------------------------------------------- #
# the treatment identity
# --------------------------------------------------------------------------- #
#: The digest R0217, R0221 and R0223 all carry under the registered mask.
#: Review-0223-01 reproduced it with its own independent masking implementation
#: on all three rounds' configs, so it is a cross-round constant rather than this
#: round's assertion. A cell that does not reproduce it is refused.
R0217_TREATMENT_INVARIANT_SHA256 = (
    "c28cfd61e744a2e19e136940a13ae0ad26bd9b9d8b9525906df57f0e7a56e784"
)

#: Applied to every graph built or consumed here, exactly as R0223 applied them.
RECALL_MEAN_FLOOR = 0.90
RECALL_P10_FLOOR = 0.80
R0171_FLOOR_SOURCE = "R0171 accepted ANN graph qualification floors (0.90 / 0.80)"
ZERO_DEGREE_TRIPWIRE = MAX_ZERO_DEGREE_ROWS

# --------------------------------------------------------------------------- #
# the comparison families
# --------------------------------------------------------------------------- #
R0223_CUVS_SEEDS: tuple[int, ...] = (42, 43, 44)
R0223_COMPARISON_SCHEMA = "round0223-minilm-mixed-2m-cuvs-graph-map-comparison-v1"
R0225_GATE_SCHEMA = "round0225-minilm-mixed-2m-tolerance-gates-v1"
R0222_GATE_SCHEMA = "round0222-minilm-mixed-2m-quality-gates-n8-v1"

#: Review-0225-01 released `ffr` and both two-sided purity bands as acceptance
#: criteria and held `density_v2` DESCRIPTIVE ONLY. This round gates on exactly
#: the released set and publishes `density_v2` beside it.
GATED_METRICS: tuple[str, ...] = ("ffr", "purity_fidelity_k256", "purity_fidelity_k1024")
DESCRIPTIVE_ONLY_METRICS: tuple[str, ...] = ("density_v2",)
PURITY_METRICS: tuple[str, ...] = ("purity_fidelity_k256", "purity_fidelity_k1024")
PURITY_RATIO_KEYS: dict[str, str] = {
    "purity_fidelity_k256": "k256",
    "purity_fidelity_k1024": "k1024",
}
DENSITY_V2_STATUS = (
    "DESCRIPTIVE ONLY (review-0225-01): anchor index 2846 / substrate row 1449227 "
    "has 1,377 duplicate coordinates and r_hd == 0, supplying ~65% of the "
    "metric's value; with that anchor dropped the cell ranking is uncorrelated "
    "(rank correlation -0.05). Reported and permutation-tested here; never used "
    "to pass or fail a map."
)
IDENTITY_BOUND_NOTE = (
    "max|z| <= (n-1)/sqrt(n) for any cell inside the family that defines the "
    "floor. At n = 8 that is 2.4749, below the one-sided k = 3.1873 and the "
    "two-sided k2 = 3.7685, so NO defining cell can fail either family and a "
    "pass count over the eight defining cells is not evidence. The first "
    "informative n is 11 one-sided and 13 two-sided (review-0225-01)."
)

EVIDENCE_LIMITS = (
    "This round trains n=3 cells per cluster count against an n=8 exact-graph "
    "family. Review-0223-01 computed that the smallest equivalence margin this "
    "design can certify is 1.47-2.02 exact-family sd, and that a real "
    "equivalence test needs n ~ 18 per arm. So no equivalence is claimed and "
    "none is available: a cell inside the family's band licenses 'no difference "
    "was detected at this n', never 'the graphs are equivalent'. A difference "
    "detected here IS informative in the strong direction, and the trend test "
    "across the three cluster counts is the instrument with the most resolution "
    "because it needs no comparison family at all."
)
ADOPTION_CLAIMED = False
EQUIVALENCE_CLAIMED = False
GATE_REGISTERABLE_HERE = False
GATE_RELEASE_CLAIMED = False


class Round0228Error(RuntimeError):
    """The registered R0228 cluster-spill map contract changed."""


def graph_capability(clusters: int) -> str:
    if int(clusters) not in CLUSTER_COUNTS:
        raise Round0228Error(f"R0228 cluster count {clusters!r} is not registered")
    return GRAPH_CAPABILITY_TEMPLATE.format(clusters=int(clusters))


def map_capability(clusters: int, seed: int) -> str:
    if (int(clusters), int(seed)) not in CELLS:
        raise Round0228Error(f"R0228 cell (c={clusters}, seed={seed}) is not registered")
    return MAP_CAPABILITY_TEMPLATE.format(clusters=int(clusters), seed=int(seed))


GRAPH_CAPABILITIES: tuple[str, ...] = tuple(
    GRAPH_CAPABILITY_TEMPLATE.format(clusters=clusters) for clusters in CLUSTER_COUNTS
)
MAP_CAPABILITIES: tuple[str, ...] = tuple(
    MAP_CAPABILITY_TEMPLATE.format(clusters=clusters, seed=seed)
    for clusters, seed in CELLS
)


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0228Error(f"R0228 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0228Error(f"R0228 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0228Error(f"R0228 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def seed_bearing_values(clusters: int, seed: int) -> dict[tuple[str, ...], Any]:
    """What each of R0217's nine seed-bearing fields holds for this cell."""
    seed = int(seed)
    capability = map_capability(clusters, seed)
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
        raise Round0228Error(
            "R0228 seed-bearing path set differs from R0217's registered set"
        )
    return values


def graph_exactness(clusters: int) -> str:
    return (
        f"approximate {CLUSTER_SPILL_BUILDER} at c={int(clusters)} "
        f"(spill {A_SPILL}, k-means seed {A_SEED}, {A_KMEANS_ITERATIONS} Lloyd "
        f"iterations on a {A_KMEANS_SUBSAMPLE_ROWS}-row subsample; per-cluster "
        f"cuVS nn-descent graph_degree {A_GRAPH_DEGREE}, intermediate "
        f"{A_INTERMEDIATE_DEGREE}, max_iterations {A_MAX_ITERATIONS}, metric "
        f"{A_METRIC}; exact fp32 cosine recompute and exact global top-k merge); "
        "never quantized, never brute force"
    )


def graph_bearing_values(
    *,
    clusters: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    dose: Mapping[str, Any],
    updates: int,
) -> dict[tuple[str, ...], Any]:
    """What each graph-derived field holds for a cluster-spill graph."""
    values: dict[tuple[str, ...], Any] = {
        ("graph", "capability"): graph_capability(clusters),
        ("graph", "source_round"): ROUND_ID,
        ("graph", "path"): str(graph_signature["canonical_path"]),
        ("graph", "sha256"): str(graph_signature["sha256"]),
        ("graph", "manifest_path"): str(graph_manifest_signature["canonical_path"]),
        ("graph", "manifest_sha256"): str(graph_manifest_signature["sha256"]),
        ("graph", "directed_edges"): int(graph_edges),
        ("graph", "exactness"): graph_exactness(clusters),
        ("family_invariant", "graph_policy"): (
            f"byte-identical R0228 cluster-spill c={int(clusters)} k15 fuzzy "
            "graph in every cell of this configuration"
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
            f"substrate with the graph replaced by the {CLUSTER_SPILL_BUILDER} "
            f"c={int(clusters)} k15 topology; recipe, precision, sampler, "
            "optimizer, residency and dose rule unchanged"
        ),
        ("dose_registration",): dict(dose),
    }
    if set(values) != set(GRAPH_BEARING_PATHS):
        raise Round0228Error(
            "R0228 graph-bearing path set drifted from R0223's registered set"
        )
    return values


def train_config(
    *,
    clusters: int,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    r0216_graph_signature: Mapping[str, Any],
    r0216_graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str, str]:
    """R0217's config with the graph swapped, and nothing else moved."""
    if (int(clusters), int(seed)) not in CELLS:
        raise Round0228Error(f"R0228 cell (c={clusters}, seed={seed}) is not registered")
    if int(rows) != ROWS:
        raise Round0228Error("R0228 population cardinality changed")
    if int(graph_edges) <= 0:
        raise Round0228Error("R0228 graph has no directed edges")
    updates = successful_updates_for_edges(int(graph_edges))
    if updates > REGISTERED_UPDATE_BOUND:
        raise Round0228Error(
            f"R0228 derived horizon {updates} exceeds the registered bound "
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
    for path, replacement in seed_bearing_values(int(clusters), int(seed)).items():
        _set_path(config, path, replacement)
    for path, replacement in graph_bearing_values(
        clusters=int(clusters),
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
        raise Round0228Error(
            "R0228 cell is not R0217's treatment outside the seed and the graph: "
            f"{invariant} != {template_invariant}"
        )
    if invariant != R0217_TREATMENT_INVARIANT_SHA256:
        raise Round0228Error(
            "R0228 cell does not reproduce the cross-round treatment digest "
            f"R0217/R0221/R0223 carry: {invariant} != "
            f"{R0217_TREATMENT_INVARIANT_SHA256}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0228Error("R0228 horizon did not reach the train config")
    return config, sha256_bytes(canonical_json(config)), invariant


def assert_configuration_family(
    configs: Mapping[int, Mapping[str, Any]],
    *,
    clusters: int,
    expected_treatment_invariant: str,
) -> dict[str, Any]:
    """Fail closed unless one configuration's cells differ only by the seed."""
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0228Error(
            f"R0228 configuration c={clusters} must be exactly seeds {list(SEEDS)}, "
            f"got {sorted(observed)}"
        )
    seed_digests: dict[int, str] = {}
    treatment_digests: dict[int, str] = {}
    per_seed: dict[str, str] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        for path, want in seed_bearing_values(clusters, seed).items():
            got = _get_path(config, path)
            if got != want:
                raise Round0228Error(
                    f"R0228 c={clusters} seed {seed} has {'.'.join(path)}={got!r}, "
                    f"expected {want!r}"
                )
        seed_digests[seed] = seed_invariant_sha256(config)
        treatment_digests[seed] = treatment_invariant_sha256(config)
        per_seed[str(seed)] = sha256_bytes(canonical_json(config))
    if len(set(seed_digests.values())) != 1:
        raise Round0228Error(
            f"R0228 c={clusters} cells differ outside the seed: {seed_digests}"
        )
    if set(treatment_digests.values()) != {expected_treatment_invariant}:
        raise Round0228Error(
            f"R0228 c={clusters} cells are not R0217's treatment outside the graph"
        )
    if len(set(per_seed.values())) != len(SEEDS):
        raise Round0228Error(f"R0228 c={clusters} produced duplicate cell configs")
    return {
        "clusters": int(clusters),
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": sorted(set(seed_digests.values()))[0],
        "treatment_invariant_sha256": expected_treatment_invariant,
        "per_seed_config_sha256": per_seed,
    }


def validate_cluster_spill_graph(
    *,
    clusters: int,
    degrees: Mapping[str, Any],
    recall: Mapping[str, float],
    edges: int,
    structural: Mapping[str, int],
) -> dict[str, Any]:
    """R0171's ANN floors plus the R0215 zero-degree tripwire, fail-closed."""
    mean = float(recall["mean_recall_at_k"])
    p10 = float(recall["p10_recall_at_k"])
    if not math.isfinite(mean) or not math.isfinite(p10):
        raise Round0228Error(f"R0228 c={clusters} recall is not finite")
    if mean < RECALL_MEAN_FLOOR or p10 < RECALL_P10_FLOOR:
        raise Round0228Error(
            f"R0228 c={clusters} recall {mean:.6f}/{p10:.6f} is below the "
            f"{RECALL_MEAN_FLOOR}/{RECALL_P10_FLOOR} floors ({R0171_FLOOR_SOURCE})"
        )
    for key in ("self_loop_entries", "duplicate_entries", "out_of_range_entries"):
        if int(structural.get(key, -1)) != 0:
            raise Round0228Error(
                f"R0228 c={clusters} k15 slice has {structural.get(key)} {key}"
            )
    if int(structural.get("rows_below_k", -1)) != 0:
        raise Round0228Error(
            f"R0228 c={clusters} k15 slice has rows with fewer than k ids"
        )
    zero = int(degrees["zero_degree_rows"])
    if zero > ZERO_DEGREE_TRIPWIRE:
        raise Round0228Error(
            f"{zero} rows have zero edges at c={clusters}. R0215 showed this is "
            "exactly what produced the v1 150M clumps; the rung does not proceed "
            "with edgeless rows."
        )
    if int(edges) <= 0:
        raise Round0228Error(f"R0228 c={clusters} fuzzy graph has no directed edges")
    return {
        "clusters": int(clusters),
        "mean_recall_at_k": mean,
        "p10_recall_at_k": p10,
        "mean_recall_floor": RECALL_MEAN_FLOOR,
        "p10_recall_floor": RECALL_P10_FLOOR,
        "recall_floor_source": R0171_FLOOR_SOURCE,
        "recall_population": RECALL_POPULATION,
        "zero_degree_rows": zero,
        "zero_degree_tripwire": ZERO_DEGREE_TRIPWIRE,
        "directed_edges": int(edges),
        "structural": dict(structural),
        "exactness": graph_exactness(int(clusters)),
    }


def validate_full_population_map(coordinates: Any) -> dict[str, Any]:
    array = np.asarray(coordinates)
    if array.shape != (ROWS, OUTPUT_DIMENSION):
        raise Round0228Error(
            f"R0228 full-population transform produced {array.shape}, expected "
            f"({ROWS}, {OUTPUT_DIMENSION})"
        )
    finite = int(np.isfinite(array).all(axis=1).sum())
    if finite != ROWS:
        raise Round0228Error(
            f"R0228 full-population transform has {ROWS - finite} nonfinite rows"
        )
    published = validate_published_map(array)
    return {
        **published,
        "transform_rows": ROWS,
        "transform_rows_finite": finite,
        "full_population_finite": True,
    }


# --------------------------------------------------------------------------- #
# statistics: exact permutation tests, done three ways
# --------------------------------------------------------------------------- #
def _mean_sd(values: Sequence[float]) -> tuple[float, float, int]:
    numbers = [float(value) for value in values]
    if len(numbers) < 2 or any(not math.isfinite(value) for value in numbers):
        raise Round0228Error("R0228 summary needs >= 2 finite values")
    return statistics.fmean(numbers), statistics.stdev(numbers), len(numbers)


def _variance(values: Sequence[float]) -> float:
    return float(statistics.variance([float(value) for value in values]))


def exact_permutation_two_sample(
    treatment: Sequence[float], control: Sequence[float]
) -> dict[str, Any]:
    """Every relabelling of the pooled cells, no normality assumption.

    Two statistics, because review-0223-01 showed the only signal in this program
    so far was in DISPERSION and a mean test had no power to see it:

    * variance ratio `var(treatment)/var(control)`, one-sided upper — "the new
      arm is more spread out than the family";
    * `|mean(treatment) - mean(control)|`, two-sided — "the new arm sits
      somewhere else".

    Also reports how many treatment cells land inside the control's observed
    range, and the exact probability of seeing that few or fewer under
    exchangeability, because "the mean agrees while dispersion differs" is the
    failure shape this program has already seen once.
    """
    treated = [float(value) for value in treatment]
    controlled = [float(value) for value in control]
    if len(treated) < 2 or len(controlled) < 2:
        raise Round0228Error("R0228 permutation test needs >= 2 cells per arm")
    pooled = treated + controlled
    n_t = len(treated)
    total = len(pooled)
    observed_treated_var = _variance(treated)
    observed_control_var = _variance(controlled)
    # Compared by cross-multiplication rather than as a quotient, so a
    # zero-variance arm — legal, if degenerate — is an ordering rather than a
    # division by zero.
    observed_var_ratio = (
        observed_treated_var / observed_control_var
        if observed_control_var > 0.0
        else None
    )
    observed_mean_gap = abs(statistics.fmean(treated) - statistics.fmean(controlled))
    lo, hi = min(controlled), max(controlled)
    observed_inside = sum(1 for value in treated if lo <= value <= hi)

    var_ge = 0
    mean_ge = 0
    inside_le = 0
    draws = 0
    for combo in itertools.combinations(range(total), n_t):
        chosen = set(combo)
        arm = [pooled[index] for index in combo]
        rest = [pooled[index] for index in range(total) if index not in chosen]
        draws += 1
        if (
            _variance(arm) * observed_control_var
            >= observed_treated_var * _variance(rest) - 1e-30
        ):
            var_ge += 1
        if abs(statistics.fmean(arm) - statistics.fmean(rest)) >= (
            observed_mean_gap - 1e-15
        ):
            mean_ge += 1
        rest_lo, rest_hi = min(rest), max(rest)
        if sum(1 for value in arm if rest_lo <= value <= rest_hi) <= observed_inside:
            inside_le += 1
    return {
        "test": "exact permutation over every relabelling of the pooled cells",
        "n_treatment": n_t,
        "n_control": len(controlled),
        "relabellings": draws,
        "treatment_variance": observed_treated_var,
        "control_variance": observed_control_var,
        "variance_ratio": observed_var_ratio,
        "p_variance_ratio_one_sided": var_ge / draws,
        "smallest_attainable_p": 1.0 / draws,
        "mean_difference": statistics.fmean(treated) - statistics.fmean(controlled),
        "abs_mean_difference": observed_mean_gap,
        "p_mean_two_sided": mean_ge / draws,
        "cells_inside_control_observed_range": observed_inside,
        "control_observed_range": [lo, hi],
        "p_cells_inside_range_or_fewer": inside_le / draws,
    }


def _label_assignments(sizes: Sequence[int]) -> "itertools.chain[tuple[int, ...]]":
    """Every distinct assignment of group labels to positions.

    Enumerating value permutations would be wrong here: with distinct floats
    `itertools.permutations` yields `9! = 362,880` arrangements, of which only
    `9!/(3!3!3!) = 1,680` are distinct *label* assignments. The label space is
    the exchangeability null; the value space over-counts each assignment by
    `prod(size!)` and would be both slower and, if any values tie, differently
    weighted.
    """
    total = sum(int(size) for size in sizes)

    def walk(remaining: tuple[int, ...], sizes_left: Sequence[int], group: int):
        if not sizes_left:
            yield ()
            return
        head, *rest = sizes_left
        for chosen in itertools.combinations(remaining, int(head)):
            still = tuple(index for index in remaining if index not in set(chosen))
            for tail in walk(still, rest, group + 1):
                yield tuple(sorted(chosen)) + tail

    positions = tuple(range(total))
    return walk(positions, list(sizes), 0)


def exact_permutation_trend(
    values_by_group: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    """Does the metric move with `log2(c)`? Distribution-free, no control arm.

    The sharpest instrument this design has: if the graph is what moves the map,
    the metric should track the amount of missing edge mass, and the amount of
    missing edge mass is monotone in `c`. The null enumerates every distinct
    assignment of the group labels to the pooled cells (`9!/(3!3!3!) = 1,680` at
    three groups of three) and recomputes the Pearson correlation of the cell
    value with `log2(c)`. Two-sided on `|r|`.
    """
    groups = sorted(int(key) for key in values_by_group)
    if len(groups) < 3:
        raise Round0228Error("R0228 trend test needs >= 3 groups")
    predictors: list[float] = []
    outcomes: list[float] = []
    sizes: list[int] = []
    for group in groups:
        cells = [float(value) for value in values_by_group[group]]
        if len(cells) < 2:
            raise Round0228Error("R0228 trend test needs >= 2 cells per group")
        sizes.append(len(cells))
        predictors.extend([math.log2(group)] * len(cells))
        outcomes.extend(cells)

    x = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    x_centred = x - x.mean()
    x_norm = float(np.sqrt((x_centred ** 2).sum()))
    if x_norm <= 0.0:
        raise Round0228Error("R0228 trend predictor has no variation")

    def correlation(order: np.ndarray) -> float:
        centred = order - order.mean()
        norm = float(np.sqrt((centred ** 2).sum()))
        if norm <= 0.0:
            return 0.0
        return float((x_centred * centred).sum() / (x_norm * norm))

    observed = correlation(y)
    if not math.isfinite(observed):
        raise Round0228Error("R0228 trend correlation is not finite")
    extreme = 0
    draws = 0
    # The values stay put and the LABELS move: each assignment relabels which
    # positions carry which log2(c).
    for assignment in _label_assignments(sizes):
        relabelled = np.empty_like(x)
        cursor = 0
        for group_index, size in enumerate(sizes):
            for position in assignment[cursor : cursor + size]:
                relabelled[position] = math.log2(groups[group_index])
            cursor += size
        centred = relabelled - relabelled.mean()
        norm = float(np.sqrt((centred ** 2).sum()))
        draws += 1
        if norm <= 0.0:
            continue
        y_centred = y - y.mean()
        y_norm = float(np.sqrt((y_centred ** 2).sum()))
        value = float((centred * y_centred).sum() / (norm * y_norm)) if y_norm else 0.0
        if abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "test": (
            "exact permutation of cell values across the cluster counts; "
            "statistic = |Pearson r| of the cell value against log2(c)"
        ),
        "groups": groups,
        "cells_per_group": sizes,
        "distinct_arrangements": draws,
        "pearson_r_vs_log2_c": observed,
        "slope_per_doubling_of_c": slope,
        "p_two_sided": extreme / draws,
        "group_means": {
            str(group): statistics.fmean([float(v) for v in values_by_group[group]])
            for group in groups
        },
    }


def compare_to_families(
    *,
    candidate_cells: Mapping[str, Mapping[str, Mapping[str, float]]],
    exact_cells: Mapping[str, Mapping[str, float]],
    cuvs_cells: Mapping[str, Mapping[str, float]],
    tolerance_gates: Mapping[str, Any],
    candidate_purity_ratios: Mapping[str, Mapping[str, Mapping[str, float]]],
    exact_purity_ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """z, exact permutation tests, band membership, and no verdict.

    `candidate_cells` is keyed `str(clusters) -> str(seed) -> metric -> value`.
    """
    if {int(seed) for seed in exact_cells} != set(EXACT_FAMILY_SEEDS):
        raise Round0228Error(
            f"R0228 comparison needs exactly the exact family {list(EXACT_FAMILY_SEEDS)}"
        )
    if {int(seed) for seed in cuvs_cells} != set(R0223_CUVS_SEEDS):
        raise Round0228Error(
            f"R0228 comparison needs exactly R0223's seeds {list(R0223_CUVS_SEEDS)}"
        )
    if {int(key) for key in candidate_cells} != set(CLUSTER_COUNTS):
        raise Round0228Error(
            f"R0228 comparison needs exactly cluster counts {list(CLUSTER_COUNTS)}"
        )

    per_metric: dict[str, Any] = {}
    for metric in PANEL_METRICS:
        exact_values = [
            float(exact_cells[str(seed)][metric]) for seed in EXACT_FAMILY_SEEDS
        ]
        cuvs_values = [
            float(cuvs_cells[str(seed)][metric]) for seed in R0223_CUVS_SEEDS
        ]
        exact_mean, exact_sd, exact_n = _mean_sd(exact_values)
        if exact_sd <= 0.0:
            raise Round0228Error(f"R0228 exact family sd for {metric} is zero")
        exact_lo, exact_hi = min(exact_values), max(exact_values)
        gate = dict(tolerance_gates[metric])
        one_sided_floor = float(gate["one_sided_tolerance_95_95"]["floor"])
        mean_minus_2sd_floor = float(gate["mean_minus_2sd"]["floor"])

        by_clusters: dict[str, Any] = {}
        pooled_values: list[float] = []
        values_by_group: dict[int, list[float]] = {}
        for clusters in CLUSTER_COUNTS:
            values = [
                float(candidate_cells[str(clusters)][str(seed)][metric])
                for seed in SEEDS
            ]
            values_by_group[clusters] = values
            pooled_values.extend(values)
            mean, sd, n = _mean_sd(values)
            by_clusters[str(clusters)] = {
                "clusters": int(clusters),
                "n": n,
                "seeds": list(SEEDS),
                "values": values,
                "mean": mean,
                "sample_sd_ddof1": sd,
                "sd_ratio_vs_exact_family": sd / exact_sd,
                "mean_difference_vs_exact": mean - exact_mean,
                "z_of_mean_vs_exact_family": (mean - exact_mean) / exact_sd,
                "cells": {
                    str(seed): {
                        "value": float(
                            candidate_cells[str(clusters)][str(seed)][metric]
                        ),
                        "z_vs_exact_family": (
                            float(candidate_cells[str(clusters)][str(seed)][metric])
                            - exact_mean
                        )
                        / exact_sd,
                        "inside_exact_family_range": (
                            exact_lo
                            <= float(candidate_cells[str(clusters)][str(seed)][metric])
                            <= exact_hi
                        ),
                        "clears_one_sided_tolerance_floor": (
                            float(candidate_cells[str(clusters)][str(seed)][metric])
                            >= one_sided_floor
                        ),
                        "clears_registered_mean_minus_2sd_floor": (
                            float(candidate_cells[str(clusters)][str(seed)][metric])
                            >= mean_minus_2sd_floor
                        ),
                    }
                    for seed in SEEDS
                },
                "cells_inside_exact_family_range": sum(
                    1 for value in values if exact_lo <= value <= exact_hi
                ),
                "permutation_vs_exact_family": exact_permutation_two_sample(
                    values, exact_values
                ),
                "permutation_vs_r0223_cuvs": exact_permutation_two_sample(
                    values, cuvs_values
                ),
            }
        pooled_mean, pooled_sd, pooled_n = _mean_sd(pooled_values)
        per_metric[metric] = {
            "metric": metric,
            "gated": metric in GATED_METRICS,
            "descriptive_only": metric in DESCRIPTIVE_ONLY_METRICS,
            "descriptive_only_reason": (
                DENSITY_V2_STATUS if metric in DESCRIPTIVE_ONLY_METRICS else None
            ),
            "exact_family": {
                "n": exact_n,
                "seeds": list(EXACT_FAMILY_SEEDS),
                "values": exact_values,
                "mean": exact_mean,
                "sample_sd_ddof1": exact_sd,
                "min": exact_lo,
                "max": exact_hi,
            },
            "r0223_cuvs_family": {
                "n": len(cuvs_values),
                "seeds": list(R0223_CUVS_SEEDS),
                "values": cuvs_values,
                "mean": statistics.fmean(cuvs_values),
                "sample_sd_ddof1": statistics.stdev(cuvs_values),
                "z_of_mean_vs_exact_family": (
                    statistics.fmean(cuvs_values) - exact_mean
                )
                / exact_sd,
            },
            "registered_mean_minus_2sd_floor": mean_minus_2sd_floor,
            "one_sided_tolerance_floor_95_95": one_sided_floor,
            "by_clusters": by_clusters,
            "pooled_candidates": {
                "n": pooled_n,
                "mean": pooled_mean,
                "sample_sd_ddof1": pooled_sd,
                "z_of_mean_vs_exact_family": (pooled_mean - exact_mean) / exact_sd,
                "permutation_vs_exact_family": exact_permutation_two_sample(
                    pooled_values, exact_values
                ),
                "note": (
                    "pooling mixes three treatments, so its dispersion statistic "
                    "carries between-configuration heterogeneity as well as "
                    "within-configuration spread; that is a feature of this test "
                    "and it is why the trend test is reported beside it"
                ),
            },
            "trend_in_log2_c": exact_permutation_trend(values_by_group),
            "identity_bound_note": IDENTITY_BOUND_NOTE,
        }
        if metric in PURITY_METRICS:
            per_metric[metric]["unfolded_two_sided"] = _unfolded_view(
                metric=metric,
                gate=gate,
                candidate_ratios=candidate_purity_ratios,
                exact_ratios=exact_purity_ratios,
            )
    return {
        "metrics": list(PANEL_METRICS),
        "gated_metrics": list(GATED_METRICS),
        "descriptive_only_metrics": list(DESCRIPTIVE_ONLY_METRICS),
        "density_v2_status": DENSITY_V2_STATUS,
        "identity_bound_note": IDENTITY_BOUND_NOTE,
        "cluster_counts": list(CLUSTER_COUNTS),
        "seeds": list(SEEDS),
        "exact_seeds": list(EXACT_FAMILY_SEEDS),
        "r0223_cuvs_seeds": list(R0223_CUVS_SEEDS),
        "per_metric": per_metric,
        "all_gated_cells_clear_one_sided_floor": all(
            cell["clears_one_sided_tolerance_floor"]
            for metric in GATED_METRICS
            for clusters in CLUSTER_COUNTS
            for cell in per_metric[metric]["by_clusters"][str(clusters)][
                "cells"
            ].values()
        ),
        "all_gated_cells_inside_two_sided_purity_bands": all(
            cell["inside_band"]
            for metric in PURITY_METRICS
            for clusters in CLUSTER_COUNTS
            for cell in per_metric[metric]["unfolded_two_sided"]["by_clusters"][
                str(clusters)
            ]["cells"].values()
        ),
        "evidence_limits": EVIDENCE_LIMITS,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
    }


def _unfolded_view(
    *,
    metric: str,
    gate: Mapping[str, Any],
    candidate_ratios: Mapping[str, Mapping[str, Mapping[str, float]]],
    exact_ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Purity on the natural scale, two-sided, with direction stated.

    Folding `exp(-|log r|)` reflects about `r = 1.0` while the exact family
    centres at `r_bar = 1.0087` on `k256` and `0.7096` on `k1024`. The fold
    inflates |z| and destroys direction, and direction is the whole question for
    a lower-recall graph: does it over-separate or under-separate? Both the raw
    ratio and its side of 1.0 are published for every cell.
    """
    key = PURITY_RATIO_KEYS[metric]
    band = dict(gate["two_sided_log_ratio_95_95"])
    log_mean = float(band["log_ratio_mean"])
    log_sd = float(band["log_ratio_sample_sd_ddof1"])
    log_lower = float(band["log_lower"])
    log_upper = float(band["log_upper"])
    if log_sd <= 0.0:
        raise Round0228Error(f"R0228 exact family log-ratio sd for {metric} is zero")
    exact_values = [
        float(exact_ratios[str(seed)][key]) for seed in EXACT_FAMILY_SEEDS
    ]
    by_clusters: dict[str, Any] = {}
    values_by_group: dict[int, list[float]] = {}
    for clusters in CLUSTER_COUNTS:
        ratios = [
            float(candidate_ratios[str(clusters)][str(seed)][key]) for seed in SEEDS
        ]
        if any(value <= 0 for value in ratios):
            raise Round0228Error(f"R0228 {metric} ratio is not positive")
        logs = [math.log(value) for value in ratios]
        values_by_group[clusters] = logs
        by_clusters[str(clusters)] = {
            "clusters": int(clusters),
            "ratios": ratios,
            "log_ratios": logs,
            "ratio_mean": statistics.fmean(ratios),
            "log_ratio_mean": statistics.fmean(logs),
            "z_of_mean_on_log_ratio": (statistics.fmean(logs) - log_mean) / log_sd,
            "direction": (
                "over-separates (r > 1)"
                if statistics.fmean(logs) > 0.0
                else "under-separates (r < 1)"
                if statistics.fmean(logs) < 0.0
                else "matches high-D (r = 1)"
            ),
            "cells": {
                str(seed): {
                    "ratio": float(candidate_ratios[str(clusters)][str(seed)][key]),
                    "log_ratio": math.log(
                        float(candidate_ratios[str(clusters)][str(seed)][key])
                    ),
                    "z_on_log_ratio_two_sided": (
                        math.log(
                            float(candidate_ratios[str(clusters)][str(seed)][key])
                        )
                        - log_mean
                    )
                    / log_sd,
                    "direction": (
                        "over-separates (r > 1)"
                        if float(candidate_ratios[str(clusters)][str(seed)][key]) > 1.0
                        else "under-separates (r < 1)"
                        if float(candidate_ratios[str(clusters)][str(seed)][key]) < 1.0
                        else "matches high-D (r = 1)"
                    ),
                    "inside_band": (
                        log_lower
                        <= math.log(
                            float(candidate_ratios[str(clusters)][str(seed)][key])
                        )
                        <= log_upper
                    ),
                }
                for seed in SEEDS
            },
            "permutation_vs_exact_family_on_log_ratio": exact_permutation_two_sample(
                logs, [math.log(value) for value in exact_values]
            ),
        }
    return {
        "scale": "natural log of the raw purity ratio, unfolded, two-sided",
        "band_source": "review-0225-01 released two_sided_log_ratio_95_95 band",
        "k2": float(band["k2"]),
        "log_ratio_mean": log_mean,
        "log_ratio_sample_sd_ddof1": log_sd,
        "log_band": [log_lower, log_upper],
        "ratio_band": [float(band["ratio_lower"]), float(band["ratio_upper"])],
        "exact_family_ratios": exact_values,
        "exact_family_direction": (
            "over-separates (r_bar > 1)" if log_mean > 0 else "under-separates (r_bar < 1)"
        ),
        "by_clusters": by_clusters,
        "trend_in_log2_c_on_log_ratio": exact_permutation_trend(values_by_group),
        "fold_caveat": (
            "exp(-|log r|) folds about r = 1.0 while the exact family centres at "
            f"exp({log_mean}); folded z-scores are distorted and folded values "
            "cannot distinguish over- from under-separation"
        ),
    }


__all__ = [
    "ADOPTION_CLAIMED",
    "A_GRAPH_DEGREE",
    "A_INTERMEDIATE_DEGREE",
    "A_KMEANS_ITERATIONS",
    "A_KMEANS_SUBSAMPLE_ROWS",
    "A_MAX_ITERATIONS",
    "A_METRIC",
    "A_SEED",
    "A_SPILL",
    "BATCH_SIZE",
    "BUILDER_SOURCE_ROUNDS",
    "BUILD_SCHEMA",
    "CELLS",
    "CLUSTERS_BUILT_HERE",
    "CLUSTERS_FROM_R0227",
    "CLUSTER_COUNTS",
    "CLUSTER_SPILL_BUILDER",
    "COMPARISON_CAPABILITY",
    "COMPARISON_SCHEMA",
    "CROSS_CHECKED_CLUSTERS",
    "C_MIN",
    "DENSITY_V2_STATUS",
    "DESCRIPTIVE_ONLY_METRICS",
    "DIMENSION",
    "EQUIVALENCE_CLAIMED",
    "EVIDENCE_LIMITS",
    "EXACT_FAMILY_SEEDS",
    "FULL_TRANSFORM_BATCH",
    "FUZZY_LAW",
    "FUZZY_RANDOM_STATE_SEED",
    "GATED_METRICS",
    "GATE_REGISTERABLE_HERE",
    "GATE_RELEASE_CLAIMED",
    "GEOMETRY_CAPABILITY",
    "GEOMETRY_SCHEMA",
    "GRAPH_BEARING_PATHS",
    "GRAPH_CAPABILITIES",
    "GRAPH_CAPABILITY_TEMPLATE",
    "GRAPH_K",
    "GRAPH_PLACEHOLDER",
    "HOST_RSS_LIMIT_GIB",
    "IDENTITY_BOUND_NOTE",
    "MAP_CAPABILITIES",
    "MAP_CAPABILITY_TEMPLATE",
    "MIN_ADMISSIBLE_NEGATIVE_DISTANCE",
    "NEGATIVE_RNG_SEED_OFFSET",
    "OUTPUT_DIMENSION",
    "PANEL_METRICS",
    "POSITIVE_ROWS_PER_UPDATE",
    "PRODUCTION_CONFIG_SCHEMA",
    "PURITY_METRICS",
    "PURITY_RATIO_KEYS",
    "R0216_EXACT_KERNEL_MIN_DISTANCE",
    "R0216_EXACT_KERNEL_NEGATIVE_ENTRIES",
    "R0216_SEALED_DIRECTED_EDGES",
    "R0171_FLOOR_SOURCE",
    "R0217_TREATMENT_INVARIANT_SHA256",
    "R0222_GATE_SCHEMA",
    "R0223_COMPARISON_SCHEMA",
    "R0223_CUVS_SEEDS",
    "R0225_GATE_SCHEMA",
    "R0227_STRICT_RECALL_BY_C",
    "R0227_TIE_AWARE_RECALL_BY_C",
    "RECALL_CROSS_CHECK_TOLERANCE",
    "RECALL_MEAN_FLOOR",
    "RECALL_P10_FLOOR",
    "RECALL_POPULATION",
    "RECALL_POPULATION_NOTE",
    "REGISTERED_UPDATE_BOUND",
    "ROUND_ID",
    "ROWS",
    "Round0228Error",
    "SEEDS",
    "SEED_BEARING_PATHS",
    "SUBSTRATE_2M_PATH",
    "SUBSTRATE_2M_ROWS",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "TEMPLATE_SEED",
    "TRAIN_SCHEMA",
    "USE_AMP",
    "ZERO_DEGREE_TRIPWIRE",
    "assert_configuration_family",
    "compare_to_families",
    "exact_permutation_trend",
    "exact_permutation_two_sample",
    "graph_bearing_values",
    "graph_capability",
    "graph_exactness",
    "map_capability",
    "performance_windows",
    "seed_bearing_values",
    "seed_invariant_projection",
    "successful_updates_for_edges",
    "train_config",
    "treatment_invariant_sha256",
    "validate_cluster_spill_graph",
    "validate_dose",
    "validate_full_population_map",
    "validate_published_map",
]

"""Frozen contract for R0223 — does a cuVS graph cost map quality?

R0220 qualified cuVS nn-descent as a k15 *builder* on R0216's sealed 2M mixed
MiniLM substrate: `nnd-gd32-igd48-it20` reached tie-aware recall `0.994164`
against exact brute-force truth, and review-0220 independently reproduced that
number. Review-0220 also blocked the claim that such a graph preserves map
quality, and it was right to: **no map was ever trained on a cuVS graph.** A
0.994-recall graph is not evidence about a map; it is evidence about a graph.

This round trains maps on the cuVS graph and scores them on the same frozen
panel the exact-graph family was scored on. The treatment is R0217's in every
respect except the graph, and that is enforced structurally rather than
asserted: every cell config is produced by taking R0217's own `train_config`
output for the same seed and overwriting **exactly** the registered
`GRAPH_BEARING_PATHS` (plus R0217's own nine `SEED_BEARING_PATHS`). The node
then recomputes a digest over the config with both path sets masked and refuses
to train unless it equals the digest of the unmodified R0217 template. Anything
that is neither seed-derived nor graph-derived is therefore byte-identical to
the family this round is compared against.

Three things this round is careful about, because a small-n comparison invites
overreach:

1. **It registers no gate and releases none.** Review 0222-01 landed while this
   round was being written. It released three of R0222's four `mean - 2sigma`
   floors as acceptance criteria and released `purity_fidelity_k256`'s as
   *descriptive only*, because `mean - 2sigma` is **self-loosening** — an outlier
   inflates `s`, which lowers the floor, so the cell a gate should catch makes
   the gate laxer for everyone else. Every metric is therefore reported against
   **both** the registered `mean - 2sigma` floor and a one-sided 95/95 normal
   tolerance floor (`k = 3.187` at `n = 8`), side by side, and the round says
   which it treats as decisive. `GATE_REGISTERABLE_HERE` and
   `GATE_RELEASE_CLAIMED` are both `False` in every artifact.
1b. **Purity is reported unfolded.** `purity_fidelity` is `exp(-|log r|)`,
   folded about `r = 1.0`, while the exact family centres at `r_bar = 1.00863`.
   Review 0222-01 showed the fold inflates z-scores and, worse for this round,
   erases *direction*: a folded metric cannot tell over-separation from
   under-separation, and a lower-recall graph could plausibly under-separate. So
   the raw ratio, its side of `1.0`, and z on the `log r` scale are all reported.

2. **It does not claim equivalence.** Three cells cannot establish equivalence
   with an eight-cell distribution. The receipt carries `EVIDENCE_LIMITS`
   verbatim, and the comparison payload reports z-scores and band membership
   rather than a verdict.
3. **The horizon is derived, never carried.** The cuVS graph is a different
   graph, so its symmetrised fuzzy edge count differs from R0216's `48,344,648`
   and the registered R0184/R0202 `ceil` rule recomputes the horizon from the
   *sealed* count. That is the same rule R0217/R0221 ran under, applied to a
   different input, which is exactly what "same low-dose horizon rule" means.
"""
from __future__ import annotations

import copy
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
    RELOAD_PROBE_SEED,
    ROWS,
    Round0217Error,
    SEALED_DIRECTED_EDGES as R0216_SEALED_DIRECTED_EDGES,
    SEED_BEARING_PATHS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    USE_AMP,
    achieved_draws_per_edge,
    dose_quantum,
    performance_windows,
    seed_invariant_projection,
    successful_updates_for_edges,
    train_config as r0217_train_config,
    validate_dose,
    validate_published_map,
)
from .round0218_minilm_2m_panel import PANEL_METRICS
from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET


ROUND_ID = "0223"

#: The cells. Seed 42 is the minimum the round must deliver; 43 and 44 are
#: trained because one cell cannot be compared to an eight-cell distribution.
SEEDS: tuple[int, ...] = (42, 43, 44)
#: The R0217 cell whose config bytes are the template for every cell.
TEMPLATE_SEED = 42

CUVS_GRAPH_CAPABILITY = "minilm-mixed-2m-cuvs-igd48-k15-fuzzy-graph-v1"
MAP_CAPABILITY_TEMPLATE = "minilm-mixed-2m-cuvs-igd48-map-seed{seed}-low-dose-v1"
COMPARISON_CAPABILITY = "minilm-mixed-2m-cuvs-graph-map-comparison-v1"

CUVS_GRAPH_SCHEMA = "round0223-minilm-mixed-2m-cuvs-igd48-k15-fuzzy-graph-v1"
TRAIN_SCHEMA = "round0223-minilm-mixed-2m-cuvs-graph-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = "round0223-minilm-mixed-2m-cuvs-graph-production-config-v1"
COMPARISON_SCHEMA = "round0223-minilm-mixed-2m-cuvs-graph-map-comparison-v1"

#: The R0220 build this round consumes, by identity. These are the bytes whose
#: recall review-0220 independently reproduced, so the map trained here sits on
#: the graph the recall number describes rather than on a fresh, differently
#: seeded rebuild. cuVS nn-descent is not deterministic across runs (R0220's two
#: queues produced two different graph hashes for the identical setting), so
#: rebuilding would break that link.
CUVS_SETTING_ID = "nnd-gd32-igd48-it20"
CUVS_GRAPH_DEGREE = 32
CUVS_INTERMEDIATE_GRAPH_DEGREE = 48
CUVS_MAX_ITERATIONS = 20
CUVS_METRIC = "sqeuclidean"
R0220_ROUND_ID = "0220"
R0220_ARTIFACT_ROOT = (
    "/data/latent-basemap/runs/round-0220/queue-correction-1/artifacts"
)
R0220_QUALIFICATION_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": (
        f"{R0220_ARTIFACT_ROOT}/cuvs-k15-graph-builder-qualification-v1/"
        "cuvs-qualification.json"
    ),
    "bytes": 17_774,
    "sha256": "2cd756d06a51fb2334891acaf220530afb30511a05c08d326a39e897c65b0274",
}
R0220_CUVS_GRAPH_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": (
        f"{R0220_ARTIFACT_ROOT}/cuvs-k15-graph-builder-qualification-v1/builds/"
        f"{CUVS_SETTING_ID}/graph.u32.npy"
    ),
    "bytes": 256_000_128,
    "sha256": "aef9be7f8cdd80e6b14092f268545fe25b618c350ab1b580f000c18dd76c485d",
}
R0220_TRUTH_RECEIPT_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": f"{R0220_ARTIFACT_ROOT}/exact-k15-truth/truth-rebuild.json",
    "bytes": 3_306,
    "sha256": "b0cdbd0e0f5452d120e060def90bb539813c91c1057b282f4e4703b69bcf869b",
}
#: R0220's published tie-aware recall for this exact setting, over all 2,000,000
#: rows. Registered as a *cross-check*: the node re-measures recall against
#: R0220's sealed truth arrays and must land on this value, so a swapped graph
#: file with a matching hash is impossible and a silently different graph is
#: caught before a single optimizer step.
R0220_TIE_AWARE_RECALL = 0.9941635666666666
R0220_STRICT_RECALL = 0.9933069000000001
RECALL_CROSS_CHECK_TOLERANCE = 1.0e-6

#: The cuVS builder is approximate, so R0216's `0.999` exactness floor does not
#: apply and is not silently reused. The floor applied here is R0171's accepted
#: ANN qualification floor, which review-0220 noted all five cuVS settings clear
#: and which no round in this program has ever loosened.
CUVS_MEAN_RECALL_FLOOR = 0.90
CUVS_P10_RECALL_FLOOR = 0.80
R0171_FLOOR_SOURCE = "R0171 accepted ANN graph qualification floors (0.90 / 0.80)"

#: The R0215 tripwire applies to every graph build in this program, exact or not.
ZERO_DEGREE_TRIPWIRE = MAX_ZERO_DEGREE_ROWS

#: The fuzzy stage is R0216's, unchanged: `umap.umap_.fuzzy_simplicial_set` at
#: `n_neighbors=15`, `metric="cosine"`, `random_state=RandomState(42)`, fed the
#: builder's k15 ids and their `1 - cosine` distances. Only the ids change.
FUZZY_RANDOM_STATE_SEED = 42
FUZZY_LAW = (
    "umap.umap_.fuzzy_simplicial_set(n_neighbors=15, metric='cosine', "
    "random_state=RandomState(42), knn_indices=<builder k15 ids>, "
    "knn_dists=1-cosine); identical to R0216's call in every argument but the ids"
)
#: fp32 cosines of near-duplicate rows can exceed 1 by ~1 ULP, making `1 - cos`
#: very slightly negative. Those are clipped to zero, counted, and — this is the
#: check that matters — bounded in MAGNITUDE. A count cap is the wrong shape: the
#: number of ties is a property of the substrate's duplicate structure, not of
#: the builder, and R0216's own **exact** brute-force kernel produces 7,288 such
#: entries (2.43e-4 of 30,000,000, minimum -1.0728836e-06) on the same rows. What
#: a real defect would look like is a *large* negative distance, i.e. a cosine
#: that is not a cosine. The floor is set two orders above the fp32 ULP at unit
#: cosine (1.19e-07) and an order below R0216's observed exact-kernel extreme.
MIN_ADMISSIBLE_NEGATIVE_DISTANCE = -1.0e-5
R0216_EXACT_KERNEL_NEGATIVE_ENTRIES = 7_288
R0216_EXACT_KERNEL_MIN_DISTANCE = -1.0728836e-06

#: R0222's registered-not-released `n = 8` floors live in its sealed artifact and
#: are read from it, never typed here. This is the artifact identity.
R0222_GATE_ARTIFACT_ROOT = (
    "/data/latent-basemap/runs/round-0222/queue/artifacts/"
    "minilm-mixed-2m-quality-gates-n8-v1"
)
R0222_GATE_SCHEMA = "round0222-minilm-mixed-2m-quality-gate-registration-v1"
R0222_POOLED_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49)
PENDING_FLOOR_METRICS: tuple[str, ...] = tuple(PANEL_METRICS)

#: Said in every artifact this round writes.
FLOOR_STATUS = (
    "R0222 registered four mean-2sigma floors at n=8. Review 0222-01 (2026-08-08) "
    "released three of them as acceptance criteria and released "
    "purity_fidelity_k256's as DESCRIPTIVE ONLY, because the mean-2sigma "
    "estimator is self-loosening and the folded fidelity manufactures "
    "significance. This round measures against the registered floors and claims "
    "no gate release of its own."
)

#: Review 0222-01's finding, applied here as a second, non-self-loosening yard
#: stick reported side by side with the registered one. `mean - 2s` is
#: self-loosening: an outlier inflates `s`, which lowers the floor, so the cell a
#: gate should catch makes the gate laxer for everyone else (seed 48 widened the
#: k256 admissible band by 73%). A one-sided normal tolerance factor for 95%
#: content at 95% confidence does not have that property at fixed `n`. The factor
#: is *computed* here from the noncentral-t definition rather than typed, and
#: cross-checked against the reviewer's published `3.187`.
TOLERANCE_CONTENT = 0.95
TOLERANCE_CONFIDENCE = 0.95
REVIEW_0222_TOLERANCE_FACTOR_N8 = 3.187
TOLERANCE_FACTOR_CROSS_CHECK = 1.0e-3
TOLERANCE_FLOOR_SOURCE = (
    "review-0222-2026-08-08-01.md: one-sided normal tolerance factor for 95% "
    "content at 95% confidence, k = nct.ppf(0.95, n-1, z_0.95*sqrt(n))/sqrt(n)"
)

#: Purity fidelity is `exp(-|log r|)`, folded about `r = 1.0`. Review 0222-01
#: showed the family centres at `r_bar = 1.00863`, not at 1.0, so the fold
#: reflects deviations about the wrong point, inflates |z| (seed 48: `2.159`
#: folded against `+1.921` unfolded) and destroys the *direction* of any
#: difference. For a cuVS-versus-exact graph comparison the direction is the
#: whole finding — a lower-recall graph could plausibly under-separate — so
#: purity is additionally reported on the unfolded `log r` scale, with the raw
#: ratio and its side of 1.0.
PURITY_METRICS: tuple[str, ...] = ("purity_fidelity_k256", "purity_fidelity_k1024")
PURITY_RATIO_KEYS: dict[str, str] = {
    "purity_fidelity_k256": "k256",
    "purity_fidelity_k1024": "k1024",
}
UNFOLDED_SCALE_NOTE = (
    "z on log r, the natural scale: the folded fidelity reflects about r = 1.0 "
    "while the exact family centres at r_bar > 1.0, so folded z-scores are "
    "distorted (review-0222-01). Direction is reported explicitly because a "
    "folded metric cannot distinguish over-separation from under-separation."
)
GATE_REGISTERABLE_HERE = False
GATE_RELEASE_CLAIMED = False

EVIDENCE_LIMITS = (
    "This round trains n=3 cuVS-graph cells and compares them to an n=8 "
    "exact-graph family. Three cells inside the exact family's band is "
    "CONSISTENT WITH equivalence and is not proof of it: at n=3 the sample sd "
    "is a three-point estimate, and the comparison has little power against "
    "differences smaller than roughly one exact-family sd. A cell landing "
    "inside the band licenses 'no difference was detected at this n', never "
    "'the graphs are equivalent'. A cell landing outside the band, or below a "
    "pending floor, is the stronger direction of evidence and is reported as "
    "such."
)

#: Config paths whose value is a function of the graph. Everything outside these
#: and R0217's nine seed-bearing paths must be byte-identical to R0217's cell.
#: Each entry carries why it must move; a path with no reason does not belong.
GRAPH_BEARING_PATHS: tuple[tuple[str, ...], ...] = (
    ("graph", "capability"),
    ("graph", "source_round"),
    ("graph", "path"),
    ("graph", "sha256"),
    ("graph", "manifest_path"),
    ("graph", "manifest_sha256"),
    ("graph", "directed_edges"),
    ("graph", "exactness"),
    ("family_invariant", "graph_policy"),
    ("optimizer", "successful_positive_lr_updates"),
    ("execution", "expected_pipeline_stamp", "valid_canonical_edge_count"),
    ("execution", "performance_windows"),
    ("execution", "achieved_positive_draws_per_edge"),
    ("execution", "scale_change"),
    ("dose_registration",),
)
GRAPH_BEARING_REASONS: dict[str, str] = {
    "graph.capability": "names the cuVS graph capability instead of R0216's",
    "graph.source_round": "the graph is produced by R0223, not R0216",
    "graph.path": "a different edge file",
    "graph.sha256": "a different edge file",
    "graph.manifest_path": "a different sealed graph receipt",
    "graph.manifest_sha256": "a different sealed graph receipt",
    "graph.directed_edges": "the symmetrised fuzzy edge count of a different graph",
    "graph.exactness": (
        "the builder is approximate cuVS nn-descent, not brute force; leaving "
        "R0216's wording would be false in the treatment bytes"
    ),
    "family_invariant.graph_policy": "states which graph every cell shares",
    "optimizer.successful_positive_lr_updates": (
        "the registered ceil horizon applied to this graph's edge count"
    ),
    "execution.expected_pipeline_stamp.valid_canonical_edge_count": (
        "the sampler stamps the active edge count it actually loaded"
    ),
    "execution.performance_windows": "a pure function of the horizon",
    "execution.achieved_positive_draws_per_edge": "a pure function of the horizon",
    "execution.scale_change": "describes which graph this rung runs on",
    "dose_registration": "the whole derived dose payload moves with the horizon",
}
#: `execution.expected_pipeline_stamp.positive_destination_policy` is NOT in the
#: list above and is deliberately left at R0217's constant
#: `R0216-exact-k15-fuzzy-tconorm-graph`. It is emitted by the sampler class in
#: the release checkout, which this round may not modify, and it labels the
#: *sampling policy* (fuzzy t-conorm weights over a k15 topology), not the file.
#: The graph file identity travels in the same stamp's `graph` signatures, which
#: do point at the cuVS artifacts. Stated here so a reviewer meets it in the
#: contract rather than discovering it in a receipt.
PIPELINE_STAMP_LABEL_CARRYOVER = (
    "execution.expected_pipeline_stamp.positive_destination_policy stays at "
    "R0217's 'R0216-exact-k15-fuzzy-tconorm-graph' because the sampler class in "
    "the frozen release emits that constant; the bound graph signatures in the "
    "same stamp identify the cuVS artifacts"
)

SEED_PLACEHOLDER = "<seed-bearing>"
GRAPH_PLACEHOLDER = "<graph-bearing>"

#: Refuse to launch a horizon larger than this (R0217's registered bound).
REGISTERED_UPDATE_BOUND = 120_000

FULL_TRANSFORM_BATCH = 8_192


class Round0223Error(RuntimeError):
    """The registered R0223 cuVS-graph map contract changed."""


def map_capability(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0223Error(f"R0223 seed {seed!r} is not a registered cell")
    return MAP_CAPABILITY_TEMPLATE.format(seed=int(seed))


MAP_CAPABILITIES: tuple[str, ...] = tuple(
    MAP_CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0223Error(f"R0223 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0223Error(f"R0223 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0223Error(f"R0223 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def treatment_invariant_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config with every seed-derived AND graph-derived value masked out."""
    projected = seed_invariant_projection(config)
    for path in GRAPH_BEARING_PATHS:
        _set_path(projected, path, GRAPH_PLACEHOLDER)
    return projected


def treatment_invariant_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(treatment_invariant_projection(config)))


def seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    """What each of R0217's nine seed-bearing fields must hold for this cell."""
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
        raise Round0223Error(
            "R0223 seed-bearing path set differs from R0217's registered set"
        )
    return values


def graph_bearing_values(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    dose: Mapping[str, Any],
    updates: int,
) -> dict[tuple[str, ...], Any]:
    """What each graph-derived field must hold for the cuVS graph."""
    values: dict[tuple[str, ...], Any] = {
        ("graph", "capability"): CUVS_GRAPH_CAPABILITY,
        ("graph", "source_round"): ROUND_ID,
        ("graph", "path"): str(graph_signature["canonical_path"]),
        ("graph", "sha256"): str(graph_signature["sha256"]),
        ("graph", "manifest_path"): str(graph_manifest_signature["canonical_path"]),
        ("graph", "manifest_sha256"): str(graph_manifest_signature["sha256"]),
        ("graph", "directed_edges"): int(graph_edges),
        ("graph", "exactness"): (
            f"approximate cuVS nn-descent {CUVS_SETTING_ID} "
            f"(graph_degree {CUVS_GRAPH_DEGREE}, intermediate_graph_degree "
            f"{CUVS_INTERMEDIATE_GRAPH_DEGREE}, max_iterations "
            f"{CUVS_MAX_ITERATIONS}, metric {CUVS_METRIC}); never quantized, "
            "never brute force"
        ),
        ("family_invariant", "graph_policy"): (
            "byte-identical R0223 cuVS igd48 k15 fuzzy graph in every cell"
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
            "substrate with the graph replaced by the R0220-qualified cuVS "
            f"nn-descent {CUVS_SETTING_ID} k15 topology; recipe, precision, "
            "sampler, optimizer, residency and dose rule unchanged"
        ),
        ("dose_registration",): dict(dose),
    }
    if set(values) != set(GRAPH_BEARING_PATHS):
        raise Round0223Error("R0223 graph-bearing path set drifted from the register")
    return values


def train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    r0216_graph_signature: Mapping[str, Any],
    r0216_graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str, str]:
    """R0217's config with the graph swapped, and nothing else moved.

    Returns `(config, config_sha256, treatment_invariant_sha256)`. The template
    is built by R0217's own `train_config` against R0216's sealed signatures —
    the same call R0217 and R0221 made — and this function then overwrites
    exactly `SEED_BEARING_PATHS | GRAPH_BEARING_PATHS`. Equality of the
    treatment-invariant digest with the template's is what makes "identical
    except the graph" a check rather than a claim.
    """
    if int(seed) not in SEEDS:
        raise Round0223Error(f"R0223 seed {seed!r} is not a registered cell")
    if int(rows) != ROWS:
        raise Round0223Error("R0223 population cardinality changed")
    if int(graph_edges) <= 0:
        raise Round0223Error("R0223 cuVS graph has no directed edges")
    updates = successful_updates_for_edges(int(graph_edges))
    if updates > REGISTERED_UPDATE_BOUND:
        raise Round0223Error(
            f"R0223 derived horizon {updates} exceeds the registered bound "
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
        raise Round0223Error(
            "R0223 cell is not R0217's treatment outside the seed and the "
            f"graph: {invariant} != {template_invariant}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0223Error("R0223 horizon did not reach the train config")
    return config, sha256_bytes(canonical_json(config)), invariant


def assert_family_differs_only_by_seed(
    configs: Mapping[int, Mapping[str, Any]],
    *,
    expected_treatment_invariant: str,
) -> dict[str, Any]:
    """Fail closed unless the cells differ in the seed and nothing else."""
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0223Error(
            f"R0223 family must be exactly seeds {list(SEEDS)}, got {sorted(observed)}"
        )
    from .round0217_minilm_2m_seed_family import seed_invariant_sha256

    seed_digests: dict[int, str] = {}
    treatment_digests: dict[int, str] = {}
    per_seed: dict[str, str] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        for path, want in seed_bearing_values(seed).items():
            got = _get_path(config, path)
            if got != want:
                raise Round0223Error(
                    f"R0223 seed {seed} cell has {'.'.join(path)}={got!r}, "
                    f"expected {want!r}"
                )
        seed_digests[seed] = seed_invariant_sha256(config)
        treatment_digests[seed] = treatment_invariant_sha256(config)
        per_seed[str(seed)] = sha256_bytes(canonical_json(config))
    if len(set(seed_digests.values())) != 1:
        raise Round0223Error(
            f"R0223 cells differ outside the seed: {seed_digests}"
        )
    if set(treatment_digests.values()) != {expected_treatment_invariant}:
        raise Round0223Error(
            "R0223 cells are not R0217's treatment outside the graph: "
            f"{sorted(set(treatment_digests.values()))} != "
            f"{expected_treatment_invariant}"
        )
    if len(set(per_seed.values())) != len(SEEDS):
        raise Round0223Error("R0223 produced duplicate cell configs")
    return {
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": sorted(set(seed_digests.values()))[0],
        "treatment_invariant_sha256": expected_treatment_invariant,
        "per_seed_config_sha256": per_seed,
        "graph_bearing_paths": [".".join(path) for path in GRAPH_BEARING_PATHS],
        "graph_bearing_reasons": dict(GRAPH_BEARING_REASONS),
        "pipeline_stamp_label_carryover": PIPELINE_STAMP_LABEL_CARRYOVER,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    }


def validate_cuvs_graph(
    *,
    degrees: Mapping[str, Any],
    recall: Mapping[str, float],
    edges: int,
    structural: Mapping[str, int],
) -> dict[str, Any]:
    """R0171's ANN floors plus the R0215 zero-degree tripwire, fail-closed."""
    mean = float(recall["mean_recall_at_k"])
    p10 = float(recall["p10_recall_at_k"])
    if not math.isfinite(mean) or not math.isfinite(p10):
        raise Round0223Error("R0223 cuVS recall is not finite")
    if mean < CUVS_MEAN_RECALL_FLOOR or p10 < CUVS_P10_RECALL_FLOOR:
        raise Round0223Error(
            f"R0223 cuVS graph recall {mean:.6f}/{p10:.6f} is below the "
            f"{CUVS_MEAN_RECALL_FLOOR}/{CUVS_P10_RECALL_FLOOR} floors "
            f"({R0171_FLOOR_SOURCE})"
        )
    for key in ("self_loop_entries", "duplicate_entries", "out_of_range_entries"):
        if int(structural.get(key, -1)) != 0:
            raise Round0223Error(
                f"R0223 cuVS k15 slice has {structural.get(key)} {key}; a graph "
                "with self loops, duplicates or out-of-range ids is not a k15 graph"
            )
    if int(structural.get("rows_below_k", -1)) != 0:
        raise Round0223Error("R0223 cuVS k15 slice has rows with fewer than k ids")
    zero = int(degrees["zero_degree_rows"])
    if zero > ZERO_DEGREE_TRIPWIRE:
        raise Round0223Error(
            f"{zero} rows have zero edges. R0215 showed this is exactly what "
            "produced the v1 150M clumps; the rung does not proceed with "
            "edgeless rows."
        )
    if int(edges) <= 0:
        raise Round0223Error("R0223 cuVS fuzzy graph has no directed edges")
    return {
        "mean_recall_at_k": mean,
        "p10_recall_at_k": p10,
        "mean_recall_floor": CUVS_MEAN_RECALL_FLOOR,
        "p10_recall_floor": CUVS_P10_RECALL_FLOOR,
        "recall_floor_source": R0171_FLOOR_SOURCE,
        "zero_degree_rows": zero,
        "zero_degree_tripwire": ZERO_DEGREE_TRIPWIRE,
        "directed_edges": int(edges),
        "structural": dict(structural),
        "exactness": (
            f"approximate cuVS nn-descent {CUVS_SETTING_ID}; recall measured "
            "against R0220's sealed exact brute-force truth over all rows"
        ),
    }


def validate_full_population_map(coordinates: Any) -> dict[str, Any]:
    """Every one of the 2,000,000 rows must project to a finite coordinate."""
    array = np.asarray(coordinates)
    if array.shape != (ROWS, OUTPUT_DIMENSION):
        raise Round0223Error(
            f"R0223 full-population transform produced {array.shape}, expected "
            f"({ROWS}, {OUTPUT_DIMENSION})"
        )
    finite = int(np.isfinite(array).all(axis=1).sum())
    if finite != ROWS:
        raise Round0223Error(
            f"R0223 full-population transform has {ROWS - finite} nonfinite rows"
        )
    published = validate_published_map(array)
    return {
        **published,
        "transform_rows": ROWS,
        "transform_rows_finite": finite,
        "full_population_finite": True,
    }


def _mean_sd(values: Sequence[float]) -> tuple[float, float, int]:
    numbers = [float(value) for value in values]
    if len(numbers) < 2 or any(not math.isfinite(value) for value in numbers):
        raise Round0223Error("R0223 summary needs >= 2 finite values")
    return statistics.fmean(numbers), statistics.stdev(numbers), len(numbers)


def tolerance_factor(n: int) -> dict[str, Any]:
    """One-sided normal tolerance factor, 95% content at 95% confidence.

    Computed, not typed. Cross-checked against review-0222-01's published
    `3.187` at `n = 8`; a drift beyond `TOLERANCE_FACTOR_CROSS_CHECK` aborts,
    because a silently different factor would silently move every floor.
    """
    if int(n) < 3:
        raise Round0223Error("R0223 tolerance factor needs n >= 3")
    from scipy import stats

    z_p = float(stats.norm.ppf(TOLERANCE_CONTENT))
    noncentrality = z_p * math.sqrt(int(n))
    k = float(stats.nct.ppf(TOLERANCE_CONFIDENCE, int(n) - 1, noncentrality)) / math.sqrt(
        int(n)
    )
    if not math.isfinite(k) or k <= 0:
        raise Round0223Error("R0223 tolerance factor is not a finite positive number")
    payload = {
        "n": int(n),
        "content": TOLERANCE_CONTENT,
        "confidence": TOLERANCE_CONFIDENCE,
        "k": k,
        "definition": TOLERANCE_FLOOR_SOURCE,
        "review_0222_published_k_at_n8": REVIEW_0222_TOLERANCE_FACTOR_N8,
    }
    if int(n) == 8:
        delta = abs(k - REVIEW_0222_TOLERANCE_FACTOR_N8)
        if delta > TOLERANCE_FACTOR_CROSS_CHECK:
            raise Round0223Error(
                f"R0223 computed tolerance factor {k!r} disagrees with "
                f"review-0222-01's {REVIEW_0222_TOLERANCE_FACTOR_N8} by {delta:.3e}"
            )
        payload["cross_check_delta"] = delta
        payload["reproduces_review_0222"] = True
    return payload


def _log_ratio_view(
    *,
    metric: str,
    cuvs_ratios: Mapping[str, float],
    exact_ratios: Mapping[str, float],
) -> dict[str, Any]:
    """Purity on its natural, unfolded scale, with direction stated."""
    exact_values = [float(exact_ratios[str(seed)]) for seed in R0222_POOLED_SEEDS]
    cuvs_values = [float(cuvs_ratios[str(seed)]) for seed in SEEDS]
    if any(value <= 0 for value in exact_values + cuvs_values):
        raise Round0223Error(f"R0223 {metric} ratio is not positive")
    exact_logs = [math.log(value) for value in exact_values]
    cuvs_logs = [math.log(value) for value in cuvs_values]
    log_mean, log_sd, _n = _mean_sd(exact_logs)
    if log_sd <= 0.0:
        raise Round0223Error(f"R0223 exact family log-ratio sd for {metric} is zero")
    return {
        "scale": "log r (unfolded)",
        "note": UNFOLDED_SCALE_NOTE,
        "exact_family": {
            "n": len(exact_values),
            "seeds": list(R0222_POOLED_SEEDS),
            "ratios": exact_values,
            "ratio_mean": statistics.fmean(exact_values),
            "log_ratio_mean": log_mean,
            "log_ratio_sample_sd_ddof1": log_sd,
            "centre_is_above_one": log_mean > 0.0,
        },
        "cuvs_family": {
            "n": len(cuvs_values),
            "seeds": list(SEEDS),
            "ratios": cuvs_values,
            "ratio_mean": statistics.fmean(cuvs_values),
            "log_ratio_mean": statistics.fmean(cuvs_logs),
        },
        "cells": {
            str(seed): {
                "ratio": float(cuvs_ratios[str(seed)]),
                "log_ratio": math.log(float(cuvs_ratios[str(seed)])),
                "direction": (
                    "over-separates (r > 1)"
                    if float(cuvs_ratios[str(seed)]) > 1.0
                    else "under-separates (r < 1)"
                    if float(cuvs_ratios[str(seed)]) < 1.0
                    else "matches high-D (r = 1)"
                ),
                "z_on_log_ratio_vs_exact_family": (
                    math.log(float(cuvs_ratios[str(seed)])) - log_mean
                )
                / log_sd,
                "inside_exact_family_ratio_range": (
                    min(exact_values)
                    <= float(cuvs_ratios[str(seed)])
                    <= max(exact_values)
                ),
            }
            for seed in SEEDS
        },
        "mean_log_ratio_difference_cuvs_minus_exact": (
            statistics.fmean(cuvs_logs) - log_mean
        ),
        "z_of_cuvs_mean_on_log_ratio": (
            statistics.fmean(cuvs_logs) - log_mean
        )
        / log_sd,
    }


def compare_to_exact_family(
    *,
    cuvs_cells: Mapping[str, Mapping[str, float]],
    exact_cells: Mapping[str, Mapping[str, float]],
    pending_floors: Mapping[str, float],
    cuvs_purity_ratios: Mapping[str, Mapping[str, float]],
    exact_purity_ratios: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """z-scores against the exact family, and pass/fail against pending floors.

    Deliberately returns no verdict. It returns the arithmetic a reader needs
    and the sentence describing what that arithmetic can support.
    """
    if {int(seed) for seed in exact_cells} != set(R0222_POOLED_SEEDS):
        raise Round0223Error(
            f"R0223 comparison needs exactly the R0222 pooled family "
            f"{list(R0222_POOLED_SEEDS)}"
        )
    if {int(seed) for seed in cuvs_cells} != set(SEEDS):
        raise Round0223Error(f"R0223 comparison needs exactly seeds {list(SEEDS)}")
    if set(pending_floors) != set(PENDING_FLOOR_METRICS):
        raise Round0223Error(
            f"R0223 pending floors must cover exactly {list(PENDING_FLOOR_METRICS)}"
        )

    factor = tolerance_factor(len(R0222_POOLED_SEEDS))
    per_metric: dict[str, Any] = {}
    for metric in PANEL_METRICS:
        exact_values = [
            float(exact_cells[str(seed)][metric]) for seed in R0222_POOLED_SEEDS
        ]
        cuvs_values = [float(cuvs_cells[str(seed)][metric]) for seed in SEEDS]
        exact_mean, exact_sd, exact_n = _mean_sd(exact_values)
        cuvs_mean, cuvs_sd, cuvs_n = _mean_sd(cuvs_values)
        if exact_sd <= 0.0:
            raise Round0223Error(
                f"R0223 exact family sd for {metric} is zero; z-scores undefined"
            )
        floor = float(pending_floors[metric])
        tolerance_floor = exact_mean - float(factor["k"]) * exact_sd
        per_cell = {
            str(seed): {
                "value": float(cuvs_cells[str(seed)][metric]),
                "z_vs_exact_family": (
                    float(cuvs_cells[str(seed)][metric]) - exact_mean
                )
                / exact_sd,
                "inside_exact_family_range": (
                    min(exact_values)
                    <= float(cuvs_cells[str(seed)][metric])
                    <= max(exact_values)
                ),
                "clears_pending_floor": (
                    float(cuvs_cells[str(seed)][metric]) >= floor
                ),
                "clears_tolerance_floor": (
                    float(cuvs_cells[str(seed)][metric]) >= tolerance_floor
                ),
            }
            for seed in SEEDS
        }
        per_metric[metric] = {
            "registered_mean_minus_2sd_floor": floor,
            "tolerance_floor_95_95": tolerance_floor,
            "tolerance_factor": dict(factor),
            "floor_families_agree": all(
                cell["clears_pending_floor"] == cell["clears_tolerance_floor"]
                for cell in per_cell.values()
            ),
            "cells_clearing_tolerance_floor": sum(
                1 for cell in per_cell.values() if cell["clears_tolerance_floor"]
            ),
            "all_cells_clear_tolerance_floor": all(
                cell["clears_tolerance_floor"] for cell in per_cell.values()
            ),
            "exact_family": {
                "n": exact_n,
                "seeds": list(R0222_POOLED_SEEDS),
                "values": exact_values,
                "mean": exact_mean,
                "sample_sd_ddof1": exact_sd,
                "min": min(exact_values),
                "max": max(exact_values),
            },
            "cuvs_family": {
                "n": cuvs_n,
                "seeds": list(SEEDS),
                "values": cuvs_values,
                "mean": cuvs_mean,
                "sample_sd_ddof1": cuvs_sd,
                "min": min(cuvs_values),
                "max": max(cuvs_values),
            },
            "mean_difference_cuvs_minus_exact": cuvs_mean - exact_mean,
            "z_of_cuvs_mean_vs_exact_family": (cuvs_mean - exact_mean) / exact_sd,
            "cells": per_cell,
            "cells_inside_exact_family_range": sum(
                1 for cell in per_cell.values() if cell["inside_exact_family_range"]
            ),
            "pending_floor": floor,
            "pending_floor_status": FLOOR_STATUS,
            "cells_clearing_pending_floor": sum(
                1 for cell in per_cell.values() if cell["clears_pending_floor"]
            ),
            "all_cells_clear_pending_floor": all(
                cell["clears_pending_floor"] for cell in per_cell.values()
            ),
        }
        if metric in PURITY_METRICS:
            per_metric[metric]["unfolded"] = _log_ratio_view(
                metric=metric,
                cuvs_ratios={
                    str(seed): float(
                        cuvs_purity_ratios[str(seed)][PURITY_RATIO_KEYS[metric]]
                    )
                    for seed in SEEDS
                },
                exact_ratios={
                    str(seed): float(
                        exact_purity_ratios[str(seed)][PURITY_RATIO_KEYS[metric]]
                    )
                    for seed in R0222_POOLED_SEEDS
                },
            )
            per_metric[metric]["folded_scale_caveat"] = UNFOLDED_SCALE_NOTE
    return {
        "metrics": list(PANEL_METRICS),
        "per_metric": per_metric,
        "cuvs_seeds": list(SEEDS),
        "exact_seeds": list(R0222_POOLED_SEEDS),
        "tolerance_factor": dict(factor),
        "tolerance_floor_source": TOLERANCE_FLOOR_SOURCE,
        "purity_metrics_reported_unfolded": list(PURITY_METRICS),
        "unfolded_scale_note": UNFOLDED_SCALE_NOTE,
        "all_metrics_clear_tolerance_floors": all(
            per_metric[metric]["all_cells_clear_tolerance_floor"]
            for metric in PANEL_METRICS
        ),
        "floor_families_agree_on_every_metric": all(
            per_metric[metric]["floor_families_agree"] for metric in PANEL_METRICS
        ),
        "pending_floor_status": FLOOR_STATUS,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "equivalence_claimed": False,
        "evidence_limits": EVIDENCE_LIMITS,
        "all_metrics_clear_pending_floors": all(
            per_metric[metric]["all_cells_clear_pending_floor"]
            for metric in PANEL_METRICS
        ),
    }


__all__ = [
    "BATCH_SIZE",
    "COMPARISON_CAPABILITY",
    "COMPARISON_SCHEMA",
    "CUVS_GRAPH_CAPABILITY",
    "CUVS_GRAPH_DEGREE",
    "CUVS_GRAPH_SCHEMA",
    "CUVS_INTERMEDIATE_GRAPH_DEGREE",
    "CUVS_MAX_ITERATIONS",
    "CUVS_MEAN_RECALL_FLOOR",
    "CUVS_METRIC",
    "CUVS_P10_RECALL_FLOOR",
    "CUVS_SETTING_ID",
    "DIMENSION",
    "EVIDENCE_LIMITS",
    "FLOOR_STATUS",
    "FULL_TRANSFORM_BATCH",
    "FUZZY_LAW",
    "FUZZY_RANDOM_STATE_SEED",
    "GATE_REGISTERABLE_HERE",
    "GATE_RELEASE_CLAIMED",
    "GRAPH_BEARING_PATHS",
    "GRAPH_BEARING_REASONS",
    "GRAPH_K",
    "HOST_RSS_LIMIT_GIB",
    "MAP_CAPABILITIES",
    "MAP_CAPABILITY_TEMPLATE",
    "MIN_ADMISSIBLE_NEGATIVE_DISTANCE",
    "R0216_EXACT_KERNEL_MIN_DISTANCE",
    "R0216_EXACT_KERNEL_NEGATIVE_ENTRIES",
    "NEGATIVE_RNG_SEED_OFFSET",
    "OUTPUT_DIMENSION",
    "PENDING_FLOOR_METRICS",
    "PIPELINE_STAMP_LABEL_CARRYOVER",
    "PURITY_METRICS",
    "PURITY_RATIO_KEYS",
    "POSITIVE_ROWS_PER_UPDATE",
    "PRODUCTION_CONFIG_SCHEMA",
    "R0216_SEALED_DIRECTED_EDGES",
    "R0171_FLOOR_SOURCE",
    "R0220_ARTIFACT_ROOT",
    "R0220_CUVS_GRAPH_SIGNATURE",
    "R0220_QUALIFICATION_SIGNATURE",
    "R0220_ROUND_ID",
    "R0220_STRICT_RECALL",
    "R0220_TIE_AWARE_RECALL",
    "R0222_GATE_ARTIFACT_ROOT",
    "R0222_GATE_SCHEMA",
    "R0222_POOLED_SEEDS",
    "RECALL_CROSS_CHECK_TOLERANCE",
    "REGISTERED_UPDATE_BOUND",
    "RELOAD_PROBE_SEED",
    "ROUND_ID",
    "ROWS",
    "Round0217Error",
    "Round0223Error",
    "SEEDS",
    "SEED_BEARING_PATHS",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "REVIEW_0222_TOLERANCE_FACTOR_N8",
    "TEMPLATE_SEED",
    "TOLERANCE_CONFIDENCE",
    "TOLERANCE_CONTENT",
    "TOLERANCE_FACTOR_CROSS_CHECK",
    "TOLERANCE_FLOOR_SOURCE",
    "TRAIN_SCHEMA",
    "UNFOLDED_SCALE_NOTE",
    "USE_AMP",
    "ZERO_DEGREE_TRIPWIRE",
    "achieved_draws_per_edge",
    "assert_family_differs_only_by_seed",
    "compare_to_exact_family",
    "dose_quantum",
    "graph_bearing_values",
    "map_capability",
    "performance_windows",
    "seed_bearing_values",
    "successful_updates_for_edges",
    "tolerance_factor",
    "train_config",
    "treatment_invariant_sha256",
    "validate_cuvs_graph",
    "validate_dose",
    "validate_full_population_map",
    "validate_published_map",
]

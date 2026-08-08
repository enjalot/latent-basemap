"""Frozen contract for the R0221 MiniLM 2M seed extension (seeds 46-49).

Why this round exists, in one sentence: **a mean - 2 sigma gate at n = 4 cannot
be tested by its own defining cells.** For any sample of size `n` the identity
`max_i |x_i - xbar| / s <= (n-1)/sqrt(n)` holds; at `n = 4` that bound is exactly
`1.5`, so a `mean - 2 sigma` floor is unreachable from below by any cell that
helped define it, for *any* four numbers whatsoever (Review 0219). R0219's
"4/4 defining cells clear each floor" is therefore a theorem, not evidence. The
same review measured the price: leave-one-out floors moved `2.11-2.49%`, and the
`purity_fidelity_k256` floor moved by **more than its own 2 sigma band** when a
single seed was dropped. `plan-minilm-100m-v2.md` now carries the binding rule
*"register gates from >= 8 seeds, and state n next to every floor."*

This round supplies the missing four cells. It trains seeds `46/47/48/49` under a
treatment that is **byte-identical to R0217's in every field except the seed**,
so the eight cells pool into one family rather than two families that happen to
share a substrate.

Byte-identity is not asserted in prose here; it is enforced structurally. Every
R0221 config is produced by taking R0217's own `train_config` output for the
canonical cell and overwriting **exactly** the nine paths R0217 registered as
seed-bearing (`SEED_BEARING_PATHS`) with this cell's seed-derived values. Nothing
else is touched, and the node then recomputes R0217's seed-invariant digest and
refuses to train unless it equals R0217's **published** value
`241c3f6d6369e311c8e1e649bd1e8894d8cfa51c17a200c0c6f35746aa04af47`
(result-0217-2026-08-08, all four cells).

Two consequences of that construction deserve to be stated rather than
discovered by a reviewer:

1. The trained config carries `round_id: "0217"`, R0217's config schema, and
   `seed_family.seeds: [42, 43, 44, 45]`. Those are **treatment bytes**, and
   changing any of them — even cosmetically, even to say something truer about
   the extended family — would break the digest equality that makes pooling
   legitimate. The round that *ran* the cell is 0221, and that is recorded in
   the receipt (`round_id`, `treatment_config_round_id`, `pooled_seed_family`),
   which is R0221's own artifact and not part of the treatment.
2. The horizon is still derived, never carried. The node re-reads
   `directed_edge_count` from R0216's sealed `queue-correction-3` receipt and
   re-applies the registered R0184/R0202 rule, exactly as R0217 did, and asserts
   the achieved dose lands on the registered ceil-derived value
   (`80,163` updates, `0.6781860734615339` draws/edge at `48,344,648` edges).

This round adds one check R0217 did not have, because R0222 needs it: the
published checkpoint is reloaded from disk and used to project **all 2,000,000
substrate rows**, and every coordinate must be finite. R0217 probed 4,096 rows;
a panel scores all of them.

This round registers **no gate**. R0222 does, at n = 8, over the accepted
R0161/R0193 metric set.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from .round0217_minilm_2m_seed_family import (
    BATCH_SIZE,
    CAPABILITY_TEMPLATE,
    DIMENSION,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    HOST_RSS_LIMIT_GIB,
    OUTPUT_DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    RELOAD_PROBE_SEED,
    ROWS,
    Round0217Error,
    SEALED_DIRECTED_EDGES,
    SEED_BEARING_PATHS,
    SEEDS as R0217_SEEDS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    USE_AMP,
    achieved_draws_per_edge,
    dose_quantum,
    performance_windows,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config as r0217_train_config,
    validate_dose,
    validate_published_map,
)


ROUND_ID = "0221"

#: The four new cells.
SEEDS: tuple[int, ...] = (46, 47, 48, 49)

#: The R0217 cell whose config bytes are reproduced verbatim outside the seed.
TEMPLATE_SEED = 42

#: The eight-cell family R0222 pools. R0217's four plus R0221's four.
POOLED_SEEDS: tuple[int, ...] = tuple(R0217_SEEDS) + SEEDS

#: R0217's **published** seed-invariant config digest (result-0217-2026-08-08:
#: all four cells carry this one value). Every R0221 cell must reproduce it, or
#: the eight cells are not one family and must not be pooled.
R0217_SEED_INVARIANT_SHA256 = (
    "241c3f6d6369e311c8e1e649bd1e8894d8cfa51c17a200c0c6f35746aa04af47"
)

#: R0221 artifacts. The *treatment* keeps R0217's config schema; only the
#: round's own receipt and published-config wrapper are new schemas.
TRAIN_SCHEMA = "round0221-minilm-mixed-2m-seed-extension-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0221-minilm-mixed-2m-seed-extension-production-config-v1"
)

#: The registered horizon at R0216 queue-correction-3's sealed edge count. This
#: is a *cross-check* on the derivation, not a substitute for it: the node still
#: computes `ceil(1e6 * E / 603,086,368)` from the sealed receipt and this value
#: is what that computation must produce.
REGISTERED_SUCCESSFUL_UPDATES = 80_163
REGISTERED_ACHIEVED_DRAWS_PER_EDGE = 0.6781860734615339

#: Refuse to launch a horizon larger than this (R0217's registered bound).
REGISTERED_UPDATE_BOUND = 120_000

#: The three sealed R0216 `queue-correction-3` signatures R0217 trained on, as
#: published in result-0217's input table. Registering them here means an R0221
#: cell can only be built on those exact bytes — a substrate swap fails before
#: the seed-invariant digest is even computed, rather than after.
SEALED_ARTIFACT_ROOT = (
    "/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
    "minilm-mixed-2m-substrate-and-exact-k15-graph-v1"
)
SEALED_SUBSTRATE_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": f"{SEALED_ARTIFACT_ROOT}/substrate.f32.npy",
    "bytes": 3_072_000_128,
    "sha256": "372fbec511c0e9fa3b8e141529ecccaad975e469fe7b8296c019698b340b3660",
}
SEALED_GRAPH_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": f"{SEALED_ARTIFACT_ROOT}/edges-k15-fuzzy.npz",
    "bytes": 580_136_932,
    "sha256": "8d99aa0ef20c465b9f873d878ef2a5a3147265dc71a2aaaec24ee98b1b7faad7",
}
SEALED_GRAPH_MANIFEST_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": f"{SEALED_ARTIFACT_ROOT}/substrate-graph.json",
    "bytes": 6_679,
    "sha256": "afd240ef38ac20c9965d38b151e12820e6aec36667e9bac5a967fb439a330ae1",
}

#: The full-population finiteness check R0217 did not run. R0222 scores every
#: row, so every row must project finitely here first.
FULL_TRANSFORM_ROWS = ROWS
FULL_TRANSFORM_BATCH = 8_192

#: This round registers no gate; R0222 does, at n = 8.
GATE_REGISTERABLE_HERE = False


class Round0221Error(RuntimeError):
    """The registered MiniLM 2M seed-extension contract changed."""


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0221Error(f"R0221 seed {seed!r} is not a registered cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(
    CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    """What each of R0217's nine seed-bearing fields must hold for this cell."""
    seed = int(seed)
    capability = capability_for_seed(seed)
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
        raise Round0221Error(
            "R0221 seed-bearing path set differs from R0217's registered set"
        )
    return values


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0221Error(f"R0221 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0221Error(f"R0221 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str]:
    """R0217's config for this substrate, with only the seed-bearing fields moved.

    Fails closed unless the result reproduces the seed-invariant digest of the
    R0217 template it was derived from. That equality *is* the "byte-identical
    except the seed" check; there is no separate, weaker field comparison.

    The digest is a function of the bound substrate/graph signatures as well as
    the recipe, so equality with R0217's **published** `241c3f6d...` value can
    only be asserted where the real sealed signatures are bound — the prepare
    script and the node both do exactly that, against the digest read out of
    R0217's own sealed train receipts.
    """
    if int(seed) not in SEEDS:
        raise Round0221Error(f"R0221 seed {seed!r} is not a registered cell")
    for label, observed, registered in (
        ("substrate", substrate_signature, SEALED_SUBSTRATE_SIGNATURE),
        ("graph", graph_signature, SEALED_GRAPH_SIGNATURE),
        ("graph manifest", graph_manifest_signature, SEALED_GRAPH_MANIFEST_SIGNATURE),
    ):
        if dict(observed) != dict(registered):
            raise Round0221Error(
                f"R0221 {label} signature is not the sealed R0216 "
                f"queue-correction-3 one: {dict(observed)!r}"
            )
    template, _template_sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=graph_edges,
        rows=rows,
    )
    config = copy.deepcopy(template)
    for path, replacement in seed_bearing_values(seed).items():
        _set_path(config, path, replacement)
    digest = seed_invariant_sha256(config)
    template_digest = seed_invariant_sha256(template)
    if digest != template_digest:
        raise Round0221Error(
            f"R0221 seed-{int(seed)} treatment is not R0217's: seed-invariant "
            f"digest {digest} != template {template_digest}"
        )
    return config, sha256_bytes(canonical_json(config))


def validate_registered_dose(*, updates: int, edge_count: int) -> dict[str, Any]:
    """R0217's dose validation, plus the registered ceil-derived landing point.

    R0217 pinned the horizon to the exact `ceil` of the registered rule. R0221
    additionally asserts *which* value that `ceil` must produce on this sealed
    graph, so a substrate swap that still satisfies the rule cannot slip through
    as a pooled cell.
    """
    dose = validate_dose(updates=updates, edge_count=edge_count)
    if int(edge_count) != SEALED_DIRECTED_EDGES:
        raise Round0221Error(
            f"R0221 graph has {int(edge_count)} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    if int(dose["successful_updates"]) != REGISTERED_SUCCESSFUL_UPDATES:
        raise Round0221Error(
            f"R0221 derived horizon {dose['successful_updates']} is not the "
            f"registered ceil-derived {REGISTERED_SUCCESSFUL_UPDATES}"
        )
    achieved = float(dose["achieved_positive_draws_per_edge"])
    if achieved != REGISTERED_ACHIEVED_DRAWS_PER_EDGE:
        raise Round0221Error(
            f"R0221 achieved dose {achieved!r} is not the registered "
            f"{REGISTERED_ACHIEVED_DRAWS_PER_EDGE!r}"
        )
    return {
        **dose,
        "registered_successful_updates": REGISTERED_SUCCESSFUL_UPDATES,
        "registered_achieved_positive_draws_per_edge": (
            REGISTERED_ACHIEVED_DRAWS_PER_EDGE
        ),
        "landed_on_registered_ceil_value": True,
    }


def validate_full_population_map(coordinates: Any) -> dict[str, Any]:
    """Every one of the 2,000,000 rows must project to a finite coordinate."""
    array = np.asarray(coordinates)
    if array.shape != (FULL_TRANSFORM_ROWS, OUTPUT_DIMENSION):
        raise Round0221Error(
            f"R0221 full-population transform produced {array.shape}, expected "
            f"({FULL_TRANSFORM_ROWS}, {OUTPUT_DIMENSION})"
        )
    finite = int(np.isfinite(array).all(axis=1).sum())
    if finite != FULL_TRANSFORM_ROWS:
        raise Round0221Error(
            f"R0221 full-population transform has {FULL_TRANSFORM_ROWS - finite} "
            "nonfinite rows"
        )
    published = validate_published_map(array)
    return {
        **published,
        "transform_rows": FULL_TRANSFORM_ROWS,
        "transform_rows_finite": finite,
        "full_population_finite": True,
    }


def assert_extension_differs_only_by_seed(
    configs: Mapping[int, Mapping[str, Any]],
    *,
    expected_seed_invariant: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the four new cells are R0217's treatment, seed aside.

    `expected_seed_invariant` is R0217's digest as read from *its own sealed
    receipts*; the prepare script always supplies it, so the pooled-family claim
    rests on R0217's artifacts rather than on a constant typed into this file.
    """
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0221Error(
            f"R0221 extension must be exactly seeds {list(SEEDS)}, got "
            f"{sorted(observed)}"
        )
    digests: dict[int, str] = {}
    per_seed: dict[str, str] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        for path, want in seed_bearing_values(seed).items():
            cursor: Any = config
            for key in path:
                if not isinstance(cursor, Mapping) or key not in cursor:
                    raise Round0221Error(
                        f"R0221 seed-{seed} config is missing {'.'.join(path)}"
                    )
                cursor = cursor[key]
            if cursor != want:
                raise Round0221Error(
                    f"R0221 seed {seed} cell has {'.'.join(path)}={cursor!r}, "
                    f"expected {want!r}"
                )
        digests[seed] = seed_invariant_sha256(config)
        per_seed[str(seed)] = sha256_bytes(canonical_json(config))
    unique = sorted(set(digests.values()))
    if len(unique) != 1:
        raise Round0221Error(
            "R0221 cells differ outside the seed: seed-invariant digests "
            f"{ {seed: digests[seed] for seed in SEEDS} }"
        )
    if expected_seed_invariant is not None and unique[0] != expected_seed_invariant:
        raise Round0221Error(
            f"R0221 cells are not R0217's treatment: {unique[0]} != "
            f"{expected_seed_invariant}"
        )
    if len(set(per_seed.values())) != len(SEEDS):
        raise Round0221Error("R0221 produced duplicate cell configs")
    return {
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": unique[0],
        "matches_r0217_published_seed_invariant": (
            unique[0] == R0217_SEED_INVARIANT_SHA256
        ),
        "checked_against_r0217_sealed_receipts": expected_seed_invariant is not None,
        "per_seed_config_sha256": per_seed,
        "pooled_seed_family": list(POOLED_SEEDS),
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    }


__all__ = [
    "BATCH_SIZE",
    "CAPABILITIES",
    "CAPABILITY_TEMPLATE",
    "DIMENSION",
    "FULL_TRANSFORM_BATCH",
    "FULL_TRANSFORM_ROWS",
    "GATE_REGISTERABLE_HERE",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GRAPH_SOURCE_ROUND_ID",
    "HOST_RSS_LIMIT_GIB",
    "NEGATIVE_RNG_SEED_OFFSET",
    "OUTPUT_DIMENSION",
    "POOLED_SEEDS",
    "POSITIVE_ROWS_PER_UPDATE",
    "PRODUCTION_CONFIG_SCHEMA",
    "R0217_SEEDS",
    "R0217_SEED_INVARIANT_SHA256",
    "REGISTERED_ACHIEVED_DRAWS_PER_EDGE",
    "REGISTERED_SUCCESSFUL_UPDATES",
    "REGISTERED_UPDATE_BOUND",
    "RELOAD_PROBE_SEED",
    "SEALED_ARTIFACT_ROOT",
    "SEALED_GRAPH_MANIFEST_SIGNATURE",
    "SEALED_GRAPH_SIGNATURE",
    "SEALED_SUBSTRATE_SIGNATURE",
    "ROUND_ID",
    "ROWS",
    "Round0217Error",
    "Round0221Error",
    "SEALED_DIRECTED_EDGES",
    "SEEDS",
    "SEED_BEARING_PATHS",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "TEMPLATE_SEED",
    "TRAIN_SCHEMA",
    "USE_AMP",
    "achieved_draws_per_edge",
    "assert_extension_differs_only_by_seed",
    "capability_for_seed",
    "dose_quantum",
    "performance_windows",
    "seed_bearing_values",
    "seed_invariant_sha256",
    "successful_updates_for_edges",
    "train_config",
    "validate_full_population_map",
    "validate_published_map",
    "validate_registered_dose",
]

"""Frozen contract for the R0250 MiniLM 2M seed extension (seeds 55-57).

**Why this round exists.** `plan-minilm-100m-v2.md`, "Binding before any ladder map
is trained", is explicit: *do NOT judge any rung with the n = 13 gate.* At `n = 13`
the calibrated robust family detects a genuine `2 sigma` regression only
`17.150%` of the time (R0234 section B), which R0234's own power ladder places
against `21.13%` at `n = 16` and `30.07%` at `n = 29`. `n >= 16` is the standing
minimum. The exact-graph family is at `13` (seeds `42-54`), so this round buys the
three cells that reach it.

The treatment is **R0217's**, not a new one. Every R0250 config is produced by
taking R0217's own `train_config` output for its canonical cell and overwriting
**exactly** the nine paths R0217 registered as seed-bearing (`SEED_BEARING_PATHS`),
which is precisely what R0221 did for `46-49` and R0230 did for `50-54`. Three
independent statements of that fact are published per cell so a reviewer can check
it below the digest:

1. the recomputed **seed-invariant digest** must equal R0217's *published*
   `241c3f6d6369e311c8e1e649bd1e8894d8cfa51c17a200c0c6f35746aa04af47`;
2. the **masked canonical JSON bytes** are published per cell as
   `masked_config_bytes` / `masked_config_sha256`, so the digest is reproducible
   from bytes with a reviewer's own masker;
3. `assert_reconstructs_r0217_template` — imported unchanged from R0230 — restores
   R0217's own seed-bearing values and requires **byte equality** with
   `r0217_train_config`'s canonical output.

Nothing here re-types a registered check. The horizon rule, the dose validation,
the byte-for-byte reconstruction, the full-population finiteness check and the
memory budgets are all imported from R0217/R0221/R0230 and called, not restated.
The only new registered constants are the three seed values, the pooled family and
the identity bound at the new `n`.

**What the new `n` buys, stated as arithmetic rather than as a hope.**
`(n-1)/sqrt(n)` is the largest `|x - xbar|/s` any member of an `n`-sample family
can reach; it is `3.3282` at `n = 13` and `3.75` at `n = 16`. A `mean - k*s` floor
whose `k` sits below that bound can be failed by one of its own defining cells.
For the robust `median - k*MAD_n` family the bound is `+inf` by R0234's rank-slack
argument, so a defining cell can fail at **any** multiplier — but the identity is
still the honest yardstick for the variance families this round reports beside it.

This round registers **no gate**. `round0250_gate_n16` does, from these cells and
the thirteen that already exist.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

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
    SEED_PLACEHOLDER,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    USE_AMP,
    achieved_draws_per_edge,
    dose_quantum,
    performance_windows,
    seed_invariant_projection,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config as r0217_train_config,
    validate_dose,
    validate_published_map,
)
from .round0221_minilm_2m_seed_extension import (
    FULL_TRANSFORM_BATCH,
    R0217_SEED_INVARIANT_SHA256,
    REGISTERED_ACHIEVED_DRAWS_PER_EDGE,
    REGISTERED_SUCCESSFUL_UPDATES,
    REGISTERED_UPDATE_BOUND,
    SEALED_ARTIFACT_ROOT,
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
    TEMPLATE_SEED,
)
from .round0230_minilm_2m_seed_extension_n13 import (
    DEVICE_BUDGET_BYTES,
    HOST_ANON_BUDGET_BYTES,
    MEASURED_PEAK_DEVICE_BYTES,
    MEASURED_PEAK_HOST_RSS_GIB,
    MEMORY_POLICY,
    POOLED_SEEDS as R0230_POOLED_SEEDS,
    PREDICTION_SAFETY_FACTOR,
    SWAP_GROWTH_ABORT_BYTES,
    WATCHDOG_POLL_S,
    assert_reconstructs_r0217_template,
    identity_bound,
    masked_config_bytes,
    validate_full_population_map,
    validate_registered_dose,
)


ROUND_ID = "0250"

#: The three new cells that take the exact-graph family from 13 to 16.
SEEDS: tuple[int, ...] = (55, 56, 57)

#: The sixteen-cell family `round0250_gate_n16` pools: R0230's thirteen plus these.
POOLED_SEEDS: tuple[int, ...] = tuple(R0230_POOLED_SEEDS) + SEEDS

#: The standing minimum from `plan-minilm-100m-v2.md`. Registered so the queue,
#: the nodes and the receipts all agree on what this round is buying.
STANDING_MINIMUM_N = 16

N_TARGET = len(POOLED_SEEDS)

TRAIN_SCHEMA = "round0250-minilm-mixed-2m-seed-extension-n16-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0250-minilm-mixed-2m-seed-extension-n16-production-config-v1"
)

FULL_TRANSFORM_ROWS = ROWS

#: This round registers no gate. `round0250_gate_n16` does.
GATE_REGISTERABLE_HERE = False


class Round0250Error(RuntimeError):
    """The registered MiniLM 2M n=16 seed-extension contract changed."""


IDENTITY_BOUND_AT_N16 = identity_bound(N_TARGET)
IDENTITY_BOUND_AT_N13 = identity_bound(len(R0230_POOLED_SEEDS))

IDENTITY_BOUND_NOTE = (
    "max|x - xbar| / s <= (n-1)/sqrt(n). At n = 13 that is "
    f"{IDENTITY_BOUND_AT_N13!r}; at n = {N_TARGET} it is "
    f"{IDENTITY_BOUND_AT_N16!r}. A mean - k*s floor whose k sits below the bound "
    "can be failed by one of its own defining cells; for the robust "
    "median - k*MAD_n family R0234's rank-slack argument gives +inf, so a "
    "defining cell can fail at ANY multiplier. The bound is reported here as the "
    "yardstick for the variance families, not as the robust family's warrant."
)


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0250Error(f"R0250 seed {seed!r} is not a registered cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(
    CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def predict_cell_footprint(seed: int) -> dict[str, Any]:
    """Predict this cell's peak footprint and decide, before anything launches.

    Identical in construction to R0230's — R0221's measured peaks times a `2x`
    safety factor, judged against the same device and host-ANONYMOUS budgets — and
    recorded as data whether or not the cell runs.
    """
    device = int(MEASURED_PEAK_DEVICE_BYTES * PREDICTION_SAFETY_FACTOR)
    host_anon = int(
        MEASURED_PEAK_HOST_RSS_GIB * PREDICTION_SAFETY_FACTOR * 1024 ** 3
    )
    device_over = device > DEVICE_BUDGET_BYTES
    host_over = host_anon > HOST_ANON_BUDGET_BYTES
    return {
        "seed": int(seed),
        "capability": capability_for_seed(seed),
        "basis": (
            "R0221's four sealed cells under this identical treatment: peak CUDA "
            f"{MEASURED_PEAK_DEVICE_BYTES} B and peak host RSS "
            f"{MEASURED_PEAK_HOST_RSS_GIB} GiB, times a "
            f"{PREDICTION_SAFETY_FACTOR}x safety factor"
        ),
        "predicted_peak_device_bytes": device,
        "predicted_peak_host_anonymous_bytes": host_anon,
        "device_budget_bytes": DEVICE_BUDGET_BYTES,
        "host_anonymous_budget_bytes": HOST_ANON_BUDGET_BYTES,
        "predicted_device_headroom_bytes": DEVICE_BUDGET_BYTES - device,
        "predicted_host_headroom_bytes": HOST_ANON_BUDGET_BYTES - host_anon,
        "device_budget_exceeded": device_over,
        "host_budget_exceeded": host_over,
        "refused_a_priori": bool(device_over or host_over),
        "swap_growth_abort_bytes": SWAP_GROWTH_ABORT_BYTES,
        "policy": MEMORY_POLICY,
    }


def seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    """What each of R0217's nine seed-bearing fields must hold for this cell.

    The nine paths are not restated as a policy: the mapping is built and then
    cross-checked against R0217's registered `SEED_BEARING_PATHS`, so a path added
    or removed upstream fails here rather than silently widening the treatment.
    """
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
        raise Round0250Error(
            "R0250 seed-bearing path set differs from R0217's registered set"
        )
    return values


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0250Error(f"R0250 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0250Error(f"R0250 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0250Error(f"R0250 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


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
    R0217 template it was derived from **and** reconstructs that template byte for
    byte when the nine paths are restored. The sealed R0216 signatures are
    registered constants imported from R0221's contract, so a cell cannot be built
    on any other bytes.
    """
    if int(seed) not in SEEDS:
        raise Round0250Error(f"R0250 seed {seed!r} is not a registered cell")
    for label, observed, registered in (
        ("substrate", substrate_signature, SEALED_SUBSTRATE_SIGNATURE),
        ("graph", graph_signature, SEALED_GRAPH_SIGNATURE),
        ("graph manifest", graph_manifest_signature, SEALED_GRAPH_MANIFEST_SIGNATURE),
    ):
        if dict(observed) != dict(registered):
            raise Round0250Error(
                f"R0250 {label} signature is not the sealed R0216 "
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
        raise Round0250Error(
            f"R0250 seed-{int(seed)} treatment is not R0217's: seed-invariant "
            f"digest {digest} != template {template_digest}"
        )
    assert_reconstructs_r0217_template(config, template)
    return config, sha256_bytes(canonical_json(config))


def assert_extension_differs_only_by_seed(
    configs: Mapping[int, Mapping[str, Any]],
    *,
    expected_seed_invariant: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the three new cells are R0217's treatment, seed aside.

    `expected_seed_invariant` is R0217's digest as read from the *sealed receipts*
    of the thirteen cells that already exist, so the pooled-family claim rests on
    published artifacts rather than on a constant typed into this file.
    """
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0250Error(
            f"R0250 extension must be exactly seeds {list(SEEDS)}, got "
            f"{sorted(observed)}"
        )
    digests: dict[int, str] = {}
    per_seed: dict[str, str] = {}
    masked: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        for path, want in seed_bearing_values(seed).items():
            if _get_path(config, path) != want:
                raise Round0250Error(
                    f"R0250 seed {seed} cell has {'.'.join(path)}="
                    f"{_get_path(config, path)!r}, expected {want!r}"
                )
        blob = masked_config_bytes(config)
        digests[seed] = sha256_bytes(blob)
        per_seed[str(seed)] = sha256_bytes(canonical_json(config))
        masked[str(seed)] = {
            "masked_config_sha256": sha256_bytes(blob),
            "masked_config_bytes": len(blob),
            "seed_placeholder": SEED_PLACEHOLDER,
        }
    unique = sorted(set(digests.values()))
    if len(unique) != 1:
        raise Round0250Error(
            "R0250 cells differ outside the seed: seed-invariant digests "
            f"{ {seed: digests[seed] for seed in SEEDS} }"
        )
    if expected_seed_invariant is not None and unique[0] != expected_seed_invariant:
        raise Round0250Error(
            f"R0250 cells are not R0217's treatment: {unique[0]} != "
            f"{expected_seed_invariant}"
        )
    if len(set(per_seed.values())) != len(SEEDS):
        raise Round0250Error("R0250 produced duplicate cell configs")
    return {
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": unique[0],
        "matches_r0217_published_seed_invariant": (
            unique[0] == R0217_SEED_INVARIANT_SHA256
        ),
        "checked_against_prior_sealed_receipts": expected_seed_invariant is not None,
        "per_seed_config_sha256": per_seed,
        "masked_config_identity": masked,
        "masker": (
            "R0217's SEED_BEARING_PATHS replaced by SEED_PLACEHOLDER, canonical "
            "JSON, SHA-256. The masked byte length is published beside the digest "
            "so a reviewer can reproduce the equality with its own masker."
        ),
        "pooled_seed_family": list(POOLED_SEEDS),
        "n_pooled": len(POOLED_SEEDS),
        "standing_minimum_n": STANDING_MINIMUM_N,
        "reaches_the_standing_minimum": len(POOLED_SEEDS) >= STANDING_MINIMUM_N,
        "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N16,
        "identity_bound_note": IDENTITY_BOUND_NOTE,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    }


__all__ = [
    "BATCH_SIZE",
    "CAPABILITIES",
    "CAPABILITY_TEMPLATE",
    "DEVICE_BUDGET_BYTES",
    "DIMENSION",
    "FULL_TRANSFORM_BATCH",
    "FULL_TRANSFORM_ROWS",
    "GATE_REGISTERABLE_HERE",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GRAPH_SOURCE_ROUND_ID",
    "HOST_ANON_BUDGET_BYTES",
    "HOST_RSS_LIMIT_GIB",
    "IDENTITY_BOUND_AT_N13",
    "IDENTITY_BOUND_AT_N16",
    "IDENTITY_BOUND_NOTE",
    "MEASURED_PEAK_DEVICE_BYTES",
    "MEASURED_PEAK_HOST_RSS_GIB",
    "MEMORY_POLICY",
    "NEGATIVE_RNG_SEED_OFFSET",
    "N_TARGET",
    "OUTPUT_DIMENSION",
    "POOLED_SEEDS",
    "POSITIVE_ROWS_PER_UPDATE",
    "PREDICTION_SAFETY_FACTOR",
    "PRODUCTION_CONFIG_SCHEMA",
    "R0217_SEED_INVARIANT_SHA256",
    "R0230_POOLED_SEEDS",
    "REGISTERED_ACHIEVED_DRAWS_PER_EDGE",
    "REGISTERED_SUCCESSFUL_UPDATES",
    "REGISTERED_UPDATE_BOUND",
    "RELOAD_PROBE_SEED",
    "ROUND_ID",
    "ROWS",
    "Round0217Error",
    "Round0250Error",
    "SEALED_ARTIFACT_ROOT",
    "SEALED_DIRECTED_EDGES",
    "SEALED_GRAPH_MANIFEST_SIGNATURE",
    "SEALED_GRAPH_SIGNATURE",
    "SEALED_SUBSTRATE_SIGNATURE",
    "SEEDS",
    "SEED_BEARING_PATHS",
    "SEED_PLACEHOLDER",
    "STANDING_MINIMUM_N",
    "SWAP_GROWTH_ABORT_BYTES",
    "TARGET_POSITIVE_DRAWS_PER_EDGE",
    "TEMPLATE_SEED",
    "TRAIN_SCHEMA",
    "USE_AMP",
    "WATCHDOG_POLL_S",
    "achieved_draws_per_edge",
    "assert_extension_differs_only_by_seed",
    "assert_reconstructs_r0217_template",
    "capability_for_seed",
    "dose_quantum",
    "identity_bound",
    "masked_config_bytes",
    "performance_windows",
    "predict_cell_footprint",
    "seed_bearing_values",
    "seed_invariant_projection",
    "seed_invariant_sha256",
    "successful_updates_for_edges",
    "train_config",
    "validate_dose",
    "validate_full_population_map",
    "validate_published_map",
    "validate_registered_dose",
]

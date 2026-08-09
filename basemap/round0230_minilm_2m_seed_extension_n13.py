"""Frozen contract for the R0230 MiniLM 2M seed extension (seeds 50-54).

**Why this round exists.** A defining cell cannot fail its own `mean - k*s` floor
while `k > (n-1)/sqrt(n)`. That bound is `1.5` at `n = 4`, `2.4749` at `n = 8`
and `3.3282` at `n = 13`, and it first exceeds `2` only at `n = 6`. So R0219's
"4/4 pass" at `n = 4` **and** R0225's "0 failures at `n = 8` under `k = 3.187`"
are both theorems about arithmetic, and R0161 (`n = 4`) and R0193 (`n = 3`) could
never have reported a failure either (Review 0225, and the binding rule in
`guides/plan-minilm-100m-v2.md`). At `k ~ 3.19` the family only becomes able to
falsify itself around `n >= 13`.

This round buys that `n`. It trains seeds `50/51/52/53/54` under a treatment that
is **byte-identical to R0217's in every field except the seed**, exactly as R0221
did for seeds `46-49`, so the thirteen cells pool into one family. R0231 then
registers floors from them — on a **robust scale estimator**, because Review 0225
measured that `median - 3*MAD_n`, a 1-trimmed `mean - 2s` and a fixed margin off
the median all move **exactly 0.0** under the 3-sigma injection that moves
`mean - 2s` by `-0.0079..-0.0366` and the 95/95 floor `1.47-1.50x` further still.

Byte-identity is enforced structurally, not asserted in prose. Every R0230 config
is produced by taking **R0217's own `train_config`** output for its canonical cell
and overwriting **exactly** the nine paths R0217 registered as seed-bearing
(`SEED_BEARING_PATHS`). Three independent statements of that fact are published so
a reviewer can check it below the digest with its own masker, as review-0221's
reviewer did:

1. the recomputed **seed-invariant digest** (R0217's masker: the nine paths
   replaced by `SEED_PLACEHOLDER`, canonical JSON, SHA-256) must equal R0217's
   **published** value
   `241c3f6d6369e311c8e1e649bd1e8894d8cfa51c17a200c0c6f35746aa04af47`;
2. the **masked canonical JSON bytes themselves** are published per cell as
   `masked_config_bytes` / `masked_config_sha256`, so the digest can be
   reproduced from bytes rather than trusted;
3. `assert_reconstructs_r0217_template` puts R0217's own seed-bearing values back
   into an R0230 cell and requires **byte equality** with `r0217_train_config`'s
   canonical output. If any field outside the nine had moved, that equality would
   fail even where a digest collision would not.

Two consequences of the construction, stated rather than left to be discovered:

* The trained config carries `round_id: "0217"`, R0217's config schema and
  `seed_family.seeds: [42, 43, 44, 45]`. Those are **treatment bytes**; changing
  any of them — even to describe the extended family more truthfully — would break
  the digest equality that makes pooling legitimate. The round that *ran* the cell
  is 0230, recorded in the receipt (`round_id`, `treatment_config_round_id`,
  `pooled_seed_family`), which is not part of the treatment.
* The horizon is derived, never carried. The node re-reads `directed_edge_count`
  from R0216's sealed `queue-correction-3` receipt, re-applies the registered
  R0184/R0202 rule and asserts the achieved dose lands on the registered
  ceil-derived value (`80,163` updates, `0.6781860734615339` draws/edge at
  `48,344,648` edges).

This round registers **no gate**. R0231 does, from these cells and the eight that
already exist, and it registers floors that a defining cell can actually fail.
"""
from __future__ import annotations

import copy
import math
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
    SEED_PLACEHOLDER,
    SEEDS as R0217_SEEDS,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    USE_AMP,
    achieved_draws_per_edge,
    dose_quantum,
    expected_seed_bearing_values as r0217_seed_bearing_values,
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
    SEEDS as R0221_SEEDS,
    TEMPLATE_SEED,
)


ROUND_ID = "0230"

#: The five new cells.
SEEDS: tuple[int, ...] = (50, 51, 52, 53, 54)

#: The thirteen-cell family R0231 pools: R0217's four, R0221's four, these five.
POOLED_SEEDS: tuple[int, ...] = tuple(R0217_SEEDS) + tuple(R0221_SEEDS) + SEEDS

#: `(n-1)/sqrt(n)`, the largest `|z|` any member of an `n`-sample family can
#: reach. At `n = 13` it is `3.3282...`, which exceeds `mean - 2s`'s `2.0`, the
#: one-sided 95/95 factor `2.6705` and Howe's two-sided `3.1008` — so under every
#: `mean - k*s` family R0231 reports, a defining cell CAN fail. That is the whole
#: reason this round exists.
N_TARGET = len(POOLED_SEEDS)

TRAIN_SCHEMA = "round0230-minilm-mixed-2m-seed-extension-n13-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0230-minilm-mixed-2m-seed-extension-n13-production-config-v1"
)

#: The full-population finiteness check, as R0221 ran it: every one of the
#: 2,000,000 rows must project to a finite coordinate, because the panel scores
#: every row.
FULL_TRANSFORM_ROWS = ROWS

#: This round registers no gate. R0231 does, at n = 13, on a robust scale.
GATE_REGISTERABLE_HERE = False

# --------------------------------------------------------------------------- #
# the predictive memory guard
# --------------------------------------------------------------------------- #
#
# A prior round wedged this box: a build drove system swap from its baseline to
# exhaustion and the SIGKILL that followed, delivered to a process holding a CUDA
# context, deadlocked RCU and forced a reboot. The rules earned there and carried
# here: predict before launching, record the prediction as data whether or not
# the cell is refused, judge swap on GROWTH over a pre-launch baseline rather
# than on an absolute level, and never SIGKILL a process holding a CUDA context.

#: Budgets. Device is the card's usable share; host is the ANONYMOUS (swappable)
#: budget, never RSS — R0226's 16M cell held 56.96 GB of RSS against 3.43 GB
#: anonymous, and clean file-backed pages are evicted rather than swapped
#: (Review 0224).
DEVICE_BUDGET_BYTES = 24 * 1024 ** 3
HOST_ANON_BUDGET_BYTES = 60 * 1024 ** 3

#: Reaction threshold on swap GROWTH over the pre-launch baseline, not on an
#: absolute level: this box holds swap at idle.
SWAP_GROWTH_ABORT_BYTES = 1 * 1024 ** 3

#: R0221's measured per-cell peaks under this exact treatment (result-0221:
#: `peak CUDA bytes 796,540,416` in all four cells; `peak host RSS 5.281-5.282
#: GiB`). The prediction is those measurements with a 2x safety factor, which is
#: how a prediction differs from a transcription.
MEASURED_PEAK_DEVICE_BYTES = 796_540_416
MEASURED_PEAK_HOST_RSS_GIB = 5.282
PREDICTION_SAFETY_FACTOR = 2.0

WATCHDOG_POLL_S = 5.0

MEMORY_POLICY = (
    "Every cell is predicted before it is launched and the prediction is sealed "
    "whether or not the cell runs, including cells refused a priori. Device "
    "budget 24 GiB; host ANONYMOUS budget 60 GiB (never RSS: file-backed pages "
    "are evicted, not swapped). A live watchdog samples the training process and "
    "trips on system swap GROWTH over the pre-launch baseline, never on an "
    "absolute swap level. The trip is cooperative and in-process; nothing in this "
    "round ever sends SIGKILL to a process holding a CUDA context."
)


class Round0230Error(RuntimeError):
    """The registered MiniLM 2M n=13 seed-extension contract changed."""


def identity_bound(n: int) -> float:
    """`(n-1)/sqrt(n)`: the largest |z| any single sample point can reach."""
    if int(n) < 2:
        raise Round0230Error("R0230 identity bound needs at least two cells")
    return (int(n) - 1) / math.sqrt(int(n))


IDENTITY_BOUND_AT_N13 = identity_bound(N_TARGET)


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0230Error(f"R0230 seed {seed!r} is not a registered cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(
    CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def predict_cell_footprint(seed: int) -> dict[str, Any]:
    """Predict this cell's peak footprint and decide, before anything launches.

    Recorded as data in every case. `refused_a_priori` is `True` when the
    prediction breaches either budget, and such a cell is never launched — the
    prediction and its refusal are the round's evidence about it.
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
        raise Round0230Error(
            "R0230 seed-bearing path set differs from R0217's registered set"
        )
    return values


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0230Error(f"R0230 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0230Error(f"R0230 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0230Error(f"R0230 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def masked_config_bytes(config: Mapping[str, Any]) -> bytes:
    """R0217's masker, exposed as **bytes** rather than only as a digest.

    `seed_invariant_sha256` is the SHA-256 of exactly these bytes. Publishing the
    bytes' length and digest side by side lets a reviewer reproduce the equality
    with its own masker instead of trusting ours — which is how review-0221's
    reviewer checked R0221 below the digest.
    """
    return canonical_json(seed_invariant_projection(config))


def assert_reconstructs_r0217_template(
    config: Mapping[str, Any], template: Mapping[str, Any]
) -> dict[str, Any]:
    """Put R0217's seed-bearing values back and require **byte** equality.

    Stronger than digest equality: it compares the full canonical JSON of the
    reconstructed config against `r0217_train_config`'s own output, so a field
    outside the nine registered seed-bearing paths cannot have moved.
    """
    reconstructed = copy.deepcopy(dict(config))
    for path, value in r0217_seed_bearing_values(TEMPLATE_SEED).items():
        _set_path(reconstructed, path, value)
    mine = canonical_json(reconstructed)
    theirs = canonical_json(dict(template))
    if mine != theirs:
        raise Round0230Error(
            "R0230 cell does not reconstruct R0217's canonical config byte for "
            "byte once the nine seed-bearing paths are restored"
        )
    return {
        "reconstructed_sha256": sha256_bytes(mine),
        "r0217_template_sha256": sha256_bytes(theirs),
        "bytes": len(mine),
        "byte_equal": True,
        "seed_bearing_paths_restored": [
            ".".join(path) for path in SEED_BEARING_PATHS
        ],
    }


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
    registered constants (imported from R0221's contract, unchanged), so a cell
    cannot be built on any other bytes — the check fires before the config is
    assembled.
    """
    if int(seed) not in SEEDS:
        raise Round0230Error(f"R0230 seed {seed!r} is not a registered cell")
    for label, observed, registered in (
        ("substrate", substrate_signature, SEALED_SUBSTRATE_SIGNATURE),
        ("graph", graph_signature, SEALED_GRAPH_SIGNATURE),
        ("graph manifest", graph_manifest_signature, SEALED_GRAPH_MANIFEST_SIGNATURE),
    ):
        if dict(observed) != dict(registered):
            raise Round0230Error(
                f"R0230 {label} signature is not the sealed R0216 "
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
        raise Round0230Error(
            f"R0230 seed-{int(seed)} treatment is not R0217's: seed-invariant "
            f"digest {digest} != template {template_digest}"
        )
    assert_reconstructs_r0217_template(config, template)
    return config, sha256_bytes(canonical_json(config))


def validate_registered_dose(*, updates: int, edge_count: int) -> dict[str, Any]:
    """R0217's dose validation, plus R0221's registered ceil-derived landing point."""
    dose = validate_dose(updates=updates, edge_count=edge_count)
    if int(edge_count) != SEALED_DIRECTED_EDGES:
        raise Round0230Error(
            f"R0230 graph has {int(edge_count)} directed edges, registered "
            f"{SEALED_DIRECTED_EDGES}"
        )
    if int(dose["successful_updates"]) != REGISTERED_SUCCESSFUL_UPDATES:
        raise Round0230Error(
            f"R0230 derived horizon {dose['successful_updates']} is not the "
            f"registered ceil-derived {REGISTERED_SUCCESSFUL_UPDATES}"
        )
    achieved = float(dose["achieved_positive_draws_per_edge"])
    if achieved != REGISTERED_ACHIEVED_DRAWS_PER_EDGE:
        raise Round0230Error(
            f"R0230 achieved dose {achieved!r} is not the registered "
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
        raise Round0230Error(
            f"R0230 full-population transform produced {array.shape}, expected "
            f"({FULL_TRANSFORM_ROWS}, {OUTPUT_DIMENSION})"
        )
    finite = int(np.isfinite(array).all(axis=1).sum())
    if finite != FULL_TRANSFORM_ROWS:
        raise Round0230Error(
            f"R0230 full-population transform has {FULL_TRANSFORM_ROWS - finite} "
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
    """Fail closed unless the five new cells are R0217's treatment, seed aside.

    `expected_seed_invariant` is R0217's digest as read from *its own sealed
    receipts*; the prepare script always supplies it, so the pooled-family claim
    rests on R0217's artifacts rather than on a constant typed into this file.
    """
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0230Error(
            f"R0230 extension must be exactly seeds {list(SEEDS)}, got "
            f"{sorted(observed)}"
        )
    digests: dict[int, str] = {}
    per_seed: dict[str, str] = {}
    masked: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        for path, want in seed_bearing_values(seed).items():
            if _get_path(config, path) != want:
                raise Round0230Error(
                    f"R0230 seed {seed} cell has {'.'.join(path)}="
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
        raise Round0230Error(
            "R0230 cells differ outside the seed: seed-invariant digests "
            f"{ {seed: digests[seed] for seed in SEEDS} }"
        )
    if expected_seed_invariant is not None and unique[0] != expected_seed_invariant:
        raise Round0230Error(
            f"R0230 cells are not R0217's treatment: {unique[0]} != "
            f"{expected_seed_invariant}"
        )
    if len(set(per_seed.values())) != len(SEEDS):
        raise Round0230Error("R0230 produced duplicate cell configs")
    return {
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": unique[0],
        "matches_r0217_published_seed_invariant": (
            unique[0] == R0217_SEED_INVARIANT_SHA256
        ),
        "checked_against_r0217_sealed_receipts": expected_seed_invariant is not None,
        "per_seed_config_sha256": per_seed,
        "masked_config_identity": masked,
        "masker": (
            "R0217's SEED_BEARING_PATHS replaced by SEED_PLACEHOLDER, canonical "
            "JSON, SHA-256. The masked byte length is published beside the digest "
            "so a reviewer can reproduce the equality with its own masker."
        ),
        "pooled_seed_family": list(POOLED_SEEDS),
        "n_pooled": len(POOLED_SEEDS),
        "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N13,
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
    "R0217_SEEDS",
    "R0217_SEED_INVARIANT_SHA256",
    "R0221_SEEDS",
    "REGISTERED_ACHIEVED_DRAWS_PER_EDGE",
    "REGISTERED_SUCCESSFUL_UPDATES",
    "REGISTERED_UPDATE_BOUND",
    "RELOAD_PROBE_SEED",
    "ROUND_ID",
    "ROWS",
    "Round0217Error",
    "Round0230Error",
    "SEALED_ARTIFACT_ROOT",
    "SEALED_DIRECTED_EDGES",
    "SEALED_GRAPH_MANIFEST_SIGNATURE",
    "SEALED_GRAPH_SIGNATURE",
    "SEALED_SUBSTRATE_SIGNATURE",
    "SEEDS",
    "SEED_BEARING_PATHS",
    "SEED_PLACEHOLDER",
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
    "validate_full_population_map",
    "validate_published_map",
    "validate_registered_dose",
]

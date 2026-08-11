"""Frozen contract for the R0255 MiniLM 2M seed extension (seeds 58-70) at `n = 29`.

**Why this round exists.** The owner ruled on 2026-08-11 (`OWNER-DECISIONS-PENDING.md`
section 3, status DECIDED; `plan-minilm-100m-v2.md` Phase 3): register the Phase 3
gate on `MAD_n`, and close the power gap by *extending the family* rather than by
trading robustness away. Thirteen new 2M cells under the **unchanged R0217
treatment**, pooled with the existing sixteen on the **frozen panel**, taking the
exact-graph family to `n = 29` -- the size the standing rules called for as "~29 for
real power".

Three constraints from that ruling are load-bearing here.

1. **2M universe only.** These thirteen cells are 2M maps. A rung map is *judged* by
   this gate and is **never added to its family**; `round0255_treatment.py` carries
   the guard that refuses a family containing anything else, with the v0 construction
   beside it as the negative control.
2. **Unchanged treatment.** Every config is produced by taking R0217's own
   `train_config` output for its canonical cell and overwriting **exactly** the nine
   paths R0217 registered as seed-bearing, which is what R0221 (46-49), R0230 (50-54)
   and R0250 (55-57) each did. Nothing about the recipe, the sampler, the precision,
   the residency, the dose or the horizon moves. The *release* is a separate axis and
   is checked separately -- see `round0255_treatment.assert_treatment_sources_unchanged`
   and the seed-42 replay control below.
3. **Nothing is tuned to preserve a published pass.** This module fits no floor and
   reads no held-out cell. `round0255_gate_n29` registers, and its independence
   control proves the fit does not move when a held-out cell moves.

**The seed-42 replay control.** R0251 settled that the *map-side scorer* has not
drifted by re-scoring R0217's archived seed-42 checkpoint. Review-0251 observed that
the *release diff* was the stronger evidence. This round uses both, and adds the
missing third: it **retrains seed 42** on this release under R0217's own template
config and compares the production-config digest and the published checkpoint against
R0217's sealed bytes. The replay is a **control, not a cell**: seed 42 is already in
the family, so adding its replay would double-count a cell, and the family-purity
guard refuses exactly that.
"""
from __future__ import annotations

import copy
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
    PREDICTION_SAFETY_FACTOR,
    SWAP_GROWTH_ABORT_BYTES,
    WATCHDOG_POLL_S,
    assert_reconstructs_r0217_template,
    identity_bound,
    masked_config_bytes,
    validate_full_population_map,
    validate_registered_dose,
)
from .round0250_seed_extension_n16 import (
    POOLED_SEEDS as R0250_POOLED_SEEDS,
    SEEDS as R0250_SEEDS,
    STANDING_MINIMUM_N,
)


ROUND_ID = "0255"

#: The thirteen new cells. **How they were chosen:** the exact-graph 2M family has
#: consumed a single contiguous block of training seeds since it was created --
#: R0217 took 42-45, R0221 46-49, R0230 50-54, R0250 55-57 -- so the next thirteen
#: integers are the only choice that is disjoint from all sixteen existing cells by
#: construction rather than by inspection, keeps the pooled family a contiguous
#: block a reviewer can verify at a glance, and continues the convention every prior
#: extension used. `assert_extension_differs_only_by_seed` re-proves the disjointness
#: against the sixteen sealed config digests rather than trusting the convention.
SEEDS: tuple[int, ...] = tuple(range(58, 71))

#: The twenty-nine-cell family `round0255_gate_n29` pools: R0250's sixteen plus these.
POOLED_SEEDS: tuple[int, ...] = tuple(R0250_POOLED_SEEDS) + SEEDS

#: The owner's ruling names this n explicitly.
OWNER_RULING_N = 29

N_TARGET = len(POOLED_SEEDS)

#: The seed the replay control retrains. It is R0217's template seed and is ALREADY
#: a family cell, which is exactly why the replay is a control and never a cell.
REPLAY_CONTROL_SEED = TEMPLATE_SEED
REPLAY_CONTROL_CAPABILITY = "minilm-mixed-2m-replay-control-seed42-r0255-v1"

TRAIN_SCHEMA = "round0255-minilm-mixed-2m-seed-extension-n29-train-receipt-v1"
REPLAY_SCHEMA = "round0255-minilm-mixed-2m-replay-control-seed42-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0255-minilm-mixed-2m-seed-extension-n29-production-config-v1"
)

FULL_TRANSFORM_ROWS = ROWS

#: This round registers no gate here. `round0255_gate_n29` does.
GATE_REGISTERABLE_HERE = False


class Round0255Error(RuntimeError):
    """The registered MiniLM 2M n=29 seed-extension contract changed."""


IDENTITY_BOUND_AT_N29 = identity_bound(N_TARGET)
IDENTITY_BOUND_AT_N16 = identity_bound(len(R0250_POOLED_SEEDS))

IDENTITY_BOUND_NOTE = (
    "max|x - xbar| / s <= (n-1)/sqrt(n). At n = 16 that is "
    f"{IDENTITY_BOUND_AT_N16!r}; at n = {N_TARGET} it is {IDENTITY_BOUND_AT_N29!r} "
    "(= 28/sqrt(29)). A mean - k*s floor whose k sits AT OR ABOVE that bound cannot "
    "be failed by any of its own defining cells, which is the defect the whole gate "
    "redesign exists to escape. For the robust median - k*MAD_n family R0234's "
    "rank-slack argument gives +inf, so a defining cell can fail at ANY multiplier; "
    "the identity is reported as the yardstick for the variance families scored "
    "beside it, and because a k above it would be disqualifying on its face."
)


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0255Error(f"R0255 seed {seed!r} is not a registered cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(
    CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def predict_cell_footprint(seed: int, *, replay_control: bool = False) -> dict[str, Any]:
    """Predict this cell's peak footprint and decide, before anything launches.

    Identical in construction to R0230's and R0250's -- R0221's measured peaks times
    the same `2x` safety factor, judged against the same device and host-ANONYMOUS
    budgets -- and recorded as data whether or not the cell runs.
    """
    seed = int(seed)
    if replay_control:
        if seed != REPLAY_CONTROL_SEED:
            raise Round0255Error(
                f"R0255 replay control is seed {REPLAY_CONTROL_SEED}, not {seed!r}"
            )
        capability = REPLAY_CONTROL_CAPABILITY
    else:
        capability = capability_for_seed(seed)
    device = int(MEASURED_PEAK_DEVICE_BYTES * PREDICTION_SAFETY_FACTOR)
    host_anon = int(
        MEASURED_PEAK_HOST_RSS_GIB * PREDICTION_SAFETY_FACTOR * 1024 ** 3
    )
    device_over = device > DEVICE_BUDGET_BYTES
    host_over = host_anon > HOST_ANON_BUDGET_BYTES
    return {
        "seed": seed,
        "capability": capability,
        "is_a_family_cell": not replay_control,
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

    The mapping is built and then cross-checked against R0217's registered
    `SEED_BEARING_PATHS`, so a path added or removed upstream fails here rather than
    silently widening the treatment.
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
        raise Round0255Error(
            "R0255 seed-bearing path set differs from R0217's registered set"
        )
    return values


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0255Error(f"R0255 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0255Error(f"R0255 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0255Error(f"R0255 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def _assert_sealed_inputs(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
) -> None:
    for label, observed, registered in (
        ("substrate", substrate_signature, SEALED_SUBSTRATE_SIGNATURE),
        ("graph", graph_signature, SEALED_GRAPH_SIGNATURE),
        ("graph manifest", graph_manifest_signature, SEALED_GRAPH_MANIFEST_SIGNATURE),
    ):
        if dict(observed) != dict(registered):
            raise Round0255Error(
                f"R0255 {label} signature is not the sealed R0216 "
                f"queue-correction-3 one: {dict(observed)!r}"
            )


def train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str]:
    """R0217's config for this substrate, with only the seed-bearing fields moved."""
    if int(seed) not in SEEDS:
        raise Round0255Error(f"R0255 seed {seed!r} is not a registered cell")
    _assert_sealed_inputs(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
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
        raise Round0255Error(
            f"R0255 seed-{int(seed)} treatment is not R0217's: seed-invariant "
            f"digest {digest} != template {template_digest}"
        )
    assert_reconstructs_r0217_template(config, template)
    return config, sha256_bytes(canonical_json(config))


def replay_control_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str]:
    """R0217's canonical seed-42 config, unmodified -- the replay control's config.

    No seed-bearing path is overwritten, because the replay IS the template cell.
    Its digest must equal R0217's published seed-42 production-config digest, which
    is the first of the three comparisons the control makes.
    """
    _assert_sealed_inputs(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
    )
    config, sha = r0217_train_config(
        seed=REPLAY_CONTROL_SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=graph_edges,
        rows=rows,
    )
    if seed_invariant_sha256(config) != R0217_SEED_INVARIANT_SHA256:
        raise Round0255Error(
            "R0255 replay control config does not carry R0217's seed-invariant digest"
        )
    return config, sha


def assert_extension_differs_only_by_seed(
    configs: Mapping[int, Mapping[str, Any]],
    *,
    expected_seed_invariant: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the thirteen new cells are R0217's treatment, seed aside."""
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0255Error(
            f"R0255 extension must be exactly seeds {list(SEEDS)}, got "
            f"{sorted(observed)}"
        )
    digests: dict[int, str] = {}
    per_seed: dict[str, str] = {}
    masked: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        for path, want in seed_bearing_values(seed).items():
            if _get_path(config, path) != want:
                raise Round0255Error(
                    f"R0255 seed {seed} cell has {'.'.join(path)}="
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
        raise Round0255Error(
            "R0255 cells differ outside the seed: seed-invariant digests "
            f"{ {seed: digests[seed] for seed in SEEDS} }"
        )
    if expected_seed_invariant is not None and unique[0] != expected_seed_invariant:
        raise Round0255Error(
            f"R0255 cells are not R0217's treatment: {unique[0]} != "
            f"{expected_seed_invariant}"
        )
    if len(set(per_seed.values())) != len(SEEDS):
        raise Round0255Error("R0255 produced duplicate cell configs")
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
        "owner_ruling_n": OWNER_RULING_N,
        "reaches_the_owner_ruling_n": len(POOLED_SEEDS) == OWNER_RULING_N,
        "standing_minimum_n": STANDING_MINIMUM_N,
        "identity_bound_at_n_pooled": IDENTITY_BOUND_AT_N29,
        "identity_bound_note": IDENTITY_BOUND_NOTE,
        "seed_choice_rationale": (
            "the exact-graph 2M family has consumed one contiguous block of "
            "training seeds since R0217 (42-45, 46-49, 50-54, 55-57); 58-70 is the "
            "next thirteen integers, disjoint from all sixteen by construction, and "
            "the disjointness is re-proved here against the sixteen sealed config "
            "digests rather than asserted from the convention"
        ),
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
    "IDENTITY_BOUND_AT_N16",
    "IDENTITY_BOUND_AT_N29",
    "IDENTITY_BOUND_NOTE",
    "MEASURED_PEAK_DEVICE_BYTES",
    "MEASURED_PEAK_HOST_RSS_GIB",
    "MEMORY_POLICY",
    "NEGATIVE_RNG_SEED_OFFSET",
    "N_TARGET",
    "OUTPUT_DIMENSION",
    "OWNER_RULING_N",
    "POOLED_SEEDS",
    "POSITIVE_ROWS_PER_UPDATE",
    "PREDICTION_SAFETY_FACTOR",
    "PRODUCTION_CONFIG_SCHEMA",
    "R0217_SEED_INVARIANT_SHA256",
    "R0250_POOLED_SEEDS",
    "R0250_SEEDS",
    "REGISTERED_ACHIEVED_DRAWS_PER_EDGE",
    "REGISTERED_SUCCESSFUL_UPDATES",
    "REGISTERED_UPDATE_BOUND",
    "RELOAD_PROBE_SEED",
    "REPLAY_CONTROL_CAPABILITY",
    "REPLAY_CONTROL_SEED",
    "REPLAY_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0217Error",
    "Round0255Error",
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
    "replay_control_config",
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

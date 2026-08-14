"""R0265's treatment closure: the pinned fneg recipe as a masked-config invariant.

R0255 registered the Phase 3 gate on R0217's *unchanged* treatment, so its closure
proved a config is R0217's -- byte-for-byte outside the nine seed-bearing paths --
and its positive control planted a *source* change the config-level digest could not
see. R0265 is different: it is a **new treatment**. The pinned fog-targeted-negatives
recipe (`fneg`) changes the low-dim kernel, the `a`/`b` curve, the dose, and turns the
fneg reweighting on. So this closure does not prove "same as R0217"; it proves "is the
one registered fneg recipe", and its positive controls plant the four ways a config
could drift *off* that recipe while still looking like a seed-family cell.

**The recipe, pinned (promotion plan, GO 2026-08-14).** Built from R0217's own
`train_config` for the sealed R0216 2M substrate + exact k15 graph, then exactly these
fields moved, and nothing else outside the nine seed-bearing paths:

* `model.low_dim_kernel`  `legacy_lp` -> `umap`
* `model.a`               `1.0` -> `1.9328`   (the `min_dist = 0` curve fit)
* `model.b`               `1.0` -> `0.7905`
* `optimizer.fneg_weight` (absent / `0.0`) -> `1.0`   (fog-targeted negatives ON)
* `optimizer.fneg_lo`     -> `0.1`
* `optimizer.fneg_hi`     -> `0.4`
* dose `x4`: `optimizer.successful_positive_lr_updates` `80163` -> `320652`
  (`round(4 x base_horizon)`), `execution.target_positive_draws_per_edge`
  `0.67818` -> `2.71271`, and a `x4` `dose_registration` receipt.

**Guard -- `assert_registered_recipe`.** Refuses unless a config carries EXACTLY the
pinned recipe. The kernel/curve/fneg fields are checked against registered constants;
the horizon is checked against `dose_multiplier x successful_updates_for_edges(edges)`
derived from the config's own sealed edge count, so a `x1` or `x2` dose is caught by
the rule and not by a literal. The four `recipe_refusal_controls` plant wrong-kernel,
wrong-dose, fneg-off and wrong-band configs against the SHIPPED predicate and prove it
refuses each; the OLD predicate this replaces -- a shape check that reads only the seed
and the output dimension -- accepts every one, which is the point: a guard that reads
the seed but not the recipe cannot tell an fneg cell from a legacy_lp cell.

**Import-closure seal.** As in R0255, the round also seals, at prepare time, the
SHA-256 of every source file in the training import closure, and every training node
re-hashes the files it imported and refuses unless they equal the sealed bytes. The
fneg merge (commit 9794b4b) touched `core.py`, so the closure is what proves the cell
ran the merged core rather than a pre-merge one.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from .round0217_minilm_2m_seed_family import (
    DIMENSION,
    OUTPUT_DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEED_BEARING_PATHS,
    SEED_PLACEHOLDER,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    achieved_draws_per_edge,
    dose_quantum,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config as r0217_train_config,
)
from .round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
    TEMPLATE_SEED,
)
from .round0230_minilm_2m_seed_extension_n13 import identity_bound, masked_config_bytes


ROUND_ID = "0265"

CLOSURE_SCHEMA = "round0265-fneg-treatment-import-closure-v1"
FNEG_TRAIN_CONFIG_SCHEMA = "round0265-minilm-mixed-2m-fneg-x4-md000-production-config-v1"
FNEG_RECIPE_SCHEMA = "round0265-fneg-x4-md000-recipe-v1"

#: The thirteen seeds of the pinned recipe -- seeds 42-54, one map per seed, the seed
#: the only free variable across the family. They deliberately reuse the integer block
#: 42-54; the fneg capability, not the seed, distinguishes an fneg cell from any legacy
#: exact-graph cell that shares its seed.
SEEDS: tuple[int, ...] = tuple(range(42, 55))
N_TARGET = len(SEEDS)

CANONICAL_SEED = 42

CAPABILITY_TEMPLATE = "minilm-mixed-2m-fneg-x4-md000-seed{seed}-r0265-v1"

# --------------------------------------------------------------------------- #
# the pinned recipe constants
# --------------------------------------------------------------------------- #

#: The `min_dist = 0` UMAP curve fit. These are the `a`/`b` the sandbox `umap-md000`
#: arms trained under; `a`/`b` ARE the stored form of `min_dist` (there is no min_dist
#: key in the config), so pinning them pins the kernel geometry.
FNEG_LOW_DIM_KERNEL = "umap"
FNEG_A = 1.9328
FNEG_B = 0.7905

#: Fog-targeted negatives, ON. A negative pair whose current 2D endpoint distance lies
#: in `[fneg_lo * R, fneg_hi * R]` (R = p90 batch radius) gets its BCE up-weighted by
#: `1 + fneg_weight`. The lo/hi equal the core defaults; only the weight is turned on.
FNEG_WEIGHT = 1.0
FNEG_LO = 0.1
FNEG_HI = 0.4

#: The dose multiplier. `x4` of R0217's accepted low dose.
DOSE_MULTIPLIER = 4

#: The x4 horizon and draws-per-edge at the sealed 2M graph, published as the recipe's
#: registered anchors. They are DERIVED from the sealed edge count at build time; these
#: literals exist only so a reviewer sees the intended values and the round refuses if
#: the derivation lands elsewhere.
FNEG_BASE_HORIZON = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
FNEG_HORIZON = DOSE_MULTIPLIER * FNEG_BASE_HORIZON
FNEG_TARGET_DRAWS_PER_EDGE = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE

#: The paths the recipe overwrites on top of R0217's template, other than the nine
#: seed-bearing ones. Named so the invariant has an auditable membership.
RECIPE_KERNEL_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "low_dim_kernel"),
    ("model", "a"),
    ("model", "b"),
)
RECIPE_FNEG_PATHS: tuple[tuple[str, ...], ...] = (
    ("optimizer", "fneg_weight"),
    ("optimizer", "fneg_lo"),
    ("optimizer", "fneg_hi"),
)
RECIPE_DOSE_PATHS: tuple[tuple[str, ...], ...] = (
    ("optimizer", "successful_positive_lr_updates"),
    ("execution", "target_positive_draws_per_edge"),
)

#: The modules a training node imports that can change what the trainer computes -- the
#: fneg-merged core first of all. Named here so the seal has a fixed membership.
TRAIN_CLOSURE_MODULES: tuple[str, ...] = (
    "basemap.pumap.parametric_umap",
    "basemap.pumap.parametric_umap.core",
    "basemap.pumap.parametric_umap.perf",
    "basemap.pumap.parametric_umap.models.mlp",
    "basemap.pumap.parametric_umap.datasets.edge_dataset",
    "basemap.pumap.parametric_umap.datasets.edge_list_dataset",
    "basemap.pumap.parametric_umap.utils.data_prefetcher",
    "basemap.pumap.parametric_umap.utils.graph",
    "basemap.pumap.parametric_umap.utils.losses",
    "basemap.round0113_prompt_contrast",
    "basemap.round0217_minilm_2m_seed_family",
    "basemap.round0265_fneg_treatment",
)

#: The predicate this guard replaces, quoted so the control has a target. It reads the
#: seed and the output dimension and nothing about the recipe, so every recipe-level
#: defect planted below passes it -- exactly as R0255's config-only predicate passed
#: every source-level defect.
OLD_RECIPE_PREDICATE = (
    "a shape check: config['seed'] in SEEDS and model['output_dimension'] == 2. It "
    "reads the seed and the map dimension but NOTHING about the kernel, the dose or "
    "the fneg reweighting, so a legacy_lp cell, a x1-dose cell and an fneg-off cell "
    "all pass it. That is the treatment blindness this closure exists to remove."
)


class Round0265RecipeError(RuntimeError):
    """The config is not the registered R0265 fneg recipe."""


class Round0265TreatmentError(RuntimeError):
    """The registered R0265 import-closure contract changed."""


class Round0265FamilyError(RuntimeError):
    """The gate's fitting family is not the registered n=13 fneg family."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def capability_for_seed(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0265RecipeError(f"R0265 seed {seed!r} is not a registered fneg cell")
    return CAPABILITY_TEMPLATE.format(seed=int(seed))


CAPABILITIES: tuple[str, ...] = tuple(
    CAPABILITY_TEMPLATE.format(seed=seed) for seed in SEEDS
)


def exact_cell_id(seed: int) -> str:
    return f"fneg-x4-md000-seed{int(seed)}"


# --------------------------------------------------------------------------- #
# path helpers
# --------------------------------------------------------------------------- #


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0265RecipeError(f"R0265 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0265RecipeError(f"R0265 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _set_path_new(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    """Set a path that may not yet exist (the fneg keys R0217's config never carried)."""
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0265RecipeError(f"R0265 config is missing parent of {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0265RecipeError(f"R0265 config parent of {'.'.join(path)} is not a map")
    cursor[path[-1]] = replacement


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0265RecipeError(f"R0265 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def _seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    """The nine seed-bearing fields for an fneg cell, cross-checked against R0217's set."""
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
        raise Round0265RecipeError(
            "R0265 seed-bearing path set differs from R0217's registered set"
        )
    return values


# --------------------------------------------------------------------------- #
# the x4 dose receipt
# --------------------------------------------------------------------------- #


def validate_fneg_dose(*, updates: int, edge_count: int) -> dict[str, Any]:
    """Pin the x4 horizon exactly and bound the achieved dose by `x4` quantisation steps.

    The base ceil rule (R0184/R0202) rounds the x1 horizon up by at most one update, so
    the x4 horizon `4 * base` carries at most `4 *` that quantisation cost against the
    `4 * target` dose. Fails closed on either half.
    """
    base = successful_updates_for_edges(edge_count)
    exact = DOSE_MULTIPLIER * base
    if int(updates) != exact:
        raise Round0265RecipeError(
            f"R0265 x4 update horizon {int(updates)} is not "
            f"{DOSE_MULTIPLIER} * successful_updates_for_edges({int(edge_count)}) "
            f"= {DOSE_MULTIPLIER} * {base} = {exact}"
        )
    achieved = achieved_draws_per_edge(updates=exact, edge_count=edge_count)
    quantum = dose_quantum(edge_count)
    target = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE
    tolerance = DOSE_MULTIPLIER * quantum
    deviation = abs(achieved - target)
    if deviation > tolerance:
        raise Round0265RecipeError(
            f"R0265 achieved x4 dose {achieved!r} is {deviation:.3e} from the "
            f"registered {target!r}, beyond the {tolerance:.3e} "
            f"({DOSE_MULTIPLIER} x one-update) quantisation bound"
        )
    return {
        "source_round": "0184-x4",
        "dose_multiplier": DOSE_MULTIPLIER,
        "base_successful_updates": base,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": int(edge_count),
        "successful_updates": exact,
        "dose_rule": (
            f"{DOSE_MULTIPLIER} * ceil(R0184_successful_updates * active_edges / "
            "R0184_directed_edges)"
        ),
        "target_positive_draws_per_edge": target,
        "achieved_positive_draws_per_edge": achieved,
        "dose_quantum_draws_per_edge": quantum,
        "achieved_minus_target": achieved - target,
        "tolerance_basis": (
            f"{DOSE_MULTIPLIER} successful-update quantisation steps; the x4 horizon "
            "inherits the base ceil rule's rounding at each of the four multiples"
        ),
    }


# --------------------------------------------------------------------------- #
# building the recipe config
# --------------------------------------------------------------------------- #


def fneg_train_config(
    *,
    seed: int,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
) -> tuple[dict[str, Any], str]:
    """R0217's config for this substrate, with the fneg recipe applied and the seed set.

    Built from the seed-42 template, then the kernel/curve/dose/fneg fields overwritten,
    then the nine seed-bearing paths set for `seed`. The result carries the fneg recipe;
    `assert_registered_recipe` proves it, and all thirteen cells share one masked digest.
    """
    if int(seed) not in SEEDS:
        raise Round0265RecipeError(f"R0265 seed {seed!r} is not a registered fneg cell")
    template, _template_sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=graph_edges,
        rows=rows,
    )
    config = copy.deepcopy(template)

    # Round identity: this is a NEW treatment, not R0217's, so it declares itself.
    config["round_id"] = ROUND_ID
    config["schema"] = FNEG_TRAIN_CONFIG_SCHEMA
    config["treatment"] = {
        "name": "fneg-x4-md000",
        "recipe_schema": FNEG_RECIPE_SCHEMA,
        "low_dim_kernel": FNEG_LOW_DIM_KERNEL,
        "a": FNEG_A,
        "b": FNEG_B,
        "min_dist_curve": "a/b are the min_dist=0 UMAP curve fit; no min_dist key stored",
        "fneg_weight": FNEG_WEIGHT,
        "fneg_lo": FNEG_LO,
        "fneg_hi": FNEG_HI,
        "dose_multiplier": DOSE_MULTIPLIER,
    }

    # The kernel + curve.
    _set_path(config, ("model", "low_dim_kernel"), FNEG_LOW_DIM_KERNEL)
    _set_path(config, ("model", "a"), FNEG_A)
    _set_path(config, ("model", "b"), FNEG_B)

    # The fneg reweighting -- keys R0217's optimizer block never carried.
    _set_path_new(config, ("optimizer", "fneg_weight"), FNEG_WEIGHT)
    _set_path_new(config, ("optimizer", "fneg_lo"), FNEG_LO)
    _set_path_new(config, ("optimizer", "fneg_hi"), FNEG_HI)

    # The x4 dose.
    dose = validate_fneg_dose(updates=DOSE_MULTIPLIER * successful_updates_for_edges(
        graph_edges), edge_count=graph_edges)
    _set_path(config, ("optimizer", "successful_positive_lr_updates"), dose["successful_updates"])
    _set_path(
        config, ("execution", "target_positive_draws_per_edge"),
        dose["target_positive_draws_per_edge"],
    )
    _set_path(
        config, ("execution", "achieved_positive_draws_per_edge"),
        dose["achieved_positive_draws_per_edge"],
    )
    config["dose_registration"] = dose

    # The fneg family metadata (constant across cells; part of the invariant).
    _set_path(config, ("seed_family", "seeds"), list(SEEDS))
    _set_path(config, ("seed_family", "cells"), len(SEEDS))
    _set_path(config, ("seed_family", "canonical_seed"), CANONICAL_SEED)

    # Finally the nine seed-bearing paths for this cell.
    for path, replacement in _seed_bearing_values(seed).items():
        _set_path(config, path, replacement)

    return config, sha256_bytes(canonical_json(config))


def fneg_seed_invariant_sha256(config: Mapping[str, Any]) -> str:
    """The masked digest of an fneg config -- R0217's masker over the same nine paths."""
    return seed_invariant_sha256(config)


# --------------------------------------------------------------------------- #
# the recipe invariant -- the guard the four plants target
# --------------------------------------------------------------------------- #


def assert_registered_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless `config` is EXACTLY the pinned fneg recipe.

    The kernel/curve/fneg fields are checked against the registered constants; the
    horizon is checked against `dose_multiplier * successful_updates_for_edges(edges)`
    derived from the config's own sealed edge count, so a wrong dose is caught by the
    rule rather than by a literal. There is one way to hold.
    """
    problems: list[str] = []

    def _num(path: tuple[str, ...]) -> float:
        return float(_get_path(config, path))

    if str(_get_path(config, ("model", "low_dim_kernel"))) != FNEG_LOW_DIM_KERNEL:
        problems.append(f"model.low_dim_kernel != {FNEG_LOW_DIM_KERNEL!r}")
    if _num(("model", "a")) != FNEG_A:
        problems.append(f"model.a != {FNEG_A!r}")
    if _num(("model", "b")) != FNEG_B:
        problems.append(f"model.b != {FNEG_B!r}")
    if _num(("optimizer", "fneg_weight")) != FNEG_WEIGHT:
        problems.append(f"optimizer.fneg_weight != {FNEG_WEIGHT!r} (fneg is OFF or wrong)")
    if _num(("optimizer", "fneg_lo")) != FNEG_LO:
        problems.append(f"optimizer.fneg_lo != {FNEG_LO!r}")
    if _num(("optimizer", "fneg_hi")) != FNEG_HI:
        problems.append(f"optimizer.fneg_hi != {FNEG_HI!r}")

    edges = int(_get_path(config, ("dose_registration", "active_graph_edges")))
    expected_horizon = DOSE_MULTIPLIER * successful_updates_for_edges(edges)
    horizon = int(_get_path(config, ("optimizer", "successful_positive_lr_updates")))
    if horizon != expected_horizon:
        problems.append(
            f"optimizer.successful_positive_lr_updates {horizon} != "
            f"{DOSE_MULTIPLIER} * base_horizon = {expected_horizon} (wrong dose)"
        )
    expected_draws = DOSE_MULTIPLIER * TARGET_POSITIVE_DRAWS_PER_EDGE
    draws = _num(("execution", "target_positive_draws_per_edge"))
    if draws != expected_draws:
        problems.append(
            f"execution.target_positive_draws_per_edge {draws!r} != {expected_draws!r}"
        )

    if problems:
        raise Round0265RecipeError(
            "R0265 config is not the registered fneg recipe: " + "; ".join(problems)
        )
    # The recipe holds. Re-derive the dose receipt so the invariant carries the
    # validated arithmetic rather than trusting the config's stored copy.
    dose = validate_fneg_dose(updates=horizon, edge_count=edges)
    return {
        "recipe_schema": FNEG_RECIPE_SCHEMA,
        "low_dim_kernel": FNEG_LOW_DIM_KERNEL,
        "a": FNEG_A,
        "b": FNEG_B,
        "fneg_weight": FNEG_WEIGHT,
        "fneg_lo": FNEG_LO,
        "fneg_hi": FNEG_HI,
        "dose_multiplier": DOSE_MULTIPLIER,
        "successful_positive_lr_updates": horizon,
        "base_horizon": successful_updates_for_edges(edges),
        "target_positive_draws_per_edge": expected_draws,
        "achieved_positive_draws_per_edge": dose["achieved_positive_draws_per_edge"],
        "seed_invariant_sha256": fneg_seed_invariant_sha256(config),
        "masked_config_bytes": len(masked_config_bytes(config)),
        "old_predicate_this_replaces": OLD_RECIPE_PREDICATE,
    }


def old_recipe_predicate(config: Mapping[str, Any]) -> bool:
    """The shape check this guard replaces: reads the seed and the map dimension only."""
    try:
        seed = int(_get_path(config, ("seed",)))
        out_dim = int(_get_path(config, ("model", "output_dimension")))
    except Round0265RecipeError:
        return False
    return seed in set(SEEDS) and out_dim == OUTPUT_DIMENSION


def _honest_recipe_config() -> dict[str, Any]:
    """A canonical, honest fneg config for the CANONICAL_SEED, built on the sealed sigs.

    Used as the baseline the controls perturb. It reads no artifact: the sealed R0216
    substrate/graph signatures are registered constants in R0221.
    """
    config, _sha = fneg_train_config(
        seed=CANONICAL_SEED,
        graph_signature=SEALED_GRAPH_SIGNATURE,
        graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
        substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return config


def recipe_refusal_controls() -> dict[str, Any]:
    """Plant four recipe defects against the SHIPPED predicate; check the old one too.

    Every control calls the shipped `assert_registered_recipe`; none re-implements it,
    so a predicate that returned unconditionally would fail all four. Each plant also
    proves the OLD predicate ACCEPTS it -- the seed/dimension shape check cannot see the
    recipe drift.
    """
    honest = _honest_recipe_config()
    controls: list[dict[str, Any]] = []

    def _plant(name: str, description: str, mutate) -> None:
        planted = copy.deepcopy(honest)
        mutate(planted)
        refused = False
        error = None
        try:
            assert_registered_recipe(planted)
        except Round0265RecipeError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
            "old_predicate_accepted": bool(old_recipe_predicate(planted)),
        })

    def _wrong_kernel(cfg: dict[str, Any]) -> None:
        # legacy_lp with a=b=1.0 -- R0217's own kernel, the treatment fneg replaces.
        _set_path(cfg, ("model", "low_dim_kernel"), "legacy_lp")
        _set_path(cfg, ("model", "a"), 1.0)
        _set_path(cfg, ("model", "b"), 1.0)

    def _wrong_dose(cfg: dict[str, Any]) -> None:
        # x1 horizon -- the base low dose, not the x4 recipe.
        base = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
        _set_path(cfg, ("optimizer", "successful_positive_lr_updates"), base)
        _set_path(
            cfg, ("execution", "target_positive_draws_per_edge"),
            TARGET_POSITIVE_DRAWS_PER_EDGE,
        )

    def _fneg_off(cfg: dict[str, Any]) -> None:
        # fneg turned off -- the core's default, byte-inert on the loss.
        _set_path(cfg, ("optimizer", "fneg_weight"), 0.0)

    def _wrong_band(cfg: dict[str, Any]) -> None:
        # the fog band moved -- a different treatment wearing the same weight.
        _set_path(cfg, ("optimizer", "fneg_lo"), 0.05)
        _set_path(cfg, ("optimizer", "fneg_hi"), 0.5)

    _plant("wrong_kernel", "legacy_lp kernel with a=b=1.0 -- R0217's kernel, not fneg's umap curve", _wrong_kernel)
    _plant("wrong_dose", "the x1 base horizon instead of the x4 recipe dose", _wrong_dose)
    _plant("fneg_off", "fneg_weight=0.0 -- the reweighting turned off, byte-inert on the loss", _fneg_off)
    _plant("wrong_band", "fneg_lo/hi moved to 0.05/0.5 -- a different fog band", _wrong_band)

    honest_refused = False
    honest_error = None
    try:
        assert_registered_recipe(honest)
    except Round0265RecipeError as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_predicate_refused"] for item in controls
        ),
        "the_old_predicate_accepted_every_one": all(
            item["old_predicate_accepted"] for item in controls
        ),
        "the_honest_recipe_still_passes": not honest_refused,
        "honest_error": honest_error,
        "old_predicate": OLD_RECIPE_PREDICATE,
        "note": (
            "every control calls the SHIPPED assert_registered_recipe; none "
            "re-implements it. A predicate that returned unconditionally would fail "
            "all four."
        ),
    }


def assert_family_shares_one_recipe(
    configs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless the thirteen cells are the fneg recipe, seed aside.

    Each config is proved to carry the recipe by `assert_registered_recipe`, then the
    masked digests must collapse to ONE value (the recipe outside the seed) while the
    full config digests are thirteen distinct values (thirteen real cells).
    """
    observed = {int(seed) for seed in configs}
    if observed != set(SEEDS):
        raise Round0265FamilyError(
            f"R0265 family must be exactly seeds {list(SEEDS)}, got {sorted(observed)}"
        )
    masked: dict[int, str] = {}
    full: dict[str, str] = {}
    for seed in SEEDS:
        config = dict(configs[seed])
        assert_registered_recipe(config)
        for path, want in _seed_bearing_values(seed).items():
            if _get_path(config, path) != want:
                raise Round0265FamilyError(
                    f"R0265 seed {seed} has {'.'.join(path)}={_get_path(config, path)!r}, "
                    f"expected {want!r}"
                )
        masked[seed] = sha256_bytes(masked_config_bytes(config))
        full[str(seed)] = sha256_bytes(canonical_json(config))
    unique = sorted(set(masked.values()))
    if len(unique) != 1:
        raise Round0265FamilyError(
            f"R0265 cells differ outside the seed: masked digests {masked}"
        )
    if len(set(full.values())) != len(SEEDS):
        raise Round0265FamilyError("R0265 produced duplicate cell configs")
    return {
        "seeds": list(SEEDS),
        "cells": len(SEEDS),
        "seed_invariant_sha256": unique[0],
        "per_seed_config_sha256": full,
        "masked_config_seed_placeholder": SEED_PLACEHOLDER,
        "n_pooled": len(SEEDS),
        "identity_bound_at_n": identity_bound(len(SEEDS)),
        "all_thirteen_share_one_recipe_digest": len(unique) == 1,
        "thirteen_distinct_cell_configs": len(set(full.values())) == len(SEEDS),
    }


# --------------------------------------------------------------------------- #
# the import-closure seal (reused R0255 machinery, retargeted at the fneg core)
# --------------------------------------------------------------------------- #


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_closure_hashes(
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, dict[str, Any]]:
    """Hash the files behind the modules this process actually imported."""
    import importlib

    observed: dict[str, dict[str, Any]] = {}
    for name in modules:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", None)
        if not path or not os.path.exists(path):
            raise Round0265TreatmentError(
                f"R0265 closure module {name!r} has no readable source file"
            )
        observed[name] = {
            "path": os.path.realpath(path),
            "bytes": os.path.getsize(path),
            "sha256": file_sha256(path),
        }
    return observed


def assert_runtime_closure_matches_seal(
    *,
    sealed: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, Any]:
    """Refuse unless every module in the registered closure ran the sealed bytes."""
    registered = tuple(modules)
    if not registered:
        raise Round0265TreatmentError(
            "R0265 treatment closure is empty: a closure with no members certifies "
            "every possible release and is the vacuous case this guard exists for"
        )
    files = dict(sealed.get("files") or {})
    if not files:
        raise Round0265TreatmentError("R0265 sealed treatment closure carries no files")
    if set(files) != set(registered):
        raise Round0265TreatmentError(
            f"R0265 sealed closure membership {sorted(files)} is not the registered "
            f"closure {sorted(registered)}"
        )
    if set(observed) != set(registered):
        raise Round0265TreatmentError(
            f"R0265 runtime closure membership {sorted(observed)} is not the registered "
            f"closure {sorted(registered)}"
        )
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for name in registered:
        want = str(dict(files[name]).get("sha256_at_release") or "")
        got = str(dict(observed[name]).get("sha256") or "")
        if not _SHA256.match(want) or not _SHA256.match(got):
            problems.append(f"{name}: malformed digest")
            rows.append({"module": name, "matches": False, "reason": "malformed"})
            continue
        matches = want == got
        if not matches:
            problems.append(f"{name}: {got} != sealed {want}")
        rows.append({"module": name, "sealed_sha256": want, "runtime_sha256": got, "matches": matches})
    if problems:
        raise Round0265TreatmentError(
            "R0265 the training closure that ran is not the sealed release: "
            + "; ".join(problems)
        )
    return {
        "schema": CLOSURE_SCHEMA,
        "modules": list(registered),
        "modules_checked": len(rows),
        "every_module_ran_the_sealed_bytes": all(row["matches"] for row in rows),
        "rows": rows,
        "what_it_does_not_prove": (
            "that a changed file changed no number. It proves the bytes that ran are "
            "the bytes sealed at prepare time -- in particular that the fneg-merged "
            "core.py (commit 9794b4b) is the one the cell trained under."
        ),
    }


def treatment_closure_controls(
    *, sealed: Mapping[str, Any], observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Plant closure defects against the SHIPPED predicate. Every control calls it."""
    honest_observed = {name: dict(value) for name, value in observed.items()}
    controls: list[dict[str, Any]] = []

    def _run(name: str, description: str, **kwargs: Any) -> None:
        refused = False
        error = None
        try:
            assert_runtime_closure_matches_seal(**kwargs)
        except Round0265TreatmentError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
        })

    victim = TRAIN_CLOSURE_MODULES[1]  # the fneg-merged core.py
    mutated = {name: dict(value) for name, value in honest_observed.items()}
    mutated[victim]["sha256"] = "0" * 64
    _run("content_drift", "core.py ran bytes that are not the sealed fneg-merged core",
         sealed=sealed, observed=mutated)

    dropped = {name: dict(value) for name, value in honest_observed.items() if name != victim}
    _run("missing_module", "the fneg-merged core is absent from the runtime map",
         sealed=sealed, observed=dropped)

    extra = {name: dict(value) for name, value in honest_observed.items()}
    extra["basemap.round0265_missing"] = {"path": __file__, "bytes": 0, "sha256": "1" * 64}
    _run("extra_module", "the runtime map carries a module the seal does not",
         sealed=sealed, observed=extra)

    malformed = {name: dict(value) for name, value in honest_observed.items()}
    malformed[victim]["sha256"] = "not-a-digest"
    _run("malformed_digest", "a digest that is not a SHA-256 hex string",
         sealed=sealed, observed=malformed)

    _run("empty_closure", "an EMPTY registered closure -- the vacuous accept",
         sealed={"files": {}}, observed={}, modules=())

    honest_refused = False
    honest_error = None
    try:
        assert_runtime_closure_matches_seal(sealed=sealed, observed=honest_observed)
    except Round0265TreatmentError as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_predicate_refused"] for item in controls
        ),
        "the_honest_closure_still_passes": not honest_refused,
        "honest_error": honest_error,
    }


__all__ = [
    "CANONICAL_SEED",
    "CAPABILITIES",
    "CAPABILITY_TEMPLATE",
    "CLOSURE_SCHEMA",
    "DOSE_MULTIPLIER",
    "FNEG_A",
    "FNEG_B",
    "FNEG_BASE_HORIZON",
    "FNEG_HI",
    "FNEG_HORIZON",
    "FNEG_LO",
    "FNEG_LOW_DIM_KERNEL",
    "FNEG_RECIPE_SCHEMA",
    "FNEG_TARGET_DRAWS_PER_EDGE",
    "FNEG_TRAIN_CONFIG_SCHEMA",
    "FNEG_WEIGHT",
    "N_TARGET",
    "OLD_RECIPE_PREDICATE",
    "RECIPE_DOSE_PATHS",
    "RECIPE_FNEG_PATHS",
    "RECIPE_KERNEL_PATHS",
    "ROUND_ID",
    "Round0265FamilyError",
    "Round0265RecipeError",
    "Round0265TreatmentError",
    "SEEDS",
    "TRAIN_CLOSURE_MODULES",
    "assert_family_shares_one_recipe",
    "assert_registered_recipe",
    "assert_runtime_closure_matches_seal",
    "capability_for_seed",
    "exact_cell_id",
    "file_sha256",
    "fneg_seed_invariant_sha256",
    "fneg_train_config",
    "old_recipe_predicate",
    "recipe_refusal_controls",
    "runtime_closure_hashes",
    "treatment_closure_controls",
    "validate_fneg_dose",
]

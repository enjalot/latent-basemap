"""R0266's treatment closure: the promoted fneg recipe on the host-int8 X path.

R0266 is a **single-delta extension of R0265's pinned fneg recipe**. R0265
registered and gated the promoted recipe (umap kernel a=1.9328/b=0.7905, dose x4,
fneg_weight=1.0 band [0.1,0.4], UNIFORM positive-edge sampling on the device path)
across an n=13 seed family and proved int8 fidelity nowhere -- its cells ran the
fp32 device path (2M fits VRAM; `x_residency="device_fp16"`). review-0262-01 left
int8 fidelity proven only at the GRADIENT level (a 600-update proxy) and explicitly
demanded a MAP-LEVEL cell before any long int8-path train:

    "The train should not run until int8 fidelity is validated at the map level --
     the 2M/6.25M cell against the registered floors, the only remaining GPU item
     and the honest precondition."  (review-0262-01 §H.1)

R0266 IS that cell. The owner ruled step-5 = x2 dose staged through 50M on the
host-int8 path (50M exceeds the ~20M fp32 device ceiling), so a map-level int8
fidelity pass on the promoted recipe is the BLOCKING critical path before 50M.

**The delta, pinned.** Exactly ONE field moves off R0265's seed-42 recipe:

* `x_residency`  `"device_fp16"` -> `"host_int8"`  (int8 rows + fp16 per-row
  scales on the DeviceEdgeSampler uniform path). Everything else -- kernel, curve,
  dose, fneg reweighting, uniform positive sampling, the device execution routing,
  the sealed R0216 substrate + exact k15 graph, seed 42 -- is R0265's recipe
  byte-for-byte, proved by delegating to `round0265_fneg_treatment` and asserting
  the whole R0265 recipe closure holds BEFORE the int8 delta is checked.

`x_residency` is not a ParametricUMAP constructor arg the config->model bridge
threads (`_new_model` predates it), so the R0266 `_build` sets
`model.x_residency = "host_int8"` on the instance -- exactly as R0265's
`_build_fneg_model` sets the fneg fields and the manifest posture -- and asserts it
after build (fail closed). At train time the pipeline stamp must report
`weighted_effective=False`, `positive_sampling="uniform"` AND
`x_residency="host_int8"`, or the cell refuses to seal (a silent fp32 fallback is
the exact class of drift the R0265 uniform-stamp tripwire caught for the sampler).

**The guard -- `assert_registered_int8_recipe`.** Delegates the entire R0265 recipe
proof to `round0265_fneg_treatment.assert_registered_recipe` (so every one of the
six R0265 recipe defects is still refused), then adds the int8 delta: the config's
`execution.x_residency` and its declared `expected_pipeline_stamp.x_residency` must
both be exactly `"host_int8"`. The refusal controls plant the int8-specific defects
(x_residency left at the fp32 default `device_fp16`, or `auto`, or a stamp that
still declares `device_fp16`) AND re-plant an R0265 recipe defect (fneg off) to
prove the delegated R0265 proof still fires through the int8 guard.

**Import-closure seal.** Reuses R0255/R0265's machinery: the round seals, at prepare
time, the SHA-256 of every source file in the training import closure and every
training node re-hashes what it imported. R0266's closure is R0265's closure PLUS
this module, so the seal proves both the fneg-merged core (commit 9794b4b) and the
R0266 int8 recipe bytes ran.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from . import round0265_fneg_treatment as R0265
from .round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from .round0217_minilm_2m_seed_family import ROWS, SEALED_DIRECTED_EDGES


ROUND_ID = "0266"

INT8_TRAIN_CONFIG_SCHEMA = "round0266-minilm-mixed-2m-fneg-x4-md000-hostint8-config-v1"
INT8_RECIPE_SCHEMA = "round0266-fneg-x4-md000-hostint8-recipe-v1"
CLOSURE_SCHEMA = "round0266-hostint8-treatment-import-closure-v1"

#: The single map of this round: R0265's canonical seed, on the host-int8 path.
SEED = R0265.CANONICAL_SEED  # 42

#: The pinned int8 delta. int8 rows + fp16 per-row scales staged to the device by
#: the DeviceEdgeSampler uniform path (the sandbox `umap-md000-x4-fneg10-hostint8`
#: arm). `"auto"` (the fp32 device default) and `"device_fp16"` are the two values
#: that would silently run the fp32 path this cell exists to escape.
X_RESIDENCY = "host_int8"

#: The R0265 recipe's fp32 stamp residency -- the value the int8 delta replaces and
#: the control plants back. NOT a gate constant; the recipe is proved by delegation.
FP32_X_RESIDENCY = "device_fp16"

CAPABILITY = "minilm-mixed-2m-fneg-x4-md000-hostint8-seed42-r0266-v1"

#: The path the recipe overwrites on top of R0265's seed-42 config, other than the
#: expected_pipeline_stamp mirror. Named so the delta has an auditable membership.
INT8_RESIDENCY_PATHS: tuple[tuple[str, ...], ...] = (
    ("execution", "x_residency"),
    ("execution", "expected_pipeline_stamp", "x_residency"),
)

#: R0266's training import closure: R0265's closure PLUS this module. Reuses R0265's
#: closure primitives (they take an explicit `modules` argument); the added member
#: seals the int8 recipe bytes beside the fneg-merged core R0265 already sealed.
TRAIN_CLOSURE_MODULES: tuple[str, ...] = (
    *R0265.TRAIN_CLOSURE_MODULES,
    "basemap.round0266_int8_treatment",
)


class Round0266RecipeError(RuntimeError):
    """The config is not the registered R0266 host-int8 fneg recipe."""


# --------------------------------------------------------------------------- #
# path helpers (mirror R0265's, kept local so the module is self-contained)
# --------------------------------------------------------------------------- #


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = value
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise Round0266RecipeError(f"R0266 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0266RecipeError(f"R0266 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0266RecipeError(f"R0266 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def _set_path_new(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0266RecipeError(f"R0266 config is missing parent of {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise Round0266RecipeError(f"R0266 config parent of {'.'.join(path)} is not a map")
    cursor[path[-1]] = replacement


# --------------------------------------------------------------------------- #
# building the recipe config -- R0265's seed-42 config + the int8 delta
# --------------------------------------------------------------------------- #


def int8_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
    seed: int = SEED,
) -> tuple[dict[str, Any], str]:
    """R0265's seed-42 fneg config with `x_residency="host_int8"` applied, and nothing else.

    The base config is built and proved by `round0265_fneg_treatment.fneg_train_config`
    for the canonical seed; then exactly the two `x_residency` fields (the execution
    field the `_build` reads to set the model, and the declared pipeline stamp mirror)
    are moved to `"host_int8"`. `assert_registered_int8_recipe` proves both halves.
    """
    if int(seed) != SEED:
        raise Round0266RecipeError(
            f"R0266 is a single seed-{SEED} cell (seed-paired to R0265 seed-{SEED}); "
            f"got seed {seed!r}"
        )
    config, _sha = R0265.fneg_train_config(
        seed=int(seed),
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=graph_edges,
        rows=rows,
    )
    config = copy.deepcopy(config)

    # Round identity: this is R0265's recipe + the int8 delta, so it declares itself.
    config["round_id"] = ROUND_ID
    config["schema"] = INT8_TRAIN_CONFIG_SCHEMA
    treatment = dict(config.get("treatment") or {})
    treatment["name"] = "fneg-x4-md000-hostint8"
    treatment["recipe_schema"] = INT8_RECIPE_SCHEMA
    treatment["x_residency"] = X_RESIDENCY
    treatment["base_recipe_schema"] = R0265.FNEG_RECIPE_SCHEMA
    config["treatment"] = treatment

    # The int8 delta: the execution field the config->model bridge reads (set on the
    # model by `_build`), and the declared pipeline-stamp residency mirror. R0265's
    # config carries no execution.x_residency key, so this is a new key; the stamp
    # key exists (R0265 set it to device_fp16) and is overwritten.
    _set_path_new(config, ("execution", "x_residency"), X_RESIDENCY)
    _set_path(config, ("execution", "expected_pipeline_stamp", "x_residency"), X_RESIDENCY)

    return config, sha256_bytes(canonical_json(config))


# --------------------------------------------------------------------------- #
# the recipe invariant -- R0265's proof, then the int8 delta
# --------------------------------------------------------------------------- #


def assert_registered_int8_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless `config` is EXACTLY R0265's pinned recipe PLUS x_residency=host_int8.

    The whole R0265 recipe proof is delegated to `assert_registered_recipe`, so every
    R0265 recipe defect (wrong kernel/dose/band, fneg off, weighted sampling on, routing
    off device, a dishonest weighted stamp) is still refused -- through this guard. Then
    the int8 delta is checked: both the execution field and the declared stamp mirror must
    be exactly `"host_int8"`. There is one way to hold.
    """
    base = R0265.assert_registered_recipe(config)  # raises Round0265RecipeError on any drift

    problems: list[str] = []
    if str(_get_path(config, ("execution", "x_residency"))) != X_RESIDENCY:
        problems.append(
            f"execution.x_residency != {X_RESIDENCY!r} (a device_fp16/auto value silently "
            "runs the fp32 device path this cell exists to escape)"
        )
    stamp = _get_path(config, ("execution", "expected_pipeline_stamp"))
    if not isinstance(stamp, Mapping) or stamp.get("x_residency") != X_RESIDENCY:
        problems.append(
            "execution.expected_pipeline_stamp.x_residency != "
            f"{X_RESIDENCY!r} (the declared stamp still describes the fp32 residency)"
        )
    if problems:
        raise Round0266RecipeError(
            "R0266 config is not the registered host-int8 fneg recipe: "
            + "; ".join(problems)
        )

    recipe = dict(base)
    recipe["recipe_schema"] = INT8_RECIPE_SCHEMA
    recipe["base_recipe_schema"] = R0265.FNEG_RECIPE_SCHEMA
    recipe["x_residency"] = X_RESIDENCY
    recipe["expected_pipeline_stamp_x_residency"] = X_RESIDENCY
    recipe["int8_delta_over_r0265"] = "x_residency: device_fp16 -> host_int8 (one field)"
    return recipe


def _honest_int8_config() -> dict[str, Any]:
    """A canonical honest int8 config, built on the sealed R0216 signatures (no artifact read)."""
    config, _sha = int8_train_config(
        graph_signature=SEALED_GRAPH_SIGNATURE,
        graph_manifest_signature=SEALED_GRAPH_MANIFEST_SIGNATURE,
        substrate_signature=SEALED_SUBSTRATE_SIGNATURE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return config


def int8_recipe_refusal_controls() -> dict[str, Any]:
    """Plant the int8 delta defects AND an R0265 recipe defect against the SHIPPED guard.

    Every control calls the shipped `assert_registered_int8_recipe`; none re-implements
    it. The int8 plants (x_residency at the fp32 default, at auto, and a stamp that still
    declares device_fp16) prove the int8 delta is enforced; the re-planted R0265 defect
    (fneg off) proves the delegated R0265 recipe proof still fires through this guard.
    """
    honest = _honest_int8_config()
    controls: list[dict[str, Any]] = []

    def _plant(name: str, description: str, mutate) -> None:
        planted = copy.deepcopy(honest)
        mutate(planted)
        refused = False
        error = None
        try:
            assert_registered_int8_recipe(planted)
        except (Round0266RecipeError, R0265.Round0265RecipeError) as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
        })

    _plant(
        "x_residency_device_fp16",
        "execution.x_residency left at the fp32 default device_fp16 -- silent fp32 path",
        lambda c: _set_path(c, ("execution", "x_residency"), FP32_X_RESIDENCY),
    )
    _plant(
        "x_residency_auto",
        "execution.x_residency='auto' -- the fp32 device fast path, not host int8",
        lambda c: _set_path(c, ("execution", "x_residency"), "auto"),
    )
    _plant(
        "stamp_x_residency_device_fp16",
        "declared expected_pipeline_stamp.x_residency still device_fp16 -- dishonest stamp",
        lambda c: _set_path(
            c, ("execution", "expected_pipeline_stamp", "x_residency"), FP32_X_RESIDENCY
        ),
    )
    _plant(
        "base_recipe_fneg_off",
        "fneg_weight=0.0 -- an R0265 recipe defect, refused through the delegated proof",
        lambda c: _set_path(c, ("optimizer", "fneg_weight"), 0.0),
    )
    _plant(
        "base_recipe_weighted_sampling_on",
        "weighted_edge_sampling=True -- R0265's fuzzy sampler, refused through the delegated proof",
        lambda c: _set_path(c, ("optimizer", "weighted_edge_sampling"), True),
    )

    honest_refused = False
    honest_error = None
    try:
        assert_registered_int8_recipe(honest)
    except (Round0266RecipeError, R0265.Round0265RecipeError) as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_predicate_refused"] for item in controls
        ),
        "the_honest_recipe_still_passes": not honest_refused,
        "honest_error": honest_error,
        "note": (
            "every control calls the SHIPPED assert_registered_int8_recipe; none "
            "re-implements it. Three plants exercise the int8 delta, two re-plant an "
            "R0265 recipe defect to prove the delegated R0265 proof still refuses."
        ),
    }


# --------------------------------------------------------------------------- #
# the import-closure seal (reused R0265 primitives, R0266's module list)
# --------------------------------------------------------------------------- #

#: Reuse R0265's closure primitives verbatim; they take an explicit `modules` arg.
file_sha256 = R0265.file_sha256


def runtime_closure_hashes(
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, dict[str, Any]]:
    """Hash the files behind R0266's closure (R0265's + this module)."""
    return R0265.runtime_closure_hashes(modules)


def assert_runtime_closure_matches_seal(
    *,
    sealed: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, Any]:
    """Refuse unless every module in R0266's registered closure ran the sealed bytes."""
    return R0265.assert_runtime_closure_matches_seal(
        sealed=sealed, observed=observed, modules=modules
    )


def treatment_closure_controls(
    *, sealed: Mapping[str, Any], observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Plant closure defects against the SHIPPED R0266 predicate. Every control calls it.

    Mirrors R0265's `treatment_closure_controls` but pins R0266's module list on every
    call, so the added round0266 member is part of the sealed membership a defect is
    planted against.
    """
    honest_observed = {name: dict(value) for name, value in observed.items()}
    controls: list[dict[str, Any]] = []

    def _run(name: str, description: str, **kwargs: Any) -> None:
        refused = False
        error = None
        try:
            assert_runtime_closure_matches_seal(**kwargs)
        except R0265.Round0265TreatmentError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
        })

    victim = "basemap.round0266_int8_treatment"  # the int8 recipe bytes
    mutated = {name: dict(value) for name, value in honest_observed.items()}
    mutated[victim]["sha256"] = "0" * 64
    _run("content_drift", "the R0266 int8 recipe ran bytes that are not the sealed ones",
         sealed=sealed, observed=mutated)

    dropped = {name: dict(value) for name, value in honest_observed.items() if name != victim}
    _run("missing_module", "the R0266 int8 recipe module is absent from the runtime map",
         sealed=sealed, observed=dropped)

    extra = {name: dict(value) for name, value in honest_observed.items()}
    extra["basemap.round0266_missing"] = {"path": __file__, "bytes": 0, "sha256": "1" * 64}
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
    except R0265.Round0265TreatmentError as raised:
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
    "CAPABILITY",
    "CLOSURE_SCHEMA",
    "FP32_X_RESIDENCY",
    "INT8_RECIPE_SCHEMA",
    "INT8_RESIDENCY_PATHS",
    "INT8_TRAIN_CONFIG_SCHEMA",
    "ROUND_ID",
    "Round0266RecipeError",
    "SEED",
    "TRAIN_CLOSURE_MODULES",
    "X_RESIDENCY",
    "assert_registered_int8_recipe",
    "assert_runtime_closure_matches_seal",
    "file_sha256",
    "int8_recipe_refusal_controls",
    "int8_train_config",
    "runtime_closure_hashes",
    "treatment_closure_controls",
]

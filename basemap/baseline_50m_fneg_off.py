"""Matched 50M control for the promoted fneg treatment.

The control changes one training axis from R0267: ``optimizer.fneg_weight`` is
``0.0`` instead of ``1.0``.  Scale, substrate, graph, network, optimizer,
positive and negative samplers, dose, host-int8 residency, and seed family stay
fixed.  The constructor and predicate below make that claim executable: the
predicate rebuilds both the R0267 parent and the expected control and refuses
any additional config difference.

This is a comparison experiment, not another promotion gate.  Its result is
descriptive evidence for the effect of fneg at 50M.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from . import round0267_int8_treatment as R0267
from .round0217_minilm_2m_seed_family import seed_invariant_sha256


ROUND_ID = "0269"
STUDY_ID = "minilm-50m-fneg-off-matched-control-v1"
TRAIN_CONFIG_SCHEMA = "baseline-minilm-50m-x2-fneg-off-hostint8-config-v1"
RECIPE_SCHEMA = "baseline-minilm-50m-x2-fneg-off-hostint8-recipe-v1"
CLOSURE_SCHEMA = "baseline-minilm-50m-x2-fneg-off-treatment-closure-v1"

ROWS = R0267.ROWS
DIMENSION = R0267.DIMENSION
SEALED_DIRECTED_EDGES = R0267.SEALED_DIRECTED_EDGES
DOSE_MULTIPLIER = R0267.DOSE_MULTIPLIER
X_RESIDENCY = R0267.X_RESIDENCY
SEEDS: tuple[int, ...] = R0267.SEEDS
CANONICAL_SEED = R0267.CANONICAL_SEED

FNEG_WEIGHT = 0.0
PARENT_FNEG_WEIGHT = 1.0
CAPABILITY_TEMPLATE = "minilm-mixed-50000k-fneg-off-x2-md000-hostint8-seed{seed}-v1"
TRAIN_CLOSURE_MODULES: tuple[str, ...] = (
    *R0267.TRAIN_CLOSURE_MODULES,
    "basemap.baseline_50m_fneg_off",
)


class Baseline50MRecipeError(RuntimeError):
    """The proposed config is not the one-axis 50M fneg-off control."""


def capability_for_seed(seed: int) -> str:
    seed = int(seed)
    if seed not in SEEDS:
        raise Baseline50MRecipeError(
            f"50M fneg-off seed {seed!r} is not in the registered family {SEEDS}"
        )
    return CAPABILITY_TEMPLATE.format(seed=seed)


CAPABILITIES: tuple[str, ...] = tuple(capability_for_seed(seed) for seed in SEEDS)


def exact_cell_id(seed: int) -> str:
    return f"fneg-off-x2-md000-hostint8-50m-seed{int(seed)}"


def control_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    substrate_signature: Mapping[str, Any],
    graph_edges: int,
    rows: int,
    seed: int = CANONICAL_SEED,
) -> tuple[dict[str, Any], str]:
    """Build the R0267 config with only fneg disabled and honest control labels."""
    config, _ = R0267.int8_50m_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=graph_edges,
        rows=rows,
        seed=seed,
    )
    config = copy.deepcopy(config)
    capability = capability_for_seed(seed)
    config["round_id"] = ROUND_ID
    config["schema"] = TRAIN_CONFIG_SCHEMA
    config["capability"] = capability
    config["optimizer"]["fneg_weight"] = FNEG_WEIGHT
    config["seed_family"]["this_capability"] = capability
    treatment = dict(config["treatment"])
    treatment.update(
        {
            "name": "fneg-off-x2-md000-hostint8-50m",
            "recipe_schema": RECIPE_SCHEMA,
            "control_of": R0267.INT8_RECIPE_SCHEMA,
            "control_axis": "optimizer.fneg_weight",
            "fneg_weight": FNEG_WEIGHT,
        }
    )
    config["treatment"] = treatment
    return config, sha256_bytes(canonical_json(config))


def _signatures_from_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph = dict(config.get("graph") or {})
    input_config = dict(config.get("input") or {})
    try:
        graph_signature = {
            "canonical_path": str(graph["path"]),
            "sha256": str(graph["sha256"]),
        }
        graph_manifest_signature = {
            "canonical_path": str(graph["manifest_path"]),
            "sha256": str(graph["manifest_sha256"]),
        }
        substrate_signature = {
            "canonical_path": str(input_config["substrate_path"]),
            "sha256": str(input_config["substrate_sha256"]),
        }
    except KeyError as exc:
        raise Baseline50MRecipeError(
            f"50M fneg-off config is missing identity field {exc.args[0]!r}"
        ) from exc
    return graph_signature, graph_manifest_signature, substrate_signature


def assert_registered_control(config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless ``config`` is exactly the one-axis control of R0267."""
    try:
        seed = int(config["seed"])
        rows = int(config["input"]["rows"])
        edges = int(config["graph"]["directed_edges"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Baseline50MRecipeError("50M fneg-off config is incomplete") from exc

    graph_sig, manifest_sig, substrate_sig = _signatures_from_config(config)
    expected, _ = control_train_config(
        graph_signature=graph_sig,
        graph_manifest_signature=manifest_sig,
        substrate_signature=substrate_sig,
        graph_edges=edges,
        rows=rows,
        seed=seed,
    )
    if canonical_json(config) != canonical_json(expected):
        raise Baseline50MRecipeError(
            "50M fneg-off config differs from the generated one-axis control"
        )

    parent, _ = R0267.int8_50m_train_config(
        graph_signature=graph_sig,
        graph_manifest_signature=manifest_sig,
        substrate_signature=substrate_sig,
        graph_edges=edges,
        rows=rows,
        seed=seed,
    )
    parent_recipe = R0267.assert_registered_50m_int8_recipe(parent)
    if float(parent["optimizer"]["fneg_weight"]) != PARENT_FNEG_WEIGHT:
        raise Baseline50MRecipeError("R0267 parent no longer has fneg_weight=1.0")

    # Remove the control's provenance-only relabeling and the registered loss
    # change. The complete config must then be byte-equivalent to its parent.
    normalized = copy.deepcopy(expected)
    for key in ("round_id", "schema", "capability", "treatment"):
        normalized[key] = copy.deepcopy(parent[key])
    normalized["seed_family"]["this_capability"] = parent["seed_family"][
        "this_capability"
    ]
    normalized["optimizer"]["fneg_weight"] = parent["optimizer"]["fneg_weight"]
    if canonical_json(normalized) != canonical_json(parent):
        raise Baseline50MRecipeError(
            "50M control has a training delta beyond optimizer.fneg_weight"
        )

    recipe = dict(parent_recipe)
    recipe.update(
        {
            "recipe_schema": RECIPE_SCHEMA,
            "control_of": R0267.INT8_RECIPE_SCHEMA,
            "round_id": ROUND_ID,
            "rows": ROWS,
            "dose_multiplier": DOSE_MULTIPLIER,
            "x_residency": X_RESIDENCY,
            "fneg_weight": FNEG_WEIGHT,
            "fneg_active": False,
            "loss_branch": "unweighted_binary_cross_entropy",
            "seed_invariant_sha256": control_seed_invariant_sha256(config),
            "only_treatment_delta": {
                "path": "optimizer.fneg_weight",
                "parent": PARENT_FNEG_WEIGHT,
                "control": FNEG_WEIGHT,
            },
            "provenance_fields_relabelled": [
                "round_id",
                "schema",
                "capability",
                "seed_family.this_capability",
                "treatment",
            ],
        }
    )
    return recipe


def control_seed_invariant_sha256(config: Mapping[str, Any]) -> str:
    return seed_invariant_sha256(config)


def assert_family_shares_one_recipe(
    configs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(int(seed) for seed in configs) != set(SEEDS):
        raise Baseline50MRecipeError(
            f"50M fneg-off family must contain exactly seeds {SEEDS}"
        )
    digests: dict[int, str] = {}
    for seed, config in configs.items():
        assert_registered_control(config)
        digests[int(seed)] = control_seed_invariant_sha256(config)
    unique = set(digests.values())
    if len(unique) != 1:
        raise Baseline50MRecipeError("50M fneg-off cells do not share one recipe")
    return {
        "seeds": list(SEEDS),
        "n": len(SEEDS),
        "seed_invariant_sha256": next(iter(unique)),
        "per_seed": {str(seed): digest for seed, digest in sorted(digests.items())},
    }


def _honest_config(seed: int = CANONICAL_SEED) -> dict[str, Any]:
    config, _ = control_train_config(
        graph_signature={"canonical_path": "/sealed/50m/graph.npz", "sha256": "b" * 64},
        graph_manifest_signature={
            "canonical_path": "/sealed/50m/qualified-graph.json",
            "sha256": "c" * 64,
        },
        substrate_signature={
            "canonical_path": "/sealed/50m/substrate.f32.npy",
            "sha256": "a" * 64,
        },
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
        seed=seed,
    )
    return config


def recipe_refusal_controls() -> dict[str, Any]:
    """Exercise the shipped predicate against common comparison-confounding drifts."""
    honest = _honest_config()
    controls: list[dict[str, Any]] = []

    def plant(name: str, description: str, mutate) -> None:
        candidate = copy.deepcopy(honest)
        mutate(candidate)
        try:
            assert_registered_control(candidate)
        except RuntimeError as exc:
            controls.append(
                {
                    "control": name,
                    "plants": description,
                    "refused": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            controls.append(
                {"control": name, "plants": description, "refused": False, "error": None}
            )

    plant(
        "fneg_left_on",
        "the R0267 treatment rather than the fneg-off control",
        lambda value: value["optimizer"].__setitem__("fneg_weight", PARENT_FNEG_WEIGHT),
    )
    plant(
        "dose_changed",
        "one fewer successful update than the matched x2 horizon",
        lambda value: value["optimizer"].__setitem__(
            "successful_positive_lr_updates",
            int(value["optimizer"]["successful_positive_lr_updates"]) - 1,
        ),
    )
    plant(
        "weighted_sampling",
        "weighted positive-edge sampling instead of the matched uniform sampler",
        lambda value: value["optimizer"].__setitem__("weighted_edge_sampling", True),
    )
    plant(
        "fp16_residency",
        "device-fp16 X instead of the matched host-int8 path",
        lambda value: value["execution"].__setitem__("x_residency", "device_fp16"),
    )

    honest_passes = True
    honest_error = None
    try:
        assert_registered_control(honest)
    except RuntimeError as exc:
        honest_passes = False
        honest_error = f"{type(exc).__name__}: {exc}"
    return {
        "controls": controls,
        "every_planted_defect_was_refused": all(item["refused"] for item in controls),
        "the_honest_control_still_passes": honest_passes,
        "honest_error": honest_error,
    }


def runtime_closure_hashes(
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, dict[str, Any]]:
    return R0267.runtime_closure_hashes(modules)


def assert_runtime_closure_matches_seal(
    *,
    sealed: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, Any]:
    return R0267.assert_runtime_closure_matches_seal(
        sealed=sealed,
        observed=observed,
        modules=modules,
    )


def treatment_closure_controls(
    *, sealed: Mapping[str, Any], observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    honest = {name: dict(value) for name, value in observed.items()}
    controls: list[dict[str, Any]] = []

    def run(name: str, candidate: Mapping[str, Mapping[str, Any]]) -> None:
        refused = False
        error = None
        try:
            assert_runtime_closure_matches_seal(sealed=sealed, observed=candidate)
        except RuntimeError as exc:
            refused = True
            error = f"{type(exc).__name__}: {exc}"
        controls.append({"control": name, "refused": refused, "error": error})

    mutated = {name: dict(value) for name, value in honest.items()}
    mutated["basemap.baseline_50m_fneg_off"]["sha256"] = "0" * 64
    run("control_source_changed", mutated)
    run(
        "control_source_missing",
        {name: value for name, value in honest.items() if name != "basemap.baseline_50m_fneg_off"},
    )
    honest_passes = True
    honest_error = None
    try:
        assert_runtime_closure_matches_seal(sealed=sealed, observed=honest)
    except RuntimeError as exc:
        honest_passes = False
        honest_error = f"{type(exc).__name__}: {exc}"
    return {
        "controls": controls,
        "every_planted_defect_was_refused": all(item["refused"] for item in controls),
        "the_honest_closure_still_passes": honest_passes,
        "honest_error": honest_error,
    }


__all__ = [
    "CANONICAL_SEED",
    "CAPABILITIES",
    "CLOSURE_SCHEMA",
    "DIMENSION",
    "DOSE_MULTIPLIER",
    "FNEG_WEIGHT",
    "RECIPE_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "SEALED_DIRECTED_EDGES",
    "SEEDS",
    "STUDY_ID",
    "TRAIN_CLOSURE_MODULES",
    "TRAIN_CONFIG_SCHEMA",
    "X_RESIDENCY",
    "Baseline50MRecipeError",
    "assert_family_shares_one_recipe",
    "assert_registered_control",
    "assert_runtime_closure_matches_seal",
    "capability_for_seed",
    "control_seed_invariant_sha256",
    "control_train_config",
    "exact_cell_id",
    "recipe_refusal_controls",
    "runtime_closure_hashes",
    "treatment_closure_controls",
]

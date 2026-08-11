"""R0257 — R0217's treatment moved onto a Phase 2 ladder rung, and the rule that
keeps the judged maps out of the judging family.

Two things live here and they are deliberately in one module, because they are the
two halves of the same rule.

**The config.** A rung cell is R0217's 2M cell with exactly three axes moved: the
seed, the graph, and the rung. `SEED_BEARING_PATHS` (R0217) and
`GRAPH_BEARING_PATHS` (R0223) are imported, never re-typed; `RUNG_BEARING_PATHS` is
this round's addition and names every field whose *value* is a function of the row
count. `rung_invariant_sha256` masks all three sets, and a rung config that does not
reproduce R0217's template digest under that mask refuses to train. That digest --
not a prose claim -- is what "the treatment is unchanged" means here.

Two strings are deliberately NOT masked and NOT changed, because they are the
*pipeline's* identity rather than a claim about the universe:
`execution.required_pipeline` / the stamp's `pipeline` and `schema`
(`host_weighted_minilm_mixed_2m`, R0217's class), and the stamp's
`source_representation` (`minilm-mixed-2m-fp32`, a literal constant inside
`MiniLMHostFp32EndpointArray.execution_stamp`). The runtime emits those verbatim at
any N, so the expected stamp must carry them for the comparison to be a real check.
They describe the code path, not the row count, and this docstring is the disclosure.

**The family rule.** `plan-minilm-100m-v2.md`: *a gate's family never grows to
include the maps it judges.* This module does not re-implement that guard -- it
calls the SHIPPED `round0255_treatment.assert_family_is_2m_only` and additionally
asserts that no rung cell id appears among the cells any floor was fitted on. The
positive controls plant each of this round's own rung maps into the family and
require the shipped guard to refuse each one while the v0 predicate it replaces
accepts every one.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from basemap.round0217_minilm_2m_seed_family import (
    DIMENSION,
    GRAPH_K,
    SEED_BEARING_PATHS,
    SEED_PLACEHOLDER,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    achieved_draws_per_edge,
    dose_quantum,
    performance_windows,
    successful_updates_for_edges,
    train_config as r0217_train_config,
    validate_dose,
)
from basemap.round0223_cuvs_graph_map import (
    GRAPH_BEARING_PATHS,
    GRAPH_PLACEHOLDER,
)
from basemap.round0255_treatment import (
    Round0255FamilyError,
    assert_family_is_2m_only,
    old_family_predicate,
)

ROUND_ID = "0257"

#: The rung this round trains. Chosen and justified in `round-0257-2026-08-11.md`
#: BEFORE the run: it is the only rung whose graph was qualified against exact
#: brute-force k15 truth over *all* its rows, and it is a 3.125x step from the only
#: N at which this program has ever measured a parametric-UMAP train.
RUNG_ROWS = 6_250_000
RUNG_SLUG = "6250k"

#: R0233's sealed rung artifacts. Values are asserted against the sealed receipts
#: at run time; they are registered here so a swapped input is a contract failure
#: rather than a silent substitution.
RUNG_SOURCE_ROUND_ID = "0233"
SUBSTRATE_CAPABILITY = f"minilm-mixed-{RUNG_SLUG}-substrate-and-reserves-v1"
SUBSTRATE_SCHEMA = f"round0233-minilm-mixed-{RUNG_SLUG}-substrate-v1"
GRAPH_CAPABILITY = f"minilm-mixed-{RUNG_SLUG}-cluster-spill-k15-fuzzy-graph-v1"
GRAPH_SCHEMA = f"round0233-minilm-mixed-{RUNG_SLUG}-k15-fuzzy-graph-v1"
SEALED_RUNG_DIRECTED_EDGES = 153_872_500

#: R0233's sealed qualification of that graph, re-read and re-asserted in the node.
SEALED_TIE_AWARE_RECALL = 0.9999016
SEALED_ZERO_DEGREE_ROWS = 0

SEEDS: tuple[int, ...] = (42, 43, 44)
TEMPLATE_SEED = 42

#: The derived horizon at this rung. `successful_updates_for_edges` is R0184's rule
#: and has no freedom; this constant exists so a queue that binds a different number
#: is refused, not so the number is chosen.
REGISTERED_SUCCESSFUL_UPDATES = successful_updates_for_edges(SEALED_RUNG_DIRECTED_EDGES)
#: Round-level ceiling on the derived horizon. 300,000 is above the derived 255,142
#: and below the 12.5M rung's, so a mis-bound graph cannot silently buy GPU hours.
REGISTERED_UPDATE_BOUND = 300_000

TRAIN_SCHEMA = f"round0257-minilm-mixed-{RUNG_SLUG}-ladder-map-train-v1"
PRODUCTION_CONFIG_SCHEMA = (
    f"round0257-minilm-mixed-{RUNG_SLUG}-ladder-map-production-config-v1"
)
PANEL_SCHEMA = f"round0257-minilm-mixed-{RUNG_SLUG}-ladder-panel-v1"
VERDICT_SCHEMA = f"round0257-minilm-mixed-{RUNG_SLUG}-rung-gate-verdict-v1"
PANEL_CAPABILITY = f"minilm-mixed-{RUNG_SLUG}-ladder-panel-v1"
VERDICT_CAPABILITY = f"minilm-mixed-{RUNG_SLUG}-rung-gate-verdict-v1"

GATE_REGISTERABLE_HERE = False

RUNG_PLACEHOLDER = "<rung-bearing>"

#: Every field whose value is a function of the row count or of which rung's sealed
#: substrate is bound. Masked alongside the seed- and graph-bearing sets so what
#: remains is the treatment itself.
RUNG_BEARING_PATHS: tuple[tuple[str, ...], ...] = (
    ("schema",),
    ("round_id",),
    ("family_invariant", "rows"),
    ("input", "rows"),
    ("input", "representation"),
    ("input", "substrate_path"),
    ("input", "substrate_sha256"),
    ("execution", "expected_pipeline_stamp", "negative_sampling"),
    ("execution", "expected_pipeline_stamp", "compact_retained_rows"),
    ("seed_family", "seeds"),
    ("seed_family", "cells"),
    ("seed_family", "canonical_seed"),
    ("seed_family", "cells_required_for_gate"),
    ("seed_family", "gate_note"),
)

#: Not masked, on purpose. Each is a constant of the code path, emitted verbatim by
#: the runtime at any N, so leaving it in the invariant makes the stamp comparison a
#: real check rather than a tautology.
PIPELINE_IDENTITY_STRINGS_NOT_MASKED = {
    "execution.required_pipeline": "host_weighted_minilm_mixed_2m",
    "execution.expected_pipeline_stamp.schema": (
        "round0217-host-weighted-minilm-mixed-2m-pipeline-v1"
    ),
    "execution.expected_pipeline_stamp.pipeline": "host_weighted_minilm_mixed_2m",
    "execution.expected_pipeline_stamp.source_representation": (
        "minilm-mixed-2m-fp32"
    ),
}
PIPELINE_IDENTITY_NOTE = (
    "these four strings name R0217's training pipeline class and its literal "
    "execution-stamp constants, not the row count. `MiniLMHostFp32EndpointArray."
    "execution_stamp` returns 'minilm-mixed-2m-fp32' at every N. They are left "
    "unmasked so the expected-stamp comparison is a real check; they are NOT a "
    "claim that this rung has 2,000,000 rows."
)

FAMILY_RULE = (
    "BINDING (plan-minilm-100m-v2.md; owner ruling 2026-08-11): the Phase 3 gate is "
    "fitted on the 2M universe and judges rung maps from OUTSIDE. A rung map is "
    "scored AGAINST the registered floors and is NEVER added to the cells any floor "
    "is fitted on. Adding one re-creates the v0 wrong pass exactly, where R0195/"
    "R0204 qualified on R0161's own fitting cells and max|z| <= (n-1)/sqrt(n) "
    "guaranteed the result before any map existed."
)


class Round0257Error(RuntimeError):
    """The registered R0257 rung contract changed."""


class Round0257FamilyError(RuntimeError):
    """A rung map reached the family a floor is fitted on."""


# --------------------------------------------------------------------------- #
# identities
# --------------------------------------------------------------------------- #


def rung_cell_id(seed: int) -> str:
    """The cell id the shipped R0255 family-purity guard already names."""
    if int(seed) not in SEEDS:
        raise Round0257Error(f"R0257 seed {seed!r} is not a registered rung cell")
    return f"ladder-{RUNG_SLUG}-h2048-seed{int(seed)}"


def map_capability(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0257Error(f"R0257 seed {seed!r} is not a registered rung cell")
    return f"minilm-mixed-{RUNG_SLUG}-ladder-map-seed{int(seed)}-v1"


def rung_cell_ids() -> tuple[str, ...]:
    return tuple(rung_cell_id(seed) for seed in SEEDS)


# --------------------------------------------------------------------------- #
# the config
# --------------------------------------------------------------------------- #


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0257Error(f"R0257 config is missing {'.'.join(path)}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0257Error(f"R0257 config is missing {'.'.join(path)}")
    cursor[path[-1]] = replacement


def rung_invariant_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Mask the seed-, graph- and rung-bearing fields; what remains is treatment."""
    projected = copy.deepcopy(dict(config))
    for path in SEED_BEARING_PATHS:
        _set_path(projected, path, SEED_PLACEHOLDER)
    for path in GRAPH_BEARING_PATHS:
        _set_path(projected, path, GRAPH_PLACEHOLDER)
    for path in RUNG_BEARING_PATHS:
        _set_path(projected, path, RUNG_PLACEHOLDER)
    return projected


def rung_invariant_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(rung_invariant_projection(config)))


def r0217_template(
    *,
    substrate_signature: Mapping[str, Any],
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """R0217's own 2M cell, built by R0217's own function. The comparison anchor."""
    from basemap.round0217_minilm_2m_seed_family import (
        ROWS as R0217_ROWS,
        SEALED_DIRECTED_EDGES as R0217_EDGES,
    )

    template, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=R0217_EDGES,
        rows=R0217_ROWS,
    )
    return template


def rung_bearing_values(
    *,
    rows: int,
    substrate_signature: Mapping[str, Any],
) -> dict[tuple[str, ...], Any]:
    rows = int(rows)
    return {
        ("schema",): TRAIN_SCHEMA,
        ("round_id",): ROUND_ID,
        ("family_invariant", "rows"): rows,
        ("input", "rows"): rows,
        ("input", "representation"): (
            f"sealed-R{RUNG_SOURCE_ROUND_ID}-{RUNG_SLUG}-fp32-substrate"
        ),
        ("input", "substrate_path"): str(substrate_signature["canonical_path"]),
        ("input", "substrate_sha256"): str(substrate_signature["sha256"]),
        ("execution", "expected_pipeline_stamp", "negative_sampling"): (
            f"uniform-{rows}-substrate-rows-nonself"
        ),
        ("execution", "expected_pipeline_stamp", "compact_retained_rows"): rows,
        ("seed_family", "seeds"): list(SEEDS),
        ("seed_family", "cells"): len(SEEDS),
        ("seed_family", "canonical_seed"): TEMPLATE_SEED,
        ("seed_family", "cells_required_for_gate"): 0,
        ("seed_family", "gate_note"): (
            "these cells are NOT a gate family and never become one. They are "
            "JUDGED by the registered n = 29 MAD_n floors fitted on the 2M "
            "universe. " + FAMILY_RULE
        ),
    }


def seed_bearing_values(seed: int) -> dict[tuple[str, ...], Any]:
    seed = int(seed)
    capability = map_capability(seed)
    return {
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


def graph_bearing_values(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    dose: Mapping[str, Any],
    updates: int,
) -> dict[tuple[str, ...], Any]:
    graph_edges = int(graph_edges)
    return {
        ("graph", "capability"): GRAPH_CAPABILITY,
        ("graph", "source_round"): RUNG_SOURCE_ROUND_ID,
        ("graph", "path"): str(graph_signature["canonical_path"]),
        ("graph", "sha256"): str(graph_signature["sha256"]),
        ("graph", "manifest_path"): str(graph_manifest_signature["canonical_path"]),
        ("graph", "manifest_sha256"): str(graph_manifest_signature["sha256"]),
        ("graph", "directed_edges"): graph_edges,
        ("graph", "exactness"): (
            "approximate cluster-spill-nnd at c=16, spill 8, qualified by R0233 "
            "against exact brute-force fp32 cosine k15 truth over ALL 6,250,000 "
            f"rows: tie-aware recall@15 {SEALED_TIE_AWARE_RECALL}, "
            f"{SEALED_ZERO_DEGREE_ROWS} zero-degree rows"
        ),
        ("family_invariant", "graph_policy"): (
            f"byte-identical sealed R{RUNG_SOURCE_ROUND_ID} {RUNG_SLUG} "
            "cluster-spill k15 fuzzy graph in every cell"
        ),
        ("optimizer", "successful_positive_lr_updates"): int(updates),
        ("execution", "expected_pipeline_stamp", "valid_canonical_edge_count"): (
            graph_edges
        ),
        ("execution", "performance_windows"): performance_windows(int(updates)),
        ("execution", "achieved_positive_draws_per_edge"): achieved_draws_per_edge(
            updates=int(updates), edge_count=graph_edges
        ),
        ("execution", "scale_change"): (
            "R0257: R0217's 2M treatment moved onto the Phase 2 ladder's "
            f"{RUNG_SLUG} rung. Architecture, optimizer, schedule, precision, "
            "sampler, residency and the R0184 dose rule are frozen; the rung, its "
            "sealed graph and the seed are the only axes that move."
        ),
        ("dose_registration",): dict(dose),
    }


def rung_train_config(
    *,
    seed: int,
    rows: int,
    graph_edges: int,
    substrate_signature: Mapping[str, Any],
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    r0217_substrate_signature: Mapping[str, Any],
    r0217_graph_signature: Mapping[str, Any],
    r0217_graph_manifest_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """R0217's config with the rung, its graph and the seed swapped in.

    Returns (config, config_sha256, rung_invariant_sha256).
    """
    if int(rows) != RUNG_ROWS:
        raise Round0257Error(
            f"R0257 rung cardinality is {int(rows)}, registered {RUNG_ROWS}"
        )
    if int(graph_edges) != SEALED_RUNG_DIRECTED_EDGES:
        raise Round0257Error(
            f"R0257 rung graph has {int(graph_edges)} directed edges, sealed "
            f"R{RUNG_SOURCE_ROUND_ID} reports {SEALED_RUNG_DIRECTED_EDGES}"
        )
    updates = successful_updates_for_edges(int(graph_edges))
    if updates != REGISTERED_SUCCESSFUL_UPDATES:
        raise Round0257Error(
            f"R0257 derived horizon {updates} is not the registered "
            f"{REGISTERED_SUCCESSFUL_UPDATES}"
        )
    if updates > REGISTERED_UPDATE_BOUND:
        raise Round0257Error(
            f"R0257 derived horizon {updates} exceeds the round bound "
            f"{REGISTERED_UPDATE_BOUND}"
        )
    dose = validate_dose(updates=updates, edge_count=int(graph_edges))

    template = r0217_template(
        substrate_signature=r0217_substrate_signature,
        graph_signature=r0217_graph_signature,
        graph_manifest_signature=r0217_graph_manifest_signature,
    )
    config = copy.deepcopy(template)
    for path, replacement in rung_bearing_values(
        rows=int(rows), substrate_signature=substrate_signature
    ).items():
        _set_path(config, path, replacement)
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

    invariant = rung_invariant_sha256(config)
    template_invariant = rung_invariant_sha256(template)
    if invariant != template_invariant:
        raise Round0257Error(
            "R0257 rung cell is not R0217's treatment outside the seed, the graph "
            f"and the rung: {invariant} != {template_invariant}"
        )
    for dotted, expected in PIPELINE_IDENTITY_STRINGS_NOT_MASKED.items():
        cursor: Any = config
        for key in dotted.split("."):
            cursor = cursor[key]
        if cursor != expected:
            raise Round0257Error(
                f"R0257 pipeline identity string {dotted} moved: {cursor!r} != "
                f"{expected!r}"
            )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0257Error("R0257 horizon did not reach the train config")
    if float(config["execution"]["minimum_train_upd_s"]) <= 0:
        raise Round0257Error("R0257 performance floor did not survive the swap")
    return config, sha256_bytes(canonical_json(config)), invariant


def dose_view(graph_edges: int) -> dict[str, Any]:
    """The dose arithmetic, published so a reviewer reproduces it without the run."""
    edges = int(graph_edges)
    updates = successful_updates_for_edges(edges)
    return {
        "active_directed_edges": edges,
        "successful_updates": updates,
        "rule": "ceil(R0184_successful_updates * active_edges / R0184_directed_edges)",
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved_draws_per_edge(
            updates=updates, edge_count=edges
        ),
        "dose_quantum_draws_per_edge": dose_quantum(edges),
        "the_deviation_clause_is_a_theorem_of_the_ceil": (
            "review-0217: bounding |achieved - target| by one quantum follows from "
            "asserting the horizon equals the exact ceil. It is not an independent "
            "guard and is not presented as one."
        ),
    }


# --------------------------------------------------------------------------- #
# the family rule
# --------------------------------------------------------------------------- #


def assert_no_rung_map_in_the_gate_family(
    defining_cell_ids: Sequence[str],
    *,
    judged_cell_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Call the SHIPPED 2M-only guard, then assert no judged map is in the family.

    The first half is R0255's guard, unmodified and imported. The second half is
    this round's own obligation: the cells this round *judges* must be disjoint
    from the cells the floors were *fitted* on.
    """
    judged = list(judged_cell_ids if judged_cell_ids is not None else rung_cell_ids())
    observed = [str(item) for item in defining_cell_ids]
    purity = assert_family_is_2m_only(observed)
    intruders = sorted(set(judged) & set(observed))
    if intruders:
        raise Round0257FamilyError(
            "R0257 rung maps reached the fitting family: "
            + ", ".join(intruders)
            + ". "
            + FAMILY_RULE
        )
    return {
        "rule": FAMILY_RULE,
        "shipped_guard": "round0255_treatment.assert_family_is_2m_only",
        "shipped_guard_verdict": purity,
        "defining_cell_ids": observed,
        "judged_cell_ids": judged,
        "judged_and_defining_are_disjoint": True,
        "n_defining": len(observed),
        "n_judged": len(judged),
    }


def rung_family_purity_controls(
    defining_cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Plant each rung map into the family; require the shipped guard to refuse."""
    honest = [str(item) for item in defining_cell_ids]
    controls: list[dict[str, Any]] = []
    for cell in rung_cell_ids():
        planted = honest + [cell]
        refused = False
        error = None
        try:
            assert_no_rung_map_in_the_gate_family(planted)
        except (Round0255FamilyError, Round0257FamilyError) as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": f"rung_map_{cell}_in_the_family",
            "plants": (
                f"{cell}, a map this round judges, added to the cells the n = 29 "
                "floors are fitted on -- the v0 defect exactly"
            ),
            "planted_cells": len(planted),
            "shipped_guard_refused": refused,
            "shipped_guard_error": error,
            "old_predicate_accepted": bool(old_family_predicate(planted)),
        })
    # The whole family replaced by the three rung maps: the degenerate case.
    degenerate_refused = False
    degenerate_error = None
    try:
        assert_no_rung_map_in_the_gate_family(list(rung_cell_ids()))
    except (Round0255FamilyError, Round0257FamilyError) as raised:
        degenerate_refused = True
        degenerate_error = f"{type(raised).__name__}: {raised}"
    controls.append({
        "control": "family_is_only_rung_maps",
        "plants": "the fitting family replaced wholesale by the three judged maps",
        "planted_cells": len(rung_cell_ids()),
        "shipped_guard_refused": degenerate_refused,
        "shipped_guard_error": degenerate_error,
        "old_predicate_accepted": bool(old_family_predicate(list(rung_cell_ids()))),
    })
    honest_refused = False
    honest_error = None
    try:
        assert_no_rung_map_in_the_gate_family(honest)
    except (Round0255FamilyError, Round0257FamilyError) as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_guard_refused"] for item in controls
        ),
        "the_old_predicate_accepted_every_one": all(
            item["old_predicate_accepted"] for item in controls
        ),
        "the_honest_family_still_passes": not honest_refused,
        "honest_error": honest_error,
        "note": (
            "every control calls assert_no_rung_map_in_the_gate_family, which "
            "calls the SHIPPED round0255_treatment.assert_family_is_2m_only. "
            "Nothing here re-implements either guard."
        ),
    }


__all__ = [
    "FAMILY_RULE",
    "GATE_REGISTERABLE_HERE",
    "GRAPH_CAPABILITY",
    "GRAPH_SCHEMA",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "PIPELINE_IDENTITY_NOTE",
    "PIPELINE_IDENTITY_STRINGS_NOT_MASKED",
    "PRODUCTION_CONFIG_SCHEMA",
    "REGISTERED_SUCCESSFUL_UPDATES",
    "REGISTERED_UPDATE_BOUND",
    "ROUND_ID",
    "RUNG_BEARING_PATHS",
    "RUNG_PLACEHOLDER",
    "RUNG_ROWS",
    "RUNG_SLUG",
    "RUNG_SOURCE_ROUND_ID",
    "Round0257Error",
    "Round0257FamilyError",
    "SEALED_RUNG_DIRECTED_EDGES",
    "SEALED_TIE_AWARE_RECALL",
    "SEALED_ZERO_DEGREE_ROWS",
    "SEEDS",
    "SUBSTRATE_CAPABILITY",
    "SUBSTRATE_SCHEMA",
    "TEMPLATE_SEED",
    "TRAIN_SCHEMA",
    "VERDICT_CAPABILITY",
    "VERDICT_SCHEMA",
    "assert_no_rung_map_in_the_gate_family",
    "dose_view",
    "graph_bearing_values",
    "map_capability",
    "r0217_template",
    "rung_bearing_values",
    "rung_cell_id",
    "rung_cell_ids",
    "rung_family_purity_controls",
    "rung_invariant_projection",
    "rung_invariant_sha256",
    "rung_train_config",
    "seed_bearing_values",
]

"""Frozen contract for the seed-43 replay of the prompted-diverse U12 rung.

R0211 found the diverse rung's registered matched-2M retention gate is not
composition-matched: it divides by a model trained only on English 2M, so it
mixes composition change with scale and cannot separate them. Every commensurate
comparison in that same panel passed — native FFR and purity-k1024 above the
English-2M floors, all 19 in-mix languages clear of the relative floor with no
collapse, and both OOD cells above the R0132 raw diverse control.

The gate that *would* decide the rung is a family gate on the diverse population
itself, and that needs seeds. R0212 is the first replay: byte-identical treatment
to R0210 except the seed, and byte-exact reuse of the sealed R0209 graph so the
only thing that varies between the two cells is the random state. Two more seeds
are needed before a mean - 2 sigma gate can be registered; this round releases a
second cell, not a gate.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0169_prompted_diverse import diverse_train_config
from basemap.round0210_prompted_diverse_low_dose import (
    HIDDEN_DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROWS,
    Round0210Error,
    SUPERSEDED_FIXED_UPDATES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
    achieved_draws_per_edge,
    successful_updates_for_edges,
)


ROUND_ID = "0212"
SEED = 43
CANONICAL_SEED = 42
CAPABILITY = "jina-prompted-diverse-u12-map-seed43-low-dose-v1"
TRAIN_SCHEMA = "round0212-prompted-diverse-u12-seed43-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = (
    "round0212-prompted-diverse-u12-seed43-production-config-v1"
)
#: The seed-42 cell this replay pairs with. A family gate needs at least three
#: cells, so two further seeds remain before mean - 2 sigma can be registered.
PAIRED_CAPABILITY = "jina-prompted-diverse-u12-map-seed42-low-dose-v1"
SEEDS_REQUIRED_FOR_FAMILY_GATE = 3


class Round0212Error(Round0210Error):
    """The registered seed-43 diverse replay contract changed."""


def seed43_train_config(
    *,
    graph_signature: Mapping[str, Any],
    graph_manifest_signature: Mapping[str, Any],
    graph_edges: int,
    retained_rows: int,
) -> tuple[dict[str, Any], str]:
    """R0210's config with the seed changed and nothing else."""
    if retained_rows != ROWS:
        raise Round0212Error("R0212 population cardinality changed")
    config, _digest = diverse_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=graph_manifest_signature,
        graph_edges=graph_edges,
        retained_rows=retained_rows,
        seed=SEED,
    )
    config = copy.deepcopy(config)
    if int(config["model"]["hidden_dimension"]) != HIDDEN_DIMENSION:
        raise Round0212Error("R0212 requires the registered h2048 recipe")
    if int(config["paired_invariant"]["seed"]) != SEED:
        raise Round0212Error("R0212 seed did not reach the train config")
    updates = successful_updates_for_edges(graph_edges)
    achieved = achieved_draws_per_edge(updates=updates, edge_count=graph_edges)
    config["schema"] = "round0212-prompted-diverse-u12-seed43-train-config-v1"
    config["optimizer"]["successful_positive_lr_updates"] = updates
    config["paired_invariant"].update({
        "successful_positive_lr_updates": updates,
        "dose_rule": (
            "ceil(R0184_successful_updates * active_edges / R0184_directed_edges)"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "only_treatment_relative_to_r0210": f"seed {CANONICAL_SEED} -> seed {SEED}",
    })
    config["execution"].update({
        "scale_change": (
            "seed replay of the exact R0210 treatment on the byte-identical "
            "sealed R0209 graph; population, recipe, width, dose, prompt, "
            "precision, sampler, optimizer, and residency frozen"
        ),
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "width_by_n_role": "second cell of the prompted-diverse seed family",
    })
    config["dose_registration"] = {
        "source_round": "0184",
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "active_graph_edges": int(graph_edges),
        "successful_updates": updates,
        "target_positive_draws_per_edge": TARGET_POSITIVE_DRAWS_PER_EDGE,
        "achieved_positive_draws_per_edge": achieved,
        "superseded_r0169_fixed_updates": SUPERSEDED_FIXED_UPDATES,
    }
    config["seed_family"] = {
        "paired_seed": CANONICAL_SEED,
        "paired_capability": PAIRED_CAPABILITY,
        "this_seed": SEED,
        "cells_after_this_round": 2,
        "cells_required_for_gate": SEEDS_REQUIRED_FOR_FAMILY_GATE,
        "gate_registerable_here": False,
    }
    return config, sha256_bytes(canonical_json(config))


__all__ = [
    "CANONICAL_SEED",
    "CAPABILITY",
    "PAIRED_CAPABILITY",
    "PRODUCTION_CONFIG_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "Round0212Error",
    "SEED",
    "SEEDS_REQUIRED_FOR_FAMILY_GATE",
    "TRAIN_SCHEMA",
    "seed43_train_config",
]

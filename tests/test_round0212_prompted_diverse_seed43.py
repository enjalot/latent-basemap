from __future__ import annotations

import json
import os

import pytest

from basemap.round0210_prompted_diverse_low_dose import (
    low_dose_train_config,
    successful_updates_for_edges,
)
from basemap.round0212_prompted_diverse_seed43 import (
    CANONICAL_SEED,
    CAPABILITY,
    ROUND_ID,
    ROWS,
    Round0212Error,
    SEED,
    SEEDS_REQUIRED_FOR_FAMILY_GATE,
    seed43_train_config,
)
from experiments import round0212_nodes as nodes


GRAPH = {"canonical_path": "/data/edges.npz", "bytes": 1, "sha256": "a" * 64}
MANIFEST = {"canonical_path": "/data/graph.json", "bytes": 1, "sha256": "b" * 64}
SEALED_EDGES = 957_799_410


def _configs():
    seed43, _ = seed43_train_config(
        graph_signature=GRAPH, graph_manifest_signature=MANIFEST,
        graph_edges=SEALED_EDGES, retained_rows=ROWS,
    )
    seed42, _ = low_dose_train_config(
        graph_signature=GRAPH, graph_manifest_signature=MANIFEST,
        graph_edges=SEALED_EDGES, retained_rows=ROWS,
    )
    return seed43, seed42


def test_identity() -> None:
    assert ROUND_ID == "0212" and SEED == 43 and CANONICAL_SEED == 42
    assert CAPABILITY == "jina-prompted-diverse-u12-map-seed43-low-dose-v1"


def test_seed_reaches_every_place_it_must() -> None:
    seed43, _ = _configs()
    assert seed43["paired_invariant"]["seed"] == 43
    assert json.dumps(seed43).count('"seed": 42') == 0


def test_the_only_difference_from_r0210_is_the_seed() -> None:
    """Every field except the seed and its own labels must be byte-identical."""
    seed43, seed42 = _configs()
    ignore = {"schema", "paired_invariant", "execution", "dose_registration",
              "seed_family", "optimizer"}
    for key in set(seed43) | set(seed42):
        if key in ignore:
            continue
        assert json.dumps(seed43.get(key), sort_keys=True) == json.dumps(
            seed42.get(key), sort_keys=True
        ), f"{key} drifted between the seed cells"
    # the dose and horizon must be identical, not merely close
    assert (
        seed43["optimizer"]["successful_positive_lr_updates"]
        == seed42["optimizer"]["successful_positive_lr_updates"]
        == successful_updates_for_edges(SEALED_EDGES)
    )
    assert (
        seed43["execution"]["achieved_positive_draws_per_edge"]
        == seed42["execution"]["achieved_positive_draws_per_edge"]
    )
    # ...and only the seed differs inside paired_invariant
    a = dict(seed43["paired_invariant"]); b = dict(seed42["paired_invariant"])
    assert a.pop("seed") == 43 and b.pop("seed") == 42
    a.pop("only_treatment_relative_to_r0210", None)
    b.pop("only_treatment_relative_to_r0169", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_this_round_cannot_register_a_family_gate() -> None:
    seed43, _ = _configs()
    family = seed43["seed_family"]
    assert family["gate_registerable_here"] is False
    assert family["cells_after_this_round"] == 2
    assert family["cells_required_for_gate"] == SEEDS_REQUIRED_FOR_FAMILY_GATE == 3


def test_wrong_population_fails_closed() -> None:
    with pytest.raises(Round0212Error):
        seed43_train_config(
            graph_signature=GRAPH, graph_manifest_signature=MANIFEST,
            graph_edges=SEALED_EDGES, retained_rows=7_952_419,
        )


def test_node_rejects_another_action_or_queue() -> None:
    with pytest.raises(Round0212Error):
        nodes.run_job({"manifest": {"round_id": ROUND_ID}},
                      {"action": "train_prompted_diverse_u12_low_dose"})
    with pytest.raises(Round0212Error):
        nodes.run_train({"manifest": {"round_id": "0210"}},
                        {"action": "train_prompted_diverse_u12_seed43"})


def test_configure_binds_seed43_and_the_cross_round_graph() -> None:
    from experiments import round0166_nodes as q2
    from experiments import round0169_nodes as diverse
    from basemap.round0209_prompted_diverse_graph import GRAPH_SCHEMA

    q2_state, diverse_state = dict(q2.__dict__), dict(diverse.__dict__)
    try:
        nodes._configure(1_588_163)
        assert q2.SEED == 43
        assert q2.ROUND_ID == "0212"
        assert q2.GRAPH_SCHEMA == GRAPH_SCHEMA
        assert q2.GRAPH_SOURCE_ROUND_ID == "0209"
        assert q2.GRAPH_BUILT_IN_ROUND is False
        assert q2.scale_train_config is seed43_train_config
        assert q2.SUCCESSFUL_UPDATES == 1_588_163
    finally:
        q2.__dict__.clear(); q2.__dict__.update(q2_state)
        diverse.__dict__.clear(); diverse.__dict__.update(diverse_state)


def test_graph_reuse_is_byte_exact_with_the_seed42_cell() -> None:
    """Both cells must bind the same sealed graph file, not a rebuild."""
    from experiments.prepare_round0210_queue import GRAPH_MANIFEST as G42
    from experiments.prepare_round0212_queue import GRAPH_MANIFEST as G43

    assert G42 == G43
    if not os.path.exists(G43):
        pytest.skip("sealed R0209 graph is not present on this machine")

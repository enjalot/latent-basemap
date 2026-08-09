"""R0229 phase 2 — the contract for the spill-lifted arm and its map test.

Phase 1 fired the registered structural-gain trigger: at matched device cost the
reachability ceiling rises steeply with spill, and two of the three
**100M-feasible** configurations clear the `c = 4` level that R0228 found clean.
Phase 2 builds the selected configuration at 2M, trains three seeds under
R0217's treatment with the graph swapped, and runs R0228's registered
displacement statistic on it.

The arm is chosen by the rule registered in `round-0229` § "Node 3" and its
addendum 2: among the spill grid's 100M-feasible cells, the one with the highest
strict ceiling over all 2,000,000 rows, ties broken by lowest `s`; built with the
nn-descent setting that won the quality sweep. `select_arm` implements exactly
that and is re-run inside the node against the sealed phase-1 artifacts, so the
queue cannot bind an arm the rule does not choose.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from basemap.round0229_quality_contract import (
    BASELINE_CELL,
    Round0229Error,
)

ROUND_ID = "0229"
ROWS = 2_000_000
DIMENSION = 384
GRAPH_K = 15

SEEDS = (42, 43, 44)
ARM_NAME = "spill-lifted"

GRAPH_CAPABILITY = "minilm-mixed-2m-spill-lifted-k15-fuzzy-graph-v1"
GEOMETRY_CAPABILITY = "minilm-mixed-2m-spill-lifted-map-geometry-v1"
GRAPH_SCHEMA = "round0229-spill-lifted-fuzzy-graph-v1"
TRAIN_SCHEMA = "round0229-spill-lifted-train-v1"
GEOMETRY_SCHEMA = "round0229-spill-lifted-geometry-v1"
PRODUCTION_CONFIG_SCHEMA = "round0229-spill-lifted-production-config-v1"

GATE_REGISTERABLE_HERE = False
GATE_RELEASE_CLAIMED = False
ADOPTION_CLAIMED = False
EQUIVALENCE_CLAIMED = False

#: R0217/R0221/R0223/R0228 all carry this. A cell that does not reproduce it
#: from its own constructed config refuses to train.
TREATMENT_INVARIANT_SHA256 = (
    "c28cfd61e744a2e19e136940a13ae0ad26bd9b9d8b9525906df57f0e7a56e784"
)

#: The three arms of the registered DiD trend test (review-0228-01 #8).
TREND_ARMS = ("c4", ARM_NAME, "c16")


def map_capability(seed: int) -> str:
    return f"minilm-mixed-2m-spill-lifted-map-seed{int(seed)}-v1"


def select_arm(
    *,
    sweep: Mapping[str, Any],
    spill: Mapping[str, Any],
) -> dict[str, Any]:
    """The registered arm-selection rule, applied to phase 1's sealed artifacts.

    Registered in `round-0229` § "Node 3" and narrowed to one arm by addendum 2:

    * if the trigger fired on **structural gain**, the arm is `spill-lifted`:
      among the spill grid's 100M-feasible cells, the highest strict ceiling
      over all 2,000,000 rows, ties broken by lowest `s`;
    * the nn-descent setting is the quality sweep's highest uniform tie-aware
      recall over all 2,000,000 rows, ties broken by lowest wall, then by
      earliest ladder position.
    """
    scored = [cell for cell in sweep["cells"] if cell.get("scored")]
    if not scored:
        raise Round0229Error("R0229 phase 2 needs a scored quality-sweep cell")
    baseline = next(
        (cell for cell in scored if str(cell["cell"]) == BASELINE_CELL), None
    )
    if baseline is None:
        raise Round0229Error("R0229 phase 2 needs the q0-baseline control")
    order = {str(cell["cell"]): index for index, cell in enumerate(sweep["cells"])}
    best_setting = min(
        scored,
        key=lambda cell: (
            -float(cell["tie_aware_recall_all_rows"]),
            float(cell["build_seconds"]),
            order[str(cell["cell"])],
        ),
    )

    feasible = [
        cell for cell in spill["cells"]
        if bool(cell.get("feasible_at_100m"))
        and cell.get("strict_ceiling_all_rows") is not None
    ]
    if not feasible:
        raise Round0229Error(
            "R0229 phase 2 found no 100M-feasible spill cell; the fallback in "
            "the round file applies and must be taken explicitly"
        )
    best_cell = min(
        feasible,
        key=lambda cell: (
            -float(cell["strict_ceiling_all_rows"]), int(cell["spill"])
        ),
    )
    return {
        "arm": ARM_NAME,
        "rule": (
            "highest strict ceiling over all 2,000,000 rows among the "
            "100M-feasible spill cells, ties broken by lowest spill; built with "
            "the quality sweep's highest uniform tie-aware recall setting"
        ),
        "cell": str(best_cell["cell"]),
        "clusters": int(best_cell["clusters"]),
        "spill": int(best_cell["spill"]),
        "strict_ceiling_all_rows": float(best_cell["strict_ceiling_all_rows"]),
        "tie_ceiling_query_sample": best_cell.get("tie_ceiling_query_sample"),
        "projected_100m_max_cluster_rows": best_cell.get(
            "projected_100m_max_cluster_rows"
        ),
        "projected_50m_max_cluster_rows": best_cell.get(
            "projected_50m_max_cluster_rows"
        ),
        "realised_imbalance_at_2m": best_cell.get("realised_imbalance"),
        "mean_cluster_rows_at_2m": best_cell.get("mean_cluster_rows_at_2m"),
        "nn_descent": {
            "cell": str(best_setting["cell"]),
            "graph_degree": int(best_setting["graph_degree"]),
            "intermediate_graph_degree": int(
                best_setting["intermediate_graph_degree"]
            ),
            "max_iterations": int(best_setting["max_iterations"]),
            "tie_aware_recall_all_rows": float(
                best_setting["tie_aware_recall_all_rows"]
            ),
            "build_seconds": float(best_setting["build_seconds"]),
        },
        "baseline_tie_aware_recall": float(baseline["tie_aware_recall_all_rows"]),
        "candidates_considered": [
            {
                "cell": str(cell["cell"]),
                "clusters": int(cell["clusters"]),
                "spill": int(cell["spill"]),
                "strict_ceiling_all_rows": float(cell["strict_ceiling_all_rows"]),
            }
            for cell in feasible
        ],
    }


def per_map_did(
    *, candidate_gaps: Sequence[float], exact_gaps: Sequence[float]
) -> list[float]:
    """Each candidate map's gap, centred on its own configuration's null arm.

    Centring per configuration is what makes the nine values of the trend test
    commensurate even though the lost/control row sets differ by configuration.
    """
    exact = [float(value) for value in exact_gaps]
    if len(exact) < 2:
        raise Round0229Error("R0229 per-map DiD needs a null arm")
    centre = sum(exact) / len(exact)
    return [float(value) - centre for value in candidate_gaps]


__all__ = [
    "ADOPTION_CLAIMED",
    "ARM_NAME",
    "DIMENSION",
    "EQUIVALENCE_CLAIMED",
    "GATE_REGISTERABLE_HERE",
    "GATE_RELEASE_CLAIMED",
    "GEOMETRY_CAPABILITY",
    "GEOMETRY_SCHEMA",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "PRODUCTION_CONFIG_SCHEMA",
    "ROUND_ID",
    "ROWS",
    "SEEDS",
    "TRAIN_SCHEMA",
    "TREATMENT_INVARIANT_SHA256",
    "TREND_ARMS",
    "map_capability",
    "per_map_did",
    "select_arm",
]

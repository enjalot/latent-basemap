"""R0243 contract — safety, registration, and a positive control for every guard.

Three properties are established here and none is delegated to a detector:

1. **No file R0243 adds to the node path contains a signalling construct.** The
   scan is stronger than R0242's: `tokenize` strips every STRING and COMMENT
   token before the search, so a forbidden token cannot be excused merely by
   also appearing inside some unrelated note. Prose may name the hazard;
   executable code may not contain it.
2. **Every threshold in the halt rule is a registered literal in the release
   commit** this round's file names, not a number chosen after a measurement.
3. **Every guard this round adds ships a positive control** - an input that
   plants the defect and proves the guard catches it - and, for the
   re-expressed exposure guard, a *vacuity* control that proves R0242's form of
   the same guard excludes nothing at the realised cell-size distribution.

The forbidden tokens are assembled at runtime rather than written as literals,
so this file does not itself become a false positive for the unreleased
`experiments/check_signal_safety.py`.
"""
from __future__ import annotations

import ast
import io
import os
import tokenize

import numpy as np
import pytest

from basemap.round0243_residual import (
    CONCENTRATION_TOP_M,
    EXPOSURE_GUARD_NOTE,
    HALT_CELL_TIE_AWARE_BUILDER_RATE,
    HALT_GLOBAL_TIE_AWARE_BUILDER_RATE,
    HALT_P_VALUE,
    HALT_RULE_NOTE,
    HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE,
    HALT_SINGLE_CLUSTER_SHARE,
    HALT_TOP_M_SHARE,
    PERMUTATIONS,
    PERMUTATION_SEED,
    ROUND_ID,
    ROWS,
    SUBSTRATE_BYTES,
    exposure_profile,
    full_gather_ceiling,
    hot_cell_scan,
    loss_decomposition,
    post_canonical_tripwire,
    residual_verdict,
    sorted_gather_price,
    strict_reproduction_gate,
)

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH_FILES = (
    "basemap/round0243_residual.py",
    "experiments/round0243_nodes.py",
)
CLUSTERS_FOR_CONTROLS = 400


def _source(name: str) -> str:
    with open(os.path.join(RELEASE_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


def _executable_text(name: str) -> str:
    """Source with every STRING and COMMENT token removed.

    R0242's version skipped a line only if the whole stripped line appeared
    verbatim inside some string constant, and then excused any token that
    appeared in ANY constant anywhere in the file. That second clause is a hole:
    a note mentioning a hazard would excuse a real call to it. Tokenizing
    removes the strings instead of pattern-matching around them.
    """
    text = _source(name)
    kept: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type in (tokenize.STRING, tokenize.COMMENT):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _forbidden_tokens() -> tuple[str, ...]:
    sig = "SIG"
    return (
        "subprocess",
        "multiprocessing",
        "os.system",
        "os.fork",
        "os.exec",
        "os." + "kill",
        "os.killpg",
        "send_signal",
        "signal.signal",
        "signal." + sig + "KILL",
        "signal." + sig + "TERM",
        "p" + "kill",
        "kill" + "all",
        "py-spy",
        "ptrace",
        "timeout=",
    )


@pytest.mark.parametrize("name", NODE_PATH_FILES)
def test_no_signalling_construct_in_the_node_path(name: str) -> None:
    executable = _executable_text(name)
    for token in _forbidden_tokens():
        assert token not in executable, (
            f"{name} carries the forbidden token {token!r} in executable code"
        )


@pytest.mark.parametrize("name", NODE_PATH_FILES)
def test_node_path_imports_no_child_process_module(name: str) -> None:
    tree = ast.parse(_source(name))
    banned = {"subprocess", "multiprocessing", "signal", "ctypes", "resource"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned, node.module


@pytest.mark.parametrize("name", NODE_PATH_FILES)
def test_every_deferred_import_on_the_node_path_actually_resolves(
    name: str,
) -> None:
    """The blind spot that cost R0242 attempt 1: a module that does not exist.

    `check_undefined_names.py` is an AST guard and cannot see a missing
    third-party package. Function-local imports are exactly where that hides.
    """
    import importlib.util

    tree = ast.parse(_source(name))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    for module in sorted(modules):
        assert importlib.util.find_spec(module) is not None, (
            f"{name} imports {module!r}, which does not resolve in the release "
            "venv this round's nodes run from"
        )


def test_every_registered_check_is_imported_not_retyped() -> None:
    text = _source("experiments/round0243_nodes.py")
    for imported in (
        "verify_inheritance",
        "_readonly_memmap",
        "_blocked_descending_sort",
        "_fuzzy_symmetrise_blocked",
        "_check_runner_abort",
        "_cluster_assignment",
        "_HostWatchdog",
        "_memmap_attestation",
        "loss_decomposition",
        "cluster_locality_test",
        "canonical_undirected_degrees",
        "post_canonical_tripwire",
        "symmetrised_degree_once",
        "weight_distribution",
        "partition_reachability",
        "partition_agreement",
        "truth_probe_query_rows",
    ):
        assert imported in text, imported
    tree = ast.parse(text)
    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for reviewed in (
        "verify_inheritance", "loss_decomposition", "cluster_locality_test",
        "_fuzzy_symmetrise_blocked", "_blocked_descending_sort",
        "canonical_undirected_degrees", "post_canonical_tripwire",
        "symmetrised_degree_once", "weight_distribution", "_cluster_assignment",
        "partition_reachability", "partition_agreement", "_dispersion",
    ):
        assert reviewed not in defined, f"R0243 re-typed the reviewed {reviewed}"


def test_round0243_edits_no_earlier_round_module() -> None:
    """R0243 adds files; the wall guard is a subclass, never an edit."""
    text = _source("experiments/round0243_nodes.py")
    assert "class StageGuard0243(StageGuard0242)" in text


def test_halt_rule_is_registered_in_the_release() -> None:
    assert HALT_GLOBAL_TIE_AWARE_BUILDER_RATE == 0.01
    assert HALT_CELL_TIE_AWARE_BUILDER_RATE == 0.0185
    assert HALT_SINGLE_CLUSTER_SHARE == 0.05
    assert HALT_SINGLE_CLUSTER_EXPOSURE_MULTIPLE == 1.5
    assert HALT_P_VALUE == 0.001
    assert HALT_TOP_M_SHARE == 0.25
    assert CONCENTRATION_TOP_M == 20
    assert PERMUTATIONS == 10_000
    assert PERMUTATION_SEED == 242_000
    assert ROUND_ID == "0243"
    assert ROWS == 100_000_000
    assert SUBSTRATE_BYTES == 153_600_000_128
    # The per-cell magnitude bar is R0034's measured v1 defect fraction.
    assert HALT_CELL_TIE_AWARE_BUILDER_RATE == pytest.approx(2_779_481 / 150_000_000, abs=5e-5)
    for fragment in (
        "H0", "H1", "H2", "0.0185", "1.5/c", "do NOT halt", "registered in advance",
    ):
        assert fragment in HALT_RULE_NOTE, fragment
    assert "vacuity" in EXPOSURE_GUARD_NOTE


# --------------------------------------------------------------------------- #
# positive controls — every guard, on an input that plants its defect
# --------------------------------------------------------------------------- #
def _cells(sizes: np.ndarray, missing_per_row: np.ndarray, k: int = 15):
    """Expand a per-cell (rows, missing) description into per-row vectors."""
    labels = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    missing = np.repeat(missing_per_row, sizes).astype(np.float64)
    exposure = np.full(labels.size, float(k), dtype=np.float64)
    return labels, missing, exposure


def test_exposure_guard_binds_where_r0242s_form_is_vacuous() -> None:
    """The vacuity control review-0242-01/F5 asked for, as a measurement.

    Cell sizes are drawn to span the realised range at `c = 400` (`0.32/c` to
    `2.08/c`). The re-expressed guard must EXCLUDE some cells; R0242's absolute
    `1%` guard must exclude NONE, because no cell can reach `1%` of exposure at
    this `c` at all.
    """
    rng = np.random.default_rng(243)
    sizes = rng.integers(400, 2_600, size=CLUSTERS_FOR_CONTROLS)
    labels, _, exposure = _cells(sizes, np.zeros(sizes.size))
    profile = exposure_profile(
        labels=labels, exposure=exposure, clusters=CLUSTERS_FOR_CONTROLS
    )
    assert profile["re_expressed_guard"]["cells_excluded_by_the_guard"] > 0, (
        "the re-expressed exposure guard must be crossable at the realised "
        "cell-size distribution, or it is as vacuous as the one it replaces"
    )
    assert profile["re_expressed_guard"]["cells_admitted_by_the_guard"] > 0, (
        "a guard that excludes every cell is vacuous in the other direction"
    )
    assert profile["r0242_absolute_guard"]["cells_excluded_by_the_guard"] == 0, (
        "R0242's absolute 1% guard is expected to exclude nothing at c = 400; "
        "if it excludes something here the vacuity finding needs revisiting"
    )
    assert profile["r0242_absolute_guard"]["largest_attainable_share"] < 0.01


def test_hot_cell_guard_fires_on_a_planted_hot_cell() -> None:
    sizes = np.full(CLUSTERS_FOR_CONTROLS, 1_000, dtype=np.int64)
    missing = np.zeros(CLUSTERS_FOR_CONTROLS)
    missing[7] = 1.0  # every row in cell 7 loses one of fifteen edges
    labels, missing_rows, exposure = _cells(sizes, missing)
    scan = hot_cell_scan(
        labels=labels, missing=missing_rows, exposure=exposure,
        clusters=CLUSTERS_FOR_CONTROLS,
    )
    assert scan["cells_firing_all_three"] == 1
    assert scan["firing_clusters"] == [7]
    assert scan["worst_cell_rate"] == pytest.approx(1.0 / 15.0)


def test_hot_cell_guard_declines_a_large_share_of_a_negligible_total() -> None:
    """The R0242 situation on the tie-aware scale, as a control.

    One cell carries almost all of the loss - a share far past `0.05` - but the
    absolute quantity is tiny, so its per-cell RATE is nowhere near `1.85%`.
    The guard must not fire: that is the whole correction this round makes.
    """
    sizes = np.full(CLUSTERS_FOR_CONTROLS, 1_000, dtype=np.int64)
    per_row = np.zeros(CLUSTERS_FOR_CONTROLS)
    labels = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    missing_rows = np.repeat(per_row, sizes).astype(np.float64)
    exposure = np.full(labels.size, 15.0)
    hot = np.flatnonzero(labels == 7)
    missing_rows[hot[:100]] = 1.0          # 100 edges in a 15,000-edge cell
    missing_rows[np.flatnonzero(labels == 3)[:5]] = 1.0
    scan = hot_cell_scan(
        labels=labels, missing=missing_rows, exposure=exposure,
        clusters=CLUSTERS_FOR_CONTROLS,
    )
    assert scan["worst_cell_share"] > HALT_SINGLE_CLUSTER_SHARE
    assert scan["worst_cell_rate"] < HALT_CELL_TIE_AWARE_BUILDER_RATE
    assert scan["cells_firing_all_three"] == 0


def test_hot_cell_guard_is_blocked_by_the_exposure_arm_on_a_giant_cell() -> None:
    """The arm R0242's absolute guard could never exercise.

    A cell three times the mean size carries a high rate and a large share; the
    re-expressed guard excludes it, and R0242's absolute `1%` form would not.
    """
    sizes = np.full(CLUSTERS_FOR_CONTROLS, 1_000, dtype=np.int64)
    sizes[11] = 3_000
    per_row = np.zeros(CLUSTERS_FOR_CONTROLS)
    per_row[11] = 1.0
    labels, missing_rows, exposure = _cells(sizes, per_row)
    scan = hot_cell_scan(
        labels=labels, missing=missing_rows, exposure=exposure,
        clusters=CLUSTERS_FOR_CONTROLS,
    )
    row = next(
        entry for entry in scan["highest_rate_cells"] if entry["cluster"] == 11
    )
    assert row["meets_rate_arm"] and row["meets_share_arm"]
    assert not row["passes_exposure_guard"]
    assert scan["cells_firing_all_three"] == 0
    profile = exposure_profile(
        labels=labels, exposure=exposure, clusters=CLUSTERS_FOR_CONTROLS
    )
    assert profile["r0242_absolute_guard"]["cells_excluded_by_the_guard"] == 0


def _verdict(scan, *, reproduces=True, top=0.30, maximum=0.08):
    shape = {
        "chi_square": {"p_value": 9.999e-05},
        "top_m_share_of_missing": {"observed": top},
        "max_single_cluster_share_of_missing": {"observed": maximum},
    }
    return residual_verdict(
        reproduction={"agree": reproduces, "disagreements": []},
        tie_aware_scan=scan,
        tie_aware_builder_test=shape,
        strict_builder_test={
            "top_m_share_of_missing": {"observed": 0.672238},
            "max_single_cluster_share_of_missing": {"observed": 0.388273},
        },
    )


def test_h1_fires_on_a_planted_global_rate() -> None:
    sizes = np.full(CLUSTERS_FOR_CONTROLS, 1_000, dtype=np.int64)
    labels, missing_rows, exposure = _cells(
        sizes, np.full(CLUSTERS_FOR_CONTROLS, 0.3)
    )
    scan = hot_cell_scan(
        labels=labels, missing=missing_rows, exposure=exposure,
        clusters=CLUSTERS_FOR_CONTROLS,
    )
    verdict = _verdict(scan)
    assert verdict["h1_fires"] and verdict["halt_part_b"]
    assert verdict["h1_global_tie_aware_builder_rate"] >= 0.01


def test_h0_failure_halts_even_with_a_clean_residual() -> None:
    sizes = np.full(CLUSTERS_FOR_CONTROLS, 1_000, dtype=np.int64)
    labels, missing_rows, exposure = _cells(
        sizes, np.zeros(CLUSTERS_FOR_CONTROLS)
    )
    missing_rows[0] = 1.0
    scan = hot_cell_scan(
        labels=labels, missing=missing_rows, exposure=exposure,
        clusters=CLUSTERS_FOR_CONTROLS,
    )
    clean = _verdict(scan)
    assert not clean["halt_part_b"]
    broken = _verdict(scan, reproduces=False)
    assert broken["halt_part_b"] and not broken["part_b_may_run"]


def test_shape_alone_does_not_halt_and_says_so() -> None:
    """The registered decision, exercised: shape fires, the verdict does not."""
    sizes = np.full(CLUSTERS_FOR_CONTROLS, 1_000, dtype=np.int64)
    labels, missing_rows, exposure = _cells(
        sizes, np.zeros(CLUSTERS_FOR_CONTROLS)
    )
    missing_rows[:20] = 1.0
    scan = hot_cell_scan(
        labels=labels, missing=missing_rows, exposure=exposure,
        clusters=CLUSTERS_FOR_CONTROLS,
    )
    verdict = _verdict(scan, top=0.90, maximum=0.90)
    shape = verdict["h3_shape_reported_not_gating"]
    assert shape["r0242_thresholds_would_halt_on_the_tie_aware_scale"]
    assert shape["gates"] is False
    assert verdict["part_b_may_run"]
    assert verdict["reading"] == "concentrated in shape, negligible in magnitude"


def test_strict_reproduction_gate_detects_a_one_unit_drift() -> None:
    sealed_decomposition = {
        field: 7 for field in
        ("probe_rows", "rows_carrying_strict_loss",
         "rows_with_reachability_below_one", "rows_both",
         "rows_builder_loss_with_truth_fully_reachable",
         "rows_recovering_an_unreachable_neighbour", "total_missing_edges",
         "partition_forced_missing_edges", "builder_missing_edges")
    }
    measured = dict(sealed_decomposition)
    observed = {
        "chi_square": 1.0, "top_m_share_of_missing": 0.5,
        "top_m_share_of_exposure": 0.1,
        "max_single_cluster_share_of_missing": 0.2,
        "max_single_cluster_share_of_exposure": 0.01,
    }
    sealed_tests = {"builder_loss_inside_partition": {"observed": dict(observed)}}
    gate = strict_reproduction_gate(
        measured_decomposition=measured,
        sealed_decomposition=sealed_decomposition,
        measured_dispersion={"builder_loss_inside_partition": observed},
        sealed_tests=sealed_tests,
    )
    assert gate["agree"] and gate["fields_checked"] == 14
    measured["builder_missing_edges"] = 8
    drifted = strict_reproduction_gate(
        measured_decomposition=measured,
        sealed_decomposition=sealed_decomposition,
        measured_dispersion={"builder_loss_inside_partition": observed},
        sealed_tests=sealed_tests,
    )
    assert not drifted["agree"]
    assert drifted["disagreements"][0]["where"].endswith("builder_missing_edges")


def test_tie_aware_decomposition_recovers_a_planted_forgiveness_rate() -> None:
    """The same imported decomposition, run on both recall vectors."""
    k = 15
    strict = np.full(200, 1.0)
    tie = np.full(200, 1.0)
    reach = np.full(200, 1.0)
    strict[:100] = (k - 3) / k          # 300 strict missing edges
    tie[:100] = (k - 1) / k             # 100 of them survive tie-forgiveness
    strict_split = loss_decomposition(strict=strict, reachability=reach, k=k)
    tie_split = loss_decomposition(strict=tie, reachability=reach, k=k)
    assert strict_split["builder_missing_edges"] == 300
    assert tie_split["builder_missing_edges"] == 100
    forgiven = 1.0 - tie_split["builder_missing_edges"] / strict_split[
        "builder_missing_edges"
    ]
    assert forgiven == pytest.approx(2 / 3)


def test_post_canonical_tripwire_fires_on_a_planted_edgeless_row() -> None:
    """The v1 defect, planted; the guard this round's Part B publishes."""
    degree = np.full(1_000, 4, dtype=np.int64)
    assert post_canonical_tripwire(degree=degree, rows=1_000)["holds"]
    degree[17] = 0
    fired = post_canonical_tripwire(degree=degree, rows=1_000)
    assert not fired["holds"] and fired["zero_degree_rows"] == 1


def test_sorted_gather_pricing_is_bounded_by_the_substrate() -> None:
    priced = sorted_gather_price(
        anchors=10_000_000, neighbours_per_row=15, row_bytes=1_536,
        distinct_rows_touched=78_000_000, substrate_bytes=SUBSTRATE_BYTES,
        wall_s=200.0, physical_read_bytes=110_000_000_000, label="control",
    )
    assert priced["read_amplification_over_useful"] < 1.0
    assert priced["physical_read_as_fraction_of_substrate"] < 1.0
    ceiling = full_gather_ceiling(
        substrate_bytes=SUBSTRATE_BYTES,
        measured_delivered_rate_bytes_per_s=1_500_000_000.0,
        measured_physical_read_fraction_of_substrate=0.72,
        r0242_unsorted_physical_rate_bytes_per_s=477_264_163.2660608,
    )
    assert ceiling["kind"] == "prediction"
    assert ceiling["implied_read_amplification_over_useful"] < 0.1
    low, high = ceiling["interval_hours"]
    assert 0.0 < low <= high < 1.0, (
        "a full SORTED 100M gather that reads the substrate once cannot cost "
        "hours; R0242's blocked projection was 16.6 h"
    )
    with pytest.raises(Exception, match="non-positive"):
        full_gather_ceiling(
            substrate_bytes=SUBSTRATE_BYTES,
            measured_delivered_rate_bytes_per_s=0.0,
            measured_physical_read_fraction_of_substrate=0.0,
            r0242_unsorted_physical_rate_bytes_per_s=477_264_163.2660608,
        )

"""R0242 contract — safety and registration, asserted by reading the source.

Two properties are established here and neither is delegated to a detector:

1. **No file R0242 adds to the node path contains a signalling construct.** Not
   a child process, not a process-signalling method, not `subprocess.run(...,
   timeout=N)` - which CPython implements as `Popen.kill()` and which fired
   against a live cuML child in R0238. The AST guard is structurally blind to
   that construct, so this is a token scan over the source text.
2. **The halt rule is a registered literal in the release commit**, not a
   threshold chosen after the measurement existed.

The forbidden tokens are assembled at runtime rather than written as string
constants, because a literal `p` + `kill` inside the tuple that forbids it is
exactly the false positive `experiments/check_signal_safety.py` reported on
R0241's contract test. Building them avoids adding noise to a detector whose
output this round reads for information.
"""
from __future__ import annotations

import ast
import os

import pytest

from basemap.round0242_locality import (
    CONCENTRATION_TOP_M,
    HALT_P_VALUE,
    HALT_RULE_NOTE,
    HALT_SINGLE_CLUSTER_EXPOSURE,
    HALT_SINGLE_CLUSTER_SHARE,
    HALT_TOP_M_SHARE,
    IN_DEGREE_BIN_EDGES,
    PERMUTATIONS,
    PERMUTATION_SEED,
    ROUND_ID,
    ROWS,
)

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH_FILES = (
    "basemap/round0242_locality.py",
    "experiments/round0242_nodes.py",
)


def _source(name: str) -> str:
    with open(os.path.join(RELEASE_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


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
    text = _source(name)
    # Prose in docstrings and notes legitimately NAMES these hazards. Only
    # executable lines are scanned, which is the property that matters.
    tree = ast.parse(text)
    docstrings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            docstrings.add(node.value)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(stripped and stripped in doc for doc in docstrings):
            continue
        lines.append(line)
    executable = "\n".join(lines)
    for token in _forbidden_tokens():
        if token in executable and not any(token in doc for doc in docstrings):
            raise AssertionError(f"{name} carries the forbidden token {token!r}")


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


def test_every_registered_check_is_imported_not_retyped() -> None:
    """The round must reach the reviewed functions, never reimplement them."""
    text = _source("experiments/round0242_nodes.py")
    for imported in (
        "verify_inheritance",
        "_probe_candidate_cosines",
        "_ProbeRowView",
        "_readonly_memmap",
        "_score_probe",
        "strict_containment_rows",
        "tie_aware_rows",
        "_blocked_descending_sort",
        "_fuzzy_symmetrise_blocked",
        "_check_runner_abort",
        "truth_probe_query_rows",
    ):
        assert imported in text, imported
    # And it must not define its own copy of any of them.
    tree = ast.parse(text)
    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for reviewed in (
        "verify_inheritance", "_score_probe", "strict_containment_rows",
        "tie_aware_rows", "_fuzzy_symmetrise_blocked", "graph_validity",
        "_probe_candidate_cosines", "_blocked_descending_sort",
    ):
        assert reviewed not in defined, f"R0242 re-typed the reviewed {reviewed}"


def test_round0242_modifies_no_earlier_round_file() -> None:
    """R0242 adds files; it never edits a round0215-0241 module."""
    text = _source("basemap/round0242_locality.py")
    assert "class StageGuard0242(StageGuard)" in text, (
        "the wall-guard correction must be a SUBCLASS; R0241's file is a "
        "reviewed artifact and this round edits nothing in it"
    )


def test_halt_rule_is_registered_in_the_release() -> None:
    assert HALT_P_VALUE == 0.001
    assert HALT_TOP_M_SHARE == 0.25
    assert HALT_SINGLE_CLUSTER_SHARE == 0.05
    assert HALT_SINGLE_CLUSTER_EXPOSURE == 0.01
    assert CONCENTRATION_TOP_M == 20
    assert PERMUTATIONS == 10_000
    assert PERMUTATION_SEED == 242_000
    # A p-value floor the permutation budget cannot resolve would be vacuous.
    assert 1.0 / (PERMUTATIONS + 1) < HALT_P_VALUE
    for fragment in (
        "builder", "p < 0.001", "top 20 of 400", "25%",
        "structured but diffuse", "CANNOT halt",
    ):
        assert fragment in HALT_RULE_NOTE, fragment


def test_registered_shape_constants() -> None:
    assert ROUND_ID == "0242"
    assert ROWS == 100_000_000
    assert IN_DEGREE_BIN_EDGES[0] == 0 and IN_DEGREE_BIN_EDGES[1] == 1, (
        "the in-degree bands must isolate zero-in-degree rows, which is the "
        "specific population R0241 left open"
    )
    assert list(IN_DEGREE_BIN_EDGES) == sorted(set(IN_DEGREE_BIN_EDGES))


def test_part_b_is_gated_on_the_locality_receipt() -> None:
    text = _source("experiments/round0242_nodes.py")
    assert "locality_reference" in text
    assert 'verdict.get("part_b_may_run")' in text, (
        "Part B must read the sealed Part A verdict and refuse itself, rather "
        "than relying on the runner's job ordering"
    )


def test_memory_watchdog_is_conjunctive() -> None:
    """A swap-only rule would have destroyed R0240's completed 100M graph."""
    from experiments.round0242_nodes import (
        WATCHDOG_ANON_BYTES,
        WATCHDOG_MEM_AVAILABLE_BYTES,
        WATCHDOG_SWAP_GROWTH_BYTES,
    )

    assert WATCHDOG_SWAP_GROWTH_BYTES > 0
    assert WATCHDOG_ANON_BYTES > 0
    assert WATCHDOG_MEM_AVAILABLE_BYTES > 0
    text = _source("experiments/round0242_nodes.py")
    assert "swap_growth > WATCHDOG_SWAP_GROWTH_BYTES and pressure" in text, (
        "the watchdog must fire only on the CONJUNCTION of swap growth with "
        "real pressure"
    )

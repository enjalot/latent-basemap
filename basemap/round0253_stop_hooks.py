"""R0253 — install the cooperative-stop hook where a real ladder train runs.

R0252 added `set_abort_poll` to `basemap/artifact_identity.py` and
`basemap/panel_v2.py` and measured that chunk-polling takes a 153.6 GB integrity
hash from `50.21589363922704x` the registered ceiling to
`0.02596942086149069x`.  review-0252-01 §A then found the fix was installed
nowhere that matters:

    `set_abort_poll` is called from exactly one file -- this round's measurement
    node.  The trainer node's setup hash is still `0.5318989691995224x`
    (deviation 6), and `score_panel` in any real node is still
    `0.7843683944260816x`.  That is two node-side installs, not one, and neither
    has been run.

This module is the install.  It is deliberately *not* a new mechanism: it calls
the `set_abort_poll` R0252 already released, with a poll that reads the runner's
own cooperative abort flag through `basemap.gpu_child_supervision`.  Nothing
about any digest, metric, neighbour set, ordering or rounding changes; the only
new outcome is that a hash or a panel score may raise when the operator has
asked the node to stop, which is the outcome the whole exercise is for.

Three things are published rather than asserted:

* `REGISTERED_ENTRIES` names every node entry point a real ladder train passes
  through.  `entry_install_audit()` reads the release's own AST and reports, per
  entry, whether the install is the first statement of its body.  A removed
  install is a failing audit, not a silently widened interval.
* `stop_hooks_state()` reports what is installed *right now*, by reading the
  modules' `abort_poll` attributes.  A node publishes it at handler entry and
  again after installing, so "installed" is a receipt.
* `assert_stop_hooks_installed()` is the guard.  Its positive control
  (`tests/test_round0253_contract.py`) clears the hook and proves the guard
  raises, and a second control plants an entry function with the install removed
  and proves the AST audit catches it.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from basemap.gpu_child_supervision import runner_abort_reason
from basemap.round0247_registry import registered_value


ROUND_ID = "0253"

INSTALL_CAPABILITY = "round0253-ladder-node-stop-hook-install-v1"
INSTALL_SCHEMA = "round0253-ladder-node-stop-hook-install-v1"
WRITE_CAPABILITY = "round0253-polled-write-path-v1"
WRITE_SCHEMA = "round0253-polled-write-path-v1"
CUDA_CAPABILITY = "round0253-cuda-holding-setup-gap-v1"
CUDA_SCHEMA = "round0253-cuda-holding-setup-gap-v1"

#: The release modules that carry R0252's `set_abort_poll`.  Both, always: the
#: review's finding is that installing one of the two is what R0252 effectively
#: did, and a node that hashes but never scores is not the node that binds.
HOOKED_MODULES: tuple[str, ...] = (
    "basemap.artifact_identity",
    "basemap.panel_v2",
)

#: Every node entry point a real ladder train passes through, as
#: `(module, function)`.  This is the answer to review-0252-01 §J.1's "two
#: node-side installs": it is fourteen more than two, because the trainer node's
#: setup path and the scorer are not the only places a 100M-scale
#: `expected_input_signature` / `ordered_array_sha256` / `score_panel` call sits.
#:
#: Derived by reading the call graph (see `reachability_audit`), then fixed here
#: so the AST guard has something to check rather than something to infer.
REGISTERED_ENTRIES: tuple[tuple[str, str], ...] = (
    # the base trainer + graph family -- `run_train` is the trainer node's own
    # setup path, the `0.5318989691995224x` interval review-0252-01 named.
    ("experiments.round0113_nodes", "run_assemble"),
    ("experiments.round0113_nodes", "run_build_graph"),
    ("experiments.round0113_nodes", "run_train"),
    ("experiments.round0113_nodes", "run_evaluate"),
    # the panel node -- `score_panel`, the `0.7843683944260816x` interval.
    ("experiments.round0218_nodes", "run_panel"),
    # the 2M-rung seed-family trainer and panel.
    ("experiments.round0230_nodes", "run_train"),
    ("experiments.round0230_nodes", "run_panel"),
    ("experiments.round0250_nodes", "run_train"),
    ("experiments.round0250_nodes", "run_panel"),
    # the 100M rung's own nodes: assembly hashes 153.6 GB, ladder and qualify
    # hold a live CUDA context while they do it.
    ("experiments.round0238_nodes", "run_reachability"),
    ("experiments.round0238_nodes", "run_assemble"),
    ("experiments.round0238_nodes", "run_truth"),
    ("experiments.round0238_nodes", "run_ladder"),
    ("experiments.round0238_nodes", "run_qualify"),
    ("experiments.round0240_nodes", "run_ladder"),
    ("experiments.round0240_nodes", "run_qualify"),
    # this round's own nodes, so the install is proved on the real entry path
    # the runner actually dispatches through.
    ("experiments.round0253_nodes", "run_install"),
    ("experiments.round0253_nodes", "run_writepath"),
    ("experiments.round0253_nodes", "run_cudasetup"),
)

#: The call names whose presence makes a node entry "a place a stop must be
#: observable": the two hash entry points R0252 instrumented, and the scorer.
BINDING_CALLS: tuple[str, ...] = (
    "expected_input_signature",
    "expected_input_signatures",
    "ordered_array_sha256",
    "sha256_file",
    "path_signature",
    "score_panel",
)

INSTALL_CALL_NAME = "install_stop_hooks"


class Round0253Error(RuntimeError):
    """The registered R0253 stop-hook install contract changed."""


class Round0253CooperativeAbort(RuntimeError):
    """A cooperative abort observed through an installed stop hook."""


# --------------------------------------------------------------------------- #
# the poll
# --------------------------------------------------------------------------- #


class RunnerAbortPoll:
    """Raise, in band, once the runner's cooperative abort flag exists.

    This is `experiments/round0238_nodes.py::_check_runner_abort`'s behaviour
    moved into `basemap/` so a release module can install it without importing
    `experiments`.  It reads `ROUNDRUN_ABORT_FLAG` through
    `basemap.gpu_child_supervision.runner_abort_reason`, allocates nothing on the
    common path, and never signals anything: raising unwinds the node through the
    normal Python path so whatever CUDA context it holds is torn down by the
    interpreter rather than by a signal.
    """

    __slots__ = ("label", "reads", "observed_site", "_reason")

    def __init__(self, *, label: str) -> None:
        self.label = str(label)
        self.reads = 0
        self.observed_site: str | None = None
        self._reason = runner_abort_reason

    def __call__(self, where: str) -> None:
        self.reads += 1
        reason = self._reason()
        if reason is not None:
            self.observed_site = str(where)
            raise Round0253CooperativeAbort(
                f"the round runner requested a cooperative abort (observed at "
                f"{where!r} under {self.label!r}): {reason}. Unwinding this node "
                "in-band; no signal was delivered and no CUDA context was killed."
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "RunnerAbortPoll",
            "label": self.label,
            "reads": int(self.reads),
            "observed_site": self.observed_site,
            "flag_env": "ROUNDRUN_ABORT_FLAG",
            "flag_path": os.environ.get("ROUNDRUN_ABORT_FLAG"),
            "signal_delivered": False,
        }


# --------------------------------------------------------------------------- #
# install / inspect / assert
# --------------------------------------------------------------------------- #


def _hooked_module(name: str):
    module = importlib.import_module(name)
    if not callable(getattr(module, "set_abort_poll", None)):
        raise Round0253Error(
            f"R0253 requires {name}.set_abort_poll; this release does not carry "
            "R0252's chunk-poll hook and a node on it cannot be stopped"
        )
    return module


def install_stop_hooks(*, label: str, poll: Any = None) -> dict[str, Any]:
    """Install the cooperative-stop poll into every hooked release module.

    Idempotent and additive.  Returns the receipt a node seals, which names the
    modules, the poll, and the previous value of each hook so a reader can see
    the process started unhooked.
    """
    installed = RunnerAbortPoll(label=str(label)) if poll is None else poll
    if not callable(installed):
        raise Round0253Error("R0253 stop hook must be callable")
    before: dict[str, Any] = {}
    for name in HOOKED_MODULES:
        module = _hooked_module(name)
        previous = module.set_abort_poll(installed)
        before[name] = None if previous is None else repr(previous)
    return {
        "schema": "round0253-stop-hook-install-v1",
        "label": str(label),
        "modules": list(HOOKED_MODULES),
        "hook": repr(installed),
        "hook_reads_the_runner_abort_flag": isinstance(installed, RunnerAbortPoll),
        "previous_hook_by_module": before,
        "the_process_started_unhooked": all(
            value is None for value in before.values()
        ),
        "installed_at_monotonic_s": time.monotonic(),
        "poll": installed,
    }


def stop_hooks_state() -> dict[str, Any]:
    """What is installed right now, read from the modules themselves."""
    by_module: dict[str, Any] = {}
    for name in HOOKED_MODULES:
        module = importlib.import_module(name)
        hook = getattr(module, "abort_poll", None)
        by_module[name] = {
            "has_set_abort_poll": callable(getattr(module, "set_abort_poll", None)),
            "abort_poll_installed": hook is not None,
            "abort_poll": None if hook is None else repr(hook),
            "declared_sites": list(getattr(module, "ABORT_POLL_SITES", ()) or ()),
        }
    return {
        "schema": "round0253-stop-hook-state-v1",
        "modules": by_module,
        "every_hooked_module_is_installed": all(
            entry["abort_poll_installed"] for entry in by_module.values()
        ),
        "modules_not_installed": sorted(
            name for name, entry in by_module.items()
            if not entry["abort_poll_installed"]
        ),
    }


def assert_stop_hooks_installed(*, label: str) -> dict[str, Any]:
    """Fail closed if any hooked module is unhooked.  Positive control in tests."""
    state = stop_hooks_state()
    if not state["every_hooked_module_is_installed"]:
        raise Round0253Error(
            f"R0253 {label}: the cooperative-stop hook is NOT installed in "
            f"{state['modules_not_installed']}. A node on this path cannot "
            "observe an abort request inside an integrity hash or a panel score."
        )
    return state


# --------------------------------------------------------------------------- #
# the AST audit -- installed, or not, per registered entry
# --------------------------------------------------------------------------- #


def _module_tree(name: str) -> tuple[str, ast.Module]:
    module = importlib.import_module(name)
    path = inspect.getsourcefile(module)
    if not path:
        raise Round0253Error(f"R0253 cannot locate source for {name}")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return path, ast.parse(source, filename=path)


def _function(tree: ast.Module, function: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            return node  # type: ignore[return-value]
    raise Round0253Error(f"R0253 registered entry {function!r} is not a module-level function")


def _first_statement(function_node: ast.FunctionDef) -> ast.stmt | None:
    body = list(function_node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body[0] if body else None


def _calls_install(statement: ast.stmt | None) -> bool:
    if statement is None:
        return False
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name == INSTALL_CALL_NAME:
                return True
    return False


def entry_install_audit(
    entries: Sequence[tuple[str, str]] = REGISTERED_ENTRIES,
) -> dict[str, Any]:
    """Per registered entry: is the install the FIRST statement of its body?

    First statement, not merely present: an install that happens after a
    153.6 GB `expected_input_signature` call is exactly the defect R0252 shipped,
    and "present somewhere in the function" would pass it.
    """
    rows: list[dict[str, Any]] = []
    trees: dict[str, tuple[str, ast.Module]] = {}
    for module_name, function in entries:
        if module_name not in trees:
            trees[module_name] = _module_tree(module_name)
        path, tree = trees[module_name]
        node = _function(tree, function)
        first = _first_statement(node)
        installs_first = _calls_install(first)
        binding = sorted({
            name
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call)
            for name in (
                [inner.func.id] if isinstance(inner.func, ast.Name)
                else [inner.func.attr] if isinstance(inner.func, ast.Attribute)
                else []
            )
            if name in BINDING_CALLS
        })
        rows.append({
            "module": module_name,
            "function": function,
            "source": path,
            "lineno": int(node.lineno),
            "installs_the_hook_first": bool(installs_first),
            "binding_calls_in_this_entry": binding,
        })
    missing = [row for row in rows if not row["installs_the_hook_first"]]
    return {
        "schema": "round0253-entry-install-audit-v1",
        "entries_audited": len(rows),
        "entries": rows,
        "entries_missing_the_install": [
            f"{row['module']}.{row['function']}" for row in missing
        ],
        "every_registered_entry_installs_the_hook_first": not missing,
    }


def assert_every_entry_installs(
    entries: Sequence[tuple[str, str]] = REGISTERED_ENTRIES,
) -> dict[str, Any]:
    audit = entry_install_audit(entries)
    if not audit["every_registered_entry_installs_the_hook_first"]:
        raise Round0253Error(
            "R0253: these ladder-train node entries do not install the "
            f"cooperative-stop hook as their first statement: "
            f"{audit['entries_missing_the_install']}"
        )
    return audit


# --------------------------------------------------------------------------- #
# the wider enumeration -- which node entries CAN bind, installed or not
# --------------------------------------------------------------------------- #


def reachability_audit(experiments_dir: str) -> dict[str, Any]:
    """Every `run_*` node entry in the release, and whether it can bind.

    A directly-textual scan of each `experiments/*_nodes.py` module: which
    module-level `run_*` functions call one of `BINDING_CALLS` anywhere in their
    own body, and whether the install is their first statement.  Modules are read
    as source, never imported, so a historical node with an unsatisfiable import
    still appears in the census.

    This is deliberately a *lower* bound on the binding set -- a call reached
    through a helper in another module is not counted here -- and it is published
    as such.  Its purpose is to say how much of the release the fourteen
    registered installs actually cover, rather than to assert that they cover it.
    """
    rows: list[dict[str, Any]] = []
    registered = {f"{module}.{function}" for module, function in REGISTERED_ENTRIES}
    for name in sorted(os.listdir(experiments_dir)):
        if not name.endswith("_nodes.py"):
            continue
        path = os.path.join(experiments_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        module_name = f"experiments.{name[: -len('.py')]}"
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("run_") or node.name == "run_job":
                continue
            binding = sorted({
                called
                for inner in ast.walk(node)
                if isinstance(inner, ast.Call)
                for called in (
                    [inner.func.id] if isinstance(inner.func, ast.Name)
                    else [inner.func.attr] if isinstance(inner.func, ast.Attribute)
                    else []
                )
                if called in BINDING_CALLS
            })
            if not binding:
                continue
            key = f"{module_name}.{node.name}"
            rows.append({
                "entry": key,
                "source": path,
                "lineno": int(node.lineno),
                "binding_calls": binding,
                "registered_for_install": key in registered,
                "installs_the_hook_first": _calls_install(_first_statement(node)),
            })
    installed = [row for row in rows if row["installs_the_hook_first"]]
    return {
        "schema": "round0253-binding-entry-census-v1",
        "entries_that_can_bind": len(rows),
        "entries_with_the_install": len(installed),
        "entries_without_the_install": len(rows) - len(installed),
        "coverage_fraction": (len(installed) / len(rows)) if rows else None,
        "entries": rows,
        "what_this_is_not": (
            "This counts a direct call in the entry's own body. An entry that "
            "reaches a binding call through a helper in another module is NOT "
            "counted, so `entries_that_can_bind` is a lower bound and "
            "`coverage_fraction` is an upper bound on coverage. It is published "
            "to size the gap between 'the two installs the review asked for' and "
            "'every node in the release', not to claim the release is covered."
        ),
        "why_the_uninstalled_ones_are_uninstalled": (
            "They belong to rounds that have already run to a terminal result. "
            "A real ladder train dispatches through the entries in "
            "REGISTERED_ENTRIES; the rest are historical and are listed so the "
            "residual is a number rather than an impression."
        ),
    }


# --------------------------------------------------------------------------- #
# what the hook is worth: the ceiling it is measured against
# --------------------------------------------------------------------------- #


def registered_ceiling_s() -> float:
    return float(registered_value("r0246_max_poll_spacing_s"))


def over_the_ceiling(seconds: float) -> float:
    return float(seconds) / registered_ceiling_s()


THE_INSTRUMENT_IS_DEFEATABLE = (
    "Every ceiling verdict this round publishes that comes from an AbortPollGate "
    "rests on review-0249-01 §B.1/§B.2, which are OPEN: the gate's `max_gap_s` is "
    "a plain mutable attribute and one assignment yields a passing receipt "
    "indistinguishable from an honest one. The owner has stopped guard work and "
    "R0253 adds none. The gap series this round publishes are recorded by "
    "PollRecorder / CoverageLedger independently of the gate, so the NUMBERS do "
    "not depend on the gate; the VERDICT strings do."
)

NOT_A_FAMILY_CELL = (
    "R0253 registers no floor, band, multiplier, estimator, gate or map. No "
    "measurement here is a family cell and no floor may be fitted to one."
)


__all__ = [
    "BINDING_CALLS",
    "CUDA_CAPABILITY",
    "CUDA_SCHEMA",
    "HOOKED_MODULES",
    "INSTALL_CAPABILITY",
    "INSTALL_SCHEMA",
    "NOT_A_FAMILY_CELL",
    "REGISTERED_ENTRIES",
    "ROUND_ID",
    "Round0253CooperativeAbort",
    "Round0253Error",
    "RunnerAbortPoll",
    "THE_INSTRUMENT_IS_DEFEATABLE",
    "WRITE_CAPABILITY",
    "WRITE_SCHEMA",
    "assert_every_entry_installs",
    "assert_stop_hooks_installed",
    "entry_install_audit",
    "install_stop_hooks",
    "over_the_ceiling",
    "reachability_audit",
    "registered_ceiling_s",
    "stop_hooks_state",
]

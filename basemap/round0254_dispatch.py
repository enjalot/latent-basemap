"""R0254 — install the stop hook where the runner actually dispatches.

review-0253-01 §A.2, the material finding:

    R0253 installed `install_stop_hooks` on 19 ladder-train entries and proved it
    three ways — but the proof resolved `run_train` / `run_panel` while the
    runner resolves `run_job`.  Across all queue manifests, **0 of 206 dispatched
    handlers were among the 19**.

`rounds/runner.py` does exactly three things to reach a node
(`runner.py:1925--1928`):

    module = importlib.import_module(handler_module)
    handler = getattr(module, handler_callable)
    handler(active, job)

So the entry the runner resolves is whatever `handler_callable` the queue
manifest names, and in this program that is almost always `run_job`.  Every
`run_job` in the live ladder modules is a thin action dispatcher that delegates
to a module-level `run_*` function.

This module therefore does two things R0253 did by hand:

* **Derives the entry list mechanically.**  `dispatch_census` reads every
  `queue.json` on this box and reports the `(handler_module, handler_callable)`
  pairs the runner has actually resolved.  `derived_entries` then takes a module
  and returns the dispatched callables for it **plus the transitive closure of
  the module-level `run_*` functions they delegate to**, computed from the AST.
  Nothing is nominated.  If someone adds an action to a `run_job` dispatch table,
  the derived list grows by itself and `assert_derived_entries_install` fails
  until the new entry carries the install.
* **Audits the install for EFFECTIVENESS, not for the presence of a name.**
  review-0253-01 §A.3 planted `if False: install_stop_hooks(...)`, a
  module-locally shadowed no-op, and a deferred `lambda` against R0253's own
  `entry_install_audit`, and **all three passed**, because `_calls_install`
  walked the first statement looking for a matching `Name.id` or
  `Attribute.attr`.  `install_effectiveness` below requires all four of:

  1. the first non-docstring statement is an *unconditional* expression or
     assignment whose value is directly the call — not an `If`, `Try`, `With`,
     loop, `Lambda`, comprehension or nested `def`;
  2. the module imports `install_stop_hooks` from this round's own source module
     and does not rebind it at module level;
  3. the entry function does not bind the name locally anywhere in its body (in
     Python a later `def install_stop_hooks` in the same function makes the
     first-statement call an `UnboundLocalError`, which no AST position check
     would notice);
  4. the name resolves **at runtime** to the identical function object this
     module imports, checked with `is`.

  (4) is what a shadow cannot survive and (1) is what dead or deferred code
  cannot survive.  Each of the four is exercised by a planted-defect control in
  `tests/test_round0254_contract.py`, and each control is itself verified against
  a deliberately weakened copy of the predicate that reproduces R0253's logic, so
  a control that could not fail is a test failure.

The scope decision this module does **not** hide: which *modules* are in scope is
a judgement (the live ladder), and it is published with its residual — every
dispatched module on the box, and how many of them carry the install.  What is
mechanical is the entry list *within* the scope, and its completeness check.
"""
from __future__ import annotations

import ast
import glob
import importlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from basemap.round0253_stop_hooks import (
    BINDING_CALLS,
    HOOKED_MODULES,
    install_stop_hooks,
)


class Round0254DispatchError(RuntimeError):
    """The registered R0254 dispatch-derivation contract changed."""


ROUND_ID = "0254"

DISPATCH_CAPABILITY = "round0254-dispatch-derived-stop-hook-install-v1"
DISPATCH_SCHEMA = "round0254-dispatch-derived-stop-hook-install-v1"
WRITEBACK_CAPABILITY = "round0254-write-stall-distribution-v1"
WRITEBACK_SCHEMA = "round0254-write-stall-distribution-v1"

INSTALL_CALL_NAME = "install_stop_hooks"
#: The one module the name is allowed to come from.  An install imported from
#: anywhere else is not the release's install.
INSTALL_SOURCE_MODULE = "basemap.round0253_stop_hooks"

#: Every `queue.json` the runner has ever bound on this box.
QUEUE_GLOB = "/data/latent-basemap/runs/**/queue.json"

#: The modules a Phase 2 ladder train dispatches, per
#: `guides/plan-minilm-100m-v2.md`: the trainer/graph family (`0113`), the panel
#: nodes (`0218`, `0230`, `0250`), the 100M rung's own nodes (`0238`, `0240`),
#: and the two stoppability rounds' nodes (`0253`, `0254`) so the install is
#: proved on the module the runner is dispatching right now.
#:
#: **This list is a scope decision, not a derivation.**  It is the one hand
#: choice in this module and it is published as such; `dispatch_census` prints
#: every dispatched module on the box beside it so the residual is a number.
SCOPE_MODULES: tuple[str, ...] = (
    "experiments.round0113_nodes",
    "experiments.round0218_nodes",
    "experiments.round0230_nodes",
    "experiments.round0238_nodes",
    "experiments.round0240_nodes",
    "experiments.round0250_nodes",
    "experiments.round0253_nodes",
    "experiments.round0254_nodes",
    "experiments.round0258_nodes",
    "experiments.round0259_nodes",
    "experiments.round0260_nodes",
    "experiments.round0263_nodes",
    "experiments.round0264_nodes",
    "experiments.round0265_nodes",
    "experiments.round0266_nodes",
    "experiments.round0267_nodes",
)

#: The callable the runner resolves when a module has never been dispatched yet
#: (this round's own module, at prepare time).  It is not a guess: `runner.py`
#: refuses any `handler_callable` that does not start with `run_`, and every
#: queue this program has issued since R0042 names `run_job`.
DEFAULT_HANDLER_CALLABLE = "run_job"

#: What counts as "a gate exists on this path".
GATE_CONSTRUCTORS: tuple[str, ...] = ("_node_gate", "AbortPollGate")


# --------------------------------------------------------------------------- #
# what the runner has actually dispatched
# --------------------------------------------------------------------------- #


def dispatch_census(queue_glob: str = QUEUE_GLOB) -> dict[str, Any]:
    """Every `(handler_module, handler_callable)` pair in every queue manifest.

    This is the ground truth review-0253-01 §A.2 used and R0253 did not: the
    runner resolves what the manifest names, so what the manifests name is what
    the install has to be on.
    """
    manifests = sorted(glob.glob(queue_glob, recursive=True))
    pairs: dict[tuple[str, str], int] = {}
    unreadable = 0
    jobs_total = 0
    jobs_with_a_handler = 0
    for path in manifests:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            unreadable += 1
            continue
        for job in manifest.get("jobs") or []:
            if not isinstance(job, dict):
                continue
            jobs_total += 1
            module = job.get("handler_module")
            callable_name = job.get("handler_callable")
            if not isinstance(module, str) or not isinstance(callable_name, str):
                continue
            jobs_with_a_handler += 1
            pairs[(module, callable_name)] = pairs.get((module, callable_name), 0) + 1

    rows = [
        {"module": module, "callable": name, "queue_jobs": count}
        for (module, name), count in sorted(pairs.items())
    ]
    by_callable: dict[str, int] = {}
    for row in rows:
        by_callable[row["callable"]] = by_callable.get(row["callable"], 0) + 1
    return {
        "schema": "round0254-dispatch-census-v1",
        "queue_glob": queue_glob,
        "queue_manifests_scanned": len(manifests),
        "queue_manifests_unreadable": unreadable,
        "jobs_seen": jobs_total,
        "jobs_naming_a_handler": jobs_with_a_handler,
        "distinct_dispatched_handlers": len(rows),
        "distinct_dispatched_modules": len({row["module"] for row in rows}),
        "dispatched_callables_by_name": dict(sorted(by_callable.items())),
        "handlers": rows,
        "how_the_runner_resolves_one": (
            "rounds/runner.py: importlib.import_module(handler_module); "
            "getattr(module, handler_callable); handler(active, job). Three "
            "lines, and the second one is the entry an install has to be on."
        ),
    }


def dispatched_callables_for(
    module: str, census: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    rows = (census or dispatch_census())["handlers"]
    names = sorted({
        str(row["callable"]) for row in rows if str(row["module"]) == module
    })
    return tuple(names) if names else (DEFAULT_HANDLER_CALLABLE,)


# --------------------------------------------------------------------------- #
# the delegation graph, from the AST
# --------------------------------------------------------------------------- #


def _module_source(module_name: str) -> tuple[str, str]:
    module = importlib.import_module(module_name)
    path = inspect.getsourcefile(module)
    if not path:
        raise Round0254DispatchError(f"R0254 cannot locate source for {module_name}")
    with open(path, "r", encoding="utf-8") as handle:
        return path, handle.read()


def _module_tree(module_name: str) -> tuple[str, ast.Module]:
    path, source = _module_source(module_name)
    return path, ast.parse(source, filename=path)


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node  # type: ignore[misc]
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            func = inner.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def delegation_graph(module_name: str) -> dict[str, Any]:
    """Which module-level `run_*` functions each `run_*` function calls."""
    path, tree = _module_tree(module_name)
    functions = _module_functions(tree)
    runs = {name for name in functions if name.startswith("run_")}
    edges = {
        name: sorted(_called_names(functions[name]) & runs - {name})
        for name in sorted(runs)
    }
    return {
        "module": module_name,
        "source": path,
        "run_functions": sorted(runs),
        "delegates_to": edges,
    }


def derived_entries(
    modules: Sequence[str] = SCOPE_MODULES,
    census: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The entry list, derived: dispatched callables plus their delegates.

    For each module in scope, start from the callables the runner has actually
    resolved for it (`dispatch_census`), then close transitively over the
    module-level `run_*` functions those callables call.  That is the set of
    functions a dispatched node can execute at top level, and therefore the set
    an install has to cover.
    """
    census = census or dispatch_census()
    entries: list[tuple[str, str]] = []
    detail: list[dict[str, Any]] = []
    for module_name in modules:
        graph = delegation_graph(module_name)
        roots = [
            name for name in dispatched_callables_for(module_name, census)
            if name in graph["run_functions"]
        ]
        if not roots:
            raise Round0254DispatchError(
                f"R0254 scope module {module_name} exposes none of the callables "
                f"the runner dispatches for it "
                f"({dispatched_callables_for(module_name, census)})"
            )
        seen: list[str] = []
        frontier = list(roots)
        while frontier:
            name = frontier.pop(0)
            if name in seen:
                continue
            seen.append(name)
            frontier.extend(graph["delegates_to"].get(name, ()))
        for name in sorted(seen):
            entries.append((module_name, name))
        detail.append({
            "module": module_name,
            "dispatched_callables": list(dispatched_callables_for(module_name, census)),
            "roots_present_in_this_module": sorted(roots),
            "derived_entries": sorted(seen),
            "derived_entry_count": len(seen),
            "delegates_to": graph["delegates_to"],
        })
    return {
        "schema": "round0254-derived-entry-list-v1",
        "scope_modules": list(modules),
        "entries": [{"module": module, "function": name} for module, name in entries],
        "entry_count": len(entries),
        "by_module": detail,
        "how_this_is_derived": (
            "Dispatched callables come from every queue.json on the box, not "
            "from an author's list; the rest is the transitive closure of "
            "module-level run_* calls inside them, taken from the AST. Adding an "
            "action to a run_job dispatch table grows this list automatically."
        ),
        "the_one_hand_choice": (
            "Which MODULES are in scope. That is a judgement about which rounds "
            "a Phase 2 ladder train will re-dispatch, and it is published beside "
            "`dispatch_census`'s full list of dispatched modules so the residual "
            "is a number rather than an impression."
        ),
    }


def entry_tuples(derived: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(row["module"]), str(row["function"])) for row in derived["entries"]
    )


# --------------------------------------------------------------------------- #
# the audit that cannot be satisfied by a dead call
# --------------------------------------------------------------------------- #


def _strip_docstring(body: Sequence[ast.stmt]) -> list[ast.stmt]:
    statements = list(body)
    if (statements and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        return statements[1:]
    return statements


def _first_statement_call(function_node: ast.FunctionDef) -> tuple[ast.Call | None, str]:
    """The call the first statement IS, not a call the first statement contains.

    `ast.walk` over the first statement is what let `if False:
    install_stop_hooks(...)` pass R0253's audit. Only three statement shapes can
    execute an install unconditionally as the first thing a function does: a bare
    expression, an assignment, and an annotated assignment.
    """
    body = _strip_docstring(function_node.body)
    if not body:
        return None, "the function body is empty"
    statement = body[0]
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return statement.value, ""
    if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(
            getattr(statement, "value", None), ast.Call):
        return statement.value, ""  # type: ignore[return-value]
    return None, (
        "the first statement is a "
        f"{type(statement).__name__}, which cannot execute an install "
        "unconditionally (dead, deferred or conditional code is not an install)"
    )


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _module_level_binding(tree: ast.Module) -> dict[str, Any]:
    imported_from: str | None = None
    rebound_at: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound == INSTALL_CALL_NAME:
                    if alias.name == INSTALL_CALL_NAME:
                        imported_from = node.module or ""
                    else:
                        rebound_at.append(int(node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == INSTALL_CALL_NAME:
                rebound_at.append(int(node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == INSTALL_CALL_NAME:
                    rebound_at.append(int(node.lineno))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == INSTALL_CALL_NAME:
                rebound_at.append(int(node.lineno))
    return {
        "imported_from": imported_from,
        "imported_from_the_release_module": imported_from == INSTALL_SOURCE_MODULE,
        "rebound_at_module_level_linenos": sorted(rebound_at),
    }


def _local_shadow(function_node: ast.FunctionDef) -> list[int]:
    """Any binding of the name inside the function makes it local THROUGHOUT.

    Python decides local-vs-global per scope, not per line, so a `def
    install_stop_hooks(...)` anywhere in the body turns the first-statement call
    into an `UnboundLocalError` — a shape a first-statement position check
    cannot see.
    """
    shadows: list[int] = []
    for node in ast.walk(function_node):
        if node is function_node:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == INSTALL_CALL_NAME:
                shadows.append(int(node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == INSTALL_CALL_NAME:
                    shadows.append(int(node.lineno))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            target = getattr(node, "target", None)
            if isinstance(target, ast.Name) and target.id == INSTALL_CALL_NAME:
                shadows.append(int(node.lineno))
        elif isinstance(node, ast.arg) and node.arg == INSTALL_CALL_NAME:
            shadows.append(int(getattr(node, "lineno", -1)))
    return sorted(shadows)


def install_effectiveness(
    module_name: str, function_name: str, *, resolve: bool = True
) -> dict[str, Any]:
    """Is the install present, unconditional, unshadowed and correctly bound?

    Four independent checks, each with its own planted-defect control. The audit
    passes only if all four hold; every failure names which one and why.
    """
    path, tree = _module_tree(module_name)
    functions = _module_functions(tree)
    node = functions.get(function_name)
    if node is None:
        raise Round0254DispatchError(
            f"R0254 entry {function_name!r} is not a module-level function of "
            f"{module_name}"
        )
    call, position_reason = _first_statement_call(node)
    unconditional = call is not None and _call_name(call) == INSTALL_CALL_NAME
    if call is not None and not unconditional:
        position_reason = (
            f"the first statement calls {_call_name(call)!r}, not "
            f"{INSTALL_CALL_NAME!r}"
        )

    binding = _module_level_binding(tree)
    shadows = _local_shadow(node)

    resolves = None
    resolved_to = None
    if resolve:
        module = importlib.import_module(module_name)
        candidate = getattr(module, INSTALL_CALL_NAME, None)
        resolved_to = None if candidate is None else repr(candidate)
        resolves = candidate is install_stop_hooks

    reasons: list[str] = []
    if not unconditional:
        reasons.append(position_reason or "the first statement is not the install")
    if not binding["imported_from_the_release_module"]:
        reasons.append(
            f"the module does not import {INSTALL_CALL_NAME} from "
            f"{INSTALL_SOURCE_MODULE} under its own name"
        )
    if binding["rebound_at_module_level_linenos"]:
        reasons.append(
            f"{INSTALL_CALL_NAME} is rebound at module level at lines "
            f"{binding['rebound_at_module_level_linenos']}"
        )
    if shadows:
        reasons.append(
            f"{INSTALL_CALL_NAME} is bound locally inside the entry at lines "
            f"{shadows}, which makes the first-statement call an UnboundLocalError"
        )
    if resolve and not resolves:
        reasons.append(
            f"{module_name}.{INSTALL_CALL_NAME} does not resolve to the release "
            f"function (got {resolved_to})"
        )

    binding_calls = sorted(_called_names(node) & set(BINDING_CALLS))
    return {
        "entry": f"{module_name}.{function_name}",
        "module": module_name,
        "function": function_name,
        "source": path,
        "lineno": int(node.lineno),
        "install_is_the_first_statement_unconditionally": bool(unconditional),
        "imported_from_the_release_module": binding["imported_from_the_release_module"],
        "imported_from": binding["imported_from"],
        "rebound_at_module_level_linenos": binding["rebound_at_module_level_linenos"],
        "locally_shadowed_at_linenos": shadows,
        "resolves_to_the_release_function": resolves,
        "resolved_to": resolved_to,
        "binding_calls_in_this_entry": binding_calls,
        "the_install_is_effective": not reasons,
        "why_not": reasons,
    }


def entry_install_audit(
    entries: Sequence[tuple[str, str]], *, resolve: bool = True
) -> dict[str, Any]:
    rows = [
        install_effectiveness(module, function, resolve=resolve)
        for module, function in entries
    ]
    ineffective = [row for row in rows if not row["the_install_is_effective"]]
    return {
        "schema": "round0254-effective-install-audit-v1",
        "entries_audited": len(rows),
        "entries_with_an_effective_install": len(rows) - len(ineffective),
        "entries_without_an_effective_install": [row["entry"] for row in ineffective],
        "every_entry_installs_effectively": not ineffective,
        "entries": rows,
        "what_this_checks_that_r0253_did_not": (
            "Unconditional position (not `ast.walk` over the first statement, "
            "which passed `if False:`), module-level and function-local "
            "rebinding, and runtime identity with the release function. "
            "review-0253-01 §A.3 planted three shapes against R0253's audit and "
            "all three passed; all three fail here, each with its own control."
        ),
    }


def assert_derived_entries_install(
    modules: Sequence[str] = SCOPE_MODULES,
    census: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The completeness check R0253's hand-written list could not have.

    Derives the entry list from the dispatch path, then requires every derived
    entry to install effectively.  A new action in a `run_job` dispatch table
    enlarges the derived list and fails this guard until it carries the install.
    """
    derived = derived_entries(modules, census)
    audit = entry_install_audit(entry_tuples(derived))
    if not audit["every_entry_installs_effectively"]:
        raise Round0254DispatchError(
            "R0254: these dispatch-derived node entries do not install the "
            "cooperative-stop hook effectively as their first statement: "
            f"{audit['entries_without_an_effective_install']}"
        )
    return {
        "schema": "round0254-derived-entry-install-guard-v1",
        "derived": derived,
        "audit": audit,
    }


# --------------------------------------------------------------------------- #
# installation is not enforcement
# --------------------------------------------------------------------------- #


def gate_census(
    entries: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Does each entry construct an `AbortPollGate`, at any depth in its module?

    review-0253-01 §F: 17 of R0253's 19 entries construct no gate at any depth,
    so they emit no gap series at all and their stop latency reads as *silence*
    rather than as a measured zero.  Installation is not enforcement, and this
    counts both.

    Transitive within the entry's own module: an entry that reaches `_node_gate`
    through a module-local helper counts.  An entry that reaches one through a
    helper in a different module does not, so `entries_that_construct_a_gate` is
    a **lower** bound and `install_and_gate` with it.
    """
    rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[dict[str, ast.FunctionDef], dict[str, set[str]]]] = {}
    for module_name, function_name in entries:
        if module_name not in cache:
            _path, tree = _module_tree(module_name)
            functions = _module_functions(tree)
            calls = {name: _called_names(node) for name, node in functions.items()}
            cache[module_name] = (functions, calls)
        functions, calls = cache[module_name]
        node = functions.get(function_name)
        if node is None:
            continue
        seen: set[str] = set()
        frontier = [function_name]
        gate_at: str | None = None
        while frontier:
            name = frontier.pop(0)
            if name in seen or name not in calls:
                continue
            seen.add(name)
            called = calls[name]
            if called & set(GATE_CONSTRUCTORS):
                gate_at = name
                break
            frontier.extend(sorted(called & set(functions) - seen))
        install = install_effectiveness(module_name, function_name)
        rows.append({
            "entry": f"{module_name}.{function_name}",
            "constructs_a_gate": gate_at is not None,
            "gate_constructed_in": gate_at,
            "module_local_functions_walked": len(seen),
            "installs_effectively": install["the_install_is_effective"],
            "binding_calls_in_this_entry": install["binding_calls_in_this_entry"],
        })
    with_gate = [row for row in rows if row["constructs_a_gate"]]
    with_install = [row for row in rows if row["installs_effectively"]]
    both = [row for row in rows
            if row["constructs_a_gate"] and row["installs_effectively"]]
    return {
        "schema": "round0254-install-and-gate-census-v1",
        "entries": rows,
        "entries_audited": len(rows),
        "entries_that_install_effectively": len(with_install),
        "entries_that_construct_a_gate": len(with_gate),
        "entries_that_both_install_and_gate": len(both),
        "entries_that_install_but_construct_no_gate": sorted(
            row["entry"] for row in rows
            if row["installs_effectively"] and not row["constructs_a_gate"]
        ),
        "what_a_missing_gate_means": (
            "Not a measured zero. An entry with no gate constructs no poll "
            "recorder and emits no gap series, so its stop latency is SILENCE "
            "and it will read as `not instrumented` in the next census exactly "
            "as R0252's write path did. An install lets such a node stop; it "
            "does not let anyone say how fast."
        ),
        "why_this_is_a_lower_bound": (
            "The walk is transitive within the entry's own module only. An entry "
            "reaching a gate through a helper in another module is counted as "
            "having none."
        ),
    }


def scope_residual(
    census: Mapping[str, Any], modules: Sequence[str] = SCOPE_MODULES
) -> dict[str, Any]:
    """Every dispatched module on the box, and whether it is in scope.

    The honest denominator, published without being asked for: review-0253-01
    §B's point is that `19` was an author-nominated set with no mechanical tie to
    anything.  The tie here is the queue manifests, and this is what they say
    about the part of the release this round does NOT cover.
    """
    dispatched = sorted({str(row["module"]) for row in census["handlers"]})
    in_scope = [name for name in dispatched if name in set(modules)]
    out_of_scope = [name for name in dispatched if name not in set(modules)]
    never_dispatched = [name for name in modules if name not in set(dispatched)]
    return {
        "schema": "round0254-scope-residual-v1",
        "dispatched_modules": len(dispatched),
        "dispatched_modules_in_scope": in_scope,
        "dispatched_modules_out_of_scope": len(out_of_scope),
        "dispatched_modules_out_of_scope_names": out_of_scope,
        "scope_modules_never_yet_dispatched": never_dispatched,
        "module_coverage_fraction": (
            len(in_scope) / len(dispatched) if dispatched else None
        ),
        "why_the_out_of_scope_ones_are_out_of_scope": (
            "They belong to rounds that have run to a terminal result and that "
            "no Phase 2 ladder train re-dispatches. That is a claim about the "
            "future queue, which does not exist yet and is chosen by the author "
            "of the round that issues it — so this fraction is published rather "
            "than argued, and a Phase 2 queue naming any of these modules must "
            "extend SCOPE_MODULES before it can pass "
            "`assert_derived_entries_install`."
        ),
    }


__all__ = [
    "DEFAULT_HANDLER_CALLABLE",
    "DISPATCH_CAPABILITY",
    "DISPATCH_SCHEMA",
    "GATE_CONSTRUCTORS",
    "HOOKED_MODULES",
    "INSTALL_CALL_NAME",
    "INSTALL_SOURCE_MODULE",
    "QUEUE_GLOB",
    "ROUND_ID",
    "Round0254DispatchError",
    "SCOPE_MODULES",
    "WRITEBACK_CAPABILITY",
    "WRITEBACK_SCHEMA",
    "assert_derived_entries_install",
    "delegation_graph",
    "derived_entries",
    "dispatch_census",
    "dispatched_callables_for",
    "entry_install_audit",
    "entry_tuples",
    "gate_census",
    "install_effectiveness",
    "scope_residual",
]

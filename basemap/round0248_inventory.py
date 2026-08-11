"""R0248 — derive the safety-parameter inventory MECHANICALLY, from the source.

Four rounds have now shipped a hand-written inventory and four reviews have
found something it omitted. review-0247-01 found `replay`; review-0246-01 found
the coverage denominator; review-0245-01 found the slope and the headroom. The
omission is never the same parameter twice, so the fix is not "add this one" —
it is to stop writing the list by hand.

This module enumerates, by parsing the source of the guard modules with `ast`:

1. **Every module-level constant that a gate compares against.**
   `derive_gate_constant_inventory()` walks each guarded module, collects the
   module-level `UPPER_CASE` names it assigns or imports, and then finds every
   `ast.Compare` whose operands read one of them. Each hit is classified:

   * `registry_read` — the comparison calls `registered_value("…")` or
     `sampler_max_anonymous_bytes()`. This is the sanctioned shape.
   * `bare_registered_symbol` — the comparison reads a bare module-level name
     whose symbol is REGISTERED. **This is the R0248 gap-1/gap-2 defect**:
     registering a number changes nothing when the `if` statement reads a
     mirror of it. `round0246_guard:278,289` and the three
     `SAMPLER_MAX_ANONYMOUS_BYTES` verdict arms were exactly this, and two
     assignments defeated them while the receipt kept printing the registry's
     number.
   * `triaged` — the constant is in `EXAMINED_NOT_SAFETY` or in
     `NOT_A_BOUND` below, with a reason.
   * `untriaged` — nobody has looked at it. Fails closed.

2. **Every declaration that waives a gate arm.**
   `derive_arm_waiver_inventory()` finds the failure-arm tuples inside
   `require`-shaped functions and reports every arm whose truth value is a
   disjunction with a `self.<attr>` or a bare name — the shape by which
   `replay` switched off two arms and `training_performed` switched off one.
   Each waiver must name a REGISTERED parameter.

`require_inventory_complete()` raises if either derivation finds a row that is
not registry-read or triaged. It is called by every R0248 node and by the
contract tests, so the next omission is a failing test rather than a review
finding.

**What this does not do.** It reads source, so it cannot see a bound reached by
`getattr`, by a dict lookup on a computed key, or by a C extension; and it
cannot stop a node that deliberately rewrites the registry at runtime. It
catches *mistakes* — a bound registered and then compared against its mirror —
which is the failure mode of a runner following instructions in good faith.
The bound that does not live in this process is `basemap.round0248_external`.

Nothing here signals anything, starts a child process, or touches the GPU.
"""
from __future__ import annotations

import ast
import os
from typing import Any

from basemap.round0247_registry import (
    EXAMINED_NOT_SAFETY,
    REGISTERED_SAFETY_PARAMETERS,
    Round0247Error,
    registry_fingerprint,
    verify_registry,
)

ROUND_ID = "0248"

#: The modules whose comparisons are gate decisions. A module added to the
#: guard family and not added here is itself a defect, so the contract test
#: asserts this list against the `basemap/round024*_guard.py` glob.
GUARDED_MODULES: tuple[str, ...] = (
    "basemap/round0244_guard.py",
    "basemap/round0244_prereq.py",
    "basemap/round0245_guard.py",
    "basemap/round0246_guard.py",
    "basemap/round0247_guard.py",
    "basemap/round0247_registry.py",
    "basemap/round0248_guard.py",
    "basemap/round0248_external.py",
    "experiments/round0244_nodes.py",
    "experiments/round0245_nodes.py",
    "experiments/round0246_nodes.py",
    "experiments/round0247_nodes.py",
    "experiments/round0248_nodes.py",
)

#: The sanctioned comparison-site reads. A comparison whose operand is one of
#: these calls resolves the registry at the moment of the comparison, so a
#: module-global assignment is not a decision surface.
REGISTRY_READERS: frozenset[str] = frozenset({
    "registered_value",
    "_registered_value",
    "sampler_max_anonymous_bytes",
    "registered_bounds",
})

#: Module-level constants a gate compares against that are examined and are NOT
#: safety bounds, with the reason. This is the ONLY hand-maintained list left,
#: and every entry in it is a claim a reviewer can check against the source.
NOT_A_BOUND: dict[str, str] = {
    "R0243_DIRECTED_EDGES": (
        "an identity: the edge count R0243 sealed. A comparison against it "
        "asks 'is this the array R0243 sealed', not 'is this within a bound'. "
        "Moving it makes the binding check FAIL, never pass"
    ),
    "R0243_UNDIRECTED_EDGES": "same identity check as R0243_DIRECTED_EDGES",
    "R0238_PROVENANCE_SHA256": "a digest identity, not a bound",
    "REGISTERED_REGISTRY_SHA256": (
        "the pinned registry digest. It is the thing verify_registry() "
        "compares the observed fingerprint AGAINST; moving it is a source "
        "change that shows in the diff, which is the whole mechanism"
    ),
    "SAMPLER_MIN_CHI_SQUARE_P": (
        "a statistical acceptance threshold for a fidelity test on a SAMPLER "
        "DRAW, not a bound on machine resources or on whether a stage may "
        "continue. Loosening it makes the round's own science claim weaker, "
        "which is self-punishing rather than self-serving"
    ),
    "SAMPLER_MAX_ABS_Z": "same class as SAMPLER_MIN_CHI_SQUARE_P",
    "SAMPLER_MIN_DRAWS_PER_S": (
        "a throughput floor on a measurement stage. Missing it aborts the "
        "node; it governs no resource and stops no runaway"
    ),
    "SAMPLER_EPOCHS": (
        "UMAP's epoch count, used to derive which edges are never sampled. A "
        "property of the distribution being profiled, not a guard"
    ),
    "GRAPH_K": "the graph's k. An identity of the inherited artifact",
    "DIMENSION": "the substrate's width. An identity of the inherited artifact",
    "ROWS": "the substrate's row count. An identity",
    "TRUTH_PROBE_ROWS": "the probe's row count. An identity",
    "PROBE_CANDIDATE_DECISIONS": (
        "already published in EXAMINED_NOT_SAFETY: the size of the probe's "
        "decision population, a fact about R0238's truth build"
    ),
    "MIN_SOFT_DEADLINE_SECONDS": (
        "a runner-side floor on the soft deadline, not importable by a node"
    ),
    # -- dispatch identities: `if action == X`. Moving one routes the node to a
    #    different handler or to a hard refusal; none of them relaxes anything.
    "ROUND_ID": "the round's own id, compared to refuse another queue's manifest",
    "DID_ACTION": "an action name, compared for handler dispatch",
    "GUARD_ACTION": "an action name, compared for handler dispatch",
    "SAMPLER_ACTION": "an action name, compared for handler dispatch",
    "TEXT_ACTION": "an action name, compared for handler dispatch",
    "WATCHDOG_ACTION": "an action name, compared for handler dispatch",
    "TIE_ACTION": "an action name, compared for handler dispatch",
    "PARAMGUARD_ACTION": "an action name, compared for handler dispatch",
    "TRUTHCOS_ACTION": "an action name, compared for handler dispatch",
    "GAPGUARD_ACTION": "an action name, compared for handler dispatch",
    "EXTERNAL_ACTION": "an action name, compared for handler dispatch",
    # -- the registry's own vocabulary
    "CEILING": (
        "a direction TAG ('ceiling'/'floor') compared inside the registry to "
        "decide which way 'weaker' points. It is a string, not a bound"
    ),
    "FLOOR": "the other direction tag; same reason as CEILING",
    "REGISTERED_SAFETY_PARAMETERS": (
        "the registry mapping itself. The comparison is a MEMBERSHIP test "
        "(`'replay' in REGISTERED_SAFETY_PARAMETERS`) asking whether a "
        "parameter is registered at all - it reads no value and bounds "
        "nothing. Its integrity is the fingerprint's job"
    ),
    "REGISTERED_ABORT_READERS": (
        "the source-level allowlist itself. The comparison `name in "
        "REGISTERED_ABORT_READERS` IS the sanction mechanism R0248 gap 3 "
        "moved sanction onto; adding to it is a source change in the diff"
    ),
    # -- sealed identities from prior rounds: `does the array I loaded equal
    #    the array R0243/R0244/R0245 sealed?`. Moving one makes the binding
    #    check FAIL, never pass, so they are self-punishing.
    "R0243_ENTRIES_AT_OR_ABOVE_ONE": "a sealed identity of R0243's graph",
    "R0243_RHOS_MEAN": "a sealed identity of R0243's graph",
    "R0243_SIGMAS_MEAN": "a sealed identity of R0243's graph",
    "R0243_WEIGHT_MAX": "a sealed identity of R0243's graph",
    "R0243_WEIGHT_MIN": "a sealed identity of R0243's graph",
    "R0243_STRICT_BUILDER_MISSING_EDGES": "a sealed identity of R0243's build",
    "R0243_TIE_AWARE_BUILDER_MISSING_EDGES": "a sealed identity of R0243's build",
    "R0244_TRIP_TRACE_SLOPE_BYTES_PER_S": (
        "a sealed identity of R0244's trip trace, compared for agreement"
    ),
    "R0244_WATCHDOG_RECEIPT_PEAK_BYTES": (
        "a sealed identity of R0244's watchdog receipt, compared for agreement"
    ),
    "R0245_SEALED_SAMPLER_SAMPLES": "a sealed identity of R0245's sampler node",
    "R0245_SEALED_SAMPLER_BOUNDARY_POLLS": "a sealed identity of R0245's node",
    "R0245_SEALED_SAMPLER_EXPECTED_SAMPLES": "a sealed identity of R0245's node",
    "R0245_SEALED_DISTINCT_EDGES_DRAWN": "a sealed identity of R0245's draw",
    # -- measurement thresholds whose risk is not a machine risk
    "NEAR_IDENTICAL_JACCARD": (
        "a LABEL boundary for describing how much two neighbour sets overlap. "
        "It classifies a reported number; it stops nothing and bounds nothing"
    ),
    "SUBSTANTIAL_JACCARD": "same label boundary as NEAR_IDENTICAL_JACCARD",
    "SOME_OVERLAP_JACCARD": "same label boundary as NEAR_IDENTICAL_JACCARD",
    "TEXT_BINDING_COSINE_FLOOR": (
        "a floor on how well recovered text must match its sealed embedding "
        "for a binding claim. Lowering it weakens the round's own scientific "
        "claim rather than permitting a stage to run"
    ),
    "CLUSTER_UNDER_TEST": "the cluster id under test; a selection, not a bound",
    "EXCLUDED_SHARDS": "a shard exclusion set; a selection, not a bound",
    # -- R0242's conjunctive machine rule, two of whose three thresholds are
    #    already published in EXAMINED_NOT_SAFETY
    "WATCHDOG_MEM_AVAILABLE_BYTES": (
        "the second term of R0242's conjunctive machine rule (available "
        "memory). Like WATCHDOG_SWAP_GROWTH_BYTES it is inherited unchanged "
        "through two rounds and is not reachable through any keyword; the ONE "
        "of the three a node can reach - the anonymous budget it declares - "
        "is registered, and R0248 routes the anonymous term itself through "
        "the registry at the comparison site"
    ),
    "NODE_HEADROOM_BYTES": (
        "a DERIVED quantity: WATCHDOG_ANON_BYTES - NODE_ANON_BUDGET_BYTES, "
        "both of which are themselves triaged or registered. It is the node's "
        "own arithmetic on two upstream numbers, published in the receipt, "
        "and the gate's binding headroom is the registered "
        "max_declared_headroom_bytes rather than this"
    ),
}


def _module_source(repo_root: str, relative: str) -> str | None:
    path = os.path.join(repo_root, relative)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _module_level_constants(tree: ast.Module) -> dict[str, str]:
    """Every `UPPER_CASE` name this module assigns or imports at module level."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants[target.id] = "assigned"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                constants[node.target.id] = "assigned"
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name.isupper():
                    constants[name] = f"imported from {node.module}"
    return constants


def _compare_operand_names(node: ast.Compare) -> set[str]:
    """Names read by a comparison, seeing through `float(X)` / `int(X)`."""
    names: set[str] = set()
    for operand in [node.left, *node.comparators]:
        target = operand
        while (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and target.func.id in {"float", "int", "bool", "abs"}
            and target.args
        ):
            target = target.args[0]
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _compare_reads_the_registry(node: ast.Compare) -> bool:
    for operand in [node.left, *node.comparators]:
        for inner in ast.walk(operand):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in REGISTRY_READERS
            ):
                return True
    return False


def _registered_symbols() -> dict[str, str]:
    symbols: dict[str, str] = {}
    for parameter in REGISTERED_SAFETY_PARAMETERS.values():
        symbols[str(parameter.symbol).split(".")[-1]] = parameter.name
    return symbols


def _triaged_symbols() -> dict[str, str]:
    triaged = dict(NOT_A_BOUND)
    for row in EXAMINED_NOT_SAFETY:
        triaged[str(row["symbol"]).split(".")[-1]] = str(row["why_not"])
    return triaged


def derive_gate_constant_inventory(*, repo_root: str) -> dict[str, Any]:
    """Every module-level constant any guarded module compares against."""
    verify_registry(label="R0248 gate-constant inventory")
    registered = _registered_symbols()
    triaged = _triaged_symbols()
    rows: list[dict[str, Any]] = []
    modules_read: list[str] = []
    for relative in GUARDED_MODULES:
        source = _module_source(repo_root, relative)
        if source is None:
            continue
        modules_read.append(relative)
        tree = ast.parse(source, filename=relative)
        constants = _module_level_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            reads_registry = _compare_reads_the_registry(node)
            for name in sorted(_compare_operand_names(node) & set(constants)):
                if name in registered:
                    status = (
                        "registry_read" if reads_registry
                        else "bare_registered_symbol"
                    )
                    reason = registered[name]
                elif name in triaged:
                    status = "triaged"
                    reason = triaged[name]
                else:
                    status = "untriaged"
                    reason = ""
                rows.append({
                    "module": relative,
                    "symbol": name,
                    "line": int(node.lineno),
                    "origin": constants[name],
                    "status": status,
                    "registered_parameter_or_reason": reason,
                    "the_comparison_reads_the_registry": bool(reads_registry),
                })
    defects = [
        row for row in rows
        if row["status"] in {"bare_registered_symbol", "untriaged"}
    ]
    return {
        "instrument": "round0248-gate-constant-inventory-v1",
        "modules_read": modules_read,
        "modules_declared": list(GUARDED_MODULES),
        "comparisons_against_module_level_constants": len(rows),
        "rows": rows,
        "distinct_symbols": sorted({str(row["symbol"]) for row in rows}),
        "defects": defects,
        "holds": not defects,
        "registry_fingerprint": registry_fingerprint(),
        "note": (
            "a comparison that reads a bare module-level name whose symbol is "
            "registered is the R0248 gap-1/gap-2 defect: the number is "
            "registered and the decision reads a mirror of it"
        ),
    }


class _ArmWaiverVisitor(ast.NodeVisitor):
    """Find `("arm_name", <x> or self.<flag>)` inside a failure-arm tuple."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.rows: list[dict[str, Any]] = []

    def visit_Tuple(self, node: ast.Tuple) -> None:  # noqa: N802 - ast API
        if (
            len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[0].value, str)
        ):
            arm = str(node.elts[0].value)
            for inner in ast.walk(node.elts[1]):
                if not isinstance(inner, ast.BoolOp):
                    continue
                if not isinstance(inner.op, ast.Or):
                    continue
                for operand in inner.values:
                    flag = None
                    if (
                        isinstance(operand, ast.Attribute)
                        and isinstance(operand.value, ast.Name)
                        and operand.value.id == "self"
                    ):
                        flag = operand.attr
                    elif isinstance(operand, ast.Name) and operand.id.islower():
                        flag = operand.id
                    if flag is not None:
                        self.rows.append({
                            "module": self.module,
                            "arm": arm,
                            "waived_by": flag,
                            "line": int(inner.lineno),
                        })
        self.generic_visit(node)


def derive_arm_waiver_inventory(*, repo_root: str) -> dict[str, Any]:
    """Every gate arm whose truth value a declaration can switch off."""
    verify_registry(label="R0248 arm-waiver inventory")
    rows: list[dict[str, Any]] = []
    for relative in GUARDED_MODULES:
        source = _module_source(repo_root, relative)
        if source is None:
            continue
        visitor = _ArmWaiverVisitor(relative)
        visitor.visit(ast.parse(source, filename=relative))
        rows.extend(visitor.rows)
    for row in rows:
        name = str(row["waived_by"])
        row["registered"] = bool(name in REGISTERED_SAFETY_PARAMETERS)
        row["registered_value"] = (
            float(REGISTERED_SAFETY_PARAMETERS[name].value)
            if row["registered"] else None
        )
    defects = [row for row in rows if not row["registered"]]
    return {
        "instrument": "round0248-arm-waiver-inventory-v1",
        "arms_waivable_by_a_declaration": len(rows),
        "rows": rows,
        "distinct_declarations": sorted({str(row["waived_by"]) for row in rows}),
        "defects": defects,
        "holds": not defects,
        "registry_fingerprint": registry_fingerprint(),
        "note": (
            "review-0247-01 A.6: `replay` waived two arms of require() and was "
            "not in the inventory, which is the shape R0247 retired "
            "`training_performed` for. Every declaration that waives an arm "
            "must name a registered parameter"
        ),
    }


def derive_inventory(*, repo_root: str) -> dict[str, Any]:
    constants = derive_gate_constant_inventory(repo_root=repo_root)
    waivers = derive_arm_waiver_inventory(repo_root=repo_root)
    return {
        "instrument": "round0248-mechanical-inventory-v1",
        "gate_constants": constants,
        "arm_waivers": waivers,
        "holds": bool(constants["holds"] and waivers["holds"]),
        "what_it_reads": (
            "the source of every module in GUARDED_MODULES, parsed with ast. "
            "It is derived on every run, so a bound added tomorrow and "
            "compared against its own module global fails the contract test "
            "rather than waiting for a review"
        ),
        "what_it_cannot_see": (
            "a bound reached by getattr, by a dict lookup on a computed key, "
            "or through a C extension; and a node that rewrites the registry "
            "at runtime. It catches MISTAKES, not a node that cheats"
        ),
    }


def require_inventory_complete(*, repo_root: str) -> dict[str, Any]:
    """Fail closed if any gate comparison or arm waiver is unregistered."""
    inventory = derive_inventory(repo_root=repo_root)
    if not inventory["holds"]:
        raise Round0247Error(
            "R0248 STOP: the mechanically derived inventory found gate "
            "decisions that do not read the registry: "
            f"constants {inventory['gate_constants']['defects']}, "
            f"arm waivers {inventory['arm_waivers']['defects']}. A bound that "
            "is registered and then compared against its module-level mirror "
            "is not a bound; two assignments defeated exactly this in "
            "review-0247-01 A.3."
        )
    return inventory


__all__ = [
    "GUARDED_MODULES",
    "NOT_A_BOUND",
    "REGISTRY_READERS",
    "ROUND_ID",
    "derive_arm_waiver_inventory",
    "derive_gate_constant_inventory",
    "derive_inventory",
    "require_inventory_complete",
]

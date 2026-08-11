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

#: R0249, review-0248-01 §D.3. **`GUARDED_MODULES` used to be a hand-written
#: tuple of thirteen paths, and that is how the seventh
#: `bare_registered_symbol` survived: `experiments/round0242_nodes.py:246`
#: compared a bare `WATCHDOG_ANON_BYTES` against a second literal copy of a
#: registered number, one file outside the list.** A hand-written module list
#: is the same artifact the inventory exists to eliminate, so it is gone.
#:
#: Two scopes are now DISCOVERED from the tree, and they answer different
#: questions:
#:
#: * `discover_round_modules()` — **every** `round0*.py` under `basemap/` and
#:   `experiments/` in the release. Over this scope the derivation enforces the
#:   one thing it can decide with no human input at all: a comparison whose
#:   operand is a **registered** symbol read as a bare name is a defect. The
#:   registry supplies the symbol set, so this needs no list of any kind.
#: * `discover_registry_regime_modules()` — the modules whose source names
#:   `round0247_registry`, i.e. those that have opted into the registry regime,
#:   against the thirteen that were hand-written before. Over this narrower
#:   scope the derivation additionally requires every constant a comparison
#:   reads to be triaged, which is the part that does need a human sentence per
#:   symbol (`NOT_A_BOUND`).
#:
#: Both counts are MEASURED at each derivation and published as
#: `discovery.round_modules_scanned` and `discovery.registry_regime_modules`;
#: no count is written here, because a count written here is the same kind of
#: hand-maintained claim the list was.
#:
#: The rule is mechanical in both directions: a new guard module that imports
#: the registry joins the triage scope automatically, and a module that does
#: not import the registry is still scanned for bare registered symbols.
ROUND_MODULE_DIRECTORIES: tuple[str, ...] = ("basemap", "experiments")
ROUND_MODULE_PREFIX = "round0"
#: The import whose presence in a module's source means "this module has opted
#: into the safety-parameter registry", and therefore into full triage.
REGISTRY_REGIME_MARKER = "round0247_registry"

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
    # -- R0249: the symbols the DISCOVERED scope brought in that the
    #    hand-written module list had waived by omission.
    "REGISTRY_READERS": (
        "the allowlist of sanctioned registry-reader function names, compared "
        "as `inner.func.id in REGISTRY_READERS` inside this module's own ast "
        "walk. It classifies source, it bounds nothing, and widening it makes "
        "the inventory report MORE comparisons as sanctioned - which is a "
        "source change in the diff, exactly like REGISTERED_ABORT_READERS"
    ),
    "REGISTRY_REGIME_MARKER": (
        "the substring whose presence in a module's source puts that module in "
        "the TRIAGE scope. It is a discovery rule, not a bound: narrowing it "
        "removes modules from triage, which shows as a source change in the "
        "diff and cannot make a bare registered symbol pass - the wide scope "
        "that catches those does not consult it at all"
    ),
    "LOCALITY_ACTION": "an action name, compared for handler dispatch",
    "FUZZY_ACTION": "an action name, compared for handler dispatch",
    "GUARDFIX_ACTION": "an action name, compared for handler dispatch",
    "KNOWN_EXTERNAL_MEMORY_MODES": (
        "the set of external-memory modes that are IMPLEMENTED, compared as "
        "`mode not in KNOWN_EXTERNAL_MEMORY_MODES` to refuse a mode nobody has "
        "written. Removing an entry makes MORE modes refuse, never fewer; it "
        "bounds no resource"
    ),
    "CONTROL_MAX_THROTTLED_RATE_RATIO": (
        "an acceptance threshold on R0249's OWN positive control - how far the "
        "measured allocation rate must collapse for the throttle arm to hold. "
        "Loosening it weakens the round's evidence for its own claim, which is "
        "self-punishing. Same class as SAMPLER_MIN_CHI_SQUARE_P"
    ),
    "DEFAULT_EXTERNAL_MEMORY_MODE": (
        "the mode a caller gets when it names none. It is a mode NAME, not a "
        "bound, and it is not the enforcement: require_external_memory_mode() "
        "refuses an unplaceable mode and never downgrades, and every R0249 "
        "receipt carries cgroup_self_report(), which reads back the limit the "
        "KERNEL applied. So weakening the default shows up in the artifact as "
        "a changed cgroup and a changed mode field, not only in the source"
    ),
    "LOCALITY_SCHEMA": (
        "a schema-version identity, compared to refuse an artifact written "
        "under a different schema. Moving it makes the load FAIL, never pass"
    ),
    "ENFORCEMENT_REFUSED": (
        "an enforcement-class TAG ('refused'/'clamped'/'declared') compared "
        "inside the registry to decide which set a weakening record belongs "
        "to. Same class as CEILING and FLOOR: a string, not a bound"
    ),
    "ENFORCEMENT_CLAMPED": "the second enforcement tag; same as ENFORCEMENT_REFUSED",
    "ENFORCEMENT_DECLARED": "the third enforcement tag; same as ENFORCEMENT_REFUSED",
    "NODE_HEADROOM_BYTES": (
        "a DERIVED quantity: WATCHDOG_ANON_BYTES - NODE_ANON_BUDGET_BYTES, "
        "both of which are themselves triaged or registered. It is the node's "
        "own arithmetic on two upstream numbers, published in the receipt, "
        "and the gate's binding headroom is the registered "
        "max_declared_headroom_bytes rather than this"
    ),
}


def discover_round_modules(*, repo_root: str) -> tuple[str, ...]:
    """Every `round0*.py` under `basemap/` and `experiments/`, sorted.

    Discovery, not a list. This is the scope over which a bare comparison
    against a REGISTERED symbol is a defect — a judgement the registry makes
    on its own, with no hand-written adjudication anywhere in it.
    """
    found: list[str] = []
    for directory in ROUND_MODULE_DIRECTORIES:
        absolute = os.path.join(repo_root, directory)
        if not os.path.isdir(absolute):
            continue
        for name in os.listdir(absolute):
            if not name.startswith(ROUND_MODULE_PREFIX):
                continue
            if not name.endswith(".py"):
                continue
            found.append(f"{directory}/{name}")
    return tuple(sorted(found))


def discover_registry_regime_modules(*, repo_root: str) -> tuple[str, ...]:
    """The discovered round modules whose source names the registry.

    A module that imports `basemap.round0247_registry` has opted into the
    registry regime, so every module-level constant its comparisons read is
    expected to be either a registry read or triaged. Adding a new guard
    module puts it in this scope on the next derivation, without an edit here.
    """
    regime: list[str] = []
    for relative in discover_round_modules(repo_root=repo_root):
        source = _module_source(repo_root, relative)
        if source is not None and REGISTRY_REGIME_MARKER in source:
            regime.append(relative)
    return tuple(regime)


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


def _scan_modules(
    *, repo_root: str, modules: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify every comparison against a module-level constant in `modules`."""
    registered = _registered_symbols()
    triaged = _triaged_symbols()
    rows: list[dict[str, Any]] = []
    modules_read: list[str] = []
    for relative in modules:
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
    return rows, modules_read


def derive_gate_constant_inventory(*, repo_root: str) -> dict[str, Any]:
    """Every module-level constant a round module compares against.

    Two scopes, two different obligations — see `ROUND_MODULE_DIRECTORIES`.
    The wide scope is where `bare_registered_symbol` is enforced and needs no
    hand-written list at all; the registry-regime scope is where `untriaged`
    is additionally enforced.
    """
    verify_registry(label="R0248 gate-constant inventory")
    all_modules = discover_round_modules(repo_root=repo_root)
    regime_modules = discover_registry_regime_modules(repo_root=repo_root)
    all_rows, all_read = _scan_modules(repo_root=repo_root, modules=all_modules)
    regime = set(regime_modules)
    rows = [row for row in all_rows if row["module"] in regime]
    #: The wide-scope defect: a REGISTERED symbol compared as a bare name,
    #: anywhere in the release. This is the class that produced
    #: `round0242_nodes:246`, and deciding it needs no human sentence.
    bare_registered_anywhere = [
        row for row in all_rows if row["status"] == "bare_registered_symbol"
    ]
    #: The regime-scope defect: a constant nobody has triaged, in a module that
    #: opted into the registry.
    untriaged_in_the_regime = [
        row for row in rows if row["status"] == "untriaged"
    ]
    defects = bare_registered_anywhere + untriaged_in_the_regime
    return {
        "instrument": "round0248-gate-constant-inventory-v1",
        "modules_read": [row for row in all_read if row in regime],
        "modules_declared": list(regime_modules),
        "discovery": {
            "how_the_scope_is_found": (
                "every round0*.py under basemap/ and experiments/ is scanned "
                "for bare registered symbols; the subset whose source names "
                f"{REGISTRY_REGIME_MARKER!r} is additionally required to have "
                "every compared constant triaged. Neither scope is written by "
                "hand - review-0248-01 §D.3 found the seventh defect one file "
                "outside the hand-written list"
            ),
            "round_modules_scanned": len(all_read),
            "registry_regime_modules": len(regime_modules),
            "comparisons_over_the_whole_scope": len(all_rows),
        },
        "comparisons_against_module_level_constants": len(rows),
        "rows": rows,
        "distinct_symbols": sorted({str(row["symbol"]) for row in rows}),
        "bare_registered_symbols_anywhere_in_the_release": (
            bare_registered_anywhere
        ),
        "untriaged_in_the_registry_regime": untriaged_in_the_regime,
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
    for relative in discover_registry_regime_modules(repo_root=repo_root):
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
    "NOT_A_BOUND",
    "REGISTRY_READERS",
    "REGISTRY_REGIME_MARKER",
    "ROUND_ID",
    "ROUND_MODULE_DIRECTORIES",
    "ROUND_MODULE_PREFIX",
    "discover_registry_regime_modules",
    "discover_round_modules",
    "derive_arm_waiver_inventory",
    "derive_gate_constant_inventory",
    "derive_inventory",
    "require_inventory_complete",
]

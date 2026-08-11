"""R0255's two guards: the treatment closure, and the purity of the gate's family.

Both exist because of a named prior failure, and both ship a positive control that
plants the defect against the **shipped** predicate -- review-0253-01 caught a
control that re-implemented the guard it was testing, and R0254 set the standard by
showing five planted defects refused by the shipped predicate and accepted by the
predicate it replaces. These follow that shape exactly, and each names the *old*
predicate it replaces so the controls have something to be accepted by.

**Guard 1 -- the treatment closure.** The owner ruling requires the thirteen new
cells to be new *cells*, not a new instrument: "any treatment or panel drift makes
the pooled n=29 family a mixed population". The config side of that is already
proved three ways by `round0255_seed_extension_n29` (seed-invariant digest, masked
bytes, byte-for-byte reconstruction of R0217's template). What none of those sees is
the **release**: R0250's cells, R0251's cells and this round's cells run different
checkouts, and R0251 landed 33 insertions inside `basemap/pumap/parametric_umap/`
between R0250's family and this one. So the round seals, at prepare time, the
SHA-256 of every source file in the training import closure at R0217's release
commit and at this one, and every training node re-hashes the files it actually
imported and refuses unless they equal the release side of that seal. The delta
against R0217 is published as evidence rather than gated -- a changed file is a
finding for a reviewer to weigh, and the round's positive claim about the treatment
rests on the seed-42 replay control, not on the diff being empty.

The **old predicate** this replaces is R0250's, still shipped in
`experiments/round0250_nodes.py`: `receipt["seed_invariant_sha256"] ==
R0217_SEED_INVARIANT_SHA256`. It is a statement about the config and cannot see a
source change at all, which is precisely why all five planted defects below pass it.

**Guard 2 -- the family is 2M-only.** `plan-minilm-100m-v2.md`: *a gate's family
never grows to include the maps it judges*. That is the v0 wrong pass -- R0195/R0204
qualified on R0161's own fitting cells, and `max|z| <= (n-1)/sqrt(n)` guaranteed the
result before any map existed. The **old predicate** is the v0 construction: take
whatever cells the panel pooled and call them the family. It accepts a rung map, a
held-out cell, the replay control, a duplicate and an empty family; the shipped one
refuses all five.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .round0250_gate_n16 import CANDIDATE_CLUSTER_COUNTS, CANDIDATE_SEEDS, CUVS_FAMILY_SEEDS
from .round0255_seed_extension_n29 import (
    POOLED_SEEDS,
    REPLAY_CONTROL_CAPABILITY,
    REPLAY_CONTROL_SEED,
    R0217_SEED_INVARIANT_SHA256,
)


ROUND_ID = "0255"

CLOSURE_SCHEMA = "round0255-treatment-import-closure-v1"

#: R0217's release commit -- the checkout the first four cells of this family were
#: trained from. The diff is taken against this and nothing else.
R0217_RELEASE_SHA = "8ae96e16a0d328a8c816958a866f436c426ccfaa"

#: The modules a training node imports that can change what the trainer computes.
#: The trainer itself, the pipeline that feeds it, the config that describes it, and
#: the contract modules that derive the horizon and the dose. Named here so the seal
#: has a fixed membership a reviewer can audit; the runtime check then proves the
#: files behind these names are the sealed ones.
TRAIN_CLOSURE_MODULES: tuple[str, ...] = (
    "basemap.pumap.parametric_umap.core",
    "basemap.pumap.parametric_umap.model",
    "basemap.round0113_prompt_contrast",
    "basemap.round0217_minilm_2m_pipeline",
    "basemap.round0217_minilm_2m_seed_family",
    "basemap.round0221_minilm_2m_seed_extension",
    "basemap.round0230_minilm_2m_seed_extension_n13",
)

#: The predicate this guard replaces, quoted so the control has a target.
OLD_TREATMENT_PREDICATE = (
    "R0250's: receipt['seed_invariant_sha256'] == R0217_SEED_INVARIANT_SHA256. A "
    "statement about the CONFIG. It cannot observe a source change, so every "
    "source-level defect planted below passes it."
)

OLD_FAMILY_PREDICATE = (
    "the v0 construction: the gate's family is whatever cells the panel pooled. It "
    "accepts a rung map, a held-out cell, a control cell, a duplicate and an empty "
    "family -- which is how R0195/R0204 qualified on R0161's own fitting cells."
)


class Round0255TreatmentError(RuntimeError):
    """The registered R0255 treatment-closure contract changed."""


class Round0255FamilyError(RuntimeError):
    """The gate's family is not the registered 2M-only universe."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_closure_hashes(
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, dict[str, Any]]:
    """Hash the files behind the modules this process actually imported.

    Imports them if they are not already imported, then reads `__file__` -- so the
    hash is of the bytes Python loaded, not of a path this module guessed.
    """
    import importlib

    observed: dict[str, dict[str, Any]] = {}
    for name in modules:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", None)
        if not path or not os.path.exists(path):
            raise Round0255TreatmentError(
                f"R0255 closure module {name!r} has no readable source file"
            )
        observed[name] = {
            "path": os.path.realpath(path),
            "bytes": os.path.getsize(path),
            "sha256": file_sha256(path),
        }
    return observed


def assert_runtime_closure_matches_seal(
    *,
    sealed: Mapping[str, Any],
    observed: Mapping[str, Mapping[str, Any]],
    modules: Sequence[str] = TRAIN_CLOSURE_MODULES,
) -> dict[str, Any]:
    """Refuse unless every module in the registered closure ran the sealed bytes.

    `sealed` is the prepare-time seal: `{"files": {module: {"sha256_at_release":
    ..., "sha256_at_r0217": ..., ...}}}`. Refuses on an empty or partial closure,
    a membership difference in either direction, a malformed digest, and any
    content difference. There is no allow-list and no tolerance: this predicate has
    exactly one way to hold.
    """
    registered = tuple(modules)
    if not registered:
        raise Round0255TreatmentError(
            "R0255 treatment closure is empty: a closure with no members accepts "
            "every possible release and is the vacuous case this guard exists for"
        )
    files = dict(sealed.get("files") or {})
    if not files:
        raise Round0255TreatmentError("R0255 sealed treatment closure carries no files")
    if set(files) != set(registered):
        raise Round0255TreatmentError(
            f"R0255 sealed closure membership {sorted(files)} is not the registered "
            f"closure {sorted(registered)}"
        )
    if set(observed) != set(registered):
        raise Round0255TreatmentError(
            f"R0255 runtime closure membership {sorted(observed)} is not the "
            f"registered closure {sorted(registered)}"
        )
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for name in registered:
        want = str(dict(files[name]).get("sha256_at_release") or "")
        got = str(dict(observed[name]).get("sha256") or "")
        if not _SHA256.match(want) or not _SHA256.match(got):
            problems.append(f"{name}: malformed digest")
            rows.append({"module": name, "matches": False, "reason": "malformed"})
            continue
        matches = want == got
        if not matches:
            problems.append(f"{name}: {got} != sealed {want}")
        rows.append({
            "module": name,
            "sealed_sha256_at_release": want,
            "runtime_sha256": got,
            "matches": matches,
            "changed_since_r0217": (
                str(dict(files[name]).get("sha256_at_r0217") or "") != want
            ),
        })
    if problems:
        raise Round0255TreatmentError(
            "R0255 the training closure that ran is not the sealed release: "
            + "; ".join(problems)
        )
    return {
        "schema": CLOSURE_SCHEMA,
        "modules": list(registered),
        "modules_checked": len(rows),
        "every_module_ran_the_sealed_bytes": all(row["matches"] for row in rows),
        "modules_changed_since_r0217": [
            row["module"] for row in rows if row.get("changed_since_r0217")
        ],
        "rows": rows,
        "r0217_release_sha": R0217_RELEASE_SHA,
        "old_predicate_this_replaces": OLD_TREATMENT_PREDICATE,
        "what_it_does_not_prove": (
            "that a changed file changed no number. It proves the bytes that ran "
            "are the bytes that were sealed and names which of them differ from "
            "R0217's release; the positive claim that the treatment is unchanged "
            "rests on the seed-42 replay control, which retrains R0217's own cell "
            "on this release and compares the published checkpoint."
        ),
    }


def old_treatment_predicate(
    *, seed_invariant_sha256: str, **_ignored: Any
) -> bool:
    """R0250's check, reproduced so the controls have something to be accepted by."""
    return str(seed_invariant_sha256) == R0217_SEED_INVARIANT_SHA256


def treatment_closure_controls(
    *, sealed: Mapping[str, Any], observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Plant five defects against the SHIPPED predicate; check the old one too.

    Nothing here re-implements `assert_runtime_closure_matches_seal`: every control
    calls it. If it returned unconditionally, all five would fail.
    """
    honest_observed = {name: dict(value) for name, value in observed.items()}
    controls: list[dict[str, Any]] = []

    def _run(name: str, description: str, **kwargs: Any) -> None:
        refused = False
        error = None
        try:
            assert_runtime_closure_matches_seal(**kwargs)
        except Round0255TreatmentError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "shipped_predicate_refused": refused,
            "shipped_predicate_error": error,
            "old_predicate_accepted": old_treatment_predicate(
                seed_invariant_sha256=R0217_SEED_INVARIANT_SHA256
            ),
        })

    mutated = {name: dict(value) for name, value in honest_observed.items()}
    victim = TRAIN_CLOSURE_MODULES[0]
    mutated[victim]["sha256"] = "0" * 64
    _run(
        "content_drift",
        "one closure module ran bytes that are not the sealed release bytes",
        sealed=sealed,
        observed=mutated,
    )

    dropped = {
        name: dict(value)
        for name, value in honest_observed.items()
        if name != victim
    }
    _run(
        "missing_module",
        "a registered closure module is absent from the runtime map",
        sealed=sealed,
        observed=dropped,
    )

    extra = {name: dict(value) for name, value in honest_observed.items()}
    extra["basemap.round0255_treatment"] = {
        "path": __file__,
        "bytes": 0,
        "sha256": "1" * 64,
    }
    _run(
        "extra_module",
        "the runtime map carries a module the seal does not",
        sealed=sealed,
        observed=extra,
    )

    malformed = {name: dict(value) for name, value in honest_observed.items()}
    malformed[victim]["sha256"] = "not-a-digest"
    _run(
        "malformed_digest",
        "a digest that is not a SHA-256 hex string",
        sealed=sealed,
        observed=malformed,
    )

    _run(
        "empty_closure",
        "an EMPTY registered closure -- the vacuous accept, which is how a guard "
        "with no members certifies every release",
        sealed={"files": {}},
        observed={},
        modules=(),
    )

    honest_refused = False
    honest_error = None
    try:
        assert_runtime_closure_matches_seal(sealed=sealed, observed=honest_observed)
    except Round0255TreatmentError as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"
    return {
        "controls": controls,
        "planted": len(controls),
        "every_planted_defect_was_refused": all(
            item["shipped_predicate_refused"] for item in controls
        ),
        "the_old_predicate_accepted_every_one": all(
            item["old_predicate_accepted"] for item in controls
        ),
        "the_honest_closure_still_passes": not honest_refused,
        "honest_error": honest_error,
        "old_predicate": OLD_TREATMENT_PREDICATE,
        "note": (
            "every control calls the SHIPPED assert_runtime_closure_matches_seal; "
            "none re-implements it. review-0253-01 rejected a control that "
            "re-implemented the guard it was testing."
        ),
    }


# --------------------------------------------------------------------------- #
# Guard 2 -- the family is the 2M universe and nothing else
# --------------------------------------------------------------------------- #


def exact_cell_id(seed: int) -> str:
    return f"exact-seed{int(seed)}"


#: Every cell id that exists in this program and is NOT a member of the fitting
#: family: the twelve held-out cells the gate is scored against, and this round's
#: replay control. Named so the guard can say *why* a cell is refused.
HELD_OUT_CELL_IDS: tuple[str, ...] = tuple(
    [f"cuvs-igd48-seed{seed}" for seed in CUVS_FAMILY_SEEDS]
    + [
        f"cluster-spill-c{clusters}-seed{seed}"
        for clusters in CANDIDATE_CLUSTER_COUNTS
        for seed in CANDIDATE_SEEDS
    ]
)

CONTROL_CELL_IDS: tuple[str, ...] = (REPLAY_CONTROL_CAPABILITY,)

FAMILY_RULE = (
    "BINDING (plan-minilm-100m-v2.md; owner ruling 2026-08-11): the Phase 3 gate is "
    "fitted on the 2M universe and judges rung maps from OUTSIDE. The fitting family "
    "is exactly the twenty-nine exact-graph 2M cells, one per registered seed, no "
    "duplicates, no held-out cell, no control cell and no rung map. A gate whose "
    "family grows to include the maps it judges is not a test: that is the v0 wrong "
    "pass, where R0195/R0204 qualified on R0161's own fitting cells and "
    "max|z| <= (n-1)/sqrt(n) guaranteed the result before any map existed."
)


def assert_family_is_2m_only(
    defining_cell_ids: Sequence[str],
    *,
    pooled_seeds: Sequence[int] = POOLED_SEEDS,
) -> dict[str, Any]:
    """Refuse unless the fitting family is exactly the registered 2M cells."""
    expected = [exact_cell_id(seed) for seed in pooled_seeds]
    observed = [str(item) for item in defining_cell_ids]
    if not observed:
        raise Round0255FamilyError(
            "R0255 fitting family is EMPTY: an empty family fits no floor and "
            "cannot be the 2M universe"
        )
    if len(observed) != len(set(observed)):
        duplicates = sorted({item for item in observed if observed.count(item) > 1})
        raise Round0255FamilyError(
            f"R0255 fitting family repeats {duplicates}: a duplicated cell inflates "
            "n without adding evidence"
        )
    foreign = [item for item in observed if item not in set(expected)]
    if foreign:
        why = []
        for item in foreign:
            if item in set(HELD_OUT_CELL_IDS):
                why.append(f"{item} is a HELD-OUT cell this gate judges")
            elif item in set(CONTROL_CELL_IDS):
                why.append(f"{item} is a CONTROL cell, not a family cell")
            else:
                why.append(f"{item} is not a registered 2M exact-graph cell")
        raise Round0255FamilyError(
            "R0255 fitting family contains cells the gate judges: " + "; ".join(why)
        )
    if sorted(observed) != sorted(expected):
        missing = sorted(set(expected) - set(observed))
        raise Round0255FamilyError(
            f"R0255 fitting family is missing registered cells {missing}"
        )
    return {
        "rule": FAMILY_RULE,
        "n": len(expected),
        "defining_cell_ids": list(expected),
        "held_out_cell_ids": list(HELD_OUT_CELL_IDS),
        "control_cell_ids": list(CONTROL_CELL_IDS),
        "family_is_the_2m_universe_only": True,
        "no_judged_map_is_in_the_family": True,
        "old_predicate_this_replaces": OLD_FAMILY_PREDICATE,
    }


def old_family_predicate(defining_cell_ids: Sequence[str], **_ignored: Any) -> bool:
    """The v0 construction: whatever the panel pooled is the family."""
    return True


def family_purity_controls(
    *, pooled_seeds: Sequence[int] = POOLED_SEEDS
) -> dict[str, Any]:
    """Plant five family defects against the SHIPPED guard and against the old one."""
    honest = [exact_cell_id(seed) for seed in pooled_seeds]
    plants: list[tuple[str, str, list[str]]] = [
        (
            "rung_map_in_the_family",
            "a Phase 2 rung map added to the cells the floor is fitted on -- the "
            "exact v0 defect",
            honest + ["ladder-6250k-h2048-seed42"],
        ),
        (
            "held_out_cell_in_the_family",
            "cluster-spill-c8-seed42, a cell this gate is scored against, added to "
            "its own fitting family",
            honest + ["cluster-spill-c8-seed42"],
        ),
        (
            "control_cell_in_the_family",
            "this round's seed-42 replay control added as a twenty-ninth cell",
            honest + [REPLAY_CONTROL_CAPABILITY],
        ),
        (
            "duplicated_cell",
            "one family cell counted twice, inflating n without adding evidence",
            honest + [exact_cell_id(pooled_seeds[0])],
        ),
        (
            "empty_family",
            "an empty fitting family -- the vacuous case",
            [],
        ),
    ]
    controls: list[dict[str, Any]] = []
    for name, description, planted in plants:
        refused = False
        error = None
        try:
            assert_family_is_2m_only(planted, pooled_seeds=pooled_seeds)
        except Round0255FamilyError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "planted_cells": len(planted),
            "shipped_guard_refused": refused,
            "shipped_guard_error": error,
            "old_predicate_accepted": bool(old_family_predicate(planted)),
        })
    honest_refused = False
    honest_error = None
    try:
        assert_family_is_2m_only(honest, pooled_seeds=pooled_seeds)
    except Round0255FamilyError as raised:
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
        "old_predicate": OLD_FAMILY_PREDICATE,
        "note": (
            "every control calls the SHIPPED assert_family_is_2m_only; none "
            "re-implements it."
        ),
    }


__all__ = [
    "CLOSURE_SCHEMA",
    "CONTROL_CELL_IDS",
    "FAMILY_RULE",
    "HELD_OUT_CELL_IDS",
    "OLD_FAMILY_PREDICATE",
    "OLD_TREATMENT_PREDICATE",
    "R0217_RELEASE_SHA",
    "ROUND_ID",
    "Round0255FamilyError",
    "Round0255TreatmentError",
    "TRAIN_CLOSURE_MODULES",
    "assert_family_is_2m_only",
    "assert_runtime_closure_matches_seal",
    "exact_cell_id",
    "family_purity_controls",
    "file_sha256",
    "old_family_predicate",
    "old_treatment_predicate",
    "runtime_closure_hashes",
    "treatment_closure_controls",
]

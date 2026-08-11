"""R0250 — measure the TRAINER's own loops against the registered poll ceiling.

Six rounds (R0244-R0249) have closed with the same sentence: *the trainer's own
loops are unmeasured, so a ~10 h training node cannot run.* review-0246-01 item 3,
review-0247-01 open item 1, review-0248-01 item 8 and result-0249 deviation 4 are
the same finding four times. This module closes it by measuring, not by fixing.

**What is being measured, and why it is the right quantity.** `AbortPollGate`
scores the widest interval between two cooperative-abort reads against
`registered_value("r0246_max_poll_spacing_s")`, which is R0244's sealed
`budget_headroom_bytes / measured slope`: the longest a stage may go unobserved
before an allocating loop could consume the whole headroom between two looks. A
training node holds a live CUDA context, so the abort flag is the *only* safe way
to stop it -- signalling one is what wedged this box twice.

**Two arms, because "the trainer" has two honest readings.**

* `AS_SHIPPED` -- `ParametricUMAP.fit` as it exists in the release. Nothing inside
  it reads the cooperative abort flag: `grep` over `basemap/pumap/` finds no
  reference to `runner_abort_reason`, `_check_runner_abort` or
  `ROUNDRUN_ABORT_FLAG` anywhere. The only reads available to a training node are
  the ones its own handler makes on either side of `fit()`, so the widest
  abort-read gap of a training node **is the whole `fit()` wall**. The arm
  measures that wall on a short rung and reports the ratio.
* `PER_BATCH` -- the same fit with one poll installed at the per-batch boundary,
  to answer the question the fix will need: *if the trainer polled once per batch,
  would it clear the ceiling, and where would the widest gap then sit?* The hook
  is `ParametricUMAP._low_dim_qs`, which `core.py:1631` calls exactly once per
  training batch on the main thread inside the forward pass. It is installed by a
  context manager that restores the original attribute unconditionally, so the
  release module is never edited and no node outside this one is affected.

**The short rung is deliberate and is not a map.** The loop structure, not the
map, is the object of study, so the rung runs `SHORT_HORIZON_UPDATES` updates
against the same sealed R0216 substrate, the same exact k15 graph, the same
sampler, the same precision and the same batch size as R0217's treatment. Its
config carries this round's id and a distinct capability, its horizon is not the
registered `80,163`, and `is_a_family_cell` is `False` in every receipt: it must
never be mistaken for a cell of the seed family.

**The extrapolation is a projection and is labelled as one.** `project_to_hours`
carries the AS_SHIPPED gap to a longer node by the only mechanism that arm has --
the gap *is* the wall, so it scales with it -- and carries the PER_BATCH gap
unscaled, because a per-batch gap is a property of one batch and does not grow
with the number of batches. Both are stamped `kind: "projection"` with the
measured basis beside them. Neither is a measurement at 10 h and neither is
offered as one.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0217_minilm_2m_seed_family import (
    BATCH_SIZE,
    DIMENSION,
    GRAPH_K,
    OUTPUT_DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    ROWS,
    WARMUP_SUCCESSFUL_UPDATES,
    performance_windows,
)
from .round0247_registry import registered_value


ROUND_ID = "0250"

TRAINER_LOOP_CAPABILITY = "round0250-trainer-loop-abort-poll-measurement-v1"
TRAINER_LOOP_SCHEMA = "round0250-trainer-loop-abort-poll-measurement-v1"

#: The short rung. Large enough that the steady-state loop dominates the arm's
#: own wall and the canary profiler still partitions a post-warmup window, small
#: enough that the measurement costs minutes rather than the 706 s R0230 paid per
#: cell. It is NOT the registered horizon and must never be read as one.
SHORT_HORIZON_UPDATES = 10_000

#: `plan-minilm-100m-v2.md`: "A 100M train is ~10 h". The projection target.
PROJECTION_TARGET_HOURS = 10.0
PROJECTION_TARGET_SECONDS = PROJECTION_TARGET_HOURS * 3600.0

ARM_AS_SHIPPED = "as_shipped"
ARM_PER_BATCH = "per_batch_poll"
ARMS: tuple[str, ...] = (ARM_AS_SHIPPED, ARM_PER_BATCH)

#: The one call `core.py` makes exactly once per training batch, on the main
#: thread, inside the forward pass. Recorded as data so a reviewer can check the
#: hook against the source rather than take the docstring's word for it.
PER_BATCH_HOOK = {
    "target": "basemap.pumap.parametric_umap.ParametricUMAP._low_dim_qs",
    "call_site": "basemap/pumap/parametric_umap/core.py:1631",
    "calls_per_training_batch": 1,
    "thread": "the main training thread",
    "installation": (
        "a context manager sets the class attribute and restores the original in "
        "a finally, so the release module is not edited and nothing outside the "
        "with-block is affected"
    ),
    "why_not_the_sampler": (
        "the R0217 pipeline takes core.py's `_fast_device_path` branch, so the "
        "batch arrives through `next(iter(loader))` on a sampler whose iteration "
        "protocol is an implementation detail; and the host prefetch producer "
        "runs on its own thread, where an abort read would unwind the wrong "
        "stack. `_low_dim_qs` is on the consumer's stack by construction."
    ),
}

NOT_A_FAMILY_CELL = (
    "This rung exists to measure loop structure. Its horizon is "
    f"{SHORT_HORIZON_UPDATES} updates, not the registered 80,163; its capability "
    f"is {TRAINER_LOOP_CAPABILITY}; its config carries round_id 0250. It is not a "
    "cell of the exact-graph seed family, no floor may be fitted to it, and no "
    "map-quality statement follows from it."
)

TRAINER_LOOP_NOTE = (
    "The registered ceiling is read at the comparison site with "
    "registered_value('r0246_max_poll_spacing_s'); it is R0244's sealed "
    "budget headroom over R0244's sealed measured allocation slope, i.e. the "
    "longest a stage may go unobserved before an allocating loop could consume "
    "the whole headroom between two abort reads."
)


class Round0250TrainerLoopError(RuntimeError):
    """The registered R0250 trainer-loop measurement contract changed."""


# --------------------------------------------------------------------------- #
# the short rung's config
# --------------------------------------------------------------------------- #

#: Exactly the fields the short rung moves away from R0217's canonical config.
#: Anything not on this list must be byte-identical to the treatment, and
#: `short_rung_config` proves it by diffing the canonical JSON.
SHORT_RUNG_CHANGED_PATHS: tuple[tuple[str, ...], ...] = (
    ("schema",),
    ("round_id",),
    ("capability",),
    ("optimizer", "successful_positive_lr_updates"),
    ("execution", "performance_windows"),
    ("execution", "scale_change"),
    ("seed_family",),
    ("dose_registration",),
)

SHORT_RUNG_CONFIG_SCHEMA = "round0250-trainer-loop-short-rung-config-v1"


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    cursor: Any = value
    for key in path[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise Round0250TrainerLoopError(
                f"R0250 short-rung config is missing {'.'.join(path)}"
            )
        cursor = cursor[key]
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        raise Round0250TrainerLoopError(
            f"R0250 short-rung config is missing {'.'.join(path)}"
        )
    cursor[path[-1]] = replacement


def _differing_paths(
    left: Mapping[str, Any], right: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    changed: list[tuple[str, ...]] = []
    for key in sorted(set(left) | set(right)):
        here = prefix + (key,)
        a = left.get(key)
        b = right.get(key)
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            changed.extend(_differing_paths(a, b, here))
        elif a != b:
            changed.append(here)
    return changed


def short_rung_config(
    template: Mapping[str, Any], *, updates: int = SHORT_HORIZON_UPDATES
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """R0217's canonical config at a short horizon, with every change enumerated.

    Fails closed if the diff against the template reaches any path outside
    `SHORT_RUNG_CHANGED_PATHS`, so the rung cannot quietly become a different
    treatment while claiming to exercise the same loop.
    """
    updates = int(updates)
    if updates < WARMUP_SUCCESSFUL_UPDATES * 2:
        raise Round0250TrainerLoopError(
            f"R0250 short rung needs more than twice the {WARMUP_SUCCESSFUL_UPDATES} "
            f"warmup updates to reach steady state, got {updates}"
        )
    config = copy.deepcopy(dict(template))
    _set_path(config, ("schema",), SHORT_RUNG_CONFIG_SCHEMA)
    _set_path(config, ("round_id",), ROUND_ID)
    _set_path(config, ("capability",), TRAINER_LOOP_CAPABILITY)
    _set_path(config, ("optimizer", "successful_positive_lr_updates"), updates)
    _set_path(config, ("execution", "performance_windows"), performance_windows(updates))
    _set_path(
        config,
        ("execution", "scale_change"),
        (
            "R0250 trainer-loop instrumentation rung: R0217's recipe, graph, "
            "precision, sampler, optimizer and residency at a short horizon of "
            f"{updates} updates. Loop structure is the object of study; no map is "
            "published and no floor may be fitted to it."
        ),
    )
    config["seed_family"] = {
        "is_a_family_cell": False,
        "why_not": NOT_A_FAMILY_CELL,
    }
    config["dose_registration"] = {
        "dose_registered": False,
        "why_not": (
            "the registered R0184/R0202 low-dose rule derives 80,163 updates from "
            "the sealed edge count. This rung deliberately runs a shorter horizon, "
            "so it carries NO dose registration rather than a weakened one."
        ),
    }
    changed = _differing_paths(dict(template), config)
    allowed = {tuple(path) for path in SHORT_RUNG_CHANGED_PATHS}
    escaped = [
        path for path in changed if not any(path[: len(a)] == a for a in allowed)
    ]
    if escaped:
        raise Round0250TrainerLoopError(
            "R0250 short rung differs from R0217's treatment outside the declared "
            f"paths: {['.'.join(path) for path in escaped]}"
        )
    identity = {
        "template_sha256": sha256_bytes(canonical_json(dict(template))),
        "short_rung_sha256": sha256_bytes(canonical_json(config)),
        "changed_paths": sorted(".".join(path) for path in changed),
        "declared_changed_paths": sorted(".".join(path) for path in allowed),
        "changes_are_inside_the_declared_set": True,
        "short_horizon_updates": updates,
        "registered_treatment_horizon_not_used": True,
        "batch_size": BATCH_SIZE,
        "positive_rows_per_update": POSITIVE_ROWS_PER_UPDATE,
        "rows": ROWS,
        "dimension": DIMENSION,
        "graph_k": GRAPH_K,
        "output_dimension": OUTPUT_DIMENSION,
        "is_a_family_cell": False,
        "not_a_family_cell_note": NOT_A_FAMILY_CELL,
    }
    return config, identity["short_rung_sha256"], identity


# --------------------------------------------------------------------------- #
# the per-batch poll, installed at runtime and always removed
# --------------------------------------------------------------------------- #


class PerBatchAbortPoll:
    """Install one abort read per training batch, then put the class back.

    `poll` is called with a stable `where` label before every batch's low-dim
    kernel evaluation. The original attribute is restored in `__exit__` whether
    or not the body raised, and `restored` is published in the receipt so a
    reviewer can see the release was left as it was found.
    """

    def __init__(self, *, poll: Any, label: str = "R0250 trainer batch") -> None:
        if not callable(poll):
            raise Round0250TrainerLoopError("R0250 per-batch poll must be callable")
        self._poll = poll
        self._label = str(label)
        self._original: Any = None
        self._owner: Any = None
        self.installed = False
        self.restored = False
        self.calls = 0

    def __enter__(self) -> "PerBatchAbortPoll":
        from .pumap.parametric_umap import ParametricUMAP

        original = ParametricUMAP._low_dim_qs
        poll = self._poll
        label = self._label
        counter = {"n": 0}

        def _instrumented(inner_self: Any, src: Any, dst: Any):
            counter["n"] += 1
            poll(f"{label} {counter['n']}")
            return original(inner_self, src, dst)

        self._original = original
        self._owner = ParametricUMAP
        self._counter = counter
        ParametricUMAP._low_dim_qs = _instrumented
        self.installed = True
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._owner is not None and self._original is not None:
            self._owner._low_dim_qs = self._original
            self.restored = True
        self.calls = int(getattr(self, "_counter", {"n": 0})["n"])

    def receipt(self) -> dict[str, Any]:
        return {
            "hook": dict(PER_BATCH_HOOK),
            "installed": bool(self.installed),
            "restored_the_release_attribute": bool(self.restored),
            "per_batch_polls_installed": int(self.calls),
        }


# --------------------------------------------------------------------------- #
# the ceiling, and the projection
# --------------------------------------------------------------------------- #


def ceiling_report(
    *, arm: str, widest_gap_s: float, stage_wall_s: float, polls: int
) -> dict[str, Any]:
    """The measured widest gap against the registered ceiling, read at the site."""
    if arm not in ARMS:
        raise Round0250TrainerLoopError(f"R0250 has no trainer-loop arm {arm!r}")
    gap = float(widest_gap_s)
    wall = float(stage_wall_s)
    if not math.isfinite(gap) or gap < 0.0 or not math.isfinite(wall) or wall <= 0.0:
        raise Round0250TrainerLoopError(
            f"R0250 arm {arm} produced an unusable gap/wall pair: {gap!r} / {wall!r}"
        )
    ceiling = registered_value("r0246_max_poll_spacing_s")
    return {
        "arm": arm,
        "widest_abort_read_gap_s": gap,
        "stage_wall_s": wall,
        "abort_reads": int(polls),
        "registered_ceiling_s_at_the_comparison_site": ceiling,
        "gap_over_the_registered_ceiling": gap / ceiling,
        "meets_the_registered_ceiling": bool(gap <= ceiling),
        "gap_as_a_fraction_of_the_stage_wall": gap / wall,
        "note": TRAINER_LOOP_NOTE,
    }


def project_to_hours(
    *,
    arm: str,
    widest_gap_s: float,
    stage_wall_s: float,
    target_seconds: float = PROJECTION_TARGET_SECONDS,
) -> dict[str, Any]:
    """Carry a measured arm to a longer node. **A projection, never a measurement.**

    `as_shipped` has no abort read inside `fit()`, so its gap *is* the wall and
    scales with it: the projected gap is the target wall itself. `per_batch_poll`
    measures a property of one batch, which does not grow with the number of
    batches, so its projected gap is the measured one carried unscaled -- with the
    caveat that a longer run has more batches and therefore more chances to draw a
    worse tail, which the receipt states rather than models.
    """
    if arm not in ARMS:
        raise Round0250TrainerLoopError(f"R0250 has no trainer-loop arm {arm!r}")
    gap = float(widest_gap_s)
    wall = float(stage_wall_s)
    target = float(target_seconds)
    ceiling = registered_value("r0246_max_poll_spacing_s")
    if arm == ARM_AS_SHIPPED:
        projected = target
        basis = (
            "the trainer makes no abort read inside fit(), so the widest gap IS "
            "the fit wall; at a target wall of "
            f"{target} s the projected widest gap is {target} s. The measured "
            f"short-rung gap {gap} s over its own wall {wall} s is "
            f"{gap / wall} of it, which is the identity this projection rests on."
        )
        scaling = "the gap equals the wall; it scales one-for-one"
    else:
        projected = gap
        basis = (
            "a per-batch gap is a property of one batch, so it does not grow with "
            f"the number of batches; the measured {gap} s is carried unscaled. A "
            f"{target} s node runs about {target / wall} times as many batches as "
            "this rung, so it draws that many more samples from the same tail and "
            "the true maximum is expected to be somewhat larger than the measured "
            "one. This projection does not model that tail."
        )
        scaling = "flat in the number of batches; tail growth not modelled"
    return {
        "kind": "projection",
        "arm": arm,
        "measured_widest_gap_s": gap,
        "measured_stage_wall_s": wall,
        "target_wall_s": target,
        "target_wall_hours": target / 3600.0,
        "projected_widest_gap_s": projected,
        "projected_gap_over_the_registered_ceiling": projected / ceiling,
        "projected_meets_the_registered_ceiling": bool(projected <= ceiling),
        "registered_ceiling_s_at_the_comparison_site": ceiling,
        "scaling_law": scaling,
        "basis": basis,
        "is_a_measurement_at_the_target_wall": False,
    }


__all__ = [
    "ARMS",
    "ARM_AS_SHIPPED",
    "ARM_PER_BATCH",
    "BATCH_SIZE",
    "NOT_A_FAMILY_CELL",
    "PER_BATCH_HOOK",
    "PROJECTION_TARGET_HOURS",
    "PROJECTION_TARGET_SECONDS",
    "PerBatchAbortPoll",
    "ROUND_ID",
    "Round0250TrainerLoopError",
    "SHORT_HORIZON_UPDATES",
    "SHORT_RUNG_CHANGED_PATHS",
    "SHORT_RUNG_CONFIG_SCHEMA",
    "TRAINER_LOOP_CAPABILITY",
    "TRAINER_LOOP_NOTE",
    "TRAINER_LOOP_SCHEMA",
    "WARMUP_SUCCESSFUL_UPDATES",
    "ceiling_report",
    "project_to_hours",
    "short_rung_config",
]

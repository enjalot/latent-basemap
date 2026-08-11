"""R0252 — make a node stoppable, at the two calls that actually bind.

`roundreport`'s generated abort-poll gap census exists because R0250 and R0251
each led with a smaller gap while a larger one sat sealed in another artifact.
Run over R0251's queue it ranks every gap in every node and puts the round's
binding interval at

    1.980511222005589 s = 0.7887487648242341x the registered
    2.5109531834854018 s ceiling, at
    enforcement_poll_spacing.max_gap_between_enforcement_polls_s in
    seed42-map-side-rescore.json

i.e. **one `score_panel` call in the rescore node**, not the trainer figure
R0251 led with. The next distinct mechanism is one `sha256` over the
`3,072,000,128`-byte substrate at `1.300640341010876` s `= 0.517986695078673x`,
which R0251 *projected* to `65.03201434087654` s `= 25.89933367479475x` at a
100M substrate. Neither of those is inside `fit()`, so R0251's per-batch poll —
correct, cheap, and worth `14.79x` inside the loop — did not make a node
stoppable. This module is the instrument for the two calls that did bind.

**The hash.** Three candidate fixes were on the table and only one of them keeps
the guarantee intact:

* *chunk the hash and poll between chunks* — the loops in
  `basemap/artifact_identity.py` are **already** chunked at 8 MiB, so this adds
  one global load and one comparison per chunk and nothing else. Every byte is
  still read in order into the same `hashlib.sha256`; the no-follow open, the
  `st_nlink == 1` refusal and the before/after `fstat` identity comparison are
  untouched. The only new outcome is that a poll may raise, in which case the
  caller receives an exception and **no digest at all** — never a partial one.
  The gap becomes the time to read and hash 8 MiB, which is independent of the
  file's size. **Adopted.**
* *fold the hash into the streaming read that already touches the substrate* —
  there is no such read. The substrate is consumed as a read-only `np.memmap` in
  sampler order, and the standing safety rule is that cuVS and the loaders are
  handed a memmap rather than a materialised array. Folding would either force a
  linear pass the science path does not perform, or verify the bytes *after* the
  trainer had already consumed them, which inverts verify-before-use.
  **Rejected.**
* *hash once and cache by `(path, size, mtime, inode)`* — this substitutes a
  `stat` guarantee for a content guarantee. `mtime` is settable with
  `os.utime`, a same-length in-place write moves no size, and inode numbers are
  reused after `unlink`. The cache would also have to survive across processes
  to help a ladder at all, which means persisting it, which means the cache file
  becomes an input nobody hashes. Chunked polling costs nothing and keeps the
  content guarantee, so there is no reason to accept a weaker one.
  **Rejected.**

**The scorer.** `basemap/panel_v2.score_panel` had no abort read anywhere inside
it, so the whole call was one interval. Ten sites are added: seven between the
existing phases of `score_panel` and three between the existing bounded loop
iterations of `_self_knn`, which is where the time is. No metric, neighbour set,
ordering or rounding is touched, and the round proves that by rescoring R0218's
archived seed-42 checkpoint with the hook installed and requiring the twelve
sealed values and the coordinate digest to be **byte-identical** to the arm
without it.

**Stoppability is measured, not asserted.** For each of the three binding sites
— hash, scorer, trainer loop — a control writes a flag file mid-work and the
node measures the wall time from the write to the exception arriving. That is
the quantity a ten-hour node's operator cares about, and no round in this
program has ever measured it.

Every ceiling verdict this module produces is an `AbortPollGate` verdict, and
that class is open from review-0249-01 §B.1/§B.2: `AbortPollGate.max_gap_s` is a
plain mutable attribute `verdict()` reads, so one assignment yields a passing
receipt indistinguishable from an honest one. The owner has stopped guard work;
this round edits no guard module and mutates no gate attribute, and it says so
here rather than letting the exposure pass silently.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .round0247_registry import registered_value
from .round0251_trainer_setup import (
    PollRecorder,
    SHAPE_IDENTIFICATION_SPREAD_LIMIT,
    tail_model,
    tail_verdict,
)


ROUND_ID = "0252"

HASH_CAPABILITY = "round0252-chunked-integrity-hash-stoppability-v1"
HASH_SCHEMA = "round0252-chunked-integrity-hash-stoppability-v1"
PANEL_CAPABILITY = "round0252-score-panel-abort-poll-and-rescore-v1"
PANEL_SCHEMA = "round0252-score-panel-abort-poll-and-rescore-v1"
TAIL_CAPABILITY = "round0252-long-rung-batch-tail-v1"
TAIL_SCHEMA = "round0252-long-rung-batch-tail-v1"

#: The rung that answers review-0251-01 §E.3. R0251 fitted a tail to `10,000`
#: batches and refused to publish a return level because the pre-registered
#: threshold ladder moved it `86.69863492594678x` against an identification
#: limit of `10.0`. This is `60x` more batches, chosen so the whole queue stays
#: inside the round's 3.0 GPU-h cap with room for one retry.
TAIL_RUNG_UPDATES = 600_000
#: The short fits the stop-latency controls interrupt. Long enough to be past
#: setup and into steady state, short enough to cost seconds.
STOP_CONTROL_UPDATES = 3_000
#: How long a control lets the work run before it asks it to stop.
STOP_CONTROL_DELAY_S = 5.0

NOT_A_FAMILY_CELL = (
    "R0252's rungs measure stoppability. Their horizons are "
    f"{STOP_CONTROL_UPDATES} and {TAIL_RUNG_UPDATES} updates, not the registered "
    "80,163; they publish no map, define no cell of the exact-graph seed family, "
    "and no floor may be fitted to them."
)

#: The 100M substrate this program is heading for: 100,000,000 x 384 fp32.
TARGET_ROWS = 100_000_000
TARGET_DIMENSION = 384
TARGET_SUBSTRATE_BYTES = TARGET_ROWS * TARGET_DIMENSION * 4

THE_INSTRUMENT_IS_DEFEATABLE = (
    "Every ceiling verdict in this artifact is an AbortPollGate verdict, and "
    "`AbortPollGate.max_gap_s` is a plain mutable attribute that `verdict()` "
    "reads: one assignment produces a PASSING receipt indistinguishable from an "
    "honest one, with no override record, no waived arm and no "
    "declared/effective pair for a reviewer to notice. The EnforcedHostWatchdog "
    "beside it has zero read-only properties. review-0249-01 items B.1 and B.2 "
    "are OPEN; the owner has stopped guard work, so R0252 edits no guard module, "
    "mutates no gate or watchdog attribute, and discloses the exposure instead "
    "of closing it. The gap series in this artifact comes from a separate "
    "PollRecorder that wraps the gate rather than replacing it, so the two can "
    "be cross-checked against each other. That is corroboration, not a fix."
)

INTEGRITY_GUARANTEE = {
    "still_guaranteed": [
        "every byte of the file is read in file order and fed to the same "
        "hashlib.sha256 object; the digest is the same function of the same "
        "bytes it was before this round",
        "_regular_file_identity still opens with O_NOFOLLOW, still refuses a "
        "hard-linked input (st_nlink != 1), and still compares st_dev, st_ino, "
        "st_mode, st_nlink, st_size, st_mtime_ns and st_ctime_ns before and "
        "after the read, so 'the file did not change while it was hashed' holds "
        "unchanged",
        "no digest is ever cached, memoised, reused across calls, or derived "
        "from anything but a complete in-order read",
    ],
    "newly_possible": [
        "a hash can now terminate early. When it does it raises and the caller "
        "receives NO digest -- there is no partial digest, no truncated digest, "
        "and no code path on which a short read yields a value",
    ],
    "not_weakened_because": (
        "the change is a call to a module-level hook between two existing loop "
        "iterations. It reads no file state, writes nothing, and returns "
        "nothing. With no hook installed (the default everywhere outside a "
        "measured stage) it is one global load and one `is not None` comparison "
        "per 8 MiB."
    ),
    "rejected_alternatives": {
        "cache_by_path_size_mtime_inode": (
            "REJECTED. It substitutes a stat guarantee for a content guarantee: "
            "os.utime sets mtime freely, an in-place same-length write moves "
            "neither size nor inode, and inode numbers are reused after unlink. "
            "To help a ladder at all the cache must outlive the process, which "
            "makes it a persisted input that nothing hashes. Chunked polling "
            "costs nothing and keeps the content guarantee."
        ),
        "fold_into_the_streaming_read": (
            "REJECTED. No streaming read touches the substrate linearly: the "
            "loaders and cuVS take a read-only np.memmap in sampler order, which "
            "is a standing safety precondition, not an optimisation. Folding "
            "would either force a linear pass the science path does not perform "
            "or verify bytes after the trainer consumed them, inverting "
            "verify-before-use."
        ),
    },
}


class Round0252Error(RuntimeError):
    """The registered R0252 stoppability contract changed."""


class Round0252CooperativeAbort(RuntimeError):
    """Raised by a control poll once its flag file has appeared."""


# --------------------------------------------------------------------------- #
# gap reporting -- keys chosen so `roundreport`'s census enumerates every one
# --------------------------------------------------------------------------- #


def _widest(entries: Sequence[tuple[str, float]]) -> tuple[str, float]:
    if not entries:
        return ("", 0.0)
    site, gap = max(entries, key=lambda item: item[1])
    return (str(site), float(gap))


def gap_report(records: Sequence[tuple[str, float]], *, arm: str) -> dict[str, Any]:
    """Widest abort-read gap overall and per site, with its ratio to the ceiling.

    Key names are deliberately the ones `roundreport`'s abort-poll gap census
    enumerates (`widest_gap_s` beside `widest_gap_over_the_ceiling`, under a
    `registered_ceiling_s`), so every gap this round measures is ranked by the
    generator rather than nominated by the author. That includes the control
    arms, whose gaps are *larger* by construction: a census the round can
    quietly stay out of is the defect the census was built to remove.
    """
    entries = [(str(site), float(gap)) for site, gap in records]
    if not entries:
        raise Round0252Error(f"R0252 arm {arm!r} recorded no abort reads")
    ceiling = registered_value("r0246_max_poll_spacing_s")
    by_site: dict[str, Any] = {}
    for site, gap in entries:
        entry = by_site.setdefault(
            site, {"reads": 0, "widest_gap_s": 0.0, "total_s": 0.0}
        )
        entry["reads"] += 1
        entry["total_s"] += gap
        entry["widest_gap_s"] = max(entry["widest_gap_s"], gap)
    for entry in by_site.values():
        entry["widest_gap_over_the_ceiling"] = entry["widest_gap_s"] / ceiling
    site, gap = _widest(entries)
    return {
        "arm": arm,
        "registered_ceiling_s": ceiling,
        "reads": len(entries),
        "wall_s": sum(value for _site, value in entries),
        "widest_gap_s": gap,
        "widest_gap_over_the_ceiling": gap / ceiling,
        "widest_gap_after": site,
        "meets_the_registered_ceiling": bool(gap <= ceiling),
        "gaps_by_site": by_site,
    }


def gap_reduction(
    *, before: Mapping[str, Any], after: Mapping[str, Any], margin_required: float = 2.0
) -> dict[str, Any]:
    """What polling bought, and whether it cleared the ceiling with margin.

    Polling never makes work faster. It replaces "the widest gap is the whole
    call" with "the widest gap is the largest single un-polled unit", and the
    number that matters is whether THAT clears the ceiling with the
    pre-registered `2.0x` margin R0250 §A5 fixed. Reported either way, including
    the shortfall in the margin-missed-but-under-ceiling branch that
    review-0251-01 §C.4 found R0251 published as `null`.
    """
    ceiling = registered_value("r0246_max_poll_spacing_s")
    b = float(before["widest_gap_s"])
    a = float(after["widest_gap_s"])
    if not math.isfinite(b) or not math.isfinite(a) or a <= 0.0:
        raise Round0252Error(f"R0252 gap reduction needs finite positive gaps: {b!r} / {a!r}")
    margin = ceiling / a
    below = a <= ceiling
    with_margin = margin >= float(margin_required)
    return {
        "registered_ceiling_s": ceiling,
        "widest_gap_before_s": b,
        "widest_gap_after_s": a,
        "before_over_the_ceiling": b / ceiling,
        "after_over_the_ceiling": a / ceiling,
        "reduction_factor": b / a,
        "margin_after": margin,
        "required_margin": float(margin_required),
        "is_below_the_ceiling": bool(below),
        "is_below_the_ceiling_with_the_required_margin": bool(with_margin),
        # review-0251-01 §C.4: R0251 populated this only when the CEILING was
        # breached, so the branch that actually occurred reported `null` and its
        # positive control exercised the other branch. Both branches are covered
        # here and both are exercised by the contract suite.
        "shortfall_over_the_ceiling_if_over": (None if below else a / ceiling),
        "shortfall_factor_against_the_required_margin_if_under": (
            None if (not below or with_margin) else float(margin_required) / margin
        ),
        "gap_that_would_clear_the_required_margin_s": ceiling / float(margin_required),
    }


# --------------------------------------------------------------------------- #
# the size law -- is the polled gap independent of the file, or not?
# --------------------------------------------------------------------------- #


def size_law(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Least-squares fit of widest polled gap against file bytes.

    The whole claim of the chunked fix is that the gap stops depending on the
    file size. That is a slope, so it is fitted rather than asserted: an
    unpolled hash has a slope of `1 / throughput` s per byte and a polled one
    should have a slope indistinguishable from zero. Reported with the implied
    gap at the 100M substrate under BOTH readings, so the reader can see how far
    apart they are without taking the round's word for which applies.
    """
    xs = [float(point["bytes"]) for point in points]
    ys = [float(point["widest_gap_s"]) for point in points]
    n = len(xs)
    if n < 2:
        raise Round0252Error("R0252 size law needs at least two points")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        raise Round0252Error("R0252 size law needs at least two distinct sizes")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ceiling = registered_value("r0246_max_poll_spacing_s")
    linear = slope * TARGET_SUBSTRATE_BYTES + intercept
    return {
        "points": [dict(point) for point in points],
        "slope_seconds_per_byte": slope,
        "intercept_s": intercept,
        "residual_max_abs_s": max(abs(value) for value in residuals),
        "observed_max_gap_s": max(ys),
        "observed_max_gap_over_the_ceiling": max(ys) / ceiling,
        "span_of_sizes": max(xs) / min(xs),
        "gap_at_the_100m_substrate_under_the_linear_fit_s": linear,
        "gap_at_the_100m_substrate_under_the_linear_fit_over_the_ceiling": (
            linear / ceiling
        ),
        "registered_ceiling_s": ceiling,
        "target_substrate_bytes": TARGET_SUBSTRATE_BYTES,
        "note": (
            "a fit is not needed to answer this round's question -- the 100M-scale "
            "point is MEASURED, not extrapolated. The fit is published so the "
            "shape of the dependence is visible: a per-chunk gap should have a "
            "slope near zero and a whole-file gap should have slope 1/throughput."
        ),
    }


# --------------------------------------------------------------------------- #
# stop latency -- the quantity a ten-hour node's operator actually cares about
# --------------------------------------------------------------------------- #


class FlagFileAbortPoll:
    """A poll that raises once a flag file exists, forwarding to an inner reader.

    This is the shape the runner already uses: the cooperative abort flag is a
    file, and the node observes it between units of work. The control writes its
    OWN flag path, never the runner's `<queue_root>/logs/<node>.abort`, so
    measuring stoppability cannot stop the queue.
    """

    def __init__(self, *, flag_path: str, inner: Any = None, clock: Any = time.monotonic) -> None:
        if not callable(clock):
            raise Round0252Error("R0252 flag poll needs a callable clock")
        if inner is not None and not callable(inner):
            raise Round0252Error("R0252 flag poll inner reader must be callable")
        self.flag_path = str(flag_path)
        self._inner = inner
        self._clock = clock
        self.reads = 0
        self.observed_at: float | None = None
        self.observed_site: str | None = None

    def __call__(self, where: str) -> None:
        self.reads += 1
        if self._inner is not None:
            self._inner(where)
        if os.path.exists(self.flag_path):
            self.observed_at = float(self._clock())
            self.observed_site = str(where)
            raise Round0252CooperativeAbort(
                f"R0252 cooperative abort observed at {where!r}"
            )


def measure_stop_latency(
    *,
    label: str,
    flag_path: str,
    delay_s: float,
    run: Any,
    inner: Any = None,
    clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Run `run(poll)`, plant the flag after `delay_s`, and time the stop.

    `run` installs `poll` wherever the site under test reads it and performs the
    work. The returned latency is wall time from the flag file's `close()` to the
    `Round0252CooperativeAbort` arriving back here, so it includes the poll's own
    `os.path.exists`, the remaining work in the unit that was in flight, and the
    unwinding of every frame between the site and this call.
    """
    if os.path.exists(flag_path):
        raise Round0252Error(f"R0252 stop-latency flag already exists: {flag_path}")
    poll = FlagFileAbortPoll(flag_path=flag_path, inner=inner, clock=clock)
    written: dict[str, float] = {}
    cancelled = threading.Event()
    staging = f"{flag_path}.staging"

    def plant() -> None:
        # The timestamp is taken BEFORE the flag can become visible, so the
        # measured latency can never be negative and always includes the rename.
        # The content is written and fsync'd first, then moved into place
        # atomically, which is how the runner publishes its own abort flag.
        if cancelled.wait(float(delay_s)):
            return
        with open(staging, "w", encoding="utf-8") as handle:
            handle.write('{"reason":"R0252 stop-latency control"}\n')
            handle.flush()
            os.fsync(handle.fileno())
        written["at"] = float(clock())
        os.replace(staging, flag_path)

    planter = threading.Thread(target=plant, name="r0252-stop-control", daemon=True)
    started = float(clock())
    stopped = False
    completed = False
    planter.start()
    try:
        try:
            run(poll)
            completed = True
        except Round0252CooperativeAbort:
            stopped = True
    finally:
        cancelled.set()
        planter.join()
        for path in (staging, flag_path):
            if os.path.exists(path):
                os.unlink(path)
    observed = poll.observed_at
    planted = written.get("at")
    latency = (
        (float(observed) - float(planted))
        if (stopped and observed is not None and planted is not None)
        else None
    )
    ceiling = registered_value("r0246_max_poll_spacing_s")
    return {
        "label": str(label),
        "registered_ceiling_s": ceiling,
        "flag_delay_s": float(delay_s),
        "abort_reads_before_the_flag": poll.reads,
        "the_work_stopped_cooperatively": bool(stopped),
        "the_work_ran_to_completion_instead": bool(completed),
        "stopped_at_site": poll.observed_site,
        "wall_from_start_to_stop_s": (
            (float(observed) - started) if observed is not None else None
        ),
        "stop_latency_s": latency,
        "stop_latency_over_the_ceiling": (
            None if latency is None else latency / ceiling
        ),
        "stop_latency_meets_the_registered_ceiling": (
            None if latency is None else bool(latency <= ceiling)
        ),
        "measures": (
            "wall time from the flag file being fsync'd and closed to the "
            "cooperative-abort exception arriving at the caller. It includes the "
            "poll's own os.path.exists, the remainder of the unit of work in "
            "flight, and the unwinding of every frame in between. It is NOT a "
            "model."
        ),
        "not_the_runner_flag": (
            "this control writes its own flag path. The runner's "
            "<queue_root>/logs/<node>.abort is never written, so measuring "
            "stoppability cannot stop the queue."
        ),
    }


# --------------------------------------------------------------------------- #
# the tail, at a rung sixty times longer
# --------------------------------------------------------------------------- #


def tail_identification(
    gaps: Sequence[float], *, arm_wall_s: float, prior_rung: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """R0251's tail model and verdict, refitted on this round's longer rung.

    Nothing about the estimator changes -- same peaks-over-threshold fit, same
    pre-registered threshold ladder, same bootstrap, same exponential reference,
    same distribution-free bound, same `10.0` identification limit. The only
    change is `m`. R0251 refused to publish a return level because the ladder
    moved it `86.7x`; whether sixty times the batches identifies the fit is the
    measurement, and the refusal stands unchanged if it does not.
    """
    model = tail_model(gaps, arm_wall_s=float(arm_wall_s))
    verdict = tail_verdict(model)
    out = {
        "tail_model": model,
        "tail_verdict": verdict,
        "identification_limit": SHAPE_IDENTIFICATION_SPREAD_LIMIT,
        "the_estimator_is_r0251s_unchanged": True,
    }
    if prior_rung is not None:
        prior_spread = float(prior_rung["threshold_ladder_return_level_spread"])
        spread = float(verdict["threshold_ladder_return_level_spread"])
        out["against_the_prior_rung"] = {
            "prior_batches": int(prior_rung["batches"]),
            "prior_threshold_ladder_spread": prior_spread,
            "prior_fit_was_identified": bool(prior_rung["identified"]),
            "batches_now": int(model["peaks_over_threshold"]["batches_observed"]),
            "batch_multiple": (
                int(model["peaks_over_threshold"]["batches_observed"])
                / max(1, int(prior_rung["batches"]))
            ),
            "threshold_ladder_spread_now": spread,
            "spread_shrank_by": (prior_spread / spread) if spread > 0 else None,
        }
    return out


#: R0251's sealed tail result, carried so this round can say whether the longer
#: rung changed the answer. Sourced from R0251's artifact at run time, never
#: typed here -- these are only the keys the comparison needs.
PRIOR_RUNG_KEYS = ("batches", "threshold_ladder_return_level_spread", "identified")


def prior_rung_from_artifact(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Pull R0251's rung size, ladder spread and verdict out of its sealed bytes."""
    verdict = dict(receipt["tail_verdict"])
    model = dict(receipt["tail_model"])
    return {
        "batches": int(dict(model["peaks_over_threshold"])["batches_observed"]),
        "threshold_ladder_return_level_spread": float(
            verdict["threshold_ladder_return_level_spread"]
        ),
        "identified": bool(verdict["the_extreme_value_fit_is_identified"]),
    }


# --------------------------------------------------------------------------- #
# the release check: do the declared sites exist where this module says?
# --------------------------------------------------------------------------- #


def declared_sites_match_the_release() -> dict[str, Any]:
    """The hook sites this round declares, against the release modules' own.

    A docstring that lists poll sites is a claim; this is the check. It reads the
    modules, so a site renamed or a call removed without updating this module
    fails closed here rather than silently measuring a wider interval.
    """
    import ast

    from . import artifact_identity, panel_v2
    from .pumap.parametric_umap import ParametricUMAP

    report: dict[str, Any] = {}
    for name, module, expected in (
        ("artifact_identity", artifact_identity, artifact_identity.ABORT_POLL_SITES),
        ("panel_v2", panel_v2, panel_v2.ABORT_POLL_SITES),
    ):
        if getattr(module, "abort_poll", "missing") is not None:
            raise Round0252Error(
                f"R0252 requires {name}.abort_poll to default to None; it is "
                f"{getattr(module, 'abort_poll', 'missing')!r}"
            )
        if not callable(getattr(module, "set_abort_poll", None)):
            raise Round0252Error(f"R0252 requires {name}.set_abort_poll")
        source = ast.parse(open(module.__file__, encoding="utf-8").read())
        called = sorted({
            node.args[0].id
            for node in ast.walk(source)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_poll_abort"
            and node.args
            and isinstance(node.args[0], ast.Name)
        })
        declared = sorted(
            attribute
            for attribute in dir(module)
            if attribute.startswith("ABORT_POLL_SITE_")
        )
        if called != declared:
            raise Round0252Error(
                f"R0252 {name} call sites {called!r} do not match its declared "
                f"site constants {declared!r}"
            )
        report[name] = {
            "declared_sites": list(expected),
            "call_site_constants": called,
            "hook_default_is_none": True,
            "sites_match": True,
        }
    if not callable(getattr(ParametricUMAP, "_poll_abort", None)):
        raise Round0252Error("R0252 requires the R0251 trainer hook to be intact")
    report["parametric_umap"] = {
        "hook_attribute": "ParametricUMAP.abort_poll",
        "hook_default_is_none": getattr(ParametricUMAP(), "abort_poll", "missing") is None,
        "sites_match": True,
    }
    return report


__all__ = [
    "FlagFileAbortPoll",
    "HASH_CAPABILITY",
    "HASH_SCHEMA",
    "INTEGRITY_GUARANTEE",
    "NOT_A_FAMILY_CELL",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "PRIOR_RUNG_KEYS",
    "PollRecorder",
    "ROUND_ID",
    "Round0252CooperativeAbort",
    "Round0252Error",
    "STOP_CONTROL_DELAY_S",
    "STOP_CONTROL_UPDATES",
    "TAIL_CAPABILITY",
    "TAIL_RUNG_UPDATES",
    "TAIL_SCHEMA",
    "TARGET_DIMENSION",
    "TARGET_ROWS",
    "TARGET_SUBSTRATE_BYTES",
    "THE_INSTRUMENT_IS_DEFEATABLE",
    "declared_sites_match_the_release",
    "gap_reduction",
    "gap_report",
    "measure_stop_latency",
    "prior_rung_from_artifact",
    "size_law",
    "tail_identification",
]

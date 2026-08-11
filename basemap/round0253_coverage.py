"""R0253 — make poll coverage quantitative, so silence stops reading as safety.

`roundreport`'s abort-poll gap census grew a poll-coverage table after
review-0252-01, and on R0252 every node reads `not instrumented`: the round's
`0.5318989691995224x` non-control maximum was computed over an unknown fraction
of a `2042.770` s node.  The tool already looks for a covered span under any of

    observed_span_s | polled_span_s | poll_window_s | covered_wall_s |
    instrumented_wall_s

and finds none, because no node emits one.  That is the blind spot at its
source: a region that polls zero times has no gap *between* polls, so it emits
nothing, and a census with no denominator cannot distinguish "we looked and found
0.03x" from "we looked at 12% of the node".

`CoverageLedger` is the denominator.  A node opens one window per instrumented
stage; every abort read inside that window stamps the window's first and last
observation.  `observed_span_s` is the union of those windows in wall time --
union, not sum, so overlapping or nested instrumentation cannot inflate it.  The
node publishes it beside its own measured wall, and `roundreport` then prints a
real percentage.

Deliberate conservatism, in both directions:

* A window's span runs from its FIRST observation to its LAST.  Work before the
  first read and after the last read is *not* claimed as covered, which is right:
  those are precisely the intervals `_write_sized_file` hid in.
* A window with fewer than two observations covers `0.0` s.  One read tells you
  nothing about an interval.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any


class Round0253CoverageError(RuntimeError):
    """The registered R0253 coverage-ledger contract changed."""


class CoverageWindow:
    """One instrumented stage.  Wrap the stage's poll and the span records itself."""

    __slots__ = ("label", "first_at", "last_at", "observations", "_clock", "_closed")

    def __init__(self, *, label: str, clock: Any) -> None:
        self.label = str(label)
        self.first_at: float | None = None
        self.last_at: float | None = None
        self.observations = 0
        self._clock = clock
        self._closed = False

    def observe(self, _where: str = "") -> None:
        now = float(self._clock())
        if self.first_at is None:
            self.first_at = now
        self.last_at = now
        self.observations += 1

    def wrap(self, poll: Any) -> Any:
        """Return a poll that stamps this window and then forwards to `poll`.

        The stamp happens BEFORE the inner poll runs, so a poll that raises (the
        cooperative-abort case) still contributes its observation and the window
        does not silently shrink at exactly the moment that matters.
        """
        if not callable(poll):
            raise Round0253CoverageError("R0253 coverage window can only wrap a callable")

        def wrapped(where: str) -> None:
            self.observe(where)
            poll(where)

        wrapped.coverage_window = self  # type: ignore[attr-defined]
        return wrapped

    @property
    def span_s(self) -> float:
        if self.observations < 2 or self.first_at is None or self.last_at is None:
            return 0.0
        return float(self.last_at - self.first_at)

    def close(self) -> None:
        self._closed = True

    def receipt(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "observations": int(self.observations),
            "span_s": self.span_s,
            "covers_nothing_because_it_saw_fewer_than_two_reads": self.observations < 2,
        }


class CoverageLedger:
    """Every instrumented window in one node, and their union in wall time."""

    def __init__(self, *, node: str, clock: Any = time.monotonic) -> None:
        if not callable(clock):
            raise Round0253CoverageError("R0253 coverage ledger needs a callable clock")
        self.node = str(node)
        self._clock = clock
        self.node_started_at = float(clock())
        self.windows: list[CoverageWindow] = []

    def window(self, label: str) -> CoverageWindow:
        window = CoverageWindow(label=label, clock=self._clock)
        self.windows.append(window)
        return window

    # -- the union ---------------------------------------------------------- #

    def _intervals(self) -> list[tuple[float, float]]:
        return sorted(
            (float(w.first_at), float(w.last_at))
            for w in self.windows
            if w.observations >= 2 and w.first_at is not None and w.last_at is not None
        )

    def observed_span_s(self) -> float:
        merged: list[list[float]] = []
        for start, end in self._intervals():
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return float(sum(end - start for start, end in merged))

    def node_wall_s(self) -> float:
        return float(self._clock() - self.node_started_at)

    def receipt(self, *, node_wall_s: float | None = None) -> dict[str, Any]:
        """The block a node seals.  `observed_span_s` is the key `roundreport` reads.

        It is emitted at the TOP LEVEL of the node's artifact deliberately: the
        census takes the maximum of every `observed_span_s` it finds in a document,
        so a nested per-stage copy would let the largest stage stand in for the
        node.  Per-window spans are published under `windows`, where they are
        evidence rather than a denominator.
        """
        wall = float(node_wall_s) if node_wall_s is not None else self.node_wall_s()
        span = self.observed_span_s()
        return {
            "schema": "round0253-poll-coverage-v1",
            "node": self.node,
            #: read by `roundreport`'s `_COVERED_SPAN_KEYS`.
            "observed_span_s": span,
            "node_wall_s": wall,
            "uncovered_wall_s": max(0.0, wall - span),
            "covered_fraction": (span / wall) if wall > 0 else None,
            "windows": [window.receipt() for window in self.windows],
            "windows_opened": len(self.windows),
            "windows_that_saw_at_least_two_reads": sum(
                1 for window in self.windows if window.observations >= 2
            ),
            "what_uncovered_wall_time_means": (
                "A LOWER BOUND on an unobserved gap, not a measurement. Code that "
                "polls zero times emits no gap, so it cannot appear in the census "
                "at all; the only trace it leaves is wall time this ledger did not "
                "cover. review-0252-01 §H: a 93.135091 s cooperative stop -- "
                "37.09x the registered ceiling -- sat inside an uninstrumented "
                "write path and appeared in no census, in no artifact and in no "
                "result."
            ),
            "what_covered_wall_time_does_not_mean": (
                "Covered does not mean bounded. A window's span is the interval "
                "over which reads happened; the WIDEST gap inside it is the "
                "separate quantity the census ranks. A node can be 100% covered "
                "and still breach the ceiling, and a node can be 3% covered with "
                "every observed gap tiny."
            ),
        }


def coverage_summary(receipts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold several nodes' coverage receipts into one honest statement."""
    rows = [dict(item) for item in receipts]
    if not rows:
        raise Round0253CoverageError("R0253 coverage summary needs at least one node")
    wall = sum(float(row["node_wall_s"]) for row in rows)
    span = sum(float(row["observed_span_s"]) for row in rows)
    return {
        "schema": "round0253-poll-coverage-summary-v1",
        "nodes": len(rows),
        "total_node_wall_s": wall,
        "total_observed_span_s": span,
        "total_uncovered_wall_s": max(0.0, wall - span),
        "covered_fraction": (span / wall) if wall > 0 else None,
        "by_node": {
            str(row["node"]): {
                "node_wall_s": float(row["node_wall_s"]),
                "observed_span_s": float(row["observed_span_s"]),
                "covered_fraction": row.get("covered_fraction"),
            }
            for row in rows
        },
    }


__all__ = [
    "CoverageLedger",
    "CoverageWindow",
    "Round0253CoverageError",
    "coverage_summary",
]

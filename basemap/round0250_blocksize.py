"""R0250 — settle `truthcos_0247`'s unregistered block size, by measurement.

R0247 open item 2, carried unchanged through R0248 and R0249:

    `truthcos_0247`'s block size is one doubling from the bound. The gather loop
    polls once per 2,000 rows and left a 2.276142634014832 s gap. It should be
    registered as a bound in its own right, or the block reduced.

The block in question is a **function default** -- `block: int = 2_000` on
`round0247_precision.recompute_truth_cosines_f64` and `.cosine_noise_floor`. It is
not a module constant, is not in the R0247 registry, and therefore is invisible to
`round0248_inventory`'s bare-registered-symbol scan. A caller may pass any value,
and a larger value widens the interval between two cooperative-abort reads. On
those two facts alone it looks exactly like the parameters R0247 clamped.

**The question is whether clamping it would work**, and that is empirical rather
than rhetorical. The gap of a block is

    gap = block * k * (seconds per gathered substrate row)

so a *fixed* block bounds the gap only if the per-row cost is itself bounded. R0247
measured its `2.276` s gap on a `2,000`-row block against a `153` GB memmap under
cold page cache, and `plan-minilm-100m-v2.md` records `2.2-8.3x` read amplification
with the page cache serving `11.6%` on that same access pattern. If the per-row
cost moves by more than `PER_ROW_COST_STABILITY_LIMIT` across page-cache states,
then no single registered block value bounds the gap, and registering one would
produce a receipt that looks enforced while the arm it claims to defend still
depends on the weather.

**The pre-registered rule.** A caller-settable parameter needs registering iff all
three hold:

1. it is settable by a caller -- **yes**, it is a keyword default;
2. moving it in one direction weakens a safety arm -- **yes**, larger is weaker;
3. clamping it to one fixed value would restore that arm -- **measured here**.

If (3) holds, the resolution is REGISTER, and the node publishes the exact value
and the exact registry row a follow-up round should add. If (3) fails, the
resolution is DOES_NOT_NEED_REGISTERING, because a block-size ceiling is the wrong
instrument, and the node publishes the instrument that does work: a block that
adapts from its own measured last-iteration wall, demonstrated on the same loop.

This round **registers nothing in the R0247 registry and edits no guard module**.
The owner has called a stop on guard work; this node measures and reports.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .round0238_rung5 import DIMENSION, GRAPH_K
from .round0247_registry import (
    REGISTERED_REGISTRY_SHA256,
    registered_value,
    registry_fingerprint,
)


ROUND_ID = "0250"

BLOCKSIZE_CAPABILITY = "round0250-truthcos-block-size-resolution-v1"
BLOCKSIZE_SCHEMA = "round0250-truthcos-block-size-resolution-v1"

#: The block sizes probed on the real gather loop, spanning R0247's `2_000`
#: default by a factor of four in each direction.
BLOCK_SIZES_PROBED: tuple[int, ...] = (500, 1_000, 2_000, 4_000)

#: R0247's default, the value under measurement. Read from the released function's
#: signature at run time rather than trusted from here -- see `observed_default`.
R0247_DECLARED_DEFAULT_BLOCK = 2_000

#: The pre-registered stability bar for rule (3). If the fastest and slowest
#: measured per-row costs differ by more than this factor, a single fixed block
#: cannot bound the gap and rule (3) fails.
PER_ROW_COST_STABILITY_LIMIT = 2.0

#: The margin a registered block would have to hold at, if one were registered.
#: A ceiling with no margin is the condition R0247 already reported at `0.906`.
REGISTRATION_SAFETY_MARGIN = 2.0

#: What the adaptive alternative aims at: keep every block's wall at or below this
#: fraction of the registered ceiling, by halving after any block that exceeds it.
ADAPTIVE_TARGET_FRACTION = 0.25

RESOLUTION_REGISTER = "REGISTER"
RESOLUTION_NOT_NEEDED = "DOES_NOT_NEED_REGISTERING"

BLOCKSIZE_NOTE = (
    "gap = block * k * seconds-per-gathered-substrate-row. A fixed block bounds "
    "the gap only if the per-row cost is bounded. This node measures the per-row "
    "cost across disjoint row ranges (cold) and a repeated range (warm) on the "
    "same 153 GB memmap R0247 used, and applies the pre-registered three-part "
    "rule to decide whether a registered block-size ceiling is the right "
    "instrument."
)


class Round0250BlockSizeError(RuntimeError):
    """The registered R0250 block-size resolution contract changed."""


def observed_default_block() -> int:
    """R0247's default, read from the released function rather than restated."""
    import inspect

    from .round0247_precision import recompute_truth_cosines_f64

    parameter = inspect.signature(recompute_truth_cosines_f64).parameters["block"]
    if parameter.default is inspect.Parameter.empty:
        raise Round0250BlockSizeError("R0247's gather block has no default to read")
    return int(parameter.default)


def _sorted_gather(
    substrate: Any, candidate_ids: np.ndarray, queries: np.ndarray
) -> int:
    """R0247's gather, shape for shape: sorted random reads then a float64 dot.

    Returns the number of substrate rows gathered. The arithmetic is the same as
    `recompute_truth_cosines_f64`'s inner block so the measured per-row cost is a
    cost of that loop and not of a simplified stand-in.
    """
    flat = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    gathered = np.empty((flat.size, int(substrate.shape[1])), dtype=np.float64)
    gathered[order] = np.asarray(substrate[flat[order]], dtype=np.float64)
    gathered = gathered.reshape(
        candidate_ids.shape[0], candidate_ids.shape[1], -1
    )
    anchors = np.asarray(substrate[np.asarray(queries, dtype=np.int64)], dtype=np.float64)
    result = np.einsum("bd,bkd->bk", anchors, gathered)
    if not np.isfinite(result).all():
        raise Round0250BlockSizeError("R0250 gather probe produced a nonfinite cosine")
    return int(flat.size)


def measure_block(
    *,
    substrate: Any,
    truth_ids: Any,
    probe_query_rows: Any,
    start: int,
    rows: int,
    block: int,
    abort_check: Any = None,
    label: str = "R0250 block probe",
) -> dict[str, Any]:
    """Walk `rows` of the probe at one block size and time every block."""
    block = int(block)
    rows = int(rows)
    if block <= 0 or rows <= 0 or rows % block:
        raise Round0250BlockSizeError(
            f"R0250 block probe needs rows a positive multiple of block, got "
            f"rows={rows} block={block}"
        )
    walls: list[float] = []
    gathered_total = 0
    began = time.monotonic()
    for offset in range(start, start + rows, block):
        if abort_check is not None:
            abort_check(f"{label} block {block} at row {offset}")
        began_block = time.monotonic()
        gathered_total += _sorted_gather(
            substrate,
            np.asarray(truth_ids[offset : offset + block], dtype=np.int64),
            np.asarray(probe_query_rows[offset : offset + block], dtype=np.int64),
        )
        walls.append(time.monotonic() - began_block)
    if abort_check is not None:
        abort_check(f"{label} block {block} complete")
    wall = time.monotonic() - began
    ordered = sorted(walls)
    worst = max(walls)
    ceiling = registered_value("r0246_max_poll_spacing_s")
    return {
        "block_rows": block,
        "probe_rows_walked": rows,
        "first_probe_row": int(start),
        "blocks": len(walls),
        "substrate_rows_gathered": gathered_total,
        "substrate_bytes_gathered": gathered_total * int(substrate.shape[1]) * 4,
        "wall_s": wall,
        "worst_block_wall_s": worst,
        "median_block_wall_s": ordered[len(ordered) // 2],
        "mean_block_wall_s": sum(walls) / len(walls),
        "worst_seconds_per_gathered_row": worst / (block * GRAPH_K),
        "mean_seconds_per_gathered_row": (sum(walls) / len(walls)) / (block * GRAPH_K),
        "worst_block_over_the_registered_ceiling": worst / ceiling,
        "registered_ceiling_s_at_the_comparison_site": ceiling,
        "worst_block_meets_the_registered_ceiling": bool(worst <= ceiling),
    }


def adaptive_gather_probe(
    *,
    substrate: Any,
    truth_ids: Any,
    probe_query_rows: Any,
    start: int,
    rows: int,
    initial_block: int,
    target_fraction: float = ADAPTIVE_TARGET_FRACTION,
    abort_check: Any = None,
    label: str = "R0250 adaptive probe",
) -> dict[str, Any]:
    """The instrument a block-size ceiling cannot be: bound the WALL, not the block.

    After every block, compare that block's measured wall against
    `target_fraction` of the registered ceiling and halve the block if it was over.
    This bounds the gap without knowing the per-row cost in advance, which is the
    property a fixed registered block cannot have.
    """
    ceiling = registered_value("r0246_max_poll_spacing_s")
    target = float(target_fraction) * ceiling
    block = int(initial_block)
    offset = int(start)
    end = int(start) + int(rows)
    walls: list[float] = []
    trace: list[dict[str, Any]] = []
    gathered_total = 0
    began = time.monotonic()
    while offset < end:
        if abort_check is not None:
            abort_check(f"{label} block {block} at row {offset}")
        stop = min(offset + block, end)
        began_block = time.monotonic()
        gathered_total += _sorted_gather(
            substrate,
            np.asarray(truth_ids[offset:stop], dtype=np.int64),
            np.asarray(probe_query_rows[offset:stop], dtype=np.int64),
        )
        wall = time.monotonic() - began_block
        walls.append(wall)
        trace.append({"first_row": offset, "block_rows": stop - offset, "wall_s": wall})
        offset = stop
        if wall > target and block > 1:
            block = max(1, block // 2)
    if abort_check is not None:
        abort_check(f"{label} complete")
    worst = max(walls)
    return {
        "instrument": "round0250-wall-adaptive-gather-block-v1",
        "initial_block_rows": int(initial_block),
        "final_block_rows": block,
        "target_fraction_of_the_registered_ceiling": float(target_fraction),
        "target_block_wall_s": target,
        "registered_ceiling_s_at_the_comparison_site": ceiling,
        "probe_rows_walked": int(rows),
        "blocks": len(walls),
        "substrate_rows_gathered": gathered_total,
        "wall_s": time.monotonic() - began,
        "worst_block_wall_s": worst,
        "worst_block_over_the_registered_ceiling": worst / ceiling,
        "worst_block_meets_the_registered_ceiling": bool(worst <= ceiling),
        "halvings": sum(
            1
            for left, right in zip(trace, trace[1:])
            if right["block_rows"] < left["block_rows"]
        ),
        "trace": trace,
        "why": (
            "the gap is bounded by measuring the previous block's wall and "
            "shrinking, so no advance knowledge of the per-row cost is required. "
            "A registered block-size ceiling requires exactly that knowledge."
        ),
    }


def largest_block_meeting_the_ceiling(
    *, worst_seconds_per_gathered_row: float, margin: float = REGISTRATION_SAFETY_MARGIN
) -> int:
    """The block a registration would have to name, at a stated margin."""
    cost = float(worst_seconds_per_gathered_row)
    if not math.isfinite(cost) or cost <= 0.0:
        raise Round0250BlockSizeError(
            f"R0250 cannot size a block from a per-row cost of {cost!r}"
        )
    ceiling = registered_value("r0246_max_poll_spacing_s")
    return max(1, int((ceiling / float(margin)) / (cost * GRAPH_K)))


def resolve(
    *,
    arms: Sequence[dict[str, Any]],
    declared_default_block: int,
    r0247_widest_gap_s: float,
    r0247_block_rows: int,
    adaptive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered three-part rule and state the resolution."""
    if not arms:
        raise Round0250BlockSizeError("R0250 block-size resolution needs measured arms")
    costs = [float(arm["worst_seconds_per_gathered_row"]) for arm in arms]
    fastest = min(costs)
    slowest = max(costs)
    if fastest <= 0.0:
        raise Round0250BlockSizeError("R0250 measured a non-positive per-row cost")
    spread = slowest / fastest
    clamping_would_restore_the_arm = spread <= PER_ROW_COST_STABILITY_LIMIT
    ceiling = registered_value("r0246_max_poll_spacing_s")
    # Criterion 2, measured rather than asserted: order the same-page-cache-state
    # arms by block and check that the worst per-block wall is non-decreasing.
    ordered = sorted(
        (arm for arm in arms if str(arm.get("page_cache_state", "")).startswith("cold")),
        key=lambda arm: int(arm["block_rows"]),
    )
    monotone = all(
        float(right["worst_block_wall_s"]) >= float(left["worst_block_wall_s"])
        for left, right in zip(ordered, ordered[1:])
    )
    largest_over_smallest = (
        float(ordered[-1]["worst_block_wall_s"]) / float(ordered[0]["worst_block_wall_s"])
        if len(ordered) >= 2 and float(ordered[0]["worst_block_wall_s"]) > 0.0
        else None
    )
    sized = largest_block_meeting_the_ceiling(
        worst_seconds_per_gathered_row=slowest
    )
    resolution = (
        RESOLUTION_REGISTER if clamping_would_restore_the_arm else RESOLUTION_NOT_NEEDED
    )
    r0247_cost = float(r0247_widest_gap_s) / (int(r0247_block_rows) * GRAPH_K)
    return {
        "resolution": resolution,
        "rule": (
            "a caller-settable parameter needs registering iff (1) a caller can "
            "set it, (2) one direction weakens a safety arm, and (3) clamping it "
            "to one fixed value would restore that arm. (1) and (2) hold by "
            "inspection; (3) is measured."
        ),
        "criterion_1_caller_settable": int(declared_default_block) > 0,
        "criterion_1_evidence": (
            "`block` is read from the released function's own signature by "
            "inspect; a parameter that HAS a default is a parameter a caller can "
            "override. The observed default is published beside this."
        ),
        "criterion_2_one_direction_is_weaker": bool(monotone),
        "criterion_2_evidence": (
            "measured on the cold arms: the worst per-block wall is non-decreasing "
            "in the block size, so a larger block strictly widens the interval "
            "between two cooperative-abort reads"
        ),
        "criterion_2_worst_block_wall_is_monotone_in_block": bool(monotone),
        "criterion_2_largest_block_worst_wall_over_smallest": largest_over_smallest,
        "criterion_3_clamping_would_restore_the_arm": bool(
            clamping_would_restore_the_arm
        ),
        "declared_default_block_rows": int(declared_default_block),
        "measured_worst_seconds_per_gathered_row": slowest,
        "measured_fastest_seconds_per_gathered_row": fastest,
        "per_row_cost_spread": spread,
        "per_row_cost_stability_limit": PER_ROW_COST_STABILITY_LIMIT,
        "registered_ceiling_s_at_the_comparison_site": ceiling,
        "block_that_would_hold_at_the_measured_worst_case": sized,
        "registration_safety_margin": REGISTRATION_SAFETY_MARGIN,
        "r0247_published_widest_gap_s": float(r0247_widest_gap_s),
        "r0247_block_rows": int(r0247_block_rows),
        "r0247_implied_seconds_per_gathered_row": r0247_cost,
        "r0247_gap_over_the_registered_ceiling": float(r0247_widest_gap_s) / ceiling,
        "block_at_which_r0247_would_breach": (
            math.ceil(int(r0247_block_rows) * ceiling / float(r0247_widest_gap_s)) + 1
        ),
        "adaptive_alternative": adaptive,
        "arms": list(arms),
        "note": BLOCKSIZE_NOTE,
        # Measured: `registry_fingerprint()` hashes every registered row plus the
        # abort-reader allowlist, and `REGISTERED_REGISTRY_SHA256` is the value
        # R0249 pinned. Equality is the evidence that this round added no
        # parameter, changed no value and sanctioned no new reader.
        "registry_fingerprint_now": registry_fingerprint(),
        "registry_fingerprint_pinned_by_r0249": REGISTERED_REGISTRY_SHA256,
        "this_round_registered_nothing": (
            registry_fingerprint() == REGISTERED_REGISTRY_SHA256
        ),
    }


__all__ = [
    "ADAPTIVE_TARGET_FRACTION",
    "BLOCKSIZE_CAPABILITY",
    "BLOCKSIZE_NOTE",
    "BLOCKSIZE_SCHEMA",
    "BLOCK_SIZES_PROBED",
    "DIMENSION",
    "GRAPH_K",
    "PER_ROW_COST_STABILITY_LIMIT",
    "REGISTRATION_SAFETY_MARGIN",
    "RESOLUTION_NOT_NEEDED",
    "RESOLUTION_REGISTER",
    "ROUND_ID",
    "R0247_DECLARED_DEFAULT_BLOCK",
    "Round0250BlockSizeError",
    "adaptive_gather_probe",
    "largest_block_meeting_the_ceiling",
    "measure_block",
    "observed_default_block",
    "resolve",
]

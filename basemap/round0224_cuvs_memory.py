"""Frozen contract for R0224 — measure cuVS memory with an instrument that sees it.

Review 0220-01 threw out R0220's memory evidence, and was right to. Peak GPU was
**byte-identical** (`2,625,634,304`) across intermediate degrees 48/96/128
despite a 2.67x spread in that parameter, because `nvidia-smi` polled from the
parent cannot resolve an allocation inside the child. The `0.104` GPU-h headline
was then fitted on `igd96`, a setting the round's own memory argument rules out
at 100M, and compared against another projection as though the ratio were
measured.

This round replaces the instrument and the argument.

**Four instruments run at once, and the round's first job is to say which of them
can see intermediate degree at all:**

* `rmm_peak` — RMM statistics adaptor installed over the CUDA memory resource
  *before* `cuvs` is imported. cuVS allocates through
  `rmm::mr::get_current_device_resource`, so a device allocation cannot hide
  from this unless cuVS installs its own pool, which the receipt records.
* `device_peak_sampled` — `cudaMemGetInfo` at 5 ms inside the build process.
  Allocator-agnostic: it measures the device, not a bookkeeper.
* `host_peak_sampled` / `host_vmhwm` — `/proc/self/statm` at 5 ms and `VmHWM`,
  in a **fresh process per build**. This matters because RAFT/cuVS nn-descent
  holds its intermediate graph in *host* memory; if intermediate degree costs
  anything, this is where it lands.
* `nvidia_smi_peak` — R0220's instrument, polled from the parent, kept
  deliberately as a **control**.

`SENSITIVITY_RULE` is registered in advance: an instrument is *sensitive* iff its
peak differs across `igd` at fixed `N`. **If no instrument is sensitive, that is
the round's finding and no memory projection is published** — a second
meaningless fit is the one outcome this round refuses to produce.

**Projections are labelled, never compared to each other.** A 100M number is
emitted only for settings that fit, only from a fit whose measured range and
extrapolation factor are stated beside it, and never as a ratio against another
projection. That specific error is why R0220 was downgraded.

**Both budgets bind.** A 100M x 384 fp32 substrate is `153.6 GB` against this
box's `123 GiB` of RAM and `32 GiB` of VRAM, so the round evaluates the device
budget *and* the host budget, and a setting "fits" only if it clears both.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


ROUND_ID = "0224"

SUBSTRATE_CAPABILITY = "minilm-mixed-16m-benchmark-substrate-v1"
SWEEP_CAPABILITY = "cuvs-nn-descent-memory-and-wall-scaling-v1"
SUBSTRATE_SCHEMA = "round0224-minilm-mixed-16m-benchmark-substrate-v1"
SWEEP_SCHEMA = "round0224-cuvs-nn-descent-memory-scaling-v1"

DIMENSION = 384
GRAPH_K = 15

#: The benchmark substrate. R0216's selection law with the composition shares
#: scaled by 8, assembled once, then written in a **seeded global row
#: permutation** so that any prefix is a uniform subsample of the whole rather
#: than one corpus block. That is the one deliberate departure from R0216's
#: layout and it exists so the N-sweep does not confound scale with composition.
#: These are BENCHMARK bytes: they seal no training capability and no map.
BENCHMARK_ROWS = 16_000_000
BENCHMARK_SELECTION_SEED = 224
BENCHMARK_SHUFFLE_SEED = 2_240_224
BENCHMARK_COMPOSITION_SCALE = 8
BENCHMARK_NOTE = (
    "benchmark substrate: R0216's per-corpus uniform-over-all-rows selection "
    "law with replacement rounds and the >=99.9% shard-span assertion, "
    "composition shares scaled x8, then a seeded global row permutation so "
    "every prefix is a uniform subsample. No training capability is sealed on "
    "these bytes."
)
#: Minimum shard span per corpus. R0216's assertion, unchanged: the defect it
#: guards against (a leading prefix of each corpus) is invisible in the output.
SHARD_SPAN_FLOOR = 0.999
#: Realized composition in a prefix is binomial, not exact, so a prefix is
#: admissible when every corpus share lands within this of its target.
PREFIX_SHARE_TOLERANCE = 0.01

#: The sweep. `graph_degree` and `max_iterations` are held FIXED so that
#: `intermediate_graph_degree` is the only moving parameter — R0220 varied all
#: three at once, which is why its curves could not attribute anything.
SWEEP_GRAPH_DEGREE = 32
SWEEP_MAX_ITERATIONS = 20
SWEEP_METRIC = "sqeuclidean"
SWEEP_INTERMEDIATE_DEGREES: tuple[int, ...] = (48, 96, 128)
SWEEP_ROWS: tuple[int, ...] = (2_000_000, 4_000_000, 8_000_000, 16_000_000)
SAMPLE_INTERVAL_S = 0.005

#: The dataset-residency probe, run at the smallest N only: does the builder
#: accept a memmap without materializing it in host RAM? A 100M x 384 fp32
#: substrate cannot be a resident host array on this box, so the answer bounds
#: the top rung regardless of what the device does.
RESIDENCY_PROBE_ROWS = 2_000_000
DATASET_MODES: tuple[str, ...] = ("materialize", "memmap")

PROJECTION_ROWS = 100_000_000
#: 100M x 384 x 4 bytes. Stated as a constant because it is the number that
#: decides the top rung before any builder parameter is chosen.
PROJECTION_SUBSTRATE_BYTES = PROJECTION_ROWS * DIMENSION * 4
#: The RTX 5090's 32 GiB, and this box's 123 GiB of RAM. Both are re-read at
#: runtime from the device and `/proc/meminfo`; these are the registered
#: expectations, and a material disagreement is reported rather than ignored.
REGISTERED_DEVICE_TOTAL_BYTES = 34_359_738_368
REGISTERED_HOST_TOTAL_BYTES = 132_000_000_000
BUDGET_TOLERANCE = 0.05

#: Instruments, and what each is for.
INSTRUMENTS: tuple[str, ...] = (
    "rmm_peak_bytes",
    "device_peak_sampled_bytes",
    "host_peak_sampled_bytes",
    "host_vmhwm_bytes",
    "nvidia_smi_peak_bytes",
)
DEVICE_INSTRUMENTS: tuple[str, ...] = (
    "rmm_peak_bytes",
    "device_peak_sampled_bytes",
    "nvidia_smi_peak_bytes",
)
HOST_INSTRUMENTS: tuple[str, ...] = ("host_peak_sampled_bytes", "host_vmhwm_bytes")
CONTROL_INSTRUMENT = "nvidia_smi_peak_bytes"
CONTROL_NOTE = (
    "nvidia_smi_peak_bytes is R0220's instrument, polled from the parent "
    "against the child pid. It is carried as a control, not as evidence."
)

#: An instrument is sensitive to intermediate degree iff its peak differs across
#: igd at fixed N by more than this fraction of its own value. Registered in
#: advance so 'sensitive' cannot be decided after seeing the numbers.
SENSITIVITY_RELATIVE_THRESHOLD = 0.01
SENSITIVITY_RULE = (
    "an instrument is sensitive to intermediate_graph_degree iff, at some fixed "
    "N, its peak spread across igd exceeds "
    f"{SENSITIVITY_RELATIVE_THRESHOLD:.0%} of its own maximum. If NO instrument "
    "is sensitive, the round publishes that as its finding and emits no memory "
    "projection: a second unfalsifiable fit is worse than no fit."
)

#: Projections are labelled and never divided by each other.
PROJECTION_DISCIPLINE = (
    "every 100M number in this round is a PROJECTION. Each carries the measured "
    "N range it was fitted on and the extrapolation factor from the largest "
    "measured N. No projection is divided by another projection, and no ratio "
    "of projections is reported as a speedup (review-0220-01)."
)

GPU_HOURS_CAP = 2.5
#: A single build that has not finished in this long has stopped being a
#: measurement and starts being a budget risk.
BUILD_TIMEOUT_S = 3_600.0
HOST_RSS_LIMIT_GIB = 96.0


class Round0224Error(RuntimeError):
    """The registered R0224 memory-measurement contract changed."""


def sweep_settings() -> tuple[dict[str, Any], ...]:
    """Every (N, igd) cell of the registered matrix, in run order."""
    out: list[dict[str, Any]] = []
    for rows in SWEEP_ROWS:
        for igd in SWEEP_INTERMEDIATE_DEGREES:
            out.append({
                "id": f"nnd-gd{SWEEP_GRAPH_DEGREE}-igd{igd}-it{SWEEP_MAX_ITERATIONS}"
                f"-n{rows}",
                "rows": int(rows),
                "graph_degree": SWEEP_GRAPH_DEGREE,
                "intermediate_graph_degree": int(igd),
                "max_iterations": SWEEP_MAX_ITERATIONS,
                "metric": SWEEP_METRIC,
                "dataset_mode": "materialize",
            })
    return tuple(out)


def residency_probe_settings() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "id": f"residency-{mode}-igd{SWEEP_INTERMEDIATE_DEGREES[0]}"
            f"-n{RESIDENCY_PROBE_ROWS}",
            "rows": RESIDENCY_PROBE_ROWS,
            "graph_degree": SWEEP_GRAPH_DEGREE,
            "intermediate_graph_degree": SWEEP_INTERMEDIATE_DEGREES[0],
            "max_iterations": SWEEP_MAX_ITERATIONS,
            "metric": SWEEP_METRIC,
            "dataset_mode": mode,
        }
        for mode in DATASET_MODES
    )


def validate_prefix_composition(
    *, shares: Mapping[str, float], targets: Mapping[str, float]
) -> dict[str, Any]:
    """A prefix is admissible when every corpus share is near its target."""
    if set(shares) != set(targets):
        raise Round0224Error("R0224 prefix composition covers the wrong corpora")
    deviations = {
        name: float(shares[name]) - float(targets[name]) for name in sorted(targets)
    }
    worst = max(abs(value) for value in deviations.values())
    if worst > PREFIX_SHARE_TOLERANCE:
        raise Round0224Error(
            f"R0224 prefix composition deviates by {worst:.4f}, beyond the "
            f"registered {PREFIX_SHARE_TOLERANCE}"
        )
    return {
        "shares": {name: float(value) for name, value in shares.items()},
        "targets": {name: float(value) for name, value in targets.items()},
        "deviations": deviations,
        "worst_absolute_deviation": worst,
        "tolerance": PREFIX_SHARE_TOLERANCE,
    }


def instrument_sensitivity(
    measurements: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Which instruments can see intermediate degree at all?

    Computed from measured builds only. This runs before any fit, and the
    round's projections are gated on its answer.
    """
    by_rows: dict[int, list[Mapping[str, Any]]] = {}
    for item in measurements:
        if not item.get("fit"):
            continue
        by_rows.setdefault(int(item["rows"]), []).append(item)

    report: dict[str, Any] = {}
    for instrument in INSTRUMENTS:
        per_n: dict[str, Any] = {}
        sensitive = False
        for rows, group in sorted(by_rows.items()):
            values = {
                int(item["intermediate_graph_degree"]): float(item.get(instrument) or 0.0)
                for item in group
            }
            if len(values) < 2:
                continue
            largest = max(values.values())
            spread = largest - min(values.values())
            relative = spread / largest if largest > 0 else 0.0
            here = relative > SENSITIVITY_RELATIVE_THRESHOLD
            sensitive = sensitive or here
            per_n[str(rows)] = {
                "by_intermediate_degree": {
                    str(key): value for key, value in sorted(values.items())
                },
                "spread_bytes": spread,
                "relative_spread": relative,
                "sensitive_here": here,
            }
        report[instrument] = {
            "role": (
                "control (R0220's instrument)"
                if instrument == CONTROL_INSTRUMENT
                else "device" if instrument in DEVICE_INSTRUMENTS else "host"
            ),
            "per_n": per_n,
            "sensitive_to_intermediate_degree": sensitive,
            "threshold": SENSITIVITY_RELATIVE_THRESHOLD,
        }
    sensitive_names = sorted(
        name for name, cell in report.items()
        if cell["sensitive_to_intermediate_degree"]
    )
    return {
        "rule": SENSITIVITY_RULE,
        "instruments": report,
        "sensitive_instruments": sensitive_names,
        "any_instrument_sensitive": bool(sensitive_names),
        "control_instrument": CONTROL_INSTRUMENT,
        "control_note": CONTROL_NOTE,
        "control_is_blind": not report[CONTROL_INSTRUMENT][
            "sensitive_to_intermediate_degree"
        ],
    }


def linear_fit(
    sizes: Sequence[int], values: Sequence[float]
) -> dict[str, Any]:
    """Least-squares `y = a + b*N`, with R^2. Memory is expected to be affine."""
    x = [float(value) for value in sizes]
    y = [float(value) for value in values]
    if len(x) != len(y) or len(x) < 2:
        raise Round0224Error("R0224 linear fit needs at least two matched points")
    if any(not math.isfinite(value) for value in x + y):
        raise Round0224Error("R0224 linear fit needs finite points")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator <= 0:
        raise Round0224Error("R0224 linear fit needs distinct sizes")
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(x, y))
    total = sum((b - mean_y) ** 2 for b in y)
    return {
        "model": "y = a + b*N",
        "intercept_a": intercept,
        "slope_b_bytes_per_row": slope,
        "r_squared": (1.0 - residual / total) if total > 0 else 1.0,
        "fitted_rows_min": int(min(sizes)),
        "fitted_rows_max": int(max(sizes)),
        "points": [
            {"rows": int(a), "value": float(b)} for a, b in zip(sizes, values)
        ],
    }


def power_law_fit(sizes: Sequence[int], seconds: Sequence[float]) -> dict[str, Any]:
    """Least-squares `t = a * N**b` in log space, with R^2."""
    x = [float(value) for value in sizes]
    t = [float(value) for value in seconds]
    if len(x) != len(t) or len(x) < 2:
        raise Round0224Error("R0224 power-law fit needs at least two matched points")
    if any(value <= 0 for value in x + t):
        raise Round0224Error("R0224 power-law fit needs strictly positive points")
    log_x = [math.log(value) for value in x]
    log_t = [math.log(value) for value in t]
    mean_x = statistics.fmean(log_x)
    mean_t = statistics.fmean(log_t)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator <= 0:
        raise Round0224Error("R0224 power-law fit needs distinct sizes")
    slope = sum(
        (a - mean_x) * (b - mean_t) for a, b in zip(log_x, log_t)
    ) / denominator
    intercept = mean_t - slope * mean_x
    residual = sum(
        (b - (intercept + slope * a)) ** 2 for a, b in zip(log_x, log_t)
    )
    total = sum((b - mean_t) ** 2 for b in log_t)
    return {
        "model": "t = a * N**b",
        "exponent_b": slope,
        "coefficient_a": math.exp(intercept),
        "r_squared": (1.0 - residual / total) if total > 0 else 1.0,
        "fitted_rows_min": int(min(sizes)),
        "fitted_rows_max": int(max(sizes)),
        "points": [
            {"rows": int(a), "seconds": float(b)} for a, b in zip(sizes, seconds)
        ],
    }


def project_linear(
    fit: Mapping[str, Any], *, rows: int = PROJECTION_ROWS
) -> dict[str, Any]:
    value = float(fit["intercept_a"]) + float(fit["slope_b_bytes_per_row"]) * float(rows)
    return {
        "is_measurement": False,
        "kind": "projection",
        "rows": int(rows),
        "projected_bytes": value,
        "projected_gib": value / (1024 ** 3),
        "fitted_rows_min": int(fit["fitted_rows_min"]),
        "fitted_rows_max": int(fit["fitted_rows_max"]),
        "extrapolation_factor": float(rows) / float(fit["fitted_rows_max"]),
        "r_squared": float(fit["r_squared"]),
        "discipline": PROJECTION_DISCIPLINE,
    }


def project_wall(
    fit: Mapping[str, Any], *, rows: int = PROJECTION_ROWS
) -> dict[str, Any]:
    seconds = float(fit["coefficient_a"]) * float(rows) ** float(fit["exponent_b"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise Round0224Error("R0224 projected wall is not a finite positive time")
    return {
        "is_measurement": False,
        "kind": "projection",
        "rows": int(rows),
        "projected_seconds": seconds,
        "projected_gpu_hours": seconds / 3600.0,
        "exponent_b": float(fit["exponent_b"]),
        "fitted_rows_min": int(fit["fitted_rows_min"]),
        "fitted_rows_max": int(fit["fitted_rows_max"]),
        "extrapolation_factor": float(rows) / float(fit["fitted_rows_max"]),
        "r_squared": float(fit["r_squared"]),
        "discipline": PROJECTION_DISCIPLINE,
        "compared_to_another_projection": False,
    }


def budget_verdict(
    *,
    intermediate_degree: int,
    device_projection: Mapping[str, Any],
    host_projection: Mapping[str, Any],
    device_budget_bytes: int,
    host_budget_bytes: int,
) -> dict[str, Any]:
    """Does this setting fit 100M, on the device AND on the host?"""
    device_bytes = float(device_projection["projected_bytes"])
    host_bytes = float(host_projection["projected_bytes"])
    device_fits = device_bytes <= float(device_budget_bytes)
    host_fits = host_bytes <= float(host_budget_bytes)
    return {
        "intermediate_graph_degree": int(intermediate_degree),
        "device": {
            **dict(device_projection),
            "budget_bytes": int(device_budget_bytes),
            "budget_gib": float(device_budget_bytes) / (1024 ** 3),
            "fits": device_fits,
            "headroom_bytes": float(device_budget_bytes) - device_bytes,
        },
        "host": {
            **dict(host_projection),
            "budget_bytes": int(host_budget_bytes),
            "budget_gib": float(host_budget_bytes) / (1024 ** 3),
            "fits": host_fits,
            "headroom_bytes": float(host_budget_bytes) - host_bytes,
        },
        "fits_100m": bool(device_fits and host_fits),
        "binding_constraint": (
            "device" if not device_fits and host_fits
            else "host" if device_fits and not host_fits
            else "both" if not device_fits and not host_fits
            else "none"
        ),
    }


def summarize_sweep(
    *,
    measurements: Sequence[Mapping[str, Any]],
    device_total_bytes: int,
    host_total_bytes: int,
) -> dict[str, Any]:
    """The whole scientific payload: sensitivity, fits, verdicts, projections."""
    sensitivity = instrument_sensitivity(measurements)
    fitted = [item for item in measurements if item.get("fit")]
    failed = [item for item in measurements if not item.get("fit")]
    largest_fitting = {}
    for igd in SWEEP_INTERMEDIATE_DEGREES:
        rows = [
            int(item["rows"])
            for item in fitted
            if int(item["intermediate_graph_degree"]) == igd
        ]
        largest_fitting[str(igd)] = max(rows) if rows else None

    payload: dict[str, Any] = {
        "sensitivity": sensitivity,
        "measured_cells": len(fitted),
        "failed_cells": [
            {
                "rows": int(item["rows"]),
                "intermediate_graph_degree": int(item["intermediate_graph_degree"]),
                "oom": bool(item.get("oom")),
                "error_type": item.get("error_type"),
            }
            for item in failed
        ],
        "largest_measured_rows_that_fit_by_igd": largest_fitting,
        "device_total_bytes": int(device_total_bytes),
        "host_total_bytes": int(host_total_bytes),
        "projection_substrate_bytes_at_100m": PROJECTION_SUBSTRATE_BYTES,
        "projection_discipline": PROJECTION_DISCIPLINE,
    }

    if not sensitivity["any_instrument_sensitive"]:
        payload.update({
            "projections_emitted": False,
            "finding": (
                "No instrument resolved intermediate_graph_degree. The round "
                "publishes that rather than another unfalsifiable fit."
            ),
            "per_igd": {},
            "settings_that_fit_100m": [],
            "no_setting_fits_100m": None,
        })
        return payload

    per_igd: dict[str, Any] = {}
    fitting_settings: list[int] = []
    for igd in SWEEP_INTERMEDIATE_DEGREES:
        cells = sorted(
            (item for item in fitted if int(item["intermediate_graph_degree"]) == igd),
            key=lambda item: int(item["rows"]),
        )
        if len(cells) < 2:
            per_igd[str(igd)] = {
                "measured_cells": len(cells),
                "fittable": False,
                "reason": "fewer than two measured sizes",
            }
            continue
        sizes = [int(item["rows"]) for item in cells]
        device_fit = linear_fit(
            sizes, [float(item["device_peak_sampled_bytes"]) for item in cells]
        )
        host_fit = linear_fit(
            sizes, [float(item["host_peak_sampled_bytes"]) for item in cells]
        )
        wall_fit = power_law_fit(
            sizes, [float(item["builder_seconds"]) for item in cells]
        )
        verdict = budget_verdict(
            intermediate_degree=igd,
            device_projection=project_linear(device_fit),
            host_projection=project_linear(host_fit),
            device_budget_bytes=device_total_bytes,
            host_budget_bytes=host_total_bytes,
        )
        cell: dict[str, Any] = {
            "measured_cells": len(cells),
            "fittable": True,
            "measured": [
                {
                    "rows": int(item["rows"]),
                    "builder_seconds": float(item["builder_seconds"]),
                    "device_peak_sampled_bytes": int(item["device_peak_sampled_bytes"]),
                    "host_peak_sampled_bytes": int(item["host_peak_sampled_bytes"]),
                    "host_vmhwm_bytes": int(item["host_vmhwm_bytes"]),
                    "rmm_peak_bytes": int(item["rmm_peak_bytes"]),
                    "nvidia_smi_peak_bytes": int(item.get("nvidia_smi_peak_bytes") or 0),
                }
                for item in cells
            ],
            "device_memory_fit": device_fit,
            "host_memory_fit": host_fit,
            "wall_fit": wall_fit,
            "budget_verdict": verdict,
        }
        #: A wall projection is emitted ONLY for a setting that fits. Projecting
        #: the wall of a build that cannot run is the error R0220 made.
        if verdict["fits_100m"]:
            cell["wall_projection_100m"] = project_wall(wall_fit)
            fitting_settings.append(int(igd))
        else:
            cell["wall_projection_100m"] = None
            cell["wall_projection_withheld_because"] = (
                "this setting does not fit 100M within the measured budgets, so a "
                "wall projection for it would describe a build that cannot run"
            )
        per_igd[str(igd)] = cell

    payload.update({
        "projections_emitted": True,
        "per_igd": per_igd,
        "settings_that_fit_100m": sorted(fitting_settings),
        "no_setting_fits_100m": not fitting_settings,
    })
    if not fitting_settings:
        payload["finding"] = (
            "No measured intermediate_graph_degree setting fits a 100,000,000-row "
            "build within this box's device and host budgets. Phase 2's top rung "
            "needs a batched or out-of-core builder; a single-shot cuVS "
            "nn-descent call is not a candidate at 100M on this hardware."
        )
    return payload


__all__ = [
    "BENCHMARK_COMPOSITION_SCALE",
    "BENCHMARK_NOTE",
    "BENCHMARK_ROWS",
    "BENCHMARK_SELECTION_SEED",
    "BENCHMARK_SHUFFLE_SEED",
    "BUDGET_TOLERANCE",
    "BUILD_TIMEOUT_S",
    "CONTROL_INSTRUMENT",
    "CONTROL_NOTE",
    "DATASET_MODES",
    "DEVICE_INSTRUMENTS",
    "DIMENSION",
    "GPU_HOURS_CAP",
    "GRAPH_K",
    "HOST_INSTRUMENTS",
    "HOST_RSS_LIMIT_GIB",
    "INSTRUMENTS",
    "PREFIX_SHARE_TOLERANCE",
    "PROJECTION_DISCIPLINE",
    "PROJECTION_ROWS",
    "PROJECTION_SUBSTRATE_BYTES",
    "REGISTERED_DEVICE_TOTAL_BYTES",
    "REGISTERED_HOST_TOTAL_BYTES",
    "RESIDENCY_PROBE_ROWS",
    "ROUND_ID",
    "Round0224Error",
    "SAMPLE_INTERVAL_S",
    "SENSITIVITY_RELATIVE_THRESHOLD",
    "SENSITIVITY_RULE",
    "SHARD_SPAN_FLOOR",
    "SUBSTRATE_CAPABILITY",
    "SUBSTRATE_SCHEMA",
    "SWEEP_CAPABILITY",
    "SWEEP_GRAPH_DEGREE",
    "SWEEP_INTERMEDIATE_DEGREES",
    "SWEEP_MAX_ITERATIONS",
    "SWEEP_METRIC",
    "SWEEP_ROWS",
    "SWEEP_SCHEMA",
    "budget_verdict",
    "instrument_sensitivity",
    "linear_fit",
    "power_law_fit",
    "project_linear",
    "project_wall",
    "residency_probe_settings",
    "summarize_sweep",
    "sweep_settings",
    "validate_prefix_composition",
]

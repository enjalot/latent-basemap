"""R0224 contract tests: the sensitivity gate, the fits, and the projections.

The tests that matter here are not the arithmetic ones. They are:

* `test_a_blind_instrument_is_detected` — R0220's failure mode, reproduced as a
  fixture: an instrument whose peak is byte-identical across intermediate
  degrees must be reported as blind.
* `test_no_sensitive_instrument_means_no_projection` — the registered refusal.
  If nothing can see intermediate degree, the round publishes that and emits no
  fit, rather than a second unfalsifiable one.
* `test_wall_projection_is_withheld_for_a_setting_that_does_not_fit` — the
  specific error that downgraded R0220: projecting the wall of a build that
  cannot run, and comparing projections to each other.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from basemap.round0224_cuvs_memory import (
    BENCHMARK_COMPOSITION_SCALE,
    BENCHMARK_ROWS,
    CONTROL_INSTRUMENT,
    DIMENSION,
    PROJECTION_ROWS,
    PROJECTION_SUBSTRATE_BYTES,
    REGISTERED_DEVICE_TOTAL_BYTES,
    Round0224Error,
    SWEEP_GRAPH_DEGREE,
    SWEEP_INTERMEDIATE_DEGREES,
    SWEEP_MAX_ITERATIONS,
    SWEEP_ROWS,
    budget_verdict,
    instrument_sensitivity,
    linear_fit,
    power_law_fit,
    project_linear,
    project_wall,
    residency_probe_settings,
    summarize_sweep,
    sweep_settings,
    validate_prefix_composition,
)
from basemap.round0216_minilm_2m_substrate import COMPOSITION


DEVICE_TOTAL = REGISTERED_DEVICE_TOTAL_BYTES
HOST_TOTAL = 132_000_000_000


def _cell(
    rows: int,
    igd: int,
    *,
    device_slope: float = 1_540.0,
    host_slope: float = 1_600.0,
    host_igd_bytes: float = 8.0,
    blind_device: bool = True,
    seconds_per_million: float = 4.5,
) -> dict[str, Any]:
    device = 600_000_000 + device_slope * rows + (0.0 if blind_device else 8.0 * igd * rows)
    host = 500_000_000 + host_slope * rows + host_igd_bytes * igd * rows
    return {
        "rows": rows,
        "intermediate_graph_degree": igd,
        "fit": True,
        "builder_seconds": seconds_per_million * rows / 1_000_000.0,
        "device_peak_sampled_bytes": device,
        "host_peak_sampled_bytes": host,
        "host_vmhwm_bytes": host + 1_000_000,
        "rmm_peak_bytes": 524_000_000,
        "device_peak_bytes": max(device, 524_000_000),
        CONTROL_INSTRUMENT: 2_625_634_304,
    }


def _matrix(**kwargs: Any) -> list[dict[str, Any]]:
    return [
        _cell(rows, igd, **kwargs)
        for rows in (2_000_000, 4_000_000, 8_000_000)
        for igd in SWEEP_INTERMEDIATE_DEGREES
    ]


def test_the_top_rung_is_memmap_fed_and_the_rest_materialize() -> None:
    from basemap.round0224_cuvs_memory import DATASET_MODE_BY_ROWS

    assert DATASET_MODE_BY_ROWS[16_000_000] == "memmap"
    assert set(DATASET_MODE_BY_ROWS) == set(SWEEP_ROWS)
    modes = {item["rows"]: item["dataset_mode"] for item in sweep_settings()}
    assert modes[16_000_000] == "memmap"
    assert modes[2_000_000] == "materialize"


def test_device_budget_instrument_dominates_every_device_instrument() -> None:
    """The budget figure is the max over all three device instruments.

    Addendum 2 adds `device_wide_peak_bytes` — the parent-side nvidia-smi
    reading — and it is the term that carries the verdict, because it is the
    only one that survives both failure modes: GIL starvation of the
    in-process sampler, and cuVS allocating outside RMM's device resource.
    """
    from basemap.round0224_cuvs_memory import (
        DEVICE_BUDGET_INSTRUMENT,
        DEVICE_BUDGET_NOTE,
        DEVICE_INSTRUMENTS,
    )

    assert DEVICE_BUDGET_INSTRUMENT == "device_peak_bytes"
    assert "LOWER bound" in DEVICE_BUDGET_NOTE
    assert "device_wide_peak_bytes" in DEVICE_BUDGET_NOTE
    assert "device_wide_peak_bytes" in DEVICE_INSTRUMENTS
    cell = _cell(8_000_000, 48)
    assert cell["device_peak_bytes"] >= cell["rmm_peak_bytes"]
    assert cell["device_peak_bytes"] >= cell["device_peak_sampled_bytes"]


def test_sweep_matrix_holds_everything_but_intermediate_degree_fixed() -> None:
    settings = sweep_settings()
    assert len(settings) == len(SWEEP_ROWS) * len(SWEEP_INTERMEDIATE_DEGREES)
    assert {item["graph_degree"] for item in settings} == {SWEEP_GRAPH_DEGREE}
    assert {item["max_iterations"] for item in settings} == {SWEEP_MAX_ITERATIONS}
    assert {item["intermediate_graph_degree"] for item in settings} == set(
        SWEEP_INTERMEDIATE_DEGREES
    )
    assert len({item["id"] for item in settings}) == len(settings)
    probes = residency_probe_settings()
    assert {item["dataset_mode"] for item in probes} == {"materialize", "memmap"}


def test_a_blind_instrument_is_detected() -> None:
    """R0220's exact failure: a byte-identical peak across igd is blindness."""
    report = instrument_sensitivity(_matrix())
    control = report["instruments"][CONTROL_INSTRUMENT]
    assert control["sensitive_to_intermediate_degree"] is False
    assert report["control_is_blind"] is True
    assert report["instruments"]["rmm_peak_bytes"][
        "sensitive_to_intermediate_degree"
    ] is False
    assert report["instruments"]["host_peak_sampled_bytes"][
        "sensitive_to_intermediate_degree"
    ] is True
    assert "host_peak_sampled_bytes" in report["sensitive_instruments"]
    assert report["any_instrument_sensitive"] is True


def test_no_sensitive_instrument_means_no_projection() -> None:
    blind = _matrix(host_igd_bytes=0.0)
    report = instrument_sensitivity(blind)
    assert report["any_instrument_sensitive"] is False
    summary = summarize_sweep(
        measurements=blind,
        device_total_bytes=DEVICE_TOTAL,
        host_total_bytes=HOST_TOTAL,
    )
    assert summary["projections_emitted"] is False
    assert summary["per_igd"] == {}
    assert "unfalsifiable" in summary["finding"]


def test_linear_and_power_fits_recover_a_known_law() -> None:
    sizes = [2_000_000, 4_000_000, 8_000_000, 16_000_000]
    values = [1_000.0 + 3.0 * size for size in sizes]
    fit = linear_fit(sizes, values)
    assert abs(fit["slope_b_bytes_per_row"] - 3.0) < 1e-6
    assert fit["r_squared"] > 0.999999
    seconds = [1.5e-6 * size ** 1.0 for size in sizes]
    wall = power_law_fit(sizes, seconds)
    assert abs(wall["exponent_b"] - 1.0) < 1e-6
    with pytest.raises(Round0224Error):
        linear_fit([1_000_000], [1.0])
    with pytest.raises(Round0224Error):
        power_law_fit(sizes, [0.0] * len(sizes))


def test_projections_are_labelled_and_carry_their_extrapolation_factor() -> None:
    fit = linear_fit([2_000_000, 8_000_000], [1.0e9, 4.0e9])
    projection = project_linear(fit)
    assert projection["is_measurement"] is False
    assert projection["kind"] == "projection"
    assert projection["rows"] == PROJECTION_ROWS
    assert abs(projection["extrapolation_factor"] - 12.5) < 1e-9
    assert projection["fitted_rows_max"] == 8_000_000
    wall = project_wall(power_law_fit([2_000_000, 8_000_000], [9.0, 36.0]))
    assert wall["compared_to_another_projection"] is False
    assert wall["projected_gpu_hours"] > 0


def test_budget_verdict_binds_both_device_and_host() -> None:
    tiny = project_linear(linear_fit([2_000_000, 8_000_000], [1.0e8, 2.0e8]))
    huge = project_linear(linear_fit([2_000_000, 8_000_000], [1.0e11, 4.0e11]))
    both_fit = budget_verdict(
        intermediate_degree=48,
        device_projection=tiny,
        host_projection=tiny,
        device_budget_bytes=DEVICE_TOTAL,
        host_budget_bytes=HOST_TOTAL,
    )
    assert both_fit["fits_100m"] is True
    assert both_fit["binding_constraint"] == "none"
    host_bound = budget_verdict(
        intermediate_degree=48,
        device_projection=tiny,
        host_projection=huge,
        device_budget_bytes=DEVICE_TOTAL,
        host_budget_bytes=HOST_TOTAL,
    )
    assert host_bound["fits_100m"] is False
    assert host_bound["binding_constraint"] == "host"
    device_bound = budget_verdict(
        intermediate_degree=48,
        device_projection=huge,
        host_projection=tiny,
        device_budget_bytes=DEVICE_TOTAL,
        host_budget_bytes=HOST_TOTAL,
    )
    assert device_bound["binding_constraint"] == "device"


def test_wall_projection_is_withheld_for_a_setting_that_does_not_fit() -> None:
    """The R0220 error: never project the wall of a build that cannot run."""
    measurements = _matrix(blind_device=False)
    summary = summarize_sweep(
        measurements=measurements,
        device_total_bytes=DEVICE_TOTAL,
        host_total_bytes=HOST_TOTAL,
    )
    assert summary["projections_emitted"] is True
    for igd in SWEEP_INTERMEDIATE_DEGREES:
        cell = summary["per_igd"][str(igd)]
        if cell["budget_verdict"]["fits_100m"]:
            assert cell["wall_projection_100m"] is not None
        else:
            assert cell["wall_projection_100m"] is None
            assert "cannot run" in cell["wall_projection_withheld_because"]
    # With a real device dependence at these slopes nothing fits 100M, and the
    # round is required to say so plainly.
    assert summary["no_setting_fits_100m"] is True
    assert "out-of-core" in summary["finding"]


def test_a_failed_cell_is_a_measurement_not_a_crash() -> None:
    measurements = _matrix()
    measurements.append({
        "rows": 16_000_000,
        "intermediate_graph_degree": 128,
        "fit": False,
        "oom": True,
        "timed_out": False,
        "error_type": "MemoryError",
    })
    measurements.append({
        "rows": 16_000_000,
        "intermediate_graph_degree": 96,
        "fit": False,
        "oom": False,
        "timed_out": True,
        "error_type": "TimeoutExpired",
    })
    summary = summarize_sweep(
        measurements=measurements,
        device_total_bytes=DEVICE_TOTAL,
        host_total_bytes=HOST_TOTAL,
    )
    # `failed_cells` carries more fields since addendum 2 (refusals, watchdog
    # aborts, predictions), so the record is checked as a superset rather than
    # by exact equality. The assertion itself is unchanged.
    expected = {
        "rows": 16_000_000,
        "intermediate_graph_degree": 128,
        "oom": True,
        "timed_out": False,
        "error_type": "MemoryError",
    }
    assert any(
        expected.items() <= cell.items() for cell in summary["failed_cells"]
    ), summary["failed_cells"]
    assert len(summary["failed_cells"]) == 2
    assert summary["largest_measured_rows_that_fit_by_igd"]["128"] == 8_000_000
    timed = [c for c in summary["failed_cells"] if c["timed_out"]]
    assert len(timed) == 1 and timed[0]["intermediate_graph_degree"] == 96


def test_prefix_composition_tolerance_is_binomial_not_fixed() -> None:
    from basemap.round0224_cuvs_memory import prefix_share_tolerance

    targets = {name: rows / 2_000_000 for name, rows in COMPOSITION}
    # A fixed 0.01 would be ~29 binomial sd at 2M rows: vacuous where it matters.
    tight = prefix_share_tolerance(rows=2_000_000, target=0.4)
    loose = prefix_share_tolerance(rows=2_000, target=0.4)
    assert tight < 0.01 < loose
    assert prefix_share_tolerance(rows=10 ** 12, target=0.4) == 0.002
    ok = validate_prefix_composition(
        shares={name: value + 0.0005 for name, value in targets.items()},
        targets=targets,
        rows=2_000_000,
    )
    assert ok["rows"] == 2_000_000
    assert all(value > 0 for value in ok["tolerances"].values())
    with pytest.raises(Round0224Error):
        validate_prefix_composition(
            shares={name: value + 0.05 for name, value in targets.items()},
            targets=targets,
            rows=2_000_000,
        )
    with pytest.raises(Round0224Error):
        validate_prefix_composition(shares={"nope": 1.0}, targets=targets, rows=100)


def test_the_100m_substrate_does_not_fit_host_ram_and_the_constant_says_so() -> None:
    assert PROJECTION_SUBSTRATE_BYTES == PROJECTION_ROWS * DIMENSION * 4
    assert PROJECTION_SUBSTRATE_BYTES > HOST_TOTAL
    assert PROJECTION_SUBSTRATE_BYTES > REGISTERED_DEVICE_TOTAL_BYTES
    assert BENCHMARK_ROWS == 2_000_000 * BENCHMARK_COMPOSITION_SCALE
    assert sum(rows for _n, rows in COMPOSITION) * BENCHMARK_COMPOSITION_SCALE == (
        BENCHMARK_ROWS
    )
    assert max(SWEEP_ROWS) == BENCHMARK_ROWS
    assert all(math.isfinite(float(rows)) for rows in SWEEP_ROWS)

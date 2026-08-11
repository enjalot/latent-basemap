"""R0257 — R0217's training pipeline over a Phase 2 ladder rung.

`MiniLMMixedTrainingInput.__init__` asserts its substrate is exactly
`(2_000_000, 384)`. That assertion is correct for R0217 and wrong for a rung, so
this module subclasses it and asserts the **rung's own** cardinality instead.
Nothing else is overridden: the sampler, the endpoint gather, the RNG streams, the
execution stamp and `prepare_round0034_training` are R0217's, inherited unchanged.
The subclass adds no numerical behaviour, and a contract test asserts that the only
method it defines is `__init__`.

The full-population map validator and the memory prediction are likewise rung
variants of R0230's and R0255's 2M-shaped ones, built on the same generic
`validate_published_map`.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0217_minilm_2m_pipeline import (
    MiniLMHostFp32EndpointArray,
    MiniLMMixedTrainingInput,
)
from basemap.round0217_minilm_2m_seed_family import (
    DIMENSION,
    OUTPUT_DIMENSION,
    validate_published_map,
)
from basemap.round0257_rung_contract import (
    RUNG_ROWS,
    RUNG_SLUG,
    Round0257Error,
)

#: R0255's registered budgets, reused unchanged. They are host-anonymous and
#: device budgets, never RSS (the rule earned in R0226/R0233).
from basemap.round0255_seed_extension_n29 import (  # noqa: E402
    DEVICE_BUDGET_BYTES,
    HOST_ANON_BUDGET_BYTES,
    HOST_RSS_LIMIT_GIB,
)

#: Measured at 2M in R0217/R0221/R0230/R0250/R0255, byte for byte identical in
#: every one of those cells. Published here as a PREDICTION for this rung with a
#: registered check, never as a carried measurement: the feature residency is a
#: host memmap, so device peak should not move with N, and that proposition is
#: what the check tests.
MEASURED_PEAK_DEVICE_BYTES_AT_2M = 796_540_416
DEVICE_PEAK_PREDICTION_NOTE = (
    "PREDICTION, not a carried measurement. All fourteen R0255 cells peaked at "
    f"{MEASURED_PEAK_DEVICE_BYTES_AT_2M} device bytes at 2M. Feature residency is "
    "host-mmap-l2-normalized-fp32-substrate, so the device working set is the "
    "model plus one batch and should be N-independent; the host side grows "
    "3.07 GB -> 9.60 GB as page cache, which is file-backed and not charged to "
    "the anonymous budget. The registered check is the DEVICE_BUDGET_BYTES "
    "refusal, and the observed peak is published beside the prediction."
)


class RungMixedTrainingInput(MiniLMMixedTrainingInput):
    """R0217's training input with the rung's cardinality asserted instead of 2M."""

    def __init__(
        self,
        dataset: MiniLMHostFp32EndpointArray,
        graph: Mapping[str, Any],
        *,
        seed: int,
        rows: int = RUNG_ROWS,
    ) -> None:
        # Deliberately does NOT call super().__init__: its only difference is the
        # 2M geometry assertion, which is the one thing that must move.
        self.dataset = dataset
        self.graph = dict(graph)
        self.seed = int(seed)
        self.shape = dataset.shape
        self._last_sampler = None
        if self.shape != (int(rows), DIMENSION):
            raise Round0257Error(
                f"R0257 rung training input is {self.shape}, expected "
                f"({int(rows)}, {DIMENSION})"
            )
        if int(self.graph.get("n_nodes", -1)) != len(dataset):
            raise Round0257Error(
                "R0257 rung graph node count does not match the rung substrate"
            )


def validate_full_rung_map(coordinates: Any, *, rows: int = RUNG_ROWS) -> dict[str, Any]:
    """Every one of the rung's rows must project to a finite coordinate."""
    array = np.asarray(coordinates)
    rows = int(rows)
    if array.shape != (rows, OUTPUT_DIMENSION):
        raise Round0257Error(
            f"R0257 full-population transform produced {array.shape}, expected "
            f"({rows}, {OUTPUT_DIMENSION})"
        )
    finite = int(np.isfinite(array).all(axis=1).sum())
    if finite != rows:
        raise Round0257Error(
            f"R0257 full-population transform has {rows - finite} nonfinite rows"
        )
    published = validate_published_map(array)
    return {
        **published,
        "transform_rows": rows,
        "transform_rows_finite": finite,
        "full_population_finite": True,
    }


def predict_rung_footprint(seed: int) -> dict[str, Any]:
    """A labelled prediction with a registered check, per the standing rule."""
    return {
        "rung": RUNG_SLUG,
        "rows": RUNG_ROWS,
        "seed": int(seed),
        "predicted_peak_device_bytes": MEASURED_PEAK_DEVICE_BYTES_AT_2M,
        "prediction_basis": DEVICE_PEAK_PREDICTION_NOTE,
        "device_budget_bytes": DEVICE_BUDGET_BYTES,
        "host_anonymous_budget_bytes": HOST_ANON_BUDGET_BYTES,
        "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
        "substrate_bytes_host_page_cache": RUNG_ROWS * DIMENSION * 4,
        "guarded_on": "host ANONYMOUS bytes, never RSS (R0226/R0233)",
        "refused_a_priori": False,
    }


__all__ = [
    "DEVICE_BUDGET_BYTES",
    "DEVICE_PEAK_PREDICTION_NOTE",
    "HOST_ANON_BUDGET_BYTES",
    "HOST_RSS_LIMIT_GIB",
    "MEASURED_PEAK_DEVICE_BYTES_AT_2M",
    "RungMixedTrainingInput",
    "predict_rung_footprint",
    "validate_full_rung_map",
]

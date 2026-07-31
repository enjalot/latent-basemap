"""Balanced nested-universe primitives for Round 0043.

Round 0043 holds the accepted R0036 model coordinates fixed and changes only
the candidate universe used by the evaluator.  The 150M source is three
contiguous 50M corpus blocks.  A rung of width ``w`` is therefore the retained
R0033 representatives in::

    [0, w) + [50M, 50M + w) + [100M, 100M + w)

The explicit global-row mapping is structural: compact slices are never
mistaken for the disjoint balanced source intervals.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from .artifact_identity import ordered_array_sha256


ROUND_ID = "0043"
SOURCE_ROWS = 150_000_000
CORPUS_BLOCK_ROWS = 50_000_000
CORPUS_COUNT = 3
DIMENSION = 384
RUNG_WIDTHS = (10_000_000, 20_000_000, 40_000_000, 50_000_000)
CORE_WIDTH = 10_000_000
EXPECTED_RUNG_IDENTITIES = {
    10_000_000: {
        "retained_counts": (9_958_301, 9_901_491, 9_919_561),
        "retained_count": 29_779_353,
        "global_rows_sha256": (
            "4e5ad1deec7567d24aa9ba0253f896d5c1357bdd6e6d853304d75a1e6ad9abe4"
        ),
    },
    20_000_000: {
        "retained_counts": (19_819_673, 19_792_137, 19_784_154),
        "retained_count": 59_395_964,
        "global_rows_sha256": (
            "4bf8c59af624811b7779f411b8bc2f1de63058e9ddeefd382453be25b18c679f"
        ),
    },
    40_000_000: {
        "retained_counts": (38_999_905, 39_681_544, 39_384_099),
        "retained_count": 118_065_548,
        "global_rows_sha256": (
            "7a6a7f8640188d895ff0c3d382a30d954885f279b3afb41b1ba508b7e1a64724"
        ),
    },
    50_000_000: {
        "retained_counts": (48_529_276, 49_567_453, 49_125_028),
        "retained_count": 147_221_757,
        "global_rows_sha256": (
            "61ac5ebdbbb82cac0955a781311e017f6904340bcfce2e38aec0be7874f0572c"
        ),
    },
}

ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0033/queue/artifacts/eligibility/"
    "minilm-150m-row-eligibility-v1.npz"
)
ELIGIBILITY_SHA256 = (
    "cd9738d1cb35b7847923ec24e343583ac91dea4d76381ec28c8c2c8bf6412aca"
)
COORDINATE_ROOT = (
    "/data/latent-basemap/runs/round-0036/queue/artifacts/coordinates"
)
COORDINATE_RECEIPT_SHA256 = (
    "3f3a04721027d9f0d4d90adb1b20d9fd28ce6e8f1ce11b1a1b35884644f9eb73"
)
R0036_PANEL = (
    "/data/latent-basemap/runs/round-0036/queue/artifacts/panel/panel.json"
)
R0036_PANEL_SHA256 = (
    "5d30384f1c3af89f952357c4ae52686950a817908bb7e4634740cb5ca9195423"
)
R0025_MANIFEST = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/"
    "int8-shards-v1.json"
)
R0025_MANIFEST_SHA256 = (
    "38c3847f2811725d571d4861a74864598faa4c76f56caf81a5d3a89cdb4a3f7d"
)
INT8_SHA256 = (
    "2171e4bf3c21e7156435b4b4021ca62b2ef8a57d9404b2764e6e968d210b7090"
)
SCALES_SHA256 = (
    "d282d4f5a5abbe17e981d957fce1cd9e227cbd67aa3262803542d496dbbecb49"
)

PANEL_CONFIG = {
    "frac": 0.001,
    "k_density": 15,
    "k_hit": 10,
    "n_anchors": 10_000,
    "anchor_seed": 123,
    "corpus_chunk": 500_000,
    "overselect": 8,
    "block_elems": 500_000_000,
    "rerank_byte_cap": 2_000_000_000,
    "rerank_scratch": 3.0,
    "peak_byte_cap": 26_000_000_000,
}


class Round0043Error(RuntimeError):
    """The registered R0043 nested-universe contract changed."""


class BalancedRungSelector:
    """Exact compact/global mapping for one balanced three-corpus rung."""

    def __init__(
        self,
        excluded_rows: np.ndarray,
        *,
        per_corpus_rows: int,
        corpus_block_rows: int = CORPUS_BLOCK_ROWS,
        corpus_count: int = CORPUS_COUNT,
    ) -> None:
        excluded = np.asarray(excluded_rows, dtype=np.int64)
        source_rows = int(corpus_block_rows) * int(corpus_count)
        if (
            excluded.ndim != 1
            or not np.array_equal(excluded, np.unique(excluded))
            or (
                len(excluded)
                and (excluded[0] < 0 or excluded[-1] >= source_rows)
            )
        ):
            raise ValueError(
                "excluded rows must be sorted, unique, and in source range"
            )
        if (
            not isinstance(per_corpus_rows, int)
            or isinstance(per_corpus_rows, bool)
            or not 0 < per_corpus_rows <= corpus_block_rows
            or corpus_count <= 0
        ):
            raise ValueError("balanced rung geometry is invalid")
        self.excluded_rows = excluded
        self.per_corpus_rows = int(per_corpus_rows)
        self.corpus_block_rows = int(corpus_block_rows)
        self.corpus_count = int(corpus_count)
        self.source_rows = source_rows
        self.intervals = tuple(
            (
                corpus * self.corpus_block_rows,
                corpus * self.corpus_block_rows + self.per_corpus_rows,
            )
            for corpus in range(self.corpus_count)
        )
        self._excluded_bounds = tuple(
            (
                int(np.searchsorted(excluded, start, side="left")),
                int(np.searchsorted(excluded, stop, side="left")),
            )
            for start, stop in self.intervals
        )
        self.retained_counts = tuple(
            (stop - start) - (right - left)
            for (start, stop), (left, right) in zip(
                self.intervals, self._excluded_bounds
            )
        )
        self.retained_count = int(sum(self.retained_counts))
        self.excluded_count = int(
            self.per_corpus_rows * self.corpus_count
            - self.retained_count
        )
        expected = (
            EXPECTED_RUNG_IDENTITIES.get(self.per_corpus_rows)
            if (
                self.corpus_block_rows == CORPUS_BLOCK_ROWS
                and self.corpus_count == CORPUS_COUNT
            )
            else None
        )
        if (
            expected is not None
            and (
                self.retained_counts != expected["retained_counts"]
                or self.retained_count != expected["retained_count"]
            )
        ):
            raise Round0043Error(
                "registered balanced-rung row counts changed"
            )
        self._global_rows: np.ndarray | None = None
        self._identity: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.retained_count

    def _materialize_global_rows(self) -> np.ndarray:
        cached = self._global_rows
        if cached is not None:
            return cached
        rows = np.empty(self.retained_count, dtype=np.int64)
        cursor = 0
        for (
            (start, stop),
            (left, right),
            expected_count,
        ) in zip(
            self.intervals,
            self._excluded_bounds,
            self.retained_counts,
        ):
            mask = np.ones(stop - start, dtype=np.bool_)
            local_excluded = self.excluded_rows[left:right] - start
            mask[local_excluded] = False
            local_rows = np.flatnonzero(mask)
            if len(local_rows) != expected_count:
                raise Round0043Error(
                    "balanced rung retained-row accounting changed"
                )
            rows[cursor : cursor + expected_count] = local_rows + start
            cursor += expected_count
            del mask, local_rows
        if (
            cursor != self.retained_count
            or (len(rows) and np.any(rows[1:] <= rows[:-1]))
            or not np.all(self.is_member(rows))
        ):
            raise Round0043Error(
                "balanced rung global-row materialization is malformed"
            )
        rows.flags.writeable = False
        self._global_rows = rows
        return rows

    def compact_to_global(self, compact_rows: Any) -> np.ndarray:
        compact = np.asarray(compact_rows, dtype=np.int64)
        if np.any(compact < 0) or np.any(compact >= self.retained_count):
            raise IndexError("balanced rung compact row is out of range")
        return self._materialize_global_rows()[compact]

    def global_to_compact(self, global_rows: Any) -> np.ndarray:
        rows = np.asarray(global_rows, dtype=np.int64)
        materialized = self._materialize_global_rows()
        positions = np.searchsorted(materialized, rows)
        clipped = np.minimum(positions, max(len(materialized) - 1, 0))
        if (
            not len(materialized)
            or np.any(positions >= len(materialized))
            or not np.array_equal(materialized[clipped], rows)
        ):
            raise IndexError("global row is outside the balanced rung")
        return positions.astype(np.int64, copy=False)

    def is_member(self, global_rows: Any) -> np.ndarray:
        rows = np.asarray(global_rows, dtype=np.int64)
        in_interval = np.zeros(rows.shape, dtype=np.bool_)
        for start, stop in self.intervals:
            in_interval |= (rows >= start) & (rows < stop)
        positions = np.searchsorted(self.excluded_rows, rows)
        excluded = np.zeros(rows.shape, dtype=np.bool_)
        if len(self.excluded_rows):
            clipped = np.minimum(positions, len(self.excluded_rows) - 1)
            excluded = (
                positions < len(self.excluded_rows)
            ) & (self.excluded_rows[clipped] == rows)
        return in_interval & ~excluded

    def identity(self) -> dict[str, Any]:
        cached = self._identity
        if cached is None:
            cached = {
                "schema": "round0043-balanced-retained-rung-v1",
                "source_rows": self.source_rows,
                "corpus_count": self.corpus_count,
                "corpus_block_rows": self.corpus_block_rows,
                "per_corpus_rows": self.per_corpus_rows,
                "raw_rung_rows": (
                    self.per_corpus_rows * self.corpus_count
                ),
                "retained_rows": self.retained_count,
                "excluded_rows": self.excluded_count,
                "intervals": [list(value) for value in self.intervals],
                "retained_rows_per_corpus": list(self.retained_counts),
                "global_rows_sha256": ordered_array_sha256(
                    self._materialize_global_rows()
                ),
                "eligibility_sha256": ELIGIBILITY_SHA256,
                "row_order": (
                    "ascending global IDs across balanced fineweb/"
                    "redpajama/pile intervals"
                ),
            }
            expected = (
                EXPECTED_RUNG_IDENTITIES.get(self.per_corpus_rows)
                if (
                    self.corpus_block_rows == CORPUS_BLOCK_ROWS
                    and self.corpus_count == CORPUS_COUNT
                )
                else None
            )
            if (
                expected is not None
                and cached["global_rows_sha256"]
                != expected["global_rows_sha256"]
            ):
                raise Round0043Error(
                    "registered balanced-rung global row bytes changed"
                )
            self._identity = cached
        return dict(cached)


class BalancedRungView:
    """Lazy row view backed by an exact :class:`BalancedRungSelector`."""

    round0043_balanced_view = True

    def __init__(self, base: Any, selector: BalancedRungSelector) -> None:
        if len(base) != selector.source_rows:
            raise ValueError("balanced view/base source row counts differ")
        self.base = base
        self.selector = selector
        self.shape = (len(selector), int(base.shape[1]))
        self.dtype = np.dtype(base.dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            rows, columns = key
            return self[rows][..., columns]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            compact = np.arange(start, stop, step, dtype=np.int64)
        else:
            compact = np.asarray(key, dtype=np.int64)
        if compact.ndim == 0:
            global_row = int(self.selector.compact_to_global(compact))
            return self.base[global_row]
        shape = compact.shape
        flat = compact.reshape(-1)
        values = self.base[self.selector.compact_to_global(flat)]
        return np.asarray(values).reshape(shape + (self.shape[1],))

    def scientific_identity(self) -> dict[str, Any]:
        base = (
            self.base.scientific_identity()
            if hasattr(self.base, "scientific_identity")
            else {
                "kind": "row-aligned-array",
                "shape": list(self.base.shape),
                "dtype": np.dtype(self.base.dtype).str,
            }
        )
        return {
            "schema": "round0043-balanced-view-identity-v1",
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "base": base,
            "selector": self.selector.identity(),
        }


def rung_label(per_corpus_rows: int) -> str:
    total_millions = per_corpus_rows * CORPUS_COUNT // 1_000_000
    if per_corpus_rows not in RUNG_WIDTHS:
        raise ValueError("unregistered R0043 rung width")
    return f"{total_millions:03d}m"


def validate_manifest_universe(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the reviewed R0025 150M signatures after exact schema checks."""
    universe = (manifest.get("universes") or {}).get("minilm-int8-150m")
    if (
        manifest.get("schema") != "round0025-int8-shards-v1"
        or not isinstance(universe, dict)
        or universe.get("rows") != SOURCE_ROWS
        or universe.get("dimension") != DIMENSION
        or universe.get("embedding_dtype") != "int8"
        or universe.get("row_scale_dtype") != "<f2"
    ):
        raise Round0043Error("R0025 150M nested source geometry changed")
    int8 = universe.get("int8") or {}
    scales = universe.get("scales") or {}
    for value, expected_bytes, expected_sha256 in (
        (int8, SOURCE_ROWS * DIMENSION, INT8_SHA256),
        (scales, SOURCE_ROWS * 2, SCALES_SHA256),
    ):
        path = value.get("canonical_path")
        if (
            not isinstance(path, str)
            or not os.path.isfile(path)
            or os.path.getsize(path) != expected_bytes
            or value.get("bytes") != expected_bytes
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64
            or value["sha256"] != expected_sha256
        ):
            raise Round0043Error("R0025 150M nested input signature changed")
    return {"int8": dict(int8), "scales": dict(scales)}

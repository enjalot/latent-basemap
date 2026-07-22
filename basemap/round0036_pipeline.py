"""Reviewed-model and retained-row primitives for Round 0036.

R0036 deliberately separates two row universes:

* the transform is row aligned to all 150,000,000 R0025 rows; and
* scientific scoring addresses only the compact sequence of R0033-retained
  representatives.

The classes in this module make that distinction structural.  Callers cannot
accidentally pass an excluded global row to the representative scorer, and no
150M-sized retained-ID array is materialised merely to translate positions.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .int8_eligibility import load_int8_eligibility
from .round0034_program import (
    INT8_PATH,
    INT8_SHA256,
    SCALES_PATH,
    SCALES_SHA256,
)


ROUND_ID = "0036"
ROW_COUNT = 150_000_000
DIMENSION = 384
RETAINED_ROWS = 147_221_757
ZERO_ROWS = 235_469
DUPLICATE_COPIES = 2_542_774
ELIGIBILITY_SHA256 = (
    "cd9738d1cb35b7847923ec24e343583ac91dea4d76381ec28c8c2c8bf6412aca"
)
INDEX_PATH = "/data/checkpoints/pumap/faiss_ivf_pq_150m.index"
COORDINATE_SCHEMA = "round0036-coordinate-stream-v1"
TRANSFORM_SCHEMA = "round0036-transform-capability-v1"


class Round0036Error(RuntimeError):
    """Fail-closed R0036 contract error."""


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0036Error(f"{label} identity seal is invalid")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_reviewed_model_bundle(
    *,
    model_path: str,
    model_sha256: str,
    train_receipt_path: str,
    train_receipt_sha256: str,
    expected_rows: int = ROW_COUNT,
) -> dict[str, Any]:
    """Authenticate the exact custom R0034 checkpoint and train receipt.

    R0034 intentionally writes a smaller checkpoint than ``ParametricUMAP``'s
    generic ``save`` format.  Treating it as a generic checkpoint would either
    fail late or invite an unreviewed compatibility fallback, so R0036 binds
    its exact three-field schema here.
    """
    if not _valid_sha256(model_sha256) or not _valid_sha256(train_receipt_sha256):
        raise Round0036Error("reviewed model/receipt SHA-256 is malformed")
    model_signature = expected_input_signature(model_path)
    receipt_signature = expected_input_signature(train_receipt_path)
    if model_signature["sha256"] != model_sha256:
        raise Round0036Error("reviewed R0034 model bytes changed")
    if receipt_signature["sha256"] != train_receipt_sha256:
        raise Round0036Error("reviewed R0034 train receipt bytes changed")
    with open(train_receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0034 train receipt")
    config = receipt.get("train_config")
    config_sha256 = receipt.get("train_config_sha256")
    accounting = receipt.get("train_accounting") or {}
    execution = receipt.get("exact_execution_receipt") or {}
    expected_stamp = (config or {}).get("execution", {}).get(
        "expected_pipeline_stamp"
    )
    required_updates = (config or {}).get("optimizer", {}).get(
        "successful_positive_lr_updates"
    )
    row_universe = (config or {}).get("row_universe") or {}
    model_config = (config or {}).get("model") or {}
    expected_architecture_fields = {
        "architecture",
        "input_dimension",
        "hidden_dimension",
        "hidden_layers",
        "output_dimension",
        "use_batchnorm",
        "use_dropout",
        "low_dim_kernel",
        "a",
        "b",
    }
    architecture = {
        key: model_config.get(key) for key in expected_architecture_fields
    }
    expected_architecture = {
        "architecture": "residual_bottleneck",
        "input_dimension": DIMENSION,
        "hidden_dimension": 2048,
        "hidden_layers": 3,
        "output_dimension": 2,
        "use_batchnorm": False,
        "use_dropout": False,
        "low_dim_kernel": "legacy_lp",
        "a": 1.0,
        "b": 1.0,
    }
    stamp_fields = (
        "pipeline",
        "sampler_class",
        "x_residency",
        "positive_sampling",
        "negative_sampling",
    )
    observed_stamp = {key: execution.get(key) for key in stamp_fields}
    required_stamp = {
        key: (expected_stamp or {}).get(key) for key in stamp_fields
    }
    if (
        receipt.get("schema") != "round0034-train-receipt-v1"
        or receipt.get("round_id") != "0034"
        or receipt.get("model") != model_signature
        or not isinstance(config, dict)
        or config_sha256 != sha256_bytes(canonical_json(config))
        or row_universe.get("rows") != expected_rows
        or row_universe.get("input_dimension") != DIMENSION
        or row_universe.get("int8_sha256") != INT8_SHA256
        or row_universe.get("scale_sha256") != SCALES_SHA256
        or architecture != expected_architecture
        or not isinstance(required_updates, int)
        or required_updates <= 0
        or accounting.get("budget_satisfied") is not True
        or accounting.get("positive_lr_optimizer_steps") != required_updates
        or accounting.get("optimizer_steps_attempted") != required_updates
        or accounting.get("optimizer_steps_succeeded") != required_updates
        or accounting.get("amp_overflow_skips") != 0
        or accounting.get("nonfinite_loss_skips") != 0
        or accounting.get("nonfinite_gradient_skips") != 0
        or required_stamp != observed_stamp
    ):
        raise Round0036Error("reviewed R0034 train/model contract is incomplete")
    return {
        "model": model_signature,
        "train_receipt": receipt_signature,
        "receipt": receipt,
        "production_config": config,
        "production_config_sha256": config_sha256,
    }


def load_reviewed_model(
    bundle: Mapping[str, Any],
    *,
    device: str,
    model_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    """Load R0034's exact state dict after validating keys, shapes and dtypes."""
    import torch

    model_path = bundle["model"]["canonical_path"]
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "state_dict",
        "production_config",
        "production_config_sha256",
    }:
        raise Round0036Error("R0034 checkpoint fields changed")
    config = bundle["production_config"]
    config_sha256 = bundle["production_config_sha256"]
    if (
        checkpoint["production_config"] != config
        or checkpoint["production_config_sha256"] != config_sha256
    ):
        raise Round0036Error("R0034 checkpoint and train receipt configs differ")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not state:
        raise Round0036Error("R0034 checkpoint state dict is empty")
    if any(
        not isinstance(name, str)
        or not torch.is_tensor(value)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for name, value in state.items()
    ):
        raise Round0036Error("R0034 checkpoint parameters are not finite fp32")
    if model_factory is None:
        from experiments.run_round0034_node import _exact_model

        model_factory = _exact_model
    instance = model_factory(config)
    instance.device = str(device)
    instance._init_model(input_dim=DIMENSION)
    expected_state = instance.model.state_dict()
    if set(state) != set(expected_state) or any(
        tuple(state[name].shape) != tuple(expected_state[name].shape)
        for name in state
    ):
        raise Round0036Error("R0034 checkpoint state keys/shapes changed")
    instance.model.load_state_dict(state, strict=True)
    instance.model.to(device)
    instance.model.eval()
    instance.is_fitted = True
    return instance


class RetainedRowSelector:
    """Rank/select adapter over a sorted excluded-row array.

    Compact position ``p`` denotes the p-th retained global row.  Selection is
    calculated with a vectorised binary search, so even a 141,158-row run of
    exclusions closes exactly without an over-fetch constant.
    """

    def __init__(self, excluded_rows: np.ndarray, *, row_count: int) -> None:
        excluded = np.asarray(excluded_rows, dtype=np.int64)
        if (
            excluded.ndim != 1
            or not np.array_equal(excluded, np.unique(excluded))
            or (len(excluded) and (excluded[0] < 0 or excluded[-1] >= row_count))
        ):
            raise ValueError("excluded rows must be sorted, unique and in range")
        self.excluded_rows = excluded
        self.row_count = int(row_count)
        self.retained_count = int(row_count - len(excluded))

    def is_retained(self, global_rows: Any) -> np.ndarray:
        rows = np.asarray(global_rows, dtype=np.int64)
        positions = np.searchsorted(self.excluded_rows, rows)
        clipped = np.minimum(positions, max(len(self.excluded_rows) - 1, 0))
        matches = np.zeros(rows.shape, dtype=bool)
        if len(self.excluded_rows):
            matches = (positions < len(self.excluded_rows)) & (
                self.excluded_rows[clipped] == rows
            )
        return (
            (rows >= 0)
            & (rows < self.row_count)
            & ~matches
        )

    def compact_to_global(self, compact_rows: Any) -> np.ndarray:
        compact = np.asarray(compact_rows, dtype=np.int64)
        if np.any(compact < 0) or np.any(compact >= self.retained_count):
            raise IndexError("compact retained-row position is out of range")
        flat = compact.reshape(-1)
        low = flat.copy()
        high = flat + len(self.excluded_rows)
        # Find the smallest global g for which retained rows through g exceed p.
        while np.any(low < high):
            middle = low + (high - low) // 2
            retained_through = middle + 1 - np.searchsorted(
                self.excluded_rows, middle, side="right"
            )
            move_right = retained_through <= flat
            low = np.where(move_right, middle + 1, low)
            high = np.where(move_right, high, middle)
        if not np.all(self.is_retained(low)):
            raise Round0036Error("retained rank/select returned an excluded row")
        return low.reshape(compact.shape)

    def global_to_compact(self, global_rows: Any) -> np.ndarray:
        rows = np.asarray(global_rows, dtype=np.int64)
        if not np.all(self.is_retained(rows)):
            raise IndexError("global-to-compact input includes an excluded row")
        return rows - np.searchsorted(self.excluded_rows, rows, side="left")

    def bitmap(self, *, maximum_global_row: int | None = None) -> np.ndarray:
        """Return FAISS' little-bit-order include bitmap for this universe."""
        limit = self.row_count if maximum_global_row is None else int(maximum_global_row)
        if not 0 < limit <= self.row_count:
            raise ValueError("bitmap row limit is invalid")
        result = np.full((limit + 7) // 8, 255, dtype=np.uint8)
        excluded = self.excluded_rows[self.excluded_rows < limit]
        if len(excluded):
            byte = excluded >> 3
            bit = excluded & 7
            masks = np.bitwise_not(np.left_shift(1, bit).astype(np.uint8))
            # ``byte`` contains repeats whenever adjacent rows are excluded;
            # ufunc.at is required because ordinary advanced-index assignment
            # would apply only one of those clear masks per byte.
            np.bitwise_and.at(result, byte, masks)
        tail = limit & 7
        if tail:
            result[-1] &= np.uint8((1 << tail) - 1)
        return result

    def identity(self) -> dict[str, Any]:
        return {
            "schema": "round0036-retained-row-selector-v1",
            "row_count": self.row_count,
            "retained_count": self.retained_count,
            "excluded_count": int(len(self.excluded_rows)),
            "excluded_rows_sha256": ordered_array_sha256(self.excluded_rows),
            "selection": "compact-rank/select-over-sorted-R0033-exclusions",
        }


def load_released_selector(
    eligibility_path: str,
    *,
    eligibility_sha256: str = ELIGIBILITY_SHA256,
    row_count: int = ROW_COUNT,
) -> tuple[RetainedRowSelector, dict[str, Any]]:
    eligibility = load_int8_eligibility(
        eligibility_path,
        expected_sha256=eligibility_sha256,
        row_count=row_count,
    )
    summary = eligibility["metadata"]["summary"]
    if row_count == ROW_COUNT and (
        summary.get("retained_row_count") != RETAINED_ROWS
        or summary.get("zero_row_count") != ZERO_ROWS
        or summary.get("duplicate_copy_rows_excluded") != DUPLICATE_COPIES
    ):
        raise Round0036Error("R0033 reviewed row accounting changed")
    selector = RetainedRowSelector(
        eligibility["excluded_rows"], row_count=row_count
    )
    return selector, eligibility


class EncodedInt8Array:
    """Read-only fp32 dequantized view of the R0025 int8+fp16-scale pair."""

    round0036_encoded_input = True

    def __init__(
        self,
        encoded: np.ndarray,
        scales: np.ndarray,
        *,
        signatures: Mapping[str, Any],
    ) -> None:
        if (
            encoded.ndim != 2
            or encoded.dtype != np.dtype("int8")
            or scales.shape != (len(encoded),)
            or scales.dtype != np.dtype("<f2")
        ):
            raise ValueError("encoded evaluation input geometry changed")
        self.encoded = encoded
        self.scales = scales
        self.signatures = dict(signatures)
        self.shape = encoded.shape
        self.dtype = np.dtype("<f4")

    @classmethod
    def from_files(
        cls,
        *,
        int8_path: str = INT8_PATH,
        int8_sha256: str = INT8_SHA256,
        scales_path: str = SCALES_PATH,
        scales_sha256: str = SCALES_SHA256,
        row_count: int = ROW_COUNT,
        dimension: int = DIMENSION,
        verify_hashes: bool = True,
    ) -> "EncodedInt8Array":
        if verify_hashes:
            int8_signature = expected_input_signature(int8_path)
            scales_signature = expected_input_signature(scales_path)
        else:
            int8_signature = {
                "canonical_path": os.path.realpath(int8_path),
                "bytes": os.path.getsize(int8_path),
                "sha256": int8_sha256,
            }
            scales_signature = {
                "canonical_path": os.path.realpath(scales_path),
                "bytes": os.path.getsize(scales_path),
                "sha256": scales_sha256,
            }
        if (
            int8_signature.get("sha256") != int8_sha256
            or scales_signature.get("sha256") != scales_sha256
            or int8_signature.get("bytes") != row_count * dimension
            or scales_signature.get("bytes") != row_count * 2
        ):
            raise Round0036Error("R0025 encoded evaluation bytes changed")
        encoded = np.memmap(
            int8_path, dtype=np.int8, mode="r", shape=(row_count, dimension)
        )
        scales = np.memmap(
            scales_path, dtype="<f2", mode="r", shape=(row_count,)
        )
        return cls(
            encoded,
            scales,
            signatures={"int8": int8_signature, "scales": scales_signature},
        )

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, key: Any) -> np.ndarray:
        values = np.asarray(self.encoded[key], dtype=np.float32)
        scales = np.asarray(self.scales[key], dtype=np.float32)
        if values.ndim == 1:
            return values * scales
        return values * scales[..., None]

    def scientific_identity(self) -> dict[str, Any]:
        return {
            "kind": "ordered_shards",
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "dtype": np.dtype("<f4").str,
            "shards": [
                {
                    "position": position,
                    "name": os.path.basename(signature["canonical_path"]),
                    "bytes": int(signature["bytes"]),
                    "sha256": signature["sha256"],
                }
                for position, signature in enumerate(
                    (self.signatures["int8"], self.signatures["scales"])
                )
            ],
        }


class RetainedArrayView:
    """Compact retained-row view over any row-aligned base array."""

    round0036_retained_view = True

    def __init__(self, base: Any, selector: RetainedRowSelector) -> None:
        if len(base) != selector.row_count:
            raise ValueError("retained view/base row counts differ")
        self.base = base
        self.selector = selector
        self.shape = (selector.retained_count, int(base.shape[1]))
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
            return self.base[self.selector.compact_to_global(compact)]
        compact = np.asarray(key, dtype=np.int64)
        if compact.ndim == 0:
            global_row = int(self.selector.compact_to_global(compact))
            return self.base[global_row]
        flat = compact.reshape(-1)
        values = self.base[self.selector.compact_to_global(flat)]
        return np.asarray(values).reshape(compact.shape + (self.shape[1],))

    def _reduce(self, op: Any, seed: float, *, axis: int | None):
        if axis not in (None, 0):
            raise ValueError("retained-view reduction only supports axis 0 or all")
        value = np.full(self.shape[1], seed, dtype=np.float32)
        for start in range(0, len(self), 1_000_000):
            block = np.asarray(self[start : min(start + 1_000_000, len(self))])
            value = op(value, op.reduce(block, axis=0))
        return value if axis == 0 else op.reduce(value)

    def min(self, axis: int | None = None):
        return self._reduce(np.minimum, np.inf, axis=axis)

    def max(self, axis: int | None = None):
        return self._reduce(np.maximum, -np.inf, axis=axis)

    def scale_admission_identity(self) -> dict[str, Any]:
        base = (
            self.base.scientific_identity()
            if hasattr(self.base, "scientific_identity")
            else None
        )
        body = {
            "schema": "round0036-retained-scale-input-v1",
            "row_count": len(self),
            "dimensions": self.shape[1],
            "base": base,
            "selector": self.selector.identity(),
        }
        return seal(body)


class CoordinateStream:
    """Authenticated lazy view over a row-aligned R0036 coordinate stream."""

    def __init__(self, root: str, *, expected_receipt_sha256: str | None = None):
        self.root = os.path.realpath(root)
        receipt_path = os.path.join(self.root, "actual-transform.json")
        signature = expected_input_signature(receipt_path)
        if expected_receipt_sha256 and signature["sha256"] != expected_receipt_sha256:
            raise Round0036Error("R0036 coordinate receipt bytes changed")
        with open(receipt_path, encoding="utf-8") as handle:
            self.receipt = json.load(handle)
        validate_seal(self.receipt, label="R0036 transform receipt")
        stream = self.receipt.get("coordinate_stream") or {}
        members = stream.get("ordered_chunks")
        if (
            self.receipt.get("schema") != TRANSFORM_SCHEMA
            or stream.get("schema") != COORDINATE_SCHEMA
            or not isinstance(members, list)
            or stream.get("row_count") != self.receipt.get("row_accounting", {}).get(
                "all_rows"
            )
            or stream.get("dtype") != "<f4"
            or stream.get("dimension") != 2
        ):
            raise Round0036Error("R0036 coordinate stream contract changed")
        self._members: list[dict[str, Any]] = []
        self._arrays: list[np.ndarray] = []
        cursor = 0
        for position, member in enumerate(members):
            path = os.path.join(
                self.root,
                f"chunk-{int(member['chunk_index']):05d}",
                "coordinates.npy",
            )
            observed = expected_input_signature(path)
            if (
                member.get("chunk_index") != position
                or member.get("global_row_start") != cursor
                or member.get("global_row_stop", 0) <= cursor
                or observed.get("sha256") != member.get("sha256")
                or observed.get("bytes") != member.get("bytes")
            ):
                raise Round0036Error("R0036 coordinate member identity/order changed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                array.shape != (member["global_row_stop"] - cursor, 2)
                or array.dtype.str != "<f4"
            ):
                raise Round0036Error("R0036 coordinate member geometry changed")
            self._members.append({**member, "path": path})
            # Keep one read-only memmap per immutable chunk. Reopening it inside
            # every 500k-row scorer tile adds thousands of avoidable mmap calls.
            self._arrays.append(array)
            cursor = int(member["global_row_stop"])
        if cursor != stream["row_count"]:
            raise Round0036Error("R0036 coordinate stream row coverage is incomplete")
        self.shape = (cursor, 2)
        self.dtype = np.dtype("<f4")
        self.shard_paths = [member["path"] for member in self._members]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            rows, columns = key
            return self[rows][..., columns]
        if isinstance(key, (int, np.integer)):
            return self[np.asarray([key], dtype=np.int64)][0]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            rows = np.arange(start, stop, step, dtype=np.int64)
        else:
            rows = np.asarray(key, dtype=np.int64)
        shape = rows.shape
        flat = rows.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= len(self)):
            raise IndexError("coordinate row is out of range")
        output = np.empty((len(flat), 2), dtype="<f4")
        if len(flat) < 2 or np.all(flat[:-1] <= flat[1:]):
            ordered_rows = flat
            order = None
            ordered_output = output
        else:
            # Sort once, find each immutable member interval with two binary
            # searches, then restore the caller's arbitrary order.  The former
            # member-by-member boolean masks did 30 * len(rows) comparisons per
            # scorer tile (about 44B comparisons over the 150M panel).
            order = np.argsort(flat, kind="stable")
            ordered_rows = flat[order]
            ordered_output = np.empty_like(output)
        copied = 0
        for member, array in zip(self._members, self._arrays):
            low = int(member["global_row_start"])
            high = int(member["global_row_stop"])
            left = int(np.searchsorted(ordered_rows, low, side="left"))
            right = int(np.searchsorted(ordered_rows, high, side="left"))
            if right > left:
                ordered_output[left:right] = array[ordered_rows[left:right] - low]
                copied += right - left
        if copied != len(flat):
            raise Round0036Error("coordinate gather did not cover every requested row")
        if order is not None:
            output[order] = ordered_output
        return output.reshape(shape + (2,))

    def min(self, axis: int | None = None) -> np.ndarray | np.float32:
        return self._reduce(np.minimum, np.inf, axis=axis)

    def max(self, axis: int | None = None) -> np.ndarray | np.float32:
        return self._reduce(np.maximum, -np.inf, axis=axis)

    def _reduce(self, op: Any, seed: float, *, axis: int | None):
        if axis not in (None, 0):
            raise ValueError("coordinate reduction only supports axis 0 or all")
        value = np.full(2, seed, dtype=np.float32)
        for array in self._arrays:
            value = op(value, op.reduce(array, axis=0))
        return value if axis == 0 else op.reduce(value)

    def scientific_identity(self) -> dict[str, Any]:
        return {
            "kind": "ordered_shards",
            "shape": [len(self), 2],
            "dtype": "<f4",
            "shards": [
                {
                    "position": index,
                    "name": f"chunk-{index:05d}-coordinates.npy",
                    "bytes": int(member["bytes"]),
                    "sha256": member["sha256"],
                }
                for index, member in enumerate(self._members)
            ],
        }


class RetainedFaissIndex:
    """Exact retained-ID filtering for an IVF index via ``IDSelectorBitmap``."""

    def __init__(
        self,
        index: Any,
        selector: RetainedRowSelector,
        *,
        nprobe: int,
        maximum_global_row: int | None = None,
    ) -> None:
        import faiss

        self.index = index
        self.rows = selector.row_count if maximum_global_row is None else int(
            maximum_global_row
        )
        if int(index.ntotal) < self.rows:
            raise Round0036Error("FAISS index is smaller than retained selector universe")
        self.selector = selector
        self.bitmap = selector.bitmap(maximum_global_row=self.rows)
        self.faiss_selector = faiss.IDSelectorBitmap(
            len(self.bitmap), faiss.swig_ptr(self.bitmap)
        )
        self.params = faiss.SearchParametersIVF()
        self.params.nprobe = int(nprobe)
        self.params.sel = self.faiss_selector

    def search_global(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.ascontiguousarray(queries, dtype=np.float32)
        distances, neighbors = self.index.search(values, int(k), params=self.params)
        if (
            neighbors.shape != (len(values), k)
            or np.any(neighbors < 0)
            or np.any(neighbors >= self.rows)
            or not np.all(self.selector.is_retained(neighbors))
        ):
            raise Round0036Error("filtered IVF search returned missing/excluded rows")
        return distances, neighbors.astype(np.int64, copy=False)

    def search_compact(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, global_rows = self.search_global(queries, k)
        return distances, self.selector.global_to_compact(global_rows)


def panel_config_identity(
    *,
    frac: float = 0.001,
    n_anchors: int = 10_000,
    anchor_seed: int = 123,
) -> dict[str, Any]:
    return {
        "formula_version": "panel_v2.2-2026-07-15",
        "frac": float(frac),
        "k_clust": [256, 1024],
        "k_density": 15,
        "k_hit": 10,
        "n_anchors": int(n_anchors),
        "anchor_seed": int(anchor_seed),
        "corpus_chunk": 500_000,
        "overselect": 8,
        "block_elems": 500_000_000,
        "rerank_byte_cap": 2_000_000_000,
        "rerank_scratch": 3.0,
        "peak_byte_cap": 26_000_000_000,
    }


def estimated_queue_gpu_seconds() -> dict[str, float]:
    """P90 node estimates after correcting the 150M low-D access strategy.

    The transform is a small fraction of the queue on the observed 30M rate;
    most uncertainty belongs to the growing-k high-D reference and registered
    panel.  Their envelopes are deliberately widened while retaining the
    round's hard eight-hour bound.
    """
    values = {
        "production_canary": 300.0,
        "transform_150m": 900.0,
        "high_d_reference": 14_400.0,
        "registered_panel": 10_800.0,
        "ood_canaries_and_panels": 1_800.0,
    }
    values["total"] = float(sum(values.values()))
    if values["total"] > 8 * 3600:
        raise AssertionError("R0036 p90 GPU job estimates exceed eight hours")
    return values


def low_dim_search_work_model(
    *,
    rows: int = RETAINED_ROWS,
    anchors: int = 10_000,
    dimensions: int = 2,
    corpus_chunk: int = 500_000,
    block_elems: int = 500_000_000,
) -> dict[str, Any]:
    """Shape-derived traffic model for the exact chunked low-D scorer."""
    if min(rows, anchors, dimensions, corpus_chunk, block_elems) <= 0:
        raise ValueError("low-D work-model dimensions must be positive")
    chunk = min(int(rows), int(corpus_chunk))
    anchor_tile = max(1, min(int(anchors), int(block_elems) // chunk))
    anchor_tiles = math.ceil(int(anchors) / anchor_tile)
    corpus_chunks = math.ceil(int(rows) / chunk)
    coordinate_bytes_per_pass = int(rows) * int(dimensions) * 4
    legacy_anchor_tile = max(1, min(int(anchors), int(block_elems) // int(rows)))
    legacy_passes = math.ceil(int(anchors) / legacy_anchor_tile)
    return {
        "schema": "round0036-low-d-search-work-model-v1",
        "rows": int(rows),
        "anchors": int(anchors),
        "dimensions": int(dimensions),
        "corpus_chunk": chunk,
        "anchor_tile": anchor_tile,
        "anchor_tiles": anchor_tiles,
        "corpus_chunks_per_anchor_tile": corpus_chunks,
        "distance_elements": int(rows) * int(anchors),
        "coordinate_bytes_per_pass": coordinate_bytes_per_pass,
        "coordinate_bytes_total": coordinate_bytes_per_pass * anchor_tiles,
        "legacy_full_corpus_anchor_tile": legacy_anchor_tile,
        "legacy_full_corpus_passes": legacy_passes,
        "legacy_coordinate_bytes_total": coordinate_bytes_per_pass * legacy_passes,
    }

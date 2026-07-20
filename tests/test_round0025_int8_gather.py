from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import bench_int8_gather as target


def _write_raw(path: Path, *, rows: int, seed: int,
               trailing: bytes = b"") -> np.ndarray:
    rng = np.random.RandomState(seed)
    array = rng.normal(size=(rows, target.DIMENSION)).astype("<f4")
    path.write_bytes(array.tobytes(order="C") + trailing)
    return array


def _small_sources(tmp_path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    roots: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}
    for index, corpus in enumerate(("fineweb", "redpajama", "pile")):
        root = tmp_path / corpus
        root.mkdir()
        trailing = b"abc" if corpus == "fineweb" else b""
        arrays[corpus] = _write_raw(
            root / "data-00000-of-00001.npy",
            rows=5,
            seed=2500 + index,
            trailing=trailing,
        )
        roots[corpus] = str(root)
    return roots, arrays


def test_source_plan_uses_complete_rows_and_records_trailing_fragment(
        tmp_path, monkeypatch):
    monkeypatch.setattr(target, "FIRST_ROWS_PER_CORPUS", 3)
    roots, _arrays = _small_sources(tmp_path)

    plan = target.build_source_plan(roots, hash_files=True)

    assert plan["schema"] == "round0025-source-plan-v1"
    assert plan["universes"]["minilm-int8-150m"]["rows"] == 9
    assert plan["universes"]["minilm-int8-405m"]["rows"] == 15
    assert len(plan["sources"]) == 3
    assert len(plan["trailing_fragments"]) == 1
    fragment = plan["trailing_fragments"][0]
    assert fragment["corpus"] == "fineweb"
    assert fragment["bytes"] == 3
    assert fragment["full_rows"] == 5
    assert all(source["signature"]["sha256"] for source in plan["sources"])


def test_read_source_blocks_fails_closed_on_source_hash_change(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "FIRST_ROWS_PER_CORPUS", 3)
    roots, _arrays = _small_sources(tmp_path)
    plan = target.build_source_plan(roots, hash_files=True)
    path = Path(plan["sources"][0]["signature"]["canonical_path"])
    with path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(target.Round0025Error, match="source (grew|hash mismatch)"):
        list(target._read_source_blocks(plan["sources"][0], block_rows=2))


def test_quantize_full_writes_150m_subset_and_all_rows(
        tmp_path, monkeypatch):
    monkeypatch.setattr(target, "FIRST_ROWS_PER_CORPUS", 3)
    monkeypatch.setattr(target, "FULL_SAMPLE_SIZE", 6)
    monkeypatch.setattr(target, "_disk_free_gib", lambda path: 1000.0)
    roots, arrays = _small_sources(tmp_path)
    plan = target.build_source_plan(roots, hash_files=True)

    manifest = target.quantize_full(plan, str(tmp_path / "int8"), block_rows=2)

    assert manifest["schema"] == "round0025-int8-shards-v1"
    assert manifest["universes"]["minilm-int8-150m"]["rows"] == 9
    assert manifest["universes"]["minilm-int8-405m"]["rows"] == 15
    assert manifest["quantization_sanity"]["cosine"]["minimum"] >= 0.999
    assert manifest["quantization_sanity"]["threshold_passed"] is True

    q150 = np.memmap(
        tmp_path / "int8" / "minilm-int8-150m" / "embeddings.i8",
        dtype=np.int8,
        mode="r",
        shape=(9, target.DIMENSION),
    )
    expected_first = np.vstack([
        arrays["fineweb"][:3],
        arrays["redpajama"][:3],
        arrays["pile"][:3],
    ]).astype("<f4")
    expected_q, _expected_scales = target._quantize_block(expected_first)
    assert np.array_equal(np.asarray(q150), expected_q)

    all_file = tmp_path / "int8" / "minilm-int8-405m" / "embeddings.i8"
    assert all_file.stat().st_size == 15 * target.DIMENSION


def test_quantize_block_handles_zero_rows_without_nonfinite_scale():
    block = np.zeros((2, target.DIMENSION), dtype="<f4")

    quantized, scales = target._quantize_block(block)

    assert np.array_equal(quantized, np.zeros_like(quantized))
    assert scales.dtype == np.dtype("<f2")
    assert np.isfinite(scales).all()
    assert np.all(scales > 0)

from __future__ import annotations

import numpy as np
import pytest

from basemap.jina_historical_selection import (
    HISTORICAL_CORPORA,
    HistoricalJinaSelectionError,
    IndexedInventoryFp16Array,
    derive_first_eligible_historical_rows,
    evenly_spaced_validation_positions,
    map_dataset_rows_to_global,
    map_historical_positions,
    materialize_indexed_fp16_npy,
    validate_historical_provenance,
    verify_embedding_rows,
)


def _provenance() -> dict[str, np.ndarray]:
    # Pre-shuffle layout is FineWeb [10, 12, 14], RPJ [3, 8], Pile [1, 5, 9].
    # The output order deliberately crosses every corpus boundary.
    return {
        "seed": np.asarray(42, dtype=np.int64),
        "perm": np.asarray([3, 0, 5, 1, 6, 4, 2, 7], dtype=np.int64),
        "fineweb_idx": np.asarray([10, 12, 14], dtype=np.int64),
        "rpj_idx": np.asarray([3, 8], dtype=np.int64),
        "pile_idx": np.asarray([1, 5, 9], dtype=np.int64),
    }


def _selection(paths: tuple[str, str, str] | None = None) -> dict:
    ranges = []
    cursor = 0
    for corpus_id, (_, dataset) in enumerate(HISTORICAL_CORPORA):
        # Two ranges per dataset exercise shard-boundary mapping.
        for dataset_start, dataset_stop in ((0, 8), (8, 16)):
            item = {
                "dataset": dataset,
                "dataset_row_start": dataset_start,
                "dataset_row_stop": dataset_stop,
                "global_row_start": cursor,
                "global_row_stop": cursor + 8,
                "shard_row_start": 0,
                "shard_row_stop": 8,
            }
            if paths is not None:
                item["shard"] = {"canonical_path": paths[corpus_id]}
            ranges.append(item)
            cursor += 8
    return {"selected_rows": cursor, "ranges": ranges}


def test_exact_historical_mapping_crosses_corpora_and_shards() -> None:
    mapped = map_historical_positions(
        _provenance(),
        _selection(),
        np.arange(8, dtype=np.int64),
    )
    # Dataset offsets are 0, 16, 32 in the synthetic inventory.
    assert mapped["global_rows"].tolist() == [19, 10, 33, 12, 37, 24, 14, 41]
    assert mapped["dataset_rows"].tolist() == [3, 10, 1, 12, 5, 8, 14, 9]
    assert mapped["corpus_ids"].tolist() == [1, 0, 2, 0, 2, 1, 0, 2]
    assert map_dataset_rows_to_global(
        np.asarray([0, 7, 8, 15]),
        dataset=HISTORICAL_CORPORA[1][1],
        inventory_or_selection=_selection(),
    ).tolist() == [16, 23, 24, 31]


def test_eligible_selector_replaces_copies_instead_of_shrinking_prefix() -> None:
    selected = derive_first_eligible_historical_rows(
        _provenance(),
        _selection(),
        np.asarray([12, 19, 33], dtype=np.int64),
        target_rows=4,
    )
    # Raw prefix [19, 10, 33, 12] has three excluded rows.  Scan onward until
    # four eligible rows exist, preserving their historical order.
    assert selected["arrays"]["historical_positions"].tolist() == [1, 4, 5, 6]
    assert selected["arrays"]["global_rows"].tolist() == [10, 37, 24, 14]
    assert selected["summary"]["scan_rows"] == 7
    assert selected["summary"]["skipped_excluded_rows"] == 3
    assert selected["summary"]["raw_prefix_excluded_rows"] == 3
    assert selected["summary"]["replacement_rows_beyond_raw_prefix"] == 3
    assert selected["summary"]["raw_prefix_corpus_counts"] == [2, 1, 1]
    assert selected["summary"]["eligible_selector_corpus_counts"] == [2, 1, 1]


def test_provenance_and_selection_fail_closed() -> None:
    repeated = _provenance()
    repeated["perm"] = repeated["perm"].copy()
    repeated["perm"][0] = repeated["perm"][1]
    with pytest.raises(HistoricalJinaSelectionError, match="exact permutation"):
        validate_historical_provenance(repeated)

    with pytest.raises(HistoricalJinaSelectionError, match="outside"):
        map_dataset_rows_to_global(
            np.asarray([16]),
            dataset=HISTORICAL_CORPORA[0][1],
            inventory_or_selection=_selection(),
        )

    with pytest.raises(HistoricalJinaSelectionError, match="sorted, unique"):
        derive_first_eligible_historical_rows(
            _provenance(),
            _selection(),
            np.asarray([19, 12], dtype=np.int64),
            target_rows=4,
        )


def test_exact_embedding_validation_uses_inventory_shards(tmp_path) -> None:
    source_paths = []
    source_arrays = []
    for corpus_id in range(3):
        # Each synthetic dataset has two eight-row shards.
        dataset = np.arange(
            corpus_id * 160,
            corpus_id * 160 + 16 * 2,
            dtype=np.float16,
        ).reshape(16, 2)
        source_arrays.append(dataset)
        first = tmp_path / f"corpus-{corpus_id}-0.npy"
        second = tmp_path / f"corpus-{corpus_id}-1.npy"
        np.save(first, dataset[:8])
        np.save(second, dataset[8:])
        source_paths.append((str(first), str(second)))

    selection = _selection()
    for corpus_id in range(3):
        selection["ranges"][2 * corpus_id]["shard"] = {
            "canonical_path": source_paths[corpus_id][0]
        }
        selection["ranges"][2 * corpus_id + 1]["shard"] = {
            "canonical_path": source_paths[corpus_id][1]
        }
    mapped = map_historical_positions(
        _provenance(), selection, np.arange(8, dtype=np.int64)
    )
    historical = np.stack([
        source_arrays[corpus_id][dataset_row]
        for corpus_id, dataset_row in zip(
            mapped["corpus_ids"], mapped["dataset_rows"]
        )
    ])
    historical_path = tmp_path / "historical.npy"
    np.save(historical_path, historical)
    receipt = verify_embedding_rows(historical_path, mapped, selection)
    assert receipt["validated_rows"] == 8
    assert receipt["exact_array_equal"] is True
    assert receipt["source_shards_opened"] == 5

    historical[3, 0] += 1
    changed_path = tmp_path / "changed.npy"
    np.save(changed_path, historical)
    with pytest.raises(HistoricalJinaSelectionError, match="differ byte-for-byte"):
        verify_embedding_rows(changed_path, mapped, selection)


def test_evenly_spaced_validation_positions_are_unique_and_cover_ends() -> None:
    assert evenly_spaced_validation_positions(10, count=4).tolist() == [0, 3, 6, 9]
    assert evenly_spaced_validation_positions(3, count=10).tolist() == [0, 1, 2]


def test_indexed_inventory_fp16_array_preserves_arbitrary_order(tmp_path) -> None:
    source_paths = []
    source_arrays = []
    for corpus_id in range(3):
        dataset = np.arange(
            corpus_id * 160,
            corpus_id * 160 + 16 * 2,
            dtype=np.float16,
        ).reshape(16, 2)
        source_arrays.append(dataset)
        first = tmp_path / f"indexed-{corpus_id}-0.npy"
        second = tmp_path / f"indexed-{corpus_id}-1.npy"
        np.save(first, dataset[:8])
        np.save(second, dataset[8:])
        source_paths.append((first, second))

    selection = _selection()
    for corpus_id in range(3):
        for shard_id in range(2):
            path = source_paths[corpus_id][shard_id]
            selection["ranges"][2 * corpus_id + shard_id]["shard"] = {
                "canonical_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": f"synthetic-{corpus_id}-{shard_id}",
                "rows": 8,
            }
    ordered_global_rows = np.asarray([41, 10, 24, 19, 33, 14], dtype=np.int64)
    source = IndexedInventoryFp16Array(
        ordered_global_rows,
        selection,
        dimension=2,
    )
    expected = np.stack([
        source_arrays[2][9],
        source_arrays[0][10],
        source_arrays[1][8],
        source_arrays[1][3],
        source_arrays[2][1],
        source_arrays[0][14],
    ])
    assert source.shape == (6, 2)
    assert source.dtype == np.dtype("<f2")
    assert np.array_equal(source[:], expected)
    assert np.array_equal(source[[5, 0, 2]], expected[[5, 0, 2]])
    assert np.array_equal(source[-1], expected[-1])
    assert len(source.segments) == 6
    staged_path = tmp_path / "staged.npy"
    signature = materialize_indexed_fp16_npy(
        staged_path, source, block_rows=2
    )
    assert signature["canonical_path"] == str(staged_path)
    assert np.array_equal(
        np.load(staged_path, mmap_mode="r", allow_pickle=False), expected
    )
    with pytest.raises(IndexError, match="logical row"):
        _ = source[6]


def test_indexed_inventory_fp16_array_rejects_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "rows.npy"
    np.save(path, np.zeros((8, 2), dtype=np.float16))
    selection = {
        "selected_rows": 8,
        "ranges": [{
            "dataset": "only",
            "dataset_row_start": 0,
            "dataset_row_stop": 8,
            "global_row_start": 0,
            "global_row_stop": 8,
            "shard_row_start": 0,
            "shard_row_stop": 8,
            "shard": {
                "canonical_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": "synthetic",
                "rows": 8,
            },
        }],
    }
    with pytest.raises(HistoricalJinaSelectionError, match="unique"):
        IndexedInventoryFp16Array(
            np.asarray([1, 1], dtype=np.int64), selection, dimension=2
        )

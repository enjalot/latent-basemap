from __future__ import annotations

import inspect

import numpy as np

from basemap.round0087_inventory import (
    ENGLISH_BUDGETS,
    FINEWEB,
    MULTILINGUAL_BASE,
    MULTILINGUAL_REMAINDER,
    POLISH,
    TARGET_ROWS,
    build_selection,
    duplicate_census,
    inspect_shard,
    inventory_datasets,
    registered_budgets,
)
from experiments import prepare_round0087_queue, round0087_nodes


def _inventory(rows: int = 2_000_000) -> dict:
    values = {
        name: {
            "rows": budget,
            "shards": [{
                "canonical_path": f"/data/{name}.npy",
                "sha256": "a" * 64,
                "bytes": budget * 768 * 2 + 128,
                "rows": budget,
            }],
        }
        for name, budget in ENGLISH_BUDGETS.items()
    }
    codes = [
        "arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
        "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
        "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
        "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn",
    ]
    for code in codes:
        name = f"fineweb2-{code}-chunked-500-jina-v5-nano"
        values[name] = {
            "rows": rows,
            "shards": [{
                "canonical_path": f"/data/{name}.npy",
                "sha256": "b" * 64,
                "bytes": rows * 768 * 2 + 128,
                "rows": rows,
            }],
        }
    return values


def test_registered_mix_is_exactly_25m_and_withholds_polish() -> None:
    budgets = registered_budgets(_inventory())
    assert sum(budgets.values()) == TARGET_ROWS == 25_000_000
    assert POLISH not in budgets
    multilingual = [
        value
        for key, value in budgets.items()
        if key.startswith("fineweb2-")
    ]
    assert len(multilingual) == 19
    assert multilingual.count(MULTILINGUAL_BASE + 1) == (
        MULTILINGUAL_REMAINDER
    )
    assert multilingual.count(MULTILINGUAL_BASE) == (
        19 - MULTILINGUAL_REMAINDER
    )


def test_selection_uses_fixed_budgets_without_redistribution() -> None:
    inventory = _inventory()
    missing_name = next(
        name for name in inventory
        if name.startswith("fineweb2-") and name != POLISH
    )
    inventory[missing_name]["rows"] = 10
    inventory[missing_name]["shards"][0]["rows"] = 10
    selection = build_selection(inventory)
    assert selection["complete"] is False
    assert len(selection["gaps"]) == 1
    assert selection["gaps"][0]["dataset"] == missing_name
    assert selection["selected_rows"] < TARGET_ROWS
    assert selection["budgets"][missing_name] > 10


def test_duplicate_census_crosses_shard_boundaries_and_excludes_bad_rows(
    tmp_path,
) -> None:
    first = np.asarray([
        np.ones(768),
        np.full(768, 2),
        np.zeros(768),
        np.r_[np.nan, np.ones(767)],
    ], dtype="<f2")
    second = np.asarray([
        np.ones(768),
        np.full(768, 3),
        np.full(768, 2),
    ], dtype="<f2")
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    selection = {
        "selected_rows": 7,
        "ranges": [
            {
                "global_row_start": 0,
                "global_row_stop": 4,
                "shard_row_start": 0,
                "shard_row_stop": 4,
                "shard": {"canonical_path": str(first_path)},
            },
            {
                "global_row_start": 4,
                "global_row_stop": 7,
                "shard_row_start": 0,
                "shard_row_stop": 3,
                "shard": {"canonical_path": str(second_path)},
            },
        ],
    }
    census = duplicate_census(selection)
    arrays = census["arrays"]
    assert arrays["zero_rows"].tolist() == [2]
    assert arrays["nonfinite_rows"].tolist() == [3]
    assert arrays["representative_rows"].tolist() == [0, 1]
    assert arrays["duplicate_excluded_rows"].tolist() == [4, 6]
    assert arrays["duplicate_representative_rows"].tolist() == [0, 1]
    assert arrays["excluded_rows"].tolist() == [2, 3, 4, 6]
    assert census["summary"]["retained_row_count"] == 3
    assert census["summary"]["fingerprint_collision_splits"] == 0


def test_inventory_itemizes_invalid_shards_and_rejects_trailing_bytes(
    tmp_path,
) -> None:
    dataset = tmp_path / FINEWEB
    dataset.mkdir()
    good = dataset / "000.npy"
    trailing = dataset / "001.npy"
    np.save(good, np.ones((2, 768), dtype="<f2"))
    np.save(trailing, np.ones((3, 768), dtype="<f2"))
    with open(trailing, "ab") as handle:
        handle.write(b"unregistered trailing bytes")

    try:
        inspect_shard(str(trailing))
    except Exception as exc:
        assert "trailing or incomplete bytes" in str(exc)
    else:
        raise AssertionError("trailing shard bytes were accepted")

    inventory = inventory_datasets(tmp_path)
    observed = inventory[FINEWEB]
    assert observed["enumerated_shard_count"] == 2
    assert len(observed["shards"]) == 1
    assert observed["shards"][0]["rows"] == 2
    assert len(observed["invalid_shards"]) == 1
    assert observed["invalid_shards"][0]["file"]["bytes"] > 0


def test_queue_is_one_cpu_io_heavy_nontraining_job() -> None:
    source = inspect.getsource(prepare_round0087_queue.prepare_round0087)
    assert source.count('"action": "inventory"') == 1
    assert '"queue_class"] = "cpu-io-heavy"' in source
    assert "gpu_hours_cap=0.0" in source
    assert '"must_not_overlap_active_gpu_queue": True' in source
    assert '"training_performed": False' in source


def test_result_releases_capability_only_for_complete_selection() -> None:
    source = inspect.getsource(round0087_nodes.run_inventory)
    assert 'selection["complete"] is True' in source
    assert 'selection["selected_rows"] == TARGET_ROWS' in source
    assert '"capability_ready": capability_ready' in source
    assert "if capability_ready else None" in source

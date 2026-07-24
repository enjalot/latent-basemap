from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.round0019_program import TRAIN_CONFIG as R0019_CONFIG
from basemap.round0034_program import INT8_SHA256, SCALES_SHA256
from basemap.round0036_pipeline import (
    COORDINATE_SCHEMA,
    TRANSFORM_SCHEMA,
    CoordinateStream,
    EncodedInt8Array,
    RetainedArrayView,
    RetainedFaissIndex,
    RetainedRowSelector,
    estimated_queue_gpu_seconds,
    low_dim_search_work_model,
    load_reviewed_model,
    seal,
    validate_reviewed_model_bundle,
)
from experiments import run_round0036_node as node
from experiments import prepare_round0036_queue as prepare


def test_retained_rank_select_closes_long_exclusion_run() -> None:
    excluded = np.concatenate((
        np.array([1, 3, 8], dtype=np.int64),
        np.arange(100, 141_258, dtype=np.int64),
    ))
    selector = RetainedRowSelector(excluded, row_count=150_000)
    expected = np.setdiff1d(np.arange(150_000, dtype=np.int64), excluded)
    positions = np.array([0, 1, 2, 7, 50, len(expected) - 2, len(expected) - 1])
    global_rows = selector.compact_to_global(positions)
    assert np.array_equal(global_rows, expected[positions])
    assert np.array_equal(selector.global_to_compact(global_rows), positions)
    assert np.all(selector.is_retained(global_rows))
    assert not np.any(selector.is_retained(excluded[[0, 3, -1]]))


def test_retained_bitmap_clears_every_adjacent_exclusion() -> None:
    excluded = np.array([1, 2, 3, 7, 8, 9, 10, 31], dtype=np.int64)
    selector = RetainedRowSelector(excluded, row_count=35)
    bitmap = selector.bitmap()
    observed = np.array([
        bool((bitmap[row >> 3] >> (row & 7)) & 1) for row in range(35)
    ])
    expected = np.ones(35, dtype=bool)
    expected[excluded] = False
    assert np.array_equal(observed, expected)
    assert not any((bitmap[-1] >> bit) & 1 for bit in range(3, 8))


def test_retained_array_view_never_addresses_excluded_rows() -> None:
    base = np.arange(60, dtype=np.float32).reshape(20, 3)
    selector = RetainedRowSelector(np.array([0, 2, 9, 19]), row_count=20)
    view = RetainedArrayView(base, selector)
    retained = np.array([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    assert view.shape == (16, 3)
    assert np.array_equal(view[:], base[retained])
    query = np.array([[0, 5], [10, 15]])
    assert np.array_equal(view[query], base[retained[query]])


def test_faiss_bitmap_selector_excludes_large_family() -> None:
    faiss = pytest.importorskip("faiss")
    rng = np.random.RandomState(36)
    values = rng.normal(size=(500, 8)).astype(np.float32)
    quantizer = faiss.IndexFlatL2(8)
    index = faiss.IndexIVFFlat(quantizer, 8, 16)
    index.train(values)
    index.add(values)
    excluded = np.arange(1, 401, dtype=np.int64)
    selector = RetainedRowSelector(excluded, row_count=500)
    filtered = RetainedFaissIndex(index, selector, nprobe=16)
    _distances, global_rows = filtered.search_global(values[[0, 450]], 10)
    assert global_rows.shape == (2, 10)
    assert np.all(selector.is_retained(global_rows))
    _distances, compact = filtered.search_compact(values[[0]], 10)
    assert np.array_equal(selector.compact_to_global(compact), global_rows[:1])


def _r0034_config() -> dict:
    config = copy.deepcopy(R0019_CONFIG)
    config["schema"] = "round0034-production-config-v1"
    config["row_universe"] = {
        "rows": 150_000_000,
        "input_dimension": 384,
        "int8_sha256": INT8_SHA256,
        "scale_sha256": SCALES_SHA256,
    }
    config["optimizer"]["successful_positive_lr_updates"] = 123
    config["execution"] = {
        "expected_pipeline_stamp": {
            "pipeline": "host_int8_canonical",
            "sampler_class": "HostInt8CanonicalSampler",
            "x_residency": "host_int8_materialized",
            "positive_sampling": (
                "uniform-retained-positive-source-then-uniform-valid-canonical-"
                "destination-with-replacement"
            ),
            "negative_sampling": "uniform-R0033-retained-rows-nonself",
        }
    }
    return config


def _model_bundle(tmp_path: Path) -> tuple[dict, dict]:
    torch = pytest.importorskip("torch")
    config = _r0034_config()
    config_sha = sha256_bytes(canonical_json(config))
    layer = torch.nn.Linear(384, 2)
    model_path = tmp_path / "model.pt"
    torch.save({
        "state_dict": layer.state_dict(),
        "production_config": config,
        "production_config_sha256": config_sha,
    }, model_path)
    model_signature = expected_input_signature(model_path)
    stamp = config["execution"]["expected_pipeline_stamp"]
    receipt_body = {
        "schema": "round0034-train-receipt-v1",
        "round_id": "0034",
        "model": model_signature,
        "train_config": config,
        "train_config_sha256": config_sha,
        "train_accounting": {
            "budget_satisfied": True,
            "positive_lr_optimizer_steps": 123,
            "optimizer_steps_attempted": 123,
            "optimizer_steps_succeeded": 123,
            "amp_overflow_skips": 0,
            "nonfinite_loss_skips": 0,
            "nonfinite_gradient_skips": 0,
        },
        "exact_execution_receipt": dict(stamp),
    }
    receipt = seal(receipt_body)
    receipt_path = tmp_path / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    bundle = validate_reviewed_model_bundle(
        model_path=str(model_path),
        model_sha256=model_signature["sha256"],
        train_receipt_path=str(receipt_path),
        train_receipt_sha256=expected_input_signature(receipt_path)["sha256"],
    )
    return bundle, layer.state_dict()


def test_reviewed_model_loader_binds_custom_checkpoint_schema(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    bundle, expected_state = _model_bundle(tmp_path)

    class FakeMap:
        def __init__(self):
            self.model = None
            self.device = "cpu"

        def _init_model(self, input_dim: int) -> None:
            assert input_dim == 384
            self.model = torch.nn.Linear(384, 2)

    loaded = load_reviewed_model(
        bundle, device="cpu", model_factory=lambda _config: FakeMap()
    )
    assert loaded.is_fitted is True
    assert all(torch.equal(loaded.model.state_dict()[key], value)
               for key, value in expected_state.items())


def test_reviewed_model_bundle_rejects_execution_stamp_drift(tmp_path: Path) -> None:
    bundle, _state = _model_bundle(tmp_path)
    path = Path(bundle["train_receipt"]["canonical_path"])
    receipt = json.loads(path.read_text())
    receipt["exact_execution_receipt"]["pipeline"] = "silent-fallback"
    body = {key: value for key, value in receipt.items() if key != "identity_sha256"}
    receipt["identity_sha256"] = sha256_bytes(canonical_json(body))
    drift = tmp_path / "drift.json"
    drift.write_text(json.dumps(receipt))
    with pytest.raises(Exception, match="incomplete"):
        validate_reviewed_model_bundle(
            model_path=bundle["model"]["canonical_path"],
            model_sha256=bundle["model"]["sha256"],
            train_receipt_path=str(drift),
            train_receipt_sha256=expected_input_signature(drift)["sha256"],
        )


def _coordinate_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "coordinates"
    root.mkdir()
    members = []
    cursor = 0
    for index, rows in enumerate((4, 3)):
        chunk = root / f"chunk-{index:05d}"
        chunk.mkdir()
        path = chunk / "coordinates.npy"
        value = np.arange(cursor * 2, (cursor + rows) * 2, dtype=np.float32).reshape(rows, 2)
        np.save(path, value)
        signature = expected_input_signature(path)
        members.append({
            "chunk_index": index,
            "global_row_start": cursor,
            "global_row_stop": cursor + rows,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
        cursor += rows
    body = {
        "schema": TRANSFORM_SCHEMA,
        "row_accounting": {"all_rows": cursor},
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": cursor,
            "dimension": 2,
            "dtype": "<f4",
            "ordered_chunks": members,
        },
    }
    (root / "actual-transform.json").write_text(json.dumps(seal(body)))
    return root


def test_coordinate_stream_and_retained_view_preserve_global_alignment(tmp_path: Path) -> None:
    root = _coordinate_fixture(tmp_path)
    stream = CoordinateStream(str(root))
    assert np.array_equal(stream[[6, 0, 4]], np.array([[12, 13], [0, 1], [8, 9]], np.float32))
    selector = RetainedRowSelector(np.array([1, 5]), row_count=7)
    retained = RetainedArrayView(stream, selector)
    assert np.array_equal(retained[:], stream[[0, 2, 3, 4, 6]])
    assert np.array_equal(retained.min(0), np.array([0, 1], np.float32))
    assert np.array_equal(retained.max(0), np.array([12, 13], np.float32))


def test_coordinate_stream_restores_unsorted_repeated_fancy_index(tmp_path: Path) -> None:
    stream = CoordinateStream(str(_coordinate_fixture(tmp_path)))
    rows = np.array([[6, 0, 4], [2, 6, 1]], dtype=np.int64)
    expected = np.arange(14, dtype=np.float32).reshape(7, 2)[rows]
    assert np.array_equal(stream[rows], expected)
    assert len(stream._arrays) == 2


def test_r0036_p90_gpu_estimate_stays_under_eight_hours() -> None:
    estimate = estimated_queue_gpu_seconds()
    assert estimate["total"] == 28_200.0
    assert estimate["total"] / 3600 < 8.0


def test_low_dim_work_model_removes_thousands_of_full_corpus_passes() -> None:
    work = low_dim_search_work_model()
    assert work["anchor_tile"] == 1_000
    assert work["anchor_tiles"] == 10
    assert work["anchor_group"] == 5_000
    assert work["corpus_passes"] == 2
    assert work["k_fraction"] == 147_222
    assert work["candidate_width"] == 147_223
    assert work["corpus_chunks_per_anchor_tile"] == 295
    assert work["legacy_full_corpus_anchor_tile"] == 3
    assert work["legacy_full_corpus_passes"] == 3_334
    assert work["coordinate_bytes_total"] == 2_355_548_112
    assert work["legacy_coordinate_bytes_total"] == 3_926_698_702_704


def test_shape_stable_transform_pads_every_short_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    values[-1] = values[0]
    batch_shapes: list[tuple[int, int]] = []

    def fake_project(_model, block, *, batch_rows):
        batch_shapes.append(tuple(block.shape))
        assert batch_rows == 4
        return np.asarray(block[:, :2], dtype=np.float32)

    monkeypatch.setattr(node, "_project_float", fake_project)
    projected = node._project_encoded_block(
        object(), values, 0, len(values), batch_rows=4
    )
    assert batch_shapes == [(4, 3), (4, 3), (4, 3)]
    assert np.array_equal(projected[-1], projected[0])
    assert np.array_equal(projected, values[:, :2])


def test_low_dim_chunked_topk_matches_dense_exact() -> None:
    from basemap.panel_v2 import PanelV2Config, _self_knn

    rng = np.random.RandomState(360)
    coordinates = rng.normal(size=(53, 2)).astype(np.float32)
    anchors = np.array([0, 7, 19, 52], dtype=np.int64)
    config = PanelV2Config(corpus_chunk=7, block_elems=21)
    observed, distances, _guard = _self_knn(
        coordinates,
        anchors,
        9,
        config,
        hi_dim=False,
        want_dist=True,
    )
    for position, anchor in enumerate(anchors):
        delta = coordinates - coordinates[anchor]
        squared = np.einsum("ij,ij->i", delta, delta)
        squared[anchor] = np.inf
        expected = np.argsort(squared, kind="stable")[:9]
        assert np.array_equal(observed[position], expected)
        assert np.allclose(distances[position], np.sqrt(squared[expected]), atol=1e-6)


def test_retained_2d_canary_never_admits_excluded_rows() -> None:
    selector = RetainedRowSelector(np.array([1, 4, 7]), row_count=12)
    coordinates = np.arange(selector.retained_count * 2, dtype=np.float32).reshape(-1, 2)
    result = node._exact_retained_2d_slice_search(
        coordinates,
        selector,
        anchors=np.array([0, 3]),
        k=3,
    )
    assert result["passed"] is True
    assert result["all_candidates_retained"] is True
    assert result["self_excluded"] is True


def test_large_retained_scan_materializes_exact_reusable_rank_select() -> None:
    excluded = np.array([1, 17, 500_000, 999_999], dtype=np.int64)
    selector = RetainedRowSelector(excluded, row_count=1_000_004)
    compact = np.arange(100_000, 200_000, dtype=np.int64)

    observed = selector.compact_to_global(compact)
    cached = selector._compact_to_global_cache

    assert cached is not None
    assert cached.dtype == np.dtype("int64")
    assert cached.flags.writeable is False
    assert len(cached) == selector.retained_count
    assert np.array_equal(observed, cached[compact])
    assert np.all(selector.is_retained(observed))
    assert np.array_equal(
        selector.global_to_compact(observed),
        compact,
    )


def test_retained_prefix_identity_passes_strict_hid_reference_validation() -> None:
    from basemap.panel_v2 import (
        PanelV2Config,
        build_hiD_reference,
        sample_anchors,
        validate_hiD_reference,
    )

    rng = np.random.RandomState(36)
    base = rng.normal(size=(12, 3)).astype(np.float32)
    selector = RetainedRowSelector(np.array([2, 7]), row_count=len(base))
    retained = RetainedArrayView(base, selector)
    source_identity = {
        "kind": "ordered_array",
        "shape": [12, 3],
        "dtype": "<f4",
        "sha256": ordered_array_sha256(base),
    }
    identity = node._retained_prefix_reference_identity(
        retained,
        global_row_interval=(0, 12),
        source_identity=source_identity,
    )
    assert set(identity["data_identity"]) == {"kind", "shape", "dtype", "sha256"}
    assert identity["data_identity"]["kind"] == "ordered_array"
    assert identity["convention"]["global_row_interval"] == [0, 12]
    assert identity["convention"]["selector"] == selector.identity()
    assert identity["convention"]["source"] == source_identity

    config = PanelV2Config(
        frac=0.3,
        k_hit=2,
        k_density=2,
        n_anchors=4,
        anchor_seed=3,
        corpus_chunk=5,
        block_elems=20,
        overselect=2,
    )
    anchors = sample_anchors(len(retained), config).astype(np.int64)
    reference = build_hiD_reference(
        retained,
        anchors,
        config,
        data_identity=identity["data_identity"],
        convention=identity["convention"],
    )
    assert validate_hiD_reference(reference)["key_parts"]["convention"] == (
        identity["convention"]
    )


def test_production_canary_persists_failure_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_job, *, started):
        raise RuntimeError(f"fixture failed after {started > 0}")

    monkeypatch.setattr(node, "_production_canary_body", fail)
    output = tmp_path / "canary"
    with pytest.raises(Exception, match="production canary failed"):
        node.run_production_canary({}, {"outputs": [str(output)]})
    verdict = json.loads((output / "verdict.json").read_text())
    assert verdict["passed"] is False
    assert verdict["failure"]["type"] == "RuntimeError"
    body = {key: value for key, value in verdict.items() if key != "identity_sha256"}
    assert verdict["identity_sha256"] == sha256_bytes(canonical_json(body))


def test_accepted_training_review_must_name_exact_model_and_receipt(
    tmp_path: Path,
) -> None:
    model_sha = "a" * 64
    receipt_sha = "b" * 64
    review = tmp_path / "review.md"
    review.write_text(f"accepted model {model_sha}\ntrain receipt {receipt_sha}\n")
    prepare._assert_review_binds_artifacts(
        str(review),
        model_sha256=model_sha,
        train_receipt_sha256=receipt_sha,
    )
    review.write_text(f"accepted model {model_sha}\n")
    with pytest.raises(RuntimeError, match="does not bind"):
        prepare._assert_review_binds_artifacts(
            str(review),
            model_sha256=model_sha,
            train_receipt_sha256=receipt_sha,
        )


def test_registry_publication_requires_every_projection_explorer(
    tmp_path: Path,
) -> None:
    probes = {"dadabase", "trec-covid", "code", "science", "latin"}
    entries = [
        {
            "kind": "projection-map",
            "map_id": f"round-0036-{probe}",
            "projection": {"probe": probe},
        }
        for probe in probes
    ]
    for probe in probes - {"latin"}:
        page = tmp_path / "projections" / f"round-0036-{probe}" / "index.html"
        page.parent.mkdir(parents=True)
        page.write_text("working explorer")
    with pytest.raises(Exception, match="latin"):
        node._required_projection_page_signatures(
            entries, site_dir=tmp_path, required_probes=probes
        )
    latin = tmp_path / "projections/round-0036-latin/index.html"
    latin.parent.mkdir(parents=True)
    latin.write_text("working explorer")
    signatures = node._required_projection_page_signatures(
        entries, site_dir=tmp_path, required_probes=probes
    )
    assert set(signatures) == probes
    assert all(value["bytes"] > 0 for value in signatures.values())

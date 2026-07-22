import json

import numpy as np
import pytest


def test_prefix_l2_array_slices_then_normalizes_and_stamps(tmp_path):
    from basemap.data_loader import PrefixL2NormalizedArray

    path = tmp_path / "source.npy"
    source = np.array([
        [3.0, 4.0, 12.0, 7.0],
        [5.0, 12.0, 9.0, 8.0],
        [8.0, 15.0, 2.0, 1.0],
    ], dtype=np.float16)
    np.save(path, source)
    memmap = np.load(path, mmap_mode="r", allow_pickle=False)
    view = PrefixL2NormalizedArray(
        memmap, source_dimension=4, output_dimension=2,
        normalize=True, source_paths=[path])

    assert view.shape == (3, 2)
    assert view.dtype == np.dtype("float32")
    assert np.allclose(view[:2], [[0.6, 0.8], [5 / 13, 12 / 13]])
    assert np.allclose(np.linalg.norm(view[[2, 0]], axis=1), 1.0)
    assert np.allclose(view[1], [5 / 13, 12 / 13])
    assert view[:, 0].shape == (3,)
    assert view.loaded_shard_paths == [str(path.resolve())]
    stamp = view.execution_preprocessing_stamp
    assert stamp["input_source_dimension"] == 4
    assert stamp["input_effective_dimension"] == 2
    assert stamp["input_l2_renormalized"] is True
    assert len(stamp["input_preprocessing_sha256"]) == 64
    json.dumps(stamp)
    with pytest.raises(RuntimeError, match="materialize"):
        np.asarray(view)


def test_full_dimension_control_is_identity_cast(tmp_path):
    from basemap.data_loader import PrefixL2NormalizedArray

    path = tmp_path / "source.npy"
    source = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float16)
    np.save(path, source)
    view = PrefixL2NormalizedArray(
        np.load(path, mmap_mode="r"), source_dimension=2,
        output_dimension=2, normalize=False, source_paths=[path])
    assert np.array_equal(view[:], source.astype(np.float32))
    assert view.execution_preprocessing_stamp[
        "input_preprocessing_operation"] == "identity-fp32-cast"
    assert view.execution_preprocessing_stamp["input_l2_renormalized"] is False


def test_reduced_prefix_requires_renorm_and_rejects_bad_rows(tmp_path):
    from basemap.data_loader import PrefixL2NormalizedArray

    source = np.array([[0.0, 0.0, 1.0], [np.inf, 1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="must be L2-renormalized"):
        PrefixL2NormalizedArray(
            source, source_dimension=3, output_dimension=2, normalize=False)
    view = PrefixL2NormalizedArray(
        source, source_dimension=3, output_dimension=2, normalize=True)
    with pytest.raises(ValueError, match="zero/non-finite"):
        view[0]
    with pytest.raises(ValueError, match="non-finite"):
        view[1]


def test_trainer_accepts_explicit_manifest_and_stamps_preprocessing(tmp_path):
    from basemap.artifact_identity import sha256_file
    from basemap.data_loader import PrefixL2NormalizedArray
    from basemap.graph_validation import graph_manifest_v2, write_manifest
    from basemap.pumap.parametric_umap import ParametricUMAP

    source_path = tmp_path / "source.npy"
    rng = np.random.RandomState(27)
    np.save(source_path, rng.normal(size=(64, 4)).astype(np.float32))
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    view = PrefixL2NormalizedArray(
        source, source_dimension=4, output_dimension=2, normalize=True,
        source_paths=[source_path])
    sources = np.repeat(np.arange(64), 2).astype(np.int32)
    targets = ((sources + 1) % 64).astype(np.int32)
    graph_path = tmp_path / "graph.npz"
    np.savez(
        graph_path, sources=sources, targets=targets,
        weights=np.ones(len(sources), np.float32), n_nodes=64, k=2)
    manifest = graph_manifest_v2(
        sources, targets, 64, X=view, graph_path=str(graph_path),
        data_paths=[str(source_path)], k=2)
    manifest["input_preprocessing"] = view.execution_preprocessing_stamp
    manifest_path = tmp_path / "manifests" / "cell.json"
    manifest_path.parent.mkdir()
    write_manifest(str(manifest_path), manifest)

    model = ParametricUMAP(
        a=1.0, b=1.0, correlation_weight=0.0, n_epochs=2,
        batch_size=32, total_steps_estimate=4, lr_schedule="cosine",
        warmup_steps=0, device="cpu", positive_target_mode="binary",
        gpu_resident_data=False, use_amp=False,
        graph_manifest_path=str(manifest_path.resolve()),
        graph_manifest_sha256=sha256_file(manifest_path),
    )
    model.fit(view, precomputed_edges_path=str(graph_path))
    stats = model._train_stats
    assert stats["pipeline_input_source_dimension"] == 4
    assert stats["pipeline_input_effective_dimension"] == 2
    assert stats["pipeline_input_l2_renormalized"] is True
    assert stats["pipeline_input_preprocessing_sha256"] == \
        view.execution_preprocessing_stamp["input_preprocessing_sha256"]
    assert stats["verified_hashes"]["graph_manifest_sha256"] == \
        sha256_file(manifest_path)


def test_explicit_manifest_preprocessing_mismatch_fails_before_training(tmp_path):
    from basemap.artifact_identity import sha256_file
    from basemap.data_loader import PrefixL2NormalizedArray
    from basemap.graph_validation import graph_manifest_v2, write_manifest
    from basemap.pumap.parametric_umap import ParametricUMAP

    source_path = tmp_path / "source.npy"
    source = np.random.RandomState(2).normal(size=(32, 4)).astype(np.float32)
    np.save(source_path, source)
    view = PrefixL2NormalizedArray(
        np.load(source_path, mmap_mode="r"), source_dimension=4,
        output_dimension=2, normalize=True, source_paths=[source_path])
    sources = np.repeat(np.arange(32), 2).astype(np.int32)
    targets = ((sources + 1) % 32).astype(np.int32)
    graph_path = tmp_path / "graph.npz"
    np.savez(graph_path, sources=sources, targets=targets,
             weights=np.ones(len(sources), np.float32), n_nodes=32, k=2)
    manifest = graph_manifest_v2(
        sources, targets, 32, X=view, graph_path=str(graph_path),
        data_paths=[str(source_path)], k=2)
    manifest["input_preprocessing"] = {
        **view.execution_preprocessing_stamp,
        "input_effective_dimension": 3,
    }
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), manifest)
    model = ParametricUMAP(
        a=1.0, b=1.0, correlation_weight=0.0, n_epochs=1,
        batch_size=16, total_steps_estimate=1, lr_schedule="cosine",
        warmup_steps=0, device="cpu", positive_target_mode="binary",
        gpu_resident_data=False, use_amp=False,
        graph_manifest_path=str(manifest_path.resolve()),
        graph_manifest_sha256=sha256_file(manifest_path),
    )
    with pytest.raises(ValueError, match="preprocessing"):
        model.fit(view, precomputed_edges_path=str(graph_path))

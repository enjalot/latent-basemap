from __future__ import annotations

import numpy as np
import pytest

from basemap.round0042_pipeline import (
    DeviceCanonicalSampler,
    Round0042PipelineError,
    Round0042TrainingInput,
)


class _Dataset:
    def __init__(self, values):
        torch = pytest.importorskip("torch")
        self.tensor = torch.as_tensor(values, dtype=torch.float32)
        self.device = "cpu"

    def __len__(self):
        return len(self.tensor)

    def index_select(self, rows):
        return self.tensor.index_select(0, rows)


def _sampler() -> DeviceCanonicalSampler:
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    targets = np.full((8, 15), -1, dtype="<i4")
    degrees = np.zeros(8, dtype="u1")
    targets[0, :1] = [1]
    targets[1, :2] = [0, 3]
    targets[3, :3] = [0, 1, 4]
    targets[4, :1] = [3]
    targets[6, :2] = [0, 4]
    targets[7, :1] = [6]
    degrees[:] = [1, 2, 0, 3, 1, 0, 2, 1]
    return DeviceCanonicalSampler(
        _Dataset(values),
        targets=targets,
        degrees=degrees,
        excluded_rows=np.asarray([2, 5], dtype=np.int64),
        positive_source_count=6,
        valid_edge_count=int(degrees.sum()),
        batch_size=20,
        pos_ratio=0.5,
        random_state=42,
        graph_signature={"sha256": "g" * 64},
        eligibility_signature={"sha256": "e" * 64},
        device="cpu",
        upload_chunk=3,
    )


def test_sampler_never_draws_excluded_self_or_invalid_destination() -> None:
    sampler = _sampler()
    positive_source, positive_destination = sampler._draw_positive_pairs(
        10_000
    )
    source = positive_source.numpy()
    destination = positive_destination.numpy()
    assert not np.isin(source, [2, 5]).any()
    assert not np.isin(destination, [2, 5]).any()
    assert not np.any(source == destination)
    for src, dst in zip(source, destination):
        assert dst in sampler.targets_t[src, : sampler.degrees_t[src]].numpy()


def test_positive_source_law_is_degree_independent() -> None:
    sampler = _sampler()
    source, _destination = sampler._draw_positive_pairs(120_000)
    counts = np.bincount(source.numpy(), minlength=8)
    observed = counts[[0, 1, 3, 4, 6, 7]]
    # A wide deterministic guard catches accidental edge-uniform sampling:
    # degree-3 row 3 would otherwise appear roughly 3x degree-1 rows.
    assert observed.max() / observed.min() < 1.08


def test_valid_destination_slots_are_uniform() -> None:
    torch = pytest.importorskip("torch")
    sampler = _sampler()
    slots = sampler._draw_slots(torch.full((120_000,), 3)).numpy()
    counts = np.bincount(slots, minlength=3)
    assert counts.max() / counts.min() < 1.03


def test_negative_pairs_use_retained_nonself_universe() -> None:
    sampler = _sampler()
    source, destination = sampler._draw_negative_pairs(20_000)
    source = source.numpy()
    destination = destination.numpy()
    assert not np.isin(source, [2, 5]).any()
    assert not np.isin(destination, [2, 5]).any()
    assert not np.any(source == destination)


def test_batch_and_execution_stamp_are_literal() -> None:
    sampler = _sampler()
    source, destination, labels = next(iter(sampler))
    assert source.shape == destination.shape == (20, 3)
    assert int((labels == 1).sum()) == 10
    assert int((labels == 0).sum()) == 10
    stamp = sampler.execution_stamp()
    assert stamp["pipeline"] == "device_fp16_canonical"
    assert stamp["sampler_class"] == "DeviceCanonicalSampler"
    assert stamp["positive_source_count"] == 6
    assert stamp["valid_canonical_edge_count"] == 10


def test_sampler_rejects_excluded_positive_source() -> None:
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    targets = np.full((8, 15), -1, dtype="<i4")
    degrees = np.zeros(8, dtype="u1")
    targets[0, 0] = 1
    degrees[0] = 1
    targets[2, 0] = 1
    degrees[2] = 1
    with pytest.raises(
        Round0042PipelineError,
        match="positive-source universe",
    ):
        DeviceCanonicalSampler(
            _Dataset(values),
            targets=targets,
            degrees=degrees,
            excluded_rows=np.asarray([2, 5], dtype=np.int64),
            positive_source_count=2,
            valid_edge_count=2,
            batch_size=20,
            pos_ratio=0.5,
            random_state=42,
            graph_signature={"sha256": "g" * 64},
            eligibility_signature={"sha256": "e" * 64},
            device="cpu",
        )


def test_sampler_rejects_valid_edge_count_mismatch() -> None:
    sampler = _sampler()
    with pytest.raises(
        Round0042PipelineError,
        match="geometry",
    ):
        DeviceCanonicalSampler(
            sampler.dataset,
            targets=sampler.targets_t.numpy().astype("<i4"),
            degrees=sampler.degrees_t.numpy().astype("u1"),
            excluded_rows=np.asarray([2, 5], dtype=np.int64),
            positive_source_count=6,
            valid_edge_count=11,
            batch_size=20,
            pos_ratio=0.5,
            random_state=42,
            graph_signature={"sha256": "g" * 64},
            eligibility_signature={"sha256": "e" * 64},
            device="cpu",
        )


def test_core_dispatches_explicit_round0042_hook_before_generic_loading(
    monkeypatch,
) -> None:
    from basemap.pumap.parametric_umap import ParametricUMAP

    class Adapter:
        shape = (8, 3)

        def __init__(self):
            self.called = False

        def prepare_round0042_training(self, **kwargs):
            self.called = True
            assert kwargs["required_input_pipeline"] == (
                "device_fp16_canonical"
            )
            return self, object(), 17, {"pipeline": "sentinel"}, {
                "graph": "verified"
            }

    adapter = Adapter()
    model = ParametricUMAP(
        batch_size=20,
        pos_ratio=0.5,
        device="cpu",
        required_input_pipeline="device_fp16_canonical",
        gpu_resident_data=False,
    )
    monkeypatch.setattr(
        "basemap.pumap.parametric_umap.datasets.edge_list_dataset.load_edge_arrays",
        lambda *_args, **_kwargs: pytest.fail("generic graph loader called"),
    )
    dataset, loader, edges = model._prepare_edge_list_training(
        adapter,
        "/tmp/canonical-graph.json",
        8,
        False,
        42,
    )
    assert adapter.called
    assert dataset is adapter
    assert edges == 17
    assert model._pipeline_info == {"pipeline": "sentinel"}
    assert model._pipeline_verified_hashes == {"graph": "verified"}


def test_training_adapter_dispatches_the_literal_canonical_sampler() -> None:
    class SmallTrainingInput(Round0042TrainingInput):
        expected_shape = (8, 3)

    sampler = _sampler()
    targets = sampler.targets_t.numpy().astype("<i4")
    degrees = sampler.degrees_t.numpy().astype("u1")
    graph_path = "/tmp/canonical-graph-v1.json"
    graph = {
        "signature": {
            "canonical_path": graph_path,
            "sha256": "g" * 64,
        },
        "targets": targets,
        "degrees": degrees,
        "manifest": {
            "inputs": {
                "eligibility": {"sha256": "e" * 64},
            },
            "outputs": {
                "targets": {"sha256": "t" * 64},
                "degrees": {"sha256": "d" * 64},
            },
            "summary": {
                "eligibility_excluded_source_count": 2,
                "eligibility_retained_row_count": 6,
                "retained_positive_source_count": 6,
                "valid_canonical_edge_count": 10,
            },
        },
    }
    wrapper = SmallTrainingInput(
        sampler.dataset,
        graph=graph,
        excluded_rows=np.asarray([2, 5], dtype=np.int64),
        feature_signature={"sha256": "f" * 64},
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP(
        batch_size=20,
        pos_ratio=0.5,
        device="cpu",
        positive_target_mode="binary",
        required_input_pipeline="device_fp16_canonical",
        gpu_resident_data=False,
    )
    dataset, loader, edges = model._prepare_edge_list_training(
        wrapper,
        graph_path,
        8,
        False,
        42,
    )
    assert dataset is wrapper
    assert isinstance(loader, DeviceCanonicalSampler)
    assert edges == 10
    assert model._pipeline_info["pipeline"] == "device_fp16_canonical"
    assert model._pipeline_verified_hashes["features"]["sha256"] == (
        "f" * 64
    )

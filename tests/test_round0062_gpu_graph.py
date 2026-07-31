from __future__ import annotations

import importlib
import inspect

import pytest

from basemap.round0049_program import Round0049Error
from experiments import prepare_round0062_queue, round0062_nodes


def _signature(path: str, digest: str) -> dict:
    return {
        "canonical_path": path,
        "sha256": digest,
        "bytes": 123,
        "kind": "file",
    }


def _qualification() -> tuple[dict, dict, dict]:
    substrate = _signature("/data/substrate.json", "a" * 64)
    eligibility = _signature("/data/eligibility.npz", "b" * 64)
    value = {
        "validity_passed": True,
        "training_performed": False,
        "optimizer_updates": 0,
        "selected_nprobe": 40,
        "substrate": substrate,
        "eligibility": eligibility,
        "candidate_universe": {
            "physical_exclusions": 600_712,
            "retained_rows": 59_399_288,
            "filtered_index": _signature(
                "/data/balanced-60m.ivfpq",
                "c" * 64,
            ),
        },
        "quality": {
            "gpu_mean_recall_at_15_unambiguous": 0.901953125,
        },
        "checks": {
            "runtime_matches": True,
            "filtered_candidate_count": True,
        },
    }
    return value, substrate, eligibility


def test_r0062_qualification_requires_exact_released_geometry() -> None:
    value, substrate, eligibility = _qualification()
    filtered = round0062_nodes._validate_qualification(
        value,
        substrate_signature=substrate,
        eligibility_signature=eligibility,
    )
    assert filtered["sha256"] == "c" * 64
    assert round0062_nodes.NPROBE == 40
    assert round0062_nodes.EXPECTED_RETAINED_ROWS == 59_399_288

    value["selected_nprobe"] = 32
    with pytest.raises(
        Round0049Error,
        match="qualification capability changed",
    ):
        round0062_nodes._validate_qualification(
            value,
            substrate_signature=substrate,
            eligibility_signature=eligibility,
        )


def test_r0062_is_one_bounded_no_training_gpu_graph() -> None:
    prep = inspect.getsource(prepare_round0062_queue.prepare_round0062)
    node = inspect.getsource(round0062_nodes)
    assert prepare_round0062_queue.RELEASE_ROOT == (
        "/home/enjalot/code/latent-basemap-run"
    )
    assert "gpu_hours_cap=2.0" in prep
    assert '"total": 7_200.0' in prep
    assert '"id": "build_gpu_native_graph_balanced_60m"' in prep
    assert '"action": "build_gpu_graph"' in prep
    assert "training_performed" in prep
    assert '"action": "train"' not in prep
    assert "index_cpu_to_gpu" in node
    assert "_write_shard" in node
    assert "_assemble_graph" in node


def test_r0062_does_not_reuse_failed_cpu_shards() -> None:
    prep = inspect.getsource(prepare_round0062_queue)
    node = inspect.getsource(round0062_nodes)
    assert "round-0050" not in prep
    assert "round-0050" not in node
    assert "round-0059" in prep
    assert "physically removed" in node


def test_r0062_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.reload(round0062_nodes)
    importlib.reload(prepare_round0062_queue)
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"

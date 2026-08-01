"""CUDA-hidden contract, encoder, and preparer tests for R0143."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from basemap import round0143_prompted_multilingual as contract
from basemap.artifact_identity import expected_input_signature
from experiments import prepare_round0143_queue as prepare
from experiments import round0143_nodes as nodes


def _fixture_inventory(tmp_path) -> tuple[dict, str]:
    ranges = []
    catalog = {}
    for tranche in contract.LANGUAGE_TRANCHES:
        dataset = str(tranche["dataset"])
        raw = (
            tmp_path
            / "embeddings"
            / dataset
            / "train"
            / "000_00000.npy"
        )
        raw.parent.mkdir(parents=True)
        raw.write_bytes((dataset + " raw").encode())
        signature = expected_input_signature(str(raw))
        selected = {
                "dataset": dataset,
                "dataset_row_start": 0,
                "dataset_row_stop": contract.ROWS_PER_LANGUAGE,
                "language": tranche["language"],
                "shard": {
                    "canonical_path": str(raw),
                    "rows": 2_000_000,
                    "bytes": signature["bytes"],
                    "sha256": signature["sha256"],
                },
                "shard_row_start": 0,
                "shard_row_stop": contract.ROWS_PER_LANGUAGE,
            }
        if tranche["r0087_global_row_range"] is None:
            catalog[dataset] = {
                "dataset": dataset,
                "role": "heldout-language-probe",
                "language": tranche["language"],
                "rows": 2_000_000,
                "shards": [selected["shard"]],
            }
        else:
            selected["global_row_start"] = tranche["r0087_global_row_range"][0]
            selected["global_row_stop"] = tranche["r0087_global_row_range"][1]
            ranges.append(selected)
    return {
        "selection": {"ranges": ranges},
        "inventory": catalog,
    }, str(tmp_path / "chunks")


def test_registered_tranches_and_budget_close_exactly() -> None:
    assert [item["language"] for item in contract.LANGUAGE_TRANCHES] == [
        "pol_Latn",
        "por_Latn",
        "rus_Cyrl",
    ]
    assert [item["r0087_global_row_range"] for item in contract.LANGUAGE_TRANCHES] == [
        None,
        [19_151_824, 19_987_278],
        [19_987_278, 20_822_732],
    ]
    assert contract.CORPUS_ROWS == 2_506_362
    assert contract.production_payload_bytes() == 3_849_772_032
    assert contract.CHUNK_ROWS == 25_000
    assert contract.BATCH_SIZE == 16
    assert contract.expected_gpu_seconds() / 3_600 == pytest.approx(
        3.100881944444444
    )
    assert (
        len(contract.LANGUAGE_TRANCHES) * contract.node_p90_seconds() / 3_600
        == pytest.approx(3.914271929824561)
    )
    assert contract.worst_passing_gpu_seconds() == pytest.approx(
        17_609.08
    )
    assert contract.worst_passing_gpu_seconds() < contract.GPU_HOURS_CAP * 3_600
    assert contract.EMBED_WARNING_ROWS_PER_S > contract.EMBED_MINIMUM_ROWS_PER_S
    assert contract.CAPABILITY == (
        "jina-document-multilingual-pol-por-rus-2p506m-v1"
    )
    assert prepare.OUTPUT_NAMESPACE.endswith("multilingual-005-v1")
    assert prepare.TOKEN_LENGTH_CALIBRATION["pol_Latn"] == {
        "mean": 476.218,
        "p95": 628.10,
        "maximum": 745,
    }
    assert prepare.TOKEN_LENGTH_CALIBRATION["por_Latn"] == {
        "mean": 437.357,
        "p95": 587.0,
        "maximum": 670,
    }
    assert prepare.TOKEN_LENGTH_CALIBRATION["rus_Cyrl"] == {
        "mean": 432.372,
        "p95": 562.05,
        "maximum": 668,
    }


def test_source_layout_binds_exact_r0087_rows_and_both_byte_streams(
    tmp_path,
) -> None:
    inventory, text_root = _fixture_inventory(tmp_path)
    for tranche in contract.LANGUAGE_TRANCHES:
        source_name = str(tranche["dataset"]).removesuffix("-jina-v5-nano")
        text = (
            tmp_path
            / "chunks"
            / source_name
            / "train"
            / "000_00000.parquet"
        )
        text.parent.mkdir(parents=True)
        text.write_bytes((source_name + " text").encode())

    layout = contract.source_layout_from_inventory(
        inventory,
        text_root=text_root,
        parquet_inspector=lambda path: (2_000_000, "large_string"),
        npy_inspector=lambda path: ((2_000_000, 768), "<f2"),
    )
    assert [item["language"] for item in layout] == [
        "pol_Latn",
        "por_Latn",
        "rus_Cyrl",
    ]
    assert [item["dataset_row_range"] for item in layout] == [
        [0, 835_454],
        [0, 835_454],
        [0, 835_454],
    ]
    assert [item["r0087_global_row_range"] for item in layout] == [
        None,
        [19_151_824, 19_987_278],
        [19_987_278, 20_822_732],
    ]
    assert all(item["text"]["sha256"] for item in layout)
    assert all(item["accepted_raw_embedding"]["sha256"] for item in layout)
    contract.validate_source_layout(layout)

    broken = copy.deepcopy(inventory)
    broken["selection"]["ranges"][1]["global_row_start"] += 1
    with pytest.raises(contract.Round0143Error, match="malformed"):
        contract.source_layout_from_inventory(
            broken,
            text_root=text_root,
            parquet_inspector=lambda path: (2_000_000, "large_string"),
            npy_inspector=lambda path: ((2_000_000, 768), "<f2"),
        )


def test_encoder_uses_batch16_without_mutating_r0116(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_encode(model, texts, *, requested_batch_size):
        observed.update(
            model=model,
            texts=list(texts),
            requested_batch_size=requested_batch_size,
        )
        return np.zeros((len(texts), 768), dtype=np.float32), {
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": requested_batch_size,
            "oom_retries": 0,
        }

    monkeypatch.setattr(nodes, "_encode_document", fake_encode)
    model = object()
    values, telemetry = nodes._encode_multilingual_document(
        model, ["Document: one", "Document: two"]
    )
    assert values.shape == (2, 768)
    assert observed["requested_batch_size"] == 16
    assert telemetry["effective_batch_size"] == 16


def test_inherited_encoder_adaptively_retries_batch16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from experiments.round0116_nodes import _encode_document

    calls: list[int] = []

    class Model:
        def encode(self, texts, *, batch_size, **kwargs):
            calls.append(batch_size)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("synthetic CPU-only OOM")
            return np.zeros((len(texts), 768), dtype=np.float32)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    values, telemetry = _encode_document(
        Model(), ["Document: test"], requested_batch_size=16
    )
    assert values.shape == (1, 768)
    assert calls == [16, 8]
    assert telemetry == {
        "requested_batch_size": 16,
        "effective_batch_size": 8,
        "oom_retries": 1,
    }


def test_boundary_rehash_rejects_mutated_large_input_equivalent(tmp_path) -> None:
    path = tmp_path / "bound.bin"
    path.write_bytes(b"before")
    signature = expected_input_signature(str(path))
    roles = [
        "round",
        "review-0087",
        "review-0114",
        "inventory",
        "model-prompt-manifest",
        "model-member",
        "source-parquet",
        "raw-embedding",
    ]
    bindings = [
        {"role": role, "signature": signature}
        for role in roles
    ]
    observed = nodes._rehash_boundary_inputs(
        bindings, require_all_sources=False
    )
    assert len(observed) == len(bindings)
    path.write_bytes(b"after")
    with pytest.raises(contract.Round0143Error, match="bytes changed"):
        nodes._rehash_boundary_inputs(bindings, require_all_sources=False)


def test_preparer_materializes_three_independent_gpu_nodes_and_cpu_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    round_file = tmp_path / "round-0143.md"
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    bound_files = []
    for index in range(12):
        path = tmp_path / f"input-{index:02d}.bin"
        path.write_bytes(f"input {index}".encode())
        bound_files.append(expected_input_signature(str(path)))
    layout = []
    for index, tranche in enumerate(contract.LANGUAGE_TRANCHES):
        layout.append(
            {
                "node_id": tranche["node_id"],
                "language": tranche["language"],
                "dataset": tranche["dataset"],
                "dataset_row_range": [0, contract.ROWS_PER_LANGUAGE],
                "dataset_row_start": 0,
                "dataset_row_stop": contract.ROWS_PER_LANGUAGE,
                "corpus_global_row_range": tranche["corpus_global_row_range"],
                "r0087_global_row_range": tranche["r0087_global_row_range"],
                "inventory_role": tranche["inventory_role"],
                "shard_row_range": [0, contract.ROWS_PER_LANGUAGE],
                "shard_row_start": 0,
                "shard_row_stop": contract.ROWS_PER_LANGUAGE,
                "shard_rows": 2_000_000,
                "text_column": "chunk_text",
                "text_column_type": "large_string",
                "text": bound_files[6 + index],
                "accepted_raw_embedding": {
                    **bound_files[9 + index],
                    "rows": 2_000_000,
                    "dimension": 768,
                    "dtype": "<f2",
                    "selected_row_range": [0, contract.ROWS_PER_LANGUAGE],
                },
            }
        )
    environment = {
        "schema": "round0116-python-environment-freeze-v1",
        "python_executable": prepare.RUN_PYTHON,
        "python_prefix": prepare.RUN_ENVIRONMENT_PREFIX,
        "python_version": "test",
        "packages": [],
        "freeze_sha256": "0" * 64,
    }
    authenticated = {
        "reviews": {"0087": bound_files[0], "0114": bound_files[1]},
        "layout": layout,
        "inventory_manifest": bound_files[2],
        "r0114_model_prompt_manifest": bound_files[3],
        "model_members": [
            {**bound_files[4], "model_relative_path": "model.safetensors"},
            {**bound_files[5], "model_relative_path": "config.json"},
        ],
        "environment_freeze": environment,
        "disk": {
            "filesystem": "/data",
            "free_bytes_observed": 10**12,
            "payload_bytes": contract.production_payload_bytes(),
            "required_free_bytes": contract.required_free_bytes(),
            "passed": True,
        },
        "authentication_wall_s": 1.0,
    }
    monkeypatch.setattr(prepare, "_require_dedicated_run_environment", lambda: None)
    monkeypatch.setattr(prepare, "_require_issued_round", lambda: str(round_file))
    monkeypatch.setattr(prepare, "_authenticate_real_inputs", lambda: authenticated)
    monkeypatch.setattr(
        prepare,
        "ensure_data_directory",
        lambda path, **kwargs: (os.makedirs(path, exist_ok=True) or path),
    )
    queue_path = prepare.prepare_round0143(
        release_sha="a" * 40,
        queue_root=str(tmp_path / "queue"),
    )
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    assert queue["queue_class"] == "gpu-data-production-filler"
    assert queue["gpu_hours_cap"] == 5.0
    assert queue["required_reviews"] == ["0087", "0114"]
    assert queue["ordering_dependencies"] == []
    assert queue["capability_dependencies"] == [
        "jina-diverse-25m-inventory-v1",
        "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ]
    assert len(queue["jobs"]) == 4
    gpu_jobs = [job for job in queue["jobs"] if job["node_policy"]["gpu_required"]]
    assert [job["id"] for job in gpu_jobs] == [
        "embed_pol_Latn",
        "embed_por_Latn",
        "embed_rus_Cyrl",
    ]
    assert all(job["deps"] == [] for job in gpu_jobs)
    assert all(
        {item["role"] for item in job["authenticated_boundary_inputs"]}
        >= {"source-parquet", "raw-embedding", "model-member"}
        for job in gpu_jobs
    )
    finalizer = queue["jobs"][-1]
    assert finalizer["node_policy"]["gpu_required"] is False
    assert finalizer["deps"] == [job["id"] for job in gpu_jobs]
    assert queue["scientific_contract"]["no_graph"] is True
    assert queue["scientific_contract"]["no_training"] is True
    assert queue["scientific_contract"]["no_quality_claim"] is True


@pytest.mark.parametrize(
    ("python_executable", "python_prefix"),
    [
        ("/wrong/python", prepare.RUN_ENVIRONMENT_PREFIX),
        (prepare.RUN_PYTHON, "/wrong/prefix"),
    ],
)
def test_preparer_rejects_mutable_or_wrong_environment(
    monkeypatch: pytest.MonkeyPatch,
    python_executable: str,
    python_prefix: str,
) -> None:
    monkeypatch.setattr(prepare.sys, "executable", python_executable)
    monkeypatch.setattr(prepare.sys, "prefix", python_prefix)
    with pytest.raises(RuntimeError, match="dedicated run environment"):
        prepare._require_dedicated_run_environment()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basemap.output_safety import atomic_save_new_npz
from basemap.duplicate_census import save_cap_npz
from basemap.duplicate_multiplicity import load_duplicate_cap
from basemap.round0014_transform import build_transform_template
from basemap.round0021_program import TRAIN_CONFIG as ROUND0021_TRAIN_CONFIG
from basemap.round0024_program import train_config_for_cell
from experiments import run_round0014_node as node
from experiments.universality_panel import (
    _array_source_signature,
    _cosine_topk,
    _dtype_identity_receipt,
    _ffr_from_neighbors,
)


def test_round0020_global_cap_uses_duplicate_cap_schema(tmp_path):
    representatives = np.asarray([10, 20], dtype=np.int64)
    family_counts = np.asarray([3, 2], dtype=np.int64)
    member_rows = np.asarray([10, 11, 12, 20, 21], dtype=np.int64)
    offsets = np.asarray([0, 3, 5], dtype=np.int64)
    path = tmp_path / "global-cap-v1.npz"
    save_cap_npz(
        str(path),
        representative_rows=representatives,
        family_counts=family_counts,
        member_rows=member_rows,
        family_offsets=offsets,
        census_identity_sha256="0" * 64,
    )
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    loaded = load_duplicate_cap(
        str(path),
        expected_sha256=digest,
        row_count=30_000_000,
        fixed_edges_per_source=15,
    )
    assert loaded["representative_rows"].tolist() == [10, 20]
    assert loaded["excluded_rows"].tolist() == [11, 12, 21]
    assert loaded["metadata"]["retained_row_count"] == 29_999_997


def test_round0020_npz_writer_is_byte_deterministic(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    arrays = {
        "metadata": np.asarray('{"schema":"test"}'),
        "rows": np.asarray([3, 1, 4], dtype=np.int64),
    }
    atomic_save_new_npz(str(left), **arrays)
    atomic_save_new_npz(str(right), **arrays)
    assert left.read_bytes() == right.read_bytes()


def test_round0022_ffr_math_counts_true_neighbors_inside_hit_set():
    true_top10 = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]])
    hit_neighbors = np.asarray([[2, 4, 6, 8, 10, 12], [1, 2, 3, 4, 5, 6]])
    ffr, per_query = _ffr_from_neighbors(true_top10, hit_neighbors)
    assert per_query.tolist() == [0.5, 0.0]
    assert ffr == 0.25


def test_round0020_0022_configure_existing_runner_node_names():
    node.configure_round0020()
    assert node.ROUND_ID == "0020"
    assert node._run_canary.__name__ == "_run_round0020_duplicate_census"
    node.configure_round0022()
    assert node.ROUND_ID == "0022"
    assert node._run_canary.__name__ == "_run_round0022_canary"
    assert node._run_panel.__name__ == "_run_round0022_panel"
    assert node._run_semantic_renders.__name__ == "_run_round0022_renders"
    node.configure_round0028()
    assert node.ROUND_ID == "0028"
    assert node._run_canary.__name__ == "_run_round0028_canary"
    assert node._run_panel.__name__ == "_run_round0028_panel"
    assert node._run_semantic_renders.__name__ == "_run_round0028_renders"


def test_round0021_configure_uses_global_cap_artifact():
    node.configure_round0021()
    treatment = node.TRAIN_CONFIG["execution"]["duplicate_multiplicity"]
    assert node.ROUND_ID == "0021"
    assert node.SCHEMA_PREFIX == "round0021"
    assert treatment["artifact_sha256"] == (
        "9511ceca802da603bfbfe9164f8c6ffd7006df82df17b9499d4ed33288fde7cb"
    )
    assert treatment["excluded_rows"] == 218_242
    assert treatment["retained_rows"] == 29_781_758
    assert treatment["represented_exact_families"] == 138_601
    assert treatment["effective_positive_edges"] == 446_726_370
    assert node.TRAIN_CONFIG["graph"]["directed_edges_effective"] == 446_726_370


def test_round0021_top_family_gate_is_per_family():
    baseline = [
        {"rank": 1, "representative_row": 10, "family_count": 5, "representative_mahalanobis": 10.0},
        {"rank": 2, "representative_row": 20, "family_count": 4, "representative_mahalanobis": 2.0},
    ]
    report = node._compare_global_top_families(baseline, np.asarray([4.0, 1.1]))
    assert report["all_top50_representatives_reduced_50pct"] is False
    assert report["failed_representative_rows"] == [20]
    assert report["families"][0]["passed"] is True
    assert report["families"][1]["required_representative_mahalanobis"] == 1.0


def test_round0024_configure_uses_job_scoped_width():
    node.configure_round0024(job={"cell": "h4096"})
    assert node.ROUND_ID == "0024"
    assert node.SCHEMA_PREFIX == "round0024-h4096"
    assert node.TRAIN_CONFIG["model"]["hidden_dimension"] == 4096
    assert node.TRAIN_CONFIG["execution"]["round0024_capacity_ladder_cell"][
        "minimum_train_upd_s"
    ] == 25.0


def test_round0024_transform_template_binds_active_config_digest():
    config, digest = train_config_for_cell("h1024")
    assert ROUND0021_TRAIN_CONFIG["schema"] == "round0021-production-config-v1"
    template = build_transform_template(
        release_root=str(Path(__file__).resolve().parents[1]),
        release_sha="2" * 40,
        train_output_relative_path="artifacts/h1024/train/model.pt",
        production_config=config,
        production_config_sha256=digest,
    )
    assert template["template_config"]["production_config_sha256"] == digest
    assert template["template_config"]["architecture"]["hidden_dimension"] == 1024
    assert template["template_config"]["trained_model"][
        "controller_output_relative_path"
    ] == "artifacts/h1024/train/model.pt"
def test_round0028_dtype_receipt_fails_closed_on_fp32_narrowing(tmp_path):
    path = tmp_path / "probe.npy"
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    np.save(path, values, allow_pickle=False)
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    selected = np.asarray(loaded[[0, 2, 5]])
    source = _array_source_signature(str(path), loaded)
    receipt = _dtype_identity_receipt(
        label="synthetic.probe_queries",
        source=source,
        selected=selected,
        expected_source_dtype=np.dtype(np.float32).str,
        expected_selected_dtype=np.dtype(np.float32).str,
    )
    assert receipt["observed_selected_array"]["dtype"] == np.dtype(np.float32).str
    assert receipt["declared"]["cosine_compute_dtype"] == np.dtype(np.float32).str

    with pytest.raises(RuntimeError, match="selected_array_dtype"):
        _dtype_identity_receipt(
            label="synthetic.probe_queries",
            source=source,
            selected=selected.astype(np.float16),
            expected_source_dtype=np.dtype(np.float32).str,
            expected_selected_dtype=np.dtype(np.float32).str,
        )


def test_round0028_fp32_cosine_regression_changes_after_fp16_rounding():
    rng = np.random.default_rng(123)
    dim = 32
    corpus_rows = 16
    query_rows = 3
    corpus = rng.normal(size=(corpus_rows, dim)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.normal(size=(query_rows, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    for query_index in range(query_rows):
        noise = rng.normal(scale=0.002, size=dim).astype(np.float32)
        corpus[2 * query_index] = queries[query_index] + noise
        corpus[2 * query_index + 1] = (
            queries[query_index]
            + noise
            + rng.normal(scale=0.0002, size=dim).astype(np.float32)
        )

    fp32_order = _cosine_topk(corpus, queries, 5, device="cpu")
    rounded_order = _cosine_topk(
        corpus.astype(np.float16).astype(np.float32),
        queries.astype(np.float16).astype(np.float32),
        5,
        device="cpu",
    )
    assert fp32_order[1, :2].tolist() == [2, 3]
    assert rounded_order[1, :2].tolist() == [3, 2]

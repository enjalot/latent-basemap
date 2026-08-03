"""Contract tests for the R0179 NUMAP 200k OOS baseline."""
from __future__ import annotations

import types

import numpy as np
import pytest

from basemap.round0179_numap_baseline import (
    CAPABILITY,
    HELD_HASH,
    N_QUERIES,
    ROWS,
    Round0179Error,
    build_synthesis,
    validate_execution,
)


def _execution() -> dict:
    return {
        "schema": "round0179-numap-reference-execution-v1",
        "mode": "real",
        "package_versions": {
            "numap": "0.2.3",
            "grease-embeddings": "0.1.5",
        },
        "config": {
            "n_neighbors": 10,
            "min_dist": 0.1,
            "metric": "cosine",
            "n_components": 2,
            "se_dim": 5,
            "se_neighbors": 10,
            "random_state": 42,
            "lr": 1.0e-3,
            "epochs": 10,
            "batch_size": 64,
            "use_se": True,
            "use_residual_connections": True,
            "use_grease": True,
            "grease_batch_size": 1024,
            "grease_lr": 1.0e-3,
            "learn_from_se": True,
            "negative_sample_rate": 5,
            "use_concat": False,
            "use_alpha": False,
            "alpha": 0.0,
            "init_method": "identity",
            "grease_hiddens": [128, 256, 256],
            "use_true_eigenvectors": True,
        },
        "cuda_available": True,
        "train_rows": ROWS,
        "query_rows": N_QUERIES,
        "dimension": 768,
        "train_accounting": {
            "selected_pipeline": (
                "numap==0.2.3 official-example GrEASE spectral extension + "
                "residual PUMAP encoder"
            ),
            "grease_architecture_actual": [128, 256, 256, 6],
            "grease_optimizer_updates": 100,
            "pumap_optimizer_updates": 31_250,
            "pumap_expected_updates": 31_250,
        },
        "checkpoint": {"reload_max_abs_error": 1.0e-6},
        "train_coordinates": {
            "shape": [ROWS, 2],
            "axis_standard_deviation": [1.0, 2.0],
        },
        "query_coordinates": {
            "shape": [N_QUERIES, 2],
            "axis_standard_deviation": [1.0, 2.0],
        },
    }


def _cell() -> dict:
    return {
        "schema": "round0179-numap-cell-v1",
        "round_id": "0179",
        "rows": ROWS,
        "n_queries": N_QUERIES,
        "held_hash": HELD_HASH,
        "guards_passed": True,
        "execution": _execution(),
        "heldout_projection": {"ffr": 0.4, "recall_at_10": 0.03},
    }


def _aumap() -> dict:
    return {
        "round_id": "0175",
        "outcome": "aumap-oos-baseline-measured",
        "scales": {
            "200k": {
                "aumap_inverse_distance": {
                    "ffr": 0.323410,
                    "recall_at_10": 0.040625,
                }
            }
        },
    }


def test_execution_requires_exact_reference_path_and_updates() -> None:
    execution = _execution()
    validate_execution(execution)
    execution["config"]["use_residual_connections"] = False
    with pytest.raises(Round0179Error):
        validate_execution(execution)


def test_execution_rejects_incomplete_pumap_training() -> None:
    execution = _execution()
    execution["train_accounting"]["pumap_optimizer_updates"] -= 1
    with pytest.raises(Round0179Error):
        validate_execution(execution)


def test_synthesis_is_diagnostic_and_preserves_paired_arithmetic() -> None:
    synthesis = build_synthesis(cell=_cell(), aumap_context=_aumap())
    assert synthesis["outcome"] == "numap-grease-oos-baseline-measured"
    assert synthesis["capabilities"] == [CAPABILITY]
    delta = synthesis["comparison_to_reviewed_r0175"]["numap_minus_aumap"]
    assert delta["ffr"] == pytest.approx(0.07659)
    assert delta["recall_at_10"] == pytest.approx(-0.010625)
    assert synthesis["production_or_publishing"] is False


def test_synthesis_rejects_wrong_held_universe() -> None:
    cell = _cell()
    cell["held_hash"] = "wrong"
    with pytest.raises(Round0179Error):
        build_synthesis(cell=cell, aumap_context=_aumap())


def test_numap_adapter_keeps_import_caches_out_of_reference_output(
    tmp_path, monkeypatch
) -> None:
    from experiments import round0179_nodes as nodes

    held = np.arange(nodes.N_QUERIES, dtype=np.int64)
    queries = np.zeros((nodes.N_QUERIES, 4), dtype=np.float32)
    observed: dict[str, object] = {}

    monkeypatch.setattr(nodes, "_signature", lambda expected, *, label: dict(expected))
    monkeypatch.setattr(nodes, "_heldout_queries", lambda scale: (None, held, queries))
    monkeypatch.setattr(nodes, "_ids_hash", lambda ids: nodes.HELD_HASH)

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        kwargs["stderr"].write("stopped before package execution\n")
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(nodes.subprocess, "run", fake_run)

    output = tmp_path / "numap-200k"
    job = {
        "outputs": [str(output)],
        "testbed_embeddings": {"canonical_path": "/tmp/frozen-matrix.npy"},
        "sample_indices": {},
        "source_manifest": {},
        "source_shards": [],
        "reference_script": {},
        "toolchain_python": {"resolved_interpreter": {}, "pyvenv_config": {}},
        "package_files": [],
    }

    with pytest.raises(nodes.Round0179Error):
        nodes.run_numap_cell({"manifest": {"release_sha": "test"}}, job)

    assert observed["cwd"] == str(output / "reference")
    assert observed["env"]["MPLCONFIGDIR"] == str(output / "numap-mplconfig")

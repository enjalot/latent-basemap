"""Execute R0160's shared prompted-reference four-seed panel."""
from __future__ import annotations

import copy
import gc
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.panel_v2 import (
    _hiD_reference_content_sha256,
    _hiD_reference_payloads,
    _label_by_centroids,
    hiD_reference_key,
    load_hiD_reference,
    sample_anchors,
    save_hiD_reference,
    score_panel,
    validate_hiD_reference,
)
from basemap.round0104_training import L2NormalizedArray
from basemap.round0113_prompt_contrast import (
    read_sealed,
    seal,
    validate_seal,
)
from basemap.round0160_prompted_seed_family import (
    CAPABILITY,
    DIMENSION,
    NEW_SEEDS,
    ROUND_ID,
    ROWS,
    SEEDS,
    Round0160Error,
    build_family_evidence,
    metric_view,
)
from experiments import round0113_nodes as prompt_nodes
from experiments.score_complete_panel import frozen_centroids


EXPECTED_SCORE_ROUNDS = {42: "0115", 43: "0117", 44: ROUND_ID, 45: ROUND_ID}


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0160Error(f"{label} is unavailable or changed") from error


def _score_cell(cell: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    seed = int(cell.get("seed", -1))
    if seed not in SEEDS:
        raise Round0160Error("prompted family seed changed")
    score_path = str(cell.get("score_path") or "")
    score_signature = _signature(score_path, label=f"seed-{seed} native score")
    score = read_sealed(score_path, label=f"seed-{seed} native prompted score")
    coordinates_path = str(cell.get("coordinates_path") or "")
    coordinates_signature = _signature(
        coordinates_path, label=f"seed-{seed} coordinates"
    )
    coordinates = score.get("coordinates")
    if (
        score.get("round_id") != EXPECTED_SCORE_ROUNDS[seed]
        or score.get("arm") != "document"
        or int(score.get("training_seed", 42 if seed == 42 else -1)) != seed
        or not isinstance(coordinates, Mapping)
        or coordinates.get("training") != coordinates_signature
        or not all(bool(value) for value in (score.get("execution_gates") or {}).values())
    ):
        raise Round0160Error(f"seed-{seed} native score identity changed")
    train_path = str((score.get("train_receipt") or {}).get("canonical_path") or "")
    if _signature(train_path, label=f"seed-{seed} train receipt") != score["train_receipt"]:
        raise Round0160Error(f"seed-{seed} train receipt bytes changed")
    return seed, score, score_signature


def _extend_reference(
    *,
    source: Any,
    assembly: Mapping[str, Any],
    accepted_reference_path: str,
    centroids: Mapping[int, Any],
) -> dict[str, Any]:
    """Attach prompted labels without recomputing map-independent neighbours."""
    cfg = prompt_nodes.panel_config()
    anchors = sample_anchors(ROWS, cfg)
    data_identity = prompt_nodes._data_identity(assembly, arm="document")
    convention = {
        "row_order": "R0113 shared source/raw/document union-representative compact order",
        "distance": "cosine via fp32-L2-normalized squared L2",
        "self_exclusion": True,
        "anchor_namespace": "R0113 compact IDs",
        "embedding_prompt": "document",
    }
    reference_identity = {"data_identity": data_identity, "convention": convention}
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * ROWS)))
    expected_old_key, expected_old_parts = hiD_reference_key(
        source,
        anchors,
        cfg,
        None,
        kf=kf,
        **reference_identity,
    )
    old = load_hiD_reference(
        accepted_reference_path,
        expected_key=expected_old_key,
        expected_key_parts=expected_old_parts,
    )
    key, parts = hiD_reference_key(
        source,
        anchors,
        cfg,
        centroids,
        kf=kf,
        **reference_identity,
    )
    extended = copy.deepcopy(old)
    extended["key"] = key
    extended["key_parts"] = parts
    extended["labels"] = _label_by_centroids(source, centroids)
    extended["payloads"] = _hiD_reference_payloads(extended)
    extended["content_sha256"] = _hiD_reference_content_sha256(extended)
    validate_hiD_reference(extended, expected_key=key, expected_key_parts=parts)
    return extended


def run_family_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0160Error("R0160 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0160Error("R0160 full prompted panel requires CUDA")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0160 prompted seed-family panel"
    )
    started = time.monotonic()

    assembly_path = str(job["assembly_manifest"])
    assembly = read_sealed(assembly_path, label="R0113 compact assembly")
    document_signature = _signature(
        str(job["document_compact"]), label="R0113 document compact matrix"
    )
    if (
        assembly.get("round_id") != "0113"
        or int(assembly.get("retained_rows", -1)) != ROWS
        or int(assembly.get("dimension", -1)) != DIMENSION
        or assembly.get("outputs", {}).get("document") != document_signature
        or assembly.get("paired_row_population_identical") is not True
    ):
        raise Round0160Error("R0113 prompted population changed")
    source_raw = np.memmap(
        document_signature["canonical_path"],
        dtype=np.dtype("<f2"),
        mode="r",
        shape=(ROWS, DIMENSION),
    )
    source = L2NormalizedArray(source_raw)

    centroid_root = create_fresh_directory(
        os.path.join(output, "centroids"), label="R0160 prompted centroids"
    )
    centroids = frozen_centroids(source, (256, 1024), centroid_root, seed=0, iters=25)
    centroid_signatures = {
        str(k): _signature(
            os.path.join(centroid_root, f"centroids_k{k}.npy"),
            label=f"prompted k{k} centroids",
        )
        for k in (256, 1024)
    }
    reference = _extend_reference(
        source=source,
        assembly=assembly,
        accepted_reference_path=str(job["accepted_high_d_reference"]),
        centroids=centroids,
    )
    reference_path = os.path.join(output, "prompted-purity-reference.npz")
    save_hiD_reference(reference, reference_path)
    reference_signature = _signature(reference_path, label="prompted purity reference")
    reference_identity = {
        "data_identity": prompt_nodes._data_identity(assembly, arm="document"),
        "convention": {
            "row_order": "R0113 shared source/raw/document union-representative compact order",
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": "R0113 compact IDs",
            "embedding_prompt": "document",
        },
    }

    selected = job.get("cells")
    if not isinstance(selected, list) or {int(cell.get("seed", -1)) for cell in selected} != set(SEEDS):
        raise Round0160Error("R0160 four-seed input matrix changed")
    cells: dict[int, dict[str, Any]] = {}
    cfg = prompt_nodes.panel_config()
    for cell in sorted(selected, key=lambda item: int(item["seed"])):
        seed, native_score, native_signature = _score_cell(cell)
        coordinates_signature = _signature(
            str(cell["coordinates_path"]), label=f"seed-{seed} coordinates"
        )
        coordinates = np.load(
            coordinates_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        if (
            coordinates.shape != (ROWS, 2)
            or coordinates.dtype != np.dtype("float32")
            or not np.isfinite(np.asarray(coordinates[:4096])).all()
        ):
            raise Round0160Error(f"seed-{seed} coordinate geometry changed")
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "seed": seed,
                "source": document_signature,
                "coordinates": coordinates_signature,
                "native_score": native_signature,
                "shared_prompted_reference": reference_signature,
            },
        )
        metrics = metric_view(panel=panel, native_score=native_score)
        cells[seed] = {
            "seed": seed,
            "role": "new-training" if seed in NEW_SEEDS else "accepted-context-rescore",
            "native_score": native_signature,
            "train_receipt": dict(native_score["train_receipt"]),
            "coordinates": coordinates_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "panel": panel,
            "native_projection": dict(native_score["projections"]["matched"]),
            "polish_ood_diagnostic": dict(native_score.get("ood", {}).get("pol_Latn", {})),
            "decision_metrics": metrics,
        }
        del coordinates
        gc.collect()

    evidence = build_family_evidence(cells)
    receipt = seal({
        **evidence,
        "release_sha": active["manifest"]["release_sha"],
        "lineage": {
            "assembly": _signature(assembly_path, label="R0113 compact assembly"),
            "document_compact": document_signature,
            "accepted_high_d_reference": _signature(
                str(job["accepted_high_d_reference"]),
                label="R0115 document high-D reference",
            ),
            "accepted_reviews": [dict(item) for item in job["accepted_reviews"]],
        },
        "shared_prompted_reference": reference_signature,
        "centroids": centroid_signatures,
        "reference_reuse": {
            "map_independent_neighbor_and_radius_arrays_reused_byte_exact": True,
            "only_added_payload": "nearest prompted-centroid labels for every row",
            "centroid_recipe": "full 1,993,761-row GPU Lloyd k-means; seed 0; 25 iterations",
        },
        "new_trains": list(NEW_SEEDS),
        "old_maps_rescored_against_same_reference": [42, 43],
        "wall_seconds": time.monotonic() - started,
    })
    validate_seal(receipt, label="R0160 prompted seed-family evidence")
    atomic_write_new_json(
        os.path.join(output, "prompted-seed-family.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "score_prompted_seed_family":
        raise Round0160Error("unknown R0160 action")
    run_family_panel(active, job)

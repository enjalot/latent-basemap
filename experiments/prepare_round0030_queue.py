#!/usr/bin/env python3
"""Prepare the matched two-arm R0030 training queue."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0030_program import (
    CAP_SHA256,
    GRAPH_BUILD_CONTRACT_SHA256,
    GRAPH_BUILD_RECEIPT_SHA256,
    GRAPH_EFFECTIVE_EDGES,
    GRAPH_MANIFEST_SHA256,
    GRAPH_PATH,
    GRAPH_RESIDENT_EDGES,
    GRAPH_SHA256,
    OOD_RETENTION_NONINFERIORITY_MARGIN,
    R0023_LAYOUT_SHA256,
    R0023_SEED_SPREAD,
    train_config_for_arm,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    R0019_HIGH_D_REFERENCE,
    RUN_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
    _hf_snapshot_file_inputs,
    _materialized_chunk_inputs,
)


ROUND_FILE = os.path.join(LAB_ROOT, "round-0030-2026-07-21.md")
ROUND_ROOT = "/data/latent-basemap/runs/round-0030"
CAP_PATH = (
    "/data/latent-basemap/runs/round-0020/queue/artifacts/"
    "duplicate-census/global-cap-v1.npz"
)
BUILD_RECEIPT = (
    "/data/latent-basemap/runs/round-0029/queue/artifacts/"
    "weighted-graph-v2/build-receipt.json"
)
R0029_V3 = (
    "/data/latent-basemap/runs/round-0029/queue/artifacts/"
    "cpu-validation/v3-consumer-contract.json"
)
R0032_V4 = (
    "/data/latent-basemap/runs/round-0032/queue/artifacts/"
    "v4-requalification/v4-requalification-receipt.json"
)
R0032_V4_PAIRS = (
    "/data/latent-basemap/runs/round-0032/queue/artifacts/"
    "v4-requalification/v4-cuda-physical-check.json"
)
R0032_CANARY = (
    "/data/latent-basemap/runs/round-0032/queue/artifacts/"
    "production-canary/production-canary.json"
)
R0023_LAYOUT = (
    "/data/latent-basemap/runs/round-0023/queue/artifacts/"
    "layout-disparity/layout-disparity-v1.json"
)
R0028_PANEL = (
    "/data/latent-basemap/runs/round-0028/queue/artifacts/panel/"
    "universality-panel-v1.json"
)
R0023_PANELS = {
    "seed42": "/data/latent-basemap/runs/round-0019/queue/artifacts/panel/panel.json",
    "seed43": "/data/latent-basemap/runs/round-0023/queue/artifacts/seed43/panel/panel.json",
    "seed44": "/data/latent-basemap/runs/round-0023/queue/artifacts/seed44/panel/panel.json",
}
EXACT_EVIDENCE_SHA256 = {
    GRAPH_PATH: GRAPH_SHA256,
    GRAPH_PATH + ".manifest.json": GRAPH_MANIFEST_SHA256,
    BUILD_RECEIPT: GRAPH_BUILD_RECEIPT_SHA256,
    R0029_V3: "084016f28b4b8279f7d26c20c46ec356c78d3dac8538d1de52c4b2b940596066",
    R0032_V4: "9895541acc0f4667226f70785d308ab6e15c1742d00bc067d3bae229667394af",
    R0032_V4_PAIRS: "41a086fdd013c695631a563a739368a6b0ec4bda1250c24662b4427a2e8d7334",
    R0032_CANARY: "18bd15f3979b167bc8a16009533c214beec6a668e44f15277c166ec0444bc2c6",
    CAP_PATH: CAP_SHA256,
    R0023_LAYOUT: R0023_LAYOUT_SHA256,
    R0023_PANELS["seed42"]: (
        "2abfb6a5fe0ab3d4fbea67709d595cfe7c5d2b437468b2f19a2c6a0373334649"
    ),
    R0023_PANELS["seed43"]: (
        "d5cea47b2e1cc4eb9d5448da2d0b8e35a759e4270c22410a6d6cfccf5c61a5ba"
    ),
    R0023_PANELS["seed44"]: (
        "b15424fa20ee82c80f9cf92671ff73d4c005e44b7d22a6d0b3f4358399fe38f6"
    ),
    R0028_PANEL: "ffb4646bf8165ad2e1f26e0760426e3681e491dd463d02862c0b33ffa5c5d8c3",
}


def _transform_templates(queue_root: str, release_sha: str) -> dict[str, str]:
    from basemap.round0014_transform import build_transform_template

    inputs_root = ensure_data_directory(os.path.join(queue_root, "inputs"))
    templates: dict[str, str] = {}
    for arm in ("uniform", "fuzzy"):
        config, digest = train_config_for_arm(arm)
        template = build_transform_template(
            release_root=RUN_ROOT,
            release_sha=release_sha,
            train_output_relative_path=f"artifacts/{arm}/train/model.pt",
            production_config=config,
            production_config_sha256=digest,
        )
        path = os.path.join(inputs_root, f"{arm}-transform-spec-template.json")
        atomic_write_new_json(path, template, immutable=True)
        templates[arm] = path
    return templates


def _static_inputs(templates: dict[str, str]) -> list[dict[str, Any]]:
    files = [
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0013-2026-07-18-02.md"),
        os.path.join(LAB_ROOT, "review-0020-2026-07-19.md"),
        os.path.join(LAB_ROOT, "review-0023-2026-07-21.md"),
        os.path.join(LAB_ROOT, "review-0028-2026-07-20.md"),
        os.path.join(LAB_ROOT, "review-0032-2026-07-21.md"),
        "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json",
        GRAPH_PATH,
        GRAPH_PATH + ".manifest.json",
        BUILD_RECEIPT,
        R0029_V3,
        R0032_V4,
        R0032_V4_PAIRS,
        R0032_CANARY,
        CAP_PATH,
        "/data/latent-basemap/runs/round-0020/queue/artifacts/duplicate-census/global-duplicate-census-v1.npz",
        "/data/latent-basemap/runs/round-0020/queue/artifacts/duplicate-census/r0019-global-baseline.json",
        os.path.join(R0019_HIGH_D_REFERENCE, "reference.npz"),
        os.path.join(R0019_HIGH_D_REFERENCE, "reference-receipt.json"),
        os.path.join(R0019_HIGH_D_REFERENCE, "recall50-truth.npy"),
        "/data/latent-basemap/runs/round-0018/posthoc-untrained-floor.json",
        "/data/checkpoints/pumap/faiss_ivf_pq_30m.index",
        "/data/latent-basemap/track1/centroids_minilm_k256.npy",
        "/data/latent-basemap/track1/centroids_minilm_k1024.npy",
        "/data/latent-basemap/track1/minilm_queries.npy",
        "/data/latent-basemap/track1/minilm_queries_prov.json",
        R0023_LAYOUT,
        *R0023_PANELS.values(),
        R0028_PANEL,
        "/data/embeddings/beir/scifact-pooled-minilm/corpus_vectors.npy",
        "/data/embeddings/beir/scifact-pooled-minilm/query_vectors.npy",
        "/data/embeddings/beir/scifact-pooled-minilm/corpus_ids.json",
        "/data/embeddings/beir/scifact-pooled-minilm/query_ids.json",
        "/data/hf/datasets/mteb___scifact/corpus/0.0.0/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/scifact-corpus.arrow",
        "/data/embeddings/beir/trec-covid-pooled-minilm/corpus_vectors.npy",
        "/data/embeddings/beir/trec-covid-pooled-minilm/queries_vectors.npy",
        "/data/embeddings/beir/trec-covid-pooled-minilm/corpus_ids.json",
        "/data/embeddings/beir/trec-covid-pooled-minilm/queries_ids.json",
        "/data/embeddings/beir/trec-covid-pooled-minilm/topk_indices.npy",
        "/data/embeddings/beir/trec-covid-pooled-minilm/topk_meta.json",
        "/data/embeddings/dadabase/minilm.npy",
        "/data/embeddings/dadabase/jokes.parquet",
        *templates.values(),
    ]
    inputs = _dedupe([
        *_file_inputs(files),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])
    by_path = {item["canonical_path"]: item for item in inputs}
    mismatches = {
        path: {
            "expected": digest,
            "observed": by_path.get(os.path.realpath(path), {}).get("sha256"),
        }
        for path, digest in EXACT_EVIDENCE_SHA256.items()
        if by_path.get(os.path.realpath(path), {}).get("sha256") != digest
    }
    if mismatches:
        raise RuntimeError(f"R0030 reviewed evidence tuple changed: {mismatches}")
    return inputs


def _arm_paths(artifacts: str) -> dict[str, dict[str, str]]:
    return {
        arm: {
            "train": os.path.join(artifacts, arm, "train"),
            "coordinates": os.path.join(artifacts, arm, "coordinates"),
            "panel": os.path.join(artifacts, arm, "panel"),
            "semantic_renders": os.path.join(artifacts, arm, "semantic-renders"),
            "ood_canary": os.path.join(artifacts, arm, "ood-canary"),
            "ood_panel": os.path.join(artifacts, arm, "ood-panel"),
        }
        for arm in ("uniform", "fuzzy")
    }


def _jobs(
    *, artifacts: str, templates: dict[str, str], inputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    paths = _arm_paths(artifacts)
    canary = os.path.join(artifacts, "sampler-canary")

    def job(
        node_id: str,
        handler: str,
        deps: list[str],
        output: str,
        p90: float,
        *,
        gpu: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "id": node_id,
            "handler": handler,
            "deps": deps,
            "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
            "outputs": [output],
            "expected_inputs": inputs,
            "p90_wall_s": p90,
            "node_policy": {
                "gpu_required": gpu,
                "training_performed": handler == "train_seed42_30m",
            },
            **extra,
        }

    jobs: list[dict[str, Any]] = [
        job("sampler_canary", "round0030_sampler_canary", [], canary, 300.0,
            arm="uniform"),
        job("uniform_train_30m", "train_seed42_30m", ["sampler_canary"],
            paths["uniform"]["train"], 6000.0, arm="uniform", canary_output=canary),
        job("fuzzy_train_30m", "train_seed42_30m", ["uniform_train_30m"],
            paths["fuzzy"]["train"], 6000.0, arm="fuzzy", canary_output=canary),
    ]
    for arm in ("uniform", "fuzzy"):
        jobs.append(job(
            f"{arm}_transform_30m", "transform_30m", [f"{arm}_train_30m"],
            paths[arm]["coordinates"], 300.0, arm=arm,
            train_output=paths[arm]["train"],
            transform_spec_template=templates[arm],
        ))
    for arm in ("uniform", "fuzzy"):
        jobs.append(job(
            f"{arm}_registered_panel", "registered_panel",
            [f"{arm}_transform_30m"], paths[arm]["panel"], 2700.0,
            arm=arm, canary_output=canary, train_output=paths[arm]["train"],
            transform_output=paths[arm]["coordinates"],
            reference_output=R0019_HIGH_D_REFERENCE,
        ))
    for arm in ("uniform", "fuzzy"):
        jobs.append(job(
            f"{arm}_semantic_renders", "semantic_renders",
            [f"{arm}_registered_panel"], paths[arm]["semantic_renders"], 180.0,
            arm=arm, transform_output=paths[arm]["coordinates"],
            panel_output=paths[arm]["panel"],
        ))
    for arm in ("uniform", "fuzzy"):
        jobs.extend([
            job(
                f"{arm}_ood_canary", "round0030_ood_canary",
                [f"{arm}_transform_30m"], paths[arm]["ood_canary"], 180.0,
                arm=arm, train_output=paths[arm]["train"],
                transform_output=paths[arm]["coordinates"],
            ),
            job(
                f"{arm}_ood_panel", "round0030_ood_panel",
                [f"{arm}_ood_canary"], paths[arm]["ood_panel"], 300.0,
                arm=arm, train_output=paths[arm]["train"],
                transform_output=paths[arm]["coordinates"],
                ood_canary_output=paths[arm]["ood_canary"],
            ),
        ])
    comparison_inputs = {
        arm: {
            "train": paths[arm]["train"],
            "panel": paths[arm]["panel"],
            "ood_panel": paths[arm]["ood_panel"],
        }
        for arm in ("uniform", "fuzzy")
    }
    jobs.append(job(
        "comparison", "round0030_comparison",
        ["uniform_semantic_renders", "fuzzy_semantic_renders",
         "uniform_ood_panel", "fuzzy_ood_panel"],
        os.path.join(artifacts, "comparison"), 120.0, gpu=False,
        arm="uniform", arm_outputs=comparison_inputs,
    ))
    return jobs


def prepare_round0030(release_sha: str) -> str:
    round_root = ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0030 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    templates = _transform_templates(queue_root, release_sha)
    inputs = _static_inputs(templates)
    manifest = _base_manifest(
        round_id="0030",
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=6.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0013", "0020", "0023", "0028", "0032"]
    manifest["capability_dependencies"] = [
        "30m-input-pack-v1",
        "30m-duplicate-census-v1",
        "30m-layout-disparity-v1",
        "universality-panel-v1",
        "30m-weighted-fuzzy-graph-v2",
    ]
    manifest["capabilities_produced"] = ["30m-fuzzy-sampling-decision-v1"]
    manifest["training_performed"] = True
    manifest["scientific_contract"] = {
        "graph": expected_input_signature(GRAPH_PATH),
        "graph_sha256": GRAPH_SHA256,
        "manifest_sha256": GRAPH_MANIFEST_SHA256,
        "build_receipt_sha256": GRAPH_BUILD_RECEIPT_SHA256,
        "build_contract_sha256": GRAPH_BUILD_CONTRACT_SHA256,
        "resident_edges": GRAPH_RESIDENT_EDGES,
        "effective_retained_source_edges": GRAPH_EFFECTIVE_EDGES,
        "duplicate_cap_sha256": CAP_SHA256,
        "r0023_layout_sha256": R0023_LAYOUT_SHA256,
        "r0023_metric_seed_spreads": R0023_SEED_SPREAD,
        "ood_retention_noninferiority_margin": OOD_RETENTION_NONINFERIORITY_MARGIN,
        "arms": {
            arm: {
                "production_config_sha256": train_config_for_arm(arm)[1],
                "weighted_edge_sampling": arm == "fuzzy",
            }
            for arm in ("uniform", "fuzzy")
        },
        "successful_positive_lr_updates_per_arm": 500_000,
        "seed": 42,
    }
    manifest["jobs"] = _jobs(
        artifacts=artifacts, templates=templates, inputs=inputs)
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        {"queue_manifest": prepare_round0030(args.release_sha)},
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

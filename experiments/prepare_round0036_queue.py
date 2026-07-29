#!/usr/bin/env python3
"""Materialize R0036 only after a reviewed R0034 model is late-bound."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0034_program import INT8_PATH, SCALES_PATH
from basemap.round0036_pipeline import (
    INDEX_PATH,
    estimated_queue_gpu_seconds,
    low_dim_search_work_model,
    panel_config_identity,
    validate_reviewed_model_bundle,
)
from experiments.common_corpus_ood_round0035 import (
    CHUNK_SCRIPT,
    EMBED_SCRIPT,
    INPUT_PACK_MANIFEST,
    PROBES,
)
from experiments.prepare_round0020_0022_queues import (
    _dedupe,
    _hf_snapshot_file_inputs,
    _materialized_chunk_inputs,
)
from experiments.run_round0036_node import (
    CENTROIDS,
    MINILM_QUERIES,
    MINILM_QUERY_PROVENANCE,
)


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0036-2026-07-22.md")
ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0033/queue/artifacts/eligibility/"
    "minilm-150m-row-eligibility-v1.npz"
)
ELIGIBILITY_SHA256 = (
    "cd9738d1cb35b7847923ec24e343583ac91dea4d76381ec28c8c2c8bf6412aca"
)
R0034_IMPLEMENTATION_PREDECESSOR = (
    "7ac11bea86283d90f66100b92ddd5cd0a8490188"
)


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*([^\s]+)\s*$", text)
    return match.group(1) if match else None


def _assert_review_binds_artifacts(
    review_path: str,
    *,
    model_sha256: str,
    train_receipt_sha256: str,
) -> None:
    """Require the accepted R0034 review to name the exact released tuple."""
    with open(review_path, encoding="utf-8") as handle:
        text = handle.read()
    missing = [
        digest
        for digest in (model_sha256, train_receipt_sha256)
        if digest not in text
    ]
    if missing:
        raise RuntimeError(
            "accepted R0034 review does not bind the supplied model/train "
            f"receipt hashes: missing={missing}"
        )


def _assert_issuance(round_file: str, review_file: str) -> None:
    if _frontmatter_status(round_file) != "issued":
        raise RuntimeError("R0036 remains draft; refuse to materialize its queue")
    if _frontmatter_status(review_file) != "accepted":
        raise RuntimeError("R0034 review is not accepted; R0036 model is not released")


def _static_ood_paths() -> list[str]:
    values = [
        INPUT_PACK_MANIFEST,
        MINILM_QUERIES,
        MINILM_QUERY_PROVENANCE,
        EMBED_SCRIPT,
        CHUNK_SCRIPT,
        "/data/embeddings/beir/trec-covid-pooled-minilm/corpus_vectors.npy",
        "/data/embeddings/beir/trec-covid-pooled-minilm/queries_vectors.npy",
        "/data/embeddings/beir/trec-covid-pooled-minilm/corpus_ids.json",
        "/data/embeddings/beir/trec-covid-pooled-minilm/queries_ids.json",
        "/data/embeddings/beir/trec-covid-pooled-minilm/topk_indices.npy",
        "/data/embeddings/beir/trec-covid-pooled-minilm/topk_meta.json",
        "/data/embeddings/dadabase/minilm.npy",
        "/data/embeddings/dadabase/jokes.parquet",
        "/data/latent-basemap/runs/round-0022/queue/artifacts/panel/"
        "universality-panel-v1.json",
    ]
    for probe in PROBES.values():
        values.extend([probe.vectors, probe.ids, probe.manifest, probe.chunks])
    return values


def _job(
    *,
    job_id: str,
    deps: list[str],
    output: str,
    p90_wall_s: float,
    inputs: list[dict[str, Any]],
    common: dict[str, Any],
    gpu: bool,
    **extra: Any,
) -> dict[str, Any]:
    return {
        **common,
        **extra,
        "id": job_id,
        "handler_module": "experiments.run_round0036_node",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(os.path.dirname(output), f"{job_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": float(p90_wall_s),
        "node_policy": {"gpu_required": gpu, "training_performed": False},
    }


def prepare(
    *,
    release_sha: str,
    model_path: str,
    model_sha256: str,
    train_receipt_path: str,
    train_receipt_sha256: str,
    r0034_review_path: str,
    r0034_review_sha256: str,
    queue_root: str = "/data/latent-basemap/runs/round-0036/queue",
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0036 release SHA must be one full commit")
    _assert_issuance(ROUND_FILE, r0034_review_path)
    review_signature = expected_input_signature(r0034_review_path)
    if review_signature["sha256"] != r0034_review_sha256:
        raise RuntimeError("accepted R0034 review bytes changed")
    bundle = validate_reviewed_model_bundle(
        model_path=model_path,
        model_sha256=model_sha256,
        train_receipt_path=train_receipt_path,
        train_receipt_sha256=train_receipt_sha256,
    )
    _assert_review_binds_artifacts(
        r0034_review_path,
        model_sha256=bundle["model"]["sha256"],
        train_receipt_sha256=bundle["train_receipt"]["sha256"],
    )
    queue_root = create_fresh_directory(queue_root, label="R0036 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "canary": os.path.join(artifacts, "canary"),
        "coordinates": os.path.join(artifacts, "coordinates"),
        "reference": os.path.join(artifacts, "high-d-reference"),
        "panel": os.path.join(artifacts, "panel"),
        "ood": os.path.join(artifacts, "ood"),
        "render": os.path.join(artifacts, "semantic-renders"),
        "registry": os.path.join(artifacts, "registry"),
    }
    core = _dedupe([
        expected_input_signature(ROUND_FILE),
        review_signature,
        bundle["model"],
        bundle["train_receipt"],
        expected_input_signature(ELIGIBILITY_PATH),
    ])
    encoded = _dedupe([
        *core,
        expected_input_signature(INT8_PATH),
        expected_input_signature(SCALES_PATH),
    ])
    centroids = _dedupe([
        *encoded,
        *(expected_input_signature(path) for path in CENTROIDS.values()),
    ])
    queries = _dedupe([
        *centroids,
        expected_input_signature(MINILM_QUERIES),
        expected_input_signature(MINILM_QUERY_PROVENANCE),
    ])
    ood = _dedupe([
        *core,
        *(expected_input_signature(path) for path in _static_ood_paths()),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])
    common = {
        "model_path": bundle["model"]["canonical_path"],
        "model_sha256": bundle["model"]["sha256"],
        "train_receipt_path": bundle["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle["train_receipt"]["sha256"],
        "eligibility_path": ELIGIBILITY_PATH,
        "eligibility_sha256": ELIGIBILITY_SHA256,
    }
    estimates = estimated_queue_gpu_seconds()
    jobs = [
        _job(
            job_id="production_canary", deps=[], output=paths["canary"],
            p90_wall_s=estimates["production_canary"], inputs=_dedupe([
                *encoded, expected_input_signature(INDEX_PATH)
            ]), common=common, gpu=True,
        ),
        _job(
            job_id="transform_150m", deps=["production_canary"],
            output=paths["coordinates"], p90_wall_s=estimates["transform_150m"],
            inputs=queries, common=common, gpu=True,
            coordinate_chunk_rows=5_000_000, model_batch_rows=65_536,
        ),
        _job(
            job_id="high_d_reference", deps=["production_canary"],
            output=paths["reference"], p90_wall_s=estimates["high_d_reference"],
            inputs=centroids, common=common, gpu=True,
        ),
        _job(
            job_id="registered_panel",
            deps=["transform_150m", "high_d_reference"], output=paths["panel"],
            p90_wall_s=estimates["registered_panel"], inputs=queries,
            common=common, gpu=True, transform_output=paths["coordinates"],
            reference_output=paths["reference"],
        ),
        _job(
            job_id="ood_panels", deps=["transform_150m"], output=paths["ood"],
            p90_wall_s=estimates["ood_canaries_and_panels"], inputs=ood,
            common=common, gpu=True, transform_output=paths["coordinates"],
        ),
        _job(
            job_id="semantic_render", deps=["registered_panel"],
            output=paths["render"], p90_wall_s=300.0, inputs=core,
            common=common, gpu=False, transform_output=paths["coordinates"],
            panel_output=paths["panel"],
        ),
        _job(
            job_id="registry_publication", deps=["semantic_render", "ood_panels"],
            output=paths["registry"], p90_wall_s=300.0, inputs=core,
            common=common, gpu=False,
        ),
    ]
    manifest = {
        "schema_version": 1,
        "schema": "round0036-evaluation-queue-v1",
        "program": "basemap-100m-round-0036",
        "round_id": "0036",
        "round_sha256": sha256_file(ROUND_FILE),
        "release_sha": release_sha,
        "implementation_predecessor": {
            "round_id": "0034",
            "release_sha": R0034_IMPLEMENTATION_PREDECESSOR,
            "relationship": "git-merged-reviewed-training-implementation",
        },
        "execution_authority": "autonomous-gpu",
        "gpu_hours_cap": 8.0,
        "queue_class": "gpu-research",
        "training_performed": False,
        "deadline_utc": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=12)
        ).isoformat(timespec="seconds"),
        "repo_root": RUN_ROOT,
        "lease_path": "/data/latent-basemap/.gpu_lease",
        "required_reviews": ["0028", "0033", "0034", "0035"],
        "capability_dependencies": [
            "150m-minilm-trained-model-coverage-aligned-seed42",
            "minilm-150m-row-eligibility-v1",
            "universality-panel-v1",
            "common-corpus-ood-panel-v1",
        ],
        "capabilities_produced": ["150m-minilm-map-coverage-aligned-seed42"],
        "late_bound_r0034": {
            "model": bundle["model"],
            "train_receipt": bundle["train_receipt"],
            "review": review_signature,
            "production_config_sha256": bundle["production_config_sha256"],
        },
        "scientific_contract": {
            "eligibility": expected_input_signature(ELIGIBILITY_PATH),
            "panel": panel_config_identity(),
            "low_dim_search_work_model": low_dim_search_work_model(),
            "all_row_transform_rows": 150_000_000,
            "scientific_representative_rows": 147_221_757,
            "excluded_duplicate_copies": 2_542_774,
            "zero_invalid_rows": 235_469,
            "ood_probes": ["dadabase", "trec-covid", "code", "science", "latin"],
        },
        "p90_gpu_seconds": estimates,
        "child_environment": {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/data/hf",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SENTENCE_TRANSFORMERS_HOME": "/data/hf/sentence-transformers",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPYCACHEPREFIX": os.path.join(queue_root, "cache", "pycache"),
            "NUMBA_CACHE_DIR": os.path.join(queue_root, "cache", "numba"),
            "TRITON_CACHE_DIR": os.path.join(queue_root, "cache", "triton"),
        },
        "jobs": jobs,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--train-receipt", required=True)
    parser.add_argument("--train-receipt-sha256", required=True)
    parser.add_argument("--r0034-review", required=True)
    parser.add_argument("--r0034-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default="/data/latent-basemap/runs/round-0036/queue"
    )
    args = parser.parse_args(argv)
    print(prepare(
        release_sha=args.release_sha,
        model_path=args.model,
        model_sha256=args.model_sha256,
        train_receipt_path=args.train_receipt,
        train_receipt_sha256=args.train_receipt_sha256,
        r0034_review_path=args.r0034_review,
        r0034_review_sha256=args.r0034_review_sha256,
        queue_root=args.queue_root,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

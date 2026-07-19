#!/usr/bin/env python3
"""Round 0022 MiniLM universality panel harness."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_file,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)


ROUND_ID = "0022"
SEED = 20260722
MODEL_PATH = "/data/latent-basemap/runs/round-0019/queue/artifacts/train/model.pt"
MODEL_SHA256 = "2f5eb27582e26735491b4bed9417cf27992bb213ef942e433a5bcba97d481a32"
R0019_COORDINATES = "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates"
R0019_COORDINATE_RECEIPT_SHA256 = (
    "8d6d5ab2b16be0e08b636a248f667d6a963217d1ec3a223af0b0730875d491d9"
)
MINILM_QUERIES = "/data/latent-basemap/track1/minilm_queries.npy"
SENTENCE_MODEL_SNAPSHOT = (
    "/data/hf/sentence-transformers/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
SCIFACT_ARROW = (
    "/data/hf/datasets/mteb___scifact/corpus/0.0.0/"
    "cf10ab6856b15b0e670ef8ae5dae4e266c12d035/scifact-corpus.arrow"
)


@dataclass(frozen=True)
class Probe:
    name: str
    corpus_vectors: np.ndarray
    query_vectors: np.ndarray
    corpus_ids: np.ndarray
    query_ids: np.ndarray
    inputs: dict[str, Any]


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _string_ids_sha256(ids: np.ndarray) -> str:
    return sha256_bytes(canonical_json([str(value) for value in np.asarray(ids).tolist()]))


def _hf_snapshot_signature(root: str) -> dict[str, Any]:
    """Bind a Hugging Face snapshot that intentionally contains symlinks."""
    root = os.path.realpath(root)
    members: list[dict[str, Any]] = []
    total = 0
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            link_target = os.readlink(path) if os.path.islink(path) else None
            resolved = os.path.realpath(path)
            if not os.path.isfile(resolved):
                raise RuntimeError(f"HF snapshot member does not resolve to a file: {path}")
            size = os.path.getsize(resolved)
            total += size
            members.append(
                {
                    "relative_path": rel,
                    "link_target": link_target,
                    "resolved_path": resolved,
                    "bytes": int(size),
                    "sha256": sha256_file(resolved),
                }
            )
    payload = [
        {k: v for k, v in item.items() if k in {"relative_path", "link_target", "bytes", "sha256"}}
        for item in members
    ]
    return {
        "canonical_path": root,
        "kind": "hf-symlink-snapshot",
        "bytes": int(total),
        "members": members,
        "sha256": sha256_bytes(canonical_json(payload)),
    }


def _cosine_topk(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    device: str = "cuda",
    return_scores: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        c = torch.as_tensor(np.array(corpus, dtype=np.float32, copy=True), device=device)
        q = torch.as_tensor(np.array(queries, dtype=np.float32, copy=True), device=device)
        c = c / torch.clamp(torch.linalg.norm(c, dim=1, keepdim=True), min=1e-30)
        q = q / torch.clamp(torch.linalg.norm(q, dim=1, keepdim=True), min=1e-30)
        scores = q @ c.T
        values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        out_i = indices.cpu().numpy().astype(np.int64)
        out_v = values.cpu().numpy().astype(np.float32)
        del c, q, scores, values, indices
        if device == "cuda":
            torch.cuda.synchronize()
    return (out_i, out_v) if return_scores else out_i


def _l2_topk(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    device: str = "cuda",
) -> np.ndarray:
    import torch

    with torch.no_grad():
        c = torch.as_tensor(np.array(corpus, dtype=np.float32, copy=True), device=device)
        q = torch.as_tensor(np.array(queries, dtype=np.float32, copy=True), device=device)
        c2 = torch.sum(c * c, dim=1)[None, :]
        q2 = torch.sum(q * q, dim=1)[:, None]
        scores = -(q2 + c2 - 2.0 * (q @ c.T))
        _, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        out = indices.cpu().numpy().astype(np.int64)
        del c, q, c2, q2, scores, indices
        if device == "cuda":
            torch.cuda.synchronize()
    return out


def _ffr_from_neighbors(true_top10: np.ndarray, hit_neighbors: np.ndarray) -> tuple[float, np.ndarray]:
    per_query = np.empty(true_top10.shape[0], dtype=np.float32)
    for index in range(true_top10.shape[0]):
        per_query[index] = np.isin(true_top10[index], hit_neighbors[index]).sum() / 10.0
    return float(np.mean(per_query, dtype=np.float64)), per_query


def _project(model: Any, vectors: np.ndarray, *, batch_size: int = 65536) -> np.ndarray:
    import torch

    model.model.eval()
    output = np.empty((len(vectors), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = np.asarray(vectors[start : start + batch_size], dtype=np.float32)
            tensor = torch.from_numpy(batch).to(model.device)
            coords = model.model(tensor).detach().cpu().numpy().astype(np.float32)
            if coords.shape != (len(batch), 2) or not np.isfinite(coords).all():
                raise RuntimeError("non-finite or malformed projected coordinates")
            output[start : start + len(batch)] = coords
            del tensor
        if str(model.device).startswith("cuda"):
            torch.cuda.synchronize()
    return output


def _load_model() -> Any:
    signature = expected_input_signature(MODEL_PATH)
    if signature["sha256"] != MODEL_SHA256:
        raise RuntimeError("R0019 model bytes changed")
    from basemap.pumap.parametric_umap import ParametricUMAP

    return ParametricUMAP.load(MODEL_PATH, device="cuda")


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_beir_probe(root: str, *, name: str) -> Probe:
    corpus_name = "corpus_vectors.npy"
    query_name = "query_vectors.npy" if os.path.exists(os.path.join(root, "query_vectors.npy")) else "queries_vectors.npy"
    qids_name = "query_ids.json" if os.path.exists(os.path.join(root, "query_ids.json")) else "queries_ids.json"
    corpus_vectors = np.load(os.path.join(root, corpus_name), mmap_mode="r", allow_pickle=False)
    query_vectors = np.load(os.path.join(root, query_name), mmap_mode="r", allow_pickle=False)
    corpus_ids = np.asarray(_read_json(os.path.join(root, "corpus_ids.json")), dtype=object)
    query_ids = np.asarray(_read_json(os.path.join(root, qids_name)), dtype=object)
    return Probe(
        name=name,
        corpus_vectors=corpus_vectors,
        query_vectors=query_vectors,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        inputs={
            "corpus_vectors": expected_input_signature(os.path.join(root, corpus_name)),
            "query_vectors": expected_input_signature(os.path.join(root, query_name)),
            "corpus_ids": expected_input_signature(os.path.join(root, "corpus_ids.json")),
            "query_ids": expected_input_signature(os.path.join(root, qids_name)),
        },
    )


def _load_dadabase_probe() -> Probe:
    vectors = np.load("/data/embeddings/dadabase/minilm.npy", mmap_mode="r", allow_pickle=False)
    rng = np.random.RandomState(SEED)
    query_rows = np.sort(rng.choice(len(vectors), 500, replace=False)).astype(np.int64)
    mask = np.ones(len(vectors), dtype=bool)
    mask[query_rows] = False
    corpus_rows = np.flatnonzero(mask).astype(np.int64)
    return Probe(
        name="dadabase",
        corpus_vectors=np.asarray(vectors[corpus_rows], dtype=np.float16),
        query_vectors=np.asarray(vectors[query_rows], dtype=np.float16),
        corpus_ids=corpus_rows,
        query_ids=query_rows,
        inputs={
            "vectors": expected_input_signature("/data/embeddings/dadabase/minilm.npy"),
            "texts": expected_input_signature("/data/embeddings/dadabase/jokes.parquet"),
            "query_rows_sha256": ordered_array_sha256(query_rows),
            "corpus_rows_sha256": ordered_array_sha256(corpus_rows),
        },
    )


def load_probe(name: str) -> Probe:
    if name == "scifact":
        return _load_beir_probe("/data/embeddings/beir/scifact-pooled-minilm", name=name)
    if name == "trec-covid":
        return _load_beir_probe("/data/embeddings/beir/trec-covid-pooled-minilm", name=name)
    if name == "dadabase":
        return _load_dadabase_probe()
    raise KeyError(name)


def _sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SENTENCE_MODEL_SNAPSHOT)


def _cosine_mean(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    cos = np.einsum("ij,ij->i", a, b) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )
    return {
        "mean_cosine": float(np.mean(cos, dtype=np.float64)),
        "min_cosine": float(np.min(cos)),
        "max_cosine": float(np.max(cos)),
        "sample_count": int(len(cos)),
        "sample_cosines_sha256": ordered_array_sha256(cos.astype(np.float32)),
    }


def _dadabase_canary() -> dict[str, Any]:
    import pandas as pd

    vectors = np.load("/data/embeddings/dadabase/minilm.npy", mmap_mode="r", allow_pickle=False)
    jokes = pd.read_parquet("/data/embeddings/dadabase/jokes.parquet")["joke"]
    rng = np.random.RandomState(SEED)
    rows = np.sort(rng.choice(len(vectors), 40, replace=False)).astype(np.int64)
    model = _sentence_model()
    embedded = model.encode(
        jokes.iloc[rows].tolist(),
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    stats = _cosine_mean(embedded, np.asarray(vectors[rows], dtype=np.float32))
    return {
        "status": "included" if stats["mean_cosine"] >= 0.98 else "excluded",
        "rule": "mean cosine of 40 re-embedded seed-fixed jokes >= 0.98",
        "sample_rows_sha256": ordered_array_sha256(rows),
        "stats": stats,
    }


def _load_scifact_texts(corpus_ids: np.ndarray, rows: np.ndarray) -> list[str]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(SCIFACT_ARROW, "r") as source:
        table = ipc.open_stream(source).read_all()
    by_id = {item["_id"]: item for item in table.to_pylist()}
    texts: list[str] = []
    for row in rows.tolist():
        item = by_id[str(corpus_ids[row])]
        texts.append((str(item["title"]) + " " + str(item["text"])).strip())
    return texts


def _scifact_canary() -> dict[str, Any]:
    probe = load_probe("scifact")
    rng = np.random.RandomState(SEED)
    rows = np.sort(rng.choice(len(probe.corpus_vectors), 40, replace=False)).astype(np.int64)
    try:
        texts = _load_scifact_texts(probe.corpus_ids, rows)
        model = _sentence_model()
        embedded = model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        stats = _cosine_mean(embedded, np.asarray(probe.corpus_vectors[rows], dtype=np.float32))
        status = "included" if stats["mean_cosine"] >= 0.98 else "excluded"
        reason = None if status == "included" else "mean cosine below registered 0.98 threshold"
    except Exception as exc:  # fail closed by excluding and naming the reason
        stats = None
        status = "excluded"
        reason = f"text canary unavailable or failed: {exc!r}"
    return {
        "status": status,
        "reason": reason,
        "rule": "mean cosine of 40 re-embedded public BEIR scifact corpus texts >= 0.98",
        "sample_rows_sha256": ordered_array_sha256(rows),
        "stats": stats,
        "inputs": {
            **probe.inputs,
            "scifact_arrow": expected_input_signature(SCIFACT_ARROW),
        },
    }


def _trec_canary() -> dict[str, Any]:
    root = "/data/embeddings/beir/trec-covid-pooled-minilm"
    probe = load_probe("trec-covid")
    topk = np.load(os.path.join(root, "topk_indices.npy"), allow_pickle=False)
    meta = _read_json(os.path.join(root, "topk_meta.json"))
    if meta.get("model") != "sentence-transformers/all-MiniLM-L6-v2":
        return {
            "status": "excluded",
            "reason": f"topk_meta model changed: {meta.get('model')!r}",
        }
    observed, scores = _cosine_topk(
        probe.corpus_vectors, probe.query_vectors, 100, return_scores=True
    )
    set_equal = all(set(observed[i].tolist()) == set(topk[i].tolist()) for i in range(len(topk)))
    ordered_equal = bool(np.array_equal(observed, topk))
    tie_swaps: list[dict[str, Any]] = []
    if set_equal and not ordered_equal:
        # Stored and recomputed order can differ only where cosine scores are exactly tied.
        corpus = np.asarray(probe.corpus_vectors, dtype=np.float32)
        queries = np.asarray(probe.query_vectors, dtype=np.float32)
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        full_scores = queries @ corpus.T
        for q, pos in np.argwhere(observed != topk):
            observed_score = float(full_scores[q, observed[q, pos]])
            stored_score = float(full_scores[q, topk[q, pos]])
            if not np.isclose(observed_score, stored_score, rtol=0.0, atol=1e-7):
                set_equal = False
                break
            tie_swaps.append(
                {
                    "query_index": int(q),
                    "position": int(pos),
                    "observed_index": int(observed[q, pos]),
                    "stored_index": int(topk[q, pos]),
                    "cosine": stored_score,
                }
            )
    status = "included" if set_equal else "excluded"
    return {
        "status": status,
        "reason": None if set_equal else "stored topk_indices not reproduced from vectors",
        "rule": "stored top-100 cosine neighbor sets reproduced from stored vectors; tied tail order named",
        "ordered_exact": ordered_equal,
        "set_exact": set_equal,
        "tie_swaps": tie_swaps,
        "observed_topk_sha256": ordered_array_sha256(observed),
        "observed_topk_scores_sha256": ordered_array_sha256(scores),
        "inputs": {
            **probe.inputs,
            "topk_indices": expected_input_signature(os.path.join(root, "topk_indices.npy")),
            "topk_meta": expected_input_signature(os.path.join(root, "topk_meta.json")),
        },
    }


def run_canary(*, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0022 canary output")
    started = time.monotonic()
    model = _load_model()
    smoke = np.load(
        "/data/latent-basemap/runs/round-0010/materialized/chunk-00000/embeddings.npy",
        mmap_mode="r",
        allow_pickle=False,
    )[:1000]
    smoke_coords = _project(model, smoke, batch_size=1000)
    if smoke_coords.shape != (1000, 2) or not np.isfinite(smoke_coords).all():
        raise RuntimeError("R0022 model smoke projection failed")
    canaries = {
        "dadabase": _dadabase_canary(),
        "trec-covid": _trec_canary(),
        "scifact": _scifact_canary(),
    }
    included = sorted(name for name, item in canaries.items() if item.get("status") == "included")
    body = {
        "schema": "round0022-universality-canary-v1",
        "round_id": ROUND_ID,
        "model": expected_input_signature(MODEL_PATH),
        "model_sha256_expected": MODEL_SHA256,
        "smoke_projection": {
            "rows": 1000,
            "finite": True,
            "coords_sha256": ordered_array_sha256(smoke_coords),
        },
        "probe_canaries": canaries,
        "included_probes": included,
        "included_probe_count": len(included),
        "acceptance_minimum_included_probes": 2,
        "can_build_panel": len(included) >= 2,
        "sentence_model_snapshot": _hf_snapshot_signature(SENTENCE_MODEL_SNAPSHOT),
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "verdict.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "verdict": expected_input_signature(path)}


def _gather_control_rows(rows: np.ndarray) -> np.ndarray:
    result = np.empty((len(rows), 384), dtype=np.float16)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    for chunk_index in np.unique(sorted_rows // 1_000_000).tolist():
        chunk_index = int(chunk_index)
        lo = chunk_index * 1_000_000
        hi = lo + 1_000_000
        positions = np.flatnonzero((sorted_rows >= lo) & (sorted_rows < hi))
        array = np.load(
            f"/data/latent-basemap/runs/round-0010/materialized/chunk-{chunk_index:05d}/embeddings.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        result[order[positions]] = array[sorted_rows[positions] - lo]
        del array
    return result


def _matched_control(n_corpus: int, n_query: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(SEED)
    corpus_rows = np.sort(rng.choice(30_000_000, n_corpus, replace=False)).astype(np.int64)
    query_pool = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)
    query_rows = np.sort(rng.choice(len(query_pool), n_query, replace=False)).astype(np.int64)
    corpus = _gather_control_rows(corpus_rows)
    queries = np.asarray(query_pool[query_rows], dtype=np.float16)
    return corpus, queries, corpus_rows, query_rows


def _score_one(
    *,
    name: str,
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    corpus_coords: np.ndarray,
    query_coords: np.ndarray,
) -> dict[str, Any]:
    hit_k = max(50, int(math.ceil(0.01 * len(corpus_vectors))))
    true_top10 = _cosine_topk(corpus_vectors, query_vectors, 10)
    hit_neighbors = _l2_topk(corpus_coords, query_coords, hit_k)
    ffr, per_query = _ffr_from_neighbors(true_top10, hit_neighbors)
    return {
        "name": name,
        "corpus_rows": int(len(corpus_vectors)),
        "query_rows": int(len(query_vectors)),
        "true_k": 10,
        "hit_k": int(hit_k),
        "ffr": ffr,
        "per_query_ffr_sha256": ordered_array_sha256(per_query),
        "true_top10_sha256": ordered_array_sha256(true_top10),
        "hit_neighbors_sha256": ordered_array_sha256(hit_neighbors),
        "map_dispersion": {
            "axis_mean": np.mean(corpus_coords, axis=0).astype(float).tolist(),
            "axis_std": np.std(corpus_coords, axis=0).astype(float).tolist(),
            "axis_span": (np.max(corpus_coords, axis=0) - np.min(corpus_coords, axis=0)).astype(float).tolist(),
        },
    }


def run_panel(*, canary_path: str, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0022 panel output")
    started = time.monotonic()
    canary = _read_json(canary_path)
    canary_body = {key: canary[key] for key in canary if key != "identity_sha256"}
    if canary.get("identity_sha256") != sha256_bytes(canonical_json(canary_body)):
        raise RuntimeError("R0022 canary verdict seal is invalid")
    model = _load_model()
    probes: dict[str, Any] = {}
    for name in ("scifact", "trec-covid", "dadabase"):
        canary_item = canary["probe_canaries"][name]
        if canary_item.get("status") != "included":
            probes[name] = {
                "status": "excluded",
                "reason": canary_item.get("reason") or "substrate canary excluded probe",
                "canary": canary_item,
            }
            continue
        probe = load_probe(name)
        corpus = np.asarray(probe.corpus_vectors, dtype=np.float16)
        queries = np.asarray(probe.query_vectors, dtype=np.float16)
        corpus_coords = _project(model, corpus)
        query_coords = _project(model, queries)
        control_corpus, control_queries, control_rows, control_query_rows = _matched_control(
            len(corpus), len(queries)
        )
        control_corpus_coords = _project(model, control_corpus)
        control_query_coords = _project(model, control_queries)
        probe_score = _score_one(
            name=name,
            corpus_vectors=corpus,
            query_vectors=queries,
            corpus_coords=corpus_coords,
            query_coords=query_coords,
        )
        control_score = _score_one(
            name=f"{name}-matched-control",
            corpus_vectors=control_corpus,
            query_vectors=control_queries,
            corpus_coords=control_corpus_coords,
            query_coords=control_query_coords,
        )
        retention = (
            float(probe_score["ffr"] / control_score["ffr"])
            if control_score["ffr"] > 0
            else None
        )
        coords_path = os.path.join(output_root, f"{name}-coordinates.npz")
        atomic_save_new_npz(
            coords_path,
            immutable=True,
            probe_corpus_coords=corpus_coords,
            probe_query_coords=query_coords,
            control_corpus_coords=control_corpus_coords,
            control_query_coords=control_query_coords,
            probe_corpus_ids=np.asarray([str(value) for value in np.asarray(probe.corpus_ids).tolist()]),
            probe_query_ids=np.asarray([str(value) for value in np.asarray(probe.query_ids).tolist()]),
            control_corpus_rows=control_rows,
            control_query_rows=control_query_rows,
        )
        probes[name] = {
            "status": "included",
            "canary": canary_item,
            "probe": probe_score,
            "matched_control": control_score,
            "retention": retention,
            "verdict": (
                "undefined-control-zero"
                if retention is None
                else "pass"
                if retention >= 0.7
                else "failure"
                if retention < 0.5
                else "amber"
            ),
            "coordinates": expected_input_signature(coords_path),
            "sample_hashes": {
                "probe_corpus_ids": _string_ids_sha256(np.asarray(probe.corpus_ids)),
                "probe_query_ids": _string_ids_sha256(np.asarray(probe.query_ids)),
                "control_corpus_rows": ordered_array_sha256(control_rows),
                "control_query_rows": ordered_array_sha256(control_query_rows),
            },
            "inputs": probe.inputs,
        }
    included = {name: item for name, item in probes.items() if item["status"] == "included"}
    body = {
        "schema": "universality-panel-v1",
        "round_id": ROUND_ID,
        "r0019_model": expected_input_signature(MODEL_PATH),
        "r0019_coordinate_receipt": expected_input_signature(
            os.path.join(R0019_COORDINATES, "actual-transform.json")
        ),
        "canary": expected_input_signature(canary_path),
        "seed": SEED,
        "scoring_formula": {
            "true_set": "top-10 exact cosine neighbors within the probe/control corpus",
            "hit_set": "top max(50, ceil(0.01*N_corpus)) map-space neighbors",
            "ffr": "mean fraction of true_set inside hit_set",
            "retention": "ffr(probe) / ffr(shape-matched in-domain control)",
        },
        "probes": probes,
        "included_probe_count": len(included),
        "acceptance": {
            "substrate_canaries_pass_for_at_least_two_probes": len(included) >= 2,
            "included_probe_verdicts": {
                name: item["verdict"] for name, item in included.items()
            },
            "all_included_retention_at_least_0_7": all(
                item["retention"] is not None and item["retention"] >= 0.7
                for item in included.values()
            ),
            "named_universality_failures": [
                name
                for name, item in included.items()
                if item["retention"] is not None and item["retention"] < 0.5
            ],
            "amber_probes": [
                name
                for name, item in included.items()
                if item["retention"] is not None and 0.5 <= item["retention"] < 0.7
            ],
        },
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "universality-panel-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "panel": expected_input_signature(path)}


def run_renders(*, panel_path: str, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="Round 0022 render output")
    started = time.monotonic()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments import run_round0014_node as node

    node.configure_round0019()
    coordinates = node.StreamedCoordinateArray(R0019_COORDINATES)
    sample_ids = np.load(
        "/data/latent-basemap/runs/round-0019/queue/artifacts/semantic-renders/sample-semantic-ids.npy",
        allow_pickle=False,
    )
    base = coordinates[sample_ids]
    panel = _read_json(panel_path)
    renders: dict[str, Any] = {}
    for name, item in panel["probes"].items():
        if item.get("status") != "included":
            continue
        coords_file = item["coordinates"]["canonical_path"]
        with np.load(coords_file, allow_pickle=False) as archive:
            corpus_coords = np.asarray(archive["probe_corpus_coords"], dtype=np.float32)
            query_coords = np.asarray(archive["probe_query_coords"], dtype=np.float32)
        rng = np.random.RandomState(SEED)
        take = np.sort(
            rng.choice(len(corpus_coords), min(5000, len(corpus_coords)), replace=False)
        )
        image_path = os.path.join(output_root, f"{name}-overlay.png")
        figure, axis = plt.subplots(figsize=(7, 5), dpi=140)
        axis.scatter(base[:, 0], base[:, 1], s=1, c="#d0d0d0", alpha=0.35, linewidths=0)
        axis.scatter(
            corpus_coords[take, 0],
            corpus_coords[take, 1],
            s=3,
            c="#2364aa",
            alpha=0.55,
            linewidths=0,
            label=f"{name} corpus sample",
        )
        axis.scatter(
            query_coords[:, 0],
            query_coords[:, 1],
            s=10,
            c="#d62828",
            alpha=0.8,
            linewidths=0,
            label=f"{name} queries",
        )
        axis.set_title(f"R0022 {name} over R0019 50k semantic sample")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(loc="best", markerscale=2)
        figure.tight_layout()
        figure.savefig(image_path, format="png")
        plt.close(figure)
        renders[name] = {
            "image": expected_input_signature(image_path),
            "drawn_probe_corpus_rows": int(len(take)),
            "drawn_probe_corpus_rows_sha256": ordered_array_sha256(take.astype(np.int64)),
            "query_rows": int(len(query_coords)),
            "base_sample_rows": int(len(base)),
        }
    body = {
        "schema": "round0022-universality-renders-v1",
        "round_id": ROUND_ID,
        "panel": expected_input_signature(panel_path),
        "r0019_coordinate_receipt": expected_input_signature(
            os.path.join(R0019_COORDINATES, "actual-transform.json")
        ),
        "r0019_semantic_sample_ids": expected_input_signature(
            "/data/latent-basemap/runs/round-0019/queue/artifacts/semantic-renders/sample-semantic-ids.npy"
        ),
        "renders": renders,
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "render-manifest.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "manifest": expected_input_signature(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    canary = sub.add_parser("canary")
    canary.add_argument("--out", required=True)
    panel = sub.add_parser("panel")
    panel.add_argument("--canary", required=True)
    panel.add_argument("--out", required=True)
    renders = sub.add_parser("renders")
    renders.add_argument("--panel", required=True)
    renders.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "canary":
        result = run_canary(output_root=os.path.realpath(args.out))
    elif args.command == "panel":
        result = run_panel(canary_path=os.path.realpath(args.canary), output_root=os.path.realpath(args.out))
    else:
        result = run_renders(panel_path=os.path.realpath(args.panel), output_root=os.path.realpath(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score three byte-bound Common Corpus OOD probes on the accepted R0019 map.

The two public ``run_*_job`` functions are standalone slim-runner entry points.
The canary proves that each stored embedding row still maps to its source chunk
text and that the registered MiniLM snapshot reproduces a deterministic sample.
The panel then evaluates a disjoint 49,500-row corpus / 500-row query split and
an identically shaped in-domain control using the corrected R0028 fp32 rules.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
    sha256_file,
)
from basemap.output_safety import (
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)


ROUND_ID = "0035"
MAP_LABEL = "r0019"
BASE_SEED = 20260735
PROBE_ROWS = 50_000
QUERY_ROWS = 500
CORPUS_ROWS = PROBE_ROWS - QUERY_ROWS
TRUE_K = 10
HIT_FRACTION = 0.01
CANARY_ROWS = 40
CANARY_MIN_MEAN_COSINE = 0.98

MODEL_PATH = "/data/latent-basemap/runs/round-0019/queue/artifacts/train/model.pt"
COORDINATE_RECEIPT = (
    "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates/"
    "actual-transform.json"
)
INPUT_PACK_MANIFEST = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
MINILM_QUERIES = "/data/latent-basemap/track1/minilm_queries.npy"
R0019_REVIEW = (
    "/home/enjalot/code/latent-labs/basemap-100m/"
    "review-0019-2026-07-19.md"
)
R0028_REVIEW = (
    "/home/enjalot/code/latent-labs/basemap-100m/"
    "review-0028-2026-07-20.md"
)
SENTENCE_MODEL_SNAPSHOT = (
    "/data/hf/sentence-transformers/"
    "models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
SENTENCE_MODEL_SNAPSHOT_SHA256 = (
    "8af1d9f7875a6c653a34813a8b97389c0306f397fd42f3f1c540b82f797bbc1a"
)
EMBED_SCRIPT = "/data/raw/common-corpus/embed_probes.py"
CHUNK_SCRIPT = "/data/raw/common-corpus/chunk_commoncorpus.py"

EXACT_SHA256 = {
    MODEL_PATH: "2f5eb27582e26735491b4bed9417cf27992bb213ef942e433a5bcba97d481a32",
    COORDINATE_RECEIPT: "8d6d5ab2b16be0e08b636a248f667d6a963217d1ec3a223af0b0730875d491d9",
    INPUT_PACK_MANIFEST: "bf6537b83559c1eb5b52b8e73c4a47315191dfcf1b1128c141d32308b3b167f8",
    MINILM_QUERIES: "74e459785c0496904b385a1d11eb229b6164580d3877b9e00f1eed5679dee1b4",
    R0019_REVIEW: "803288e4f686f57d8da0add13711be10c7e721bc21152cab8517397efc1d9593",
    R0028_REVIEW: "7034af8793cf616fe277cf38943826e5daa489a421bab6b5e63362562bf7611c",
    EMBED_SCRIPT: "f1a6437c055ea8f4717463117678d2077a6f62c1d824969264dd9a6f29751d97",
    CHUNK_SCRIPT: "15cd62cd85a1820431e697bba1debcb6290c235ae7f6937dab6e3ce252181904",
}

KEEP_METADATA = (
    "identifier",
    "chunk_index",
    "collection",
    "open_type",
    "language",
    "date",
    "title",
)


@dataclass(frozen=True)
class ProbePaths:
    name: str
    vectors: str
    ids: str
    manifest: str
    chunks: str
    vectors_sha256: str
    ids_sha256: str
    manifest_sha256: str
    chunks_sha256: str


def _probe_paths(
    name: str,
    *,
    vectors_sha256: str,
    ids_sha256: str,
    manifest_sha256: str,
    chunks_sha256: str,
) -> ProbePaths:
    embedding_root = (
        f"/data/embeddings/common-corpus-{name}-chunked-120-"
        "all-MiniLM-L6-v2/train"
    )
    return ProbePaths(
        name=name,
        vectors=os.path.join(embedding_root, "data-00000-of-00001.npy"),
        ids=os.path.join(embedding_root, "corpus_ids.json"),
        manifest=os.path.join(embedding_root, "manifest.json"),
        chunks=(
            f"/data/chunks/common-corpus-{name}-chunked-120/train/"
            "000_00000.parquet"
        ),
        vectors_sha256=vectors_sha256,
        ids_sha256=ids_sha256,
        manifest_sha256=manifest_sha256,
        chunks_sha256=chunks_sha256,
    )


PROBES = {
    item.name: item
    for item in (
        _probe_paths(
            "code",
            vectors_sha256="e7f23e3ec86f3057ccc445be35bb389939ec8cef5a0747d229bf7854a25c494a",
            ids_sha256="6427ac4e0585e8cbc3dbf97f5c1ec17bdd744f4f9a7822bf87d3e3fe6ad4f66c",
            manifest_sha256="8183babde0338d8b66b2c72e485a3c686f8276046abbe0d642bdd6bbb7089000",
            chunks_sha256="8eed0dd528f9123692c3b9c2772393993bd74366966637bb6663f2bce3774efe",
        ),
        _probe_paths(
            "science",
            vectors_sha256="376de5b4716321cf73bd8014dfec4e796a36ef58a8427f5715631cd69feda2a4",
            ids_sha256="bb60848d36f16150f8c4f0454b8de0713a6ae83da2104ca47f1f67c5d68da52e",
            manifest_sha256="baa001c20d30cd7d9e25b4c5f87613dedf2065f4dec9c369a95ef50164731ca9",
            chunks_sha256="34b7e353e940782f1818aa8176b2becce9dec50e2a4ae851a7912149daeca779",
        ),
        _probe_paths(
            "latin",
            vectors_sha256="a24e5544ab1db794c770734245e420b7f7341481c6a6cc5d49902881d6a58b2c",
            ids_sha256="7ff15265efdaf2e283a1d18d7bbec77f7da6b6a62dd1aed5d75a2a0cf6e4197b",
            manifest_sha256="8bc282cbb6d52e2d429e3d79663aae7bb3da6f51d259d4126e25732d076c4571",
            chunks_sha256="f55c2ba887717ffd9f8b203f06d61ec9257854dcac9e07963013e25f68fe8921",
        ),
    )
}
_VERIFIED_CONTROL_MEMBERS: dict[str, dict[str, Any]] = {}
_MODEL_LOADER: Callable[[], Any] | None = None


def configure_map(
    *,
    round_id: str,
    map_label: str,
    model_path: str,
    model_sha256: str,
    coordinate_receipt: str,
    coordinate_receipt_sha256: str,
    model_loader: Callable[[], Any] | None = None,
) -> None:
    """Reuse the accepted R0035 probe definitions on another reviewed map."""
    global ROUND_ID, MAP_LABEL, MODEL_PATH, COORDINATE_RECEIPT, _MODEL_LOADER
    previous_model = MODEL_PATH
    previous_coordinates = COORDINATE_RECEIPT
    ROUND_ID = str(round_id)
    MAP_LABEL = str(map_label)
    MODEL_PATH = os.path.realpath(model_path)
    COORDINATE_RECEIPT = os.path.realpath(coordinate_receipt)
    EXACT_SHA256.pop(previous_model, None)
    EXACT_SHA256.pop(previous_coordinates, None)
    EXACT_SHA256[MODEL_PATH] = str(model_sha256)
    EXACT_SHA256[COORDINATE_RECEIPT] = str(coordinate_receipt_sha256)
    _MODEL_LOADER = model_loader


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _validate_seal(value: dict[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise RuntimeError(f"{label} identity seal is invalid")


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _seed(name: str, purpose: str) -> int:
    digest = sha256_bytes(f"{BASE_SEED}:{name}:{purpose}".encode("utf-8"))
    return int(digest[:8], 16) & 0x7FFFFFFF


def deterministic_split(name: str, n_rows: int = PROBE_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, disjoint corpus/query rows for one registered probe."""
    if n_rows != PROBE_ROWS:
        raise ValueError(f"R0035 requires exactly {PROBE_ROWS} rows, got {n_rows}")
    rng = np.random.RandomState(_seed(name, "probe-query-split"))
    query = np.sort(rng.choice(n_rows, QUERY_ROWS, replace=False)).astype(np.int64)
    keep = np.ones(n_rows, dtype=bool)
    keep[query] = False
    corpus = np.flatnonzero(keep).astype(np.int64)
    if len(corpus) != CORPUS_ROWS or len(query) != QUERY_ROWS:
        raise RuntimeError("R0035 split cardinality changed")
    return corpus, query


def _normalize_metadata(value: Any) -> Any:
    if value is None:
        return None
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_metadata(row.get(key)) for key in KEEP_METADATA}


def _projection_row_id(metadata: dict[str, Any], row: int) -> str:
    """Stable chunk-level ID for registry explorers.

    ``identifier`` names the source document and is intentionally repeated for
    every chunk.  Binding both ``chunk_index`` and the proved vector row keeps
    labels unique without discarding the human-recognisable document identity.
    """
    return json.dumps(
        {
            "identifier": metadata.get("identifier"),
            "chunk_index": metadata.get("chunk_index"),
            "vector_row": int(row),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def verify_source_mapping(paths: ProbePaths) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Prove that stored vector row i corresponds to parquet row i."""
    import pandas as pd

    vectors = np.load(paths.vectors, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (PROBE_ROWS, 384) or np.dtype(vectors.dtype) != np.dtype(np.float32):
        raise RuntimeError(
            f"{paths.name} vectors must be exactly ({PROBE_ROWS}, 384) fp32; "
            f"observed shape={vectors.shape} dtype={vectors.dtype}"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"{paths.name} vectors contain non-finite values")

    manifest = _read_json(paths.manifest)
    expected_manifest = {
        "collection": paths.name,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "n_chunks": PROBE_ROWS,
        "dtype": "float32",
        "npy": paths.vectors,
        "source_chunks": paths.chunks,
    }
    changed = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if changed:
        raise RuntimeError(f"{paths.name} manifest tuple changed: {changed}")

    ids_raw = _read_json(paths.ids)
    if not isinstance(ids_raw, list) or len(ids_raw) != PROBE_ROWS:
        raise RuntimeError(f"{paths.name} identity sidecar is not exactly 50,000 rows")
    ids = [_metadata_projection(item) for item in ids_raw]
    frame = pd.read_parquet(paths.chunks, columns=[*KEEP_METADATA]).iloc[:PROBE_ROWS]
    if len(frame) != PROBE_ROWS:
        raise RuntimeError(f"{paths.name} source parquet has fewer than 50,000 rows")
    source_ids = [_metadata_projection(item) for item in frame.to_dict(orient="records")]
    first_mismatch = next(
        (index for index, (left, right) in enumerate(zip(ids, source_ids)) if left != right),
        None,
    )
    if first_mismatch is not None:
        raise RuntimeError(
            f"{paths.name} vector/text row mapping failed at row {first_mismatch}: "
            f"sidecar={ids[first_mismatch]!r} source={source_ids[first_mismatch]!r}"
        )
    sidecar_digest = sha256_bytes(canonical_json(ids))
    source_digest = sha256_bytes(canonical_json(source_ids))
    if sidecar_digest != source_digest:
        raise RuntimeError(f"{paths.name} metadata mapping digests differ")
    return vectors, ids, {
        "proved": True,
        "rows_compared": PROBE_ROWS,
        "ordered_metadata_sha256": sidecar_digest,
        "vector_observation": {
            "shape": [PROBE_ROWS, 384],
            "dtype": np.dtype(vectors.dtype).str,
            "ordered_array_sha256": ordered_array_sha256(vectors),
        },
    }


def verify_exact_identities() -> dict[str, dict[str, Any]]:
    expected = dict(EXACT_SHA256)
    for probe in PROBES.values():
        expected.update(
            {
                probe.vectors: probe.vectors_sha256,
                probe.ids: probe.ids_sha256,
                probe.manifest: probe.manifest_sha256,
                probe.chunks: probe.chunks_sha256,
            }
        )
    observed: dict[str, dict[str, Any]] = {}
    for path, digest in sorted(expected.items()):
        signature = expected_input_signature(path)
        if signature["sha256"] != digest:
            raise RuntimeError(
                f"R0035 exact input changed: {path}: "
                f"expected={digest} observed={signature['sha256']}"
            )
        observed[path] = signature
    # The registered HF snapshot intentionally contains symlinks, so use the
    # accepted R0028 snapshot identity algorithm instead of treating it as a
    # normal directory.  This hashes all eleven resolved members, not names or
    # sizes alone.
    from experiments.universality_panel import _hf_snapshot_signature

    snapshot = _hf_snapshot_signature(SENTENCE_MODEL_SNAPSHOT)
    if snapshot["sha256"] != SENTENCE_MODEL_SNAPSHOT_SHA256:
        raise RuntimeError(
            "R0035 sentence-model snapshot changed: "
            f"expected={SENTENCE_MODEL_SNAPSHOT_SHA256} "
            f"observed={snapshot['sha256']}"
        )
    observed[SENTENCE_MODEL_SNAPSHOT] = snapshot
    return observed


def _sentence_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SENTENCE_MODEL_SNAPSHOT, device="cuda")


def _cosine_agreement(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    cosine = np.einsum("ij,ij->i", left, right) / (
        np.maximum(np.linalg.norm(left, axis=1), 1e-30)
        * np.maximum(np.linalg.norm(right, axis=1), 1e-30)
    )
    return {
        "mean_cosine": float(np.mean(cosine, dtype=np.float64)),
        "min_cosine": float(np.min(cosine)),
        "max_cosine": float(np.max(cosine)),
        "sample_count": int(len(cosine)),
        "sample_cosines_sha256": ordered_array_sha256(cosine.astype(np.float32)),
    }


def _load_map() -> Any:
    if sha256_file(MODEL_PATH) != EXACT_SHA256[MODEL_PATH]:
        raise RuntimeError(f"accepted {MAP_LABEL} model bytes changed")
    if _MODEL_LOADER is not None:
        return _MODEL_LOADER()
    from basemap.pumap.parametric_umap import ParametricUMAP

    return ParametricUMAP.load(MODEL_PATH, device="cuda")


def _project(model: Any, vectors: np.ndarray, *, batch_size: int = 65_536) -> np.ndarray:
    import torch

    model.model.eval()
    output = np.empty((len(vectors), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = np.asarray(vectors[start : start + batch_size], dtype=np.float32)
            tensor = torch.from_numpy(np.array(batch, copy=True)).to(model.device)
            projected = model.model(tensor).detach().cpu().numpy().astype(np.float32)
            if projected.shape != (len(batch), 2) or not np.isfinite(projected).all():
                raise RuntimeError("R0035 projection produced malformed coordinates")
            output[start : start + len(batch)] = projected
            del tensor
        torch.cuda.synchronize()
    return output


def run_canary(*, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="R0035 canary output")
    started = time.monotonic()
    exact_inputs = verify_exact_identities()
    sentence_model = _sentence_model()
    probes: dict[str, Any] = {}
    for name, paths in PROBES.items():
        try:
            vectors, _, mapping = verify_source_mapping(paths)
            import pandas as pd

            frame = pd.read_parquet(paths.chunks, columns=["chunk_text"]).iloc[:PROBE_ROWS]
            rng = np.random.RandomState(_seed(name, "embedding-canary"))
            rows = np.sort(rng.choice(PROBE_ROWS, CANARY_ROWS, replace=False)).astype(np.int64)
            texts = frame.iloc[rows]["chunk_text"].astype(str).tolist()
            embedded = sentence_model.encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            agreement = _cosine_agreement(embedded, np.asarray(vectors[rows]))
            included = agreement["mean_cosine"] >= CANARY_MIN_MEAN_COSINE
            probes[name] = {
                "status": "included" if included else "blocked",
                "blocker": None if included else "re-embedded mean cosine below 0.98",
                "rule": (
                    "all 50,000 metadata rows match source order and mean cosine of "
                    "40 seed-fixed re-embedded chunk texts is at least 0.98"
                ),
                "mapping": mapping,
                "sample_rows_sha256": ordered_array_sha256(rows),
                "sample_texts_sha256": sha256_bytes(canonical_json(texts)),
                "agreement": agreement,
            }
        except Exception as exc:  # publish the blocker before the panel fails closed
            probes[name] = {
                "status": "blocked",
                "blocker": f"{type(exc).__name__}: {exc}",
                "rule": "fail closed when exact vector/text mapping or model identity is unproved",
            }

    model = _load_map()
    smoke = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)[:1000]
    smoke_coords = _project(model, smoke, batch_size=1000)
    included = sorted(name for name, value in probes.items() if value["status"] == "included")
    body = {
        "schema": f"round{ROUND_ID}-common-corpus-canary-v1",
        "round_id": ROUND_ID,
        "seed": BASE_SEED,
        "registered_probes": sorted(PROBES),
        "probe_canaries": probes,
        "included_probes": included,
        "can_build_panel": included == sorted(PROBES),
        "accepted_map": {
            "model": exact_inputs[MODEL_PATH],
            "coordinate_receipt": exact_inputs[COORDINATE_RECEIPT],
            "review_0019": exact_inputs[R0019_REVIEW],
        },
        "corrected_ood_semantics": {
            "review_0028": exact_inputs[R0028_REVIEW],
            "probe_source_dtype": "float32",
            "projection_compute_dtype": "float32",
            "cosine_compute_dtype": "float32",
        },
        "embedding_provenance": {
            "snapshot": exact_inputs[SENTENCE_MODEL_SNAPSHOT],
            "embed_script": exact_inputs[EMBED_SCRIPT],
            "chunk_script": exact_inputs[CHUNK_SCRIPT],
        },
        "smoke_projection": {
            "rows": 1000,
            "finite": True,
            "coordinates_sha256": ordered_array_sha256(smoke_coords),
        },
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "verdict.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "verdict": expected_input_signature(path)}


def _matmul_settings() -> dict[str, Any]:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "torch_dtype": "float32",
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _cosine_topk(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        c = torch.as_tensor(np.array(corpus, dtype=np.float32, copy=True), device=device)
        q = torch.as_tensor(np.array(queries, dtype=np.float32, copy=True), device=device)
        c = c / torch.clamp(torch.linalg.norm(c, dim=1, keepdim=True), min=1e-30)
        q = q / torch.clamp(torch.linalg.norm(q, dim=1, keepdim=True), min=1e-30)
        scores = q @ c.T
        values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        result_indices = indices.cpu().numpy().astype(np.int64)
        result_values = values.cpu().numpy().astype(np.float32)
        del c, q, scores, values, indices
        if device == "cuda":
            torch.cuda.synchronize()
    return result_indices, result_values


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
        result = indices.cpu().numpy().astype(np.int64)
        del c, q, c2, q2, scores, indices
        if device == "cuda":
            torch.cuda.synchronize()
    return result


def _score_one(
    *,
    name: str,
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    corpus_coords: np.ndarray,
    query_coords: np.ndarray,
    device: str = "cuda",
) -> dict[str, Any]:
    if len(corpus_vectors) != CORPUS_ROWS or len(query_vectors) != QUERY_ROWS:
        raise RuntimeError(f"{name} is not the registered 49,500/500 shape")
    hit_k = int(math.ceil(HIT_FRACTION * len(corpus_vectors)))
    true_top11, true_scores = _cosine_topk(
        corpus_vectors, query_vectors, TRUE_K + 1, device=device
    )
    true_top10 = true_top11[:, :TRUE_K]
    hit_neighbors = _l2_topk(corpus_coords, query_coords, hit_k, device=device)
    per_query = np.empty(QUERY_ROWS, dtype=np.float32)
    for index in range(QUERY_ROWS):
        per_query[index] = (
            np.isin(true_top10[index], hit_neighbors[index]).sum() / TRUE_K
        )
    ffr = float(np.mean(per_query, dtype=np.float64))
    boundary_ties = np.isclose(
        true_scores[:, TRUE_K - 1], true_scores[:, TRUE_K], rtol=0.0, atol=1e-7
    )
    return {
        "name": name,
        "corpus_rows": CORPUS_ROWS,
        "query_rows": QUERY_ROWS,
        "true_k": TRUE_K,
        "hit_k": hit_k,
        "ffr": ffr,
        "per_query_ffr_sha256": ordered_array_sha256(per_query),
        "true_top10_sha256": ordered_array_sha256(true_top10),
        "hit_neighbors_sha256": ordered_array_sha256(hit_neighbors),
        "top10_boundary_tied_query_count_at_1e_7": int(boundary_ties.sum()),
        "map_dispersion": {
            "axis_mean": np.mean(corpus_coords, axis=0).astype(float).tolist(),
            "axis_std": np.std(corpus_coords, axis=0).astype(float).tolist(),
            "axis_span": (
                np.max(corpus_coords, axis=0) - np.min(corpus_coords, axis=0)
            ).astype(float).tolist(),
        },
    }


def _load_input_pack_members() -> dict[str, dict[str, Any]]:
    manifest = _read_json(INPUT_PACK_MANIFEST)
    members = manifest["capability_payload"]["materialized_fp16"]["ordered_members"]
    return {
        os.path.realpath(item["path"]): {
            "canonical_path": os.path.realpath(item["path"]),
            "kind": "file",
            "bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in members
    }


def _gather_control_rows(rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    members = _load_input_pack_members()
    output: np.ndarray | None = None
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    chunks: list[dict[str, Any]] = []
    source_dtype: np.dtype | None = None
    for chunk_index in np.unique(sorted_rows // 1_000_000).tolist():
        chunk_index = int(chunk_index)
        positions = np.flatnonzero(
            (sorted_rows >= chunk_index * 1_000_000)
            & (sorted_rows < (chunk_index + 1) * 1_000_000)
        )
        path = (
            "/data/latent-basemap/runs/round-0010/materialized/"
            f"chunk-{chunk_index:05d}/embeddings.npy"
        )
        canonical = os.path.realpath(path)
        registered = members.get(canonical)
        if registered is None:
            raise RuntimeError(f"control chunk is absent from accepted input pack: {path}")
        observed = _VERIFIED_CONTROL_MEMBERS.get(canonical)
        if observed is None:
            observed = expected_input_signature(path)
            if observed != registered:
                raise RuntimeError(
                    f"control chunk bytes differ from accepted input pack: {path}: "
                    f"expected={registered} observed={observed}"
                )
            _VERIFIED_CONTROL_MEMBERS[canonical] = observed
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if source_dtype is None:
            source_dtype = np.dtype(array.dtype)
            output = np.empty((len(rows), array.shape[1]), dtype=source_dtype)
        if np.dtype(array.dtype) != source_dtype:
            raise RuntimeError("matched-control source dtypes differ")
        output[order[positions]] = array[sorted_rows[positions] - chunk_index * 1_000_000]
        chunks.append({**observed, "array_dtype": np.dtype(array.dtype).str})
    if output is None or source_dtype != np.dtype(np.float16):
        raise RuntimeError("matched-control corpus is empty or no longer native fp16")
    return output, {
        "kind": "accepted-input-pack-selection",
        "selected_rows_sha256": ordered_array_sha256(rows),
        "source_dtype": source_dtype.str,
        "selected_dtype": np.dtype(output.dtype).str,
        "chunks": chunks,
    }


def _matched_control(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.RandomState(_seed(name, "matched-control"))
    corpus_rows = np.sort(
        rng.choice(30_000_000, CORPUS_ROWS, replace=False)
    ).astype(np.int64)
    query_pool = np.load(MINILM_QUERIES, mmap_mode="r", allow_pickle=False)
    if np.dtype(query_pool.dtype) != np.dtype(np.float32):
        raise RuntimeError("held-out MiniLM query pool is no longer native fp32")
    query_rows = np.sort(
        rng.choice(len(query_pool), QUERY_ROWS, replace=False)
    ).astype(np.int64)
    corpus, corpus_source = _gather_control_rows(corpus_rows)
    queries = np.asarray(query_pool[query_rows])
    return corpus, queries, corpus_rows, query_rows, {
        "corpus": corpus_source,
        "queries": {
            **expected_input_signature(MINILM_QUERIES),
            "source_dtype": np.dtype(query_pool.dtype).str,
            "selected_dtype": np.dtype(queries.dtype).str,
            "selected_rows_sha256": ordered_array_sha256(query_rows),
        },
    }


def retention_verdict(retention: float | None) -> str:
    if retention is None:
        return "undefined-control-zero"
    if retention >= 0.7:
        return "pass"
    if retention < 0.5:
        return "failure"
    return "amber"


def run_panel(*, canary_path: str, output_root: str) -> dict[str, Any]:
    output_root = create_fresh_directory(output_root, label="R0035 panel output")
    started = time.monotonic()
    verify_exact_identities()
    canary = _read_json(canary_path)
    _validate_seal(canary, label="R0035 canary")
    if canary.get("schema") != f"round{ROUND_ID}-common-corpus-canary-v1":
        raise RuntimeError(f"R{ROUND_ID} Common Corpus canary schema changed")
    if not canary.get("can_build_panel"):
        blockers = {
            name: value.get("blocker")
            for name, value in canary.get("probe_canaries", {}).items()
            if value.get("status") != "included"
        }
        raise RuntimeError(f"R0035 source/model canary blocked panel: {blockers}")

    compute = _matmul_settings()
    model = _load_map()
    results: dict[str, Any] = {}
    for name, paths in PROBES.items():
        vectors, ids, mapping = verify_source_mapping(paths)
        corpus_rows, query_rows = deterministic_split(name, len(vectors))
        corpus = np.asarray(vectors[corpus_rows])
        queries = np.asarray(vectors[query_rows])
        if np.dtype(corpus.dtype) != np.dtype(np.float32) or np.dtype(queries.dtype) != np.dtype(np.float32):
            raise RuntimeError(f"{name} probe narrowed before fp32 compute boundary")
        corpus_coords = _project(model, corpus)
        query_coords = _project(model, queries)

        control_corpus, control_queries, control_rows, control_query_rows, control_source = (
            _matched_control(name)
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
            name=f"{name}-shape-matched-in-domain-control",
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
        coordinates_path = os.path.join(output_root, f"{name}-coordinates.npz")
        atomic_save_new_npz(
            coordinates_path,
            immutable=True,
            probe_corpus_coords=corpus_coords,
            probe_query_coords=query_coords,
            control_corpus_coords=control_corpus_coords,
            control_query_coords=control_query_coords,
            probe_corpus_rows=corpus_rows,
            probe_query_rows=query_rows,
            probe_corpus_ids=np.asarray([
                _projection_row_id(ids[index], index)
                for index in corpus_rows.tolist()
            ]),
            probe_query_ids=np.asarray([
                _projection_row_id(ids[index], index)
                for index in query_rows.tolist()
            ]),
            control_corpus_rows=control_rows,
            control_query_rows=control_query_rows,
        )
        results[name] = {
            "status": "included",
            "probe": probe_score,
            "shape_matched_in_domain_control": control_score,
            "retention": retention,
            "verdict": retention_verdict(retention),
            "source_mapping": mapping,
            "sample_hashes": {
                "probe_corpus_rows": ordered_array_sha256(corpus_rows),
                "probe_query_rows": ordered_array_sha256(query_rows),
                "probe_corpus_metadata": sha256_bytes(
                    canonical_json([ids[index] for index in corpus_rows.tolist()])
                ),
                "probe_query_metadata": sha256_bytes(
                    canonical_json([ids[index] for index in query_rows.tolist()])
                ),
                "control_corpus_rows": ordered_array_sha256(control_rows),
                "control_query_rows": ordered_array_sha256(control_query_rows),
            },
            "dtype_identity": {
                "probe_source": "float32",
                "probe_selected": "float32",
                "control_corpus_source": "float16",
                "control_corpus_selected": "float16",
                "control_query_source": "float32",
                "control_query_selected": "float32",
                "projection_compute": "float32",
                "cosine_compute": "float32",
            },
            "control_source": control_source,
            "coordinates": expected_input_signature(coordinates_path),
        }

    body = {
        "schema": "common-corpus-ood-panel-v1",
        "round_id": ROUND_ID,
        "map": {
            "label": MAP_LABEL,
            "model": expected_input_signature(MODEL_PATH),
            "coordinate_receipt": expected_input_signature(COORDINATE_RECEIPT),
        },
        "canary": expected_input_signature(canary_path),
        "seed": BASE_SEED,
        "split": {
            "per_probe_source_rows": PROBE_ROWS,
            "corpus_rows": CORPUS_ROWS,
            "query_rows": QUERY_ROWS,
            "query_selection": "sorted without-replacement RandomState rows; collection/purpose seed-bound",
            "queries_disjoint_from_probe_corpus": True,
        },
        "matched_control": {
            "corpus_source": "accepted R0013 30M materialized fp16 input pack",
            "query_source": "accepted held-out MiniLM fp32 query pool",
            "shape_exact": [CORPUS_ROWS, QUERY_ROWS],
        },
        "scoring_formula": {
            "true_set": "top-10 exact brute-force fp32 cosine neighbors within each 49,500-row corpus",
            "hit_set": "top ceil(0.01 * 49,500) = 495 exact fp32 L2 map-space neighbors",
            "ffr": "mean fraction of true top-10 inside the 1% map-space hit set",
            "retention": "ffr(probe) / ffr(shape-matched in-domain control)",
            "verdict": "pass >= 0.7; amber [0.5, 0.7); failure < 0.5",
        },
        "compute": compute,
        "probes": results,
        "acceptance": {
            "all_three_source_model_canaries_included": True,
            "probe_verdicts": {name: value["verdict"] for name, value in results.items()},
            "all_retention_at_least_0_7": all(
                value["retention"] is not None and value["retention"] >= 0.7
                for value in results.values()
            ),
            "named_failures": [
                name
                for name, value in results.items()
                if value["retention"] is not None and value["retention"] < 0.5
            ],
            "amber_probes": [
                name
                for name, value in results.items()
                if value["retention"] is not None and 0.5 <= value["retention"] < 0.7
            ],
        },
        "wall_seconds": time.monotonic() - started,
    }
    receipt = _seal(body)
    path = os.path.join(output_root, "common-corpus-ood-panel-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "panel": expected_input_signature(path)}


def run_canary_job(_active: dict[str, Any], job: dict[str, Any]) -> None:
    run_canary(output_root=job["outputs"][0])


def run_panel_job(_active: dict[str, Any], job: dict[str, Any]) -> None:
    run_panel(
        canary_path=os.path.join(job["canary_output"], "verdict.json"),
        output_root=job["outputs"][0],
    )

"""O1 — PROMPTED jina embedding of the exact same 200k rows as the unprompted
jina testbed at /data/latent-basemap/jina-en-200k, so prompted vs. unprompted
maps become comparable.

Row identity contract
----------------------
``/data/latent-basemap/jina-en-200k/sample_indices.npy`` holds 200000 int64
row ids into the CONCATENATED (ordered, positional) source embedding shards at
``/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train``.
Those embeddings were produced from the CONCATENATED (ordered, positional)
text shards at ``/data/chunks/fineweb-edu-sample-10BT-chunked-500/train``.

Verified alignment (documented here, re-asserted at runtime by
``verify_shard_alignment``): the first 11 text-parquet shards
(``data-00000-of-00099.parquet`` .. ``data-00010-of-00099.parquet``) have ROW
COUNTS IDENTICAL, shard-for-shard, to the 11 embedding shards that actually
exist on disk (``data-00000-of-00099.npy`` .. ``data-00010-of-00099.npy``;
99 is a total-shard-count in the filename, not the number of shards actually
produced by the truncated embedding run):

  265419, 260422, 260748, 264661, 266637, 263875, 264737, 262636, 260251,
  260043, 260933  (total 2,890,362 == max(sample_indices) + 1)

This is exactly the shard layout ``experiments/build_testbed.py`` samples over
(sort shards by name, concatenate positionally, offsets = cumsum(sizes)), so
``global_idx -> (shard, local_row)`` is identical for text and embeddings.
Cross-shard row-count equality is asserted before any text is read — a silent
mis-alignment (e.g. a differently-chunked text corpus) would show up as a
per-shard size mismatch and hard-fails the run.

Model / faithfulness
---------------------
The model card at
``/data/hf/hub/models--jinaai--jina-embeddings-v5-text-nano-retrieval`` is a
sentence-transformers checkpoint: Transformer(EuroBertModel) -> Pooling
(``lasttoken``, ``include_prompt=True``) -> Normalize. NOTE this is
LAST-TOKEN pooling, not mean pooling (contra the workspace notes) — do not
assume mean pooling when reasoning about this model. Its own
``config_sentence_transformers.json`` ships native prompts
``{"query": "Query: ", "document": "Document: "}``; the literal prefix this
script prepends is byte-identical to that ``document`` prompt (verified: an
explicit ``"Document: " + text`` encode and a ``prompt_name="document"``
encode produce bit-identical output).

This script's local backend is ``sentence-transformers`` (``trust_remote_code=
True``), loaded from the HF cache (``HF_HOME=/data/hf``) at float32 compute
precision (the checkpoint's declared dtype is bfloat16; forcing float32
compute makes ``Normalize`` output land within 1e-6 of unit norm instead of
drifting ~0.2%). Before the prompted pass, ``measure_faithfulness`` embeds a
handful of the 200k rows WITHOUT the prompt through this exact pipeline and
compares cosine similarity to the stored (Modal/vLLM-produced) reference
embeddings for those same rows. A held-out 8-row probe (seed 0, disjoint from
the smoke set) measured **mean cosine 0.9950, min 0.9860** — comfortably
above the 0.98 faithfulness floor demanded by the plan. See EMBED_PROMPTED_
FAITHFULNESS in the manifest for the measurement made on the run that
actually wrote the output.

Usage (orchestrator, under a held/inherited GPU lease):
  python experiments/embed_prompted_200k.py \
      --testbed /data/latent-basemap/jina-en-200k \
      --out /data/latent-basemap/jina-en-200k-prompted \
      --text-dir /data/chunks/fineweb-edu-sample-10BT-chunked-500/train \
      --embed-dir /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train \
      --batch-size 256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       ordered_array_sha256, sha256_bytes)
from basemap.output_safety import (atomic_save_new_npy, atomic_write_new_json,
                                   create_fresh_directory)
from basemap.round0005_staging import (
    ROUND0005_DIMENSIONS, ROUND0005_MODEL_ID, ROUND0005_MODEL_REVISION,
    ROUND0005_NORMALIZATION, ROUND0005_POOLING,
)

PROMPT_PREFIX = "Document: "
MODEL_ID = ROUND0005_MODEL_ID
MANIFEST_SCHEMA = "prompted_embed_manifest.v1"
CHUNK_SCHEMA = "jina_atomic_embedding_chunk.v1"
OUTER_CHUNK_ROWS = 25_000
REQUIRED_MANIFEST_KEYS = [
    "schema", "model_id", "model_commit", "prompt_prefix", "prompt_prefix_hex",
    "n_rows", "dim", "dtype", "row_ids_sha", "text_shards", "text_shard_sha",
    "out_shards", "norm_check", "faithfulness", "canary", "created_utc",
]


# --------------------------------------------------------------------------
# Pure / CPU-testable helpers (row resolution, prompting, manifest shape)
# --------------------------------------------------------------------------

def build_shard_offsets(sizes) -> np.ndarray:
    """Cumulative offsets for a list of ordered shard sizes: offsets[i] is the
    first global row id in shard i; offsets[-1] is the total row count."""
    sizes = np.asarray(list(sizes), dtype=np.int64)
    return np.concatenate([[0], np.cumsum(sizes)])


def locate_shard(indices, offsets):
    """Vectorized global-id -> (shard_id, local_row), order-preserving (the
    output arrays are positionally aligned with ``indices`` as given, NOT
    sorted). Raises if any index is out of the covered range."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= offsets[-1]):
        raise ValueError(
            f"index range [{idx.min() if idx.size else None}, "
            f"{idx.max() if idx.size else None}] outside covered range "
            f"[0, {offsets[-1]}) — row resolution would silently misalign.")
    shard_ids = np.searchsorted(offsets, idx, side="right") - 1
    local = idx - offsets[shard_ids]
    return shard_ids, local


def assert_row_identity(resolved_ids, expected_ids, *, context="row resolution"):
    """The drift/abort guard: resolved text row ids must equal the requested
    ids EXACTLY, in the SAME order (not just as a set). Any mismatch — count,
    order, or content — means the embedding would be silently mispaired with
    the wrong text, which invalidates the prompted/unprompted comparison.
    Raises ValueError with a description of the first mismatch."""
    resolved_ids = np.asarray(resolved_ids)
    expected_ids = np.asarray(expected_ids)
    if resolved_ids.shape != expected_ids.shape:
        raise ValueError(
            f"{context}: resolved {resolved_ids.shape} rows but expected "
            f"{expected_ids.shape} — row count drift.")
    bad = np.nonzero(resolved_ids != expected_ids)[0]
    if len(bad):
        i = int(bad[0])
        raise ValueError(
            f"{context}: index drift at position {i}: resolved id "
            f"{resolved_ids[i]} != expected {expected_ids[i]} "
            f"({len(bad)}/{len(resolved_ids)} positions differ overall "
            f"-> ABORT before any embedding work).")
    return True


def apply_prompt(texts, prefix: str = PROMPT_PREFIX):
    """Prepend the literal prompt string to every text. Order-preserving."""
    return [prefix + t for t in texts]


def l2_norm_stats(X) -> dict:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    return {
        "mean": float(norms.mean()), "std": float(norms.std()),
        "min": float(norms.min()), "max": float(norms.max()),
        "is_unit_norm": bool(abs(norms.mean() - 1.0) < 1e-3 and norms.std() < 1e-2),
    }


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def sha_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _atomic_json(path: str, value: dict) -> None:
    atomic_write_new_json(path, value, immutable=True)


def _atomic_npy(path: str, value: np.ndarray) -> None:
    atomic_save_new_npy(path, value, immutable=True)


def ordered_text_sha256(texts) -> str:
    """Hash every UTF-8 text with an unambiguous length prefix, in order."""
    h = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        h.update(len(raw).to_bytes(8, "big")); h.update(raw)
    return h.hexdigest()


def _chunk_context(*, chunk_index: int, start: int, stop: int, row_ids,
                   texts, prompt_prefix: str, model_id: str, model_commit: str,
                   pooling: str, compute_dtype: str, output_dtype: str,
                   normalization: str, requested_batch_size: int) -> dict:
    if (not isinstance(model_commit, str) or
            not re.fullmatch(r"[0-9a-f]{40}", model_commit) or
            len(set(model_commit)) == 1):
        raise ValueError("embedding chunk model revision must be full immutable 40-hex")
    prompted = apply_prompt(texts, prompt_prefix)
    return {
        "schema": CHUNK_SCHEMA,
        "chunk_index": int(chunk_index),
        "source_position_range": [int(start), int(stop)],
        "source_global_id_first": int(row_ids[0]) if len(row_ids) else None,
        "source_global_id_last": int(row_ids[-1]) if len(row_ids) else None,
        "source_row_count": int(len(row_ids)),
        "source_ids_ordered_sha256": ordered_array_sha256(
            np.asarray(row_ids, dtype=np.int64)),
        "source_text_ordered_sha256": ordered_text_sha256(texts),
        "prompt_bytes_hex": prompt_prefix.encode("utf-8").hex(),
        "prompted_text_ordered_sha256": ordered_text_sha256(prompted),
        "model_id": model_id,
        "model_commit": model_commit,
        "pooling": pooling,
        "compute_dtype": compute_dtype,
        "output_dtype": output_dtype,
        "normalization": normalization,
        "requested_batch_size": int(requested_batch_size),
    }


def _load_completed_chunk(receipt_path: str, output_path: str,
                          expected_context: dict) -> dict:
    """Load a completed chunk or raise; corrupt state is never treated as absent."""
    if not os.path.isfile(receipt_path) or not os.path.isfile(output_path):
        raise RuntimeError(
            f"partial existing embedding chunk refuses resume: receipt={receipt_path} "
            f"output={output_path}")
    try:
        with open(receipt_path, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"corrupt existing embedding chunk receipt: {receipt_path}: {exc}") from exc
    if receipt.get("status") != "complete":
        raise RuntimeError(f"existing embedding chunk receipt is not complete: {receipt_path}")
    if receipt.get("context") != expected_context:
        raise RuntimeError(
            f"existing embedding chunk context mismatch: expected={expected_context!r} "
            f"observed={receipt.get('context')!r}")
    output = receipt.get("output_signature") or {}
    if output.get("canonical_path") != os.path.realpath(output_path):
        raise RuntimeError("existing embedding chunk receipt names the wrong output")
    try:
        observed = expected_input_signature(output_path)
    except Exception as exc:
        raise RuntimeError(f"existing embedding chunk output cannot be signed: {exc}") from exc
    if observed != output:
        raise RuntimeError(
            f"existing embedding chunk output is corrupt: expected={output!r} observed={observed!r}")
    values = np.load(output_path, mmap_mode="r")
    if list(values.shape) != receipt.get("output_shape") or values.dtype != np.dtype("float32"):
        raise RuntimeError("existing embedding chunk shape/dtype mismatch")
    if not np.isfinite(values).all() or ordered_array_sha256(values) != receipt.get(
            "output_ordered_sha256"):
        raise RuntimeError("existing embedding chunk payload identity mismatch")
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    if len(norms) and float(np.max(np.abs(norms - 1.0))) > 1e-3:
        raise RuntimeError("existing embedding chunk falsely claims L2 normalization")
    return receipt


def build_manifest(*, model_id, model_commit, prompt_prefix, n_rows, dim, dtype,
                    row_ids, text_shards, text_shard_sha, out_shards, norm_check,
                    faithfulness, canary, extra=None) -> dict:
    """Assemble the sidecar manifest. Schema is checked structurally by the
    CPU test (``REQUIRED_MANIFEST_KEYS`` all present, correct coarse types)."""
    import datetime
    man = {
        "schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "model_commit": model_commit,
        "prompt_prefix": prompt_prefix,
        "prompt_prefix_hex": prompt_prefix.encode("utf-8").hex(),
        "n_rows": int(n_rows),
        "dim": int(dim),
        "dtype": str(dtype),
        "row_ids_sha": _sha_bytes(np.asarray(row_ids, dtype=np.int64).tobytes()),
        "text_shards": list(text_shards),
        "text_shard_sha": dict(text_shard_sha),
        "out_shards": list(out_shards),
        "norm_check": norm_check,
        "faithfulness": faithfulness,
        "canary": canary,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        man.update(extra)
    return man


def validate_manifest_shape(man: dict) -> None:
    """Structural well-formedness check reused by the CPU test and available
    for the orchestrator to call post-hoc."""
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in man]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    if man["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema {man['schema']} != {MANIFEST_SCHEMA}")
    if not isinstance(man["n_rows"], int) or man["n_rows"] <= 0:
        raise ValueError("manifest n_rows must be a positive int")
    if man["prompt_prefix_hex"] != man["prompt_prefix"].encode("utf-8").hex():
        raise ValueError("manifest prompt_prefix_hex does not match prompt_prefix")
    fh = man["faithfulness"]
    for k in ("mean_cosine", "min_cosine", "n_probe", "passed"):
        if k not in fh:
            raise ValueError(f"manifest faithfulness missing key {k}")


# --------------------------------------------------------------------------
# I/O helpers (real data — used by main(), also importable for reuse by
# experiments/build_prompted_graph.py)
# --------------------------------------------------------------------------

def embed_shard_sizes(embed_dir: str):
    shards = sorted(f for f in os.listdir(embed_dir) if f.endswith(".npy"))
    if not shards:
        raise ValueError(f"{embed_dir}: no .npy shards found")
    sizes, dim = [], None
    for s in shards:
        m = np.load(os.path.join(embed_dir, s), mmap_mode="r")
        sizes.append(int(m.shape[0]))
        if dim is None:
            dim = int(m.shape[1])
        elif dim != m.shape[1]:
            raise ValueError(f"{s}: dim {m.shape[1]} != {dim}")
    return shards, sizes, dim


def text_shard_sizes(text_dir: str, n_shards: int):
    import pyarrow.parquet as pq
    shards = sorted(f for f in os.listdir(text_dir) if f.endswith(".parquet"))[:n_shards]
    sizes = [pq.ParquetFile(os.path.join(text_dir, s)).metadata.num_rows for s in shards]
    return shards, sizes


def verify_shard_alignment(embed_dir: str, text_dir: str):
    """Assert the text parquet shards and embedding shards are POSITIONALLY
    row-count-aligned, shard for shard (P0-E-style identity check, applied to
    the text side which has no built-in fingerprint). Returns
    (shard_names, sizes, offsets, dim) to drive row resolution."""
    embed_shards, embed_sizes, dim = embed_shard_sizes(embed_dir)
    text_shards, text_sizes = text_shard_sizes(text_dir, len(embed_shards))
    if len(text_shards) != len(embed_shards):
        raise ValueError(
            f"shard count mismatch: {len(text_shards)} text shards vs "
            f"{len(embed_shards)} embedding shards — alignment cannot be "
            f"verified; refuse to proceed.")
    for i, (te, ee, ts, es) in enumerate(zip(text_shards, embed_shards, text_sizes, embed_sizes)):
        if ts != es:
            raise ValueError(
                f"shard {i} row-count mismatch: text {te}={ts} rows vs "
                f"embedding {ee}={es} rows — text/embedding corpora are NOT "
                f"positionally aligned; refuse to proceed (this would silently "
                f"pair the wrong text with the wrong embedding row).")
    offsets = build_shard_offsets(embed_sizes)
    return text_shards, embed_sizes, offsets, dim


def fetch_texts_for_indices(global_indices, text_dir: str, text_shards, offsets):
    """Fetch chunk_text for each id in ``global_indices``, IN THE SAME ORDER
    as given. Reads each needed shard's ``chunk_text`` column once (grouped),
    then reassembles by the caller's original order — never loads an unneeded
    shard, never materializes more than one shard's text column at a time."""
    import pyarrow.parquet as pq
    idx = np.asarray(global_indices, dtype=np.int64)
    shard_ids, local = locate_shard(idx, offsets)
    out = [None] * len(idx)
    for si in np.unique(shard_ids):
        pos = np.nonzero(shard_ids == si)[0]
        rows = local[pos]
        path = os.path.join(text_dir, text_shards[int(si)])
        col = pq.read_table(path, columns=["chunk_text"]).column("chunk_text")
        # pyarrow .take() gathers in the given (arbitrary) index order, so the
        # result is already positionally aligned with `pos` / `rows`.
        vals = col.take(rows.tolist()).to_pylist()
        for k, p in enumerate(pos):
            out[p] = vals[k]
    return out


def load_model(device: str = "cuda", dtype: str = "float32", *,
               model_path: str | None = None, model_revision: str | None = None):
    """Load the jina-v5-nano-retrieval sentence-transformers checkpoint from
    the local HF cache. Returns (model, commit_hash)."""
    if str(device).lower().startswith("cuda"):
        from basemap.run_controller import require_active_round0005_child_admission
        require_active_round0005_child_admission()
    os.environ.setdefault("HF_HOME", "/data/hf")
    import torch
    from sentence_transformers import SentenceTransformer
    torch_dtype = getattr(torch, dtype)
    model = SentenceTransformer(
        model_path or MODEL_ID, trust_remote_code=True, device=device,
        model_kwargs={"torch_dtype": torch_dtype},
        local_files_only=model_path is not None,
    )
    try:
        commit = model[0].auto_model.config._commit_hash
    except Exception:
        commit = None
    # A local SentenceTransformer tree normally has no runtime _commit_hash.
    # Never replace that honest None with a caller-supplied revision.  The
    # exact staged-file closure is the revision proof for local execution.
    if model_revision is not None and commit is not None and commit != model_revision:
        raise ValueError(
            f"loaded model runtime revision {commit!r} != expected {model_revision!r}")
    return model, commit


def inspect_loaded_jina_model(model) -> dict:
    """Return fail-closed runtime semantics from an already loaded model."""
    try:
        transformer_config = model[0].auto_model.config
        pooling = model[1]
        normalize = model[2]
        dimension_getter = (getattr(model, "get_embedding_dimension", None) or
                            model.get_sentence_embedding_dimension)
        dimensions = int(dimension_getter())
    except Exception as exc:
        raise ValueError(f"loaded Jina model module structure is invalid: {exc}") from exc
    runtime_pooling_mode = getattr(pooling, "pooling_mode", None)
    if isinstance(runtime_pooling_mode, str):
        pooling_flags = {
            "cls": runtime_pooling_mode == "cls",
            "mean": runtime_pooling_mode == "mean",
            "max": runtime_pooling_mode == "max",
            "mean_sqrt": runtime_pooling_mode == "mean_sqrt_len_tokens",
            "weighted_mean": runtime_pooling_mode == "weightedmean",
            "lasttoken": runtime_pooling_mode == "lasttoken",
        }
    else:
        pooling_flags = {
            "cls": bool(getattr(pooling, "pooling_mode_cls_token", False)),
            "mean": bool(getattr(pooling, "pooling_mode_mean_tokens", False)),
            "max": bool(getattr(pooling, "pooling_mode_max_tokens", False)),
            "mean_sqrt": bool(getattr(
                pooling, "pooling_mode_mean_sqrt_len_tokens", False)),
            "weighted_mean": bool(getattr(
                pooling, "pooling_mode_weightedmean_tokens", False)),
            "lasttoken": bool(getattr(pooling, "pooling_mode_lasttoken", False)),
        }
    prompts = dict(getattr(model, "prompts", {}) or {})
    proof = {
        "dimensions": dimensions,
        "transformer_architecture": type(model[0].auto_model).__name__,
        "transformer_hidden_size": int(getattr(transformer_config, "hidden_size", -1)),
        "pooling": "lasttoken" if pooling_flags["lasttoken"] else "other",
        "pooling_flags": pooling_flags,
        "pooling_include_prompt": bool(getattr(pooling, "include_prompt", False)),
        "normalized": type(normalize).__name__ == "Normalize",
        "normalize_module": type(normalize).__name__,
        "prompts": prompts,
        "default_prompt_name": getattr(model, "default_prompt_name", None),
    }
    expected_flags = {
        "cls": False, "mean": False, "max": False, "mean_sqrt": False,
        "weighted_mean": False, "lasttoken": True,
    }
    if (proof["dimensions"] != ROUND0005_DIMENSIONS or
            proof["transformer_hidden_size"] != ROUND0005_DIMENSIONS or
            proof["pooling"] != ROUND0005_POOLING or
            proof["pooling_flags"] != expected_flags or
            proof["pooling_include_prompt"] is not True or
            proof["normalized"] is not True or
            proof["prompts"] != {"query": "Query: ", "document": "Document: "} or
            proof["default_prompt_name"] is not None):
        raise ValueError(f"loaded Jina model violates the exact runtime contract: {proof!r}")
    return proof


def embed_texts(model, texts, batch_size: int = 256, show_progress: bool = False,
                *, return_telemetry: bool = False):
    """Encode a list of (already-prompted, if desired) strings -> (n, d)
    float32 normalized embeddings. Retries with a halved batch size on CUDA
    OOM so a transient VRAM pressure spike doesn't kill a multi-hour run."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        from basemap.run_controller import require_active_round0005_child_admission
        require_active_round0005_child_admission()
    import torch
    bs = batch_size
    oom_retries = 0
    while True:
        try:
            result = np.asarray(
                model.encode(texts, batch_size=bs, convert_to_numpy=True,
                             show_progress_bar=show_progress),
                dtype=np.float32,
            )
            telemetry = {"requested_batch_size": int(batch_size),
                         "final_batch_size": int(bs),
                         "oom_retries": int(oom_retries)}
            return (result, telemetry) if return_telemetry else result
        except torch.cuda.OutOfMemoryError:
            if bs <= 8:
                raise
            oom_retries += 1
            bs = max(8, bs // 2)
            torch.cuda.empty_cache()
            print(f"[embed_prompted_200k] CUDA OOM, retrying batch_size={bs}", flush=True)


def embed_outer_chunks(model, *, sample_indices, out_train: str, receipt_dir: str,
                       text_dir: str, text_shards, offsets, model_commit: str,
                       compute_dtype: str, batch_size: int = 256,
                       chunk_rows: int = OUTER_CHUNK_ROWS,
                       prompt_prefix: str = PROMPT_PREFIX,
                       fetch_fn=fetch_texts_for_indices,
                       embed_fn=embed_texts) -> dict:
    """Embed atomic outer chunks; an OOM can retry only the current chunk.

    A completed chunk is skipped only after its exact source IDs/text, prompt,
    model convention, and full output signature all re-verify.  Thus a restart
    never re-encodes a previously certified chunk and never trusts a filename.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        from basemap.run_controller import require_active_round0005_child_admission
        require_active_round0005_child_admission()
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    for path, label in ((out_train, "embedding chunk output root"),
                        (receipt_dir, "embedding chunk receipt root")):
        if os.path.exists(path):
            if not os.path.isdir(path) or os.path.islink(path):
                raise RuntimeError(f"{label} is not a regular directory: {path}")
        else:
            create_fresh_directory(path, label=label)
    rows = np.asarray(sample_indices)
    if rows.ndim != 1 or rows.dtype != np.dtype("int64"):
        raise ValueError("embedding chunk sample IDs must be an exact one-dimensional int64 array")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("embedding chunk sample IDs contain duplicates")
    chunk_count = int(np.ceil(len(rows) / chunk_rows)) if len(rows) else 0
    allowed_outputs = {f"data-{index:05d}.npy" for index in range(chunk_count)}
    allowed_receipts = {f"chunk-{index:05d}.json" for index in range(chunk_count)}
    for path, allowed, label in (
            (out_train, allowed_outputs, "embedding chunk output root"),
            (receipt_dir, allowed_receipts, "embedding chunk receipt root")):
        extras = sorted(set(os.listdir(path)) - allowed)
        if extras:
            raise FileExistsError(f"refuse unrecognized files in {label}: {extras[:5]}")
    def fetch_context(chunk_index, start, stop):
        chunk_ids = rows[start:stop]
        fetch_started = time.monotonic()
        texts = fetch_fn(chunk_ids, text_dir, text_shards, offsets)
        fetch_wall = time.monotonic() - fetch_started
        if len(texts) != len(chunk_ids) or not all(isinstance(text, str) for text in texts):
            raise RuntimeError(f"chunk {chunk_index} source fetch returned invalid text rows")
        context_started = time.monotonic()
        context = _chunk_context(
            chunk_index=chunk_index, start=start, stop=stop, row_ids=chunk_ids,
            texts=texts, prompt_prefix=prompt_prefix, model_id=MODEL_ID,
            model_commit=model_commit, pooling="lasttoken", compute_dtype=compute_dtype,
            output_dtype="float32", normalization="l2",
            requested_batch_size=batch_size)
        context_wall = time.monotonic() - context_started
        return chunk_ids, texts, context, fetch_wall, context_wall

    # Preflight every already-present chunk before producing any new output.
    # Thus a corrupt future receipt can never be discovered only after an
    # earlier missing chunk was re-embedded.
    states = []
    for chunk_index, start in enumerate(range(0, len(rows), chunk_rows)):
        stop = min(len(rows), start + chunk_rows)
        output_path = os.path.join(out_train, f"data-{chunk_index:05d}.npy")
        receipt_path = os.path.join(receipt_dir, f"chunk-{chunk_index:05d}.json")
        output_exists = os.path.lexists(output_path)
        receipt_exists = os.path.lexists(receipt_path)
        if output_exists != receipt_exists:
            raise RuntimeError(
                f"partial existing embedding chunk refuses resume: receipt={receipt_path} "
                f"output={output_path}")
        if output_exists:
            _, _, context, _, _ = fetch_context(chunk_index, start, stop)
            complete = _load_completed_chunk(receipt_path, output_path, context)
            states.append({"kind": "complete", "receipt": complete,
                           "chunk_index": chunk_index})
        else:
            states.append({"kind": "new", "chunk_index": chunk_index,
                           "start": start, "stop": stop,
                           "output_path": output_path, "receipt_path": receipt_path})

    receipts = []
    for state in states:
        chunk_index = state["chunk_index"]
        if state["kind"] == "complete":
            receipts.append({**state["receipt"], "resumed": True})
            print(f"[embed_prompted_200k] chunk {chunk_index:05d} verified; resume skip",
                  flush=True)
            continue
        chunk_started = time.monotonic()
        start, stop = state["start"], state["stop"]
        chunk_ids, texts, context, fetch_wall, context_wall = fetch_context(
            chunk_index, start, stop)
        output_path, receipt_path = state["output_path"], state["receipt_path"]
        prompted = apply_prompt(texts, prompt_prefix)
        embed_started = time.monotonic()
        embedded, telemetry = embed_fn(
            model, prompted, batch_size=batch_size, show_progress=False,
            return_telemetry=True)
        embed_wall = time.monotonic() - embed_started
        embedded = np.asarray(embedded, dtype=np.float32)
        if embedded.shape[0] != len(chunk_ids) or not np.isfinite(embedded).all():
            raise RuntimeError(f"chunk {chunk_index} produced invalid embedding rows")
        norms = np.linalg.norm(embedded.astype(np.float64), axis=1)
        if len(norms) and float(np.max(np.abs(norms - 1.0))) > 1e-3:
            raise RuntimeError(f"chunk {chunk_index} claims L2 normalization but is not normalized")
        write_started = time.monotonic()
        _atomic_npy(output_path, embedded)
        output_signature = expected_input_signature(output_path)
        write_wall = time.monotonic() - write_started
        receipt = {
            "status": "complete",
            "context": context,
            "output_signature": output_signature,
            "output_shape": [int(value) for value in embedded.shape],
            "output_ordered_sha256": ordered_array_sha256(embedded),
            "embedding": telemetry,
            "phase_wall_s": {
                "source_fetch": round(fetch_wall, 6),
                "prompt_and_context_hash": round(context_wall, 6),
                "model_encode_including_tokenization": round(embed_wall, 6),
                "validate_and_output_publish": round(write_wall, 6),
            },
            "wall_s_through_output_publish": round(time.monotonic() - chunk_started, 6),
            # Persist the best available timing before the no-overwrite receipt
            # itself is published. The fresh-run return value below replaces
            # this with the true total including receipt publication.
            "wall_s": round(time.monotonic() - chunk_started, 6),
            "resumed": False,
        }
        receipt_started = time.monotonic()
        _atomic_json(receipt_path, receipt)
        receipt_wall = time.monotonic() - receipt_started
        runtime_receipt = {
            **receipt,
            "phase_wall_s": {
                **receipt["phase_wall_s"],
                "receipt_publish": round(receipt_wall, 6),
            },
            "wall_s": round(time.monotonic() - chunk_started, 6),
        }
        receipts.append(runtime_receipt)
    return {
        "schema": "jina_atomic_embedding_chunks.v1",
        "chunk_rows": int(chunk_rows),
        "n_rows": int(len(rows)),
        "chunks": receipts,
        "completed_chunks": len(receipts),
        "preflight_complete_before_new_work": True,
        "preflight_existing_chunks": sum(state["kind"] == "complete" for state in states),
        "new_chunks": sum(not item.get("resumed", False) for item in receipts),
        "resumed_chunks": sum(bool(item.get("resumed", False)) for item in receipts),
        "oom_retries": sum(int((item.get("embedding") or {}).get("oom_retries", 0))
                           for item in receipts),
    }


def measure_faithfulness(model, *, sample_indices, ref_X, text_dir, text_shards,
                          offsets, n_probe: int = 8, seed: int = 0,
                          min_mean_cosine: float = 0.98,
                          min_row_cosine: float = 0.95) -> dict:
    """Embed ``n_probe`` of the 200k rows WITHOUT the prompt through the local
    pipeline and compare to the stored (Modal/vLLM) reference embeddings for
    those exact rows. Returns a dict with per-row/mean/min cosine and a
    ``passed`` flag; does NOT raise (the caller decides whether to abort)."""
    rng = np.random.RandomState(seed)
    n = len(sample_indices)
    positions = np.sort(rng.choice(n, size=min(n_probe, n), replace=False))
    global_ids = np.asarray(sample_indices)[positions]
    texts = fetch_texts_for_indices(global_ids, text_dir, text_shards, offsets)
    got_ids = global_ids  # fetch_texts_for_indices preserves input order by construction
    emb = embed_texts(model, texts, batch_size=max(8, len(texts)))
    ref = np.asarray(ref_X[positions], dtype=np.float32)
    a = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    b = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
    cos = (a * b).sum(axis=1)
    out = {
        "n_probe": int(len(cos)), "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()), "std_cosine": float(cos.std()),
        "per_row_cosine": [float(c) for c in cos],
        "probe_positions": [int(p) for p in positions],
        "probe_global_ids": [int(g) for g in got_ids],
        "threshold": min_mean_cosine, "min_row_threshold": min_row_cosine,
        # fable finding: gate the MIN row cosine too — one catastrophically wrong
        # row (e.g. a single misaligned shard) can hide behind a passing mean.
        "passed": bool(cos.mean() >= min_mean_cosine and cos.min() >= min_row_cosine),
    }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("experiments/embed_prompted_200k.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="/data/latent-basemap/jina-en-200k",
                     help="unprompted testbed dir (source of sample_indices.npy + reference X)")
    ap.add_argument("--out", default="/data/latent-basemap/jina-en-200k-prompted")
    ap.add_argument("--text-dir", default="/data/chunks/fineweb-edu-sample-10BT-chunked-500/train")
    ap.add_argument("--embed-dir",
                     default="/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--outer-chunk-rows", type=int, default=OUTER_CHUNK_ROWS)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--n-faithfulness-probe", type=int, default=40)  # fable: ≥33 for 11-shard coverage
    ap.add_argument("--canary-rows", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None,
                     help="debug: only embed the first N rows (smoke tests)")
    args = ap.parse_args()
    if args.batch_size != 256:
        raise ValueError("production Jina embedding requires initial batch size 256")
    if args.outer_chunk_rows != OUTER_CHUNK_ROWS:
        raise ValueError("production Jina embedding requires atomic 25,000-row outer chunks")
    if os.path.lexists(args.out):
        if not os.path.isdir(args.out) or os.path.islink(args.out):
            raise FileExistsError(f"embedding output root is not resumable: {args.out}")
        extras = sorted(set(os.listdir(args.out)) - {
            "train", "chunk-receipts", "sample_indices.npy"})
        if extras:
            raise FileExistsError(
                f"refuse nonempty embedding output root with unrecognized entries: {extras[:5]}")
        completed_manifests = [name for name in os.listdir(os.path.join(args.out, "train"))
                               if name.endswith(".manifest.json")] \
            if os.path.isdir(os.path.join(args.out, "train")) else []
        if completed_manifests:
            raise FileExistsError(
                f"refuse existing completed embedding output root: {args.out}")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()

    t_start = time.time()
    sample_indices = np.load(os.path.join(args.testbed, "sample_indices.npy"))
    if sample_indices.ndim != 1 or sample_indices.dtype != np.dtype("int64"):
        raise ValueError("production sample_indices must be exact one-dimensional int64")
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("production sample_indices contain duplicates")
    if args.limit:
        sample_indices = sample_indices[: args.limit]
    print(f"[embed_prompted_200k] {len(sample_indices):,} target rows "
          f"(range {sample_indices.min()}..{sample_indices.max()})", flush=True)

    text_shards, embed_sizes, offsets, ref_dim = verify_shard_alignment(args.embed_dir, args.text_dir)
    print(f"[embed_prompted_200k] verified {len(text_shards)} text/embedding shard pairs, "
          f"total {offsets[-1]:,} rows, dim={ref_dim}", flush=True)
    text_shard_sha = {s: sha_file(os.path.join(args.text_dir, s)) for s in text_shards}

    # resolve ALL 200k rows now, up front, and hard-gate on identity before any
    # model load / GPU work (row-count / order / index drift guard).
    shard_ids, local = locate_shard(sample_indices, offsets)
    resolved_check = offsets[shard_ids] + local
    assert_row_identity(resolved_check, sample_indices, context="pre-embed row resolution")
    print("[embed_prompted_200k] row resolution identity check PASSED "
          "(resolved ids == sample_indices, in order)", flush=True)

    ref_X = np.load(os.path.join(args.testbed, "train", "data-00000.npy"), mmap_mode="r")
    if len(ref_X) != len(sample_indices) and args.limit is None:
        raise ValueError(f"reference X has {len(ref_X)} rows but sample_indices has "
                          f"{len(sample_indices)} — testbed is internally inconsistent.")

    model, commit = load_model(device=device, dtype=args.dtype)
    print(f"[embed_prompted_200k] loaded {MODEL_ID} commit={commit} device={device} "
          f"dtype={args.dtype}", flush=True)

    faithfulness = measure_faithfulness(
        model, sample_indices=sample_indices, ref_X=ref_X, text_dir=args.text_dir,
        text_shards=text_shards, offsets=offsets, n_probe=args.n_faithfulness_probe,
    )
    print(f"[embed_prompted_200k] FAITHFULNESS (unprompted local pipeline vs. stored "
          f"reference, n={faithfulness['n_probe']}): mean_cosine={faithfulness['mean_cosine']:.4f} "
          f"min_cosine={faithfulness['min_cosine']:.4f} (threshold {faithfulness['threshold']})",
          flush=True)
    if not faithfulness["passed"]:
        msg = (f"[embed_prompted_200k] FAITHFULNESS CHECK FAILED: mean cosine "
               f"{faithfulness['mean_cosine']:.4f} < {faithfulness['threshold']} — the local "
               f"embedding pipeline does NOT faithfully reproduce the reference unprompted "
               f"embeddings; a prompted run on top of it would NOT be validly comparable to "
               f"the unprompted testbed. Refusing to proceed.")
        print(msg, flush=True)
        if os.environ.get("BASEMAP_UNSAFE_ALLOW_LOW_FAITHFULNESS") != "1":
            raise RuntimeError(msg)
        print("[embed_prompted_200k] BASEMAP_UNSAFE_ALLOW_LOW_FAITHFULNESS=1 set — "
              "proceeding anyway (UNSAFE).", flush=True)

    n = len(sample_indices)
    out_train = os.path.join(args.out, "train")
    receipt_dir = os.path.join(args.out, "chunk-receipts")
    chunk_report = embed_outer_chunks(
        model, sample_indices=sample_indices, out_train=out_train,
        receipt_dir=receipt_dir, text_dir=args.text_dir, text_shards=text_shards,
        offsets=offsets, model_commit=commit, compute_dtype=args.dtype,
        batch_size=args.batch_size, chunk_rows=args.outer_chunk_rows)
    assert_row_identity(resolved_check, sample_indices,
                        context="post-chunk embedding drift check")

    out_shard_paths = sorted(
        os.path.join(out_train, name) for name in os.listdir(out_train)
        if name.endswith(".npy"))
    norm_parts = []
    finite = True
    total_rows = 0
    for path in out_shard_paths:
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2 or values.shape[1] != ref_dim:
            raise RuntimeError(f"embedding chunk has wrong shape: {path} {values.shape}")
        finite = finite and bool(np.isfinite(values).all())
        norm_parts.append(np.linalg.norm(np.asarray(values, dtype=np.float32), axis=1))
        total_rows += len(values)
    if total_rows != n:
        raise RuntimeError(f"atomic embedding chunks contain {total_rows} rows, expected {n}")
    norms = np.concatenate(norm_parts)
    norm_check = {
        "mean": float(norms.mean()), "std": float(norms.std()),
        "min": float(norms.min()), "max": float(norms.max()),
        "is_unit_norm": bool(abs(norms.mean() - 1.0) < 1e-3 and norms.std() < 1e-2),
    }
    print(f"[embed_prompted_200k] norm check: mean={norm_check['mean']:.6f} "
          f"std={norm_check['std']:.6f} unit_norm={norm_check['is_unit_norm']}", flush=True)
    if not norm_check["is_unit_norm"]:
        print("[embed_prompted_200k] WARNING: output vectors are not tightly unit-norm "
              "— check dtype/pooling config.", flush=True)
    if not finite:
        raise RuntimeError("[embed_prompted_200k] non-finite values in output embeddings — abort.")
    output_sample_ids = os.path.join(args.out, "sample_indices.npy")
    if os.path.lexists(output_sample_ids):
        if os.path.islink(output_sample_ids) or not os.path.isfile(output_sample_ids):
            raise RuntimeError("existing prompted sample_indices output is not a regular file")
        existing_sample_ids = np.load(output_sample_ids, mmap_mode="r")
        if (existing_sample_ids.dtype != np.dtype("int64") or
                existing_sample_ids.ndim != 1 or
                not np.array_equal(existing_sample_ids, sample_indices)):
            raise RuntimeError(
                "existing prompted sample_indices output is corrupt/mismatched and will not "
                "be overwritten")
    else:
        _atomic_npy(output_sample_ids, sample_indices)

    first = chunk_report["chunks"][0]
    first_rows = first["context"]["source_row_count"]
    first_wall = float(first.get("wall_s", 0.0))
    est_total_s = first_wall * len(chunk_report["chunks"])
    canary = {
        "canary_rows": int(first_rows), "canary_wall_s": round(first_wall, 2),
        "rows_per_s": round(first_rows / max(first_wall, 1e-6), 2),
        "estimated_total_wall_s": round(est_total_s, 1),
        "actual_total_wall_s": round(time.time() - t_start, 1),
        "oom_retries": chunk_report["oom_retries"],
    }
    manifest = build_manifest(
        model_id=MODEL_ID, model_commit=commit, prompt_prefix=PROMPT_PREFIX,
        n_rows=n, dim=ref_dim, dtype="float32", row_ids=sample_indices,
        text_shards=text_shards, text_shard_sha=text_shard_sha,
        out_shards=[os.path.basename(path) for path in out_shard_paths], norm_check=norm_check,
        faithfulness=faithfulness, canary=canary,
        extra={
            "source_testbed": os.path.abspath(args.testbed),
            "source_text_dir": os.path.abspath(args.text_dir),
            "source_embed_dir": os.path.abspath(args.embed_dir),
            "compute_dtype": args.dtype, "device": device, "batch_size": args.batch_size,
            "pooling": "lasttoken", "normalize": True,
            "sentence_transformers_prompt_equivalent": "document",
            "outer_chunk_rows": args.outer_chunk_rows,
            "atomic_chunk_report": chunk_report,
            "chunk_receipts": [expected_input_signature(
                os.path.join(receipt_dir, f"chunk-{index:05d}.json"))
                for index in range(len(chunk_report["chunks"]))],
        },
    )
    validate_manifest_shape(manifest)
    man_path = out_shard_paths[0] + ".manifest.json"
    _atomic_json(man_path, manifest)
    print(f"[embed_prompted_200k] wrote {len(out_shard_paths)} atomic chunks "
          f"({n}, {ref_dim}) float32 + manifest {man_path}", flush=True)


if __name__ == "__main__":
    main()

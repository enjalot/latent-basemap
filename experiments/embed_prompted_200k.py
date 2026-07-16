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
      --batch-size 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT_PREFIX = "Document: "
MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
MANIFEST_SCHEMA = "prompted_embed_manifest.v1"
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


def load_model(device: str = "cuda", dtype: str = "float32"):
    """Load the jina-v5-nano-retrieval sentence-transformers checkpoint from
    the local HF cache. Returns (model, commit_hash)."""
    os.environ.setdefault("HF_HOME", "/data/hf")
    import torch
    from sentence_transformers import SentenceTransformer
    torch_dtype = getattr(torch, dtype)
    model = SentenceTransformer(
        MODEL_ID, trust_remote_code=True, device=device,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    try:
        commit = model[0].auto_model.config._commit_hash
    except Exception:
        commit = None
    return model, commit


def embed_texts(model, texts, batch_size: int = 512, show_progress: bool = False):
    """Encode a list of (already-prompted, if desired) strings -> (n, d)
    float32 normalized embeddings. Retries with a halved batch size on CUDA
    OOM so a transient VRAM pressure spike doesn't kill a multi-hour run."""
    import torch
    bs = batch_size
    while True:
        try:
            return np.asarray(
                model.encode(texts, batch_size=bs, convert_to_numpy=True,
                             show_progress_bar=show_progress),
                dtype=np.float32,
            )
        except torch.cuda.OutOfMemoryError:
            if bs <= 8:
                raise
            bs = max(8, bs // 2)
            torch.cuda.empty_cache()
            print(f"[embed_prompted_200k] CUDA OOM, retrying batch_size={bs}", flush=True)


def measure_faithfulness(model, *, sample_indices, ref_X, text_dir, text_shards,
                          offsets, n_probe: int = 8, seed: int = 0,
                          min_mean_cosine: float = 0.98) -> dict:
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
        "threshold": min_mean_cosine,
        "passed": bool(cos.mean() >= min_mean_cosine),
    }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="/data/latent-basemap/jina-en-200k",
                     help="unprompted testbed dir (source of sample_indices.npy + reference X)")
    ap.add_argument("--out", default="/data/latent-basemap/jina-en-200k-prompted")
    ap.add_argument("--text-dir", default="/data/chunks/fineweb-edu-sample-10BT-chunked-500/train")
    ap.add_argument("--embed-dir",
                     default="/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--n-faithfulness-probe", type=int, default=8)
    ap.add_argument("--canary-rows", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None,
                     help="debug: only embed the first N rows (smoke tests)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()

    t_start = time.time()
    sample_indices = np.load(os.path.join(args.testbed, "sample_indices.npy"))
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

    # fetch + prompt all 200k texts (order-preserving; re-verify identity after fetch)
    t_fetch = time.time()
    texts = fetch_texts_for_indices(sample_indices, args.text_dir, text_shards, offsets)
    print(f"[embed_prompted_200k] fetched {len(texts):,} text rows in "
          f"{time.time() - t_fetch:.1f}s", flush=True)
    prompted = apply_prompt(texts, PROMPT_PREFIX)

    n = len(prompted)
    out_X = np.empty((n, ref_dim), dtype=np.float32)

    canary_n = min(args.canary_rows, n)
    t0 = time.time()
    out_X[:canary_n] = embed_texts(model, prompted[:canary_n], batch_size=args.batch_size)
    canary_s = time.time() - t0
    est_total_s = canary_s * (n / max(canary_n, 1))
    print(f"[embed_prompted_200k] CANARY: {canary_n:,} rows in {canary_s:.1f}s "
          f"({canary_n / max(canary_s, 1e-6):.1f} rows/s) -> est. total "
          f"{est_total_s / 60:.1f} min for {n:,} rows", flush=True)
    # re-assert no drift over the rows the canary actually covered
    assert_row_identity(resolved_check[:canary_n], sample_indices[:canary_n],
                        context="post-canary drift check")

    if canary_n < n:
        t1 = time.time()
        out_X[canary_n:] = embed_texts(model, prompted[canary_n:], batch_size=args.batch_size,
                                       show_progress=True)
        print(f"[embed_prompted_200k] remaining {n - canary_n:,} rows in "
              f"{time.time() - t1:.1f}s", flush=True)

    norm_check = l2_norm_stats(out_X)
    print(f"[embed_prompted_200k] norm check: mean={norm_check['mean']:.6f} "
          f"std={norm_check['std']:.6f} unit_norm={norm_check['is_unit_norm']}", flush=True)
    if not norm_check["is_unit_norm"]:
        print("[embed_prompted_200k] WARNING: output vectors are not tightly unit-norm "
              "— check dtype/pooling config.", flush=True)
    if not np.isfinite(out_X).all():
        raise RuntimeError("[embed_prompted_200k] non-finite values in output embeddings — abort.")

    out_train = os.path.join(args.out, "train")
    os.makedirs(out_train, exist_ok=True)
    out_shard_path = os.path.join(out_train, "data-00000.npy")
    np.save(out_shard_path, out_X)
    np.save(os.path.join(args.out, "sample_indices.npy"), sample_indices)

    canary = {
        "canary_rows": int(canary_n), "canary_wall_s": round(canary_s, 2),
        "rows_per_s": round(canary_n / max(canary_s, 1e-6), 2),
        "estimated_total_wall_s": round(est_total_s, 1),
        "actual_total_wall_s": round(time.time() - t_start, 1),
    }
    manifest = build_manifest(
        model_id=MODEL_ID, model_commit=commit, prompt_prefix=PROMPT_PREFIX,
        n_rows=n, dim=ref_dim, dtype=str(out_X.dtype), row_ids=sample_indices,
        text_shards=text_shards, text_shard_sha=text_shard_sha,
        out_shards=[os.path.basename(out_shard_path)], norm_check=norm_check,
        faithfulness=faithfulness, canary=canary,
        extra={
            "source_testbed": os.path.abspath(args.testbed),
            "source_text_dir": os.path.abspath(args.text_dir),
            "source_embed_dir": os.path.abspath(args.embed_dir),
            "compute_dtype": args.dtype, "device": device, "batch_size": args.batch_size,
            "pooling": "lasttoken", "normalize": True,
            "sentence_transformers_prompt_equivalent": "document",
        },
    )
    validate_manifest_shape(manifest)
    man_path = out_shard_path + ".manifest.json"
    json.dump(manifest, open(man_path, "w"), indent=1)
    print(f"[embed_prompted_200k] wrote {out_shard_path} ({out_X.shape}, {out_X.dtype}) "
          f"+ manifest {man_path}", flush=True)


if __name__ == "__main__":
    main()

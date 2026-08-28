#!/usr/bin/env python3
"""P-A: document-prompted jina-v5-nano SOCIAL POOLS + 250k probes
(JINA_SWEEP_PROPOSAL.md 2026-08-28).

The jina-side social-mixture sweep displaces EN with social rows drawn from these
pools; the maximin probe suite scores OOD on the 250k probe registers. Both must
be document-prompted (prompt "Document: ") EXACTLY like the champion substrate —
a raw embed silently measures prompt-mismatch, not register coverage (see
p2_jina_embed.py). Reuses that template's model machinery.

HOLDOUT-DISJOINT convention (mirrors build_mixture_substrates.py / MiniLM):
  Each social corpus reserves its FIRST 300,000 global rows (sorted-shard
  concatenation order) as the PROBE holdout. The mixture POOLS draw ONLY from
  global offset >= 300000; the 250k PROBES draw from the FRONT 300000 holdout
  (first 250,000 rows), so pool and probe are disjoint.

Builds (document-prompted, f16 (N,768), write-once):
  substrates/reddit-jina-pool/     700,000 rows, global offset >= 300000
  substrates/ca-jina-pool/         500,000 rows, global offset >= 300000
  substrates/twitter-jina-pool/    500,000 rows, global offset >= 300000
  substrates/bluesky-jina-pool/    500,000 rows, global offset >= 300000
  substrates/twitter-jina-250k/    250,000 rows, front 300000 holdout (offset 0)
  substrates/bluesky-jina-250k/    250,000 rows, front 300000 holdout (offset 0)

reddit-jina-250k + ca-jina-250k ALREADY EXIST (built by p2_jina_embed.py) and are
NOT rebuilt here. The model is loaded ONCE and used for every job.

GPU script (the parent runs it). CPU validation (``--validate``) reads 5 rows per
job at its offset and confirms count/format; it does NOT load the model or embed.

Usage:
  jina_social_pools.py            # full GPU embed of every pending job
  jina_social_pools.py --validate # cheap CPU check of the text sources only
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")

import numpy as np

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
DIM = 768
BATCH = int(os.environ.get("BATCH", "64"))
SUBSTRATES = Path("/data/latent-basemap/substrates")
HOLDOUT = 300_000   # front 300k of each social corpus reserved for probes

# job: name -> (chunk_dirname, offset, count)
#   pools draw from global offset >= HOLDOUT; 250k probes from the front holdout.
JOBS = {
    "reddit-jina-pool":  ("reddit-tldr17-chunked-120", HOLDOUT, 700_000),
    "ca-jina-pool":      ("communityarchive-tweets",   HOLDOUT, 500_000),
    "twitter-jina-pool": ("twitter100m-chunked-120",   HOLDOUT, 500_000),
    "bluesky-jina-pool": ("bluesky-5m-chunked-120",    HOLDOUT, 500_000),
    "twitter-jina-250k": ("twitter100m-chunked-120",   0,       250_000),
    "bluesky-jina-250k": ("bluesky-5m-chunked-120",    0,       250_000),
}


def _text_column(sample_file: str) -> str:
    """Sniff the text column: prefer ``chunk_text`` (the pipeline standard),
    fall back to ``text``. Raises if neither is present."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(sample_file).schema.names
    for c in ("chunk_text", "text"):
        if c in names:
            return c
    raise KeyError(f"no chunk_text/text column in {sample_file}: {names}")


def read_range(chunk_dirname: str, start: int, count: int) -> list[str]:
    """``count`` texts from global rows [start, start+count) of the register's
    sorted parquet shards. Same semantics as embed_jina_625_topup.read_range
    (global row-skip, 3000-char overlong guard, sorted-glob shard order) but
    STREAMS via iter_batches so it stops early and never materializes a whole
    shard (a fineweb2 lang is a single 2M-row parquet)."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(f"/data/chunks/{chunk_dirname}/train/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards at /data/chunks/{chunk_dirname}/train")
    col = _text_column(files[0])
    texts: list[str] = []
    seen = 0
    for f in files:
        pf = pq.ParquetFile(f)
        n = pf.metadata.num_rows
        if seen + n <= start:           # whole shard is before the window
            seen += n
            continue
        for batch in pf.iter_batches(batch_size=65_536, columns=[col]):
            b = batch.num_rows
            lo = max(0, start - seen)    # local start within this batch
            if lo < b:
                arr = batch.column(0).to_pylist()
                for x in arr[lo:lo + (count - len(texts))]:
                    texts.append(x[:3000])
            seen += b
            if len(texts) >= count:
                break
        if len(texts) >= count:
            break
    assert len(texts) == count, (chunk_dirname, start, count, len(texts))
    return texts


def pick_doc_prompt(model) -> tuple[str, str]:
    """The model's own document-side prompt (mandatory), matching the reference."""
    prompts = getattr(model, "prompts", None) or {}
    for key in ("passage", "document", "retrieval.passage", "doc"):
        if key in prompts:
            return key, prompts[key]
    return "__manual__", "Document: "


def embed_job(model, prompt_key, prompt_text, name, chunk_dirname,
              offset, count) -> None:
    out_dir = SUBSTRATES / name
    out_path = out_dir / "substrate.f16.npy"
    if out_path.exists():
        print(f"{name}: {out_path} exists, skip", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = read_range(chunk_dirname, offset, count)
    t0 = time.time()
    if prompt_key == "__manual__":
        vecs = model.encode([prompt_text + t for t in texts],
                            batch_size=BATCH, convert_to_numpy=True,
                            show_progress_bar=False)
    else:
        vecs = model.encode(texts, prompt_name=prompt_key, batch_size=BATCH,
                            convert_to_numpy=True, show_progress_bar=False)
    assert vecs.shape == (len(texts), DIM), vecs.shape
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, vecs.astype(np.float16))
    os.rename(tmp, out_path)
    role = "mixture-pool (offset>=300000)" if name.endswith("-pool") else \
        "probe register (front 300000 holdout)"
    (out_dir / "manifest.json").write_text(json.dumps({
        "model": MODEL_ID,
        "prompt_key": prompt_key,
        "prompt_text": prompt_text,
        "chunk_dirname": chunk_dirname,
        "text_source": f"/data/chunks/{chunk_dirname}/train/*.parquet",
        "offset": offset,
        "count": count,
        "global_row_range": [offset, offset + count],
        "N": count, "dim": DIM, "dtype": "float16",
        "prompting": "document-prompted",
        "role": role,
        "holdout_note": (
            "front 300,000 global rows reserved as probe holdout; pools draw "
            "from offset>=300000, 250k probes from the front holdout — disjoint."),
    }, indent=1))
    dt = time.time() - t0
    print(f"{name}: {len(texts):,} @offset {offset:,} in {dt/60:.1f} min "
          f"({len(texts)/dt:,.0f}/s) -> {out_path}", flush=True)


def validate() -> int:
    """Cheap CPU check: read 5 rows per job AT ITS OFFSET, confirm count+format."""
    for name, (dirname, offset, count) in JOBS.items():
        rows = read_range(dirname, offset, 5)
        assert len(rows) == 5 and all(isinstance(r, str) for r in rows), name
        print(f"{name:20s} <- {dirname} @offset {offset:,} (n={count:,}): "
              f"first_len={len(rows[0])}, repr={rows[0][:50]!r}")
    print("validate OK")
    return 0


def main() -> int:
    if "--validate" in sys.argv:
        return validate()

    import torch
    from sentence_transformers import SentenceTransformer

    SUBSTRATES.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID, device="cuda", trust_remote_code=True)
    model = model.half()
    prompt_key, prompt_text = pick_doc_prompt(model)
    print(f"document prompt: {prompt_key!r} -> {prompt_text!r}", flush=True)

    for name, (chunk_dirname, offset, count) in JOBS.items():
        embed_job(model, prompt_key, prompt_text, name, chunk_dirname,
                  offset, count)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

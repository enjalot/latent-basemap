#!/usr/bin/env python3
"""P2, step 1: document-prompted jina-v5-nano substrates for two social-media
register corpora (reddit TL;DR, community-archive tweets).

These become the OOD probe inputs for the jina-space parametric-UMAP champion
(``jina-multi-2m/champion-bs16k``).  The champion's training substrate was
DOCUMENT-PROMPTED (owner order 2026-08-22, see
``latent-data-modal/embed_jina_prompted_subsets.py``): "Document: " shifts jina
cosine to 0.73-0.94, so a raw embed here would silently measure prompt-mismatch
instead of register coverage.  We therefore apply the model's own document-side
prompt EXACTLY as the reference pipeline does.

For each register (name -> chunk_dirname):
  reddit-jina-250k -> reddit-tldr17-chunked-120
  ca-jina-250k     -> communityarchive-tweets

reads N=250,000 texts from column ``chunk_text``, applies the document prompt,
jina-embeds on the 5090, and saves float16 (N,768) to
  /data/latent-basemap/substrates/<name>/substrate.f16.npy
plus a manifest.json (model, prompt provenance, chunk_dirname, N).

Write-once: a register whose substrate.f16.npy already exists is skipped.  The
model is loaded ONCE and used for both registers.

GPU script.  CPU validation (``--validate``) only reads 5 rows per corpus and
confirms count/format; it does NOT load the model or embed.

Usage:
  p2_jina_embed.py            # full GPU embed of both registers
  p2_jina_embed.py --validate # cheap CPU check of read_texts only
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
N = int(os.environ.get("N", "250000"))
SUBSTRATES = Path("/data/latent-basemap/substrates")

# register name -> chunk dirname under /data/chunks/<dirname>/train/*.parquet
REGISTERS = {
    "reddit-jina-250k": "reddit-tldr17-chunked-120",
    "ca-jina-250k": "communityarchive-tweets",
}


def read_texts(chunk_dirname: str, n: int) -> list[str]:
    """First ``n`` chunk_text rows from the register's sorted parquet shards.

    Mirrors ``embed_jina_prompted_subsets.read_texts`` exactly (column
    ``chunk_text``, 3000-char overlong guard, sorted-glob shard order)."""
    import pyarrow.parquet as pq

    texts: list[str] = []
    for f in sorted(glob.glob(f"/data/chunks/{chunk_dirname}/train/*.parquet")):
        t = pq.read_table(f, columns=["chunk_text"])["chunk_text"].to_pylist()
        texts.extend(t[:n - len(texts)])
        if len(texts) >= n:
            break
    assert len(texts) == n, (chunk_dirname, len(texts), n)
    return [t[:3000] for t in texts]  # same overlong guard as the raw pipeline


def pick_doc_prompt(model) -> tuple[str, str]:
    """The model's own document-side prompt (mandatory), matching the reference."""
    prompts = getattr(model, "prompts", None) or {}
    for key in ("passage", "document", "retrieval.passage", "doc"):
        if key in prompts:
            return key, prompts[key]
    return "__manual__", "Document: "


def embed_register(model, prompt_key, prompt_text, name, chunk_dirname) -> None:
    out_dir = SUBSTRATES / name
    out_path = out_dir / "substrate.f16.npy"
    if out_path.exists():
        print(f"{name}: {out_path} exists, skip", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = read_texts(chunk_dirname, N)
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
    (out_dir / "manifest.json").write_text(json.dumps({
        "model": MODEL_ID,
        "prompt_key": prompt_key,
        "prompt_text": prompt_text,
        "chunk_dirname": chunk_dirname,
        "N": N,
        "dim": DIM,
        "dtype": "float16",
        "prompting": "document-prompted",
        "note": "document-side prompt applied (matches the jina champion's "
                "substrate); a raw embed would measure prompt-mismatch, not "
                "register coverage.",
    }, indent=1))
    dt = time.time() - t0
    print(f"{name}: {len(texts):,} in {dt/60:.1f} min ({len(texts)/dt:,.0f}/s) "
          f"-> {out_path}", flush=True)


def validate() -> int:
    """Cheap CPU check: read 5 rows per register, confirm count + format."""
    for name, dirname in REGISTERS.items():
        rows = read_texts(dirname, 5)
        assert len(rows) == 5, (name, len(rows))
        assert all(isinstance(r, str) for r in rows), name
        print(f"{name} <- {dirname}: {len(rows)} rows, "
              f"first_len={len(rows[0])}, repr={rows[0][:60]!r}")
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

    for name, chunk_dirname in REGISTERS.items():
        embed_register(model, prompt_key, prompt_text, name, chunk_dirname)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

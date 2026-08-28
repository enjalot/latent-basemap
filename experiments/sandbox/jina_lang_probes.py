#!/usr/bin/env python3
"""P-B: document-prompted jina-v5-nano per-LANGUAGE probe registers
(JINA_SWEEP_PROPOSAL.md 2026-08-28). The standing blocker for the jina maximin —
we have never had per-language OOD truths for the jina maps.

For each of the 20 fineweb2 languages (image_map_pipeline._JINA_LANGS), embed
100,000 chunks DISJOINT from every jina substrate's language span. The jina base
substrates consumed fineweb2 rows through:
  jina-prompted multi-1m.f16   : first 50,000 / lang
  jina-prompted 6.25M topup    : through 156,250 / lang (manifest-6250k.json)
so the holdout draws from GLOBAL row >= 156,250 per language (disjoint from both).

Source text: /data/chunks/fineweb2-<lang>-chunked-500/train/*.parquet (column
chunk_text; each lang is a single 2,000,000-row shard, so offset 156,250 + 100,000
= 256,250 << 2,000,000). Document-prompted (prompt "Document: ") EXACTLY like the
champion — a raw embed would measure prompt-mismatch, not language coverage.

Output (f16 (100000,768), write-once, FULL lang code in the name):
  substrates/probe-lang-<lang>-jina/substrate.f16.npy   e.g. probe-lang-arb_Arab-jina
  + manifest.json (offset>=156250 disjointness proof).

The model is loaded ONCE for all 20 languages. GPU script (the parent runs it).
CPU validation (``--validate``) reads 5 rows/lang at the holdout offset and confirms
count/format; it does NOT load the model or embed.

Usage:
  jina_lang_probes.py            # full GPU embed of the 20 language probes
  jina_lang_probes.py --validate # cheap CPU check of the text sources only
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
N_PER_LANG = int(os.environ.get("N_PER_LANG", "100000"))
LANG_HOLDOUT = 156_250   # jina substrates used through 156,250/lang; draw beyond
SUBSTRATES = Path("/data/latent-basemap/substrates")

# the 20 languages in image_map_pipeline._JINA_LANGS order (FULL codes).
LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
         "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
         "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
         "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")


def _text_column(sample_file: str) -> str:
    import pyarrow.parquet as pq
    names = pq.ParquetFile(sample_file).schema.names
    for c in ("chunk_text", "text"):
        if c in names:
            return c
    raise KeyError(f"no chunk_text/text column in {sample_file}: {names}")


def read_range(chunk_dirname: str, start: int, count: int) -> list[str]:
    """``count`` texts from global rows [start, start+count). Same semantics as
    embed_jina_625_topup.read_range (global row-skip, 3000-char guard) but
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
    prompts = getattr(model, "prompts", None) or {}
    for key in ("passage", "document", "retrieval.passage", "doc"):
        if key in prompts:
            return key, prompts[key]
    return "__manual__", "Document: "


def embed_lang(model, prompt_key, prompt_text, lang) -> None:
    name = f"probe-lang-{lang}-jina"
    chunk_dirname = f"fineweb2-{lang}-chunked-500"
    out_dir = SUBSTRATES / name
    out_path = out_dir / "substrate.f16.npy"
    if out_path.exists():
        print(f"{name}: {out_path} exists, skip", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = read_range(chunk_dirname, LANG_HOLDOUT, N_PER_LANG)
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
        "lang": lang,
        "chunk_dirname": chunk_dirname,
        "text_source": f"/data/chunks/{chunk_dirname}/train/*.parquet",
        "offset": LANG_HOLDOUT,
        "count": N_PER_LANG,
        "global_row_range": [LANG_HOLDOUT, LANG_HOLDOUT + N_PER_LANG],
        "N": N_PER_LANG, "dim": DIM, "dtype": "float16",
        "prompting": "document-prompted",
        "role": "per-language OOD probe register (maximin suite)",
        "disjointness_proof": (
            "jina base substrates used fineweb2-%s rows through 156,250/lang "
            "(multi-1m first 50,000; 6.25M topup through 156,250, see "
            "jina-prompted/manifest-6250k.json). This probe draws global rows "
            "[156250, 256250) — disjoint from every jina substrate span." % lang),
    }, indent=1))
    dt = time.time() - t0
    print(f"{name}: {len(texts):,} @offset {LANG_HOLDOUT:,} in {dt/60:.1f} min "
          f"({len(texts)/dt:,.0f}/s) -> {out_path}", flush=True)


def validate() -> int:
    """Cheap CPU check: read 5 rows/lang at the holdout offset, confirm format."""
    for lang in LANGS:
        rows = read_range(f"fineweb2-{lang}-chunked-500", LANG_HOLDOUT, 5)
        assert len(rows) == 5 and all(isinstance(r, str) for r in rows), lang
        print(f"probe-lang-{lang}-jina <- fineweb2-{lang}-chunked-500 "
              f"@offset {LANG_HOLDOUT:,}: first_len={len(rows[0])}, "
              f"repr={rows[0][:40]!r}")
    print(f"validate OK ({len(LANGS)} langs, {N_PER_LANG:,} rows each)")
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

    for lang in LANGS:
        embed_lang(model, prompt_key, prompt_text, lang)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

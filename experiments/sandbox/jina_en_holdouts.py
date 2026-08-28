#!/usr/bin/env python3
"""P-C: document-prompted jina-v5-nano EN base-register holdout probes
(JINA_SWEEP_PROPOSAL.md 2026-08-28). The in-distribution EN registers — the
language floor's EN anchor in the maximin suite.

For each EN base corpus (fineweb-edu / RedPajama / pile), embed 150,000 chunks
DISJOINT from every jina EN substrate span. The jina EN substrates consumed the
sorted chunked-500 parquets from row 0:
  jina-prompted en-2m.f16   : first 666,667 / 666,667 / 666,666 (_JINA_EN_PER)
  jina-prompted 6.25M topup : first 1,041,667 / corpus (manifest-6250k.json)
so the holdout draws from GLOBAL row >= 1,041,667 per corpus — disjoint from BOTH
the 2M champion (666,667) AND the 6.25M map (1,041,667). (The proposal requires
disjoint-from-champion, i.e. >= 666,667; 1,041,667 is the strictly-safer offset
that also clears the 6.25M map, at no cost — all three corpora have >= 22M rows.)

Source text (column chunk_text, same parquets the champion embed used, via
embed_jina_prompted_subsets.EN):
  fineweb -> fineweb-edu-sample-10BT-chunked-500   (25,959,540 rows)
  rpj     -> RedPajama-Data-V2-sample-10B-chunked-500 (22,437,805 rows)
  pile    -> pile-uncopyrighted-chunked-500        (68,002,835 rows)

Document-prompted (prompt "Document: ") EXACTLY like the champion. Output
(f16 (150000,768), write-once):
  substrates/probe-fineweb-jina/substrate.f16.npy
  substrates/probe-rpj-jina/substrate.f16.npy
  substrates/probe-pile-jina/substrate.f16.npy
  + manifest.json (disjoint-from-champion proof).

Model loaded ONCE. GPU script (the parent runs it). CPU validation (``--validate``)
reads 5 rows/corpus at the holdout offset; it does NOT load the model or embed.

Usage:
  jina_en_holdouts.py            # full GPU embed of the 3 EN holdout probes
  jina_en_holdouts.py --validate # cheap CPU check of the text sources only
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
N_EN = int(os.environ.get("N_EN", "150000"))  # proposal range 100-250k/corpus
# jina EN substrates used through 1,041,667/corpus (2M @666,667 + 6.25M @1,041,667);
# draw beyond the larger span so the probe is disjoint from BOTH.
EN_HOLDOUT = 1_041_667
SUBSTRATES = Path("/data/latent-basemap/substrates")

# probe name -> (chunk_dirname, champion_2m_span, champion_625_span)
EN = {
    "probe-fineweb-jina": ("fineweb-edu-sample-10BT-chunked-500", 666_667, 1_041_667),
    "probe-rpj-jina":     ("RedPajama-Data-V2-sample-10B-chunked-500", 666_667, 1_041_667),
    "probe-pile-jina":    ("pile-uncopyrighted-chunked-500", 666_666, 1_041_666),
}


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
    multi-GB shard."""
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


def embed_en(model, prompt_key, prompt_text, name, chunk_dirname,
             span_2m, span_625) -> None:
    out_dir = SUBSTRATES / name
    out_path = out_dir / "substrate.f16.npy"
    if out_path.exists():
        print(f"{name}: {out_path} exists, skip", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = read_range(chunk_dirname, EN_HOLDOUT, N_EN)
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
        "text_source": f"/data/chunks/{chunk_dirname}/train/*.parquet",
        "offset": EN_HOLDOUT,
        "count": N_EN,
        "global_row_range": [EN_HOLDOUT, EN_HOLDOUT + N_EN],
        "N": N_EN, "dim": DIM, "dtype": "float16",
        "prompting": "document-prompted",
        "role": "EN base-register OOD holdout probe (maximin suite; EN anchor)",
        "disjointness_proof": {
            "champion_2m_span": span_2m,
            "champion_6250k_span": span_625,
            "holdout_offset": EN_HOLDOUT,
            "note": ("jina EN substrates used first %d (2M champion) and %d "
                     "(6.25M topup) rows of this corpus; probe draws [%d, %d) — "
                     "disjoint from both." % (span_2m, span_625, EN_HOLDOUT,
                                              EN_HOLDOUT + N_EN)),
        },
    }, indent=1))
    dt = time.time() - t0
    print(f"{name}: {len(texts):,} @offset {EN_HOLDOUT:,} in {dt/60:.1f} min "
          f"({len(texts)/dt:,.0f}/s) -> {out_path}", flush=True)


def validate() -> int:
    """Cheap CPU check: read 5 rows/corpus at the holdout offset, confirm format."""
    for name, (dirname, span_2m, span_625) in EN.items():
        rows = read_range(dirname, EN_HOLDOUT, 5)
        assert len(rows) == 5 and all(isinstance(r, str) for r in rows), name
        print(f"{name:20s} <- {dirname} @offset {EN_HOLDOUT:,} "
              f"(2m_span={span_2m:,}, 625_span={span_625:,}): "
              f"first_len={len(rows[0])}, repr={rows[0][:40]!r}")
    print(f"validate OK ({len(EN)} EN corpora, {N_EN:,} rows each)")
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

    for name, (chunk_dirname, span_2m, span_625) in EN.items():
        embed_en(model, prompt_key, prompt_text, name, chunk_dirname,
                 span_2m, span_625)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

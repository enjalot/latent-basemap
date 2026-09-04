"""Backlog jina embedder (owner never-rest-GPU mandate, overseer 2026-09-04). Embeds the next UNDONE shard(s)
of the deferred jina production corpus with jina-v5-nano DOCUMENT-prompt (byte-pinned path) -> per-shard f16
.npy at /data/embeddings/<corpus>-jina-v5-nano/train/. Per-shard resumable (skips existing .npy). Processes at
most MAX_SHARDS per invocation (default 2) then exits, so the drain releases the GPU flock promptly (preempt-
friendly). Corpus order: fineweb-edu-chunked-500 EN (99), then fineweb2 language pools. Loads model once.
Usage: jina_backlog_embed.py  (env JINA_BACKLOG_MAX_SHARDS)."""
import os, glob, json, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"; BATCH = 512
E = "/data/embeddings"; CH = "/data/chunks"
CORPORA = [("fineweb-edu-sample-10BT-chunked-500", "en")] + \
          [(d.split("/")[-1], "ml") for d in sorted(glob.glob(f"{CH}/fineweb2-*-chunked-500"))]


def _undone():
    """Yield (corpus, parquet_path, out_npy) for the next undone shards, in corpus/shard order."""
    for corpus, _ in CORPORA:
        outdir = Path(f"{E}/{corpus}-jina-v5-nano/train"); outdir.mkdir(parents=True, exist_ok=True)
        for p in sorted(glob.glob(f"{CH}/{corpus}/train/*.parquet")):
            out = outdir / (Path(p).stem + ".npy")
            if not out.exists():
                yield corpus, p, out


def main():
    maxn = int(os.environ.get("JINA_BACKLOG_MAX_SHARDS", "2"))
    jobs = []
    it = _undone()
    for _ in range(maxn):
        try:
            jobs.append(next(it))
        except StopIteration:
            break
    if not jobs:
        print("BACKLOG jina: all shards done"); return 0
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, device="cuda", trust_remote_code=True)
    prompts = getattr(model, "prompts", {}) or {}
    pkey = next((k for k in ("passage", "document") if k in prompts), None)
    for corpus, p, out in jobs:
        t = pq.read_table(p, columns=["chunk_text"]).to_pandas()["chunk_text"].tolist()
        texts = [str(x) for x in t]; t0 = time.time()
        if pkey:
            v = model.encode(texts, prompt_name=pkey, batch_size=BATCH, convert_to_numpy=True, show_progress_bar=False)
        else:
            v = model.encode(["Document: " + x for x in texts], batch_size=BATCH, convert_to_numpy=True, show_progress_bar=False)
        tmp = out.with_suffix(".tmp.npy"); np.save(tmp, v.astype(np.float16)); os.rename(tmp, out)
        print(f"BACKLOG jina: {corpus}/{out.name} {v.shape} {len(texts)/(time.time()-t0):.0f} ch/s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

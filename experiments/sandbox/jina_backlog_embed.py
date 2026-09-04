"""Backlog jina embedder (owner never-rest-GPU mandate, overseer 2026-09-04; token-budget batching 2026-09-04).
Embeds the next UNDONE shard(s) of the deferred jina production corpus with jina-v5-nano DOCUMENT-prompt ->
per-shard f16 .npy at /data/embeddings/<corpus>-jina-v5-nano/train/. Per-shard resumable (skips existing .npy).
Processes at most MAX_SHARDS per invocation (default 2) then exits, so the drain releases the GPU flock promptly.
Corpus order: fineweb-edu-chunked-500 EN (99), then fineweb2 language pools. Loads model once.

OOM FIX (overseer 2026-09-04): a FIXED batch_size=512 put jina-v5-nano at the 32GB knife-edge — the longest
batch (512 rows x 627 tokens = 321k token-slots) transiently hit ~31GB and any shard could tip over 32GB on
fragmentation (diagnosed: fineweb-edu #20 OOMed while #0, with a LONGER max token, embedded fine — pure knife-
edge, not data). Now we batch by a TOKEN BUDGET: sort by length, pack each batch so n_rows*max_len <= budget, so
the longest-sequence batches carry proportionally FEWER rows and the VRAM peak is flat (~<=22GB). Plus
expandable_segments to tame the fragmentation that made a 4.28GiB alloc fail with ~4GB free.
Usage: jina_backlog_embed.py  (env JINA_BACKLOG_MAX_SHARDS, JINA_TOKEN_BUDGET)."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # tame fragmentation (set before torch)
import glob, json, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
TOKEN_BUDGET = int(os.environ.get("JINA_TOKEN_BUDGET", "220000"))   # n_rows*max_len per batch; 512x627=321k OOMed
MAX_ROWS = 512; MIN_ROWS = 8
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


def _token_budget_batches(lens, budget=TOKEN_BUDGET):
    """Yield lists of original indices; each batch sorted-by-length has n_rows*max_len <= budget (and <= MAX_ROWS).
    Longer sequences => fewer rows => flat VRAM peak. `lens` is the per-row token count."""
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    batch = []; bmax = 0
    for i in order:
        nmax = bmax if bmax > lens[i] else lens[i]
        if batch and ((len(batch) + 1) * nmax > budget or len(batch) >= MAX_ROWS):
            yield batch; batch = []; bmax = 0; nmax = lens[i]
        batch.append(i); bmax = nmax
    if batch:
        yield batch


def main():
    maxn = int(os.environ.get("JINA_BACKLOG_MAX_SHARDS", "2"))
    jobs = []; it = _undone()
    for _ in range(maxn):
        try:
            jobs.append(next(it))
        except StopIteration:
            break
    if not jobs:
        print("BACKLOG jina: all shards done"); return 0
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, device="cuda", trust_remote_code=True)
    tokenizer = model.tokenizer
    prompts = getattr(model, "prompts", {}) or {}
    pkey = next((k for k in ("passage", "document") if k in prompts), None)
    for corpus, p, out in jobs:
        texts = [str(x) for x in pq.read_table(p, columns=["chunk_text"]).to_pandas()["chunk_text"].tolist()]
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        lens = [len(tokenizer(t, truncation=True, max_length=8192)["input_ids"]) for t in texts]
        embs = [None] * len(texts); nb = 0
        for bidx in _token_budget_batches(lens):
            bt = [texts[i] for i in bidx]
            if pkey:
                v = model.encode(bt, prompt_name=pkey, batch_size=len(bt), convert_to_numpy=True, show_progress_bar=False)
            else:
                v = model.encode(["Document: " + x for x in bt], batch_size=len(bt), convert_to_numpy=True, show_progress_bar=False)
            for j, i in enumerate(bidx):
                embs[i] = v[j]
            nb += 1
        V = np.stack(embs).astype(np.float16)
        tmp = out.with_suffix(".tmp.npy"); np.save(tmp, V); os.rename(tmp, out)
        # log both: allocated (live tensors) and reserved (allocator pool) — the OOM cited "27.89GB in use"
        # ~= reserved+alloc, so reserved is the number comparable to the 32GB knife-edge.
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9; peak_resv = torch.cuda.max_memory_reserved() / 1e9
        print(f"BACKLOG jina: {corpus}/{out.name} {V.shape} in {nb} tok-budget batches "
              f"{len(texts)/(time.time()-t0):.0f} ch/s peak_alloc={peak_alloc:.1f}GB peak_resv={peak_resv:.1f}GB -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

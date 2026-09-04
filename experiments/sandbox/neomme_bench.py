"""NeoMME-260M 10K throughput benchmark (owner NeoMME probe, correct NeoMMEForRetrieval path). Embeds a 10K
fineweb-edu chunked-120 slice via NeoMMEProcessor + NeoMMEForRetrieval -> dense_embeddings (mean-pooled,
1024-d Matryoshka) on the 5090; records dim + chunks/s. Document-prompted (retrieval 'document' task)."""
import time, glob, json, sys
import numpy as np, pyarrow.parquet as pq, torch
from transformers import NeoMMEForRetrieval, NeoMMEProcessor

MID = "Hcompany/NeoMME-260M-Retriever"; N = 10_000; B = 128
texts = []
for f in sorted(glob.glob("/data/chunks/fineweb-edu-sample-10BT-chunked-120/train/*.parquet")):
    texts += [str(x) for x in pq.read_table(f, columns=["chunk_text"]).to_pandas()["chunk_text"].tolist()]
    if len(texts) >= N: break
texts = texts[:N]
out = {"model": MID, "n": N}
try:
    proc = NeoMMEProcessor.from_pretrained(MID)
    model = NeoMMEForRetrieval.from_pretrained(MID, dtype=torch.float16).cuda().eval()
    def _encode(batch):
        # try the documented 'document' task prompt; fall back to plain text if the kwarg differs
        for kw in ({"text": batch, "task": "document"}, {"text": batch, "prompt_name": "document"}, {"text": batch}):
            try:
                inp = proc(**kw, return_tensors="pt", padding=True, truncation=True, max_length=512)
                return {k: v.cuda() for k, v in inp.items()}, kw
            except TypeError:
                continue
        raise RuntimeError("no working processor signature")
    t0 = time.time(); vs = []; used = None
    with torch.no_grad():
        for s in range(0, N, B):
            inp, used = _encode(texts[s:s+B])
            o = model(**inp)
            emb = getattr(o, "dense_embeddings", None)
            if emb is None:
                emb = o.get("dense_embeddings") if isinstance(o, dict) else o[0]
            emb = torch.nn.functional.normalize(emb, dim=1)
            vs.append(emb.float().cpu().numpy())
    v = np.concatenate(vs); dt = time.time() - t0
    out.update(ok=True, dim=int(v.shape[1]), chunks_per_s=round(N/dt, 1), wall_s=round(dt, 1),
               processor_sig=str(used), note="document-prompted dense_embeddings, fp16")
except Exception as e:
    import traceback; out.update(ok=False, error=str(e)[:300], trace=traceback.format_exc()[-500:])
open("/data/latent-basemap/sandbox/neomme-bench.json", "w").write(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "trace"}, indent=1))

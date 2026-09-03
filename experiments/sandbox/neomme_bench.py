"""NeoMME-260M throughput benchmark (owner NeoMME probe, de-risk). Loads the model, embeds a 10K fineweb-edu
chunked-120 slice -> POOLED DENSE (mean-pooled normalized), records dim + chunks/s on the 5090. Flags if the
model class isn't available or throughput is pathological (before committing the 2M embed)."""
import time, glob, json, sys
import numpy as np, pyarrow.parquet as pq, torch

MID = "Hcompany/NeoMME-260M-Retriever"
N = 10_000
texts = []
for f in sorted(glob.glob("/data/chunks/fineweb-edu-sample-10BT-chunked-120/train/*.parquet")):
    t = pq.read_table(f, columns=["chunk_text"]).to_pandas()["chunk_text"].tolist()
    texts += [str(x) for x in t]
    if len(texts) >= N: break
texts = texts[:N]
out = {"model": MID, "n": N}
try:
    # try sentence-transformers ST-dense variant first (simplest pooled path)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MID, device="cuda", trust_remote_code=True)
    t0 = time.time()
    v = m.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    dt = time.time() - t0
    out.update(path="sentence-transformers", dim=int(v.shape[1]), chunks_per_s=round(N/dt, 1), wall_s=round(dt, 1))
except Exception as e:
    out["st_error"] = str(e)[:200]
    try:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(MID, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
        t0 = time.time(); vs = []
        with torch.no_grad():
            for s in range(0, N, 256):
                b = tok(texts[s:s+256], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                o = mdl(**b); h = o.last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                pooled = (h*mask).sum(1)/mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                vs.append(pooled.float().cpu().numpy())
        v = np.concatenate(vs); dt = time.time()-t0
        out.update(path="transformers-AutoModel-meanpool", dim=int(v.shape[1]), chunks_per_s=round(N/dt,1), wall_s=round(dt,1))
    except Exception as e2:
        out["transformers_error"] = str(e2)[:200]
open("/data/latent-basemap/sandbox/neomme-bench.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

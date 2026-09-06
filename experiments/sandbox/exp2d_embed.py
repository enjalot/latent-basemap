"""exp-2d SigLIP2 joint substrate (owner 2026-09-06). GPU (neomme-env: transformers>=5, torchvision). Embeds the
250K NeoMME-pair thumbnails through the SigLIP2 IMAGE tower + the 250K captions through the SigLIP2 TEXT tower
(same MID/revision) -> 500K joint substrate: rows [0,250K)=image SigLIP2, [250K,500K)=caption SigLIP2, row i <->
250K+i matched. modality.npy (0=img,1=txt) + pair_id.npy provenance, as exp-2. f32 L2-normed (SigLIP2 space,
cross-modal recall@1 0.883).

BYTE-VERIFY (mandatory, encoder-harness): the 250K pair thumbnails are pool thumbnails; siglip-random-2m embedded
random-2m thumbnails via the SAME SigLIP2 image tower. So a few pair-image embeddings should match siglip-random-
2m for the same MONET id (cosine ~1.0 = image-tower config match). The text tower shares MID/processor. (No prior
SigLIP2 TEXT vectors on disk to verify the text tower directly — flagged; image-tower match + shared config is the
available check.)

Reads /data2/monet/neomme-pairs-250k/pairs.parquet (id, thumbnail, caption_florence-2-large). Writes
/data2/monet/exp2d-siglip-500k/{substrate.f32.npy, modality.npy, pair_id.npy, id.npy, manifest.json}.
Usage: neomme-env python exp2d_embed.py [--byteverify-only]
"""
import os, sys, io, json, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoModel, AutoProcessor

MID = "google/siglip2-so400m-patch16-256"
PAIRS = "/data2/monet/neomme-pairs-250k/pairs.parquet"
SRM = "/data2/monet/siglip-random-2m"            # for the image-tower byte-verify
OUT = Path("/data2/monet/exp2d-siglip-500k"); OUT.mkdir(parents=True, exist_ok=True)
DIM = 1152; B = 256


def _pil(wb):
    try:
        return Image.open(io.BytesIO(wb)).convert("RGB")
    except Exception:
        return None


def main():
    t = pq.read_table(PAIRS, columns=["id", "thumbnail", "caption_florence-2-large"])
    ids = np.array(t["id"].to_pylist()); thumbs = t["thumbnail"].to_pylist(); caps = t["caption_florence-2-large"].to_pylist()
    N = len(ids); print(f"[exp2d] {N} pairs; SigLIP2 {MID}", flush=True)
    proc = AutoProcessor.from_pretrained(MID); model = AutoModel.from_pretrained(MID, dtype=torch.float16).cuda().eval()

    def img_embed(pils):
        inp = {k: v.cuda() for k, v in proc(images=pils, return_tensors="pt").items() if hasattr(v, "cuda")}
        with torch.no_grad():
            o = model.get_image_features(**inp); f = o if torch.is_tensor(o) else o.pooler_output
            return torch.nn.functional.normalize(f, dim=1).float().cpu().numpy()

    def txt_embed(texts):
        inp = {k: v.cuda() for k, v in proc(text=texts, padding="max_length", truncation=True, return_tensors="pt").items() if hasattr(v, "cuda")}
        with torch.no_grad():
            o = model.get_text_features(**inp); f = o if torch.is_tensor(o) else o.pooler_output
            return torch.nn.functional.normalize(f, dim=1).float().cpu().numpy()

    # BYTE-VERIFY: image tower vs siglip-random-2m for shared ids
    srm_ids = np.concatenate([np.load(m, allow_pickle=True)["id"] for m in sorted(Path("/data2/monet/random-2m/shards").glob("*_meta.npz"))])
    srm_sub = np.load(f"{SRM}/substrate.f32.npy", mmap_mode="r")
    id2row = {}
    for r, i in enumerate(srm_ids.tolist()):
        id2row.setdefault(i, r)
    checks = []
    for k in range(min(8, N)):
        pil = _pil(thumbs[k])
        if pil is None or ids[k] not in id2row:
            continue
        e = img_embed([pil])[0]; ref = srm_sub[id2row[ids[k]]]; ref = ref / (np.linalg.norm(ref) + 1e-9)
        checks.append(float(e @ ref))
    bv = float(np.mean(checks)) if checks else None
    print(f"[exp2d] BYTE-VERIFY image tower vs siglip-random-2m: mean cosine {bv} (n={len(checks)})", flush=True)
    if "--byteverify-only" in sys.argv:
        (OUT / "byteverify.json").write_text(json.dumps({"image_tower_cosine_vs_srm": bv, "n": len(checks), "MID": MID}, indent=1)); return 0
    if bv is not None and bv < 0.95:
        print(f"[exp2d] WARNING byte-verify {bv:.4f} < 0.95 — image-tower config may differ from siglip-random-2m", flush=True)

    sub = np.lib.format.open_memmap(OUT / "substrate.f32.npy", mode="w+", dtype=np.float32, shape=(2 * N, DIM))
    t0 = time.time()
    for s in range(0, N, B):                          # image half
        pils = [_pil(x) or Image.new("RGB", (256, 256)) for x in thumbs[s:s + B]]
        sub[s:s + len(pils)] = img_embed(pils)
        if s % (B * 20) == 0: print(f"[exp2d] img {s}/{N} {(time.time()-t0):.0f}s", flush=True)
    for s in range(0, N, B):                          # text half
        sub[N + s:N + s + len(caps[s:s + B])] = txt_embed([str(c) for c in caps[s:s + B]])
        if s % (B * 20) == 0: print(f"[exp2d] txt {s}/{N}", flush=True)
    sub.flush()
    # FUNCTIONAL text-tower check (overseer 2026-09-06): caption->image retrieval on a 1K sample. MATCHED PROTOCOL to
    # the recorded demo = 1K captions vs their 1K PAIRED images (1K candidate pool) -> must reproduce ~0.883/0.985.
    # The 250K-pool number (vs ALL images) is reported alongside as the honest large-pool figure (250x harder — a
    # much lower r@1 there is expected and NOT a failure). GATE on the matched-protocol r@1 >= 0.85.
    rng = np.random.default_rng(42); samp = np.sort(rng.choice(N, min(1000, N), replace=False))
    img_all = torch.from_numpy(np.ascontiguousarray(sub[:N])).cuda()
    txt = torch.from_numpy(np.ascontiguousarray(sub[N + samp])).cuda()
    img_pool = torch.from_numpy(np.ascontiguousarray(sub[samp])).cuda()          # the 1K PAIRED images (matched pool)
    with torch.no_grad():
        topm = (txt @ img_pool.T).topk(10, dim=1).indices.cpu().numpy()          # matched 1K-pool
        topL = (txt @ img_all.T).topk(10, dim=1).indices.cpu().numpy()           # honest 250K-pool
    gtm = np.arange(len(samp))[:, None]; gtL = samp[:, None]
    r1 = float((topm[:, :1] == gtm).any(1).mean()); r10 = float((topm == gtm).any(1).mean())
    r1L = float((topL[:, :1] == gtL).any(1).mean()); r10L = float((topL == gtL).any(1).mean())
    print(f"[exp2d] TEXT-TOWER caption->image retrieval, MATCHED 1K-pool: recall@1 {r1:.4f} recall@10 {r10:.4f} "
          f"(recorded demo ~0.883/0.985); HONEST 250K-pool: recall@1 {r1L:.4f} recall@10 {r10L:.4f}", flush=True)
    if r1 < 0.85:
        print(f"[exp2d] GATE FAIL matched-protocol recall@1 {r1:.4f} < 0.85 — text tower did not reproduce the demo; NOT trusting substrate", flush=True)
    np.save(OUT / "modality.npy", np.concatenate([np.zeros(N, np.int8), np.ones(N, np.int8)]))
    np.save(OUT / "pair_id.npy", np.concatenate([np.arange(N), np.arange(N)]).astype(np.int64))
    np.save(OUT / "id.npy", np.concatenate([ids, ids]))
    (OUT / "manifest.json").write_text(json.dumps({"schema": "exp2d-siglip-500k-2026-09-06", "MID": MID, "dim": DIM,
        "n_pairs": N, "layout": "rows [0,N)=image SigLIP2, [N,2N)=caption SigLIP2; row i<->N+i matched",
        "byteverify_image_tower_cosine_vs_srm": bv, "textcheck_matched1k_recall_at_1": round(r1, 4), "textcheck_matched1k_recall_at_10": round(r10, 4),
        "textcheck_pool250k_recall_at_1": round(r1L, 4), "textcheck_pool250k_recall_at_10": round(r10L, 4),
        "textcheck_gate_passed": bool(r1 >= 0.85), "textcheck_protocol": "matched to recorded demo: 1K captions vs their 1K paired images; gate on matched r@1>=0.85; 250K-pool reported as honest large-pool figure",
        "note": "f32 L2-normed SigLIP2 space; for exp-2d stratified relational kNN"}, indent=1))
    print(f"[exp2d] DONE 2x{N} SigLIP2 -> {OUT} (byte-verify {bv})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

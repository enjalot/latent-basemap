"""SigLIP2 text-queryability demo (owner SigLIP2 probe, overseer 2026-09-04) — the capability DINOv2 can't offer.
Embed 1K MONET captions through SigLIP2's TEXT tower + their paired thumbnails through the IMAGE tower into the
SAME space; report matched-pair cosine vs random-pair cosine + recall (is the true image in the caption's top-k
over the 1K set?). A sanity demo that SigLIP2 gives a shared text-image space over these images. CPU-light GPU.
Usage: siglip_text_demo.py [N=1000]   (neomme-env python)"""
import os, sys, io, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoModel, AutoProcessor

MID = "google/siglip2-so400m-patch16-256"
PAIRS = "/data2/monet/neomme-pairs-250k/pairs.parquet"       # thumbnail(webp binary) + caption_florence-2-large
OUT = Path("/data2/monet/siglip-random-2m")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t = pq.read_table(PAIRS, columns=["thumbnail", "caption_florence-2-large"])
    thumbs = t["thumbnail"].to_pylist()[:N]; caps = [str(x) for x in t["caption_florence-2-large"].to_pylist()[:N]]
    pils = [Image.open(io.BytesIO(bytes(b))).convert("RGB") for b in thumbs]
    proc = AutoProcessor.from_pretrained(MID); model = AutoModel.from_pretrained(MID, dtype=torch.float16).cuda().eval()
    def img_emb(ps):
        inp = {k: v.cuda() for k, v in proc(images=ps, return_tensors="pt").items() if hasattr(v, "cuda")}
        with torch.no_grad():
            return torch.nn.functional.normalize(model.get_image_features(**inp), dim=1).float().cpu().numpy()
    def txt_emb(ts):
        inp = {k: v.cuda() for k, v in proc(text=ts, padding="max_length", truncation=True, return_tensors="pt").items() if hasattr(v, "cuda")}
        with torch.no_grad():
            return torch.nn.functional.normalize(model.get_text_features(**inp), dim=1).float().cpu().numpy()
    ie = np.concatenate([img_emb(pils[s:s+128]) for s in range(0, N, 128)])
    te = np.concatenate([txt_emb(caps[s:s+128]) for s in range(0, N, 128)])
    sims = te @ ie.T                                          # (N,N) caption->image cosine
    matched = float(np.diag(sims).mean()); rnd = float(sims[~np.eye(N, dtype=bool)].mean())
    rank = (sims > np.diag(sims)[:, None]).sum(1)             # rank of true image per caption (0=top)
    out = {"model": MID, "n": N, "matched_pair_cosine": round(matched, 4), "random_pair_cosine": round(rnd, 4),
           "recall@1": round(float((rank == 0).mean()), 4), "recall@10": round(float((rank < 10).mean()), 4),
           "median_rank_of_true_image": int(np.median(rank)),
           "note": "SigLIP2 shared text-image space over MONET images; caption retrieves its own thumbnail. "
                   "Text queryability is the capability DINOv2 lacks."}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "text_demo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""SigLIP2 image embed of the EXACT random-2m draw rows (owner SigLIP2 probe, overseer 2026-09-04). Decision-
grade vs DINOv2 0.826 / CLIP 0.649 on IDENTICAL rows. Reads 256px thumbnails straight from the packed pool store
(verified recipe: random-2m substrate row g <-> thumbnail shard i, local_row k; shard i == thumb shard i, same
seed-42 shuffle) -> SigLIP2 image-tower embeddings -> /data2/monet/siglip-random-2m/substrate.f32.npy, row-
aligned with random-2m/clip-substrate.f32.npy. Resumable.

--bench N : embed first N thumbnails, report throughput + dim + cone_stats(mean-norm) + assert-on-output
            (not NaN, not collapsed, L2-normed) BEFORE any full run (encoder-harness checklist steps 1-2).
Usage: siglip_embed.py [--bench 10000]   (neomme-env python: transformers>=5, torchvision)"""
import os, sys, io, json, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MID = "google/siglip2-so400m-patch16-256"          # 256px-native SigLIP2 (bench confirms dim + throughput)
RM = "/data2/monet/random-2m"; THB = "/data2/monet/pool-20m-thumbs256/shards"
OUT = Path("/data2/monet/siglip-random-2m"); OUT.mkdir(parents=True, exist_ok=True)
B = 256; BLACK = Image.new("RGB", (256, 256))       # placeholder for the rare failed-decode (0-length) row


def _shards():
    s = json.load(open(f"{RM}/manifest.json"))["shards"]; s.sort(key=lambda x: x["idx"]); return s


def _iter_webp(start_row=0):
    """Yield (global_row, webp_bytes) for every random-2m substrate row, in order, from start_row."""
    g = 0
    for s in _shards():
        i, r = s["idx"], s["rows"]
        if g + r <= start_row:                      # whole shard already done
            g += r; continue
        off = np.fromfile(f"{THB}/{i:04d}.offsets.u64", dtype=np.uint64)
        with open(f"{THB}/{i:04d}.blob", "rb") as f:
            for k in range(r):
                if g >= start_row:
                    a, b = int(off[k]), int(off[k + 1]); f.seek(a)
                    yield g, f.read(b - a)
                g += 1


def _pil(wb):
    if not wb:
        return BLACK
    try:
        return Image.open(io.BytesIO(wb)).convert("RGB")
    except Exception:
        return BLACK


def _load():
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID, dtype=torch.float16).cuda().eval()
    return proc, model


def _embed(proc, model, pils):
    inp = proc(images=pils, return_tensors="pt")
    inp = {k: v.cuda() for k, v in inp.items() if hasattr(v, "cuda")}
    with torch.no_grad():
        f = model.get_image_features(**inp)
        f = torch.nn.functional.normalize(f, dim=1)
    return f.detach().float().cpu().numpy()


def bench(n):
    proc, model = _load()
    pils, t0 = [], None
    for g, wb in _iter_webp():
        pils.append(_pil(wb))
        if len(pils) >= n:
            break
    t0 = time.time(); vs = []
    for s in range(0, len(pils), B):
        vs.append(_embed(proc, model, pils[s:s + B]))
    v = np.concatenate(vs); dt = time.time() - t0
    mean_norm = float(np.linalg.norm(v.mean(0)))            # cone_stats (v already L2-normed rows)
    out = {"model": MID, "n": len(v), "dim": int(v.shape[1]), "img_per_s": round(len(v) / dt, 1),
           "wall_s": round(dt, 1), "cone_mean_norm": round(mean_norm, 4),
           "assert_not_nan": bool(np.isfinite(v).all()), "assert_not_collapsed": bool(v.std(0).mean() > 1e-4),
           "row_norm_mean": round(float(np.linalg.norm(v, axis=1).mean()), 4),
           "cone_note": ">0.3 => center per-substrate + report raw/centered (checklist step 2)"}
    assert out["assert_not_nan"] and out["assert_not_collapsed"], f"SigLIP2 output failed sanity: {out}"
    OUT.joinpath("bench.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)); return 0


def main():
    if "--bench" in sys.argv:
        return bench(int(sys.argv[sys.argv.index("--bench") + 1]))
    N = sum(s["rows"] for s in _shards())                  # 2008321
    proc, model = _load()
    DIM = int(_embed(proc, model, [BLACK]).shape[1])
    subf = OUT / "substrate.f32.npy"; donef = OUT / "done.txt"
    sub = np.lib.format.open_memmap(subf, mode="r+" if subf.exists() else "w+", dtype=np.float32, shape=(N, DIM))
    done = int(donef.read_text()) if donef.exists() else 0
    print(f"SigLIP2 embed: N={N:,} dim={DIM} resume@{done:,}", flush=True)
    buf_r, buf_p = [], []; t0 = time.time()
    for g, wb in _iter_webp(start_row=done):
        buf_r.append(g); buf_p.append(_pil(wb))
        if len(buf_p) >= B:
            sub[buf_r[0]:buf_r[-1] + 1] = _embed(proc, model, buf_p)
            done = buf_r[-1] + 1; buf_r, buf_p = [], []
            if done % 100_096 < B:
                sub.flush(); donef.write_text(str(done))
                print(f"  {done:,}/{N:,} ({done/(time.time()-t0):.0f} img/s)", flush=True)
    if buf_p:
        sub[buf_r[0]:buf_r[-1] + 1] = _embed(proc, model, buf_p); done = buf_r[-1] + 1
    sub.flush(); donef.write_text(str(done))
    (OUT / "manifest.json").write_text(json.dumps({"model": MID, "dim": DIM, "n": int(done),
        "rows": "EXACTLY random-2m draw rows, row-aligned with random-2m/{clip,dino}-substrate.f32.npy",
        "source": "pool-20m-thumbs256 (256px webp)", "dtype": "float32-normed"}, indent=1))
    print(f"SigLIP2 substrate done: {done:,} rows -> {subf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

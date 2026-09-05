"""BL (British Library) + pool-thumb CLIP embed for the OOD projection test (owner item c, controlled design
approved 2026-09-05). GPU (tiny ViT-B/32) + CPU (decode-bound, parallel workers). Can interleave with the
champion (low VRAM).

Controlled design: BL thumbs AND a pool-thumb sample go through the BYTE-IDENTICAL pipeline (same model, same
preprocessing), so their reception DIFFERENCE isolates corpus while the common thumb/config offset (byte-verify
measured ~0.918 vs pool originals) CANCELS. Config = openai/clip-vit-base-patch32 (transformers, PIL backend),
vision_model.pooler_output -> visual_projection -> L2-norm (the byte-verified path; open_clip's torchvision dep
conflicts with the pinned cu128 torch, so transformers is used — pool-thumb control makes config-equivalence moot).

Outputs -> /data2/monet/bl-clip/:
  bl-clip.f32.npy        (N_bl, 512) f32 unit-norm, order = covers,medium,embellishments,plates; filename-sorted
  bl-segment.npy         (N_bl,) int8  0=covers 1=medium 2=embellishments 3=plates
  bl-id.npy              (N_bl,) filenames (stems)
  poolthumb-clip.f32.npy (N_ps,512) f32 unit-norm  — the reception-floor control
  poolthumb-pos.npy      (N_ps,) pool-20m positions (join to pool coords/heads)
  manifest.json          config + versions + pipeline note + BL-originals follow-up flag
Usage: bl_clip_embed.py [POOL_SAMPLE=200000] [BATCH=256] [WORKERS=12].
"""
import json, os, sys, time, io
from pathlib import Path
import numpy as np

BL_THUMBS = Path("/data/images/british-library-book-images/thumbs")
SEGMENTS = ["covers", "medium", "embellishments", "plates"]   # bl-siglip2-1m row_order
POOL = Path("/data2/monet/pool-20m")
OUT = Path("/data2/monet/bl-clip")
MODEL_ID = "openai/clip-vit-base-patch32"


def main():
    pool_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    sys.path.insert(0, "/home/enjalot/code/latent-scope-3d/pipeline/src")
    from lsvoxel.monet_thumbs import MonetThumbStore, pack_thumb_ref

    model = CLIPModel.from_pretrained(MODEL_ID).eval().to("cuda")
    proc = CLIPProcessor.from_pretrained(MODEL_ID)

    def preprocess(img):
        return proc(images=[img.convert("RGB")], return_tensors="pt")["pixel_values"][0]

    class BLSet(Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def __getitem__(self, i):
            try:
                return preprocess(Image.open(self.files[i])), 1
            except Exception:
                return torch.zeros(3, 224, 224), 0    # decode-fail -> zero px, validity 0

    class PoolThumbSet(Dataset):
        def __init__(self, positions):
            self.pos = positions; self.store = None
        def __len__(self):
            return len(self.pos)
        def __getitem__(self, i):
            if self.store is None:   # open per-worker (fds not fork-safe)
                self.psi = np.load(POOL / "prov_shard_idx.npy", mmap_mode="r")
                self.plr = np.load(POOL / "prov_local_row.npy", mmap_mode="r")
                self.store = MonetThumbStore(shards_dir=str(POOL) + "-thumbs256/shards")
            r = int(self.pos[i])
            b = self.store.read_packed(pack_thumb_ref(int(self.psi[r]), int(self.plr[r])))
            if not b:
                return torch.zeros(3, 224, 224), 0
            try:
                return preprocess(Image.open(io.BytesIO(b))), 1
            except Exception:
                return torch.zeros(3, 224, 224), 0

    @torch.no_grad()
    def embed(ds, tag):
        dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
        out = np.empty((len(ds), 512), np.float32); valid = np.empty(len(ds), bool)
        t0 = time.time(); done = 0
        for px, v in dl:
            e = model.visual_projection(model.vision_model(pixel_values=px.to("cuda")).pooler_output)
            e = torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float32)
            out[done:done + e.shape[0]] = e; valid[done:done + e.shape[0]] = v.numpy().astype(bool)
            done += e.shape[0]
            if done % (batch * 50) < batch:
                el = time.time() - t0; rate = done / el if el else 0
                print(f"[{tag}] {done:,}/{len(ds)}  {rate:.0f} img/s  eta~{(len(ds)-done)/rate/60:.0f}min", flush=True)
        return out, valid

    # BL: enumerate segments (filename-sorted), record segment labels
    bl_files = []; bl_seg = []; bl_id = []
    for si, seg in enumerate(SEGMENTS):
        fs = sorted((BL_THUMBS / seg).glob("*.webp"))
        bl_files += fs; bl_seg += [si] * len(fs); bl_id += [f.stem for f in fs]
        print(f"[bl] {seg}: {len(fs)} images", flush=True)
    print(f"[bl] total {len(bl_files)} images", flush=True)
    bl_emb, bl_valid = embed(BLSet(bl_files), "bl")
    np.save(OUT / "bl-clip.f32.npy", bl_emb); np.save(OUT / "bl-segment.npy", np.array(bl_seg, np.int8))
    np.save(OUT / "bl-id.npy", np.array(bl_id)); np.save(OUT / "bl-valid.npy", bl_valid)

    # pool-thumb control sample (byte-identical pipeline)
    rng = np.random.default_rng(42)
    ppos = np.sort(rng.choice(np.load(POOL / "prov_shard_idx.npy").shape[0], pool_sample, replace=False))
    ps_emb, ps_valid = embed(PoolThumbSet(ppos), "poolthumb")
    np.save(OUT / "poolthumb-clip.f32.npy", ps_emb); np.save(OUT / "poolthumb-pos.npy", ppos)
    np.save(OUT / "poolthumb-valid.npy", ps_valid)

    import transformers
    (OUT / "manifest.json").write_text(json.dumps({
        "schema": "bl-clip-embed-2026-09-05", "corpus": "british-library-book-images (thumbs)",
        "n_bl": len(bl_files), "n_bl_valid": int(bl_valid.sum()),
        "segments": {s: int((np.array(bl_seg) == i).sum()) for i, s in enumerate(SEGMENTS)},
        "pool_thumb_sample": int(pool_sample), "n_ps_valid": int(ps_valid.sum()),
        "config": {"model": MODEL_ID, "lib": f"transformers {transformers.__version__}", "backend": "PIL",
                   "path": "vision_model.pooler_output -> visual_projection -> L2-norm", "dim": 512},
        "control": "BL and pool-thumb via byte-identical pipeline; reception DIFFERENCE isolates corpus "
                   "(common ~0.918 thumb/config offset vs pool originals cancels).",
        "followup": "BL originals (biglam/british-library-book-images, ~1M-image download) only if the "
                    "thumb-based result warrants de-confounding; not done now (owner)."}, indent=1))
    print(f"[bl] DONE: BL {len(bl_files)} ({int(bl_valid.sum())} valid), pool-thumb {pool_sample} "
          f"({int(ps_valid.sum())} valid) -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

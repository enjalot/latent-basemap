"""Card 006 (DINO-1536) replay-bank builder — CPU, off-flock.
Builds three fixed sets with FROZEN teacher (ORIGINAL T0 champion) coords, all
using persistent POOL POSITIONS as ids (DINO seal ref/val are pool positions):
  IN  bank : 200K ACTIVE-anchor rows (40K/source) — pool pos = final_draw[active_id].
             ACTIVE only, never holdouts. These are inside the graph-training draw.
  OUT bank : 200K rows OUTSIDE both training draws AND the seal (40K/source).
  confirm  : 10K wholly-unseen rows (2K/source), excluding both draws, entire seal,
             AND both replay banks. Movement-only (never enters any loss).
Targets = T0_model(L2-normalised X) at identical preprocessing (prenormalized path:
model does not renorm internally). Banks stored fp16 X + fp32 targets + int64 pool ids.
Exclusion invariants asserted and recorded. Availability was pre-checked; re-assert here.
Usage: build_card006_banks.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

POOL = Path("/data2/monet/pool-20m"); SEAL = Path("/data2/monet/eval-common-v2")
SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
T0 = SB / "dino-arrival-t0"; UPD = T0 / "updates"
T0SUB = Path("/data/latent-basemap/substrates/dino-arrival-t0")
FINALSUB = Path("/data/latent-basemap/substrates/dino-arrival-final")
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
PER = 40000; CONF_PER = 2000; SEED = 6006


def _norm(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def _hash_ids(ids):
    return hashlib.sha256(np.ascontiguousarray(np.sort(ids).astype(np.int64)).tobytes()).hexdigest()[:16]


def project(model, X_mm, pos):
    """Gather sorted pool rows from the memmap, L2-normalise, project through the T0 model (CPU, batched)."""
    order = np.argsort(pos); inv = np.argsort(order); spos = pos[order]
    Xg = np.asarray(X_mm[spos], np.float32)         # bounded: len(pos) x 1536
    Xn = _norm(Xg).astype(np.float16)
    outs = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 50000):
            z = model.model(torch.from_numpy(Xn[i:i + 50000].astype(np.float32))).float().numpy()
            outs.append(z.astype(np.float32))
    tgt = np.concatenate(outs)
    return Xn[inv], tgt[inv]                          # restore original (source-grouped) order


def main():
    pool_src = np.load(POOL / "source.npy", allow_pickle=True).astype(str)
    npool = pool_src.shape[0]
    X_mm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    assert X_mm.shape[0] == npool, (X_mm.shape, npool)
    t0 = np.load(T0SUB / "draw_idx.npy"); fin = np.load(FINALSUB / "draw_idx.npy")
    ref = np.load(SEAL / "ref_idx.npy"); val = np.load(SEAL / "val_idx.npy")
    act = np.load(UPD / "anchor_active_ids-anchored.npy")
    model = ParametricUMAP.load(str(T0 / "champion-bs16k/model.pt"), device="cpu"); model.model.eval()

    # exclusion masks over the whole pool
    used = np.zeros(npool, bool)
    used[t0] = True; used[fin] = True; used[ref] = True; used[val] = True
    seal_mask = np.zeros(npool, bool); seal_mask[ref] = True; seal_mask[val] = True
    act_pool = fin[act]                              # active-anchor pool positions
    act_src = pool_src[act_pool]
    rng = np.random.default_rng(SEED)

    manifest = {"schema": "card006-banks-2026-09-10", "dim": 1536, "sources": SOURCES,
                "per_source": {"in": PER, "out": PER, "confirm": CONF_PER}, "seed": SEED,
                "teacher": str(T0 / "champion-bs16k/model.pt"), "availability": {}, "exclusion": {}, "banks": {}}

    # ---- IN bank: active anchors only, 40K/source ----
    in_pos = []
    for s in SOURCES:
        cand = act_pool[act_src == s]
        manifest["availability"][f"in_{s}"] = int(cand.size)
        assert cand.size >= PER, f"IN {s}: {cand.size} < {PER}"
        in_pos.append(rng.choice(cand, PER, replace=False))
    in_pos = np.concatenate(in_pos)
    assert np.isin(in_pos, act_pool).all(), "IN contains a non-active-anchor row"
    assert not seal_mask[in_pos].any(), "IN intersects seal"

    # ---- OUT bank: off both draws + seal, 40K/source ----
    out_pos = []
    for s in SOURCES:
        pos = np.where(pool_src == s)[0]; cand = pos[~used[pos]]
        manifest["availability"][f"out_{s}"] = int(cand.size)
        assert cand.size >= PER, f"OUT {s}: {cand.size} < {PER}"
        out_pos.append(rng.choice(cand, PER, replace=False))
    out_pos = np.concatenate(out_pos)
    assert not used[out_pos].any(), "OUT intersects a training draw or seal"

    # ---- confirmation: 2K/source, exclude draws+seal+BOTH banks ----
    used_conf = used.copy(); used_conf[in_pos] = True; used_conf[out_pos] = True
    conf_pos = []
    for s in SOURCES:
        pos = np.where(pool_src == s)[0]; cand = pos[~used_conf[pos]]
        assert cand.size >= CONF_PER, f"CONF {s}: {cand.size} < {CONF_PER}"
        conf_pos.append(rng.choice(cand, CONF_PER, replace=False))
    conf_pos = np.concatenate(conf_pos)
    assert not used[conf_pos].any() and not np.isin(conf_pos, in_pos).any() and not np.isin(conf_pos, out_pos).any()

    # cross-bank disjointness
    manifest["exclusion"] = {
        "in_out_disjoint": bool(len(np.intersect1d(in_pos, out_pos)) == 0),
        "in_conf_disjoint": bool(len(np.intersect1d(in_pos, conf_pos)) == 0),
        "out_conf_disjoint": bool(len(np.intersect1d(out_pos, conf_pos)) == 0),
        "out_off_both_draws": bool((~np.isin(out_pos, t0)).all() and (~np.isin(out_pos, fin)).all()),
        "in_active_only": bool(np.isin(in_pos, act_pool).all()),
        "no_bank_in_seal": bool(not seal_mask[in_pos].any() and not seal_mask[out_pos].any() and not seal_mask[conf_pos].any()),
    }

    for name, pos, is_loss in [("in", in_pos, True), ("out", out_pos, True), ("confirm", conf_pos, False)]:
        Xn, tgt = project(model, X_mm, pos)
        np.savez(OC / f"card006_{name}_bank.npz", replay_X=Xn, replay_targets=tgt.astype(np.float32),
                 replay_ids=pos.astype(np.int64), source=pool_src[pos])
        manifest["banks"][name] = {"n": int(pos.size), "for_loss": is_loss, "ids_sha16": _hash_ids(pos),
                                   "X_dtype": str(Xn.dtype), "target_dtype": "float32",
                                   "target_finite": bool(np.isfinite(tgt).all()),
                                   "target_norm_mean": round(float(np.linalg.norm(tgt, axis=1).mean()), 4),
                                   "gib": round(pos.size * Xn.shape[1] * 2 / 2**30, 3)}
    (OC / "card006-banks-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"exclusion": manifest["exclusion"], "banks": manifest["banks"]}, indent=1))
    ok = all(manifest["exclusion"].values()) and all(b["target_finite"] for b in manifest["banks"].values())
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Card 007 (Jina-768) replay-bank builder — CPU, off-flock.
Mirrors card006 for the multilingual Chinese-arrival update. 20 "old languages" =
English (its 3 registers) + 19 non-Chinese ml languages. Persistent ids = jina30m
GLOBAL BLOCKS-order indices (the same space as draw_idx / the multilingual seal),
resolved via build_jina_ladder_draws.build_layout/gather (authoritative).
  IN  bank : 200K ACTIVE-anchor rows, 10K/language. Active-anchor global id =
             final_draw_idx[active_id] (FINAL-graph-local, per build_s0_anchor).
             English's 10K split across its 3 registers by S0 ACTIVE-anchor
             proportions, recorded before training.
  OUT bank : 200K rows OUTSIDE both draws (S0 ∪ final) AND the seal, 10K/language;
             English split by the SAME recorded register proportions.
  confirm  : 20K wholly-unseen (1K/language), excluding both draws, entire seal,
             AND both replay banks. Movement-only (never in any loss).
Targets = S0_champion(L2-normalised X) at identical preprocessing. fp16 X + fp32 targets.
Usage: build_card007_banks.py
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
import build_jina_ladder_draws as BJ   # authoritative BLOCKS / build_layout / gather / _norm

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
S0SUB = Path("/data/latent-basemap/substrates/jina-ladder-s0")
FINALSUB = Path("/data/latent-basemap/substrates/jina-ladder-2m-proportional")
UPD = SB / "jina-ladder-2m-s0/updates"
S0_MODEL = SB / "jina-ladder-2m-s0/champion-bs16k/model.pt"
SEAL = Path("/data2/monet/eval-common-multilingual")
EN_BLOCKS = [n for n, _ in BJ.EN]                        # 3 registers
ML_LANGS = [f"ml-{l}" for l in BJ.LANGS if l != "cmn_Hani"]   # 19 non-Chinese
LANG_KEYS = ["en"] + ML_LANGS                            # 20 old languages
PER = 10000; CONF_PER = 1000; SEED = 7007; NPOOL = 30_000_000


def _hash_ids(ids):
    return hashlib.sha256(np.ascontiguousarray(np.sort(ids).astype(np.int64)).tobytes()).hexdigest()[:16]


def project(model, layout, gidx):
    """Gather sorted global rows from shards, L2-normalise, project through S0 (CPU, batched)."""
    order = np.argsort(gidx); inv = np.argsort(order); sg = gidx[order]
    Xn = BJ._norm(BJ.gather(layout, sg)).astype(np.float16)
    outs = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 50000):
            outs.append(model.model(torch.from_numpy(Xn[i:i + 50000].astype(np.float32))).float().numpy().astype(np.float32))
    tgt = np.concatenate(outs)
    return Xn[inv], tgt[inv]


def main():
    layout, g = BJ.build_layout(); assert g == NPOOL
    starts = np.array([gs for _, gs, _, _ in layout]); names = [n for n, _, _, _ in layout]
    block_of = lambda gi: np.searchsorted(starts, gi, side="right") - 1
    lang_of = lambda bi: "en" if names[bi].startswith("en-") else names[bi]

    s0 = np.load(S0SUB / "draw_idx.npy"); fin = np.load(FINALSUB / "draw_idx.npy")
    ref = np.load(SEAL / "ref_idx.npy"); val = np.load(SEAL / "val_idx.npy")
    act = np.load(UPD / "anchor_active_ids-anchored.npy")
    model = ParametricUMAP.load(str(S0_MODEL), device="cpu"); model.model.eval()
    rng = np.random.default_rng(SEED)

    used = np.zeros(NPOOL, bool); used[s0] = True; used[fin] = True; used[ref] = True; used[val] = True
    seal_mask = np.zeros(NPOOL, bool); seal_mask[ref] = True; seal_mask[val] = True

    # active-anchor global ids + their language / EN register
    act_g = fin[act]
    act_bi = block_of(act_g); act_blockname = np.array(names)[act_bi]
    act_lang = np.array([lang_of(b) for b in act_bi])
    # English register proportions among S0 active anchors (recorded before training)
    en_mask = act_lang == "en"; en_counts = {b: int((act_blockname[en_mask] == b).sum()) for b in EN_BLOCKS}
    en_tot = sum(en_counts.values()); en_prop = {b: en_counts[b] / en_tot for b in EN_BLOCKS}

    manifest = {"schema": "card007-banks-2026-09-10", "dim": 768, "languages": LANG_KEYS,
                "per_language": {"in": PER, "out": PER, "confirm": CONF_PER}, "seed": SEED,
                "teacher": str(S0_MODEL), "en_register_counts_S0_active": en_counts,
                "en_register_proportions": {b: round(en_prop[b], 5) for b in EN_BLOCKS},
                "availability": {}, "exclusion": {}, "banks": {}}

    def alloc_en(total):
        """split `total` across the 3 EN registers by recorded S0-active proportions (largest-remainder)."""
        raw = {b: en_prop[b] * total for b in EN_BLOCKS}; base = {b: int(np.floor(raw[b])) for b in EN_BLOCKS}
        rem = total - sum(base.values())
        for b in sorted(EN_BLOCKS, key=lambda b: raw[b] - base[b], reverse=True)[:rem]:
            base[b] += 1
        return base

    def draw_block(blockname, n, pool_mask):
        """draw n global ids from `blockname`'s range where pool_mask is True."""
        bi = names.index(blockname); a = starts[bi]; b = a + layout[bi][2]
        cand = np.arange(a, b)[pool_mask[a:b]]
        return cand, (rng.choice(cand, n, replace=False) if n <= cand.size else None)

    # ---- IN bank: active anchors only, 10K/language ----
    in_ids = []
    for lang in LANG_KEYS:
        if lang == "en":
            alloc = alloc_en(PER)
            for reg, n in alloc.items():
                cand = act_g[(act_lang == "en") & (act_blockname == reg)]
                manifest["availability"][f"in_{reg}"] = int(cand.size)
                assert cand.size >= n, f"IN {reg}: {cand.size} < {n}"
                in_ids.append(rng.choice(cand, n, replace=False))
        else:
            cand = act_g[act_lang == lang]
            manifest["availability"][f"in_{lang}"] = int(cand.size)
            assert cand.size >= PER, f"IN {lang}: {cand.size} < {PER}"
            in_ids.append(rng.choice(cand, PER, replace=False))
    in_ids = np.concatenate(in_ids)
    assert np.isin(in_ids, act_g).all() and not seal_mask[in_ids].any()

    # ---- OUT bank: off both draws + seal, 10K/language (EN by recorded register proportions) ----
    out_ids = []
    for lang in LANG_KEYS:
        if lang == "en":
            alloc = alloc_en(PER)
            for reg, n in alloc.items():
                cand, sel = draw_block(reg, n, ~used)
                manifest["availability"][f"out_{reg}"] = int(cand.size)
                assert sel is not None, f"OUT {reg}: {cand.size} < {n}"
                out_ids.append(sel)
        else:
            cand, sel = draw_block(lang, PER, ~used)
            manifest["availability"][f"out_{lang}"] = int(cand.size)
            assert sel is not None, f"OUT {lang}: {cand.size} < {PER}"
            out_ids.append(sel)
    out_ids = np.concatenate(out_ids)
    assert not used[out_ids].any()

    # ---- confirmation: 1K/language, exclude draws+seal+BOTH banks ----
    used_conf = used.copy(); used_conf[in_ids] = True; used_conf[out_ids] = True
    conf_ids = []
    for lang in LANG_KEYS:
        if lang == "en":
            alloc = alloc_en(CONF_PER)
            for reg, n in alloc.items():
                _, sel = draw_block(reg, n, ~used_conf); assert sel is not None, f"CONF {reg}"
                conf_ids.append(sel)
        else:
            _, sel = draw_block(lang, CONF_PER, ~used_conf); assert sel is not None, f"CONF {lang}"
            conf_ids.append(sel)
    conf_ids = np.concatenate(conf_ids)
    assert not used[conf_ids].any() and not np.isin(conf_ids, in_ids).any() and not np.isin(conf_ids, out_ids).any()

    manifest["exclusion"] = {
        "in_out_disjoint": bool(len(np.intersect1d(in_ids, out_ids)) == 0),
        "in_conf_disjoint": bool(len(np.intersect1d(in_ids, conf_ids)) == 0),
        "out_conf_disjoint": bool(len(np.intersect1d(out_ids, conf_ids)) == 0),
        "out_off_both_draws": bool((~np.isin(out_ids, s0)).all() and (~np.isin(out_ids, fin)).all()),
        "in_active_only": bool(np.isin(in_ids, act_g).all()),
        "no_bank_in_seal": bool(not seal_mask[in_ids].any() and not seal_mask[out_ids].any() and not seal_mask[conf_ids].any()),
        "no_chinese": bool(not any(names[b].endswith("cmn_Hani") for b in
                                   np.unique(np.concatenate([block_of(in_ids), block_of(out_ids), block_of(conf_ids)])))),
    }
    for name, ids, is_loss in [("in", in_ids, True), ("out", out_ids, True), ("confirm", conf_ids, False)]:
        Xn, tgt = project(model, layout, ids)
        lang_arr = np.array([lang_of(b) for b in block_of(ids)])
        np.savez(OC / f"card007_{name}_bank.npz", replay_X=Xn, replay_targets=tgt.astype(np.float32),
                 replay_ids=ids.astype(np.int64), language=lang_arr)
        manifest["banks"][name] = {"n": int(ids.size), "for_loss": is_loss, "ids_sha16": _hash_ids(ids),
                                   "X_dtype": str(Xn.dtype), "target_finite": bool(np.isfinite(tgt).all()),
                                   "target_norm_mean": round(float(np.linalg.norm(tgt, axis=1).mean()), 4),
                                   "gib": round(ids.size * 768 * 2 / 2**30, 3)}
    (OC / "card007-banks-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"en_register_proportions": manifest["en_register_proportions"],
                      "exclusion": manifest["exclusion"], "banks": manifest["banks"]}, indent=1))
    ok = all(manifest["exclusion"].values()) and all(b["target_finite"] for b in manifest["banks"].values())
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

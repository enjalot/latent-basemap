"""Cards 006/007 reception scorer (CPU, off-flock) — deployment criteria 2 & 3, frozen thresholds.
Per-query recall on the original sealed instrument (eval_jina_pair.per_query) for heads:
frozen (T0/S0), original anchored (card005/004), IN replay, OUT replay. Reports, per group and
equal-cohort aggregate, at B250/B2000:
  Criterion 2 (old-quality retention): equal-cohort mean recall LOSS vs frozen <= .005 and worst
    cohort loss <= .01 at BOTH budgets (Vietnamese called out for Jina).
  Criterion 3 (arriving quality): arriving aggregate no worse than ORIGINAL anchored by .005 at
    either budget, and positive B250 gain over frozen with paired-bootstrap CI > 0 (Jina also
    requires original Chinese gain >= .03).
Reports booleans + numbers only; decides no gate. Persists per-query recall arrays for paired
uncertainty. Env CARD=card006|card007. Usage: score_replay_reception.py
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import eval_jina_pair as EP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CARD = os.environ.get("CARD", "card006")

CFG = {
 "card006": {"seal": Path("/data2/monet/eval-common-v2"), "grp_file": "val_source.npy",
             "outd": SB / "dino-arrival-t0/replay-updates",
             "heads": {"frozen": SB / "dino-arrival-t0/champion-bs16k/model.pt",
                       "anchored": SB / "dino-arrival-t0/updates/model-anchored.pt",
                       "unanchored": SB / "dino-arrival-t0/updates/model-unanchored.pt"},
             "arriving": ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"],
             "old": ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"],
             "exclude": ["diffusion-aesthetic-4k"], "chinese": None},
 "card007": {"seal": Path("/data2/monet/eval-common-multilingual"), "grp_file": "val_cohort.npy",
             "outd": SB / "jina-ladder-2m-s0/replay-updates",
             "heads": {"frozen": SB / "jina-ladder-2m-s0/champion-bs16k/model.pt",
                       "anchored": SB / "jina-ladder-2m-s0/updates/model-anchored.pt",
                       "unanchored": SB / "jina-ladder-2m-s0/updates/model-unanchored.pt"},
             "arriving": ["cmn_Hani", "ml-cmn_Hani"], "old": None, "exclude": [], "chinese": ["cmn_Hani", "ml-cmn_Hani"]},
}[CARD]


def main():
    seal = CFG["seal"]
    ref_hd = np.asarray(np.load(seal / "ref_hd.f16.npy"), np.float32)
    val_hd = np.asarray(np.load(seal / "val_hd.f16.npy"), np.float32)
    truth = np.load(seal / "truth_val.npy")
    grp = np.load(seal / CFG["grp_file"], allow_pickle=True).astype(str)
    groups = sorted(set(grp.tolist()))
    arriving = [g for g in groups if g in set(CFG["arriving"])]
    old = CFG["old"] if CFG["old"] else [g for g in groups if g not in set(CFG["arriving"]) and g not in set(CFG["exclude"])]

    heads = dict(CFG["heads"])
    for tag in ("in", "out"):
        p = CFG["outd"] / f"model-{tag}.pt"
        if p.exists(): heads[tag] = p
    pq = {h: EP.per_query(str(ck), ref_hd, val_hd, truth) for h, ck in heads.items() if Path(ck).exists()}
    np.savez(OC / f"{CARD}-reception-perq.npz", val_group=grp,
             **{f"{h}_B{B}": pq[h][B] for h in pq for B in EP.BUDGETS})

    def gm(h, B, g): return float(pq[h][B][grp == g].mean())
    rep = {h: {f"B{B}": {g: round(gm(h, B, g), 4) for g in groups} for B in EP.BUDGETS} for h in pq}

    # Criterion 2: old-quality retention (recall LOSS vs frozen) — equal-cohort mean & worst, both budgets
    crit2 = {}
    for tag in [t for t in ("in", "out") if t in pq]:
        c2 = {}
        for B in EP.BUDGETS:
            losses = {g: gm("frozen", B, g) - gm(tag, B, g) for g in old}   # positive = dropped recall
            c2[f"B{B}"] = {"mean_loss": round(float(np.mean(list(losses.values()))), 5),
                           "worst_loss": round(float(max(losses.values())), 5),
                           "worst_group": max(losses, key=losses.get)}
        c2["pass"] = bool(all(c2[f"B{B}"]["mean_loss"] <= 0.005 and c2[f"B{B}"]["worst_loss"] <= 0.01 for B in EP.BUDGETS))
        crit2[tag] = c2

    # Criterion 3: arriving quality (vs anchored by .005; positive B250 gain over frozen, paired CI>0)
    crit3 = {}
    rng = np.random.default_rng(0)
    for tag in [t for t in ("in", "out") if t in pq]:
        c3 = {}
        for B in EP.BUDGETS:
            arr_tag = float(np.mean([gm(tag, B, g) for g in arriving])) if arriving else None
            arr_anch = float(np.mean([gm("anchored", B, g) for g in arriving])) if ("anchored" in pq and arriving) else None
            c3[f"B{B}"] = {"arriving_agg": round(arr_tag, 4) if arr_tag is not None else None,
                           "anchored_agg": round(arr_anch, 4) if arr_anch is not None else None,
                           "no_worse_than_anchored_by.005": bool(arr_anch is None or arr_tag >= arr_anch - 0.005)}
        # positive B250 gain over frozen with paired CI>0 (per-query over arriving)
        if arriving:
            qa = np.concatenate([(pq[tag][250] - pq["frozen"][250])[grp == g] for g in arriving])
            bs = np.array([qa[rng.integers(0, qa.size, qa.size)].mean() for _ in range(2000)])
            c3["arriving_B250_gain"] = round(float(qa.mean()), 5)
            c3["arriving_B250_ci95"] = (round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5))
            c3["positive_gain_ci_above_0"] = bool(np.percentile(bs, 2.5) > 0)
            if CFG["chinese"]:
                c3["chinese_gain_ge.03"] = bool(c3["arriving_B250_gain"] >= 0.03)
        crit3[tag] = c3

    out = {"schema": f"{CARD}-reception-2026-09-10", "card": CARD, "budgets": list(EP.BUDGETS),
           "groups": {"arriving": arriving, "old": old, "excluded": CFG["exclude"]},
           "reception_per_group": rep, "criterion2_old_retention": crit2, "criterion3_arriving": crit3,
           "note": "recall on the original sealed instrument; equal-cohort means; loss = frozen - arm (positive = drop). "
                   "Decides no gate; card006 movement already reported separately."}
    (OC / f"{CARD}-reception.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"card": CARD, "heads": list(pq), "criterion2": {t: crit2[t]["pass"] for t in crit2},
                      "criterion3": {t: {k: crit3[t].get(k) for k in ("positive_gain_ci_above_0", "arriving_B250_gain")} for t in crit3}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

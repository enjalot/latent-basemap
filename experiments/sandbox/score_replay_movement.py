"""Cards 006/007 movement + coverage scorer (CPU, off-flock) — the PRIMARY OUT-vs-IN causal
comparison under thresholds frozen in the card before training. Reporting code
was corrected after DINO results, per card006-result-review.md:
 - old_sealed aggregate = OLD cohorts ONLY (DINO 5 real; Jina non-Chinese); arriving/diagnostic
   query movements remain available in full-seal arrays, excluded from the old aggregate.
 - fixed radius read from the EXPLICIT T0/S0 field (not a fuzzy key match).
 - SYMMETRIC input: teacher target = original head applied to the SAME stored fp16 the student
   projects (both via the same stored input), so a quantization delta cannot masquerade as drift.
 - controls (original anchored / unanchored / fresh) scored on confirmation + old_sealed too.
 - per-point IDs, groups, raw-native AND rigid-aligned displacement, R/t, radius provenance all
   persisted to <card>-movement-arrays.npz; confirmation pairs joined by ID (shared bank order).
 - coverage/deployment booleans computed on FULL-precision arrays (display values are rounded).
Evaluates the fixed criteria separately from training completion. Env CARD=card006|card007. Usage: score_replay_movement.py
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch, frame as F
from basemap.pumap.parametric_umap.core import ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
CARD = os.environ.get("CARD", "card006")

CFG = {
 "card006": {"dim": 1536, "grp": "source", "radius_key": "fixed_radius_trained_T0_centroid_p90",
             "orig_champ": SB / "dino-arrival-t0/champion-bs16k/model.pt",
             "orig_coords": SB / "dino-arrival-t0/champion-bs16k/coordinates.npy",
             "anchor_meta": SB / "dino-arrival-t0/anchor.meta.json",
             "t0_draw": Path("/data/latent-basemap/substrates/dino-arrival-t0/draw_idx.npy"),
             "final_draw": Path("/data/latent-basemap/substrates/dino-arrival-final/draw_idx.npy"),
             "outd": SB / "dino-arrival-t0/replay-updates",
             "orig_active": SB / "dino-arrival-t0/updates/anchor_active_ids-anchored.npy",
             "orig_holdout": SB / "dino-arrival-t0/updates/anchor_holdout_ids-anchored.npy",
             "seal": Path("/data2/monet/eval-common-v2"), "seal_hd": "val_hd.f16.npy", "seal_grp": "val_source.npy",
             "arriving": ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"],
             "old": ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"],
             "controls": {"anchored": (SB / "dino-arrival-t0/updates/model-anchored.pt", SB / "dino-arrival-t0/updates/coords-anchored.npy"),
                          "unanchored": (SB / "dino-arrival-t0/updates/model-unanchored.pt", SB / "dino-arrival-t0/updates/coords-unanchored.npy"),
                          "fresh": (SB / "dino-arrival-final/champion-bs16k/model.pt", SB / "dino-arrival-final/champion-bs16k/coordinates.npy")}},
 "card007": {"dim": 768, "grp": "language", "radius_key": "fixed_S0_centroid_p90_radius",
             "orig_champ": SB / "jina-ladder-2m-s0/champion-bs16k/model.pt",
             "orig_coords": SB / "jina-ladder-2m-s0/champion-bs16k/coordinates.npy",
             "anchor_meta": OC / "card004-movement-audit.json",
             "t0_draw": Path("/data/latent-basemap/substrates/jina-ladder-s0/draw_idx.npy"),
             "final_draw": Path("/data/latent-basemap/substrates/jina-ladder-2m-proportional/draw_idx.npy"),
             "outd": SB / "jina-ladder-2m-s0/replay-updates",
             "orig_active": SB / "jina-ladder-2m-s0/updates/anchor_active_ids-anchored.npy",
             "orig_holdout": SB / "jina-ladder-2m-s0/updates/anchor_holdout_ids-anchored.npy",
             "seal": Path("/data2/monet/eval-common-multilingual"), "seal_hd": "val_hd.f16.npy", "seal_grp": "val_cohort.npy",
             "arriving": ["cmn_Hani", "ml-cmn_Hani"], "old": None,   # old = every non-Chinese seal cohort
             "controls": {"anchored": (SB / "jina-ladder-2m-s0/updates/model-anchored.pt", SB / "jina-ladder-2m-s0/updates/coords-anchored.npy"),
                          "unanchored": (SB / "jina-ladder-2m-s0/updates/model-unanchored.pt", SB / "jina-ladder-2m-s0/updates/coords-unanchored.npy"),
                          "fresh": (SB / "jina-ladder-2m-proportional/champion-bs16k/model.pt", SB / "jina-ladder-2m-proportional/champion-bs16k/coordinates.npy")}},
}[CARD]


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def project(model, X, normalize=True):
    o = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 50000):
            o.append(model.model(torch.from_numpy(_norm(X[i:i + 50000]) if normalize else np.asarray(X[i:i + 50000], np.float32))).float().numpy().astype(np.float32))
    return np.concatenate(o)


def stat(d):   # d already divided by radius; full-precision values, rounded only for display
    return {"n": int(d.size), "mean": round(float(d.mean()), 6), "p95": round(float(np.percentile(d, 95)), 6),
            "p99": round(float(np.percentile(d, 99)), 6), "frac_gt_.05": round(float((d > 0.05).mean()), 5),
            "frac_gt_.10": round(float((d > 0.10).mean()), 5)}


def main():
    dim, grp = CFG["dim"], CFG["grp"]
    meta = json.load(open(CFG["anchor_meta"]))
    radius = float(meta[CFG["radius_key"]])
    if not np.isfinite(radius) or radius <= 0: raise ValueError("Invalid fixed radius")   # explicit field, verified against prior movement audit
    t0_coords = np.asarray(np.load(CFG["orig_coords"]), np.float64)
    t0_draw = np.load(CFG["t0_draw"]); final_draw = np.load(CFG["final_draw"])
    active = np.load(CFG["orig_active"]); holdout = np.load(CFG["orig_holdout"])
    common, t0_local, final_local = np.intersect1d(t0_draw, final_draw, assume_unique=True, return_indices=True)
    active_mask = np.isin(final_local, active); hold_mask = np.isin(final_local, holdout)
    s0o = t0_coords[t0_local]
    orig = ParametricUMAP.load(str(CFG["orig_champ"]), device="cpu"); orig.model.eval()

    # off-graph cohorts (shared across heads): SYMMETRIC teacher target = orig head on the SAME fp16 input
    def bank(name):
        z = np.load(OC / f"{CARD}_{name}_bank.npz")
        g = z["source"] if "source" in z else z["language"]
        return np.asarray(z["replay_X"], np.float32), np.asarray(z["replay_ids"]), g.astype(str)
    conf_X, conf_ids, conf_g = bank("confirm")
    conf_tgt = project(orig, conf_X, normalize=False)                     # symmetric with student (same _norm(fp16))
    seal_hd = np.asarray(np.load(CFG["seal"] / CFG["seal_hd"]), np.float32)
    seal_g = np.load(CFG["seal"] / CFG["seal_grp"], allow_pickle=True).astype(str)
    seal_ids = np.load(CFG["seal"] / "val_idx.npy")
    seal_tgt = project(orig, seal_hd)
    OLD = CFG["old"] if CFG["old"] else [g for g in sorted(set(seal_g.tolist())) if g not in set(CFG["arriving"])]
    old_seal_mask = np.isin(seal_g, OLD)                 # OLD-only aggregate (5 real for DINO; non-Chinese for Jina)

    arrays = {"graph_ids": common, "graph_active_mask": active_mask, "graph_holdout_mask": hold_mask, "radius": radius, "radius_key": CFG["radius_key"], "confirm_ids": conf_ids, "confirm_group": conf_g,
              "seal_ids": seal_ids, "seal_group": seal_g, "old_seal_mask": old_seal_mask}
    heads = {}
    HEADSPEC = {"in": (CFG["outd"] / "model-in.pt", CFG["outd"] / "coords-in.npy"),
                "out": (CFG["outd"] / "model-out.pt", CFG["outd"] / "coords-out.npy")}
    HEADSPEC.update(CFG["controls"])
    for tag, (mp, cp) in HEADSPEC.items():
        if not (Path(mp).exists() and Path(cp).exists()):
            raise FileNotFoundError(f"Required movement head absent: {tag}: {mp}, {cp}")
        mo = ParametricUMAP.load(str(mp), device="cpu"); mo.model.eval()
        upd = np.asarray(np.load(cp), np.float64)[final_local]
        _, info = F.rigid_align(upd[active_mask], s0o[active_mask])   # fit on ORIGINAL ACTIVE anchors
        R, tvec = np.asarray(info["R"], np.float64), np.asarray(info["t"], np.float64)
        def align(c): return np.asarray(c, np.float64) @ R.T + tvec
        res = {"status": "scored", "rigid_rmsd": round(float(info.get("rmsd", np.nan)), 6),
               "R": R.tolist(), "t": tvec.tolist()}
        # graph cohorts (only meaningful for in/out and the controls that share this draw)
        d_graph = np.linalg.norm(align(upd) - s0o, axis=1) / radius
        res["graph"] = {"active": stat(d_graph[active_mask]), "anchor_holdout": stat(d_graph[hold_mask]),
                        "all_retained": stat(d_graph)}
        # off-graph cohorts: aligned + native (unaligned) displacement vs symmetric teacher target
        cohorts = {"confirmation": (conf_X, conf_tgt, conf_g, None),
                   "old_sealed": (seal_hd, seal_tgt, seal_g, old_seal_mask)}
        if tag in ("in", "out"):
            own_X, own_ids, own_g = bank(tag); cohorts["own_replay"] = (own_X, project(orig, own_X, normalize=False), own_g, None)
        for cname, (X, tgt, g, submask) in cohorts.items():
            proj = project(mo, X, normalize=(cname == "old_sealed")); tgt = np.asarray(tgt, np.float64)
            d_al = np.linalg.norm(align(proj) - tgt, axis=1) / radius       # rigid-aligned (primary)
            d_nat = np.linalg.norm(np.asarray(proj, np.float64) - tgt, axis=1) / radius  # native (secondary)
            sel = submask if submask is not None else np.ones(d_al.shape[0], bool)
            if not np.isfinite(d_al).all() or not np.isfinite(d_nat).all(): raise ValueError("Nonfinite movement")
            per_grp = {str(c): stat(d_al[(g == c)]) for c in sorted(set(g[sel].tolist()))}
            res[cname] = {"aggregate": stat(d_al[sel]), "native_aggregate": stat(d_nat[sel]), "per_group": per_grp}
            arrays[f"{tag}_{cname}_aligned"] = d_al
            arrays[f"{tag}_{cname}_native"] = d_nat
        arrays[f"{tag}_graph_aligned"] = d_graph
        arrays[f"{tag}_R"] = R; arrays[f"{tag}_t"] = tvec
        if tag in ("in", "out"):
            arrays[f"{tag}_own_replay_ids"] = own_ids
            arrays[f"{tag}_own_replay_group"] = own_g
        heads[tag] = res

    out = {"schema": f"{CARD}-movement-v2-2026-09-10", "card": CARD, "dim": dim,
           "fixed_radius": radius, "radius_key": CFG["radius_key"], "group_key": grp,
           "OLD_cohorts": OLD, "arriving_cohorts": CFG["arriving"],
           "movement": {t: heads[t] for t in heads}}

    # ---- coverage hypothesis (IN vs OUT on confirmation, paired by ID) + deployment crit1 ----
    if heads.get("in", {}).get("status") == "scored" and heads.get("out", {}).get("status") == "scored":
        di, do = arrays["in_confirmation_aligned"].astype(np.float64), arrays["out_confirmation_aligned"].astype(np.float64)
        p99_in, p99_out = float(np.percentile(di, 99)), float(np.percentile(do, 99))   # full precision
        rng = np.random.default_rng(0); n = di.size
        diffs = np.array([(lambda idx: np.percentile(do[idx], 99) - np.percentile(di[idx], 99))(rng.integers(0, n, n)) for _ in range(2000)])
        ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        out["coverage_hypothesis"] = {"confirm_p99_in": p99_in, "confirm_p99_out": p99_out,
                                      "p99_ratio_out_over_in": round(p99_out / p99_in, 4) if p99_in else None,
                                      "out_le_half_in": bool(p99_out <= 0.5 * p99_in),
                                      "p99_out_minus_in_ci95": [round(ci[0], 6), round(ci[1], 6)],
                                      "ci_below_zero": bool(ci[1] < 0),
                                      "coverage_success": bool(p99_out <= 0.5 * p99_in and ci[1] < 0),
                                      "note": "Fixed halving-plus-negative-CI criterion; interpret the actual estimate/interval, not an equivalence test."}
        dep = {}
        for tag in ("in", "out"):
            d = arrays[f"{tag}_confirmation_aligned"]
            means = [float(d[conf_g == g].mean()) for g in np.unique(conf_g)]
            tails = [float(np.percentile(d[conf_g == g], 99)) for g in np.unique(conf_g)]
            agg_mean, agg_p99 = float(d.mean()), float(np.percentile(d, 99))
            dep[tag] = {"aggregate_mean_le.01": agg_mean <= .01, "aggregate_p99_le.05": agg_p99 <= .05,
                        "worst_group_mean": max(means), "worst_group_p99": max(tails),
                        "criterion1_pass": bool(agg_mean <= .01 and agg_p99 <= .05
                                                and max(means) <= .01 and max(tails) <= .05)}
        out["deployment_criterion1_confirmation"] = dep
        # context: does adding replay help the unseen tail vs anchored-alone (no replay)?
        out["confirmation_p99_by_head"] = {t: float(np.percentile(arrays[f"{t}_confirmation_aligned"].astype(np.float64), 99))
                                           for t in heads if heads[t].get("status") == "scored"}
    else:
        out["coverage_hypothesis"] = {"status": "pending — both arms required"}

    np.savez(OC / f"{CARD}-movement-arrays.npz", **arrays)
    (OC / f"{CARD}-movement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"card": CARD, "radius": radius, "heads": [t for t in heads if heads[t].get("status") == "scored"],
                      "coverage": out.get("coverage_hypothesis"),
                      "confirm_p99_by_head": out.get("confirmation_p99_by_head")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the FFR v1-vs-v2 verdict-stability table (external review 2026-08-27).

For every headline conclusion, pull the v1 (quick_ffr_at_0.1pct) and the freshly
rescored v2 (quick_ffr_v2) from each map's summary.json, and for the OOD/register
verdicts recompute v2 from the scorers' SAVED projected coordinates (CPU, no GPU,
no retrain).  Emit /data/latent-basemap/sandbox/ffr-v2-verdict-stability.json plus
a short markdown companion.

Run AFTER rescore_ffr_v2.py --run has populated the map summaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knobs_2m import quick_ffr, quick_ffr_v2  # noqa: E402

SB = Path("/data/latent-basemap/sandbox")


def pair(rel: str):
    """(v1, v2) from a map dir's summary.json."""
    f = SB / rel / "summary.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    return d.get("quick_ffr_at_0.1pct"), d.get("quick_ffr_v2")


def order(d: dict):
    return [k for k, _ in sorted(((k, v) for k, v in d.items() if v is not None),
                                 key=lambda kv: kv[1], reverse=True)]


def recompute_from_coords(coord_map: dict[str, Path], edges_of: dict[str, Path]):
    """coord_map: label -> saved projected coords .npy; edges_of: label -> truth npz.
    Returns label -> {v1, v2}."""
    out = {}
    for label, cp in coord_map.items():
        if not cp.exists() or not edges_of[label].exists():
            out[label] = {"v1": None, "v2": None}
            continue
        xy = np.asarray(np.load(cp), dtype=np.float32)
        n = int(xy.shape[0])
        out[label] = {"v1": float(quick_ffr(xy, edges_of[label], n)),
                      "v2": float(quick_ffr_v2(xy, edges_of[label], n))}
    return out


def main() -> int:
    V: dict = {}

    # ---- A. champion transfer ordering + %-of-ceiling (per space) ----
    spaces = {
        "jina":   ("jina-multi-2m/champion-bs16k",           "jina-multi-2m/upstream-06dev"),
        "sisap":  ("sisap-clip-2m/champion-bs16k",           "sisap-clip-2m/upstream-06dev"),
        "BL":     ("bl-siglip-1m/champion-bs16k",            "bl-siglip-1m/upstream-06dev"),
        "MiniLM": ("2m-knobs/umap-md000-x4bs16k-winner",     "minilm-mix-2m/upstream-06dev"),
    }
    champ = {}
    for sp, (arm, ceil) in spaces.items():
        c1, c2 = pair(arm)
        k1, k2 = pair(ceil)
        champ[sp] = {
            "champion": {"v1": c1, "v2": c2},
            "ceiling":  {"v1": k1, "v2": k2},
            "pct_of_ceiling_v1": (round(100 * c1 / k1, 1) if c1 and k1 else None),
            "pct_of_ceiling_v2": (round(100 * c2 / k2, 1) if c2 and k2 else None),
        }
    ord_v1 = order({sp: champ[sp]["pct_of_ceiling_v1"] for sp in champ})
    ord_v2 = order({sp: champ[sp]["pct_of_ceiling_v2"] for sp in champ})
    abs_v1 = order({sp: champ[sp]["champion"]["v1"] for sp in champ})
    abs_v2 = order({sp: champ[sp]["champion"]["v2"] for sp in champ})
    V["A_champion_transfer"] = {
        "per_space": champ,
        "order_by_pct_of_ceiling_v1": ord_v1, "order_by_pct_of_ceiling_v2": ord_v2,
        "order_by_abs_champion_v1": abs_v1, "order_by_abs_champion_v2": abs_v2,
        "pct_order_preserved": ord_v1 == ord_v2,
        "abs_order_preserved": abs_v1 == abs_v2,
    }

    # ---- B. dose8 width ladder + halving increments ----
    ladder = ["champion-x8-h2048", "champion-x8-h3072", "champion-x8-h4096"]
    lv = {a: pair(f"jina-multi-2m/{a}") for a in ladder}
    inc = lambda v, i: (round(v[ladder[i + 1]][i2] - v[ladder[i]][i2], 4)
                        if v[ladder[i]][i2] is not None and v[ladder[i + 1]][i2] is not None
                        else None)
    i2 = 0
    inc1 = [round(lv[ladder[1]][0] - lv[ladder[0]][0], 4),
            round(lv[ladder[2]][0] - lv[ladder[1]][0], 4)]
    inc2 = [round(lv[ladder[1]][1] - lv[ladder[0]][1], 4),
            round(lv[ladder[2]][1] - lv[ladder[1]][1], 4)]
    mono_v1 = lv[ladder[0]][0] < lv[ladder[1]][0] < lv[ladder[2]][0]
    mono_v2 = lv[ladder[0]][1] < lv[ladder[1]][1] < lv[ladder[2]][1]
    halving_v1 = inc1[1] < inc1[0]
    halving_v2 = inc2[1] < inc2[0]
    V["B_dose8_width_ladder"] = {
        "values": {a: {"v1": lv[a][0], "v2": lv[a][1]} for a in ladder},
        "increments_v1": {"h2048->h3072": inc1[0], "h3072->h4096": inc1[1]},
        "increments_v2": {"h2048->h3072": inc2[0], "h3072->h4096": inc2[1]},
        "monotone_increasing_v1": bool(mono_v1), "monotone_increasing_v2": bool(mono_v2),
        "increment_halving_v1": bool(halving_v1), "increment_halving_v2": bool(halving_v2),
        "verdict_preserved": bool(mono_v1 == mono_v2 and halving_v1 == halving_v2
                                  and mono_v2 and halving_v2),
    }

    # ---- C. rank25 / 25%-fraction rule ----
    r = {k: pair(f"6250k-knobs/{v}") for k, v in {
        "fixed500k": "umap-md000-x4bs16k-winner",
        "norank":    "umap-md000-x4bs16k-winner-norank",
        "rank25":    "umap-md000-x4bs16k-winner-rank25"}.items()}
    ord6_v1 = order({k: r[k][0] for k in r})
    ord6_v2 = order({k: r[k][1] for k in r})
    knee = {k: pair(f"minilm-mix-1m/{v}") for k, v in {
        "12.5pct": "rankfrac-12p5", "25pct": "rankfrac-25", "50pct": "rankfrac-50"}.items()}
    knee_v1 = max(knee, key=lambda k: knee[k][0])
    knee_v2 = max(knee, key=lambda k: knee[k][1])
    V["C_rank25_fraction_rule"] = {
        "at_6.25M": {k: {"v1": r[k][0], "v2": r[k][1]} for k in r},
        "order_6.25M_v1": ord6_v1, "order_6.25M_v2": ord6_v2,
        "rank25_gt_norank_gt_fixed_v1": ord6_v1 == ["rank25", "norank", "fixed500k"],
        "rank25_gt_norank_gt_fixed_v2": ord6_v2 == ["rank25", "norank", "fixed500k"],
        "order_preserved": ord6_v1 == ord6_v2,
        "1M_knee": {k: {"v1": knee[k][0], "v2": knee[k][1]} for k in knee},
        "1M_argmax_v1": knee_v1, "1M_argmax_v2": knee_v2,
        "knee_at_25pct_preserved": knee_v1 == knee_v2 == "25pct",
    }

    # ---- D. dose-vs-N surface + int8 tax ----
    dose = {}
    for N, ds in (("500K", "minilm-mix-500k"), ("1M", "minilm-mix-1m")):
        row = {}
        for lab, arm in (("x2", "dose-x2-rf25"), ("x4", "dose-x4-rf25"), ("x8", "dose-x8-rf25")):
            v1, v2 = pair(f"{ds}/{arm}")
            row[lab] = {"v1": v1, "v2": v2}
        dose[N] = row
    # dose monotone-in-dose per N
    def mono_dose(row, key):
        vals = [row[l][key] for l in ("x2", "x4", "x8") if row[l][key] is not None]
        return len(vals) == 3 and vals[0] < vals[1] < vals[2]
    int8 = {}
    for lab, (ds, arm) in {"fp16": ("minilm-mix-500k", "int8fac-fp16"),
                           "qdq": ("minilm-mix-500k-qdq", "int8fac-qdq"),
                           "hostint8": ("minilm-mix-500k", "int8fac-hostint8")}.items():
        v1, v2 = pair(f"{ds}/{arm}")
        int8[lab] = {"v1": v1, "v2": v2}
    int8_ord_v1 = order({k: int8[k]["v1"] for k in int8})
    int8_ord_v2 = order({k: int8[k]["v2"] for k in int8})
    V["D_dose_vs_N_and_int8"] = {
        "dose_surface": dose,
        "dose_monotone_v1": {N: mono_dose(dose[N], "v1") for N in dose},
        "dose_monotone_v2": {N: mono_dose(dose[N], "v2") for N in dose},
        "int8_splits": int8,
        "int8_order_v1": int8_ord_v1, "int8_order_v2": int8_ord_v2,
        "int8_fp16_gt_qdq_gt_hostint8_v1": int8_ord_v1 == ["fp16", "qdq", "hostint8"],
        "int8_fp16_gt_qdq_gt_hostint8_v2": int8_ord_v2 == ["fp16", "qdq", "hostint8"],
        "int8_order_preserved": int8_ord_v1 == int8_ord_v2,
    }

    # ---- E. P3 reversal (recompute v2 from saved projected coords) ----
    p3_coords = SB / "p3-probe-coords"
    regs3 = ("reddit-2m", "communityarchive-2m")
    keys3 = ("a", "b", "c")
    p3_per_reg = {}
    for reg in regs3:
        cm = {k: p3_coords / f"{reg}--{k}.npy" for k in keys3}
        eo = {k: SB / reg / "edges-k15-fuzzy.npz" for k in keys3}
        p3_per_reg[reg] = recompute_from_coords(cm, eo)
    p3_mean = {}
    for k in keys3:
        v1s = [p3_per_reg[reg][k]["v1"] for reg in regs3 if p3_per_reg[reg][k]["v1"] is not None]
        v2s = [p3_per_reg[reg][k]["v2"] for reg in regs3 if p3_per_reg[reg][k]["v2"] is not None]
        p3_mean[k] = {"v1": float(np.mean(v1s)) if v1s else None,
                      "v2": float(np.mean(v2s)) if v2s else None}
    p3_ord_v1 = order({k: p3_mean[k]["v1"] for k in keys3})
    p3_ord_v2 = order({k: p3_mean[k]["v2"] for k in keys3})
    predicted = None
    sc = SB / "p3-scorecard.json"
    if sc.exists():
        try:
            predicted = json.loads(sc.read_text()).get("predicted_diversity_order")
        except Exception:
            predicted = None
    V["E_p3_reversal"] = {
        "note": "recomputed from saved p3-probe-coords (CPU, no GPU); a=current-mix "
                "b=curated c=random; predicted diversity order from scorecard.",
        "per_register": p3_per_reg,
        "per_map_mean": p3_mean,
        "probe_order_v1": p3_ord_v1, "probe_order_v2": p3_ord_v2,
        "predicted_diversity_order": predicted,
        "reversal_v1": (predicted is not None and p3_ord_v1 != predicted),
        "reversal_v2": (predicted is not None and p3_ord_v2 != predicted),
        "probe_order_preserved": p3_ord_v1 == p3_ord_v2,
    }

    # ---- F. decomposition exposure > capacity ----
    base = pair("jina-multi-2m/champion-bs16k")
    expo = pair("jina-multi-2m/champion-x8-h2048")    # exposure lever (dose8)
    capa = pair("jina-multi-2m/champion-x4-h3072")    # capacity lever (width)
    gain = lambda arm, i: (round(arm[i] - base[i], 4) if arm[i] is not None and base[i] is not None else None)
    V["F_decomposition_exposure_gt_capacity"] = {
        "baseline_champion": {"v1": base[0], "v2": base[1]},
        "exposure_x8_h2048": {"v1": expo[0], "v2": expo[1], "gain_v1": gain(expo, 0), "gain_v2": gain(expo, 1)},
        "capacity_x4_h3072": {"v1": capa[0], "v2": capa[1], "gain_v1": gain(capa, 0), "gain_v2": gain(capa, 1)},
        "exposure_gt_capacity_v1": (expo[0] is not None and capa[0] is not None and expo[0] > capa[0]),
        "exposure_gt_capacity_v2": (expo[1] is not None and capa[1] is not None and expo[1] > capa[1]),
    }
    f = V["F_decomposition_exposure_gt_capacity"]
    f["verdict_preserved"] = bool(f["exposure_gt_capacity_v1"] == f["exposure_gt_capacity_v2"]
                                  and f["exposure_gt_capacity_v2"])

    # ---- G. sweep maximin (only if the sweep coords/results exist) ----
    sweep_coords = SB / "mixture-sweep-coords"
    have_sweep = sweep_coords.exists() and any(sweep_coords.glob("*.npy"))
    V["G_sweep_maximin"] = {
        "status": "AVAILABLE" if have_sweep else "NOT_YET_RUN",
        "note": "mixture-sweep-coords is empty; the broad-probe sweep has not been "
                "scored yet, so no v1/v2 maximin comparison exists. Re-run "
                "mixture_probe.py (now v2) on GPU when the sweep maps are trained."
        if not have_sweep else "sweep coords present; recompute per mixture_probe.",
    }

    # P2 jina register probe (bonus: recompute v2 from saved coords)
    p2c = SB / "p2-jina-probe-coords"
    p2_regs = ("reddit-jina-250k", "ca-jina-250k")
    cm = {reg: p2c / f"{reg}.npy" for reg in p2_regs}
    eo = {reg: SB / reg / "edges-k15-fuzzy.npz" for reg in p2_regs}
    V["P2_jina_register_probe"] = recompute_from_coords(cm, eo)

    out = SB / "ffr-v2-verdict-stability.json"
    out.write_text(json.dumps({
        "schema": "ffr-v1-vs-v2-verdict-stability-2026-08-27",
        "instrument": "knobs_2m.quick_ffr_v2 (exact knn_indices.npy where present, "
                      "else top-15 by fuzzy weight; query excluded from its own "
                      "high-D truth and 2D disc budget).",
        "verdicts": V,
    }, indent=1))
    print(f"wrote {out}")
    # console digest
    for name, blk in V.items():
        print(f"\n=== {name} ===")
        print(json.dumps(blk, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

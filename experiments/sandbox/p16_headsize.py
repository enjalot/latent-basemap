#!/usr/bin/env python3
"""P1.6 projection head-size experiment (4th-review, delegate 2026-08-29).

Question: does a SMALLER head (2M / 4M) project onto the full 6.25M atlas as well as a
6.25M-TRAINED map? Resident floor=0 so within-experiment comparisons are EXACT (single seed 42;
seed-43 only on the decisive cell if it lands within 2x of the gate boundary).

Three composition-matched jina heads, all champion recipe (rankneg = 25% of N), seed 42:
  * 2M  = jina-multi-2m / p15-baseline-s42        (REUSE the P1.5 baseline; member mask = 2M old-block prefixes)
  * 4M  = jina-4m-head / champion-bs16k           (nested 64%-per-span; member mask = member_indices.npy)
  * ref = jina-multi-6m / p16-ref-s42 (DIRECT)    (trained ON the 6.25M; all rows are members)

For each head: (1) transform the 6.25M substrate (lazy NormMemmap) -> FFR@0.1%xN on the 6.25M
sealed truth with the head's EXACT member/unseen split + collapse/fog/occupancy; (2) score the 27
jina registers -> per-register FFR + per-group (lang/social/en) means.

Deploy gate per small head vs the DIRECT ref: (proj gap <= 0.01 OR retention >= 0.97) AND no
register group mean drops > 0.005. Output: /data/latent-basemap/sandbox/p16-headsize-results.json
GPU script; queue behind other GPU work.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUBSTRATES = Path("/data/latent-basemap/substrates")
GATE_GAP = 0.01
GATE_RETENTION = 0.97
GATE_GROUP_DROP = 0.005


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import confirm_jina_mixture as J
    from image_map_pipeline import _norm
    from knobs_2m import quick_ffr_v2, quick_ffr_v2_split
    from jina_head_membership import member_mask_2m, member_mask_4m, span_bounds
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    try:
        from analysis_v2 import map_quality as _mq
    except Exception:
        _mq = None

    _, n6 = span_bounds()
    JINA6M_SUB, JINA6M_TRUTH = J.JINA6M_SUB, J.JINA6M_TRUTH
    knn6 = J.JINA6M_KNN if J.JINA6M_KNN.exists() else None
    if not (JINA6M_SUB.exists() and JINA6M_TRUTH.exists()):
        raise SystemExit(f"6.25M substrate/truth missing: {JINA6M_SUB} / {JINA6M_TRUTH}")

    HEADS = {
        "2M": {"model": SB / "jina-multi-2m/p15-baseline-s42/model.pt",
               "mask": "2m", "N": 2_000_000},
        "4M": {"model": SB / "jina-4m-head/champion-bs16k/model.pt",
               "mask": "4m", "N": 4_000_000},
        "direct-6250k": {"model": SB / "jina-multi-6m/p16-ref-s42/model.pt",
                         "mask": "all", "N": 6_250_000},
    }
    # P1.6 seed-43 replicate of the near-boundary 4M cell: point the 4M head at an alternate model.
    if os.environ.get("P16_4M_MODEL"):
        HEADS["4M"]["model"] = Path(os.environ["P16_4M_MODEL"])
    for h, info in HEADS.items():
        if not Path(info["model"]).exists():
            print(f"[warn] {h} model missing: {info['model']}", flush=True)

    # ---- register substrate/truth cache (shared across heads) ----
    reg_status = J._resolve_registers()
    reg_cache = {}

    def _register(reg):
        if reg not in reg_cache:
            info = reg_status.get(reg, {})
            sub, edges = Path(info.get("substrate", "")), Path(info.get("truth", ""))
            if not info.get("exists"):
                reg_cache[reg] = (None, None, None)
            else:
                x = _norm(np.asarray(np.load(sub, mmap_mode="r"), dtype=np.float32))
                reg_cache[reg] = (x, edges, int(x.shape[0]))
        return reg_cache[reg]

    def _mask_for(kind):
        if kind == "2m":
            return member_mask_2m(n6)
        if kind == "4m":
            return member_mask_4m(n6)
        return None  # "all" -> member_cutoff=n6

    results = {}
    for h, info in HEADS.items():
        mp = Path(info["model"])
        if not mp.exists():
            results[h] = {"status": f"SKIPPED (no model at {mp})"}
            continue
        print(f"\n=== HEAD {h} ({mp}) ===", flush=True)
        model = ParametricUMAP.load(str(mp), device="cuda")

        # (1) 6.25M projection
        t0 = time.time()
        X6 = J.NormMemmap(np.load(JINA6M_SUB, mmap_mode="r"))
        xy = np.asarray(model.transform(X6, batch_size=8192), dtype=np.float32)
        np.save(SB / f"p16-coords-{h}.npy", xy)
        mask = _mask_for(info["mask"])
        if mask is not None:
            sp = quick_ffr_v2_split(xy, JINA6M_TRUTH, n6, member_mask=mask,
                                    knn_indices_path=(knn6 if knn6 else None))
        else:
            sp = quick_ffr_v2_split(xy, JINA6M_TRUTH, n6, member_cutoff=n6,
                                    knn_indices_path=(knn6 if knn6 else None))
        proj = {"ffr_v2": float(sp["overall"]), "ffr_v2_member": sp["member"],
                "ffr_v2_unseen": sp["unseen"], "member_frac": sp["member_frac"],
                "n_member": sp["n_member"], "n_unseen": sp["n_unseen"],
                "truth_mode": sp["truth_mode"], "wall_s": round(time.time() - t0, 1)}
        if _mq is not None:
            try:
                q = _mq(xy)
                proj["collapse"] = float(q["collapse"]["r10_over_radius_times_sqrt_n"])
                proj["fog"] = float(q["fog"]["fog"])
                proj["occupancy"] = float(q["fog"]["occupied_bin_fraction"])
            except Exception as e:
                proj["map_quality_error"] = f"{type(e).__name__}: {e}"
        print(f"  proj 6.25M: overall {proj['ffr_v2']:.4f} member {proj['ffr_v2_member']} "
              f"unseen {proj['ffr_v2_unseen']} collapse {proj.get('collapse')} "
              f"fog {proj.get('fog')} ({proj['wall_s']}s)", flush=True)

        # (2) 27-register suite
        reg_ffr = {}
        for reg in J.REGISTERS:
            x, edges, n = _register(reg)
            if x is None:
                reg_ffr[reg] = None
                continue
            rxy = np.asarray(model.transform(x, batch_size=8192), dtype=np.float32)
            reg_ffr[reg] = float(quick_ffr_v2(rxy, edges, n))
        groups = {}
        for gname, regs in (("languages", J.LANG_REGISTERS), ("social", J.SOCIAL_REGISTERS),
                            ("en_base", J.EN_REGISTERS)):
            vals = [reg_ffr[r] for r in regs if reg_ffr.get(r) is not None and np.isfinite(reg_ffr[r])]
            groups[gname] = float(np.mean(vals)) if vals else None
        maximin = min((v for v in reg_ffr.values() if v is not None and np.isfinite(v)), default=None)
        results[h] = {"status": "OK", "model": str(mp), "N": info["N"],
                      "projection": proj, "register_ffr": reg_ffr,
                      "register_groups": groups, "register_maximin": maximin}
        print(f"  registers: groups {groups} maximin {maximin}", flush=True)
        del model

    # ---- head-size gate: 2M / 4M vs direct-6250k ----
    ref = results.get("direct-6250k", {})
    gate = {}
    if ref.get("status") == "OK":
        ref_proj = ref["projection"]["ffr_v2"]
        ref_groups = ref["register_groups"]
        for h in ("2M", "4M"):
            r = results.get(h, {})
            if r.get("status") != "OK":
                gate[h] = {"status": r.get("status", "MISSING")}
                continue
            hp = r["projection"]["ffr_v2"]
            hp_unseen = r["projection"]["ffr_v2_unseen"]
            gap = round(ref_proj - hp, 4)
            gap_unseen = (round(ref_proj - hp_unseen, 4) if hp_unseen is not None else None)
            retention = round(hp / ref_proj, 4) if ref_proj else None
            grp_deltas = {g: (round(r["register_groups"][g] - ref_groups[g], 4)
                              if r["register_groups"].get(g) is not None and ref_groups.get(g) is not None
                              else None) for g in ref_groups}
            grp_ok = all(d is None or d >= -GATE_GROUP_DROP for d in grp_deltas.values())
            proj_ok = (gap <= GATE_GAP) or (retention is not None and retention >= GATE_RETENTION)
            passes = bool(proj_ok and grp_ok)
            # decisive-cell closeness: within 2x of either boundary -> flag for seed-43 replicate
            near_boundary = (abs(gap - GATE_GAP) <= GATE_GAP) or (
                retention is not None and abs(retention - GATE_RETENTION) <= (1 - GATE_RETENTION))
            gate[h] = {"proj_gap": gap, "proj_gap_unseen": gap_unseen, "retention": retention,
                       "proj_ok": proj_ok, "group_deltas": grp_deltas, "groups_ok": grp_ok,
                       "passes_deploy_gate": passes, "near_boundary_needs_seed43": bool(near_boundary)}
            print(f"\n[gate {h}] gap {gap} (unseen {gap_unseen}) retention {retention} "
                  f"proj_ok {proj_ok} groups_ok {grp_ok} -> {'PASS' if passes else 'FAIL'}"
                  f"{' (near boundary: seed-43 the decisive cell)' if near_boundary else ''}", flush=True)

    out = SB / os.environ.get("P16_OUT", "p16-headsize-results.json")
    out.write_text(json.dumps({
        "schema": "p16-headsize-2026-08-29", "seed": 42,
        "gate": {"proj_gap_max": GATE_GAP, "retention_min": GATE_RETENTION,
                 "group_drop_max": GATE_GROUP_DROP,
                 "rule": "(proj gap<=0.01 OR retention>=0.97) AND no register group mean drops>0.005; "
                         "seed-43 replicate on a head only if within 2x of a boundary"},
        "n6": n6, "heads": results, "head_size_gate": gate,
        "note": ("head SIZE is the sole variable (composition matched to the 6.25M, champion recipe "
                 "rankneg=25% of N). member/unseen use EXACT masks (2M old-block prefixes, 4M "
                 "member_indices, direct=all). unseen is the honest OOS number; overall is the "
                 "map-to-map comparison vs the direct 6.25M reference."),
    }, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

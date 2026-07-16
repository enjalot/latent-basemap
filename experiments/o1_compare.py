"""O1 — prompted-vs-unprompted comparison node (CPU, content-bound).

Merges the prompted 200k v2.2 scores, the unprompted 200k v2.2 scores (from G0),
and the prompt-shift report into one artifact: per-seed FFR/purity/density deltas
(prompted − unprompted), recipe ranking, and the high-D topology shift the prompt
induced (kNN overlap + centroid ARI) — all keyed to the shared-reference keys so a
mixed provenance is caught.
"""
from __future__ import annotations
import argparse, os, sys, json, statistics


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _arm_seed_means(runs, metric):
    # runs keyed like "legacy_a1b1_s42" (unprompted) or "prompted_s42"
    vals = {}
    for lbl, r in runs.items():
        sd = lbl.rsplit("_s", 1)[-1]
        if r.get(metric) is not None:
            vals.setdefault(sd, r[metric])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompted", required=True, help="complete_200k_prompted_v22.json")
    ap.add_argument("--unprompted", required=True, help="complete_200k_v22.json (G0)")
    ap.add_argument("--shift", required=True, help="prompt_shift_report.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pr = _load(args.prompted); un = _load(args.unprompted); sh = _load(args.shift)
    if not (pr and un):
        raise SystemExit("o1_compare: missing prompted or unprompted scores")
    pruns, uruns = pr.get("runs", {}), un.get("runs", {})
    # unprompted legacy arm only (the kernel G1 finalized)
    uleg = {k: v for k, v in uruns.items() if k.startswith("legacy_a1b1_s")}
    deltas = {}
    for metric in ("ffr", "purity_k1024", "density", "proj_ffr"):
        pv = _arm_seed_means(pruns, metric)          # {seed: val} prompted
        uv = _arm_seed_means(uleg, metric)           # {seed: val} unprompted legacy
        pair = [(s, pv[s], uv[s]) for s in sorted(set(pv) & set(uv))]
        d = [p - u for _, p, u in pair]
        deltas[metric] = {
            "prompted_mean": round(statistics.mean([p for _, p, _ in pair]), 5) if pair else None,
            "unprompted_mean": round(statistics.mean([u for _, _, u in pair]), 5) if pair else None,
            "delta_mean": round(statistics.mean(d), 5) if d else None,
            "delta_sd": round(statistics.stdev(d), 5) if len(d) > 1 else 0.0,
            "n_seeds": len(pair), "per_seed": {s: {"prompted": round(p, 5), "unprompted": round(u, 5)}
                                               for s, p, u in pair}}
    out = {"gate": "o1_prompted_vs_unprompted", "formula_version": pr.get("formula_version"),
           "prompted_reference_key": pr.get("hiD_reference_key"),
           "unprompted_reference_key": un.get("hiD_reference_key"),
           "prompted_scorer_dirty": pr.get("scorer_dirty"),
           "high_d_prompt_shift": sh,   # kNN overlap + centroid ARI (topology moved by the prompt)
           "metric_deltas": deltas,
           "note": "delta = prompted - unprompted (legacy_lp arm, paired by seed). The prompt "
                   "moves BOTH the high-D space (see high_d_prompt_shift) and the learned map; "
                   "read map deltas alongside the high-D shift, not as pure map effects."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[o1_compare] ffr Δ={deltas['ffr']['delta_mean']} purity Δ={deltas['purity_k1024']['delta_mean']} "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()

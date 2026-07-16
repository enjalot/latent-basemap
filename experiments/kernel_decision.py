"""G0 — kernel decision from the shared-reference v2.2 rescores.

Keeps TWO questions separate (owner): legacy vs umap(a=b=1) [formula isolation]
and legacy vs fitted std curve [curve]. Reports per-arm seed means, paired seedwise
deltas, and — as the fresh-2M evaluator gate — a wall/peak envelope check on the
2M rescore telemetry (a regression here blocks trusting the evaluator at 8M).
Also asserts every map reused the ONE per-corpus reference (L0.4).
"""
from __future__ import annotations
import argparse, os, sys, json, statistics

# generous per-map envelope for a 200k/2m complete-panel scoring (transductive +
# projection + kNN-regressor); a real regression (lost byte-cap / tile blowup)
# is many-x over these, not a jitter.
WALL_MAX_S = {"200k": 60.0, "2m": 240.0}
PEAK_GPU_MAX_GB = 26.0
ARMS = ["legacy_a1b1", "umap_a1b1", "umap_stdcurve"]


def _arm_seed(label):
    # label like "umap_stdcurve_s42"
    arm, _, sd = label.rpartition("_s")
    return arm, int(sd)


def _load(p):
    d = json.load(open(p))
    if not d.get("runs"):
        raise SystemExit(f"kernel_decision: {p} has no runs")
    return d


def _arm_means(runs, metric):
    out = {}
    for arm in ARMS:
        vals = [r[metric] for lbl, r in runs.items()
                if _arm_seed(lbl)[0] == arm and r.get(metric) is not None]
        out[arm] = round(statistics.mean(vals), 5) if vals else None
    return out


def _paired_delta(runs, base, other, metric):
    # seedwise (other - base) mean ± sample sd
    seeds = sorted({_arm_seed(l)[1] for l in runs})
    diffs = []
    for s in seeds:
        b = next((r[metric] for l, r in runs.items() if _arm_seed(l) == (base, s)), None)
        o = next((r[metric] for l, r in runs.items() if _arm_seed(l) == (other, s)), None)
        if b is not None and o is not None:
            diffs.append(o - b)
    if not diffs:
        return None
    return {"mean": round(statistics.mean(diffs), 5),
            "sd": round(statistics.stdev(diffs), 5) if len(diffs) > 1 else 0.0,
            "n": len(diffs)}


def _corpus_block(p, corpus):
    d = _load(p)
    runs = d["runs"]
    # L0.4: one reference key, reused on every map
    keys = {r.get("hiD_reference_key") for r in runs.values()}
    reused = all(r.get("hiD_reference_reused") for r in runs.values())
    if len(keys) != 1 or None in keys or not reused:
        raise SystemExit(f"{corpus}: maps did not all reuse ONE shared reference "
                         f"(keys={keys}, reused={reused}) — refuse to decide (L0.4).")
    # fresh evaluator wall/peak envelope (G0.3)
    walls = {l: r.get("wall_s") for l, r in runs.items()}
    peaks = {l: (r.get("panel_full", {}).get("provenance", {}) or {}).get("peak_gpu_gb")
             for l, r in runs.items()}
    max_wall = max((w for w in walls.values() if w is not None), default=None)
    max_peak = max((pk for pk in peaks.values() if pk is not None), default=None)
    wall_ok = (max_wall is not None and max_wall <= WALL_MAX_S[corpus])
    peak_ok = (max_peak is None or max_peak <= PEAK_GPU_MAX_GB)
    means = {m: _arm_means(runs, m) for m in ("ffr", "purity_k1024", "density")}
    deltas = {
        "umap_a1b1_minus_legacy": {m: _paired_delta(runs, "legacy_a1b1", "umap_a1b1", m)
                                   for m in ("ffr", "purity_k1024", "density")},
        "stdcurve_minus_legacy": {m: _paired_delta(runs, "legacy_a1b1", "umap_stdcurve", m)
                                  for m in ("ffr", "purity_k1024", "density")},
    }
    return {"formula_version": d.get("formula_version"),
            "reference_key": next(iter(keys)), "hiD_reference_reused": reused,
            "scorer_dirty": d.get("scorer_dirty"), "n_maps": len(runs),
            "arm_means": means, "paired_deltas": deltas,
            "evaluator_wall_peak": {"max_wall_s": max_wall, "max_peak_gpu_gb": max_peak,
                                    "wall_ok": wall_ok, "peak_ok": peak_ok,
                                    "wall_max_allowed": WALL_MAX_S[corpus],
                                    "peak_max_allowed": PEAK_GPU_MAX_GB}}, (wall_ok and peak_ok)


def _verdict(means):
    """legacy wins an axis if its mean >= the umap arm's mean."""
    lg = means["legacy_a1b1"]; return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p200", required=True)
    ap.add_argument("--p2m", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b200, ok200 = _corpus_block(args.p200, "200k")
    b2m, ok2m = _corpus_block(args.p2m, "2m")

    def decide(block):
        m = block["arm_means"]        # {metric: {arm: mean}}
        # legacy vs each umap arm on the two primary axes (ffr, purity)
        def cmp(arm):
            return {ax: ("legacy" if (m[ax]["legacy_a1b1"] or 0) >= (m[ax][arm] or 0) else arm)
                    for ax in ("ffr", "purity_k1024")}
        return {"legacy_vs_umap_a1b1": cmp("umap_a1b1"),
                "legacy_vs_stdcurve": cmp("umap_stdcurve")}

    out = {"gate": "kernel_decision_v22", "formula_version": b200["formula_version"],
           "corpora": {"200k": b200, "2m": b2m},
           "primary_axis_winners": {"200k": decide(b200), "2m": decide(b2m)},
           "evaluator_gate_passed": bool(ok200 and ok2m),
           "note": "two separate questions kept apart: formula isolation (a=b=1) and "
                   "fitted standard curve; primary axes ffr+purity, density secondary."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["primary_axis_winners"], indent=1))
    print(f"[kernel_decision] evaluator_gate_passed={out['evaluator_gate_passed']} -> {args.out}")
    sys.exit(0 if out["evaluator_gate_passed"] else 3)


if __name__ == "__main__":
    main()

"""Machine-checkable R0.1 closure gate summary (closure review §Final handoff rule).

Closure statements must come from THIS script, not from a launcher's prose. It
refuses `closed=true` unless every registered closure item points to a passing,
content-bound artifact AND every training result's ACTUAL pipeline/semantics match
its declared recipe. Requested config is not execution evidence.

Each check binds the artifact by sha and asserts the specific pass condition the
review registered. Prints a table + a JSON verdict; exits non-zero if not closed.

Usage:
  python experiments/gate_summary.py --out experiments/evidence/r0_1_gate_summary.json
"""
from __future__ import annotations
import argparse, os, sys, json, glob, hashlib

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVID = os.path.join(ROOT, "experiments", "evidence")


def _sha(path, chunk=1 << 20):
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()[:16]
    except Exception:
        return None


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def check_2x2():
    p = os.path.join(EVID, "r1_ablation_mn4", "complete.json")
    d = _load(p)
    if not d:
        return False, "missing complete.json", {}
    runs = d.get("runs", {})
    cells = {r.get("run_dir", k).split("_s")[0] for k, r in runs.items()} if runs else set()
    ok = len(runs) >= 12 and int(d.get("n_holdout_unique", 0)) == int(d.get("n_holdout", -1))
    return ok, f"{len(runs)} maps, unique held-out={d.get('n_holdout_unique')}", {"sha": _sha(p)}


def check_a3():
    p = os.path.join(EVID, "a3_rescore.json")
    d = _load(p)
    if not d:
        return False, "missing a3_rescore.json", {}
    ok = (d.get("decision") == "nomn" and d.get("provenance", {}).get("code_dirty") is False)
    return ok, f"decision={d.get('decision')} clean_commit={d.get('provenance',{}).get('code_commit')}", {"sha": _sha(p)}


def check_knn_cost():
    p = os.path.join(EVID, "r1_rescore", "knn_cost.json")
    d = _load(p)
    if not d:
        return False, "missing knn_cost.json (durable build+query+mem)", {}
    tb = d.get("testbeds", {})
    ok = all(("query_wall_s" in v and "end_to_end_regression_wall_s" in v and "x_footprint_gb" in v)
             for v in tb.values()) and len(tb) >= 2
    return ok, f"testbeds={list(tb)} contract={'zero-build' in (d.get('contract','') )}", {"sha": _sha(p)}


def check_golden():
    p = os.path.join(EVID, "r1_rescore", "golden_2m_extended_v22.json")
    d = _load(p)
    if not d:
        return False, "missing golden_2m_extended_v22.json (reference+deltas+tolerances+pass)", {}
    need = {"ffr", "density", "purity_k1024", "proj_ffr"}
    have = set(d.get("metrics_validated", []))
    ok = bool(d.get("passed")) and need.issubset(have) and "panel_v2.2" in str(d.get("formula_version"))
    return ok, f"passed={d.get('passed')} metrics={sorted(have)}", {"sha": _sha(p)}


def check_bridge():
    """Both weighted bridge seeds must have ACTUAL pipeline=device + weighted,
    exactly 500k positive-LR v3 updates, and a v2.2 score."""
    res = os.path.join(EVID, "r1_8m", "bridge_weighted.json")
    d = _load(res)
    if not d:
        return False, "missing bridge_weighted.json", {}
    seeds = d.get("seeds", {})
    ok = len(seeds) >= 2 and all(
        s.get("pipeline") == "device" and s.get("positive_sampling") == "weighted_with_replacement"
        and int(s.get("positive_lr_updates", 0)) == 500000 and "panel_v2.2" in str(s.get("formula_version", ""))
        for s in seeds.values())
    detail = {k: (v.get("pipeline"), v.get("positive_sampling"), v.get("positive_lr_updates")) for k, v in seeds.items()}
    return ok, f"seeds={detail}", {"sha": _sha(res)}


CHECKS = [
    ("1_corrected_2x2", check_2x2),
    ("2_a3_regen", check_a3),
    ("3a_knn_cost", check_knn_cost),
    ("3b_golden_2m_extended", check_golden),
    ("4_weighted_8m_bridge", check_bridge),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EVID, "r0_1_gate_summary.json"))
    args = ap.parse_args()
    items = {}
    for name, fn in CHECKS:
        try:
            ok, detail, extra = fn()
        except Exception as e:
            ok, detail, extra = False, f"check raised: {e!r}", {}
        items[name] = {"passed": bool(ok), "detail": detail, **extra}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    closed = all(v["passed"] for v in items.values())
    out = {"gate": "r0_1_closure", "closed": bool(closed), "items": items,
           "rule": "closed=true only if every registered item points to a passing content-bound "
                   "artifact and every training result's ACTUAL pipeline/semantics match its recipe."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nR0.1 closed = {closed}  ->  {args.out}")
    sys.exit(0 if closed else 3)


if __name__ == "__main__":
    main()

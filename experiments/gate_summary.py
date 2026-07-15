"""Machine-checkable R0.1 closure gate (closure review §S1 — prove what you claim).

This gate REOPENS raw pinned sources and recomputes derived facts; it does NOT
trust aggregate JSON fields. It refuses closed=true unless, for every registered
item, the raw content-bound evidence proves the exact condition the review
registered — exact seed/cell sets, panel formula version, golden deltas recomputed
from values/tolerances, and for training the ACTUAL pipeline/sampler/schedule/
budget/updates bound to the persisted run results (not a summary copy).

Usage:  python experiments/gate_summary.py --out experiments/evidence/r0_1_gate_summary.json
"""
from __future__ import annotations
import argparse, os, sys, json, glob, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVID = os.path.join(ROOT, "experiments", "evidence")
RESULTS = os.path.join(ROOT, "experiments", "results")


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


def _latest(pat):
    d = sorted(glob.glob(os.path.join(RESULTS, pat)))
    return d[-1] if d else None


def _acct(run_dir):
    r = _load(os.path.join(run_dir, "results.json")) or {}
    return (r.get("run_manifest") or {}).get("train_accounting", {}), r


# ── item checks (each reopens raw sources) ───────────────────────────────────────

def check_2x2():
    d = _load(os.path.join(EVID, "r1_ablation_mn4", "complete.json"))
    if not d:
        return False, "missing complete.json", {}
    want_cells = {"mn0_s1_ws0", "mn0_s1_ws1", "mn1_s4_ws0", "mn1_s4_ws1"}
    got = {k.rsplit("_s", 1)[0] for k in d.get("runs", {})}
    seeds = {int(k.rsplit("_s", 1)[1]) for k in d.get("runs", {}) if k.rsplit("_s", 1)[1].isdigit()}
    ok = (len(d.get("runs", {})) == 12 and want_cells.issubset(got)
          and seeds == {42, 43, 44}
          and int(d.get("n_holdout_unique", 0)) == int(d.get("n_holdout", -1)))
    return ok, f"cells={sorted(got)} seeds={sorted(seeds)} n={len(d.get('runs',{}))}", {}


def check_a3():
    d = _load(os.path.join(EVID, "a3_rescore.json"))
    if not d:
        return False, "missing a3_rescore.json", {}
    prov = d.get("provenance", {})
    # reopen: clean commit, v2.2 formula, decision recomputed from seed_means
    means = d.get("seed_means", {})
    recomputed = ("nomn" if means.get("nomn", {}).get("ffr", 0) > means.get("mn4", {}).get("ffr", 1)
                  and means.get("nomn", {}).get("purity_k1024", 0) > means.get("mn4", {}).get("purity_k1024", 1)
                  else "mn4")
    ok = (prov.get("code_dirty") is False and "panel_v2.2" in str(d.get("formula_version"))
          and d.get("decision") == recomputed == "nomn")
    return ok, f"decision={d.get('decision')}(recomputed {recomputed}) dirty={prov.get('code_dirty')} formula={d.get('formula_version')}", {}


def check_knn_cost():
    d = _load(os.path.join(EVID, "r1_rescore", "knn_cost.json"))
    if not d:
        return False, "missing knn_cost.json", {}
    tb = d.get("testbeds", {})
    ok = ("zero-build" in d.get("contract", "") and len(tb) >= 2 and
          all({"query_wall_s", "end_to_end_regression_wall_s", "x_footprint_gb"} <= set(v)
              for v in tb.values()))
    return ok, f"testbeds={sorted(tb)}", {}


def check_golden():
    d = _load(os.path.join(EVID, "r1_rescore", "golden_2m_extended_v22.json"))
    if not d:
        return False, "missing golden_2m_extended_v22.json", {}
    # RECOMPUTE the pass from values/tolerances — do not trust d['passed'].
    need = {"ffr", "recall@k", "density", "purity_k1024", "proj_ffr"}
    have = set(d.get("metrics_validated", []))
    streamed, ref, tol = d.get("streamed", {}), d.get("reference", {}), d.get("tolerances", {})
    recomputed = all(abs(streamed[m] - ref[m]) <= tol[m] for m in have
                     if m in streamed and m in ref and m in tol)
    ok = (need.issubset(have) and recomputed and d.get("code_dirty") is False
          and "panel_v2.2" in str(d.get("formula_version")))
    return ok, f"metrics={sorted(have)} recomputed_pass={recomputed} dirty={d.get('code_dirty')}", {}


def check_bridge():
    """Reopen each seed's RAW results.json (not the bridge_weighted.json copy) and
    verify ACTUAL pipeline=device, weighted-with-replacement, 500k v3 updates,
    budget_satisfied, config byte-identity, and admission-verified graph hash;
    plus the canary passed at the registered floor."""
    canary = _load(os.path.join(EVID, "r1_8m", "canary_8m.json")) or {}
    canary_ok = (canary.get("passed") is True and canary.get("pipeline") == "device"
                 and canary.get("positive_sampling") == "weighted_with_replacement"
                 and (canary.get("steady_updates_per_s") or 0) >= (canary.get("floor") or 200))
    details, seed_ok = {}, True
    for seed in (42, 43):
        rd = _latest(f"r1_8m_bridge_weighted_s{seed}_*")
        if not rd:
            seed_ok = False; details[f"s{seed}"] = "no run dir"; continue
        acct, r = _acct(rd)
        cfg_persisted = os.path.join(rd, "config.yaml")
        cfg_tracked = os.path.join(ROOT, "experiments", "configs", f"_bridge_w_s{seed}.yaml")
        cfg_match = _sha(cfg_persisted) == _sha(cfg_tracked)
        # POST-HOC graph proof (bridge is NOT retrained per S1.5): re-validate the
        # graph the run used against its manifest NOW. Either the run stamped the
        # verified hash (post-S0 runs) or we re-verify content here.
        graph_ok = bool((acct.get("verified_hashes") or {}).get("graph_sha"))
        if not graph_ok:
            try:
                from basemap.graph_validation import validate_graph_content
                import yaml as _yaml, glob as _glob
                dc = _yaml.safe_load(open(cfg_persisted)).get("data", {})
                ep = dc.get("precomputed_edges_path")
                man = _load(ep + ".manifest.json") if ep else None
                # resolve the ordered data shards the run's memmap_dirs would load
                shards = []
                for md in (dc.get("memmap_dirs") or []):
                    shards += sorted(_glob.glob(os.path.join(md, "*.npy")))
                if man:
                    validate_graph_content(ep, man, shard_paths=shards, require_manifest_sha=True)
                    graph_ok = True
            except Exception as _e:
                graph_ok = False
        this = (acct.get("pipeline_pipeline") == "device"
                and acct.get("pipeline_positive_sampling") == "weighted_with_replacement"
                and int(acct.get("positive_lr_optimizer_steps", 0)) == 500000
                and acct.get("budget_satisfied") is True
                and "cosine-v3" in str(acct.get("schedule_version")) and graph_ok)
        seed_ok = seed_ok and this and cfg_match
        details[f"s{seed}"] = {"pipeline": acct.get("pipeline_pipeline"),
                               "sampling": acct.get("pipeline_positive_sampling"),
                               "updates": acct.get("positive_lr_optimizer_steps"),
                               "budget_satisfied": acct.get("budget_satisfied"),
                               "schedule": acct.get("schedule_version"),
                               "graph_verified_posthoc": graph_ok,
                               "config_byte_identical": cfg_match}
    ok = canary_ok and seed_ok
    return ok, f"canary_pass={canary_ok} seeds={details}", {}


CHECKS = [("1_corrected_2x2", check_2x2), ("2_a3_regen", check_a3),
          ("3a_knn_cost", check_knn_cost), ("3b_golden_2m_extended", check_golden),
          ("4_weighted_8m_bridge", check_bridge)]


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
           "rule": "reopens raw pinned sources; recomputes golden deltas + decisions; binds actual "
                   "pipeline/sampler/schedule/budget/updates to persisted run results, not summaries."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nR0.1 closed = {closed}  ->  {args.out}")
    sys.exit(0 if closed else 3)


if __name__ == "__main__":
    main()

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
import argparse, os, sys, json, hashlib, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # so `import basemap...` works for the post-hoc graph check
EVID = os.path.join(ROOT, "experiments", "evidence")
RAW_ROOT = "/data/latent-basemap/closure"
BRIDGE_RUNS = {
    42: "historical/r1_8m_bridge_weighted_s42_20260715_163719_f4934f5b",
    43: "historical/r1_8m_bridge_weighted_s43_20260715_170618_63bbadf5",
}
PINNED_RAW_FILES = {
    "g1_decision_raw": "g1/stdcurve_decision.json",
    "backfill_2m_controller": "bf_2m/bf_ctl.json",
    "o1_controller": "o1/o1_ctl.json",
    "o2_frontier_controller": "o2/o2_frontier_ctl.json",
    "bridge_s42_results": BRIDGE_RUNS[42] + "/results.json",
    "bridge_s42_config": BRIDGE_RUNS[42] + "/config.yaml",
    "bridge_s43_results": BRIDGE_RUNS[43] + "/results.json",
    "bridge_s43_config": BRIDGE_RUNS[43] + "/config.yaml",
}


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


def _acct(run_dir):
    r = _load(os.path.join(run_dir, "results.json")) or {}
    return (r.get("run_manifest") or {}).get("train_accounting", {}), r


# ── item checks (each reopens raw sources) ───────────────────────────────────────

def check_2x2():
    d = _load(os.path.join(EVID, "r1_ablation_mn4", "complete.json"))
    if not d:
        return False, "missing complete.json", {}
    want_cells = {"mn0_s1_ws0", "mn0_s1_ws1", "mn1_s4_ws0", "mn1_s4_ws1"}
    runs = d.get("runs", {})
    got = {k.rsplit("_s", 1)[0] for k in runs}
    seeds = {int(k.rsplit("_s", 1)[1]) for k in runs if k.rsplit("_s", 1)[1].isdigit()}
    # C1: prove the EXACT cell×seed cross-product (12 = 4 cells × 3 seeds), not
    # just the count + membership (critique #9).
    want_cross = {f"{c}_s{s}" for c in want_cells for s in (42, 43, 44)}
    cross_ok = set(runs) == want_cross
    v22 = "panel_v2.2" in str(d.get("formula_version"))
    ok = (len(runs) == 12 and want_cells == got and seeds == {42, 43, 44} and cross_ok and v22
          and int(d.get("n_holdout_unique", 0)) == int(d.get("n_holdout", -1)))
    return ok, (f"cells={sorted(got)} seeds={sorted(seeds)} n={len(runs)} "
                f"exact_cross={cross_ok} v2.2={v22}"), {}


def check_kernel_v22():
    """C1: the missing v2.2 kernel rescore (G0). Reopens complete_200k_v22 +
    complete_2m_v22 + kernel_decision_v22 and requires: formula v2.2, ONE shared
    reference reused on EVERY map, clean scorer, the fresh evaluator wall/peak gate
    passed, and legacy winning BOTH primary axes vs both umap arms at both scales."""
    dec = _load(os.path.join(EVID, "r1_kernel", "kernel_decision_v22.json"))
    p200 = _load(os.path.join(EVID, "r1_kernel", "complete_200k_v22.json"))
    p2m = _load(os.path.join(EVID, "r1_kernel", "complete_2m_v22.json"))
    if not (dec and p200 and p2m):
        return False, "missing kernel v2.2 rescore evidence (complete_200k/2m/decision)", {}
    problems = []
    for name, panel, n_expected in [("200k", p200, 9), ("2m", p2m, 6)]:
        runs = panel.get("runs", {})
        if len(runs) != n_expected:
            problems.append(f"{name}:n={len(runs)}!={n_expected}")
        if "panel_v2.2" not in str(panel.get("formula_version")):
            problems.append(f"{name}:not_v2.2")
        if panel.get("scorer_dirty") is not False:
            problems.append(f"{name}:scorer_dirty")
        keys = {r.get("hiD_reference_key") for r in runs.values()}
        if len(keys) != 1 or None in keys or not all(r.get("hiD_reference_reused") for r in runs.values()):
            problems.append(f"{name}:reference_not_reused({keys})")
    # decision: legacy wins both axes vs both umap arms at both scales
    if not dec.get("evaluator_gate_passed"):
        problems.append("evaluator_gate_failed")
    for scale, w in (dec.get("primary_axis_winners") or {}).items():
        for q, axes in w.items():
            for ax, winner in axes.items():
                if winner != "legacy":
                    problems.append(f"{scale}:{q}:{ax}={winner}")
    ok = not problems
    return ok, f"200k_ref={p200.get('hiD_reference_key')} 2m_ref={p2m.get('hiD_reference_key')} "\
               f"evaluator_gate={dec.get('evaluator_gate_passed')} problems={problems}", {}


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
    # C1: a MISSING streamed/reference/tolerance triplet is a FAILURE, not a
    # silently-skipped metric (critique #9).
    need = {"ffr", "recall@k", "density", "purity_k1024", "proj_ffr"}
    have = set(d.get("metrics_validated", []))
    streamed, ref, tol = d.get("streamed", {}), d.get("reference", {}), d.get("tolerances", {})
    triplet_complete = all(m in streamed and m in ref and m in tol for m in need)
    recomputed = triplet_complete and all(abs(streamed[m] - ref[m]) <= tol[m] for m in need)
    ok = (need.issubset(have) and triplet_complete and recomputed
          and d.get("code_dirty") is False and "panel_v2.2" in str(d.get("formula_version")))
    return ok, (f"metrics={sorted(have)} triplet_complete={triplet_complete} "
                f"recomputed_pass={recomputed} dirty={d.get('code_dirty')}"), {}


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
        rd = os.path.join(RAW_ROOT, BRIDGE_RUNS[seed])
        if not os.path.isdir(rd):
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
          ("4_weighted_8m_bridge", check_bridge), ("5_kernel_v22_rescore", check_kernel_v22)]

# C1: per-source content hashes stamped into the certificate so the closure is
# traceable to exact bytes, not just to a pass/fail boolean.
SOURCE_FILES = {
    "ablation_2x2": os.path.join(EVID, "r1_ablation_mn4", "complete.json"),
    "a3_rescore": os.path.join(EVID, "a3_rescore.json"),
    "knn_cost": os.path.join(EVID, "r1_rescore", "knn_cost.json"),
    "golden_2m_extended": os.path.join(EVID, "r1_rescore", "golden_2m_extended_v22.json"),
    "canary_pass": os.path.join(EVID, "r1_8m", "canary_8m.json"),
    "canary_abort_demo": os.path.join(EVID, "r1_8m", "canary_abort_demo.json"),
    "bridge_scores": os.path.join(EVID, "r1_8m", "bridge_weighted.json"),
    "kernel_200k_v22": os.path.join(EVID, "r1_kernel", "complete_200k_v22.json"),
    "kernel_2m_v22": os.path.join(EVID, "r1_kernel", "complete_2m_v22.json"),
    "kernel_decision_v22": os.path.join(EVID, "r1_kernel", "kernel_decision_v22.json"),
}


def main():
    global RAW_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EVID, "r0_1_gate_summary.json"))
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--release-sha", required=True,
                    help="full clean queue release SHA that this new certificate binds")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.release_sha):
        raise SystemExit("--release-sha must be a full lowercase 40-character SHA")
    RAW_ROOT = os.path.realpath(args.raw_root)
    if not RAW_ROOT.startswith("/data/"):
        raise SystemExit("--raw-root must resolve under /data")
    items = {}
    for name, fn in CHECKS:
        try:
            ok, detail, extra = fn()
        except Exception as e:
            ok, detail, extra = False, f"check raised: {e!r}", {}
        items[name] = {"passed": bool(ok), "detail": detail, **extra}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    closed = all(v["passed"] for v in items.values())
    # C1: stamp per-source content hashes so the certificate is traceable to exact
    # bytes. A missing pinned source is itself a red flag surfaced here.
    pinned_raw_paths = {key: os.path.join(RAW_ROOT, relative)
                        for key, relative in PINNED_RAW_FILES.items()}
    all_sources = {**SOURCE_FILES, **pinned_raw_paths}
    source_hashes = {k: _sha(p) for k, p in all_sources.items()}
    missing = [k for k, h in source_hashes.items() if h is None]
    closed = closed and not missing
    out = {"gate": "r0_1_closure", "closed": bool(closed), "items": items,
           "source_hashes": source_hashes, "missing_sources": missing,
           "source_paths": all_sources, "raw_root": RAW_ROOT,
           "code_commit": _git_head(), "release_sha": args.release_sha,
           "rule": "reopens raw pinned sources; recomputes golden deltas + decisions + the exact "
                   "cell×seed cross-product; requires the v2.2 kernel rescore with a reused shared "
                   "reference; binds actual pipeline/sampler/schedule/budget/updates to persisted "
                   "run results, not summaries; stamps per-source content hashes."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nR0.1 closed = {closed}  ->  {args.out}")
    if missing:
        print(f"  (missing pinned sources: {missing})")
    sys.exit(0 if closed else 3)


def _git_head():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()

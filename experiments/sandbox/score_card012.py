"""Card012 scorer (per card012-prereg + scoring review). Reuses the validated score_card009 machinery.
DEVELOPMENT movement panel = the ORIGINAL card006_confirm_bank (wholly-unseen, excluded from the 1M pool)
— the fresh reserved 10K stays UNTOUCHED for a later selected-candidate confirmation. Heads: frozen_t0 /
uniform / error_directed / anchored. Recall on original eval-common-v2 (all 9 cohorts). R0=33.6717.
Primary movement = ALIGNED (rigid-fit on original active anchors); native reported alongside + used for the
SEPARATE deployment boolean. Gate: error_directed aligned p99 <= .70 x uniform with paired p99 CI<0;
fraction moving >.05 not increased; old-content losses vs frozen <=.005 mean/<=.01 worst; arriving recall
within .005 of BOTH uniform AND anchored; B250 gain over frozen positive CI. CPU. Usage: score_card012.py
"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "4"
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import frame
from score_card009 import R0, BUDGETS, sha, write_json, project, recall, stats, paired, ParametricUMAP

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"; OUT = OC / "card012-scoring"
TD = SB / "dino-arrival-t0"; SEAL = Path("/data2/monet/eval-common-v2"); SUBD = Path("/data/latent-basemap/substrates")
TRAIN = SB / "card012-train"
OLD = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
ARR = ["synthetic-flux-klein", "synthetic-flux-schnell", "synthetic-z-image"]
HEADS = {"frozen_t0": (TD / "champion-bs16k/model.pt", TD / "champion-bs16k/coordinates.npy"),
         "uniform": (TRAIN / "model-uniform.pt", TRAIN / "coords-uniform.npy"),
         "error_directed": (TRAIN / "model-error_directed.pt", TRAIN / "coords-error_directed.npy"),
         "anchored": (TD / "updates/model-anchored.pt", TD / "updates/coords-anchored.npy")}


def all_finite(a):
    return all(bool(np.isfinite(a[i:i+65536]).all()) for i in range(0, len(a), 65536))


def quality_checks(old_preserved, vs_uniform, vs_anchored, gain_ci_low):
    """Shared production guards, used by both research and deployment decisions."""
    return {"old_content_preserved": bool(old_preserved),
            "arriving_within_005_of_uniform_AND_anchored": bool(vs_uniform and vs_anchored),
            "B250_gain_over_frozen_ci_positive": bool(np.isfinite(gain_ci_low) and gain_ci_low > 0)}


def deployment_checks(mean, p99, quality):
    checks = {"native_mean_le_001": bool(np.isfinite(mean) and mean <= .01),
              "native_p99_le_005": bool(np.isfinite(p99) and p99 <= .05), **quality}
    return {"checks": checks, "DEPLOY_ELIGIBLE": all(checks.values())}


def equal_cohort_means(recalls):
    """Include every instrument cohort, including diffusion-aesthetic-4k."""
    return {b: float(np.mean(list(by_source.values()))) for b, by_source in recalls.items()}


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def prescoring_validator():
    """Fail-closed BEFORE scoring: both arms exact 140K, 3 EXPECTED refresh steps with DISTINCT banks +
    ordered-row digests, shared warm-start, admitted constant LR, finite coords, current (not stale)."""
    C = {}
    mans = {a: json.loads((TRAIN / f"manifest-{a}.json").read_text()) for a in ("uniform", "error_directed")}
    C["both_140k"] = all(m["executed_steps"] == 140000 for m in mans.values())
    C["shared_warm"] = len({m["warm_start_sha"] for m in mans.values()}) == 1
    C["constant_lr"] = all(abs(m["lr_used_min"] - 1e-4) < 1e-12 and abs(m["lr_used_max"] - 1e-4) < 1e-12 for m in mans.values())
    expected_steps = [35000, 70000, 105000]
    anchor = TD / 'anchor.npz'
    checker = ParametricUMAP.load(str(HEADS['frozen_t0'][0]), device='cpu')
    checker.anchor_ids_path = str(anchor); checker.anchor_holdout_fraction = .10
    aid, _, hid, anchor_stats = checker._load_sparse_anchor_landmarks(2400000, random_state=42)
    C['original_anchor_split'] = (np.array_equal(aid, np.load(TD/'updates/anchor_active_ids-anchored.npy'))
                                  and np.array_equal(hid, np.load(TD/'updates/anchor_holdout_ids-anchored.npy')))
    warm = state_digest(checker.model.state_dict()); del checker
    C['original_T0_warm'] = all(m['warm_start_sha'] == warm for m in mans.values())
    ok_ref = True; ok_distinct = True; ok_ordered = True
    for a, m in mans.items():
        admission_path = TRAIN / f'admission-{a}.json'
        admission = json.loads(admission_path.read_text())
        identity = admission['identity']; started = admission_path.stat().st_mtime
        C[f'{a}_admission'] = (admission['steps'] == 140000 and admission['arm'] == a
            and admission['seed'] == 42 and admission['lr'] == 1e-4 and admission['lr_schedule'] == 'constant'
            and admission['batch_size'] == 16384 and admission['anchor_hold_weight'] == .02
            and admission['replay_weight'] == .02 and admission['replay_fraction'] == .05
            and m['identity'] == identity and identity['arm'] == a
            and identity['refresh_steps'] == expected_steps and identity['refresh_enabled'] is True
            and admission['head_sha'] == sha(HEADS['frozen_t0'][0])[:16]
            and admission['anchor_sha'] == sha(anchor)[:16]
            and admission['edges_sha'] == sha(SB/'dino-arrival-final/edges-k15-fuzzy.npz')[:16]
            and admission['pool_manifest_sha'] == sha(OC/'card012-pool-manifest.json')[:16]
            and admission['loaded_module_shas'] == json.loads((OC/'card012-code-frozen/manifest.json').read_text())['module_shas'])
        C[f'{a}_outputs_fresh'] = all((TRAIN/f'{prefix}-{a}.{ext}').stat().st_mtime >= started
            for prefix,ext in [('manifest','json'),('model','pt'),('coords','npy')])
        ts = m['train_stats']
        C[f'{a}_train_stats'] = (ts['executed_iters'] == ts['optimizer_steps_succeeded'] == 140000
            and all(ts.get(k) == v for k,v in anchor_stats.items() if k != 'anchor_landmark_path'))
        rf = m.get("replay_refreshes", [])
        ok_ref = ok_ref and [r["step"] for r in rf] == expected_steps
        shas = [r["new_sha"] for r in rf]; ok_distinct = ok_distinct and len(set(shas)) == 3
        # Stage files persist across a legitimate resume even if the resumed
        # trainer's in-memory stage_meta contains only its later transitions.
        for step in expected_steps:
            meta_path = TRAIN/f'stage-{a}-{step}.json'
            meta = json.loads(meta_path.read_text())
            bank_path = TRAIN/f'banks/{a}-step{step}.npz'
            with np.load(bank_path) as bank:
                ids, x, targets = bank['replay_ids'], bank['replay_X'], bank['replay_targets']
                ordered = hashlib.sha256()
                for arr in (ids.astype(np.int64), x.astype(np.float16), targets.astype(np.float32)):
                    ordered.update(np.ascontiguousarray(arr).tobytes())
                content = hashlib.sha256()
                for arr in (np.sort(ids.astype(np.int64)), x.astype(np.float16), targets.astype(np.float32)):
                    content.update(np.ascontiguousarray(arr).tobytes())
                digest = content.hexdigest()[:16]
                ok_ordered = ok_ordered and (len(np.unique(ids)) == 200000 and x.shape == (200000,1536)
                    and targets.shape == (200000,2) and all_finite(x) and all_finite(targets)
                    and ordered.hexdigest()[:16] == meta['ordered_row_identity_sha']
                    and digest == meta['content_sha'] and meta['step'] == step and meta['arm'] == a)
            ck = torch.load(TRAIN/f'ckpt/{a}/ckpt-step{step}.pt', map_location='cpu', weights_only=False)
            C[f'{a}_checkpoint_{step}'] = (ck['global_step'] == step and ck['step_checkpoint'] is True
                and ck['card012_identity'] == identity and ck['config']['replay_bank_sha'] == digest
                and ck['train_stats']['executed_iters'] == step
                and all(key in ck for key in ('optimizer','scheduler','scaler','replay_gen','loader_gen'))
                and all(bool(torch.isfinite(v).all()) for v in ck['model'].values())
                and next(v['new_sha'] for v in rf if v['step'] == step) == digest)
            del ck
        endpoint = torch.load(TRAIN/f'model-{a}.pt', map_location='cpu', weights_only=False)
        C[f'{a}_endpoint_state'] = state_digest(endpoint['model_state_dict']) == m['trained_sha256']
        del endpoint
    C["three_refreshes_expected_steps"] = ok_ref; C["distinct_banks"] = ok_distinct; C["ordered_digests"] = ok_ordered
    for a in ("uniform", "error_directed"):
        c = np.load(TRAIN / f"coords-{a}.npy", mmap_mode="r")
        C[f"{a}_coords_finite"] = c.shape == (2400000, 2) and all_finite(c)
    return bool(all(C.values())), C


def main():
    started = time.time(); OUT.mkdir(exist_ok=True)
    try:
        ok, checks = prescoring_validator()
    except (OSError, ValueError, KeyError, AssertionError, StopIteration) as ex:
        write_json(OUT / 'prescoring-validation.json', {'PASS': False, 'error': str(ex)})
        raise
    write_json(OUT / "prescoring-validation.json", {"PASS": ok, "checks": checks})
    if not ok:
        print(json.dumps({"prescoring_validation": checks, "PASS": ok}, indent=1));
        write_json(OC / "card012-score.json", {"status": "PRESCORING_FAIL", "checks": checks})
        raise RuntimeError('Card012 prescoring validation failed')

    val = np.load(SEAL / "val_hd.f16.npy", mmap_mode="r"); truth = np.load(SEAL / "truth_val.npy")
    ref = np.load(SEAL / "ref_hd.f16.npy", mmap_mode="r")
    groups = np.load(SEAL / "val_source.npy", allow_pickle=True).astype(str)
    assert set(np.unique(groups)) == set(OLD + ARR + ['diffusion-aesthetic-4k'])
    assert truth.shape == (len(val), 15) and len(groups) == len(val)
    ref_ids = np.load(SEAL / "ref_idx.npy"); val_ids = np.load(SEAL / "val_idx.npy")
    for g in OLD + ARR: assert (groups == g).sum() == 1200, (g, int((groups == g).sum()))
    tdraw = np.load(SUBD / "dino-arrival-t0/draw_idx.npy"); fdraw = np.load(SUBD / "dino-arrival-final/draw_idx.npy")
    common, ti, fi = np.intersect1d(tdraw, fdraw, assume_unique=True, return_indices=True)
    active = np.load(TD / "updates/anchor_active_ids-anchored.npy"); held = np.load(TD / "updates/anchor_holdout_ids-anchored.npy")
    am = np.isin(fi, active); hm = np.isin(fi, held); assert am.sum() == len(active) and hm.sum() == len(held) and not (am & hm).any()
    base = np.asarray(np.load(HEADS["frozen_t0"][1])[ti], np.float64)
    # DEVELOPMENT panel = ORIGINAL card006_confirm_bank (unseen, pool-excluded). Reserved 10K stays untouched.
    bank = np.load(OC / "card006_confirm_bank.npz"); cx = np.asarray(bank["replay_X"]); ci = bank["replay_ids"]; cs = bank["source"].astype(str)
    assert cx.dtype == np.float16 and not np.isin(ci, np.concatenate([fdraw, ref_ids, val_ids])).any()
    assert len(np.unique(ci)) == len(ci) and not np.isin(ci, np.load(SB/'card012-pool/pool_ids.npy')).any()
    assert all((cs == g).sum() > 0 for g in OLD)
    provenance = {"scorer_sha256": sha(__file__), "helper_sha256": sha(Path(__file__).with_name("score_card009.py")),
                  "R0": R0, "confirmation": "card006_confirm_bank (development); reserved 10K untouched",
                  "confirmation_sha256": sha(OC/'card006_confirm_bank.npz'), "frame_helper_sha256": sha(frame.__file__),
                  "instrument": {n: sha(SEAL / n) for n in ("ref_hd.f16.npy", "val_hd.f16.npy", "truth_val.npy", "val_source.npy", "ref_idx.npy", "val_idx.npy")},
                  "heads": {h: {"model_sha256": sha(mp), "coords_sha256": sha(cp),
                                "admission": sha(TRAIN / f"admission-{h}.json") if (TRAIN / f"admission-{h}.json").exists() else None}
                            for h, (mp, cp) in HEADS.items()}}
    write_json(OUT / "provenance.json", provenance)

    arrays = {"val_ids": val_ids, "ref_ids": ref_ids, "val_groups": groups, "truth": truth,
              "confirm_ids": ci, "confirm_sources": cs, "common_graph_ids": common,
              "active_graph_ids": common[am], "holdout_graph_ids": common[hm]}
    heads = {}; target = None
    for h, (mp, cp) in HEADS.items():
        model = ParametricUMAP.load(str(mp), device="cpu").model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        rc, vc = project(model, ref, True), project(model, val, True); pq = recall(rc, vc, truth)
        cc = project(model, cx).astype(np.float64)                            # card006 replay_X (fp16->fp32 in project)
        if h == "frozen_t0":
            target = cc.copy(); updated = base; R, t = np.eye(2), np.zeros(2); info = {"rmsd": 0.0}
        else:
            updated = np.asarray(np.load(cp, mmap_mode="r")[fi], np.float64)
            _, info = frame.rigid_align(updated[am], base[am]); R, t = np.array(info["R"]), np.array(info["t"])
        native = np.linalg.norm(cc - target, axis=1) / R0
        aligned = np.linalg.norm(cc @ R.T + t - target, axis=1) / R0
        hn = np.linalg.norm(updated[hm] - base[hm], axis=1) / R0
        ha = np.linalg.norm(updated[hm] @ R.T + t - base[hm], axis=1) / R0
        for b, v in pq.items():
            assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
            arrays[f"{h}_B{b}"] = v
        arrays.update({f"{h}_disp_native": native, f"{h}_disp_aligned": aligned,
                       f"{h}_confirmation_xy_native": cc, f"{h}_confirmation_xy_aligned": cc @ R.T + t,
                       f"{h}_holdout_native": hn, f"{h}_holdout_aligned": ha, f"{h}_R": R, f"{h}_t": t})
        heads[h] = {"confirmation_native": stats(native), "confirmation_aligned": stats(aligned),
                    "fraction_gt_005_aligned": float((aligned > .05).mean()),
                    "confirmation_by_source_aligned": {g: stats(aligned[cs == g]) for g in OLD if (cs == g).any()},
                    "confirmation_by_source_native": {g: stats(native[cs == g]) for g in OLD if (cs == g).any()},
                    "graph_holdout_native": stats(hn), "graph_holdout_aligned": stats(ha),
                    "recall": {str(b): {g: float(v[groups == g].mean()) for g in np.unique(groups)} for b, v in pq.items()},
                    "old": {str(b): float(v[np.isin(groups, OLD)].mean()) for b, v in pq.items()},
                    "arrival": {str(b): float(v[np.isin(groups, ARR)].mean()) for b, v in pq.items()}}
        write_json(OUT / f"{h}.json", heads[h]); del model, rc, vc
    arrays["confirmation_teacher_xy"] = target
    np.savez(OUT / "per-query.npz", **arrays)

    ed, un, an, fz = "error_directed", "uniform", "anchored", "frozen_t0"
    arrmask = np.isin(groups, ARR)
    p99c = paired(arrays[f"{ed}_disp_aligned"], arrays[f"{un}_disp_aligned"], lambda x: np.percentile(x, 99), cs)
    u_p99 = heads[un]["confirmation_aligned"]["p99"]
    ratio = (heads[ed]["confirmation_aligned"]["p99"] / u_p99) if u_p99 > 0 else None    # zero-denom = uninformative
    old_losses = {str(b): {g: heads[fz]["recall"][str(b)][g] - heads[ed]["recall"][str(b)][g] for g in OLD} for b in (250, 2000)}
    old_ok = all(np.mean(list(ls.values())) <= .005 and max(ls.values()) <= .01 for ls in old_losses.values())
    arr_vs_u = all(heads[ed]["arrival"][str(b)] >= heads[un]["arrival"][str(b)] - .005 for b in (250, 2000))
    arr_vs_a = all(heads[ed]["arrival"][str(b)] >= heads[an]["arrival"][str(b)] - .005 for b in (250, 2000))
    fresh_gain = paired(arrays[f"{ed}_B250"][arrmask], arrays[f"{fz}_B250"][arrmask], strata=groups[arrmask])
    arr_paired = {"vs_uniform": {str(b): paired(arrays[f"{ed}_B{b}"][arrmask], arrays[f"{un}_B{b}"][arrmask], strata=groups[arrmask]) for b in (250, 2000)},
                  "vs_anchored": {str(b): paired(arrays[f"{ed}_B{b}"][arrmask], arrays[f"{an}_B{b}"][arrmask], strata=groups[arrmask]) for b in (250, 2000)}}
    gate = {"unseen_aligned_p99_le_070xuniform": bool(ratio is not None and ratio <= .70),
            "paired_p99_ci_below_zero": bool(p99c["ci95"][1] < 0),
            "fraction_gt_005_not_increased": bool(heads[ed]["fraction_gt_005_aligned"] <= heads[un]["fraction_gt_005_aligned"]),
            **quality_checks(old_ok, arr_vs_u, arr_vs_a, fresh_gain['ci95'][0])}
    # Native stability AND every quality guard; fresh confirmation still required.
    deployment = {}
    for h in (un, ed):
        hl = {str(b): {g: heads[fz]["recall"][str(b)][g] - heads[h]["recall"][str(b)][g] for g in OLD} for b in (250, 2000)}
        hold = all(np.mean(list(ls.values())) <= .005 and max(ls.values()) <= .01 for ls in hl.values())
        hu = all(heads[h]['arrival'][str(b)] >= heads[un]['arrival'][str(b)] - .005 for b in (250,2000))
        ha = all(heads[h]['arrival'][str(b)] >= heads[an]['arrival'][str(b)] - .005 for b in (250,2000))
        gain = paired(arrays[f'{h}_B250'][arrmask], arrays[f'{fz}_B250'][arrmask], strata=groups[arrmask])
        deployment[h] = deployment_checks(heads[h]['confirmation_native']['mean'], heads[h]['confirmation_native']['p99'],
                                          quality_checks(hold, hu, ha, gain['ci95'][0]))
        deployment[h]['arrival_gain_vs_frozen'] = gain
        deployment[h]['note'] = 'Development eligibility only; untouched confirmation is required before deployment.'
    pool_src = Path("/data2/monet/pool-20m/source.npy")
    src = np.load(pool_src, allow_pickle=True).astype(str); nm, ct = np.unique(src, return_counts=True)
    weights = {str(a): int(b) for a, b in zip(nm, ct) if a in OLD + ARR}
    natural = {h: {str(b): sum(weights[g] * heads[h]["recall"][str(b)][g] for g in weights) / sum(weights.values()) for b in BUDGETS} for h in heads}
    assert set(weights) == set(OLD + ARR)
    equal = {h: equal_cohort_means(heads[h]['recall']) for h in heads}
    report = {"schema": "card012-score-2026-09-11", "status": "SCORED", "R0": R0,
              "primary_movement": "ALIGNED (rigid-fit on original active anchors); native reported alongside",
              "development_panel": "card006_confirm_bank (fresh reserved 10K untouched)",
              "heads": heads, "p99_ratio_error_over_uniform_unrounded": ratio,
              "paired_p99_error_minus_uniform_aligned": p99c, "old_losses_vs_frozen": old_losses,
              "arriving_vs_uniform_AND_anchored": {"vs_uniform_ok": arr_vs_u, "vs_anchored_ok": arr_vs_a, "paired": arr_paired},
              "B250_gain_over_frozen": fresh_gain, "gate": gate, "GATE_PASS": bool(all(gate.values())),
              "native_deployment_eligibility": deployment,
              "equal_cohort": equal, "natural_pool_weighted": natural, "natural_pool_weights": weights,
              "prescoring_validation": checks,
              "note": "Development evidence; aligned=primary, native=deployment. Reserved 10K reserved for a selected candidate's fresh confirmation. Execution provenance = admission-hash guard + source copy (NOT full executable isolation).",
              "uncertainty": "2000 paired source-stratified bootstrap draws, seed9009 (helper).", "cpu_wall_s": time.time() - started}
    write_json(OUT / "result.json", report); write_json(OC / "card012-score.json", report)
    print(json.dumps({"p99_ratio": ratio, "gate": gate, "GATE_PASS": all(gate.values()),
                      "deployment": {h: deployment[h]["DEPLOY_ELIGIBLE"] for h in deployment}}, indent=1), flush=True)


if __name__ == "__main__":
    main()

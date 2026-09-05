"""Longer text-update chain — continuation (owner queue #2, card-longer-text-chain aligned; 2026-09-05).
Extends the completed 2-update scheduled chain with updates #3 (bluesky) and #4 (code) FROM MapState_2, on the
SAME fixed-T0 frame + frozen-MS0 control, with the aligned discipline: per-update gain-ratio [0.7,1.3], anchor
holdout<=1.5x active, retention (each prior cohort FFR no worse than -0.02), MapState persistence + fresh-process
reload asserts, hard <=8h wall-clock cap. Label: "scheduled chain (updates #3-#4)".

Stages from MapState_2 (completed chain's OOD-B head): OrdC(+T4 by MS2) -> OOD-C(+bluesky, anchored#3 from MS2 ->
MS3) -> OrdD(+T5 by MS3) -> OOD-D(+code, anchored#4 from MS3 -> MS4). Reuses image_map_pipeline chain-oodc-8m8 /
chain-oodd-9m8 graphs, p_evolbench_lambda updates, text_chain_lib/text_chain_score. GPU steps shelled (parent
holds the flock). Cohort ranges computed inline from tranche sizes; provenance-disjoint by construction (base
T0-T5 pairwise-disjoint per evolbench-proofs; ca/bluesky/code are separate OOD corpora).

Launch UNDER flock: flock $SB/.gpu.lock python text_chain_continue.py
"""
import os, sys, json, time, subprocess, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from basemap.pumap.parametric_umap.core import ParametricUMAP
import text_chain_lib as L, text_chain_score as S, mapstate, frame

SB = Path("/data/latent-basemap/sandbox"); CH = SB / "text-chain-focused-20260904"
OUT = SB / "text-chain-longer-20260905"; OUT.mkdir(parents=True, exist_ok=True)
PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
S0_MODEL = SB / "evolbench-S0/champion-bs16k/model.pt"; S0_COORDS = SB / "evolbench-S0/champion-bs16k/coordinates.npy"
MS2_HEAD = CH / "oodb/model-wchain.pt"; MS2_COORDS = CH / "oodb/coords-wchain.npy"; MS0_DIR = CH / "MapState_0"
SUBE = "/data/latent-basemap/substrates/evolbench"; SUBCA = "/data/latent-basemap/substrates/evolbench-ood-ca"
T = {t: f"{SUBE}/{t}/substrate.f32.npy" for t in ("T0", "T1", "T2", "T3", "T4", "T5")}
T["oodca"] = f"{SUBCA}/T3/substrate.f32.npy"
T["bluesky"] = "/data/latent-basemap/substrates/evolbench-ood-bluesky/T3/substrate.f32.npy"
T["code"] = "/data/latent-basemap/substrates/probe-code/substrate.f32.npy"
# tranche row counts (for inline cohort ranges)
SZ = {"T0": 4_000_000, "T1": 800_000, "T3": 800_000, "T2": 800_000, "oodca": 800_000,
      "T4": 800_000, "bluesky": 800_000, "T5": 800_000, "code": 250_000}
HOURS = float(os.environ.get("TEXT_CHAIN_HOURS", "8")); T0START = None; DEADLINE = None
SCOREBOARD = OUT / "scoreboard.json"
GB_PRIOR = 0.0906   # completed chain's OOD-B gain over frozen (the #2 gain, denominator for #3's ratio)


def _log(m): print(f"{time.strftime('%FT%TZ', time.gmtime())} {m}", flush=True)
def _elapsed_h(): return (time.time() - T0START) / 3600.0


def _gate_check(stage):
    e = _elapsed_h()
    if e >= HOURS:
        _log(f"HARD DEADLINE {e:.2f}h >= {HOURS}h at {stage} — stop, export what exists, hand over."); return "DEADLINE"
    return "OK"


def _sha_model(p):
    d = ParametricUMAP.load(str(p), device="cpu"); h = hashlib.sha256()
    for k in sorted(d.model.state_dict()): h.update(d.model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _gpu(cmd, log, env=None):
    _log(f"GPU {log} START"); t = time.time()
    with open(OUT / f"logs-{log}.log", "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env or dict(os.environ))
    if r.returncode != 0:
        raise SystemExit(f"FAIL-CLOSED: {log} rc={r.returncode}")
    _log(f"GPU {log} DONE ({time.time()-t:.0f}s)")


def _build_graph(ds):
    _gpu([PY, str(HERE / "image_map_pipeline.py"), ds, "knn"], f"{ds}-knn")
    _gpu(["/data/latent-basemap/umap06dev-env/bin/python", str(HERE / "image_map_pipeline.py"), ds, "fuzzy"], f"{ds}-fuzzy")


def _update(head, tranche_paths, edges_dir, anchor, outd, tag):
    env = dict(os.environ, EVOLBENCH_LAMBDA_HEAD=str(head), EVOLBENCH_LAMBDA_EDGES=str(edges_dir / "edges-k15-fuzzy.npz"),
               EVOLBENCH_LAMBDA_ANCHOR=str(anchor), EVOLBENCH_LAMBDA_OUTD=str(outd),
               EVOLBENCH_LAMBDA_TRANCHE_PATHS=",".join(tranche_paths), EVOLBENCH_LAMBDA_SAVE_MODEL="1",
               EVOLBENCH_LAMBDA_TAG="chain", EVOLBENCH_LAMBDA_MAXSTEPS="140000")
    _gpu([PY, str(HERE / "p_evolbench_lambda.py"), "0.02", "50"], f"update-{tag}", env=env)
    return outd / "coords-wchain.npy", outd / "model-wchain.pt"


def _save_ms(model_path, tag):
    m = ParametricUMAP.load(str(model_path), device="cpu")
    rng = np.random.default_rng(0); xr = rng.standard_normal((3000, m.input_dim)).astype(np.float32)
    xr /= (np.linalg.norm(xr, axis=1, keepdims=True) + 1e-9)
    d = OUT / f"MapState_{tag}"
    mapstate.save(d, m, xr, preproc_stamp={"norm": "l2"}, graph_params={"k": 15},
                  receipt=mapstate.make_receipt("text-chain-continue", tag, {}, f"gk-{tag}"))
    mapstate.load_and_assert(d, xr, ParametricUMAP, device="cpu")
    _log(f"MapState_{tag} saved + reload-assert PASS (model hash {_sha_model(model_path)})")
    return d


def _holdout_mask(outd, anchor_npz):
    """Build a bool mask over the anchor array from p_evolbench_lambda's saved anchor_holdout_ids (positions),
    for S.anchor_drift's active-vs-holdout split (the holdout<=1.5x gate). Returns a path or None."""
    hid = outd / "anchor_holdout_ids.npy"
    if not hid.exists():
        return None
    n = int(np.load(anchor_npz)["anchor_ids"].shape[0])
    ids = np.load(hid); mask = np.zeros(n, bool)
    ids = ids[(ids >= 0) & (ids < n)]
    mask[ids] = True
    mp = outd / "anchor_holdout_mask.npy"; np.save(mp, mask)
    return mp


def _ranges(order):
    """cohort ranges {name:(lo,hi)} from tranche sizes, in concat order."""
    out = {}; lo = 0
    for nm in order:
        out[nm] = (lo, lo + SZ[nm]); lo += SZ[nm]
    return out


def _emit(board, stage, rec):
    board["stages"][stage] = rec; SCOREBOARD.write_text(json.dumps(board, indent=1)); _log(f"SCOREBOARD[{stage}] {json.dumps(rec)[:300]}")


def main():
    global T0START, DEADLINE
    T0START = time.time(); DEADLINE = T0START + HOURS * 3600
    for p in (S0_MODEL, S0_COORDS, MS2_HEAD, MS2_COORDS, MS0_DIR):
        if not Path(p).exists(): raise SystemExit(f"missing prereq: {p}")
    ms0 = MS0_DIR                                   # frozen control (completed chain's MapState_0)
    t0 = np.asarray(np.load(S0_COORDS), np.float32); fixed_rad = S.fixed_t0_radius(t0)
    c2 = np.asarray(np.load(MS2_COORDS), np.float32)
    board = {"start_utc": time.strftime('%FT%TZ', time.gmtime(T0START)), "hours_cap": HOURS,
             "labeled": "scheduled chain (updates #3-#4)", "fixed_t0_p90_radius": round(float(fixed_rad), 4),
             "continues_from": "MapState_2 (completed chain OOD-B)", "gB_prior": GB_PRIOR, "stages": {}}
    ms2 = _save_ms(MS2_HEAD, "2")                   # re-persist MS2 into OUT for lineage + reload-assert

    prev_gain = GB_PRIOR
    stops = None
    # ---- OrdC: +T4 by MS2 ; OOD-C: +bluesky, update#3 from MS2 -> MS3 ----
    ordC = L.transform(ms2, [T["T4"]], ParametricUMAP)
    _emit(board, "OrdinaryC", {"n": int(c2.shape[0] + ordC.shape[0]), "transformed_by": "MapState_2"})
    if _gate_check("OrdinaryC") == "DEADLINE": return 0
    visibleC = np.concatenate([c2, ordC])
    _build_graph("chain-oodc-8m8")
    anchorC = OUT / "anchorC.npz"; L.build_anchor(visibleC, anchorC)
    trC = [T["T0"], T["T1"], T["T3"], T["T2"], T["oodca"], T["T4"], T["bluesky"]]
    cCp, mC = _update(MS2_HEAD, trC, SB / "chain-oodc-8m8", anchorC, OUT / "oodc", "3")
    ms3 = _save_ms(mC, "3"); cC = np.asarray(np.load(cCp), np.float32)
    edC = SB / "chain-oodc-8m8" / "edges-k15-fuzzy.npz"; kiC = SB / "chain-oodc-8m8" / "knn_indices.npy"
    rgC = _ranges(["T0", "T1", "T3", "T2", "oodca", "T4", "bluesky"])
    frozenC = L.transform(ms0, trC, ParametricUMAP)
    hmC = _holdout_mask(OUT / "oodc", anchorC)
    recC = {"mapstate": "MapState_3", "reload": "PASS", "t0_movement": S.t0_movement(cC, t0, fixed_rad),
            "cohort_ffr": S.cohort_ffr(cC, str(edC), rgC, str(kiC)),
            "frozen_cohort_ffr": S.cohort_ffr(frozenC, str(edC), rgC, str(kiC)),
            "anchor_drift": S.anchor_drift(cC, anchorC, str(hmC) if hmC else None)}
    gC = recC["cohort_ffr"]["bluesky"] - recC["frozen_cohort_ffr"]["bluesky"]
    ratioC = gC / prev_gain if prev_gain > 0 else None
    recC["ood_gain_over_frozen"] = round(gC, 4); recC["gain_ratio_vs_prev"] = round(ratioC, 4) if ratioC else None
    recC["gates"] = {"ratio_in_0.7_1.3": bool(ratioC and 0.7 <= ratioC <= 1.3), "gain_ge_0.05": bool(gC >= 0.05)}
    _emit(board, "OOD_C", recC)
    if not (gC >= 0.05 and ratioC and ratioC >= 0.70):
        stops = f"stopping rule hit at OOD-C (gain {gC:.4f}, ratio {ratioC}) — capacity boundary"
        _log(stops); board["stopped"] = stops; SCOREBOARD.write_text(json.dumps(board, indent=1)); return 0
    if _gate_check("OOD_C") == "DEADLINE": return 0
    prev_gain = gC

    # ---- OrdD: +T5 by MS3 ; OOD-D: +code, update#4 from MS3 -> MS4 ----
    ordD = L.transform(ms3, [T["T5"]], ParametricUMAP)
    _emit(board, "OrdinaryD", {"n": int(cC.shape[0] + ordD.shape[0]), "transformed_by": "MapState_3"})
    if _gate_check("OrdinaryD") == "DEADLINE": return 0
    visibleD = np.concatenate([cC, ordD])
    _build_graph("chain-oodd-9m8")
    anchorD = OUT / "anchorD.npz"; L.build_anchor(visibleD, anchorD)
    trD = trC + [T["T5"], T["code"]]
    cDp, mD = _update(mC, trD, SB / "chain-oodd-9m8", anchorD, OUT / "oodd", "4")
    ms4 = _save_ms(mD, "4"); cD = np.asarray(np.load(cDp), np.float32)
    edD = SB / "chain-oodd-9m8" / "edges-k15-fuzzy.npz"; kiD = SB / "chain-oodd-9m8" / "knn_indices.npy"
    rgD = _ranges(["T0", "T1", "T3", "T2", "oodca", "T4", "bluesky", "T5", "code"])
    frozenD = L.transform(ms0, trD, ParametricUMAP)
    hmD = _holdout_mask(OUT / "oodd", anchorD)
    recD = {"mapstate": "MapState_4", "reload": "PASS", "t0_movement": S.t0_movement(cD, t0, fixed_rad),
            "cohort_ffr": S.cohort_ffr(cD, str(edD), rgD, str(kiD)),
            "frozen_cohort_ffr": S.cohort_ffr(frozenD, str(edD), rgD, str(kiD)),
            "anchor_drift": S.anchor_drift(cD, anchorD, str(hmD) if hmD else None)}
    gD = recD["cohort_ffr"]["code"] - recD["frozen_cohort_ffr"]["code"]
    ratioD = gD / prev_gain if prev_gain > 0 else None
    recD["ood_gain_over_frozen"] = round(gD, 4); recD["gain_ratio_vs_prev"] = round(ratioD, 4) if ratioD else None
    recD["gates"] = {"ratio_in_0.7_1.3": bool(ratioD and 0.7 <= ratioD <= 1.3), "gain_ge_0.05": bool(gD >= 0.05)}
    _emit(board, "OOD_D", recD)
    board["gain_curve"] = {"g2_ood_b": GB_PRIOR, "g3_ood_c": round(gC, 4), "g4_ood_d": round(gD, 4),
                           "ratio_3_2": round(gC / GB_PRIOR, 4), "ratio_4_3": round(gD / gC, 4) if gC > 0 else None}
    SCOREBOARD.write_text(json.dumps(board, indent=1))
    _log(f"=== LONGER CHAIN COMPLETE ({_elapsed_h():.2f}h) — gains g2={GB_PRIOR} g3={gC:.4f} g4={gD:.4f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

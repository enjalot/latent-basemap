"""Text-chain orchestrator (owner plan-basemap-chain, 2026-09-04). Runs the 8-hour focused persisted update chain:
Base(T0) -> OrdinaryA(T1 by MS0) -> OOD-A(+T3 reddit, anchored#1 from MS0) -> OrdinaryB(T2 by MS1) ->
OOD-B(+oodca_T3, anchored#2 from MS1), plus the MANDATORY frozen control (MS0 over the whole timeline). Fixed T0
frame + fixed T0 p90-radius throughout. Emits INCREMENTAL stage receipts + a running scoreboard vs the prereg
thresholds (flags failures live). Deadline watchdog with H2/H5/H7 gates.

GPU steps (cumulative graph knn+fuzzy via image_map_pipeline; anchored updates via p_evolbench_lambda) are
shelled under the flock. CPU steps (transform, anchor, mapstate save/reload, scoring) run in-process. Records
start + hard deadline UTC the moment it takes the GPU. Usage: text_chain.py   (env TEXT_CHAIN_HOURS default 8)."""
import os, sys, json, time, subprocess, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from basemap.pumap.parametric_umap.core import ParametricUMAP
import text_chain_lib as L, text_chain_score as S, mapstate, frame

SB = Path("/data/latent-basemap/sandbox"); OUT = SB / "text-chain-focused-20260904"; OUT.mkdir(parents=True, exist_ok=True)
LOCK = str(SB / ".gpu.lock"); PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
S0_MODEL = SB / "evolbench-S0/champion-bs16k/model.pt"; S0_COORDS = SB / "evolbench-S0/champion-bs16k/coordinates.npy"
SUBE = "/data/latent-basemap/substrates/evolbench"; SUBCA = "/data/latent-basemap/substrates/evolbench-ood-ca"
T = {"T0": f"{SUBE}/T0/substrate.f32.npy", "T1": f"{SUBE}/T1/substrate.f32.npy", "T2": f"{SUBE}/T2/substrate.f32.npy",
     "T3": f"{SUBE}/T3/substrate.f32.npy", "oodca_T3": f"{SUBCA}/T3/substrate.f32.npy"}
HOURS = float(os.environ.get("TEXT_CHAIN_HOURS", "8")); DEADLINE = None; T0START = None
SCOREBOARD = OUT / "scoreboard.json"; RECEIPT = OUT / "run-receipt.json"


def _log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _elapsed_h():
    return (time.time() - T0START) / 3600.0


def _gate_check(stage):
    """H2/H5/H7 + hard deadline. Returns a directive string; caller decides. Never silently shrinks scope."""
    e = _elapsed_h()
    if e >= HOURS:
        _log(f"HARD DEADLINE reached ({e:.2f}h >= {HOURS}h) at {stage} — stop training, export what exists, hand over.")
        return "DEADLINE"
    if e >= 7:
        return "H7_EXPORT"
    if e >= 5:
        return "H5_EVAL_ONLY"
    return "OK"


def _sha_model(p):
    d = ParametricUMAP.load(str(p), device="cpu")
    h = hashlib.sha256()
    for k in sorted(d.model.state_dict()):
        h.update(d.model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _gpu(cmd, log, env=None):
    """GPU subprocess (fail-closed). NO flock here: the orchestrator is launched UNDER `flock $LOCK` and owns the
    GPU for the whole chain (backlog yields at its shard boundary); re-flocking a child would deadlock on the
    parent-held lock."""
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
                  receipt=mapstate.make_receipt("text-chain", tag, {}, f"gk-{tag}"))
    mapstate.load_and_assert(d, xr, ParametricUMAP, device="cpu")   # fresh reconstruction, raises on >1e-4
    _log(f"MapState_{tag} saved + reload-assert PASS (parent-lineage via model hash {_sha_model(model_path)})")
    return d


def main():
    global DEADLINE, T0START
    T0START = time.time(); DEADLINE = T0START + HOURS * 3600
    start_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(T0START))
    dl_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(DEADLINE))
    RECEIPT.write_text(json.dumps({"schema": "text-chain-receipt-2026-09-04", "start_utc": start_utc,
        "hard_deadline_utc": dl_utc, "hours": HOURS, "seed": 42, "lambda": 0.02, "batch_size": 16384,
        "anchor_frac": 0.05, "anchor_holdout_frac": 0.10, "max_train_steps": 140000, "substrate": "fp16",
        "optimizer_state": "RESET each update (warm-start weights, NOT optimizer continuation)",
        "fixed_frame": "T0 (S0 coords), fixed p90-radius denominator", "manifest": str(OUT / "chain-manifest.json")}, indent=1))
    _log(f"=== TEXT CHAIN START {start_utc} hard deadline {dl_utc} ({HOURS}h) ===")
    board = {"start_utc": start_utc, "deadline_utc": dl_utc, "labeled": "single-seed focused chain", "stages": {}}
    ranges = L.stage_row_ranges(str(OUT / "chain-manifest.json"))

    def _emit(stage, rec):
        board["stages"][stage] = rec; SCOREBOARD.write_text(json.dumps(board, indent=1))
        _log(f"SCOREBOARD[{stage}] {json.dumps(rec)[:400]}")

    # ---- Base: MS0 = S0 head; fixed T0 frame = S0 coords ----
    ms0 = _save_ms(S0_MODEL, "0")
    t0 = np.asarray(np.load(S0_COORDS), np.float32); fixed_rad = S.fixed_t0_radius(t0)
    board["fixed_t0_p90_radius"] = round(float(fixed_rad), 4) if fixed_rad else None
    _emit("Base", {"n": int(t0.shape[0]), "mapstate": "MapState_0", "reload": "PASS"})

    # ---- Ordinary A: transform T1 by MS0 ----
    ordA = L.transform(ms0, [T["T1"]], ParametricUMAP)
    visibleA = np.concatenate([t0, ordA])                       # T0+T1 = 4.8M in MS0 frame
    _emit("OrdinaryA", {"n": int(visibleA.shape[0]), "transformed_by": "MapState_0"})
    if _gate_check("OrdinaryA") == "DEADLINE":
        return 0

    # ---- OOD A: graph(T0+T1+T3) + anchored update#1 from MS0 ----
    _build_graph("chain-ooda-5m6")
    anchorA = OUT / "anchorA.npz"; L.build_anchor(visibleA, anchorA)
    c1p, m1 = _update(S0_MODEL, [T["T0"], T["T1"], T["T3"]], SB / "chain-ooda-5m6", anchorA, OUT / "ooda", "1")
    ms1 = _save_ms(m1, "1"); c1 = np.asarray(np.load(c1p), np.float32)
    # frozen control at OOD-A: MS0 transforms the OOD-A cohort (T3) — compare active vs frozen
    ki_a = SB / "chain-ooda-5m6" / "knn_indices.npy"; ed_a = SB / "chain-ooda-5m6" / "edges-k15-fuzzy.npz"
    frozenA = L.transform(ms0, [T["T0"], T["T1"], T["T3"]], ParametricUMAP)
    recA = {"mapstate": "MapState_1", "reload": "PASS",
            "t0_movement": S.t0_movement(c1, t0, fixed_rad),
            "cohort_ffr": S.cohort_ffr(c1, str(ed_a), ranges["OOD_A"], str(ki_a)),
            "frozen_cohort_ffr": S.cohort_ffr(frozenA, str(ed_a), ranges["OOD_A"], str(ki_a)),
            "anchor_drift": S.anchor_drift(c1, anchorA)}
    _emit("OOD_A", recA)
    g = _gate_check("OOD_A")
    if g == "DEADLINE":
        _log("deadline at OOD_A — one-update artifact saved, exporting"); return 0

    # ---- Ordinary B: transform T2 by MS1 ----
    ordB = L.transform(ms1, [T["T2"]], ParametricUMAP)
    visibleB = np.concatenate([c1, ordB])                       # [T0,T1,T3]+T2 = 6.4M in MS1 frame
    _emit("OrdinaryB", {"n": int(visibleB.shape[0]), "transformed_by": "MapState_1"})

    # ---- OOD B: graph(7.2M) + anchored update#2 FROM MS1 ----
    if g == "H5_EVAL_ONLY":
        _log("H5 gate: skipping update#2 (evaluate completed states); chain ends one-update"); return 0
    _build_graph("chain-oodb-7m2")
    anchorB = OUT / "anchorB.npz"; L.build_anchor(visibleB, anchorB)
    c2p, m2 = _update(m1, [T["T0"], T["T1"], T["T3"], T["T2"], T["oodca_T3"]], SB / "chain-oodb-7m2", anchorB, OUT / "oodb", "2")
    ms2 = _save_ms(m2, "2"); c2 = np.asarray(np.load(c2p), np.float32)
    ki_b = SB / "chain-oodb-7m2" / "knn_indices.npy"; ed_b = SB / "chain-oodb-7m2" / "edges-k15-fuzzy.npz"
    frozenB = L.transform(ms0, [T["T0"], T["T1"], T["T3"], T["T2"], T["oodca_T3"]], ParametricUMAP)
    recB = {"mapstate": "MapState_2", "reload": "PASS", "parent": "MapState_1",
            "t0_movement": S.t0_movement(c2, t0, fixed_rad),
            "cohort_ffr": S.cohort_ffr(c2, str(ed_b), ranges["OOD_B"], str(ki_b)),
            "frozen_cohort_ffr": S.cohort_ffr(frozenB, str(ed_b), ranges["OOD_B"], str(ki_b)),
            "anchor_drift": S.anchor_drift(c2, anchorB)}
    _emit("OOD_B", recB)
    # chain-integrity: 2nd/1st OOD gain ratio over frozen
    try:
        gA = recA["cohort_ffr"]["T3"] - recA["frozen_cohort_ffr"]["T3"]
        gB = recB["cohort_ffr"]["oodca_T3"] - recB["frozen_cohort_ffr"]["oodca_T3"]
        board["oodA_gain_over_frozen"] = round(gA, 4); board["oodB_gain_over_frozen"] = round(gB, 4)
        board["second_over_first_gain_ratio"] = round(gB / gA, 4) if gA > 0 else None
        board["ratio_in_0.7_1.3"] = (gA > 0 and 0.7 <= gB / gA <= 1.3)
    except Exception as e:
        board["ratio_error"] = str(e)[:120]
    SCOREBOARD.write_text(json.dumps(board, indent=1))
    _log(f"=== TEXT CHAIN COMPLETE ({_elapsed_h():.2f}h) — scoreboard {SCOREBOARD} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

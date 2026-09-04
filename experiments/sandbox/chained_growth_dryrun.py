"""G dry-run GATE (overseer-signed-off 2026-09-04) — MUST PASS before the full chained-growth experiment.
2-step MapState chain on evolbench S3, checking the two gate concerns: (1) the reload ASSERT holds at each
persisted MapState (transform(x_ref) reproduces coords), and (2) the RIGID frame (frame.py) does not collapse or
accumulate rotation across an anchored UPDATE-FROM-UPDATE.

Chain: MS0 = S0 champion head. Step1 = anchored fine-tune S0->S3 (anchor=S2 layout, w=0.02) -> MS1. Step2 =
anchored fine-tune FROM MS1's model, re-anchored to MS1's OWN S3 layout, on S3 again -> MS2 (this is the update-
from-update the harness must demonstrate; the full chain swaps S3->S4/S5 via substrate parameterization). Reuses
p_evolbench_lambda.py (EVOLBENCH_LAMBDA_SAVE_MODEL) for the GPU fine-tunes (flock-guarded). Small MAXSTEPS -> a
fast gate, not the full experiment. Emits chained-growth-dryrun-gate.json. Usage: chained_growth_dryrun.py"""
import os, sys, json, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from basemap.pumap.parametric_umap.core import ParametricUMAP
import mapstate, frame

SB = Path("/data/latent-basemap/sandbox"); DRY = SB / "chained-growth-dryrun"; DRY.mkdir(parents=True, exist_ok=True)
LOCK = str(SB / ".gpu.lock"); PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
S0_MODEL = SB / "evolbench-S0/champion-bs16k/model.pt"
S0_COORDS = SB / "evolbench-S0/champion-bs16k/coordinates.npy"
S3_EDGES = SB / "evolbench-S3/edges-k15-fuzzy.npz"; S2_ANCHOR = SB / "lambda/s2_anchor.npz"
IDIM = 384; MAXSTEPS = "2000"; W = "0.02"; EPOCHS = "50"


def _reload_assert(model_path, tag):
    m = ParametricUMAP.load(str(model_path), device="cpu")
    rng = np.random.default_rng(0)
    xr = rng.standard_normal((3000, IDIM)).astype(np.float32); xr /= (np.linalg.norm(xr, axis=1, keepdims=True) + 1e-9)
    d = DRY / f"ms-{tag}"
    mapstate.save(d, m, xr, preproc_stamp={"norm": "l2"}, graph_params={"k": 15},
                  receipt=mapstate.make_receipt("dryrun", tag, {}, f"gk-{tag}"))
    mapstate.load_and_assert(d, xr, ParametricUMAP, device="cpu")   # raises on divergence
    print(f"  reload-assert PASS: {tag}", flush=True)


def _finetune(head, anchor, outd, tag):
    outd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EVOLBENCH_LAMBDA_HEAD=str(head), EVOLBENCH_LAMBDA_EDGES=str(S3_EDGES),
               EVOLBENCH_LAMBDA_ANCHOR=str(anchor), EVOLBENCH_LAMBDA_OUTD=str(outd),
               EVOLBENCH_LAMBDA_SAVE_MODEL="1", EVOLBENCH_LAMBDA_TAG="dry", EVOLBENCH_LAMBDA_MAXSTEPS=MAXSTEPS)
    subprocess.run(["flock", LOCK, PY, str(HERE / "p_evolbench_lambda.py"), W, EPOCHS], check=True, env=env,
                   stdout=open(SB / f"logs/dryrun-{tag}.log", "w"), stderr=subprocess.STDOUT)
    return outd / "coords-wdry.npy", outd / "model-wdry.pt"


def main():
    print("MS0 (S0 head) reload gate:", flush=True); _reload_assert(S0_MODEL, "MS0")
    print("Step1: anchored S0->S3 (anchor=S2)...", flush=True)
    c1, m1 = _finetune(S0_MODEL, S2_ANCHOR, DRY / "step1", "step1"); _reload_assert(m1, "MS1")
    xy1 = np.asarray(np.load(c1), np.float32)
    # step-2 anchor: pin ALL S3 rows to their MS1 positions (update-from-update)
    a2 = DRY / "step2_anchor.npz"
    np.savez(a2, anchor_ids=np.arange(xy1.shape[0], dtype=np.int64), anchor_targets=xy1.astype(np.float32))
    print("Step2: anchored MS1->S3 re-anchored to MS1 layout (update-from-update)...", flush=True)
    c2, m2 = _finetune(m1, a2, DRY / "step2", "step2"); _reload_assert(m2, "MS2")
    xy2 = np.asarray(np.load(c2), np.float32)
    xy0 = np.asarray(np.load(S0_COORDS), np.float32)
    # drift: rigid-frame churn + learned_scale across the two updates (scale ~1 => no collapse)
    d01, i01 = frame.churn(xy1, xy0)                 # MS0->MS1 (shared T0 rows)
    d12, i12 = frame.churn(xy2, xy1)                 # MS1->MS2 (update-from-update)
    def _ok(info):
        return 0.7 <= info["learned_scale"] <= 1.4
    gate = {"schema": "chained-growth-dryrun-gate-2026-09-04",
            "reload_assert": "PASS (MS0, MS1, MS2 all reproduced transform within 1e-4)",
            "MS0_to_MS1": {"churn_mean": round(float(d01.mean()), 5), "learned_scale": i01["learned_scale"], "rmsd": i01["rmsd"]},
            "MS1_to_MS2_update_from_update": {"churn_mean": round(float(d12.mean()), 5), "learned_scale": i12["learned_scale"], "rmsd": i12["rmsd"]},
            "no_scale_collapse": bool(_ok(i01) and _ok(i12)),
            "gate": "reload asserts PASS + learned_scale in [0.7,1.4] both steps (rigid frame stable, no collapse)"}
    gate["PASS"] = bool(_ok(i01) and _ok(i12))
    (DRY / "chained-growth-dryrun-gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1), flush=True)
    return 0 if gate["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

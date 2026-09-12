"""Card022 sequential GPU chain (per card022-local-shape-floor.md). Lives INSIDE the isolated worktree. Root
owns the primary shape-instrument/viability gate (>=5% excess-thin headroom) and the GPU release + both
leases; this chain runs inside them and does budget accounting only. Stages:
  calibration (freeze coefficient; PASS-gates) -> device_canary (OFF-bitwise/ON-diverges/resume/reject) ->
  ordinary -> shape_floor (uses the frozen coefficient).
GPU occupancy cap 4500s including preparation/attempts; hard window end 2026-09-13T01:52:44Z. Both arms are
exactly 60K; admission compares remaining time to the canary-measured per-arm estimate — no automatic dose
truncation (stop + report if it will not fit). Arm validation + idempotent skip use the canonical strict
validator. Frozen isolated source re-verified before each stage; all canaries + failed attempts are costed.
Usage: run_card022_chain.py
"""
import datetime as dt, fcntl, json, math, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card022_validate as V

ROOT = Path(__file__).resolve().parents[2]; ES = ROOT / "experiments/sandbox"
OC = V.OC; PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
CARD = OC / "card022-ledger.json"; WIN = OC / "cards-24h-window-ledger.json"
END = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
CAP = 4500; WIN_CAP = 86400; DOSE = V.DOSE
CALIB_CAP = 900; CANARY_CAP = 1200; ARM_TAIL_RESERVE = 200.0; DEFAULT_ITPS = 40.0
ARMS = ["ordinary", "shape_floor"]


def atomic(p, v):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(v, indent=2) + "\n"); os.replace(t, p)
def notify(msg):
    subprocess.run([PY, str(OC / "notify.py"), "post", "basemap-runner", str(OC / "card022-execution.json"), msg], timeout=30)


def charge(tag, seconds, rc):
    with (OC / "window-ledger-write.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for path, key in [(CARD, "batch_spent_s"), (WIN, "spent_s")]:
            v = json.loads(path.read_text()); v[key] = float(v.get(key, 0)) + seconds
            v.setdefault("entries", []).append({"t": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": "gpu_stage", "card": "022", "tag": tag, "wall_s": seconds, "rc": rc,
                "accounting": "exclusive GPU stage occupancy incl. setup + endpoint writes"})
            atomic(path, v)


def _spent(path, key):
    v = json.loads(Path(path).read_text()); x = float(v[key]); assert math.isfinite(x) and x >= 0, f"bad ledger {path}"; return x
def _remaining(stage_cap):
    return min(stage_cap, CAP - _spent(CARD, "batch_spent_s"), WIN_CAP - _spent(WIN, "spent_s"), END - time.time())
def _canary_itps():
    try:
        v = json.loads((OC / "card022-canary.json").read_text()).get("it_per_s")
        return float(v) if v and math.isfinite(float(v)) and float(v) > 0 else DEFAULT_ITPS
    except Exception:
        return DEFAULT_ITPS


def run_stage(tag, script, timeout):
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen source changed before {tag}: {bad}"
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} START {tag} timeout={timeout:.1f}s", flush=True)
    t = time.monotonic(); rc = 999
    try: rc = subprocess.run([PY, str(ES / script)], cwd=ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired: rc = 124
    finally: charge(tag, time.monotonic() - t, rc)
    return rc


def run_arm(arm, timeout):
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen source changed before {arm}: {bad}"
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} START {arm} timeout={timeout:.1f}s", flush=True)
    t = time.monotonic(); rc = 999
    try: rc = subprocess.run([PY, str(ES / "run_card022_arm.py"), arm, str(DOSE)], cwd=ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired: rc = 124
    finally: charge(arm, time.monotonic() - t, rc)
    return rc


def main():
    if not CARD.exists():
        atomic(CARD, {"schema": "card022-ledger", "batch_cap_s": CAP, "batch_spent_s": 0, "entries": []})
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen isolated source changed: {bad}"
    assert time.time() < END, "past hard deadline"
    completed = []

    # 1. calibration (freeze coefficient); PASS gates the shape arm + the whole card
    to = _remaining(CALIB_CAP); assert to >= 120, f"cannot admit calibration: {to:.0f}s"
    rc = run_stage("calibration", "calibrate_card022.py", to)
    if rc == 124: atomic(OC / "card022-execution.json", {"status": "CALIBRATION_TIMEOUT", "at": dt.datetime.now(dt.timezone.utc).isoformat()}); notify("Card022 calibration timed out; halted."); return
    calib = json.loads((OC / "card022-calibration.json").read_text())
    if rc == 3 or not calib.get("PASS"):
        atomic(OC / "card022-execution.json", {"status": "CALIBRATION_STOP", "calibration": calib, "at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "note": "Required shape/pairwise grad-norm ratio undefined/unstable/nonfinite. Stop; no strength sweep, no dose change."})
        notify("Card022 calibration STOP: coefficient undefined/unstable. Admission halted; no strength sweep."); return
    assert rc == 0, f"calibration rc={rc}"; print("DONE calibration", flush=True)

    # 2. device canary (OFF-bitwise / ON-diverges / resume twin / wrong bank-weight-seed-arm rejection)
    to = _remaining(CANARY_CAP); assert to >= 120, f"cannot admit canary: {to:.0f}s"
    rc = run_stage("device_canary", "gpu_card022_canary.py", to)
    assert rc == 0, f"device_canary rc={rc}"
    assert json.loads((OC / "card022-canary.json").read_text())["PASS"], "device canary FAILED"
    print("DONE device_canary", flush=True)

    # 3. arms — measured admission from the canary it/s; no dose truncation
    itps = _canary_itps(); need = DOSE / itps + ARM_TAIL_RESERVE
    for arm in ARMS:
        try:
            completed.append(V.strict_validate_arm(arm, ROOT))
            atomic(OC / "card022-completion-validation.json", {"completed": completed, "both_valid": len(completed) == 2})
            print(f"SKIP {arm} (already strict-valid)", flush=True); continue
        except Exception:
            pass
        to = _remaining(CAP)
        assert to >= need, f"cannot admit {arm}: remaining={to:.1f}s < measured need={need:.1f}s (itps={itps}); stop, no dose truncation"
        rc = run_arm(arm, min(to, need + ARM_TAIL_RESERVE + 300))
        if rc == 124:
            atomic(OC / "card022-execution.json", {"status": "ARM_TIMEOUT_CHECKPOINTED", "arm": arm, "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "note": "checkpoint preserved; re-queue resumes from latest ckpt; no dose truncation."})
            notify(f"Card022 {arm} hit budget with checkpoint preserved; re-queue resumes. No truncation."); return
        assert rc == 0, f"{arm} rc={rc}"
        completed.append(V.strict_validate_arm(arm, ROOT))
        atomic(OC / "card022-completion-validation.json", {"completed": completed, "both_valid": len(completed) == 2})
        print(f"DONE {arm}", flush=True)

    assert len({x["model_state_sha"] for x in completed}) == 2, "arms did not diverge"
    atomic(OC / "card022-execution.json", {"status": "TRAINED_VALIDATED", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "arms": completed, "coefficient": calib.get("coefficient"), "quality": "NOT_YET_SCORED"})
    notify("Card022 both arms (ordinary/shape_floor) trained + strict-validated: 60K continuation from the Card013 "
           "baseline endpoint, LR 1e-4, device_fp16, frozen coefficient, resumable step ckpts (20/40/60K) with "
           "shape_gen + bound identity, global VRAM<30GB. CPU scoring against root's shape instrument next.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        atomic(OC / "card022-execution.json", {"status": "EXECUTION_FAILED", "error": repr(e), "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        notify(f"Card022 chain stopped fail-closed: {e}. Preserve artifacts + reconcile ledger before repair.")
        raise

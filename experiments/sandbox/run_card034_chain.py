"""Card034 sequential GPU chain. Lives INSIDE the isolated worktree. Root owns the GPU release + both leases;
this chain runs inside them and does budget accounting only. Stages: device_canary -> throughput_preflight
(admission GATE for all three arms) -> grouped_umap -> grouped_nce -> grouped_infonce. Shared GPU cap 7200s;
per-arm 1800s CUMULATIVE across resume attempts; hard window end 2026-09-13T01:52:44Z. Admission compares
remaining time to the preflight-measured per-step * steps-left; no fallback, no dose truncation. Requires the
frozen calibration (grouped_infonce coefficient). Arm validation + idempotent skip use the canonical strict
validator; frozen isolated source re-verified before each stage. Usage: run_card034_chain.py
"""
import datetime as dt, fcntl, json, math, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V

ROOT = Path(__file__).resolve().parents[2]; ES = ROOT / "experiments/sandbox"
OC = V.OC; PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
CARD = OC / "card034-ledger.json"; WIN = OC / "cards-24h-window-ledger.json"
END = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
CAP = 7200; PER_ARM_CAP = 1800; WIN_CAP = 86400; DOSE = V.DOSE
CANARY_CAP = 1200; PREFLIGHT_CAP = 700; ARM_TAIL_RESERVE = 200.0
ARMS = ["grouped_umap", "grouped_nce", "grouped_infonce"]


def atomic(p, v):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(v, indent=2) + "\n"); os.replace(t, p)
def notify(msg):
    subprocess.run([PY, str(OC / "notify.py"), "post", "basemap-runner", str(OC / "card034-execution.json"), msg], timeout=30)


def charge(tag, seconds, rc):
    with (OC / "window-ledger-write.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for path, key in [(CARD, "batch_spent_s"), (WIN, "spent_s")]:
            v = json.loads(path.read_text()); v[key] = float(v.get(key, 0)) + seconds
            v.setdefault("entries", []).append({"t": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": "gpu_stage", "card": "034", "tag": tag, "wall_s": seconds, "rc": rc,
                "accounting": "exclusive GPU stage occupancy incl. setup + endpoint writes"})
            atomic(path, v)


def _spent(path, key):
    v = json.loads(Path(path).read_text()); x = float(v[key]); assert math.isfinite(x) and x >= 0, f"bad ledger {path}"; return x
def _arm_cumulative(arm):
    v = json.loads(CARD.read_text())
    return float(sum(e.get("wall_s", 0.0) for e in v.get("entries", []) if e.get("tag") == arm))
def _remaining(stage_cap, arm=None):
    r = [stage_cap, CAP - _spent(CARD, "batch_spent_s"), WIN_CAP - _spent(WIN, "spent_s"), END - time.time()]
    if arm is not None: r.append(PER_ARM_CAP - _arm_cumulative(arm))
    return min(r)
def _completed_steps(arm):
    cdir = OC.parent / "card034-train" / arm / "ckpts"
    if not cdir.is_dir(): return 0
    from run_card034_arm import _latest_ckpt
    return _latest_ckpt(cdir)[1]
def _preflight_perstep():
    pf = json.loads((OC / "card034-preflight.json").read_text()); ps = float(pf["per_step_s"])
    assert math.isfinite(ps) and ps > 0, "preflight per_step invalid"; return ps


def run_stage(tag, script, timeout, args=()):
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen source changed before {tag}: {bad}"
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} START {tag} timeout={timeout:.1f}s", flush=True)
    t = time.monotonic(); rc = 999
    try: rc = subprocess.run([PY, str(ES / script), *args], cwd=ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired: rc = 124
    finally: charge(tag, time.monotonic() - t, rc)
    return rc


def main():
    if not CARD.exists():
        atomic(CARD, {"schema": "card034-ledger", "batch_cap_s": CAP, "batch_spent_s": 0, "entries": []})
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen isolated source changed: {bad}"
    assert time.time() < END, "past hard deadline"
    assert json.loads((OC / "card034-calibration.json").read_text()).get("PASS"), "InfoNCE calibration not frozen/PASS"
    completed = []

    to = _remaining(CANARY_CAP); assert to >= 120, f"cannot admit canary: {to:.0f}s"
    rc = run_stage("device_canary", "gpu_card034_canary.py", to); assert rc == 0, f"device_canary rc={rc}"
    assert json.loads((OC / "card034-canary.json").read_text())["PASS"], "device canary FAILED"
    print("DONE device_canary", flush=True)

    to = _remaining(PREFLIGHT_CAP); assert to >= 120, f"cannot admit preflight: {to:.0f}s"
    rc = run_stage("throughput_preflight", "gpu_card034_preflight.py", to)
    if rc == 124:
        atomic(OC / "card034-execution.json", {"status": "PREFLIGHT_TIMEOUT", "at": dt.datetime.now(dt.timezone.utc).isoformat()}); notify("Card034 preflight timed out; halted."); return
    pf = json.loads((OC / "card034-preflight.json").read_text())
    if rc == 3 or not pf.get("PASS"):
        atomic(OC / "card034-execution.json", {"status": "PREFLIGHT_STOP", "preflight": pf, "at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "note": "Measured cost of the three 60K arms does not fit 7200s cap/window/deadline. Admission stopped; dose NOT truncated."})
        notify("Card034 preflight STOP: three 60K arms do not fit the cap/deadline. Admission halted, no dose truncation."); return
    assert rc == 0, f"throughput_preflight rc={rc}"
    per_step = _preflight_perstep(); print("DONE throughput_preflight", flush=True)

    for arm in ARMS:
        try:
            completed.append(V.strict_validate_arm(arm, ROOT))
            atomic(OC / "card034-completion-validation.json", {"completed": completed, "all_valid": len(completed) == 3})
            print(f"SKIP {arm} (already strict-valid)", flush=True); continue
        except Exception:
            pass
        remaining_steps = DOSE - _completed_steps(arm); need = per_step * remaining_steps + ARM_TAIL_RESERVE
        to = _remaining(PER_ARM_CAP, arm=arm)
        assert to >= need, f"cannot admit {arm}: remaining={to:.1f}s < measured need={need:.1f}s ({remaining_steps} steps @ {per_step:.5f}s); stop, no truncation"
        rc = run_stage(arm, "run_card034_arm.py", min(to, need + 300), args=(arm, str(DOSE)))
        if rc == 124:
            atomic(OC / "card034-execution.json", {"status": "ARM_TIMEOUT_CHECKPOINTED", "arm": arm, "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "note": "checkpoint preserved; re-queue resumes from latest ckpt (cumulative 1800s/arm); no dose truncation."})
            notify(f"Card034 {arm} hit its budget with checkpoint preserved; re-queue resumes. No truncation."); return
        assert rc == 0, f"{arm} rc={rc}"
        completed.append(V.strict_validate_arm(arm, ROOT))
        atomic(OC / "card034-completion-validation.json", {"completed": completed, "all_valid": len(completed) == 3})
        print(f"DONE {arm}", flush=True)

    assert len({x["model_state_sha"] for x in completed}) == 3, "arms did not diverge"
    atomic(OC / "card034-execution.json", {"status": "TRAINED_VALIDATED", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "arms": completed, "quality": "NOT_YET_SCORED"})
    notify("Card034 all three arms (grouped_umap/grouped_nce/grouped_infonce) trained + strict-validated: fresh 2D "
           "60K, LR 1e-3, grouped 9:1 uniform nonself noise, device_fp16, resumable step+epoch ckpts, frozen "
           "InfoNCE coefficient, global VRAM<30GB. Root owns scoring (250K common, grouped_infonce primary).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        atomic(OC / "card034-execution.json", {"status": "EXECUTION_FAILED", "error": repr(e), "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        notify(f"Card034 chain stopped fail-closed: {e}. Preserve artifacts + reconcile ledger before repair.")
        raise

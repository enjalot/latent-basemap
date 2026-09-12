"""Card023 sequential GPU chain (per card023-half-scale-2m.md + root review). Lives INSIDE the isolated worktree.
Root owns the GPU queue and holds both leases; this chain runs inside them and does budget accounting only.

Stages: device_canary -> throughput_preflight (admission GATE) -> actual3d. Root-review repairs:
per-arm 8,500s cap is CUMULATIVE across resume attempts (summed from the ledger by tag), not per retry; arm
admission compares remaining time to the MEASURED remaining-dose estimate (preflight per_step × steps left),
not a fixed constant; the preflight gate=false / rc3 path is handled as PREFLIGHT_STOP BEFORE any rc==0
assertion; timings are full-precision for decisions (rounded only for display); arm validation + idempotent
skip use the ONE canonical strict validator. Frozen isolated source is re-verified before every stage. GPU
cap 9,000s, hard deadline 2026-09-13T01:52:44Z. Arms auto-resume from their latest checkpoint. Usage:
run_card023_chain.py
"""
import datetime as dt, fcntl, json, math, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card023_validate as V

ROOT = Path(__file__).resolve().parents[2]; ES = ROOT / "experiments/sandbox"
OC = V.OC; PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
CARD = OC / "card023-ledger.json"; WIN = OC / "cards-24h-window-ledger.json"
END = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
CAP = 9000; PER_ARM_CAP = 8500; WIN_CAP = 86400; DOSE = V.DOSE
CANARY_CAP = 1500; PREFLIGHT_CAP = 600; ARM_TAIL_RESERVE = 400.0
ARMS = ["actual3d"]


def atomic(p, v):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(v, indent=2) + "\n"); os.replace(t, p)
def notify(msg):
    subprocess.run([PY, str(OC / "notify.py"), "post", "codex-overseer", str(OC / "card023-execution.json"), msg], timeout=30)


def charge(tag, seconds, rc):
    with (OC / "window-ledger-write.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for path, key in [(CARD, "batch_spent_s"), (WIN, "spent_s")]:
            v = json.loads(path.read_text()); v[key] = float(v.get(key, 0)) + seconds
            v.setdefault("entries", []).append({"t": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": "gpu_stage", "card": "023", "tag": tag, "wall_s": seconds, "rc": rc,
                "accounting": "exclusive GPU stage occupancy incl. setup + endpoint writes"})
            atomic(path, v)


def _spent(path, key):
    v = json.loads(Path(path).read_text()); x = float(v[key])
    assert math.isfinite(x) and x >= 0, f"malformed ledger {path}[{key}]"; return x


def _arm_cumulative_spent(arm):
    """Sum of ledger wall charged to THIS arm across all attempts (cumulative per-arm cap)."""
    v = json.loads(CARD.read_text())
    return float(sum(e.get("wall_s", 0.0) for e in v.get("entries", []) if e.get("tag") == arm))


def _completed_steps(arm):
    cdir = OC.parent / "card023-train" / arm / "ckpts"
    if not cdir.is_dir(): return 0
    from run_card023_arm import _latest_ckpt
    return _latest_ckpt(cdir)[1]


def _preflight_per_step():
    pf = json.loads((OC / "card023-preflight.json").read_text())
    ps = float(pf["per_step_s"]); assert math.isfinite(ps) and ps > 0, "preflight per_step invalid"; return ps


def _remaining_env(stage_cap, arm=None):
    now = time.time(); card_s = _spent(CARD, "batch_spent_s"); win_s = _spent(WIN, "spent_s")
    r = [stage_cap, CAP - card_s, WIN_CAP - win_s, END - now]
    if arm is not None: r.append(PER_ARM_CAP - _arm_cumulative_spent(arm))     # CUMULATIVE per-arm cap
    return min(r)


def main():
    if not CARD.exists():
        atomic(CARD, {"schema": "card023-ledger", "batch_cap_s": CAP, "batch_spent_s": 0, "entries": []})
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen isolated source changed: {bad}"
    assert time.time() < END, "past hard deadline"
    completed = []

    def run_stage(tag, script, args, timeout):
        assert V.runtime_manifest_check(ROOT)[0], f"frozen source changed before {tag}"
        print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} START {tag} timeout={timeout:.1f}s", flush=True)
        t = time.monotonic(); rc = 999
        try:
            rc = subprocess.run([PY, str(ES / script), *args], cwd=ROOT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        finally:
            charge(tag, time.monotonic() - t, rc)     # full-precision wall
        return rc

    # 1. device canary
    to = _remaining_env(CANARY_CAP); assert to >= 300, f"cannot admit canary: {to:.0f}s"
    rc = run_stage("device_canary", "gpu_card023_canary.py", [], to)
    assert rc == 0, f"device_canary rc={rc}"
    assert json.loads((OC / "card023-canary.json").read_text())["PASS"], "resume/identity canary FAILED"
    print("DONE device_canary", flush=True)

    # 2. throughput preflight — handle gate=false / rc3 as PREFLIGHT_STOP BEFORE asserting rc0
    to = _remaining_env(PREFLIGHT_CAP); assert to >= 120, f"cannot admit preflight: {to:.0f}s"
    rc = run_stage("throughput_preflight", "gpu_card023_preflight.py", [], to)
    if rc == 124:
        atomic(OC / "card023-execution.json", {"status": "PREFLIGHT_TIMEOUT", "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        notify("Card023 preflight timed out; admission halted."); return
    pf = json.loads((OC / "card023-preflight.json").read_text())
    if rc == 3 or not pf.get("PASS"):
        atomic(OC / "card023-execution.json", {"status": "PREFLIGHT_STOP", "preflight": pf,
               "at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "note": "Measured cost does not fit the 9,000s cap/deadline. Admission stopped; dose NOT truncated."})
        notify("Card023 preflight STOP: one 400K arm does not fit the cap/deadline. Admission halted, dose not truncated.")
        return
    assert rc == 0, f"throughput_preflight rc={rc}"
    per_step = _preflight_per_step(); print("DONE throughput_preflight", flush=True)

    # 3. arms — measured remaining-dose admission, cumulative per-arm cap
    for arm in ARMS:
        if (OC.parent/"card023-train"/f"manifest-{arm}.json").exists():
            completed.append(V.strict_validate_arm(arm, ROOT))
            atomic(OC / "card023-completion-validation.json", {"completed": completed, "one_valid": len(completed) == 1})
            print(f"SKIP {arm} (already strict-valid)", flush=True); continue
        remaining_steps = DOSE - _completed_steps(arm); need = per_step * remaining_steps + ARM_TAIL_RESERVE
        to = _remaining_env(PER_ARM_CAP, arm=arm)
        assert to >= need, f"cannot admit {arm}: remaining={to:.1f}s < measured need={need:.1f}s ({remaining_steps} steps left)"
        rc = run_stage(arm, "run_card023_arm.py", [arm, str(DOSE)], to)
        if rc == 124:
            atomic(OC / "card023-execution.json", {"status": "ARM_TIMEOUT_CHECKPOINTED", "arm": arm,
                   "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "note": "checkpoint preserved; re-queue resumes from latest ckpt (cumulative 8500s/arm; no truncation)."})
            notify(f"Card023 {arm} hit its budget with checkpoint preserved; re-queue resumes (cumulative per-arm cap). No truncation.")
            return
        assert rc == 0, f"{arm} rc={rc}"
        completed.append(V.strict_validate_arm(arm, ROOT))     # canonical strict validator
        atomic(OC / "card023-completion-validation.json", {"completed": completed, "one_valid": len(completed) == 1})
        print(f"DONE {arm}", flush=True)

    assert len(completed) == 1, "one treatment expected"
    atomic(OC / "card023-execution.json", {"status": "TRAINED_VALIDATED", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "arms": completed, "quality": "NOT_YET_SCORED"})
    notify("Card023 one half-strength3D arm (actual3d; ordinary018 reused) trained + strict-validated at 2M/400K: exact positive-LR "
           "dose, device_fp16, loadable snapshots + resumable step ckpts (incl 400K) with bound identity, isolated "
           "runtime under worktree, global VRAM<30GB. CPU scoring (score_card023.py) next.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        atomic(OC / "card023-execution.json", {"status": "EXECUTION_FAILED", "error": repr(e), "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        notify(f"Card023 chain stopped fail-closed: {e}. Preserve artifacts + reconcile ledger before repair.")
        raise

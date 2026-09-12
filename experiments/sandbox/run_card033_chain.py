"""Card033 sequential GPU chain (per card033-scale4m-conditional.md + root review). Lives INSIDE the isolated worktree.
Root owns the GPU queue and holds both leases; this chain runs inside them and does budget accounting only.

Stages: graph -> CPU_validator -> device_canary -> throughput_preflight (admission GATE) -> ordinary3d -> actual3d. Root-review repairs:
per-arm 14,500s cap is CUMULATIVE across resume attempts (summed from the ledger by tag), not per retry; arm
admission compares remaining time to the MEASURED remaining-dose estimate (preflight per_step × steps left),
not a fixed constant; the preflight gate=false / rc3 path is handled as PREFLIGHT_STOP BEFORE any rc==0
assertion; timings are full-precision for decisions (rounded only for display); arm validation + idempotent
skip use the ONE canonical strict validator. Frozen isolated source is re-verified before every stage. GPU
cap 32,400s, hard deadline 2026-09-13T01:52:44Z. Arms auto-resume from their latest checkpoint. Usage:
run_card033_chain.py
"""
import datetime as dt, fcntl, json, math, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card033_validate as V

ROOT = Path(__file__).resolve().parents[2]; ES = ROOT / "experiments/sandbox"
OC = V.OC; PY = "/home/enjalot/code/latent-basemap/.venv/bin/python"
CARD = OC / "card033-ledger.json"; WIN = OC / "cards-24h-window-ledger.json"
END = dt.datetime.fromisoformat("2026-09-13T01:52:44+00:00").timestamp()
CAP = 32400; PER_ARM_CAP = 14500; WIN_CAP = 86400; DOSE = V.DOSE
CANARY_CAP = 900; PREFLIGHT_CAP = 600; ARM_TAIL_RESERVE = 400.0
ARMS = ["ordinary3d", "actual3d"]


def atomic(p, v):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(v, indent=2) + "\n"); os.replace(t, p)
def notify(msg):
    subprocess.run([PY, str(OC / "notify.py"), "post", "codex-overseer", str(OC / "card033-execution.json"), msg], timeout=30)


def charge(tag, seconds, rc):
    with (OC / "window-ledger-write.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for path, key in [(CARD, "batch_spent_s"), (WIN, "spent_s")]:
            v = json.loads(path.read_text()); v[key] = float(v.get(key, 0)) + seconds
            v.setdefault("entries", []).append({"t": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": "gpu_stage", "card": "033", "tag": tag, "wall_s": seconds, "rc": rc,
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
    cdir = OC.parent / "card033-train" / arm / "ckpts"
    if not cdir.is_dir(): return 0
    from run_card033_arm import _latest_ckpt
    return _latest_ckpt(cdir)[1]


def _preflight_per_step():
    pf = json.loads((OC / "card033-preflight.json").read_text())
    ps = float(pf["per_step_s"]); assert math.isfinite(ps) and ps > 0, "preflight per_step invalid"; return ps


def _remaining_env(stage_cap, arm=None):
    now = time.time(); card_s = _spent(CARD, "batch_spent_s"); win_s = _spent(WIN, "spent_s")
    r = [stage_cap, CAP - card_s - 30, WIN_CAP - win_s - 30, END - now - 30]
    if arm is not None: r.append(PER_ARM_CAP - _arm_cumulative_spent(arm))     # CUMULATIVE per-arm cap
    return min(r)


def main():
    if not CARD.exists():
        atomic(CARD, {"schema": "card033-ledger", "batch_cap_s": CAP, "batch_spent_s": 0, "entries": []})
    ok, bad = V.runtime_manifest_check(ROOT); assert ok, f"frozen isolated source changed: {bad}"
    assert time.time() < END, "past hard deadline"
    completed = []
    release=json.loads((OC/'card033-launch-release.json').read_text());assert release['PASS']
    assert V.full_sha(ROOT/'card033-runtime-sha.json')==release['runtime_manifest_sha']
    fresh=json.loads((OC/'card031-score.json').read_text());audit=json.loads((OC/'card031-confirmation/independent-audit.json').read_text())
    assert fresh['FRESH_GATE_PASS'] and audit['PASS'], 'fresh031 prerequisite failed'
    assert V.full_sha(OC/'card031-score.json')==release['fresh031_score_sha'] and V.full_sha(OC/'card031-confirmation/independent-audit.json')==release['fresh031_audit_sha']
    assert V.full_sha(V.DATA/'draw-manifest.json')==release['draw_manifest_sha']

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

    # Exact graph first; a numeric/resource stop prevents training and preserves progress.
    graph_spent=sum(e.get('wall_s',0) for e in json.loads(CARD.read_text())['entries'] if e.get('tag')=='graph')
    to=_remaining_env(1800-graph_spent);assert to>=120,'graph cap exhausted'
    rc=run_stage('graph','build_card033_graph.py',[],to)
    if rc in [3,124]:
        atomic(OC/'card033-execution.json',{'status':'PREFLIGHT_STOP','stage':'graph','rc':rc,'note':'No model trained; graph cost/timeout stop, no draw/dose truncation.'});notify('Card033 graph admission stopped; root owns next allocation.');return
    assert rc==0, f'graph failed rc={rc}'
    import importlib;importlib.reload(V)
    gm=json.loads((V.DATA/'manifest.json').read_text());assert gm['PASS'] and gm['graph_builder_sha']==V.full_sha(ES/'build_card033_graph.py')
    rc=run_stage('CPU_validator_after_graph','cpu_card033_selftest.py',[],_remaining_env(120));assert rc==0

    # 1. device canary
    to = _remaining_env(CANARY_CAP); assert to >= 300, f"cannot admit canary: {to:.0f}s"
    rc = run_stage("device_canary", "gpu_card033_canary.py", [], to)
    assert rc == 0, f"device_canary rc={rc}"
    assert json.loads((OC / "card033-canary.json").read_text())["PASS"], "resume/identity canary FAILED"
    print("DONE device_canary", flush=True)

    # 2. throughput preflight — handle gate=false / rc3 as PREFLIGHT_STOP BEFORE asserting rc0
    to = _remaining_env(PREFLIGHT_CAP); assert to >= 120, f"cannot admit preflight: {to:.0f}s"
    rc = run_stage("throughput_preflight", "gpu_card033_preflight.py", [], to)
    if rc == 124:
        atomic(OC / "card033-execution.json", {"status": "PREFLIGHT_TIMEOUT", "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        notify("Card033 preflight timed out; admission halted."); return
    pf = json.loads((OC / "card033-preflight.json").read_text())
    if rc == 3 or not pf.get("PASS"):
        atomic(OC / "card033-execution.json", {"status": "PREFLIGHT_STOP", "preflight": pf,
               "at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "note": "Measured cost does not fit the 32,400s cap/deadline. Admission stopped; dose NOT truncated."})
        notify("Card033 preflight STOP: both 800K arms do not fit the cap/deadline. Admission halted, dose not truncated.")
        return
    assert rc == 0, f"throughput_preflight rc={rc}"
    per_step = _preflight_per_step(); print("DONE throughput_preflight", flush=True)

    # 3. arms — measured remaining-dose admission, cumulative per-arm cap
    for arm in ARMS:
        if (OC.parent/"card033-train"/f"manifest-{arm}.json").exists():
            completed.append(V.strict_validate_arm(arm, ROOT))
            atomic(OC / "card033-completion-validation.json", {"completed": completed, "both_valid": len(completed) == 2})
            print(f"SKIP {arm} (already strict-valid)", flush=True); continue
        remaining_steps = DOSE - _completed_steps(arm); need = per_step * remaining_steps + ARM_TAIL_RESERVE
        to = _remaining_env(PER_ARM_CAP, arm=arm)
        assert to >= need, f"cannot admit {arm}: remaining={to:.1f}s < measured need={need:.1f}s ({remaining_steps} steps left)"
        rc = run_stage(arm, "run_card033_arm.py", [arm, str(DOSE)], to)
        if rc == 124:
            atomic(OC / "card033-execution.json", {"status": "ARM_TIMEOUT_CHECKPOINTED", "arm": arm,
                   "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "note": "checkpoint preserved; re-queue resumes from latest ckpt (cumulative 14500s/arm; no truncation)."})
            notify(f"Card033 {arm} hit its budget with checkpoint preserved; re-queue resumes (cumulative per-arm cap). No truncation.")
            return
        assert rc == 0, f"{arm} rc={rc}"
        completed.append(V.strict_validate_arm(arm, ROOT))     # canonical strict validator
        atomic(OC / "card033-completion-validation.json", {"completed": completed, "both_valid": len(completed) == 2})
        print(f"DONE {arm}", flush=True)

    assert len({x["model_state_sha"] for x in completed}) == 2, "arms did not diverge"
    atomic(OC / "card033-execution.json", {"status": "TRAINED_VALIDATED", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "arms": completed, "quality": "NOT_YET_SCORED"})
    notify("Card033 both 3D arms (ordinary3d/actual3d) trained + strict-validated at 4M/800K: exact positive-LR "
           "dose, device_fp16, loadable snapshots + resumable step ckpts (incl 800K) with bound identity, isolated "
           "runtime under worktree, global VRAM<30GB. CPU scoring (score_card033_exact.py) next.")


if __name__ == "__main__":
    lease_start=time.monotonic();prior_spent=json.loads(CARD.read_text())["batch_spent_s"] if CARD.exists() else 0
    try:
        main()
    except Exception as e:
        atomic(OC / "card033-execution.json", {"status": "EXECUTION_FAILED", "error": repr(e), "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        notify(f"Card033 chain stopped fail-closed: {e}. Preserve artifacts + reconcile ledger before repair.")
        raise

    finally:
        if CARD.exists():
            uncharged=max(0.,time.monotonic()-lease_start-(json.loads(CARD.read_text())["batch_spent_s"]-prior_spent))
            if uncharged:charge("controller_validation_overhead",uncharged,0)

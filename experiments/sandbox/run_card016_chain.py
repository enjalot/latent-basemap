"""Card016 GPU canary, three matched fits and endpoint benchmark under both leases."""
from pathlib import Path
import datetime as dt
import fcntl
import json
import os
import subprocess
import time
import torch
from run_card016_arm import ROOT, OC, DATA, OUT, ARMS, SNAPS, file_sha, state_sha, atomic_json
from card016_model import CompactProjector

PY = '/home/enjalot/code/latent-basemap/.venv/bin/python'
CARD = OC / 'card016-ledger.json'; WINDOW = OC / 'cards-24h-window-ledger.json'
CAP = 2700; END = dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()


def charge(tag, wall, rc):
    with (OC / 'window-ledger-write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for p, key in [(CARD, 'batch_spent_s'), (WINDOW, 'spent_s')]:
            d = json.loads(p.read_text()); d[key] = float(d.get(key, 0)) + wall
            d.setdefault('entries', []).append({'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'card': '016',
                                               'event': 'exclusive_gpu_stage', 'stage': tag, 'wall_s': wall, 'rc': rc})
            atomic_json(p, d)


def stage(tag, script, args, expected, per_stage_cap):
    available = min(CAP - json.loads(CARD.read_text())['batch_spent_s'],
                    86400 - json.loads(WINDOW.read_text())['spent_s'], END - time.time(), per_stage_cap)
    assert available >= expected, f'cannot admit {tag}: {available:.1f}s < {expected:.1f}s expected'
    print(f'{dt.datetime.now(dt.timezone.utc).isoformat()} START {tag} timeout={available:.1f}', flush=True)
    start = time.monotonic(); rc = 999
    try:
        rc = subprocess.run([PY, str(ROOT / 'experiments/sandbox' / script), *args], cwd=ROOT, timeout=available).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    finally:
        charge(tag, time.monotonic() - start, rc)
    assert rc == 0, f'{tag} failed rc={rc}'


def validate(arm):
    folder = OUT / arm
    result = json.loads((folder / 'complete.json').read_text()); identity = json.loads((folder / 'admission.json').read_text())
    assert result['successful_steps'] == identity['steps'] == 20000
    assert result['identity'] == identity and identity['arm'] == arm
    sd_hashes = []
    for step in SNAPS:
        p = folder / f'step-{step}.pt'
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck['successful_steps'] == step and ck['identity'] == identity
        assert all(torch.isfinite(t).all() for t in ck['model_state_dict'].values())
        model = CompactProjector(); model.load_state_dict(ck['model_state_dict']); sd_hashes.append(state_sha(model))
        assert p.stat().st_mtime >= (folder / 'admission.json').stat().st_mtime
    assert len(set(sd_hashes)) == len(SNAPS)
    endpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(endpoint['model_state_dict']); assert state_sha(model) == sd_hashes[-1] == result['endpoint_state_sha']
    assert file_sha(folder / 'model.pt') == result['endpoint_file_sha']
    return {'arm': arm, 'PASS': True, 'warm_state_sha': result['warm_state_sha'], 'endpoint_state_sha': result['endpoint_state_sha'],
            'prior_penalty_exposed': result['prior_penalty_exposed'], 'sampled_prior_gradient_nonzero': result['sampled_prior_gradient_nonzero']}


def notify(message):
    subprocess.run([PY, str(OC / 'notify.py'), 'post', 'codex-overseer', str(OC / 'card016-execution.json'), message], timeout=30)


def main():
    if not CARD.exists(): atomic_json(CARD, {'schema': 'card016-ledger', 'batch_cap_s': CAP, 'batch_spent_s': 0, 'entries': []})
    runtime = json.loads((ROOT / 'card016-runtime-sha.json').read_text())
    assert all(file_sha(ROOT / p) == h for p, h in runtime.items()), 'frozen source mismatch'
    cpu = json.loads((OC / 'card016-cpu-canary.json').read_text())
    assert cpu['PASS'] and json.loads((OC / 'card016-data-canary.json').read_text())['PASS']
    assert all(file_sha(ROOT / 'experiments/sandbox' / n) == h for n, h in cpu['source'].items()), 'CPU canary source stale'
    stage('gpu_canary', 'card016_canary.py', ['cuda'], 60, 300)
    gpu = json.loads((OC / 'card016-cuda-canary.json').read_text()); assert gpu['PASS']
    estimate = 20000 / gpu['preflight']['updates_per_s']
    atomic_json(OC / 'card016-gpu-admission.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'PASS': True,
                                                'estimated_fit_per_arm_s': estimate, 'runtime': runtime,
                                                'data_manifest_sha': file_sha(DATA / 'manifest.json'), 'dose': 20000,
                                                'window_end': END, 'per_card_cap_s': CAP})
    validated = []
    for arm in ARMS:
        stage(arm, 'run_card016_arm.py', [arm], estimate + 30, min(1100, estimate * 1.5 + 90))
        validated.append(validate(arm))
        atomic_json(OC / 'card016-completion-validation.json', {'arms': validated, 'all_three_valid': len(validated) == 3})
    assert len({a['warm_state_sha'] for a in validated}) == 1
    stage('projection_benchmark', 'benchmark_card016.py', [], 30, 180)
    atomic_json(OC / 'card016-execution.json', {'status': 'TRAINED_VALIDATED_BENCHMARKED', 'arms': validated, 'quality': 'NOT_YET_SCORED'})
    notify('Card016 compact MSE/L1/L1+prior arms trained, validated and GPU-benchmarked. CPU quality scoring next; no promotion yet. Codex owns next allocation. Zero prior exposure, if observed, is reported as uninformative rather than a scientific benefit.')


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        atomic_json(OC / 'card016-execution.json', {'status': 'EXECUTION_FAILED', 'error': repr(exc), 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
        notify(f'Card016 stopped fail-closed: {exc}. Preserve and charge attempts; no quality verdict.')
        raise

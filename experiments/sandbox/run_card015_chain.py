"""Sequential GPU stages. Run only inside both leases; scoring is separate."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OC = Path('/data/latent-basemap/sandbox/overseer-codex')
ROOT = OC.parent / 'card015-code'
PY = '/home/enjalot/code/latent-basemap/.venv/bin/python'
CARD = OC / 'card015-ledger.json'
WIN = OC / 'cards-24h-window-ledger.json'
END = dt.datetime.fromisoformat('2026-09-13T01:52:44+00:00').timestamp()
CAP = 5400


def atomic(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, p)


def notify(message):
    subprocess.run([PY, str(OC / 'notify.py'), 'post', 'codex-overseer',
                    str(OC / 'card015-execution.json'), message], timeout=30)


def charge(tag, seconds, rc):
    with (OC / 'window-ledger-write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path, key in [(CARD, 'batch_spent_s'), (WIN, 'spent_s')]:
            value = json.loads(path.read_text())
            value[key] = float(value.get(key, 0)) + seconds
            value.setdefault('entries', []).append({
                't': dt.datetime.now(dt.timezone.utc).isoformat(),
                'event': 'gpu_stage', 'card': '015', 'tag': tag,
                'wall_s': seconds, 'rc': rc, 'accounting': 'exclusive GPU stage occupancy, including setup and endpoint writes'})
            atomic(path, value)


def validate_arm(arm):
    import torch
    import numpy as np
    out = OC.parent / 'card015-train'
    m = json.loads((out / f'manifest-{arm}.json').read_text())
    a = json.loads((out / f'admission-{arm}.json').read_text())
    assert m['executed_steps'] == m['train_stats']['positive_lr_optimizer_steps'] == 60000
    assert m['shared_init_sha256'] == '5544a31160054bcc'
    assert m['n_components'] == a['n_components'] == 3
    assert m['loaded_modules']['verified_frozen_runtime']
    assert a['rankneg_window'] == 75000 and a['batch_size'] == 16384
    assert m['lr_used_min'] == m['lr_used_max'] == .001
    expected = json.loads((ROOT / 'card015-runtime-sha.json').read_text())
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in expected.items())
    xy = np.load(out / f'coords-{arm}.npy', mmap_mode='r')
    assert xy.shape == (300000, 3) and np.isfinite(xy).all()
    for p in [out / f'model-{arm}.pt'] + [out / arm / f'model-step{s}.pt' for s in [20000, 30000, 40000, 60000]]:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        state = ck.get('model_state_dict', ck.get('model_state'))
        assert state is not None, f'missing model state: {p}'
        assert state['proj_out.weight'].shape[0] == 3
        assert all(torch.isfinite(t).all() for t in state.values())
    return {'arm': arm, 'PASS': True, 'model_state_sha': m['trained_sha256']}


def main():
    if not CARD.exists():
        atomic(CARD, {'schema': 'card015-ledger', 'batch_cap_s': CAP, 'batch_spent_s': 0, 'entries': []})
    assert json.loads((OC / 'card015-independent-cpu-audit.json').read_text())['PASS']
    assert time.time() < END
    expected_hashes = json.loads((ROOT / 'card015-runtime-sha.json').read_text())
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in expected_hashes.items()), 'frozen source changed'
    stages = [('device_canary', 'gpu_card015_canary.py', [], 300, 90)] + [
        (a, 'run_card015_arm.py', [a, '60000'], 1500, 1020)
        for a in ['baseline3d', 'actual_full3d', 'shuffled_full3d']]
    completed = []
    for tag, script, args, cap, expected in stages:
        c = json.loads(CARD.read_text())['batch_spent_s']
        w = json.loads(WIN.read_text())['spent_s']
        remaining = min(cap, CAP - c, 86400 - w, END - time.time())
        assert remaining >= expected, f'cannot admit {tag}: remaining={remaining}, expected={expected}'
        print(f'{dt.datetime.now(dt.timezone.utc).isoformat()} START {tag} timeout={remaining:.0f}s', flush=True)
        start = time.monotonic()
        rc = 999
        try:
            result = subprocess.run([PY, str(ROOT / 'experiments/sandbox' / script), *args],
                                    cwd=ROOT, timeout=remaining)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        finally:
            charge(tag, round(time.monotonic() - start, 3), rc)
        assert rc == 0, f'{tag} failed rc={rc}'
        if tag == 'device_canary':
            assert json.loads((OC / 'card015-canary.json').read_text())['PASS']
        else:
            completed.append(validate_arm(tag))
            atomic(OC / 'card015-completion-validation.json', {'completed': completed, 'all_three_valid': len(completed) == 3})
        print(f'DONE {tag}', flush=True)
    assert len({x['model_state_sha'] for x in completed}) == 3, 'interventions did not diverge'
    atomic(OC / 'card015-execution.json', {'status': 'TRAINED_VALIDATED', 'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'arms': completed, 'quality': 'NOT_YET_SCORED'})
    notify('Card015 all three 3D arms trained and validated; exact 60K positive-LR updates, loadable snapshots, isolated runtime. Quality not yet scored. Codex owns CPU scoring and next admitted card; no runner duplicate.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic(OC / 'card015-execution.json', {'status': 'EXECUTION_FAILED', 'error': repr(exc), 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
        notify(f'Card015 execution stopped fail-closed: {exc}. Preserve artifacts and reconcile ledger before repair.')
        raise

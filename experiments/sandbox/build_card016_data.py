"""CPU-only admission, training split and fixed density grid for Card016."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from _paths import ensure_paths
ensure_paths()
from basemap.pumap.parametric_umap.core import ParametricUMAP
from card016_model import CompactProjector, interpolate_density, density_penalty

SB = Path('/data/latent-basemap/sandbox'); OC = SB / 'overseer-codex'
POOL = SB / 'card012-pool'; OUT = SB / 'card016-data'
TEACHER = SB / 'dino-arrival-t0/champion-bs16k/model.pt'
torch.set_num_threads(2)
N = 1024; H = .01; SEED = 16016


def sha_array(a):
    h = hashlib.sha256()
    for start in range(0, len(a), 4096):
        h.update(np.ascontiguousarray(a[start:start + 4096]).tobytes())
    return h.hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(p, value):
    temp = p.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(p)


def make_grid(points):
    dx = 1 / (N - 1)
    edges = np.linspace(-dx / 2, 1 + dx / 2, N + 1)
    count, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(edges, edges))
    assert int(count.sum()) == len(points)
    return gaussian_filter(count / (len(points) * dx * dx), H / dx, mode='constant', truncate=6)


def direct_kde(points, probes):
    values, gradients = [], []
    for start in range(0, len(probes), 32):
        delta = points[None] - probes[start:start + 32, None]
        w = np.exp(-(delta * delta).sum(2) / (2 * H * H)) / (len(points) * 2 * np.pi * H * H)
        values.append(w.sum(1)); gradients.append((w[:, :, None] * delta / (H * H)).sum(1))
    return np.concatenate(values), np.concatenate(gradients)


def main():
    started = time.monotonic(); OUT.mkdir(exist_ok=True)
    assert not (OUT / 'manifest.json').exists(), 'bundle already admitted; do not overwrite'
    m = json.loads((OC / 'card012-pool-manifest.json').read_text())
    ids = np.load(POOL / 'pool_ids.npy'); source = np.load(POOL / 'pool_source.npy', allow_pickle=True).astype(str)
    X = np.load(POOL / 'pool_X.f16.npy', mmap_mode='r'); Y = np.load(POOL / 'pool_teacher.npy')
    checks = {'shape': X.shape == (1000000, 1536) and Y.shape == (1000000, 2), 'teacher_finite': bool(np.isfinite(Y).all()),
              'teacher_head_identity': sha_file(TEACHER)[:16] == m['teacher_head_sha'] == 'd62b18cb0df94981', 'unique_pool_ids': len(np.unique(ids)) == len(ids)}
    for name, arr, key in [('X', X, 'X_sha'), ('targets', Y, 'teacher_sha'), ('ids', ids, 'pool_ids_sha'), ('source', source, 'source_sha')]:
        checks[name + '_hash'] = sha_array(arr)[:16] == m[key]
    checks['all_X_finite'] = all(bool(np.isfinite(X[s:s + 32768]).all()) for s in range(0, len(X), 32768))
    for fname in ['ref_idx.npy', 'val_idx.npy']:
        checks['excluded_' + fname] = not np.isin(ids, np.load(Path('/data2/monet/eval-common-v2') / fname)).any()
    conf = np.load(OC / 'card006_confirm_bank.npz', allow_pickle=True)
    id_key = 'replay_ids'; assert id_key in conf.files, 'confirmation bank ID schema mismatch'
    checks['excluded_original_confirmation'] = not np.isin(ids, conf[id_key]).any()
    rng = np.random.default_rng(SEED)
    dev = np.sort(np.concatenate([rng.choice(np.flatnonzero(source == g), 10000, replace=False) for g in np.unique(source)]))
    mask = np.ones(len(ids), bool); mask[dev] = False; fit = np.flatnonzero(mask)
    checks['split'] = len(dev) == 50000 and len(fit) == 950000 and all((source[fit] == g).sum() == 190000 for g in np.unique(source))
    checks['split_disjoint'] = not np.intersect1d(fit, dev).size
    center = (Y[fit].min(0).astype('f8') + Y[fit].max(0).astype('f8')) / 2
    span = float(np.ptp(Y[fit].astype('f8'), axis=0).max() / .8)
    target = ((Y.astype('f8') - center) / span + .5).astype('f4')
    checks['fit_box'] = bool((target[fit] >= .0999999).all() and (target[fit] <= .9000001).all())
    model = ParametricUMAP.load(str(TEACHER), device='cpu').model.eval()
    sample = np.sort(rng.choice(fit, 512, replace=False))
    with torch.inference_mode(): pred = model(torch.from_numpy(np.array(X[sample], dtype='f4'))).numpy()
    max_resid = float(np.abs(pred - Y[sample]).max())
    checks['teacher_exact_stored_input_contract'] = max_resid <= .002
    teacher_params = sum(p.numel() for p in model.parameters()); del model
    # Fixed numerical approximation audit uses fit rows only, before any quality data.
    small = target[np.sort(rng.choice(fit, 10000, replace=False))].astype('f8')
    probes = np.clip(small[rng.integers(0, len(small), 512)] + rng.normal(0, .02, (512, 2)), .06, .94)
    small_grid = make_grid(small)
    p, grad = direct_kde(small, probes)
    q = torch.tensor(probes, dtype=torch.float64, requires_grad=True)
    grid_t = torch.tensor(small_grid, dtype=torch.float64)
    approx = interpolate_density(grid_t, q)
    ag = torch.autograd.grad(approx.sum(), q)[0].numpy()
    density = approx.detach().numpy(); good = p >= 1e-8
    logerr = np.abs(np.log(density[good]) - np.log(p[good]))
    gn = np.linalg.norm(grad, axis=1); an = np.linalg.norm(ag, axis=1)
    useful = good & (gn > 1e-4) & (an > 1e-4)
    cos = (grad[useful] * ag[useful]).sum(1) / (gn[useful] * an[useful])
    checks['grid_logp95'] = float(np.percentile(logerr, 95)) <= .05
    checks['grid_grad_median_cos'] = float(np.median(cos)) >= .99
    # Check actual clipped penalty gradients in supported low-density and dense probes.
    low = np.flatnonzero((density > 1e-8) & (density < np.exp(-2)) & (an > 1e-4))
    high = np.flatnonzero(density > np.exp(-2) * 2)
    checks['low_and_dense_probes_exist'] = bool(len(low) and len(high))
    if len(low) and len(high):
        q = torch.tensor(probes[[low[0], high[0]]], dtype=torch.float32, requires_grad=True)
        g = torch.tensor(small_grid, dtype=torch.float32)
        penalty = density_penalty(g, q); pg = torch.autograd.grad(penalty, q)[0]
        checks['low_prior_gradient_nonzero'] = bool(pg[0].abs().sum() > 0)
        checks['dense_prior_gradient_zero'] = bool(pg[1].abs().sum() == 0)
    grid = make_grid(target[fit].astype('f8')).astype('f4')
    checks['grid_mass'] = abs(float(grid.sum()) / (N - 1) ** 2 - 1) < 1e-4
    torch.manual_seed(42); student = CompactProjector()
    checks['parameter_reduction'] = sum(p.numel() for p in student.parameters()) <= .3 * teacher_params
    audit = {'PASS': all(checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
             'teacher_stored_input_max_residual': max_resid, 'grid_log_error_p95': float(np.percentile(logerr, 95)),
             'grid_gradient_cos_median': float(np.median(cos)), 'grid_audit_probes': int(good.sum()),
             'teacher_parameters': teacher_params, 'student_parameters': sum(p.numel() for p in student.parameters()),
             'cpu_wall_s': time.monotonic() - started, 'scope': 'Full input hashes, split, exact stored-input teacher contract and fixed-grid numerical fidelity. No GPU training or evaluation-quality observation.'}
    write(OC / 'card016-data-canary.json', audit)
    assert audit['PASS'], {k: v for k, v in checks.items() if not v}
    for name, arr in [('fit_rows', fit), ('dev_rows', dev), ('targets_normalized', target), ('density_grid', grid), ('center', center)]:
        np.save(OUT / f'{name}.npy', arr)
    torch.save(student.state_dict(), OUT / 'init.pt')
    # Fit-target density quantile is descriptive and fixed before evaluation.
    with torch.no_grad():
        lg = interpolate_density(torch.from_numpy(grid), torch.from_numpy(target[fit])).clamp_min(1e-12).log().numpy()
    manifest = {'schema': 'card016-data', 'complete': True, 'center': center.tolist(), 'span': span, 'seed': SEED,
                'fit_n': len(fit), 'dev_n': len(dev), 'fit_density_logp01': float(np.percentile(lg, 1)),
                'source': str(POOL), 'pool_manifest_sha': sha_file(OC / 'card012-pool-manifest.json'),
                'teacher_head_sha': sha_file(TEACHER), 'files': {p.name: sha_file(p) for p in OUT.iterdir() if p.is_file()},
                'builder_sha': sha_file(__file__), 'audit': audit}
    write(OUT / 'manifest.json', manifest)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__': main()

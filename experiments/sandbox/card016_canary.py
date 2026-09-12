"""Real standalone fit/resume, intervention and pairing checks; CPU or admitted GPU."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import copy
import json
import sys
import time
import numpy as np
import torch
from run_card016_arm import DATA, POOL, OC, new_training, train_step, state_sha, save_checkpoint, restore_checkpoint, file_sha, atomic_json
from card016_model import CompactProjector, interpolate_density


def opt_equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(opt_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(opt_equal(x, y) for x, y in zip(a, b))
    return a == b


def fresh(device):
    torch.manual_seed(42)
    if device == 'cuda': torch.cuda.manual_seed_all(42)
    return new_training(device)


def main(device):
    assert device in ['cpu', 'cuda']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda': assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    start = time.monotonic()
    fit = np.load(DATA / 'fit_rows.npy')
    raw = np.load(POOL / 'pool_X.f16.npy', mmap_mode='r')
    yt = np.load(DATA / 'targets_normalized.npy')
    if device == 'cpu':
        chosen = fit[:1024]
        X = torch.from_numpy(np.array(raw[chosen], dtype='f4'))
        Y = torch.from_numpy(yt[chosen]); rows = torch.arange(len(chosen))
    else:
        X = torch.from_numpy(np.array(raw, copy=True)).cuda()
        Y = torch.from_numpy(yt).cuda(); rows = torch.from_numpy(fit).cuda()
    grid = torch.from_numpy(np.load(DATA / 'density_grid.npy')).to(device)
    identity = {'canary': True, 'arm': 'compact_l1_prior', 'initial_bank': file_sha(DATA / 'manifest.json'), 'seed': 16016, 'device': device}
    path = OC / f'card016-{device}-resume-canary.pt'
    m1, o1, g1 = fresh(device)
    for _ in range(12): train_step(m1, o1, g1, X, Y, rows, grid, 'compact_l1_prior', batch=64)
    state1 = state_sha(m1); rng1 = g1.get_state().clone(); cpu1 = torch.get_rng_state().clone()
    m2, o2, g2 = fresh(device)
    for _ in range(5): train_step(m2, o2, g2, X, Y, rows, grid, 'compact_l1_prior', batch=64)
    save_checkpoint(path, m2, o2, g2, 5, identity, {'prior_sum': 17.25, 'loss_sum': 31.5})
    m3, o3, g3 = fresh(device)
    ck = restore_checkpoint(path, m3, o3, g3, identity, device)
    assert ck['successful_steps'] == 5
    for _ in range(5, 12): train_step(m3, o3, g3, X, Y, rows, grid, 'compact_l1_prior', batch=64)
    checks = {'genuine_intermediate_resume': ck['successful_steps'] == 5,
              'model_resume_bitwise': state1 == state_sha(m3), 'optimizer_resume_bitwise': opt_equal(o1.state_dict(), o3.state_dict()),
              'batch_rng_resume_bitwise': torch.equal(rng1, g3.get_state()), 'global_cpu_rng_resume_bitwise': torch.equal(cpu1, torch.get_rng_state()),
              'accumulators_restored': ck['accumulators'] == {'prior_sum': 17.25, 'loss_sum': 31.5},
              'targets_frozen': not Y.requires_grad and not grid.requires_grad}
    for key in ['arm', 'initial_bank', 'seed']:
        wrong = dict(identity); wrong[key] = str(wrong[key]) + '-wrong'
        try:
            restore_checkpoint(path, m3, o3, g3, wrong, device)
            checks['reject_' + key] = False
        except ValueError as exc:
            checks['reject_' + key] = str(exc) == 'admission identity mismatch'
    # Force a known low-density prediction to test the actual optimizer intervention.
    gn = grid.detach().cpu().numpy()
    eligible = np.argwhere((gn > 1e-5) & (gn < .1))
    eligible = eligible[(eligible.min(1) > 2) & (eligible.max(1) < len(gn) - 3)]
    assert len(eligible)
    point = torch.tensor(eligible[len(eligible) // 2] / (len(gn) - 1), dtype=torch.float32, device=device)
    forced, _, _ = fresh(device)
    with torch.no_grad():
        forced.net[-2].weight.zero_(); forced.net[-2].bias.copy_(torch.logit(point))
    sd = copy.deepcopy(forced.state_dict())
    xx = X.index_select(0, rows[:256]).float()
    with torch.no_grad(): artificial_y = forced(xx).detach()
    hashes, prior_values = [], []
    for arm in ['compact_l1', 'compact_l1_prior']:
        mm, oo, gg = fresh(device); mm.load_state_dict(sd)
        _, _, p = train_step(mm, oo, gg, xx, artificial_y, torch.arange(len(xx), device=device), grid, arm, batch=64)
        hashes.append(state_sha(mm)); prior_values.append(float(p))
    checks['prior_optimizer_divergence'] = hashes[0] != hashes[1] and prior_values[1] > 0
    checks['coordinates_fp32'] = artificial_y.dtype == torch.float32
    if device == 'cpu':
        # Independent exact-input teacher/pairing negative control, never a fit target refresh.
        from _paths import ensure_paths
        ensure_paths()
        from basemap.pumap.parametric_umap.core import ParametricUMAP
        teacher = ParametricUMAP.load('/data/latent-basemap/sandbox/dino-arrival-t0/champion-bs16k/model.pt', device='cpu').model.eval()
        sample = fit[:128]
        with torch.no_grad(): pred = teacher(torch.from_numpy(np.array(raw[sample], dtype='f4'))).numpy()
        targets = np.load(POOL / 'pool_teacher.npy')[sample]
        right = float(np.square(pred - targets).mean()); wrong = float(np.square(pred - np.roll(targets, 1, axis=0)).mean())
        checks['wrong_pairing_detected'] = wrong > max(1., 10000 * right)
    preflight = None
    if device == 'cuda':
        mm, oo, gg = fresh(device)
        for _ in range(30): train_step(mm, oo, gg, X, Y, rows, grid, 'compact_l1_prior')
        torch.cuda.synchronize(); t = time.monotonic()
        for _ in range(200): train_step(mm, oo, gg, X, Y, rows, grid, 'compact_l1_prior')
        torch.cuda.synchronize(); seconds = time.monotonic() - t
        preflight = {'successful_steps': 200, 'steady_state_s': seconds, 'updates_per_s': 200 / seconds,
                     'projected_three_arm_fit_s': 3 * 20000 * seconds / 200,
                     'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                     'scope': 'Real 1M feature bank / 950K fit-row sampler / batch8192 / bf16 hidden and FP32 output/loss; setup excluded.'}
        checks['fits_45m_with_margin'] = preflight['projected_three_arm_fit_s'] + 240 < 2700
        checks['vram_under_30'] = preflight['peak_allocated_gib'] < 30
    result = {'PASS': bool(all(checks.values())), 'device': device, 'checks': checks,
              'preflight': preflight, 'wall_s': time.monotonic() - start, 'resume_endpoint_sha': state1,
              'source': {n: file_sha(Path(__file__).parent / n) for n in ['card016_model.py', 'run_card016_arm.py', 'card016_canary.py']}}
    atomic_json(OC / f'card016-{device}-canary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    assert result['PASS'], {k: v for k, v in checks.items() if not v}


if __name__ == '__main__': main(sys.argv[1])

"""Standalone, resumable compact-teacher fit. Production dose and identity are fixed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
from card016_model import CompactProjector, loss_for

ROOT = Path(__file__).resolve().parents[2]
SB = Path('/data/latent-basemap/sandbox'); OC = SB / 'overseer-codex'
DATA = SB / 'card016-data'; POOL = SB / 'card012-pool'; OUT = SB / 'card016-train'
ARMS = ['compact_mse', 'compact_l1', 'compact_l1_prior']
SNAPS = [1000, 5000, 10000, 20000]


def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def atomic_json(p, x):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def save_checkpoint(p, model, optimizer, generator, step, identity, accumulators=None):
    obj = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
           'batch_rng': generator.get_state(), 'cpu_rng': torch.get_rng_state(),
           'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
           'successful_steps': step, 'identity': identity, 'accumulators': accumulators or {}}
    tmp = p.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(p)


def restore_checkpoint(p, model, optimizer, generator, identity, device):
    # Deserialize on CPU; optimizer.load_state_dict moves moments to each parameter's device
    # while preserving Adam's non-capturable CPU step scalar.
    ck = torch.load(p, map_location='cpu', weights_only=False)
    if ck['identity'] != identity:
        raise ValueError('admission identity mismatch')
    model.load_state_dict(ck['model_state_dict'], strict=True)
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    generator.set_state(ck['batch_rng'].cpu()); torch.set_rng_state(ck['cpu_rng'].cpu())
    if ck['cuda_rng'] is not None: torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    return ck


def train_step(model, optimizer, generator, X, targets, rows, grid, arm, batch=8192, capture=False):
    at = torch.randint(len(rows), (batch,), device=X.device, generator=generator)
    ix = rows.index_select(0, at)
    xb = X.index_select(0, ix); yb = targets.index_select(0, ix)
    optimizer.zero_grad(set_to_none=True)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if X.is_cuda else nullcontext()
    with context: pred = model(xb.float())
    loss, base, prior = loss_for(arm, pred, yb, grid)
    assert bool(torch.isfinite(loss)), 'non-finite loss'
    if capture:
        model._last_prior_grad_norm = float(torch.linalg.vector_norm(torch.autograd.grad(.002 * prior, pred, retain_graph=True)[0])) if prior.requires_grad else 0.
    loss.backward()
    assert bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()), 'non-finite gradient'
    optimizer.step()
    return loss.detach(), base.detach(), prior.detach()


def new_training(device):
    model = CompactProjector().to(device)
    model.load_state_dict(torch.load(DATA / 'init.pt', map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator(device=device).manual_seed(16016)
    return model, optimizer, generator


def main(arm):
    assert arm in ARMS and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((DATA / 'manifest.json').read_text())
    runtime = json.loads((ROOT / 'card016-runtime-sha.json').read_text())
    assert all(file_sha(ROOT / p) == digest for p, digest in runtime.items()), 'frozen runtime changed'
    assert all(file_sha(DATA / name) == digest for name, digest in manifest['files'].items()), 'data changed'
    assert file_sha(OC / 'card012-pool-manifest.json') == manifest['pool_manifest_sha']
    identity = {'card': '016', 'arm': arm, 'steps': 20000, 'batch': 8192, 'lr': .001,
                'precision': 'bf16_hidden_matmuls_FP32_output_loss_parameters', 'seed': 16016,
                'data_manifest_sha': file_sha(DATA / 'manifest.json'), 'init_sha': file_sha(DATA / 'init.pt'),
                'runtime': runtime, 'teacher': manifest['teacher_head_sha'], 'prior_weight': .002 if arm == 'compact_l1_prior' else 0}
    folder = OUT / arm; folder.mkdir(parents=True, exist_ok=True)
    ad = folder / 'admission.json'
    if ad.exists():
        assert json.loads(ad.read_text()) == identity, 'existing admission mismatch'
    else:
        atomic_json(ad, identity)
    assert not (folder / 'complete.json').exists(), 'do not overwrite completed arm'
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    X_cpu = np.array(np.load(POOL / 'pool_X.f16.npy', mmap_mode='r'), copy=True)
    hx = hashlib.sha256()
    for start in range(0, len(X_cpu), 4096): hx.update(X_cpu[start:start + 4096].tobytes())
    pool_manifest = json.loads((OC / 'card012-pool-manifest.json').read_text())
    assert hx.hexdigest()[:16] == pool_manifest['X_sha'], 'materialized feature-bank hash mismatch'
    X = torch.from_numpy(X_cpu).cuda(); del X_cpu
    target = torch.from_numpy(np.load(DATA / 'targets_normalized.npy')).cuda()
    rows = torch.from_numpy(np.load(DATA / 'fit_rows.npy')).cuda()
    dev = torch.from_numpy(np.load(DATA / 'dev_rows.npy')).cuda()
    grid = torch.from_numpy(np.load(DATA / 'density_grid.npy')).cuda()
    model, optimizer, generator = new_training('cuda')
    warm_sha = state_sha(model)
    ckpts = [folder / f'step-{s}.pt' for s in SNAPS if (folder / f'step-{s}.pt').exists()]
    restored = restore_checkpoint(ckpts[-1], model, optimizer, generator, identity, 'cuda') if ckpts else {}
    step = int(restored.get('successful_steps', 0)); resumed = step
    model.train(); log = json.loads((folder / 'curve.json').read_text()) if (folder / 'curve.json').exists() else []
    log = [r for r in log if r['step'] <= resumed]
    accum = restored.get('accumulators', {})
    prior_sum = torch.tensor(accum.get('prior_sum', 0.), device='cuda'); loss_sum = torch.tensor(accum.get('loss_sum', 0.), device='cuda')
    fit_start = time.monotonic()
    while step < 20000:
        loss, base, prior = train_step(model, optimizer, generator, X, target, rows, grid, arm, capture=(step + 1 in SNAPS))
        step += 1; prior_sum += prior; loss_sum += loss
        if step in SNAPS:
            save_checkpoint(folder / f'step-{step}.pt', model, optimizer, generator, step, identity, {'prior_sum': float(prior_sum), 'loss_sum': float(loss_sum)})
            # Development loss is recorded, never used to select an endpoint.
            with torch.inference_mode():
                residuals = []
                for i in range(0, len(dev), 8192):
                    ix = dev[i:i + 8192]
                    pred = model(X.index_select(0, ix).float())
                    residuals.append((pred - target.index_select(0, ix)).float())
                rr = torch.cat(residuals)
                record = {'step': step, 'dev_mse': float(rr.square().mean()), 'dev_mae': float(rr.abs().mean()),
                          'cumulative_prior': float(prior_sum), 'prior_gradient_norm_at_snapshot': getattr(model, '_last_prior_grad_norm', None), 'elapsed_s': time.monotonic() - fit_start}
            log.append(record); atomic_json(folder / 'curve.json', log)
            print(json.dumps(record), flush=True)
    torch.cuda.synchronize()
    endpoint = folder / 'model.pt'
    torch.save({'model_state_dict': model.state_dict(), 'identity': identity, 'successful_steps': step,
                'center': manifest['center'], 'span': manifest['span']}, endpoint)
    result = {'status': 'TRAINED', 'arm': arm, 'successful_steps': step, 'resumed_from_step': resumed,
              'warm_state_sha': warm_sha, 'endpoint_state_sha': state_sha(model), 'endpoint_file_sha': file_sha(endpoint),
              'identity': identity, 'fit_wall_s': time.monotonic() - fit_start, 'total_wall_s': time.monotonic() - started,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'positive_prior_accumulated': float(prior_sum), 'curve': log}
    assert step == 20000 and all(torch.isfinite(p).all() for p in model.parameters())
    result['prior_penalty_exposed'] = result['positive_prior_accumulated'] > 0
    result['sampled_prior_gradient_nonzero'] = any((r.get('prior_gradient_norm_at_snapshot') or 0) > 0 for r in log)
    assert all(file_sha(ROOT / p) == digest for p, digest in runtime.items()), 'runtime changed during fit'
    atomic_json(folder / 'complete.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['identity', 'curve']}, indent=2), flush=True)


if __name__ == '__main__': main(sys.argv[1])

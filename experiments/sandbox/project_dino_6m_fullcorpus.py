"""Resource-bounded, resumable 2D projection through both completed 6M DINO heads.

--preflight is CPU-only. The GPU run acquires the existing sandbox flock before
initializing CUDA, and releases it before any viewer publishing by the caller.
PCA is applied per batch from the saved training transform; no 100M PCA matrix
is materialized. Complete manifests are written only after full finite checks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

SB = Path('/data/latent-basemap/sandbox')
D6 = Path('/data2/monet/random-dino-6m')
INPUTS = [Path('/data2/monet/pool-20m/dino1536.f16.npy'),
          Path('/data2/monet/pool-complement-88m/dino1536.f16.npy')]
ARMS = ['monet-random-dino-6m', 'monet-random-dino-6m-pca768']
OUTS = [SB / 'fullcorpus-dino-6m-2d', SB / 'fullcorpus-dino-6m-pca768-2d']
STATE = SB / 'dino-6m-fullcorpus-20260907'
BATCH = 4096


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(path):
    s = path.stat()
    return {'path': str(path), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def setup(device):
    from _paths import ensure_paths
    ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.15)
    heads = [ParametricUMAP.load(str(SB / a / 'champion-bs16k/model.pt'), device=device).model.eval()
             for a in ARMS]
    pca = np.load(D6 / 'pca768-model.npz')
    mean = torch.tensor(pca['mean'], device=device)
    comp = torch.tensor(pca['components'], device=device)
    assert mean.shape == (1536,) and comp.shape == (1536, 768)
    return torch, heads, mean, comp


def preflight():
    torch, heads, mean, comp = setup('cpu')
    train = np.load(D6 / 'dino-substrate.f16.npy', mmap_mode='r')
    train_pca = np.load(D6 / 'pca768-substrate.f32.npy', mmap_mode='r')
    ids = np.linspace(0, len(train) - 1, 64, dtype=np.int64)
    positions = np.load(D6 / 'full_pos.npy', mmap_mode='r')[ids]
    columns = [np.load(p, mmap_mode='r') for p in INPUTS]
    assert [a.shape for a in columns] == [(19344847, 1536), (84471903, 1536)]
    x = np.asarray(train[ids], np.float32)
    joined = np.stack([columns[0][p] if p < len(columns[0]) else columns[1][p-len(columns[0])]
                       for p in positions]).astype(np.float32)
    np.testing.assert_array_equal(x, joined)
    with torch.inference_mode():
        tx = torch.from_numpy(x)
        px = torch.nn.functional.normalize((tx - mean) @ comp, dim=1)
        pca_error = float(np.max(np.abs(px.numpy() - train_pca[ids])))
        assert pca_error < 2e-6, pca_error
        errors = []
        for arm, head, inp in zip(ARMS, heads, [tx, px]):
            expected = np.load(SB / arm / 'champion-bs16k/coordinates.npy', mmap_mode='r')[ids]
            got = head(inp).numpy()
            error = float(np.max(np.abs(got - expected)))
            assert got.shape == (64, 2) and np.isfinite(got).all() and error < 1e-4, (arm, error)
            errors.append(error)
    result = {'sample_rows': 64, 'row_join_exact': True, 'pca_max_abs_error': pca_error,
              'head_max_abs_errors': dict(zip(ARMS, errors))}
    print(json.dumps(result), flush=True)
    STATE.mkdir(exist_ok=True)
    save(STATE / 'preflight.json', result)


def project():
    STATE.mkdir(exist_ok=True)
    identity = {'inputs': [fingerprint(p) for p in INPUTS],
                'checkpoints': [digest(SB / a / 'champion-bs16k/model.pt') for a in ARMS],
                'pca_sha256': digest(D6 / 'pca768-model.npz'), 'batch_size': BATCH}
    if all((out / 'manifest.json').exists() for out in OUTS):
        for out in OUTS:
            receipt = json.loads((out / 'manifest.json').read_text())
            assert receipt['source_identity'] == identity and receipt['status'] == 'complete'
            assert np.load(out / 'coords.f32.npy', mmap_mode='r').shape == (103816750, 2)
        print('Both matching projections already complete; proceeding to viewer refresh.', flush=True)
        return
    save(STATE / 'status.json', {'status': 'waiting_for_gpu_lock', 'identity': identity, 'pid': os.getpid()})
    with open(SB / '.gpu.lock', 'a') as lock, open('/data/latent-basemap/.gpu_lease', 'a') as lease:
        print('Waiting for sandbox GPU lock; no CUDA context allocated.', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        fcntl.flock(lease, fcntl.LOCK_EX)
        assert identity['inputs'] == [fingerprint(p) for p in INPUTS], 'Inputs changed while queued'
        assert identity['checkpoints'] == [digest(SB / a / 'champion-bs16k/model.pt') for a in ARMS], 'Checkpoints changed while queued'
        assert identity['pca_sha256'] == digest(D6 / 'pca768-model.npz'), 'PCA changed while queued'
        # Fail closed if a GPU job outside these locks is still using the card.
        other = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if other:
            raise RuntimeError(f'GPU compute process outside the acquired locks: {other}')
        progress_file = STATE / 'progress.json'
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {'identity': identity, 'done': 0, 'compute_wall_s': 0}
        assert progress['identity'] == identity, 'Inputs changed; refusing to resume'
        columns = [np.load(p, mmap_mode='r') for p in INPUTS]
        sizes = [len(a) for a in columns]
        total = sum(sizes)
        assert sizes == [19344847, 84471903]
        coords = []
        for out in OUTS:
            out.mkdir(exist_ok=True)
            if (out / 'manifest.json').exists():
                receipt = json.loads((out / 'manifest.json').read_text())
                assert receipt['source_identity'] == identity and progress['done'] == total
            path = out / ('coords.f32.npy' if (out / 'coords.f32.npy').exists() else 'coords.partial.npy')
            coords.append(np.lib.format.open_memmap(path, mode='r+' if progress['done'] else 'w+',
                                                    dtype=np.float32, shape=(total, 2)))
        torch, heads, mean, comp = setup('cuda')
        started = time.monotonic()
        previous_wall = progress['compute_wall_s']
        start_row = progress['done']
        last_save = start_row
        save(STATE / 'status.json', {'status': 'projecting', 'pid': os.getpid(), 'identity': identity})
        with torch.inference_mode():
            base = 0
            for src in columns:
                for i in range(max(0, start_row - base), len(src), BATCH):
                    end = min(i + BATCH, len(src))
                    x = torch.from_numpy(np.asarray(src[i:end], np.float32)).to('cuda')
                    p = torch.nn.functional.normalize((x - mean) @ comp, dim=1)
                    for target, head, inp in zip(coords, heads, [x, p]):
                        y = head(inp).cpu().numpy()
                        if not np.isfinite(y).all():
                            raise RuntimeError(f'Nonfinite projection at {base+i}')
                        target[base+i:base+end] = y
                    done = base + end
                    if done - last_save >= 1_000_000 or done == total:
                        for target in coords:
                            target.flush()
                        elapsed = time.monotonic() - started
                        progress.update(done=done, compute_wall_s=previous_wall + elapsed)
                        save(progress_file, progress)
                        rate = (done - start_row) / max(elapsed, 1e-6)
                        print(f'{done:,}/{total:,} rows; {rate:,.0f} rows/s; ETA {(total-done)/rate:.0f}s', flush=True)
                        last_save = done
                base += len(src)
        peak_vram = torch.cuda.max_memory_reserved()
        for target in coords:
            for i in range(0, total, 1_000_000):
                assert np.isfinite(target[i:i+1_000_000]).all()
        for index, out in enumerate(OUTS):
            if (out / 'coords.partial.npy').exists():
                (out / 'coords.partial.npy').replace(out / 'coords.f32.npy')
            save(out / 'manifest.json', {
                'schema': 'monet-dino-6m-fullcorpus-projection-v1', 'status': 'complete',
                'dim': 2, 'n_rows': total, 'n_pool': sizes[0], 'n_complement': sizes[1],
                'row_layout': {'pool': [0, sizes[0]], 'complement': [sizes[0], total]},
                'checkpoint': str(SB / ARMS[index] / 'champion-bs16k/model.pt'),
                'checkpoint_sha256': identity['checkpoints'][index],
                'checkpoint_sha256_16': identity['checkpoints'][index][:16],
                'training_rows': 6000000, 'input_dimensions': [1536, 768][index],
                'preprocessing': 'f16 to f32; ' + ('saved PCA mean/components, then unit normalization' if index else 'already normalized DINO column'),
                'pca_sha256': identity['pca_sha256'] if index else None,
                'source_identity': identity, 'finite_rows_checked': total,
                'joint_two_head_wall_s': progress['compute_wall_s'],
                'peak_cuda_reserved_bytes': peak_vram,
                'note': 'Both heads streamed together; wall time is joint, not per head. Projection only; no new reception score.'})
        save(STATE / 'status.json', {'status': 'complete', 'outputs': list(map(str, OUTS)),
                                   'joint_two_head_wall_s': progress['compute_wall_s'], 'peak_cuda_reserved_bytes': peak_vram})
        print('Both projections complete; GPU locks released on exit.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        try:
            project()
        except Exception as error:
            STATE.mkdir(exist_ok=True)
            save(STATE / 'status.json', {'status': 'failed', 'error': str(error)})
            raise

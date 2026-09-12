"""Same-input FP32 GPU projection benchmark of actual endpoints and their teacher."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
from _paths import ensure_paths
ensure_paths()
from basemap.pumap.parametric_umap.core import ParametricUMAP
from card016_model import CompactProjector
from run_card016_arm import DATA, POOL, OUT, OC, ARMS, file_sha, atomic_json


class NativeCompact(torch.nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.model = CompactProjector()
        self.model.load_state_dict(obj['model_state_dict'], strict=True)
        self.register_buffer('center', torch.tensor(obj['center'], dtype=torch.float32))
        self.span = obj['span']

    def forward(self, x):
        return (self.model(x) - .5) * self.span + self.center


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    raw = np.load(POOL / 'pool_X.f16.npy', mmap_mode='r')
    X = torch.from_numpy(np.array(raw[:65536], dtype='f4')).cuda()
    teacher_path = Path('/data/latent-basemap/sandbox/dino-arrival-t0/champion-bs16k/model.pt')
    models = {'teacher': ParametricUMAP.load(str(teacher_path), device='cuda').model.eval()}
    hashes = {'teacher': file_sha(teacher_path)}
    for a in ARMS:
        p = OUT / a / 'model.pt'
        obj = torch.load(p, map_location='cpu', weights_only=False)
        assert obj['successful_steps'] == 20000
        models[a] = NativeCompact(obj).cuda().eval(); hashes[a] = file_sha(p)
    params = {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}
    timings = {a: [] for a in models}; batch = 8192; iterations = 128
    orders = [['teacher', *ARMS], [*reversed(ARMS), 'teacher'], [ARMS[1], 'teacher', ARMS[2], ARMS[0]]]
    with torch.inference_mode():
        for m in models.values():
            for i in range(20): m(X[(i % 8) * batch:(i % 8 + 1) * batch])
        torch.cuda.synchronize()
        for order in orders:
            for a in order:
                torch.cuda.synchronize(); start = time.monotonic()
                for i in range(iterations):
                    y = models[a](X[(i % 8) * batch:(i % 8 + 1) * batch])
                torch.cuda.synchronize(); elapsed = time.monotonic() - start
                assert y.shape == (batch, 2) and bool(torch.isfinite(y).all())
                timings[a].append({'seconds': elapsed, 'rows_per_s': batch * iterations / elapsed})
    medians = {a: float(np.median([v['rows_per_s'] for v in values])) for a, values in timings.items()}
    result = {'schema': 'card016-projection-benchmark', 'PASS': True, 'batch': batch, 'iterations_per_repeat': iterations,
              'repeats': 3, 'orders': orders, 'precision': 'FP32 parameters, inputs and arithmetic; autocast off; TF32 disabled',
              'input': 'Same preloaded 65,536 stored pool rows cast to FP32, cycling 8 batches; no I/O or host transfer in timed loop',
              'includes': 'Student conversion from normalized output to native teacher frame',
              'parameter_counts': params, 'timings': timings, 'median_rows_per_s': medians,
              'speedup_vs_teacher': {a: medians[a] / medians['teacher'] for a in ARMS}, 'model_hashes': hashes,
              'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'limit': 'Model projection benchmark. Does not measure a complete 100M feature-read, transfer, projection and coordinate-write pipeline.'}
    atomic_json(OC / 'card016-projection-benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()

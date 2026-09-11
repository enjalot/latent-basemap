"""Independent CPU checks of development baseline identities and reported arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--card', choices=['card006', 'card007'], required=True)
    args = p.parse_args()
    os.environ['CARD'] = args.card
    from score_replay_movement import CFG, OC
    root = OC / 'sharpening-baselines' / args.card
    result = json.loads((root / 'results.json').read_text())
    q = np.load(root / 'per-query.npz')
    old = np.load(OC / f'{args.card}-reception-perq.npz')
    final = np.load(CFG['final_draw'])
    active = final[np.load(CFG['orig_active'])]
    assert result['status'] == 'COMPLETE' and len(result['results']) == 17
    assert np.isin(q['frame_anchor_ids'], active).all()
    assert not np.intersect1d(q['confirmation_ids'], final).size
    assert np.array_equal(q['val_ids'], old['val_idx'])
    checks = {}
    for tag, r in result['results'].items():
        d = q[f'{tag}_movement']
        assert np.isfinite(d).all()
        assert abs(np.quantile(d, .99) - r['movement']['p99']) < 1e-12
        assert abs(d.mean() - r['movement']['mean']) < 1e-12
        for budget in (50, 100, 250, 500, 1000, 2000):
            x = q[f'{tag}_B{budget}']
            assert np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
        if '-blend' in tag:
            parent, alpha = tag.split('-blend')
            assert np.max(np.abs(d - float(alpha)*q[f'{parent}_movement'])) < 1e-7
    for tag, prev in [('frozen', 'frozen'), ('anchored', 'anchored'),
                      ('in-140000', 'in'), ('out-140000', 'out')]:
        for budget in (250, 2000):
            delta = q[f'{tag}_B{budget}'] - old[f'{prev}_B{budget}']
            assert abs(delta.mean()) < .0002
            checks[f'{tag}_B{budget}'] = dict(aggregate_delta=float(delta.mean()),
                                            changed_queries=int(np.count_nonzero(delta)))
    a, b = q['in-70000_movement'], q['in-140000_movement']
    rng = np.random.default_rng(11008)
    bs = []
    for _ in range(2000):
        ix = rng.integers(len(a), size=len(a))
        bs.append(np.quantile(a[ix], .99) - np.quantile(b[ix], .99))
    audit = dict(status='PASS', frame_training_identity=True,
                 confirmation_excluded_final_graph=True,
                 endpoint_quality_reproduction=checks,
                 note='Development 20K anchor gauge, not original full-active frame. '
                      'Float32 rigid projection can change near-tie rankings; '
                      'historical aggregate quality reproduced within .0002.',
                 in70k_minus_in140k_p99=float(np.quantile(a, .99)-np.quantile(b, .99)),
                 paired_query_ci95=np.quantile(bs, [.025, .975]).tolist())
    (root / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

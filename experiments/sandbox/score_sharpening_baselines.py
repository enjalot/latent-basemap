"""CPU development scoring of saved replay doses and coordinate blends.

No training, no gate relaxation. Writes separate artifacts from cards006/007.
The development frame uses 20K original active anchors; final experiment
reporting retains the full original active-anchor gauge unless preregistered.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import faiss

from _paths import ensure_paths
ensure_paths()
from basemap.pumap.parametric_umap.core import ParametricUMAP
import frame


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def project(model, x, normalize):
    result = []
    with torch.inference_mode():
        for i in range(0, len(x), 8192):
            batch = norm(x[i:i+8192]) if normalize else np.asarray(x[i:i+8192], np.float32)
            result.append(model.model(torch.from_numpy(batch)).float().numpy())
    z = np.concatenate(result)
    assert z.shape == (len(x), 2) and np.isfinite(z).all()
    return z


def stats(d):
    assert np.isfinite(d).all()
    return dict(mean=float(d.mean()), p95=float(np.quantile(d, .95)),
                p99=float(np.quantile(d, .99)), frac_gt05=float((d>.05).mean()),
                frac_gt10=float((d>.10).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--card', choices=['card006','card007'], default='card006')
    a = p.parse_args()
    os.environ['CARD'] = a.card
    from score_replay_movement import CFG, OC
    from score_replay_reception import CFG as QUALITY
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    started = time.time()
    out = OC/'sharpening-baselines'/a.card
    out.mkdir(parents=True, exist_ok=True)
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    seal = CFG['seal']
    ref = np.load(seal/'ref_hd.f16.npy', mmap_mode='r')
    val = np.load(seal/'val_hd.f16.npy', mmap_mode='r')
    truth = np.load(seal/'truth_val.npy')
    vi, ri = np.load(seal/'val_idx.npy'), np.load(seal/'ref_idx.npy')
    groups = np.load(seal/CFG['seal_grp'], allow_pickle=True).astype(str)
    assert not np.intersect1d(vi, ri).size
    assert truth.shape[0] == len(vi) and truth.min() >= 0 and truth.max() < len(ri)
    for dp in (CFG['t0_draw'], CFG['final_draw']):
        assert not np.intersect1d(vi, np.load(dp)).size, f'Query/train overlap: {dp}'
    old_groups = QUALITY['old'] or sorted(set(groups)-set(QUALITY['arriving']))
    old_mask = np.isin(groups, old_groups)
    arriving_mask = np.isin(groups, QUALITY['arriving'])
    assert old_mask.any() and arriving_mask.any()
    with np.load(OC/f'{a.card}_in_bank.npz') as b:
        gg = b['source' if a.card == 'card006' else 'language'].astype(str)
        rng = np.random.default_rng(11008)
        ix = np.concatenate([rng.choice(np.flatnonzero(gg==g),20000//len(set(gg)),replace=False)
                             for g in sorted(set(gg))])
        anchor_x, anchor_ids = b['replay_X'][ix], b['replay_ids'][ix]
    with np.load(OC/f'{a.card}_confirm_bank.npz') as b:
        confirm_x, confirm_ids = b['replay_X'], b['replay_ids']
        confirm_groups = b['source' if a.card == 'card006' else 'language'].astype(str)
    assert not np.intersect1d(anchor_ids, confirm_ids).size
    radius = float(json.loads(Path(CFG['anchor_meta']).read_text())[CFG['radius_key']])
    common = {'val_ids':vi, 'ref_ids':ri, 'val_groups':groups,
              'confirmation_ids':confirm_ids, 'confirmation_groups':confirm_groups,
              'frame_anchor_ids':anchor_ids, 'radius':np.array(radius)}
    np.savez(out/'identity.npz', **common)
    specs = {'frozen':CFG['orig_champ'], 'anchored':CFG['controls']['anchored'][0]}
    for arm in ('in','out'):
        specs.update({f'{arm}-{s}':CFG['outd']/f'snapshots-{arm}'/f'model-step{s}.pt'
                      for s in (35000,70000)})
        specs[f'{arm}-140000'] = CFG['outd']/f'model-{arm}.pt'
    instrument = [(str(f),f.stat().st_size,f.stat().st_mtime_ns) for f in
                  (seal/'ref_hd.f16.npy',seal/'val_hd.f16.npy',seal/'ref_idx.npy',seal/'val_idx.npy')]
    extra_identity = [sha(OC/f'{a.card}_{k}_bank.npz') for k in ('in','confirm')]
    coords, provenance = {}, {}
    for tag, path in specs.items():
        digest = sha(path)
        key = hashlib.sha256(json.dumps([digest,instrument,extra_identity,'frame-20k-seed11008-v1']).encode()).hexdigest()[:20]
        cp = cache/f'{tag}-{key}.npz'
        if cp.exists():
            with np.load(cp) as z: c = {k:z[k] for k in ('ref','val','anchor','confirmation')}
        else:
            mo = ParametricUMAP.load(str(path), device='cpu')
            mo.model.eval()
            alias = tag.replace('-140000','')
            old_key = hashlib.sha256(json.dumps([digest,instrument,'L2-f32-v1']).encode()).hexdigest()[:20]
            old_cache = OC/'projection-cache'/f'{a.card}-{alias}-{old_key}.npz'
            if old_cache.exists():
                with np.load(old_cache) as z: c = {'ref':z['ref'], 'val':z['val']}
            else:
                c = {'ref':project(mo,ref,True), 'val':project(mo,val,True)}
            c.update(anchor=project(mo,anchor_x,False), confirmation=project(mo,confirm_x,False))
            del mo
            np.savez(cp, **c)
        for name, n in [('ref',len(ref)),('val',len(val)),('anchor',len(anchor_ids)),('confirmation',len(confirm_ids))]:
            assert c[name].shape == (n,2) and np.isfinite(c[name]).all()
        coords[tag] = c
        provenance[tag] = {'model':str(path),'sha256':digest,'cache':str(cp)}
        print(tag,'projected/cached',round(time.time()-started,1),flush=True)
    budgets = [50,100,250,500,1000,2000]
    perq, report = {}, {}
    base = coords['frozen']

    def score(tag, c, native_c=None, transform=None):
        index = faiss.IndexFlatL2(2)
        index.add(np.ascontiguousarray(c['ref'],np.float32))
        _,nn = index.search(np.ascontiguousarray(c['val'],np.float32),max(budgets))
        # Obtain exact inspection ranks for each high-D truth neighbor.
        ranks = np.full(truth.shape, max(budgets)+1, np.int32)
        for j in range(truth.shape[1]):
            matches = nn == truth[:,j,None]
            ranks[:,j] = np.where(matches.any(1),matches.argmax(1)+1,max(budgets)+1)
        del nn
        pq = {B:(ranks<=B).mean(1) for B in budgets}
        d = np.linalg.norm(c['confirmation']-base['confirmation'],axis=1)/radius
        native = np.linalg.norm((native_c or c)['confirmation']-base['confirmation'],axis=1)/radius
        perq.update({f'{tag}_B{B}':x for B,x in pq.items()})
        perq[f'{tag}_movement'] = d
        perq[f'{tag}_native_movement'] = native
        rec = {'movement':stats(d),'native_movement':stats(native),'frame':transform,
               'per_group_movement':{g:stats(d[confirm_groups==g]) for g in sorted(set(confirm_groups))},
               'quality':{}}
        for B, x in pq.items():
            bygroup = {g:float(x[groups==g].mean()) for g in sorted(set(groups))}
            oldloss = {g:float((perq[f'frozen_B{B}']-x)[groups==g].mean()) for g in old_groups}
            rec['quality'][str(B)] = {'arriving':float(x[arriving_mask].mean()),
                'old_balanced':float(np.mean([bygroup[g] for g in old_groups])),
                'old_mean_loss_vs_frozen':float(np.mean(list(oldloss.values()))),
                'old_worst_loss_vs_frozen':max(oldloss.values()),'per_group':bygroup}
        report[tag] = rec
        print(tag,'scored',rec['quality']['250']['arriving'],rec['movement']['p99'],flush=True)

    aligned = {}
    for tag,c in coords.items():
        _, info = frame.rigid_align(c['anchor'],base['anchor'])
        R,t = np.asarray(info['R']),np.asarray(info['t'])
        ac = {k:np.asarray(v,np.float64)@R.T+t for k,v in c.items()}
        aligned[tag] = ac
        score(tag,ac,c,info)
    for parent in ('anchored','in-140000','out-140000'):
        for alpha in (.25,.5,.75):
            c = {k:base[k]+alpha*(aligned[parent][k]-base[k]) for k in base}
            tag = f'{parent}-blend{alpha}'
            score(tag,c,transform={'blend_alpha':alpha,'parent':parent,'uses_parent_fixed_rigid_transform':True})
            target = alpha*perq[f'{parent}_movement']
            assert np.max(np.abs(perq[f'{tag}_movement']-target)) < 1e-7
    np.savez(out/'per-query.npz', **common, **perq)
    result = {'schema':'sharpening-development-baselines-v1','card':a.card,'status':'COMPLETE',
              'created_unix':time.time(),'elapsed_cpu_wall_seconds':time.time()-started,
              'frame':'20K group-balanced original active anchors, seed11008; rotation+translation only',
              'radius':radius,'evaluation_role':'Previously exposed development seal/confirmation, no deployment selection',
              'old_groups':old_groups,'provenance':provenance,'results':report}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# {a.card}: development dose and blend baselines','',
           'CPU only. Fixed 20K training-anchor frame; this is not the original full-active gauge.',
           'Previously inspected seals are development data. No original gate is changed.','',
           '| Head | Arriving B250 | Arriving B2000 | Old B2000 mean loss | Old B2000 worst loss | Unseen mean | Unseen p99 | >.05 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for tag,r in report.items():
        q,m=r['quality'],r['movement']
        lines.append(f"| {tag} | {q['250']['arriving']:.6f} | {q['2000']['arriving']:.6f} | {q['2000']['old_mean_loss_vs_frozen']:.6f} | {q['2000']['old_worst_loss_vs_frozen']:.6f} | {m['mean']:.6f} | {m['p99']:.6f} | {m['frac_gt05']:.4%} |")
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()

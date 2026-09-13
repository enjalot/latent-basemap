"""Fixed-head gradients from exactly eight actual production batches; no updates."""
import hashlib
import math
import numpy as np
import torch
import card084_common as C
from basemap.pumap.parametric_umap.bounded_attraction import add_attraction

class CalibrationComplete(Exception):
    pass

def tensor_sha(t):
    a=t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def coefficient_from_ratios(ratios):
    if len(ratios)!=8 or not all(math.isfinite(x) and x>0 for x in ratios):
        raise ValueError('STOP: eight finite positive raw A/G ratios required')
    if max(ratios)/min(ratios)>10:
        raise ValueError('STOP: raw A/G spread exceeds10')
    c=0.1/float(np.median(ratios))
    if not math.isfinite(c) or c<=0:
        raise ValueError('STOP: coefficient invalid')
    return c

class FixedHeadCalibration:
    def __init__(self, destination):
        self.destination=destination
        self.destination.mkdir(parents=True,exist_ok=False)
        self.batches=[];self.stats={a:[] for a in C.ARMS}
        self.initial_sha=None
        self.ids=np.load(C.D/'train-ids.npy',mmap_mode='r')
        self.excluded=np.unique(np.r_[np.load('/data2/monet/eval-common-v2/ref_idx.npy'),np.load('/data2/monet/eval-common-v2/val_idx.npy')])
        self.pre=None
    def before_batch(self, loader):
        self.pre=loader.gen.get_state().cpu().clone()
    def __call__(self,p,base,src,dst,targets,scale,loader):
        assert len(self.batches)<8 and p._train_stats['optimizer_steps_attempted']==0
        sd=C.state_sha(p.model.state_dict())
        if self.initial_sha is None:
            self.initial_sha=sd
            original=torch.load(C.CHAMP,map_location='cpu',weights_only=False)['model_state_dict']
            assert sd==C.state_sha(original),'calibration must use original head'
        assert sd==self.initial_sha,'head updated during calibration'
        assert loader.n_nodes==C.N and len(self.ids)==C.N
        assert p._train_stats['use_amp'] and p._train_stats['amp_dtype']=='float16'
        assert len(targets)==C.BATCH and int((targets>.5).sum())==1638
        assert p._pipeline_info['pipeline']=='device' and p._pipeline_info['x_residency']=='device_fp16'
        assert torch.isfinite(base) and torch.isfinite(scale).all() and (scale>0).all()
        sr=loader._last_all_src.detach().cpu().numpy();dr=loader._last_all_dst.detach().cpu().numpy()
        assert not np.isin(self.ids[np.unique(np.r_[sr,dr])],self.excluded).any()
        # Production PERM selects graph rows; check exact selected positive endpoints.
        count=1638;pos_end=loader.pos_idx
        ix=loader.perm[pos_end-count:pos_end]
        assert torch.equal(loader.sources_t[ix].long(),loader._last_all_src[:count].long())
        assert torch.equal(loader.targets_t[ix].long(),loader._last_all_dst[:count].long())
        path=self.destination/f'batch{len(self.batches):02d}.npz'
        np.savez(path,src=sr,dst=dr,targets=targets.detach().cpu().numpy(),
                 pair_scale=scale.detach().cpu().numpy(),rng_before=self.pre.numpy(),
                 rng_after=loader.gen.get_state().cpu().numpy(),positive_edge_rows=ix.cpu().numpy())
        batch={'index':len(self.batches),'path':str(path),'sha':C.sha(path),
               'rng_before_sha':tensor_sha(self.pre),'rng_after_sha':tensor_sha(loader.gen.get_state()),
               'head_sha':sd,'baseline_value':float(base.detach())}
        params=tuple(p.model.parameters())
        g=torch.autograd.grad(base,params,retain_graph=True)
        g2=sum(v.detach().double().square().sum() for v in g)
        G=float(g2.sqrt());assert math.isfinite(G) and G>0,'STOP baseline gradient'
        for family in C.ARMS:
            extra=add_attraction(base.new_zeros(()),src,dst,targets>.5,scale,
                                 coefficient=1.,family=family,delta=C.DELTA)
            a=torch.autograd.grad(extra,params,retain_graph=True)
            a2=sum(v.detach().double().square().sum() for v in a)
            dot=sum((u.detach().double()*v.detach().double()).sum() for u,v in zip(g,a))
            A=float(a2.sqrt());dot=float(dot)
            assert math.isfinite(A) and A>0 and math.isfinite(dot),'STOP extra gradient'
            self.stats[family].append({'G':G,'A':A,'A_over_G':A/G,'dot':dot,
                'cosine':dot/(G*A),'extra_value':float(extra.detach()),'baseline_clips':G>1.})
            del a
        self.batches.append(batch)
        if len(self.batches)==8:
            self.pipeline=dict(p._pipeline_info)
            self.precision={'use_amp':p._train_stats['use_amp'],'amp_dtype':p._train_stats['amp_dtype']}
            assert p._train_stats['optimizer_steps_attempted']==0
            raise CalibrationComplete()
    def result(self):
        assert len(self.batches)==8
        arms={}
        for family,rows in self.stats.items():
            ratios=[r['A_over_G'] for r in rows];c=coefficient_from_ratios(ratios)
            for row in rows:
                n2=row['G']**2+2*c*row['dot']+(c*row['A'])**2
                assert math.isfinite(n2) and n2>=0
                row.update(combined_norm=math.sqrt(n2),combined_clips=n2>1.,added_ratio=c*row['A_over_G'])
            arms[family]={'coefficient':c,'raw_ratios':ratios,'max_over_min':max(ratios)/min(ratios),
                'quantiles':np.quantile(ratios,[0,.25,.5,.75,1]).tolist(),'batches':rows}
        batch_digest=hashlib.sha256(''.join(x['sha'] for x in self.batches).encode()).hexdigest()
        return {'PASS':True,'delta':C.DELTA,'arms':arms,'batches':self.batches,'batches_sha':batch_digest,
            'runtime_sha':C.source_check(),'original_head_sha':C.sha(C.CHAMP),'fixed_state_sha':self.initial_sha,
            'input_manifest_sha':C.sha(C.D/'inputs-manifest.json'),'graph_sha':C.sha(C.D/'edges-fixed15.npz'),
            'radii_sha':C.sha(C.D/'radii.npy'),'uniform_q_sha':C.sha(C.O/'degree-noise-readiness/uniform.npy'),
            'bank_sha':C.BANK_SHA,'pipeline':self.pipeline,'precision':self.precision,'loaded_modules':C.loaded_modules(),'successful_updates':0,
            'scope':'Eight actual production batches0-7 seed42; fixed original head, raw unscaled gradients before clipping. No Adam matching.'}

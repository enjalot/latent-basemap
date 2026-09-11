"""Measure actual saved fine-tuning endpoints on the frozen band discovery panel.
Straight lines on the plot join sparse observations, not a reconstructed training path.
"""
import json
from pathlib import Path
import numpy as np
import torch
from band_projection_canary import raw_features, ParametricUMAP
from band_scan import ROOT,SB,sha

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    r=json.loads((ROOT/'projection-canary.json').read_text());panel=r['panel'];gids=np.load(ROOT/'survey-identities.npz')['full_ids'];a=np.load(ROOT/'detail-neighbors.npz');nns=dict(zip(a['query_survey_indices'],a['neighbor_survey_indices']))
    query=np.array([p['full_id'] for p in panel]);neighbors=np.array([gids[nns[p['survey_index']][:15]] for p in panel]);allids=np.unique(np.r_[query,neighbors.ravel()]);X=raw_features(allids);qi=np.searchsorted(allids,query);ni=np.searchsorted(allids,neighbors)
    paths={'t0':SB/'dino-arrival-t0/champion-bs16k/model.pt'}
    for arm in ['in','out']:
        for step in [35000,70000,140000]:paths[f'{arm}_{step}']=SB/f'dino-arrival-t0/replay-updates/snapshots-{arm}/model-step{step}.pt'
    coords={};offsets={}
    for tag,p in paths.items():
        model=ParametricUMAP.load(str(p),device='cpu').model.eval()
        with torch.inference_mode():z=np.concatenate([model(torch.from_numpy(X[i:i+256])).numpy() for i in range(0,len(X),256)])
        assert np.isfinite(z).all();coords[tag]=z[qi];nz=z[ni];scale=np.sqrt(np.mean(np.sum((nz-nz.mean(1,keepdims=True))**2,axis=2),axis=1)*2)
        offsets[tag]=np.linalg.norm(z[qi]-nz.mean(1),axis=1)/np.maximum(scale,1e-12)
    result={'status':'exploratory; selected panel, not representative evaluation or a deployment gate','panel':panel,'fixed_T0_radius':33.6717,'frame':'native saved model frame; no alignment on these queries','caveat':'Card006 is a different training family from the 2M/6M/12M ladder. Points selected for ladder geometry, not drift. Checkpoint paths contain only 0/35K/70K/140K observations; connecting lines do not reveal intervening SGD trajectories. Training membership not used to label these points unseen.','models':{tag:{'path':str(p),'sha256':sha(p),'coordinates':coords[tag].tolist(),'native_movement_over_T0_radius':(np.linalg.norm(coords[tag]-coords['t0'],axis=1)/33.6717).tolist(),'query_HD_neighbor_centroid_offset_over_pair_rms':offsets[tag].tolist()} for tag,p in paths.items()},'source_sha256':sha(Path(__file__))}
    (ROOT/'snapshot-paths.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for a,arm in zip(ax,['in','out']):
        steps=[0,35000,70000,140000];tags=['t0']+[f'{arm}_{s}' for s in steps[1:]]
        for i,p in enumerate(panel):
            if p['role']!='center':continue
            move=[result['models'][t]['native_movement_over_T0_radius'][i] for t in tags]
            a.plot(steps,move,'o-',lw=1,label=f'case {p["case"]}')
        a.axhline(.05,color='black',ls='--',lw=.8);a.set_title(f'Card006 {arm.upper()} replay: selected center images');a.set_xlabel('Successful update checkpoint');a.set_ylabel('Native displacement / original T0 radius')
    ax[1].legend(fontsize=7,ncol=3);fig.suptitle('Sparse saved checkpoints; lines are guides, not observed trajectories');fig.tight_layout();fig.savefig(ROOT/'snapshot-movement.png',dpi=150);plt.close(fig)
    print('DONE',len(allids),'feature rows',flush=True)

if __name__=='__main__':main()

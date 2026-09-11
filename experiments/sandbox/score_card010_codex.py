"""Card010 exploratory scoring on the ORIGINAL common evaluation instrument.
CPU only; cached projections with model/input identities. Original viability remains false.
Run repeatedly while a corrected control is pending; final score requires every primary head.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
import json,time,hashlib
from pathlib import Path
import numpy as np
from score_card009 import project, recall, sha, write_json, BUDGETS, ParametricUMAP, torch, faiss
from scipy.spatial.distance import cdist,pdist
from scipy.stats import spearmanr
OC=Path('/data/latent-basemap/sandbox/overseer-codex');OUT=OC/'card010-scoring'
TD=Path('/data/latent-basemap/sandbox/card010-train');SEAL=Path('/data2/monet/eval-common-v2')
HEADS={h:TD/f'model-{h}.pt' for h in ['fixed15','adaptive']}
HEADS['fixed12']=TD/'fixed12-constant-repair/model-fixed_mean.pt'
HEADS['historical_plateau12']=TD/'model-fixed_mean.pt'
PRIMARY=['fixed15','adaptive','fixed12'];SEED=10010


def ranks(x):
    np.fill_diagonal(x,np.inf)
    order=np.argsort(x,axis=1,kind='stable');rank=np.empty(order.shape,dtype='i4')
    rank[np.arange(len(x))[:,None],order]=np.arange(1,len(x)+1)
    return order,rank


def reliability(xy,hi,hr,labels,hcent):
    n=len(xy);k=15;lo,lr=ranks(cdist(xy,xy,'sqeuclidean'))
    rows=np.arange(n)[:,None];h_at_l=hr[rows,lo[:,:k]];l_at_h=lr[rows,hi[:,:k]]
    factor=2/(n*k*(2*n-3*k-1))
    centroid=np.array([xy[labels==i].mean(0) for i in range(len(hcent))])
    hd=pdist(hcent,'cosine');ld=pdist(centroid,'euclidean')
    return {'sample':n,'k':k,'trustworthiness':float(1-factor*np.maximum(h_at_l-k,0).sum()),
            'continuity':float(1-factor*np.maximum(l_at_h-k,0).sum()),
            'map_k15_outside_encoder_k60_fraction':float((h_at_l>60).mean()),
            'encoder_cluster_centroid_distance_spearman':float(spearmanr(hd,ld).statistic),
            'cluster_count':len(hcent)}


def contrast(a,b,groups):
    rng=np.random.default_rng(SEED);parts=[(a-b)[groups==g] for g in np.unique(groups)]
    draws=np.zeros(2000)
    for d in parts:
        for i in range(len(draws)): draws[i]+=d[rng.integers(0,len(d),len(d))].mean()/len(parts)
    return {'delta':float(np.mean([d.mean() for d in parts])), 'ci95':np.percentile(draws,[2.5,97.5]).tolist()}


def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    names=['ref_hd.f16.npy','val_hd.f16.npy','truth_val.npy','ref_idx.npy','val_idx.npy','val_source.npy']
    instrument={n:sha(SEAL/n) for n in names}
    code_sha=sha(__file__);helper_sha=sha(Path(__file__).with_name('score_card009.py'))
    ref=np.load(SEAL/names[0],mmap_mode='r');val=np.load(SEAL/names[1],mmap_mode='r');truth=np.load(SEAL/names[2]);rid=np.load(SEAL/names[3]);qid=np.load(SEAL/names[4]);groups=np.load(SEAL/names[5],allow_pickle=True).astype(str);cohorts=np.unique(groups)
    assert truth.shape==(len(val),15) and (truth>=0).all() and (truth<len(ref)).all()
    draw=np.load('/data/latent-basemap/substrates/card010-adaptive/draw_ids.npy')
    assert not np.isin(draw,np.r_[rid,qid]).any()
    rng=np.random.default_rng(SEED);panel=np.sort(np.concatenate([rng.choice(np.flatnonzero(groups==g),200,replace=False) for g in cohorts]))
    H=np.array(val[panel],dtype='f4');H/=np.linalg.norm(H,axis=1,keepdims=True).clip(1e-12)
    hi,hr=ranks(cdist(H,H,'cosine'))
    km=faiss.Kmeans(H.shape[1],16,niter=20,nredo=1,seed=SEED,verbose=False);km.train(H);_,li=km.index.search(H,1);labels=li[:,0]
    assert len(np.unique(labels))==16
    arrays={'reference_ids':rid,'query_ids':qid,'query_sources':groups,'truth':truth,'panel_query_ids':qid[panel],'panel_local':panel,'panel_hd_cluster':labels}
    provenance={'scorer_sha256':code_sha,'helper_sha256':helper_sha,'instrument':instrument,'heads':{}}
    report={};pending=[]
    example_q=np.array([rng.choice(np.flatnonzero(groups==g)) for g in cohorts]);examples=[]
    for h,mp in HEADS.items():
        if not mp.exists(): pending.append(h);continue
        ident={'model_path':str(mp),'model_sha256':sha(mp),'instrument':instrument,'scorer_sha256':code_sha,'helper_sha256':helper_sha}
        cp=OUT/f'{h}.npz';jp=OUT/f'{h}-identity.json'
        if cp.exists() and jp.exists() and json.loads(jp.read_text())==ident:
            z=np.load(cp);pq={b:z[f'B{b}'] for b in BUDGETS};rc=z['ref_xy'];vc=z['val_xy'];top=z['top15']
        else:
            print('Scoring',h,flush=True)
            model=ParametricUMAP.load(str(mp),device='cpu').model.eval()
            rc=project(model,ref,True);vc=project(model,val,True);pq=recall(rc,vc,truth)
            ix=faiss.IndexFlatL2(2);ix.add(rc);_,top=ix.search(vc,15)
            np.savez(cp,**{f'B{b}':v for b,v in pq.items()},ref_xy=rc,val_xy=vc,top15=top)
            write_json(jp,ident);del model
        provenance['heads'][h]=ident
        by={str(b):{g:float(pq[b][groups==g].mean()) for g in cohorts} for b in BUDGETS}
        report[h]={'by_source':by,'equal_cohort':{str(b):float(np.mean(list(by[str(b)].values()))) for b in BUDGETS},
                   'micro':{str(b):float(pq[b].mean()) for b in BUDGETS},
                   'real5':{str(b):float(pq[b][np.isin(groups,['laion','coyo','commoncatalog-cc-by','megalith10m','cc12m'])].mean()) for b in BUDGETS},
                   'reliability':reliability(vc[panel].astype('f8'),hi,hr,labels,km.centroids)}
        for b,v in pq.items():
            assert np.isfinite(v).all();arrays[f'{h}_B{b}']=v
        arrays[f'{h}_val_xy']=vc;arrays[f'{h}_ref_xy']=rc;arrays[f'{h}_top15']=top
        print(json.dumps({'head':h,'equal_cohort':report[h]['equal_cohort'],'reliability':report[h]['reliability']}),flush=True)
    write_json(OUT/'provenance.json',provenance);write_json(OUT/'head-summaries.json',report)
    for q in example_q:
        examples.append({'query_pool_row':int(qid[q]),'source':str(groups[q]),'truth_pool_rows':rid[truth[q]].tolist(),
                         'map_neighbors':{h:rid[arrays[f'{h}_top15'][q]].tolist() for h in report}})
    write_json(OUT/'examples.json',examples)
    np.savez(OUT/'per-query.npz',**arrays)
    if any(h in pending for h in PRIMARY):
        write_json(OUT/'status.json',{'status':'WAITING_CORRECTED_CONTROL','pending':pending});print('Pending',pending,flush=True);return
    comparisons={};gate={}
    for ctrl in ['fixed15','fixed12']:
        contrasts={str(b):contrast(arrays[f'adaptive_B{b}'],arrays[f'{ctrl}_B{b}'],groups) for b in [250,2000]}
        per={str(b):{g:report['adaptive']['by_source'][str(b)][g]-report[ctrl]['by_source'][str(b)][g] for g in cohorts} for b in [250,2000]}
        checks={'B250_gain_ge_0015':contrasts['250']['delta']>=.015,'B250_ci_positive':contrasts['250']['ci95'][0]>0,
                'B2000_delta_ge_minus0005':contrasts['2000']['delta']>=-.005,
                'every_cohort_both_budgets_loss_le_001':all(v>=-.01 for d in per.values() for v in d.values())}
        comparisons[ctrl]={'aggregate':contrasts,'by_source_delta':per,'checks':checks};gate[ctrl]=all(checks.values())
    weights=json.loads((OC/'card009-result.json').read_text())['natural_pool_weights']
    natural={h:{str(b):sum(weights[g]*report[h]['by_source'][str(b)][g] for g in weights)/sum(weights.values()) for b in BUDGETS} for h in report}
    result={'status':'SCORED','exploratory':True,'original_preregistered_viability':False,'primary':'Equal-cohort mean across all9 original eval-common-v2 cohorts; original reference and encoder k15 truth',
            'heads':report,'comparisons':comparisons,'NUMERICAL_QUALITY_SCREEN_PASS':all(gate.values()),
            'reliability_and_visual_review':'Separate review required; numerical pass alone is not promotion or deployment',
            'natural_weights':weights,'natural_query_weighted':natural,'uncertainty':'2000 paired source-stratified equal-cohort bootstrap draws,seed10010; one training trajectory; development data',
            'historical_plateau12':'Diagnostic only; violates original constant LR and is excluded from matched controls',
            'example_path':str(OUT/'examples.json'),'provenance_path':str(OUT/'provenance.json'),'cpu_wall_s':time.time()-start}
    write_json(OUT/'result.json',result);write_json(OC/'card010-score-codex.json',result);write_json(OUT/'status.json',{'status':'SCORED'})
    print(json.dumps({'comparisons':comparisons,'NUMERICAL_QUALITY_SCREEN_PASS':all(gate.values())},indent=2),flush=True)


if __name__=='__main__':main()

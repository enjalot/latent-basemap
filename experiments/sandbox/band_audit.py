"""Independent checks of saved band diagnostics and reference identity."""
import json,time
from pathlib import Path
import numpy as np
from band_scan import ROOT,HEADS,sha
from band_forensics import features

def main():
    started=time.time();identity=np.load(ROOT/'survey-identities.npz');ids=identity['full_ids'];detail=np.load(ROOT/'detail-neighbors.npz')
    assert np.array_equal(ids,detail['reference_full_ids']) and len(np.unique(ids))==200000
    q=detail['query_survey_indices'];nn=detail['neighbor_survey_indices'];assert (nn!=q[:,None]).all()
    X=features(ids);panel=np.linspace(0,len(q)-1,12,dtype=int);sim=X[q[panel]]@X.T
    exact=[]
    for j,row in enumerate(panel):
        sim[j,q[row]]=-np.inf;order=np.argsort(-sim[j])[:15]
        # Floating-point dot-product ties can swap near-equal boundary neighbors.
        expected=nn[row,:15];delta=float(sim[j,order[-1]]-sim[j,expected].min());assert delta<2e-6,delta
        exact.append({'q':int(q[row]),'set_overlap_k15':len(set(order)&set(expected)),'boundary_similarity_gap':delta})
    result=json.loads((ROOT/'forensics-result.json').read_text());checks={}
    for h,p in HEADS.items():
        g=np.load(ROOT/(h+'-geometry.npz'));saved=np.load(p/'coords.f32.npy',mmap_mode='r')
        np.testing.assert_array_equal(g['coords'],saved[ids]);assert np.isfinite(g['coords']).all()
        flags=(g['linearity32']>=.9)&(g['linearity96']>=.8)&(g['balance']>=.2)&g['probe_support']&(g['contrast']>=1.3)
        np.testing.assert_array_equal(flags,g['ridge_candidate'])
        checks[h]={'coords_identity':True,'flags_consistent':True,'flag_count':int(flags.sum())}
        # Independent brute-force map rank check for each case's query, reported in query_B250[0].
        for c in result['cases']:
            i=c['survey_index'];dist=np.sum((g['coords']-g['coords'][i])**2,axis=1);dist[i]=np.inf;near=np.argpartition(dist,250)[:250]
            hd=nn[np.flatnonzero(q==i)[0],:15];rec=len(set(near)&set(hd))/15
            assert abs(rec-c['comparisons'][h]['query_B250'][0])<1e-9
    record={'status':'PASS','reference_identity':True,'brute_force_hd_checks':exact,'saved_map_and_geometry_checks':checks,'brute_force_case_query_B250':True,'limitations':'200K reference only; no whole-corpus exact neighbor guarantee, semantic labels or causal claim. Candidate selection exploratory. Query precision checks permit <2e-6 similarity ties.','source_sha256':sha(Path(__file__)),'forensics_sha256':sha(ROOT/'forensics-result.json'),'wall_seconds':time.time()-started}
    (ROOT/'audit.json').write_text(json.dumps(record,indent=2)+'\n');print('PASS',flush=True)

if __name__=='__main__':main()

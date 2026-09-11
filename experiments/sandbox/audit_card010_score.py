"""Independent arithmetic and rank checks for the restored card010 evaluation."""
from score_card010_codex import *
from sklearn.manifold import trustworthiness

def main():
    a=np.load(OUT/'per-query.npz');heads=[h for h in HEADS if f'{h}_B250' in a];groups=a['query_sources'];truth=a['truth'];rng=np.random.default_rng(7110);query=np.sort(rng.choice(len(groups),64,replace=False))
    checked={}
    for h in heads:
        rc=a[h+'_ref_xy'];vc=a[h+'_val_xy'];idx=faiss.IndexFlatL2(2);idx.add(rc);_,nn=idx.search(vc[query],2000)
        for b in BUDGETS:
            expected=np.array([len(set(truth[q])&set(n[:b]))/15 for q,n in zip(query,nn)])
            assert np.array_equal(expected,a[f'{h}_B{b}'][query])
        for b0,b1 in zip(BUDGETS[:-1],BUDGETS[1:]):assert (a[f'{h}_B{b0}']<=a[f'{h}_B{b1}']).all()
        checked[h]=True
    summary=json.loads((OUT/'head-summaries.json').read_text());panel=a['panel_local'];H=np.load(SEAL/'val_hd.f16.npy')[panel].astype('f4');H/=np.linalg.norm(H,axis=1,keepdims=True)
    y=a['fixed15_val_xy'][panel].astype('f8')
    independent_t=trustworthiness(H,y,n_neighbors=15,metric='cosine')
    independent_c=trustworthiness(y,H,n_neighbors=15,metric='euclidean')
    # Euclidean versus cosine ordering can differ for tied fp16 embeddings; record discrepancy.
    assert abs(independent_t-summary['fixed15']['reliability']['trustworthiness'])<1e-4
    assert abs(independent_c-summary['fixed15']['reliability']['continuity'])<1e-4
    test=contrast(np.r_[np.ones(10),np.ones(30)*3],np.zeros(40),np.array(['a']*10+['b']*30));assert test['delta']==2 and test['ci95']==[2,2]
    result={'status':'PASS','retrieval64queries_matches_independent_set_intersection_all_budgets':checked,'budget_curve_monotone':True,'equal_cohort_bootstrap_unequal_size_sanity':True,'sklearn_trustworthiness_check':float(independent_t),'sklearn_reverse_trustworthiness_check':float(independent_c)}
    if (OUT/'result.json').exists():
        r=json.loads((OUT/'result.json').read_text());flags=[]
        for ctrl in ['fixed15','fixed12']:
            changes={b:np.array([(a[f'adaptive_B{b}']-a[f'{ctrl}_B{b}'])[groups==g].mean() for g in np.unique(groups)]) for b in [250,2000]}
            c=r['comparisons'][ctrl];assert abs(changes[250].mean()-c['aggregate']['250']['delta'])<1e-12
            flag=bool(changes[250].mean()>=.015 and c['aggregate']['250']['ci95'][0]>0 and changes[2000].mean()>=-.005 and all((v>=-.01).all() for v in changes.values()))
            assert flag==all(c['checks'].values());flags.append(flag)
        assert all(flags)==r['NUMERICAL_QUALITY_SCREEN_PASS'];result['unrounded_gate_recomputation_matches']=True
    write_json(OC/'card010-score-independent-audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

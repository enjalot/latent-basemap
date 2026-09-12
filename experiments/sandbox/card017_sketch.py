"""Optimized greedy plaid covering; author-rule cover, distribution-equivalent draws.

Method: Hie et al., Cell Systems2019. Author algorithm reference is pinned in OC literature.
Independent implementation. No claim of bitwise RNG equivalence to Python-set sampling.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def intervals(X, orders, ranges, unit):
    n,d=X.shape; table=np.zeros((n,d),dtype=np.int32)
    for j in range(d):
        if ranges[j]<=unit: continue
        at=0.; label=-1
        for z in range(n):
            i=orders[z,j]
            if label<0 or at+unit<X[i,j]:
                at=X[i,j]; label+=1
            table[i,j]=label
    return table


def cover(X,k,alpha=.1,max_iter=200):
    X=np.asarray(X,dtype='f8'); assert np.isfinite(X).all() and X.ndim==2
    X=X-X.min(0); scale=X.max(); assert scale>0,'degenerate feature cover'
    X/=scale; ranges=np.ptp(X,axis=0); orders=np.argsort(X,axis=0,kind='stable')
    low=0.; high=float(ranges.max()); unit=(low+high)/4.; trace=[]
    for it in range(max_iter):
        table=intervals(X,orders,ranges,unit)
        cells,labels,count=np.unique(table,axis=0,return_inverse=True,return_counts=True)
        bins=len(cells); trace.append({'iteration':it,'unit':unit,'occupied':bins})
        if k*(1-alpha)<=bins<=k*(1+alpha):
            return {'labels':labels.astype('i4'),'counts':count,'cells':cells,'unit':unit,'trace':trace,'scale':float(scale)}
        if bins>k*(1+alpha): low=unit
        else: high=unit
        unit=(low+high)/2.
        if high-low<1e-20: break
    raise RuntimeError(f'cover search failed declared tolerance: last {trace[-1]}')


def sample_cover(labels,n,seed):
    """One uniform row per nonempty box in randomized cycles, no replacement."""
    labels=np.asarray(labels); assert 0<=n<=len(labels)
    rng=np.random.default_rng(seed)
    # Independent random order inside every box; grouping preserves that order.
    perm=rng.permutation(len(labels)); order=perm[np.argsort(labels[perm],kind='stable')]
    _,counts=np.unique(labels[order],return_counts=True)
    starts=np.r_[0,np.cumsum(counts)[:-1]]; positions=np.zeros(len(counts),dtype='i8')
    chosen=[]; remaining=n
    while remaining:
        active=np.flatnonzero(positions<counts); cycle=rng.permutation(active)[:remaining]
        chosen.append(order[starts[cycle]+positions[cycle]]); positions[cycle]+=1; remaining-=len(cycle)
    out=np.concatenate(chosen) if chosen else np.empty(0,dtype='i8')
    assert len(np.unique(out))==n
    return out


def literal_intervals(X,unit):
    X=np.asarray(X,dtype='f8'); ranges=np.ptp(X,axis=0); table=np.zeros(X.shape,dtype='i4')
    for d in range(X.shape[1]):
        if ranges[d]<=unit: continue
        start=None; label=-1
        for i in np.argsort(X[:,d],kind='stable'):
            if start is None or start+unit<X[i,d]: start=float(X[i,d]); label+=1
            table[i,d]=label
    return table


def canary():
    rng=np.random.default_rng(17017); checks={}
    cases=[rng.normal(size=(512,8)),np.repeat(rng.normal(size=(64,4)),3,axis=0),
           np.array([[0,0],[1,0],[1,1],[2,1],[3,2]],dtype='f8')]
    for j,x in enumerate(cases):
        x=x-x.min(0); x/=x.max(); order=np.argsort(x,axis=0,kind='stable')
        for unit in [.125,.25,.5,1.]:
            checks[f'literal_cover_{j}_{unit}']=np.array_equal(intervals(x,order,np.ptp(x,axis=0),unit),literal_intervals(x,unit))
    labels=np.repeat(np.arange(4),[1,2,10,20]); draw=sample_cover(labels,15,42)
    checks['first_cycle_all_boxes']=len(np.unique(labels[draw[:4]]))==4
    checks['all_unique']=len(np.unique(draw))==15
    checks['deterministic']=np.array_equal(draw,sample_cover(labels,15,42))
    checks['all_rows_exact']=np.array_equal(np.sort(sample_cover(labels,len(labels),42)),np.arange(len(labels)))
    checks['input_unchanged']=np.array_equal(labels,np.repeat(np.arange(4),[1,2,10,20]))
    c=cover(rng.normal(size=(2000,8)),600); checks['search_tolerance']=540<=len(c['counts'])<=660
    return {k:bool(v) for k,v in checks.items()}

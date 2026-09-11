"""Stream all 103.8M saved coordinates into fixed case windows, CPU only.
No resampling or per-window endpoint re-selection: bounds derive from the frozen survey cases.
"""
import json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from band_scan import ROOT,HEADS,sha

def main():
    started=time.time();cases=json.loads((ROOT/'forensics-result.json').read_text())['cases'];selected=[3,4,10];out={};grids={};bounds={}
    for h,path in HEADS.items():
        z=np.load(ROOT/(h+'-geometry.npz'))['coords'];windows={}
        for n in selected:
            c=cases[n-1];ix=np.r_[c['center_indices'],c['endpoint_indices'][0],c['endpoint_indices'][1]];low=z[ix].min(0);high=z[ix].max(0);pad=np.maximum((high-low)*.25,.05);windows[n]=(low-pad,high+pad)
        H={n:np.zeros((512,512),np.uint64) for n in selected};count={n:0 for n in selected};mm=np.load(path/'coords.f32.npy',mmap_mode='r')
        for lo in range(0,len(mm),1_000_000):
            a=np.asarray(mm[lo:lo+1_000_000])
            for n,(low,high) in windows.items():
                mask=((a>=low)&(a<=high)).all(1);v=a[mask];count[n]+=len(v)
                if len(v):hist,*_=np.histogram2d(v[:,0],v[:,1],bins=512,range=[[low[0],high[0]],[low[1],high[1]]]);H[n]+=hist.astype(np.uint64)
        out[h]={str(n):{'count':count[n],'bounds':[windows[n][0].tolist(),windows[n][1].tolist()]} for n in selected}
        for n in selected:grids[f'{h}_{n}']=H[n];bounds[(h,n)]=windows[n]
        print(h,count,flush=True)
    np.savez_compressed(ROOT/'fullcorpus-crop-histograms.npz',**grids)
    fig,axes=plt.subplots(3,4,figsize=(16,11))
    for row,n in enumerate(selected):
        c=cases[n-1]
        for ax,(h,path) in zip(axes[row],HEADS.items()):
            low,high=bounds[(h,n)];grid=grids[f'{h}_{n}'];z=np.load(ROOT/(h+'-geometry.npz'))['coords'];ax.imshow(np.log1p(grid.T),origin='lower',extent=[low[0],high[0],low[1],high[1]],cmap='magma',interpolation='nearest',aspect='equal')
            ax.scatter(*z[c['survey_index']],marker='+',s=80,c='cyan');ax.set_title(f'Case {n} · {h}\n{int(grid.sum()):,} images in fixed crop',fontsize=9)
    fig.suptitle('All corpus rows in frozen case windows; log counts, 512 × 512 bins\nEach panel uses its own count color scale; cyan = selected query',fontsize=12);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(ROOT/'fullcorpus-crops.png',dpi=140);plt.close(fig)
    (ROOT/'fullcorpus-crops.json').write_text(json.dumps({'status':'descriptive full-corpus coordinate raster','heads':out,'population':103816750,'cases':selected,'source_sha256':sha(Path(__file__)),'wall_seconds':time.time()-started,'caveat':'Fixed windows were selected on 200K survey; no new candidate detector or comparative error rate. Per-panel log-count colors are not a shared density scale.'},indent=2)+'\n');print('DONE',time.time()-started,flush=True)

if __name__=='__main__':main()

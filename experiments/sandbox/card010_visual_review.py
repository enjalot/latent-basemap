"""Self-contained image-neighbor gallery and scientific plots for the fixed card010 panel."""
from pathlib import Path
import json,base64,html
import numpy as np
from score_card010_codex import OC,OUT,PRIMARY,HEADS

def main():
    examples=json.loads((OUT/'examples.json').read_text());pool=Path('/data2/monet/pool-20m');thumb=Path('/data2/monet/pool-20m-thumbs256/shards')
    shard=np.load(pool/'prov_shard_idx.npy',mmap_mode='r');local=np.load(pool/'prov_local_row.npy',mmap_mode='r')
    refs={};missing=[]
    def im(row):
        row=int(row)
        if row not in refs:
            p=thumb/f'{int(shard[row]):04d}';offset=np.memmap(p.with_suffix('.offsets.u64'),dtype='<u8',mode='r');a,b=map(int,offset[int(local[row]):int(local[row])+2])
            with p.with_suffix('.blob').open('rb') as f:f.seek(a);data=f.read(b-a)
            if data[:4]!=b'RIFF':missing.append(row);refs[row]=''
            else:refs[row]='data:image/webp;base64,'+base64.b64encode(data).decode()
        return f'<figure><img loading="lazy" src="{refs[row]}" alt="MONET pool row {row}"><figcaption>{row}</figcaption></figure>' if refs[row] else f'<figure>Missing image {row}</figure>'
    body=['<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Card010 — adaptive neighborhood review</title><style>body{font:16px system-ui;margin:28px auto;max-width:1250px;color:#172331;background:#faf9f5}h1{font-size:30px}section{border-top:1px solid #ccc;padding:22px 0}.row{display:flex;gap:7px;align-items:center;margin:8px 0}.label{width:130px;flex-shrink:0}figure{margin:0;width:130px}img{width:130px;height:106px;object-fit:contain;background:white}figcaption{font-size:10px;color:#657}p{max-width:1000px;line-height:1.5}h3{margin:0}small{color:#526}a{color:#265}</style><h1>Adaptive neighborhoods: fixed image panel</h1><p>Exploratory 300K DINO comparison. Original viability remains FAIL (floor rule). Fixed query sample: one query from each original evaluation cohort. Rows show the first six neighbors from the <b>same original evaluation reference</b>. Encoder neighbors are a representation benchmark, not semantic ground truth. Gallery selection is fixed and independent of model performance. Recall statistics use the full B250/B2000 sets, not just these six thumbnails.</p>']
    present=[h for h in PRIMARY if h in examples[0]['map_neighbors']]
    if (OUT/'result.json').exists():
        result=json.loads((OUT/'result.json').read_text())
        body.append('<p><b>Result: adaptive does not pass the quality screen.</b> Original preregistered viability also remains FAIL. All matched heads trained for 60,000 updates at constant learning rate; the corrected fixed12 control is shown below.</p>')
        body.append('<table><tr><th>Head</th><th>B250 recall</th><th>B2000 recall</th></tr>')
        for h in present:
            v=result['heads'][h]['equal_cohort']
            body.append(f'<tr><td>{h}</td><td>{v["250"]:.6f}</td><td>{v["2000"]:.6f}</td></tr>')
        body.append('</table>')
    if len(present)<3:body.append('<p><b>Corrected fixed12 still pending. This preliminary gallery has two matched heads.</b></p>')
    for ex in examples:
        body.append(f'<section><h3>{html.escape(ex["source"])}</h3><div class="row"><div class="label">Query</div>{im(ex["query_pool_row"])}</div>')
        body.append('<div class="row"><div class="label">Encoder k15<br><small>first six</small></div>'+''.join(im(i) for i in ex['truth_pool_rows'][:6])+'</div>')
        for h in present:body.append(f'<div class="row"><div class="label">{h}<br><small>map nearest six</small></div>'+''.join(im(i) for i in ex['map_neighbors'][h][:6])+'</div>')
        body.append('</section>')
    body.append('</html>');(OUT/'image-review.html').write_text(''.join(body))
    (OUT/'thumbnail-audit.json').write_text(json.dumps({'images':len(refs),'missing':missing,'identity':'pool row -> prov_shard_idx/prov_local_row -> matching packed thumbnail store offsets'},indent=2))
    # Exact matched query sample maps; axes are native fresh-head coordinates (no movement claim).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=np.load(OUT/'per-query.npz');groups=a['query_sources'];coh=np.unique(groups);colors=plt.get_cmap('tab10')
    fig,axes=plt.subplots(1,len(present),figsize=(5*len(present),5),squeeze=False)
    for ax,h in zip(axes[0],present):
        xy=a[h+'_val_xy']
        for i,g in enumerate(coh):
            z=xy[groups==g];ax.scatter(z[:,0],z[:,1],s=2,alpha=.55,color=colors(i),label=g)
        ax.set_title(h);ax.set_aspect('equal');ax.set_xlabel('Fresh-map x');ax.set_ylabel('Fresh-map y')
    fig.legend(*axes[0][0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=7,markerscale=4)
    fig.suptitle('Identical held-out queries; original encoder inputs; native coordinates')
    fig.tight_layout(rect=[0,.12,1,.93]);fig.savefig(OUT/'maps.png',dpi=150);plt.close(fig)
    print(json.dumps({'gallery':str(OUT/'image-review.html'),'maps':str(OUT/'maps.png'),'thumbnails':len(refs),'missing':missing}))

if __name__=='__main__':main()

"""Reproducible qualitative atlas of all twelve exploratory band candidates."""
import json, html, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from band_scan import ROOT, POOL_N, sha

PUBLIC=Path('/home/enjalot/.agent/moonshine/basemap-research-arcs/public')

def main():
    result=json.loads((ROOT/'forensics-result.json').read_text());cases=result['cases']
    ids=np.load(ROOT/'survey-identities.npz')['full_ids']
    geo={h:np.load(ROOT/(h+'-geometry.npz')) for h in ['dino2m','dino6m','dino6m_pca','dino12m_pca']}
    prov={}
    for p in ['pool-20m','pool-complement-88m']:
        d=Path('/data2/monet')/p
        prov[p]=(np.load(d/'prov_shard_idx.npy',mmap_mode='r'),np.load(d/'prov_local_row.npy',mmap_mode='r'))
    fetched={}
    def thumbnail(i):
        i=int(i);out=ROOT/'thumbs'/f'{i}.webp'
        p='pool-20m' if i<POOL_N else 'pool-complement-88m';row=i if i<POOL_N else i-POOL_N
        shard,local=prov[p];si=int(shard[row]);li=int(local[row]);base=Path('/data2/monet/pool-20m-thumbs256/shards')/f'{si:04d}'
        offset=np.memmap(base.with_suffix('.offsets.u64'),dtype='<u8',mode='r');a,b=map(int,offset[li:li+2])
        with base.with_suffix('.blob').open('rb') as f:f.seek(a);data=f.read(b-a)
        assert data[:4]==b'RIFF' and data[8:12]==b'WEBP',i
        out.write_bytes(data);fetched[i]={'shard':si,'local':li,'sha256':sha(out)}
        return out
    def im(i):
        thumbnail(i)
        return f'<figure><img loading="lazy" src="band-assets/thumbs/{i}.webp" alt="Corpus row {i}"><figcaption>{i}</figcaption></figure>'
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Investigating bands in large DINO maps</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 16px;background:#faf9f5;color:#172331;line-height:1.5}section{border-top:1px solid #aaa;margin:28px 0;padding-top:20px}.row{display:flex;gap:6px;overflow:auto}.label{min-width:135px;width:135px;font-size:13px}figure{margin:0;width:120px;flex-shrink:0}figure img{width:120px;height:100px;object-fit:contain;background:white}figcaption{font:11px monospace}table{border-collapse:collapse}td,th{padding:5px 13px;text-align:right;border-bottom:1px solid #ddd}.plot{width:100%}p{max-width:1000px}.note{background:#e9eee8;padding:15px}a{color:#245b83}</style><h1>What are the bands between clusters?</h1><p>Exploratory audit, 11 September 2026. Identical 200,000 rows sampled uniformly from the 103,816,750-image corpus. Four saved 2D DINO heads and two 3D twins. Images below come from their original corpus provenance. No GPU training was used for this audit.</p><p class="note"><b>A line is not automatically a mapping error.</b> The detector selects elongated neighborhoods with denser points on both sides. These can be compact, coherent islands as well as candidate bridges. Endpoint groups are geometric probes, not certified semantic clusters. Encoder neighbors are a representation benchmark, not human ground truth.</p><p>All twelve selected cases are retained: three high-contrast, spatially separated candidates per head. This is a discovery panel, not an unbiased estimate of bridge frequency. Head size, training history, preprocessing and dimension can all affect the comparisons. Within each plot, colors refer to the exact same image IDs; axes are native to each separately trained head.</p><img class="plot" src="band-assets/survey-overview.png" alt="Four full-map sample views"><p>Recall below: how many of each center image’s 15 closest full-dimensional DINO neighbors occur in its nearest B map points, using the same 200K reference and excluding the image itself. B250 is 0.125% of this reference; it is not the existing 250K-reference evaluation instrument.</p>']
    for n,c in enumerate(cases,1):
        parts=c['full_ids_by_part'];groups=[('Query',[c['full_id']]),('Center neighbors',parts['center'][1:7]),('Endpoint A',parts['endpoint_A'][:6]),('Endpoint B',parts['endpoint_B'][:6]),('Encoder nearest',c['encoder_neighbors_of_query'][:6])]
        body.append(f'<section id="case-{n}"><h2>{n}. {html.escape(c["head"])} · row {c["full_id"]}</h2><p>Source: {html.escape(c["source"])}. Selecting-head ridge contrast: {c["contrast"]:.2f}. Encoder between/within endpoint cosine-distance ratio: {c["encoder_core_between_over_within_cosine_distance"]:.2f} (exploratory; larger means more separated relative to within-group spread).</p>')
        # Full provenance contact sheet, matching the rendered gallery.
        sheet=Image.new('RGB',(7*150,5*140),'#faf9f5');draw=ImageDraw.Draw(sheet)
        for row,(label,rr) in enumerate(groups):
            body.append('<div class="row"><div class="label">'+label+'</div>'+''.join(im(i) for i in rr)+'</div>')
            draw.text((4,row*140+4),label,fill='black')
            for col,i in enumerate(rr):
                pic=Image.open(thumbnail(i)).convert('RGB');pic.thumbnail((140,110));sheet.paste(pic,((col+1)*150,row*140+5));draw.text(((col+1)*150,row*140+118),str(i),fill='black')
        sheet.save(ROOT/f'case-{n:02d}-images.jpg',quality=90)
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        band=np.array(c['center_indices']);A,B=map(np.array,c['endpoint_indices']);sel=np.r_[band,A,B]
        for ax,(h,g) in zip(axes,geo.items()):
            z=g['coords'];low=z[sel].min(0);high=z[sel].max(0);pad=np.maximum((high-low)*.25,.05)
            mask=((z>=low-pad)&(z<=high+pad)).all(1);ax.scatter(*z[mask].T,s=2,c='#bbbbbb',alpha=.4,rasterized=True)
            for idx,color,label in [(A,'#3e7bb6','A'),(B,'#df9239','B'),(band,'#a84277','center')]:ax.scatter(*z[idx].T,s=15,c=color,label=label)
            ax.scatter(*z[c['survey_index']],s=65,c='black',marker='x');ax.set_title(h);ax.set_aspect('equal');ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
        axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/f'case-{n:02d}-maps.png',dpi=130);plt.close(fig)
        body.append(f'<img class="plot" src="band-assets/case-{n:02d}-maps.png" alt="Identical selected images across four maps"><table><tr><th>Head</th><th>B15</th><th>B50</th><th>B250</th><th>B2000</th></tr>')
        for h,v in c['comparisons'].items():
            vals=v.get('budget_recall',{'250':v['B250_k15']})
            body.append('<tr><td>'+h+'</td>'+''.join('<td>'+ (f'{100*vals[str(b)]:.1f}%' if str(b) in vals else '—')+'</td>' for b in [15,50,250,2000])+'</tr>')
        body.append('</table></section>')
    if (ROOT/'fullcorpus-crops.png').exists():
        body.append('<section><h2>Check against the complete corpus</h2><p>These fixed windows contain every corpus point, streamed from all 103,816,750 saved coordinates. The gym corridor persists in all four heads (roughly 165–172K images per crop). Some endpoint groups spread widely in other heads, so their matched windows span much of the map. Per-panel colors use separate log-count scales; this is a geometry check, not a shared density comparison.</p><a href="band-assets/fullcorpus-crops.png"><img class="plot" src="band-assets/fullcorpus-crops.png" alt="All-corpus density crops around three fixed cases"></a></section>')
    if (ROOT/'snapshot-movement.png').exists():
        body.append('<section><h2>Actual saved fine-tuning checkpoints</h2><p>The case images were also projected through card006 IN/OUT replay snapshots. These are a different model family from the ladder. Case 4 moves substantially and non-monotonically: IN displacement from original T0 is .0718, .0088, .2230 at 35K, 70K, 140K updates; OUT is .0482, .1485, .0058. Values are fractions of the fixed original T0 radius. This is measured model movement; lines between checkpoints are only visual guides, and do not prove a semantic cluster transition.</p><img class="plot" src="band-assets/snapshot-movement.png" alt="Measured displacement at three saved checkpoints"><p>CPU re-inference matches stored map coordinates to less than .001 local k32 radius on the tested panel. Small input probes reveal sensitive locations, but perturbed feature vectors need not correspond to real images. The detector and selected examples do not establish that a larger head removes bands globally.</p></section>')
    support_section = ROOT/'support-v2/support-section.html'
    if support_section.exists():
        body.append(support_section.read_text())
    body.append('<p>Literature: <a href="https://www.nature.com/articles/s41467-025-60434-9">Map-continuity reliability</a>; <a href="https://arxiv.org/abs/2107.07859">Steadiness and Cohesiveness</a>; <a href="https://arxiv.org/abs/1909.12902">MING neighborhood overlays</a>. This page uses a custom geometric screen, not those papers’ validated scores.</p></html>')
    (ROOT/'band-review.html').write_text(''.join(body))
    (ROOT/'thumbnail-audit.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)),'count':len(fetched),'rows':fetched},indent=2)+'\n')
    target=PUBLIC/'band-assets';target.mkdir(exist_ok=True);shutil.copytree(ROOT/'thumbs',target/'thumbs',dirs_exist_ok=True)
    for p in [ROOT/'survey-overview.png',*ROOT.glob('case-*-maps.png'),*ROOT.glob('snapshot-movement.png'),*ROOT.glob('fullcorpus-crops.png')]:shutil.copy2(p,target/p.name)
    shutil.copy2(ROOT/'band-review.html',PUBLIC/'band-review.html')
    # Separate document for the SPA's gallery route; resolve assets relative
    # to this folder, including when Vite is published under a base subpath.
    (target/'review.html').write_text(''.join(body).replace('band-assets/', './'))
    print('Gallery complete',len(cases),'cases',len(fetched),'images',flush=True)

if __name__=='__main__':main()

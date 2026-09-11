import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,html,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/enjalot/code/latent-basemap/experiments/sandbox')
from band_scan import ROOT,POOL_N,sha
P=ROOT/'support-v2';PUBLIC=Path('/home/enjalot/.agent/moonshine/basemap-research-arcs/public');assets=PUBLIC/'band-assets'
r=json.loads((P/'results.json').read_text());e=json.loads((P/'extraction.json').read_text());pq=np.load(P/'per-query.npz');co=np.load(P/'cohorts.npz')
heads=list(r['references']['200000']['heads']);labels=['2M full 2D','6M full 2D','6M PCA 2D','12M PCA 2D','6M PCA 3D','12M PCA 3D'];ns=[3,4,10]
fig,axs=plt.subplots(3,2,figsize=(13,12));x=np.arange(6)
for row,n in enumerate(ns):
 for col,metric in enumerate(['B250','encoder_distance_ratio_p90']):
  ax=axs[row,col];v=[r['references']['200000']['heads'][h][str(n)][metric] for h in heads]
  ax.bar(x-.18,[a['band_mean'] for a in v],.36,label='Corridor images',color='#ad4272');ax.bar(x+.18,[a['control_mean'] for a in v],.36,label='Matched ordinary images',color='#527f91')
  ax.set_xticks(x,labels,rotation=30,ha='right');ax.set_title(f'Case {n}: '+('encoder k15 recall within map B250' if col==0 else 'mean query p90 encoder distance / d60'))
  if col==0:ax.set_ylim(0,1)
  else:ax.axhline(1,ls=':',c='#777')
  ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Actual full-corpus corridor points vs matched ordinary controls\n48 pairs per selected case; same 200K reference and image IDs across heads',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(P/'support-comparison.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,n in zip(axs,ns):
 t=co[f'case{n}_axis_position'];margin=pq[f'case{n}_margin_200000'];a=pq[f'case{n}_support_A_200000'];b=pq[f'case{n}_support_B_200000'];mixed=(a>0)&(b>0)
 ax.scatter(t,margin,c=np.where(mixed,'#ad4272','#527f91'),s=25);ax.axhline(0,c='#777',lw=.6);ax.set_title(f'Case {n}');ax.set_xlabel('Position along corridor / original r32');ax.set_ylabel('Encoder affinity B − A')
fig.suptitle('Affinity is not ambiguity: pink means encoder-k15 includes neighbors from both seed supports');fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(P/'support-affinity.png',dpi=150);plt.close(fig)
body=['<section id="support-v2"><h2>Follow-up: real corridor points and matched controls</h2><p>We extracted every full-corpus point in three fixed thin corridors, then sampled 48 new images per corridor outside the original survey. All IDs are held fixed across four 2D heads and two 3D twins. The results below are exploratory case studies, not corpus-wide rates.</p><table><tr><th>Case</th><th>Actual corridor points</th><th>Selecting head</th><th>Corridor B250</th><th>Control B250</th></tr>']
md=['# Actual corridor points: encoder support and matched controls','', '**Machine:** GSV  ', '**Assets:** `gsv:/data/latent-basemap/sandbox/overseer-codex/band-investigation-20260911/support-v2/`; [gallery](http://gsv.local:5194/band-review.html#support-v2).','', 'Three purposive regions were fixed before this follow-up: cases3,4,10. A thin rectangle extends ±3 original survey r32 along the principal axis and ±.5 r32 perpendicular. Every corridor ID is retained;48 rows/case are sampled uniformly after excluding the original survey. These counts are window occupancies, not numbers of bands or comparable prevalence estimates.','', 'Each sampled row is paired to a unique ordinary survey row (linearity32<.8, outside8 original radii), matched exactly on provenance source and selecting-head training membership, then nearest log map-r32. Matching does not control image semantics or encoder density. Matching is only for the selecting head; other heads can retain membership imbalance. Query self IDs are excluded from reference neighbors.','', '| Case | Selecting head | Corridor rows | B250 corridor | B250 controls | Difference |','| --- | --- | ---: | ---: | ---: | ---: |']
for a in e['cases']:
 n=a['case'];h=a['selecting_head'];v=r['references']['200000']['heads'][h][str(n)]['B250']
 body.append(f'<tr><td>{n}</td><td>{a["corridor_count"]:,}</td><td>{h}</td><td>{v["band_mean"]:.3f}</td><td>{v["control_mean"]:.3f}</td></tr>')
 md.append(f'| {n} | {h} | {a["corridor_count"]:,} | {v["band_mean"]:.6f} | {v["control_mean"]:.6f} | {v["paired_delta_mean"]:+.6f} |')
body.append('</table><p>Controls match source, selecting-head training membership, and local map density. They do not match image semantics or encoder density. Other-head comparisons may retain training-membership differences. Pink bars show corridor images; blue bars show controls. Values are descriptive means from these regions.</p><img class="plot" src="band-assets/support-comparison.png" alt="Corridor and matched-control recall and encoder-distance distortion across six maps">')
body.append('<p class="note"><b>What this adds:</b> the gym corridor has strong encoder support and almost complete B250 recall. In the mixed and clothing cases, almost all encoder-k60 neighbors belong to neither endpoint support; their placement between those probes does not establish semantic indecision. The clothing cohort has a small 2D recall deficit versus controls and improves in the saved 3D twins. These are different sampled cohorts from the original center examples.</p>')
body.append('<h3>Do the apparent endpoints have encoder support?</h3><p>For each original map-probe group, select its encoder medoid and expand to 256 encoder neighbors. These are endpoint-seeded neighborhoods, not independently discovered semantic clusters. The table partitions corridor encoder-k60 neighbors into A-only, B-only, shared, or neither.</p><table><tr><th>Case</th><th>A only</th><th>B only</th><th>Shared</th><th>Neither</th><th>Seed-support overlap</th></tr>')
md+=['','## Endpoint-seeded encoder support','','Endpoint seeds are encoder medoids of the original map-probe groups; top256 reference neighbors define their support. Seeds remain map-selected in origin. Shared support means overlap of these neighborhoods, not a semantic probability. Similar affinity to A/B when neither is close is not evidence of ambiguity.','', '| Case | A-only k60 mass | B-only | Shared | Neither | Endpoint Jaccard |','| --- | ---: | ---: | ---: | ---: | ---: |']
for n in ns:
 v=r['references']['200000']['cases'][str(n)];q=v['HD60_support_partition'];vals=[q[k] for k in ['A_only','B_only','both','neither']]+[v['endpoint_support_jaccard']]
 body.append('<tr><td>'+str(n)+'</td>'+''.join(f'<td>{z:.1%}</td>' for z in vals)+'</tr>')
 md.append('| '+str(n)+' | '+' | '.join(f'{z:.6f}' for z in vals)+' |')
body.append('</table><img class="plot" src="band-assets/support-affinity.png" alt="Real corridor positions versus encoder affinity, with direct support indicated"><p>Magenta points have encoder-k15 neighbors in both endpoint supports. Near-zero affinity can also mean that neither endpoint describes the image. The arbitrary sign of the principal axis can reverse the plotted trend.</p>')
# Show four axial representatives and their matched controls, selected without looking at quality.
prov={}
for folder in ['pool-20m','pool-complement-88m']:
 root=Path('/data2/monet')/folder;prov[folder]=(np.load(root/'prov_shard_idx.npy',mmap_mode='r'),np.load(root/'prov_local_row.npy',mmap_mode='r'))
thumb=assets/'support-thumbs';thumb.mkdir(exist_ok=True);audit={}
def img(gid):
 gid=int(gid);folder='pool-20m' if gid<POOL_N else 'pool-complement-88m';row=gid if gid<POOL_N else gid-POOL_N
 si,li=(int(a[row]) for a in prov[folder]);base=Path('/data2/monet/pool-20m-thumbs256/shards')/f'{si:04d}';off=np.memmap(base.with_suffix('.offsets.u64'),dtype='<u8',mode='r');lo,hi=map(int,off[li:li+2])
 with base.with_suffix('.blob').open('rb') as f:f.seek(lo);data=f.read(hi-lo)
 assert data[:4]==b'RIFF' and data[8:12]==b'WEBP'
 path=thumb/f'{gid}.webp';path.write_bytes(data);audit[str(gid)]={'shard':si,'local':li,'sha256':sha(path)}
 return f'<figure><img loading="lazy" src="band-assets/support-thumbs/{gid}.webp" alt="Corpus row {gid}"><figcaption>{gid}</figcaption></figure>'
for n in ns:
 t=co[f'case{n}_axis_position'];order=np.argsort(t);sel=order[[5,17,29,41]]
 body.append(f'<h3>Case {n}: four positions along the corridor</h3><p>Four fixed axial quantiles, chosen without quality scores; controls are their exact paired comparison rows.</p>')
 for part in ['band','control']:body.append('<div class="row"><div class="label">'+part+'</div>'+''.join(img(i) for i in co[f'case{n}_{part}'][sel])+'</div>')
body.append('<p>Reference-size sensitivity: the 50K reference is a uniform nested subset of the 200K survey. Fixed B250 changes reference coverage (.5% versus .125%); B63 at 50K approximates B250 coverage at 200K. Endpoint supports are reconstructed at each size, so support differences also reflect changed neighborhoods. Directed distance ratios use each query’s encoder d60; they do not claim symmetric false-join detection.</p></section>')
interpretation = '''The follow-up separates three cases. The gym corridor preserves nearly all encoder-k15 neighbors within B250, and almost all of its encoder-k60 neighborhood mass lies in the union of the two endpoint supports. This supports a coherent connection around those seeds. Fine ordering remains imperfect: at the200K reference, gym B15 is only about.30–.36 across these heads despite B250 near1.

In the mixed region and clothing corridor, 98.85% and 99.97% of encoder-k60 neighbor mass respectively belongs to neither endpoint support. Their apparent position between the probes therefore does not establish semantic indecision between those probes. The spatial endpoints may simply be poor descriptions of the intervening content.

The clothing cohort's 12M PCA B250 is .8792 versus .9056 for controls, a descriptive deficit of2.64 percentage points. Its saved3D twin reaches .9181, up3.89 points; the6M twin also improves. This is an operational comparison, not an isolated dimensionality effect. The mixed-region cohort has B250 .8389 on its selecting6M head, above controls .7764. The earlier handpicked center examples were different rows, so their lower recall cannot be generalized to the whole extracted corridor.

Four axial representatives per case were inspected without selecting by quality. Gym samples show gym interiors/equipment; clothing samples show outerwear modeled or photographed against simple backgrounds. The mixed corridor includes horses, a carriage/street scene and a decorated toy. This small panel illustrates heterogeneity but does not certify semantics for every corridor image. Controls include unrelated visual content, as expected from source/density rather than semantic matching.

Local edge thresholds are reference-sensitive. For the gym case on its selecting2M head, the directed>1.25-distance fraction changes from0/720 map edges at50K to76/720 at200K, while B250 stays near1. The relative radius shrinks as reference density increases. That threshold must not be treated as a universal artifact detector; support, fine ordering, distance distortion and dimensional comparisons need to be read together.'''
md[4:4] = ['', '## Findings', '', interpretation, '']
body[-1] = body[-1].replace('</section>', '<p>The gym’s directed distance-threshold count changes from 0 of 720 map edges at 50K to 76 of 720 at 200K, despite near-complete B250 recall. This threshold is reference-sensitive and is not a universal artifact label.</p></section>')
md+=['','## Measurement limits','','- All six heads use the same query IDs and reference IDs. Recall uses exact normalized full1536-D DINO k15; map budgets are15/63/250.','- The directed edge diagnostic examines map-k15 edges, their encoder-k60 membership, and Euclidean distance divided by the query encoder d60. No symmetric-neighbor claim is made. Continuous ratios accompany the1.25 threshold.','- Reference50K is a seeded uniform subset of200K. B250 is reference-specific; compare B63/50K with B250/200K for similar reference fractions. Endpoint supports are rebuilt per reference, so this also changes their definition.','- Controls condition on map density and shape. Differences are descriptive associations; none identifies a semantic cause or a head-size causal effect. There are only three purposive regions, no population confidence interval or estimate of band prevalence.','- Raw IDs, per-query neighborhoods/metrics, source/membership matching and exact protocol remain in the asset directory. CPU resource ceiling8GiB,2cores; no GPU training.']
(P/'support-section.html').write_text(''.join(body));(P/'report.md').write_text('\n'.join(md)+'\n');(P/'thumbnail-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
for fn in ['support-comparison.png','support-affinity.png']:shutil.copy2(P/fn,assets/fn)
print('PUBLISHED section and figures',len(audit),'thumbs')

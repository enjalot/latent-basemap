"""Reuse identical CLIP support/graph/init; square-root the existing half-radius arrays."""
from pathlib import Path
import os,json,time
import numpy as np
import card055_common as C

def main():
 start=time.monotonic();old=Path('/data/latent-basemap/substrates/card049-clip-scale');C.D.mkdir(exist_ok=True);m=C.read(old/'inputs-manifest.json');g=C.read(old/'graph-manifest.json');assert m['PASS'] and g['PASS'];assert g['input_manifest_sha']==C.sha(old/'inputs-manifest.json')
 for n,h in {**m['files'],**g['files']}.items():
  assert C.sha(old/n)==h,n
  if not (C.D/n).exists():os.link(old/n,C.D/n)
 for target,source in [('r-quarter.npy','r-half.npy'),('r-shuffled-quarter.npy','r-shuffled-half.npy')]:np.save(C.D/target,np.sqrt(np.load(old/source)))
 m.update(card='055',parent_input_manifest_sha=C.sha(old/'inputs-manifest.json'),parent_graph_manifest_sha=C.sha(old/'graph-manifest.json'),radius_rule='sqrt(original049 half-radius); samewithin-source permutation, not new sampling',cpu_build_s=time.monotonic()-start)
 for n in ['r-quarter.npy','r-shuffled-quarter.npy']:m['files'][n]=C.sha(C.D/n)
 C.write(C.D/'inputs-manifest.json',m);C.write(C.O/'card055-parent-graph.json',g)
 gg=dict(g);gg.update(input_manifest_sha=C.sha(C.D/'inputs-manifest.json'),parent_graph_manifest_sha=C.sha(C.O/'card055-parent-graph.json'),parent_runtime_sha=g['runtime_sha'],scope='Byte-identical original049 graph, no new graph training or GPU build. Newquarter radius arrays in input manifest.');C.write(C.D/'graph-manifest.json',gg)
 print('CPU reuse build',time.monotonic()-start,flush=True)
if __name__=='__main__':main()

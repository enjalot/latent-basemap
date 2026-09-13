"""CPU-only graph reuse identity/corruption controls, no feature scan."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,json,hashlib,tempfile,copy
from pathlib import Path
import card088_graph as G
functions={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse(Path(G.__file__).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='validate_bundle'}
checks={};oldroot=G.R
for case in ['valid','runtime','producer','functions','files','input_corruption','output_corruption','builder_corruption']:
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);G.R=root/'code';(G.R/'experiments/sandbox').mkdir(parents=True);O=root/'overseer-codex';O.mkdir();D=root/'data';D.mkdir();(G.R/'card088-runtime-sha.json').write_text('{}');builder=G.R/'experiments/sandbox/build_card088_graph.py';builder.write_text('builder');f=D/'array';f.write_text('output');inp=root/'input';inp.write_text('input')
  m={'PASS':True,'n':G.N,'k_extra':45,'runtime_sha':'producer','graph_builder_sha':G.sha(builder),'files':{'array':G.sha(f)},'inputs':{str(inp):G.sha(inp)}};p=D/'manifest.json';p.write_text(json.dumps(m));proof={'PASS':True,'producer_runtime_sha':'producer','consumer_runtime_sha':G.sha(G.R/'card088-runtime-sha.json'),'producer_manifest_sha':G.sha(p),'graph_builder_sha':m['graph_builder_sha'],'unchanged_graph_functions':functions,'files':m['files'],'inputs':m['inputs']};proof=copy.deepcopy(proof)
  if case=='runtime':proof['consumer_runtime_sha']='wrong'
  if case=='producer':proof['producer_manifest_sha']='wrong'
  if case=='functions':proof['unchanged_graph_functions'].clear()
  if case=='files':proof['files']={}
  if case=='input_corruption':inp.write_text('changed')
  if case=='output_corruption':f.write_text('changed')
  if case=='builder_corruption':builder.write_text('changed')
  (O/'card088-graph-reuse.json').write_text(json.dumps(proof));failed=False
  try:G.validate_bundle(D)
  except AssertionError:failed=True
  checks[case]=failed==(case!='valid')
G.R=oldroot;assert all(checks.values()),checks
O=G.R.parent/'overseer-codex';(O/'card088-graph-reuse-cpu.json').write_text(json.dumps({'PASS':True,'checks':checks,'scope':'Mocked schema-correct graph bundle, no search or GPU. Reject mismatched runtime/producer/function/data/input/output/builder.'},indent=2)+'\n');print('PASS',len(checks))

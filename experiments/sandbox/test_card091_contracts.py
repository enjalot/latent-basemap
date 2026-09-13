"""CPU-only contracts: no production model, raw scan, ledger or GPU access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,datetime as dt,json,tempfile,unittest,hashlib,time
from pathlib import Path
from unittest.mock import patch
import card091_common as C
import card091_budget as B
import run_card091_chain as H
from card035_stream_layout import expected_ranges,validate_raw_history
CHECKS=[]
def check(name,fn):fn();CHECKS.append(name)
def rejects(fn,text):
 try:fn()
 except (AssertionError,ValueError,FileNotFoundError) as e:
  assert text in str(e),(text,str(e));return
 raise AssertionError('accepted invalid fixture: '+text)
def fixtures():
 arms={}
 for arm in C.ARMS:
  a={'card':'090','arm':arm,'seed':int(arm[-2:]),'dose':60000,'lr':.0001,'parent_sha':C.PARENT,'original_init_sha':C.PARENT,'runtime_manifest_sha':C.RUNTIME090,'rankneg_window':500000 if arm.startswith('ranked') else 0}
  v={'PASS':True,'arm':arm,'runtime_sha':C.RUNTIME090,'model_sha':'a'*64,'identity':a}
  m={'identity':a,'model_sha':'a'*64,'dose':60000,'train_stats':{'positive_lr_optimizer_steps':60000}}
  p={'READY':True,'runtime_sha':C.RUNTIME090,'parent_sha':C.PARENT};arms[arm]=(v,m,a,p)
 return {'status':'TRAINED_VALIDATED','runtime_sha':C.RUNTIME090,'quality':'FAIL'},arms

def main():
 ex,arms=fixtures();check('all_four_valid_quality_FAIL_is_eligible',lambda:C.validate_completed(ex,arms))
 for field,value,msg in [('status','PARTIAL','incomplete'),('runtime_sha','bad','wrong runtime')]:
  z=copy.deepcopy(ex);z[field]=value;check('reject_'+field,lambda z=z,msg=msg:rejects(lambda:C.validate_completed(z,arms),msg))
 for missing in C.ARMS:
  z=copy.deepcopy(arms);del z[missing];check('missing_'+missing,lambda z=z:rejects(lambda:C.validate_completed(ex,z),'missing or extra'))
 for k,v,msg in [('seed',42,'dose/seed'),('dose',59999,'dose/seed'),('parent_sha','bad','parent'),('rankneg_window',0,'policy'),('runtime_manifest_sha','bad','identity runtime')]:
  z=copy.deepcopy(arms);z['ranked43'][2][k]=v;check('wrong_'+k,lambda z=z,msg=msg:rejects(lambda:C.validate_completed(ex,z),msg))
 z=copy.deepcopy(arms);z['uniform44'][0]['PASS']=False;check('invalid_fourth_validation',lambda:rejects(lambda:C.validate_completed(ex,z),'canonical'))
 hist=[{'lo':a,'hi':b,'raw_sha':'a'*64} for a,b in expected_ranges()]
 check('full_split_history',lambda:validate_raw_history(hist));assert any(x['hi']==C.LOW for x in hist) and any(x['lo']==C.LOW for x in hist)
 for name,bad in [('missing',hist[:-1]),('duplicate',hist+[hist[0]]),('reordered',[hist[1],hist[0]]+hist[2:])]:
  check('stream_'+name,lambda bad=bad:rejects(lambda:validate_raw_history(bad),'raw history'))
 z=copy.deepcopy(hist);z[0]['raw_sha']='xyz';check('malformed_raw_sha',lambda:rejects(lambda:validate_raw_history(z),'malformed'))
 now=C.END-6000;state={'active':False};check('unstarted090_reserves1800',lambda:assert_equal(C.cpu090_bound(state,now),1800))
 rel={'PASS':True,'runtime_sha':C.RUNTIME090,'cpu_budget_started_at':dt.datetime.fromtimestamp(now,dt.timezone.utc).isoformat(),'cpu_absolute_deadline':dt.datetime.fromtimestamp(now+1800,dt.timezone.utc).isoformat()}
 state={'active':True,'memory_max_bytes':16*2**30,'memory_swap_max_bytes':0,'release':rel,'launch':{'status':'CPU_STARTED'},'launch_release_match':True}
 check('active090_remaining_bound',lambda:assert_equal(C.cpu090_bound(state,now+100),1700))
 for name,changes,msg in [('unbounded_cpu_memory',{'memory_max_bytes':32*2**30},'memory/swap'),('swap_enabled',{'memory_swap_max_bytes':1},'memory/swap'),('wrong_binding',{'launch_release_match':False},'release mismatch'),('inactive_stuck',{'active':False},'inactive without terminal'),('unbound_active',{'release':None},'unbound'),('stale_terminal',{'active':False,'terminal':{'status':'SCORED_AUDITED'},'terminal_fresh':False},'stale terminal')]:
  z={**state,**changes};check(name,lambda z=z,msg=msg:rejects(lambda:C.cpu090_bound(z,now),msg))
 z={**state,'active':False,'terminal':{'status':'CPU_FAILED'},'terminal_fresh':True};check('terminal_failed_cpu_can_end_reserve',lambda:assert_equal(C.cpu090_bound(z,now),0))
 check('expired090_stop',lambda:rejects(lambda:C.cpu090_bound(state,now+1801),'expired'))
 def admission():
  q=C.joint_checks(2500,400,10000,1800,C.END-4000,16*2**30);assert all(q.values())
  assert not C.joint_checks(2500,1200,10000,1800,C.END-4000,16*2**30)['card_cap']
  assert not C.joint_checks(500,400,10000,1800,C.END-2999,16*2**30)['joint_deadline']
  assert not C.joint_checks(2500,400,10000,1800,C.END-4000,16*2**30-1)['disk']
  assert not C.joint_checks(2500,400,164000,1800,C.END-4000,16*2**30)['window_cap']
 check('joint_max_deadline_card_window_disk',admission)
 check('active_reservation_actual_not_reserved',lambda:assert_equal(C.settled_equivalent(3500,3000,200),700))
 check('missing_active_reservation',lambda:rejects(lambda:C.settled_equivalent(100,3000,200),'reservation absent'))
 check('nonfinite_estimate',lambda:rejects(lambda:C.joint_checks(float('nan'),0,0,0,now,16*2**30),'invalid'))
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp)
  with patch.object(C,'O',d):check('no_root_release_rejected_before_device',lambda:rejects(C.require_release,'NO ROOT GPU RELEASE'))
  p=d/'receipt.json';t=time.time_ns();r={'PASS':True,'runtime_sha':'r','selection_sha':'s','release_sha':'l'};C.write(p,r);t=p.stat().st_mtime_ns
  check('fresh_receipt',lambda:C.receipt(p,t,'r','s','l'))
  check('missing_receipt',lambda:rejects(lambda:C.receipt(d/'absent',t,'r','s','l'),'missing'))
  check('stale_receipt',lambda:rejects(lambda:C.receipt(p,p.stat().st_mtime_ns+1,'r','s','l'),'stale'))
  for key,val,msg in [('PASS',False,'lacks PASS'),('runtime_sha','bad','runtime'),('selection_sha','bad','selection'),('release_sha','bad','release')]:
   C.write(p,{**r,key:val});check('receipt_'+key,lambda msg=msg:rejects(lambda:C.receipt(p,t,'r','s','l'),msg))
  with patch.object(C,'O',d),patch.object(B,'L',d/'ledger'),patch.object(B,'W',d/'window'),patch.object(B,'J',d/'journal'):
   C.write(B.W,{'spent_s':100.,'entries':[]});B.transact(3000,'reservation',True);B.transact(200-3000,'settlement')
   assert abs(C.read(B.L)['spent_s']-200)<1e-9 and abs(C.read(B.W)['spent_s']-300)<1e-9
   for p in B.J.glob('*.json'):B.apply(C.read(p))
   assert C.read(B.L)['spent_s']==200;CHECKS.append('journal_idempotence_and_active_reservation_settlement')
   rejects(lambda:B.transact(3500,'reservation',True),'cumulative');CHECKS.append('journal_cap_preserves_prior_cost')
   # Actual controller main: successful fake child, child failure, and initial verify failure.
   class Child:
    pid=999999
    def __init__(self,rc):self.returncode=rc
    def poll(self):return self.returncode
   C.write(d/'card091-release.json',{'PASS':True});runtime='r';release={'runtime_sha':runtime,'selection_sha':'s'}
   for fail in ['none','child','verify','lease','stale']:
    before=C.read(B.L)['spent_s'];real=C.require_release
    def vr():
     if fail=='verify':raise AssertionError('injected verify failure')
     return release
    def rr(*args):
     if fail=='stale':raise AssertionError('stale stage receipt')
     return {'PASS':True}
    with patch.object(C,'require_release',vr),patch.object(H,'verify_external_leases',side_effect=AssertionError('lease') if fail=='lease' else None),patch.object(H,'cpu_state',return_value={'active':False}),patch.object(H.subprocess,'Popen',return_value=Child(2 if fail=='child' else 0)),patch.object(C,'receipt',side_effect=rr),patch.object(H.shutil,'disk_usage',return_value=type('Disk',(),{'free':20*2**30})()),patch.object(C,'D',d):
     C.write(d/'execution.json',{'status':'PROJECTED_NOT_SCORED','rows':C.HIGH,'models':dict.fromkeys(C.ARMS)})
     rc=H.main();assert rc==(0 if fail=='none' else 1);assert C.read(B.L)['spent_s']>before
     assert abs(C.read(B.W)['spent_s']-C.read(B.L)['spent_s']-100)<1e-6
     assert all(abs(e['wall_s'])<3601 for e in C.read(B.L)['entries'])
    CHECKS.append('actual_chain_'+fail+'_charged_settled')
 # Actual selection builder on tiny completed payload files; no canonical model loading.
 import prepare_card091_selection as P
 for fault in ['none','missing_snapshot','missing_epoch','wrong_model_sha']:
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);o=d/'overseer';td=d/'train';r090=d/'original';out=o/'full';r090.mkdir();o.mkdir()
   ex,aa=fixtures();C.write(o/'card090-execution.json',ex);C.write(o/'card090-release.json',{'PASS':True})
   (r090/'card090-runtime-sha.json').write_text('{}')
   deep={}
   for a in C.ARMS:
    base=td/a;(base/'ckpts').mkdir(parents=True)
    for n in ['model.pt','prepared.pt']+[f'model-step{x}.pt' for x in [20000,40000,60000]]+[f'ckpts/ckpt-step{x}.pt' for x in [20000,40000,60000]]+['ckpts/ckpt-epoch2.pt']:(base/n).write_bytes(a.encode())
    v,m,adm,prep=aa[a];v['model_sha']=m['model_sha']=C.sha(base/'model.pt')
    for n,x in [('validation.json',v),('manifest.json',m),('admission.json',adm),('preparation.json',prep)]:C.write(base/n,x)
    deep[a]=v
   if fault=='missing_snapshot':(td/'uniform44/model-step40000.pt').unlink()
   if fault=='missing_epoch':(td/'uniform44/ckpts/ckpt-epoch2.pt').unlink()
   if fault=='wrong_model_sha':(td/'uniform44/model.pt').write_bytes(b'wrong')
   old={'reused_models':{},'reused_outputs':{},'head_batch':256};C.write(o/'card078-full/selection.json',old)
   for n in ['card091-full-reference-seeds.md','card091-quality-prereg.md']:(o/n).write_text('frozen')
   original_sha=C.sha
   def sh(p):return C.RUNTIME090 if Path(p)==r090/'card090-runtime-sha.json' else original_sha(p)
   with patch.object(C,'O',o),patch.object(C,'TD',td),patch.object(C,'R090',r090),patch.object(C,'D',out),patch.object(C,'source_check',return_value='runtime091'),patch.object(C,'sha',side_effect=sh),patch.object(C,'validate_selection'),patch.object(C,'verify_sources'),patch.object(P.subprocess,'run',return_value=type('Result',(),{'stdout':json.dumps(deep)})()):
    if fault=='none':
     result=P.build();assert list(result['models'])==C.ARMS and not any(k.startswith('reused_') for k in result)
     assert len(result['retained_checkpoint_files']['uniform44'])==4
     for a in C.ARMS:
      assert str(td/a/'model-step40000.pt') in result['endpoint_bindings']
      assert str(td/a/'ckpts/ckpt-epoch2.pt') in result['endpoint_bindings']
     rejects(P.build,'existing selection')
    else:rejects(P.build,{'missing_snapshot':'model-step40000','missing_epoch':'missing retained epoch','wrong_model_sha':'model changed'}[fault]);assert not (out/'selection.json').exists()
   CHECKS.append('actual_selection_builder_'+fault)
 import numpy as np
 ids=np.arange(4);zeros=np.zeros((4,3),'f4')
 check('constant_loader_outputs_eligible',lambda:C.validate_loader_output(ids,zeros,'state',ids,zeros,'state'))
 check('constant_outputs_wrong_ids_reject',lambda:rejects(lambda:C.validate_loader_output(ids[::-1],zeros,'state',ids,zeros,'state'),'query order'))
 check('loader_wrong_precision_reject',lambda:rejects(lambda:C.validate_loader_output(ids,zeros.astype('f8'),'state',ids,zeros,'state'),'precision'))
 check('loader_wrong_state_reject',lambda:rejects(lambda:C.validate_loader_output(ids,zeros,'bad',ids,zeros,'state'),'state mismatch'))
 check('loader_wrong_values_reject',lambda:rejects(lambda:C.validate_loader_output(ids,zeros+1,'state',ids,zeros,'state'),'output mismatch'))
 # Real checkpoint primitive with synthetic outputs only; reopen and corrupt a completed slice.
 import numpy as np
 from card032_engine import save_checkpoint,load_checkpoint
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp);outputs={a:np.lib.format.open_memmap(d/(a+'.npy'),mode='w+',dtype='f4',shape=(8,3)) for a in C.ARMS}
  for a,v in outputs.items():v[:4]=np.arange(12).reshape(4,3)
  h={'lo':0,'hi':4,'sha':{a:hashlib.sha256(v[:4].tobytes()).hexdigest() for a,v in outputs.items()}}
  save_checkpoint(d,4,{},outputs,'model-input-release',[h]);outputs={a:np.load(d/(a+'.npy'),mmap_mode='r+') for a in C.ARMS}
  assert load_checkpoint(d,'model-input-release',outputs,'cpu')[0]==4;CHECKS.append('actual_checkpoint_reopen')
  for identity in ['wrong-model-input-release','model-wrong-input-release']:
   rejects(lambda:load_checkpoint(d,identity,outputs,'cpu'),'admission-identity mismatch');CHECKS.append(identity)
  outputs['uniform44'][0,0]+=1;rejects(lambda:load_checkpoint(d,'model-input-release',outputs,'cpu'),'completed-output hash mismatch');CHECKS.append('corrupt_output_checkpoint_rejected')
 return {'PASS':True,'checks':CHECKS,'count':len(CHECKS),'scope':'Synthetic CPU data and mocked actual-chain calls only; device numerical and measured gates pending.'}
def assert_equal(a,b):assert a==b,(a,b)
if __name__=='__main__':print(json.dumps(main(),indent=2))

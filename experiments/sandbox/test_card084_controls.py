"""CPU negative controls for finite loss/gradient, calibration and accounting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,tempfile,math
from pathlib import Path
import torch
import card084_common as C
sys.path.insert(0,str(C.R))
from basemap.pumap.parametric_umap.bounded_attraction import add_attraction,require_finite_gradient
from card084_calibration import coefficient_from_ratios
import card084_budget as B
checks=[]
def reject(name,call):
    try:call()
    except (ValueError,AssertionError,FloatingPointError):checks.append(name)
    else:raise AssertionError(name+' accepted')
x=torch.ones(2,3,requires_grad=True);y=torch.zeros_like(x);mask=torch.ones(2,dtype=torch.bool);scale=torch.ones(2)
def call(**kw):
    args=dict(coefficient=1.,delta=C.DELTA);args.update(kw)
    return add_attraction(torch.tensor(0.),x,y,mask,scale,**args)
for c in (-1.,float('nan'),float('inf')):reject('invalid coefficient '+str(c),lambda:call(coefficient=c))
for delta in (0.,-1.,float('nan'),float('inf')):reject('invalid delta '+str(delta),lambda:call(delta=delta))
reject('unknown family',lambda:call(family='cubic'))
for family in C.ARMS:
    xx=torch.full((2,3),1e30,requires_grad=True)
    reject(family+' nonfinite extra value',lambda:add_attraction(torch.tensor(0.),xx,y,mask,scale,coefficient=1.,family=family,delta=C.DELTA))
    # Finite forward but nonfinite backward, deliberately amplify upstream derivative.
    val=call(family=family)
    reject(family+' nonfinite extra backward',lambda:torch.autograd.grad(val,x,grad_outputs=torch.tensor(float('inf'))))
reject('nonfinite explicit gradient guard',lambda:require_finite_gradient(torch.tensor([float('nan')])))
reject('no positive edges',lambda:add_attraction(torch.tensor(0.),x,y,~mask,scale,coefficient=1.,delta=C.DELTA))
reject('zero radius product',lambda:add_attraction(torch.tensor(0.),x,y,mask,scale*0,coefficient=1.,delta=C.DELTA))
# Negative-pair endpoints have NaN data; they must not contaminate the positive-only term.
xx=torch.tensor([[1.,0.,0.],[float('nan'),0.,0.]],requires_grad=True)
v=add_attraction(torch.tensor(0.),xx,y,torch.tensor([True,False]),scale,coefficient=1.,delta=C.DELTA)
assert torch.isfinite(v) and torch.isfinite(torch.autograd.grad(v,xx)[0]).all()
checks.append('negative endpoints excluded even when nonfinite')
for ratios in ([1.]*7,[1.]*7+[0.],[1.]*7+[-1.],[1.]*7+[float('nan')],[1.]*7+[float('inf')],[1.]*7+[10.000001],[5e-324]*8):
    reject('invalid calibration '+repr(ratios),lambda:coefficient_from_ratios(ratios))
assert coefficient_from_ratios([1.]*7+[10.])==.1
assert coefficient_from_ratios([2.]*8)==.05
checks.append('spread10 boundary and exact0.1/median formula')
# Reject every required identity mutation before accessing any model/CUDA payload.
ident={'arm':'quadratic','family':'quadratic','delta':C.DELTA,'bank_sha':C.BANK_SHA,
       'dose':60000,'lr':.0001,'coefficient':.2,'attraction_normalization':'positive_mean',
       'calibration_sha':'cal','calibration_batches_sha':'batches','runtime_manifest_sha':'source',
       'fneg_weight':1.,'neg_tanh_gamma':4.}
C.validate_identity(ident)
for field,value in [('family','pseudo_huber'),('coefficient',.3),('delta',.5),
                    ('calibration_sha','wrong'),('runtime_manifest_sha','wrong'),
                    ('bank_sha','wrong'),('dose',59999),('calibration_batches_sha','wrong')]:
    mutated={**ident,field:value}
    reject('resume wrong '+field,lambda:C.validate_ckpt({'card012_identity':ident},mutated))
import run_card084_chain as chain
with tempfile.TemporaryDirectory(prefix='card084-release-cpu-') as td:
    old=C.O;C.O=Path(td)
    reject('chain refuses absent root release',chain.verify)
    reject('trainer refuses absent root release',C.require_gpu_stage)
    C.O=old
checks.append('root release gates evaluated without GPU access')
# Never write real ledgers. Test reservation/settlement and crash recovery in isolated CPU fixture.
with tempfile.TemporaryDirectory(prefix='card084-budget-cpu-') as td:
    old=C.O;C.O=Path(td);B.L=C.O/'card084-ledger.json';B.W=C.O/'window.json';B.J=C.O/'journal'
    C.write(B.W,{'spent_s':12.,'entries':[]})
    e=B.transact('fixture',100.,'quadratic',check=True)
    assert C.read(B.L)['arm_spent_s']=={'quadratic':100.,'pseudo_huber':0.}
    B.transact('fixture',-75.,'quadratic',kind='settlement')
    assert C.read(B.L)['batch_spent_s']==25. and C.read(B.W)['spent_s']==37.
    B.transact('shared',10.,check=True)
    reject('cumulative per-arm cap',lambda:B.transact('overspend',1800.,'quadratic',check=True))
    # Simulate interruption after window application, before local ledger application.
    event={'transaction':'crash-test','card':'084','tag':'crash','wall_s':9.,'arm_s':B.allocation(9.),'event':'reservation'}
    C.write(B.J/'crash-test.json',event)
    w=C.read(B.W);w['spent_s']+=9.;w['entries'].append(event);C.write(B.W,w)
    B.transact('recovery',0.)
    assert C.read(B.W)['spent_s']==56. and C.read(B.L)['batch_spent_s']==44.
    B.transact('idempotent',0.)
    assert C.read(B.W)['spent_s']==56. and C.read(B.L)['batch_spent_s']==44.
    checks.append('budget reservations, settlements, per-arm caps and crash replay idempotent')
    C.O=old
print(json.dumps({'PASS':True,'checks':checks,'n_checks':len(checks),'device':'cpu'},indent=2))

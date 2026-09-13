"""CPU contracts: a zero-exit stage may not reuse stale or invalid evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile,time
from unittest.mock import patch
from types import SimpleNamespace
import run_card084_chain as chain
from pathlib import Path
import card084_common as C
from run_card084_chain import validate_stage_receipt
checks=[]
with tempfile.TemporaryDirectory(prefix='card084-receipts-cpu-') as td:
    path=Path(td)/'receipt.json';start=time.time_ns();runtime='current-runtime';calibration='current-calibration'
    good={'PASS':True,'runtime_sha':runtime,'calibration_sha':calibration}
    def check_bad(name,expected):
        try:validate_stage_receipt(path,start,runtime,calibration)
        except AssertionError as error:
            assert str(error)==expected,(name,str(error))
            checks.append({'case':name,'rejected':True,'error':str(error)})
        else:raise AssertionError('invalid receipt accepted: '+name)
    check_bad('missing','missing stage receipt')
    C.write(path,good);os.utime(path,ns=(start-1,start-1))
    check_bad('old PASS from earlier stage','stale stage receipt')
    os.utime(path,ns=(start,start))
    assert validate_stage_receipt(path,start,runtime,calibration)==good
    checks.append({'case':'mtime exactly equal to stage start','PASS':True})
    for name,change,error in [
        ('false PASS',{'PASS':False},'stage receipt lacks explicit PASS'),
        ('numeric PASS',{'PASS':1},'stage receipt lacks explicit PASS'),
        ('wrong runtime',{'runtime_sha':'old-runtime'},'stage receipt runtime mismatch'),
        ('wrong calibration',{'calibration_sha':'old-calibration'},'stage receipt calibration mismatch')]:
        C.write(path,{**good,**change});os.utime(path,ns=(start+1,start+1));check_bad(name,error)
    for field,error in [('PASS','stage receipt lacks explicit PASS'),('runtime_sha','stage receipt runtime mismatch'),('calibration_sha','stage receipt calibration mismatch')]:
        missing={k:v for k,v in good.items() if k!=field};C.write(path,missing)
        os.utime(path,ns=(start+1,start+1));check_bad('missing '+field,error)
    C.write(path,good);os.utime(path,ns=(start+1,start+1))
    assert validate_stage_receipt(path,start,runtime,calibration)==good
    checks.append({'case':'fresh matching explicit PASS','PASS':True})
# Exercise the actual chain stage with an already-finished zero-exit child.
# All accounting, paths and process creation are isolated/mocked; no GPU access.
with tempfile.TemporaryDirectory(prefix='card084-zero-exit-cpu-') as td:
    out=Path(td);cal=out/'calibration.json';C.write(cal,{'fixed':'calibration'})
    cal_sha=C.sha(cal);runtime='current-runtime';receipt=out/'evidence.json'
    child=SimpleNamespace(returncode=0,poll=lambda:0)
    with patch.object(C,'O',out),patch.object(C,'CAL',cal),patch.object(C,'source_check',lambda:runtime), \
         patch.object(chain,'verify',lambda:'release-sha'),patch.object(chain.B,'available',lambda arm:100.), \
         patch.object(chain.B,'transact',lambda *args,**kwargs:None),patch.object(chain.subprocess,'Popen',lambda *args,**kwargs:child):
        for name,expected in [('missing','missing stage receipt'),('stale','stale stage receipt'),('wrong_runtime','stage receipt runtime mismatch')]:
            if receipt.exists():receipt.unlink()
            if name!='missing':
                C.write(receipt,{'PASS':True,'runtime_sha':runtime if name=='stale' else 'old','calibration_sha':cal_sha})
                stamp=1 if name=='stale' else time.time_ns()+1_000_000_000
                os.utime(receipt,ns=(stamp,stamp))
            try:chain.stage(name,'unused-cpu-fixture.py',30,receipt_path=receipt)
            except AssertionError as error:assert str(error)==expected
            else:raise AssertionError('zero-exit child accepted '+name+' receipt')
            audit=C.read(out/f'card084-stage-{name}.json')
            assert audit['rc']==0 and audit['PASS'] is False and audit['receipt_proof'] is None
            assert expected in audit['error']
            checks.append({'case':'actual chain zero-exit '+name,'rejected':True,'recorded_rc':audit['rc'],'recorded_error':audit['error']})
print(json.dumps({'PASS':True,'checks':checks,'n_checks':len(checks),'device':'cpu','scope':'Temporary CPU receipts only; no GPU stage executed'},indent=2))

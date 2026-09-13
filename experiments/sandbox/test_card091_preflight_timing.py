"""Execute actual preflight tail AST under a mocked clock, without importing GPU code."""
import ast,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
SOURCE=Path(__file__).with_name('run_card091_gpu.py')

def run_tail(tail,positions,fail_resources=False):
    clock=[0.];events=[];timings=[]
    def work(name,seconds):
        events.append(name);clock[0]+=seconds
        if name=='resources' and fail_resources:raise RuntimeError('injected resource STOP')
    scope={'td':None,'CHUNK':32768,'out':{},'h':{},'hist32':[{}],
           'save_checkpoint':lambda *args:work('checkpoint',2),
           'torch':SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:work('synchronize',3))),
           'resources':lambda:work('resources',7),
           'time':SimpleNamespace(monotonic=lambda:clock[0]),'timings':timings}
    code=compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(tail),type_ignores=[])),str(SOURCE),'exec')
    for lo in positions:
        scope['t']=clock[0]
        try:exec(code,scope)
        except RuntimeError as e:
            assert fail_resources and str(e)=='injected resource STOP'
            assert not timings,'failed resource sample was recorded'
            return timings,events
    return timings,events

def main():
    tree=ast.parse(SOURCE.read_text())
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='timings' and c.func.attr=='append' for x in n.body for c in ast.walk(x))]
    assert len(loops)==1;loop=loops[0];tail=loop.body[-4:]
    positions=eval(compile(ast.Expression(loop.iter),str(SOURCE),'eval'),{'LOW':19344847,'HIGH':103816750,'CHUNK':32768})
    assert positions==[0,32768,20344847,39344847,103751214,103783982]
    measured,events=run_tail(tail,positions)
    assert measured==[12.]*6,'recurring resource overhead omitted from measured windows'
    assert events==['checkpoint','synchronize','resources']*6,'wrong recurring call order'
    # Reproduce prior timing-before-resources omission from the same extracted statements.
    regressed=tail[:2]+[tail[3],tail[2]]
    old,_=run_tail(regressed,positions);assert old==[5.]*6 and old!=measured,'negative timing-order control not discriminating'
    run_tail(tail,positions,fail_resources=True)
    compile(SOURCE.read_text(),str(SOURCE),'exec')
    return {'PASS':True,'checks':{'six_original_positions':True,'all_six_include_resource_seconds':True,'sync_resources_before_elapsed':True,'old_order_negative_control':True,'resource_failure_not_recorded':True,'changed_producer_compiles':True},'count':6,'source_sha':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'test_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'measured_s':measured,'old_order_s':old,'scope':'Actual preflight final four AST statements under mocked checkpoint/synchronization/resource costs; no producer import, inference, endpoint access or GPU.'}
if __name__=='__main__':print(json.dumps(main(),indent=2))

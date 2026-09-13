"""One-backbone confined inference; no training or core changes."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from _paths import ensure_paths
ensure_paths()
from basemap.pumap.parametric_umap.core import ParametricUMAP

R=Path(__file__).resolve().parents[2]
O=R.parent/'overseer-codex'
DATA=O/'card057-inputs'
HEADS=('ordinary2m','half4m')
BATCH=256

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def atomic(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def kernel(teacher,z,weight,radius):
    """Branch-free implementation of the frozen scalar operator.

    Input validity is checked at admission and output validity after projection.
    FP64 work is deliberately included in the deployment timing. Active rows
    use torch.where's original FP32 branch, preserving even signed zero.
    """
    inactive=z.amax(1)<=0
    margin=(-z.amax(1)).clamp_min(0).double()
    delta=1e-4*(margin/(margin+1))[:,None]*(z.double()@weight.double().T)
    norms=torch.linalg.vector_norm(delta,dim=1)
    factor=((.001*radius)/norms.clamp_min(torch.finfo(torch.float64).tiny)).clamp_max(1)
    delta=delta*factor[:,None]
    candidate=(teacher.double()+delta).float()
    result=torch.where(inactive[:,None],candidate,teacher)
    return result,delta,inactive

def loaded_files():
    out={}
    for name,module in list(sys.modules.items()):
        if not name.startswith('basemap') or not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve()
        assert path.is_relative_to(R),(name,str(path))
        out[name]={'path':str(path),'sha':sha(path)}
    return out

def load_head(name,device):
    assert name in HEADS
    panel=json.loads((O/'card057-panel/manifest.json').read_text())
    rec=panel['heads'][name]['model'];assert sha(rec['path'])==rec['sha']
    saved=torch.load(rec['path'],map_location='cpu',weights_only=False)
    m=ParametricUMAP.load(rec['path'],device=device).model.float().eval().requires_grad_(False)
    assert m.proj_in.in_features==1536 and m.proj_out.out_features==3
    assert isinstance(m.up[1],torch.nn.ReLU)
    assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) for k,v in m.state_dict().items())
    loaded_files()
    return m,float(panel['heads'][name]['radius'])

@torch.inference_mode()
def project(m,inputs,radius,mode='repair',collect=False):
    assert mode in ('teacher','repair')
    device=next(m.parameters()).device
    output=[];teachers=[];deltas=[];masks=[];inactive_z=[];inactive_rows=[]
    held={};hook=None
    if mode=='repair':hook=m.up[0].register_forward_hook(lambda module,args,z:held.update(z=z))
    try:
        for lo in range(0,len(inputs),BATCH):
            x=torch.from_numpy(np.array(inputs[lo:lo+BATCH],dtype='f4',copy=True)).to(device)
            teacher=m(x)
            if mode=='repair':
                z=held.pop('z');y,delta,inactive=kernel(teacher,z,m.proj_out.weight,radius)
                if collect:
                    teachers.append(teacher.cpu().numpy());deltas.append(delta.cpu().numpy());masks.append(inactive.cpu().numpy())
                    chosen=torch.nonzero(inactive).flatten()
                    inactive_z.append(z[chosen].cpu().numpy());inactive_rows.append(chosen.cpu().numpy()+lo)
            else:y=teacher
            output.append(y.cpu().numpy())
    finally:
        if hook is not None:hook.remove()
    out={'xy':np.concatenate(output)}
    if collect:
        assert mode=='repair'
        out.update(teacher=np.concatenate(teachers),delta64=np.concatenate(deltas),inactive=np.concatenate(masks),
                   inactive_preactivation=np.concatenate(inactive_z),inactive_rows=np.concatenate(inactive_rows))
    assert all(np.isfinite(v).all() for v in out.values())
    return out

def device_setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()


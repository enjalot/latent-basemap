"""Separate-process unchanged052 core control for054's default-off canary."""
from pathlib import Path
import sys,numpy as np,torch
R=Path('/data/latent-basemap/sandbox/card052-code');sys.path[:0]=[str(R),str(R/'experiments/sandbox')]
from card052_fit import fit
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
D=Path(sys.argv[1]);p,ck,r=fit('ordinary',18,D/'old052',X=np.load(D/'X.npy'),graph=D/'edges.npz',radius_path=D/'radii.npy',checkpoints=[2,4,7,9,18]);torch.save(ck,D/'old052-end.pt')

"""CPU freeze only; never launches GPU work."""
from run_card020 import *
def main():
 c=torch.load(PARENT,map_location='cpu',weights_only=False);assert c['successful_steps']==20000 and c['identity']['arm']=='compact_l1'
 b={'parent_path':str(PARENT),'parent_file_sha':sha(PARENT),'parent_identity':c['identity'],'data_manifest_sha':sha(DATA/'manifest.json'),'input_path':str(POOL/'pool_X.f16.npy'),'input_file_sha':sha(POOL/'pool_X.f16.npy')}
 for key,field in [('model_state_dict','parent_model_sha'),('optimizer_state_dict','parent_optimizer_sha'),('batch_rng','parent_batch_rng_sha'),('cpu_rng','parent_cpu_rng_sha'),('cuda_rng','parent_cuda_rng_sha')]:b[field]=objsha(c[key])
 write(ROOT/'card020-admission-base.json',b)
 files=list((ROOT/'basemap').rglob('*.py'))+[ROOT/'experiments/sandbox'/n for n in ['card016_model.py','run_card016_arm.py','benchmark_card016.py','_paths.py','run_card020.py','card020_canary.py','run_card020_chain.py','score_card020.py','card020_instrument.py','prepare_card020.py','card020_validator_canary.py']]+[ROOT/'card020-admission-base.json']
 write(ROOT/'card020-runtime-sha.json',{str(p.relative_to(ROOT)):sha(p) for p in files});validate_inputs();print('CPU inputs/runtime frozen; parent model',b['parent_model_sha'])
if __name__=='__main__':main()

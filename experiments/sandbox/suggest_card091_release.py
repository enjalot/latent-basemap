"""Root-invoked candidate only; never creates the operative GPU/CPU release."""
import time
import card091_common as C

def build_candidate():
 start=time.monotonic();s=C.validate_selection(C.read(C.D/'selection.json'));C.verify_sources(s)
 r=C.read(C.O/'card091-readiness/suggested-bindings.json');files=dict(r['files'])
 files.update(s['endpoint_bindings']);files[str(C.D/'selection.json')]=C.sha(C.D/'selection.json')
 files[str(C.D/'selection-build.json')]=C.sha(C.D/'selection-build.json')
 assert all(C.sha(p)==h for p,h in files.items()),'candidate binding changed'
 assert not any(str(C.D/(a+suffix+'.npy')) in files for a in C.ARMS for suffix in ['', '-positive-final']),'generated projection output bound as input'
 target=C.D/'release-candidate.json';assert not target.exists(),'existing candidate requires root preservation/review'
 proposal={'PASS':True,'status':'CANDIDATE_ONLY_NO_RELEASE','card':'091','runtime_sha':C.source_check(),'limits':C.LIMITS,'selection_sha':C.sha(C.D/'selection.json'),'files':files,'root_must_freeze':True,'cpu_builder_wall_s':time.monotonic()-start,'cpu_selection_wall_s':C.read(C.D/'selection-build.json')['cpu_wall_s'],'pending':['actual device loader/fidelity/sensor/buffering/resume','measured complete four-head joint admission','root GPU release and both external leases','root ONE1200CPU handoff']}
 C.write(target,proposal);return proposal
if __name__=='__main__':build_candidate()

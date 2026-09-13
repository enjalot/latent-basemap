"""CPU source manifest only. Does not create a GPU release or launch anything."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2]
paths=sorted(set(R.joinpath('basemap').rglob('*.py'))|set(R.joinpath('experiments/sandbox').glob('*card084*.py'))|{R/'experiments/sandbox/card084-protocol.md',R/'experiments/sandbox/card084-implementation.md'})
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(R/'card084-runtime-sha.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('CPU manifest',len(manifest),'files; NOT a GPU release')

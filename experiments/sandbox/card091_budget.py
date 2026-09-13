"""Idempotent cumulative reservation/settlement journal under canonical lock."""
import fcntl,uuid,datetime as dt,math
import card091_common as C
L=C.O/'card091-ledger.json';W=C.O/'cards-24h-window-ledger.json';J=C.O/'card091-accounting-journal'
def apply(e):
 for p in [W,L]:
  r=C.read(p)
  if any(x.get('transaction')==e['transaction'] for x in r['entries']):continue
  r['spent_s']+=e['wall_s'];r['entries'].append(e);C.write(p,r)
def transact(seconds,kind,check=False):
 assert math.isfinite(seconds),'nonfinite accounting'
 with (C.O/'window-ledger-write.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert W.exists(),'root window ledger required'
  if not L.exists():C.write(L,{'cap_s':3600,'spent_s':0.,'entries':[]})
  J.mkdir(exist_ok=True)
  for p in sorted(J.glob('*.json')):apply(C.read(p))
  if check:assert C.read(L)['spent_s']+seconds<=3600 and C.read(W)['spent_s']+seconds<=165491,'cumulative cap STOP'
  e={'transaction':uuid.uuid4().hex,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'card':'091','event':kind,'tag':'four_head_projection_occupancy','wall_s':seconds}
  C.write(J/(e['transaction']+'.json'),e);apply(e)
  return e

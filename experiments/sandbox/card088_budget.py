"""Conservative reservation journal, replayable under sole rolling-ledger lock."""
from pathlib import Path
import datetime as dt
import fcntl,uuid
import card088_common as C
L=C.O/'card088-ledger.json';W=C.O/'cards-24h-window-ledger.json'
J=C.O/'card088-accounting-journal'
END=dt.datetime.fromisoformat('2026-09-13T23:50:55+00:00').timestamp()
LIMITS={'card_gpu_s':4800,'stage_gpu_s':{'original15':1600,'mixture':1600},'global_vram_gib':30,'rss_gib':48,'deadline':'2026-09-13T23:50:55Z'}

def allocation(seconds,arm=None):
    return {a:(seconds if arm==a else 0.) if arm else 0. for a in C.ARMS}

def _apply(event):
    # Window first: crashes cannot leave uncharged GPU authorization. Replay is idempotent.
    for path,key in ((W,'spent_s'),(L,'batch_spent_s')):
        r=C.read(path)
        if any(e.get('transaction')==event['transaction'] for e in r['entries']):continue
        r[key]+=event['wall_s'];r['entries'].append(event)
        if path==L:
            for a,v in event['arm_s'].items():r['arm_spent_s'][a]+=v
        C.write(path,r)

def _recover():
    J.mkdir(exist_ok=True)
    for p in sorted(J.glob('*.json')):_apply(C.read(p))

def _event(tag,seconds,arm,kind):
    event={'transaction':uuid.uuid4().hex,'at':dt.datetime.now(dt.timezone.utc).isoformat(),
           'card':'088','tag':tag,'event':kind,'wall_s':seconds,'arm_s':allocation(seconds,arm)}
    C.write(J/(event['transaction']+'.json'),event);_apply(event)
    return event

def transact(tag,seconds,arm=None,kind='reservation',check=False):
    with (C.O/'window-ledger-write.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not L.exists():C.write(L,{'batch_cap_s':4800,'batch_spent_s':0.,'arm_spent_s':{a:0. for a in C.ARMS},'entries':[]})
        assert W.exists(),'root rolling ledger required'
        _recover()
        if check:
            l=C.read(L);w=C.read(W)
            assert l['batch_spent_s']+seconds<=4800 and w['spent_s']+seconds<=165491,'cumulative budget STOP'
            assert all(l['arm_spent_s'][a]+v<=LIMITS['stage_gpu_s'][a] for a,v in allocation(seconds,arm).items()),'per-arm cumulative STOP'
        return _event(tag,seconds,arm,kind)

def available(arm=None):
    import time
    l=C.read(L);w=C.read(W)
    remaining=[4800-l['batch_spent_s'],165491-w['spent_s'],END-time.time()]
    if arm:remaining.append(LIMITS['stage_gpu_s'][arm]-l['arm_spent_s'][arm])
    return min(remaining)

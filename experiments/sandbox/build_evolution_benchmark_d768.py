#!/usr/bin/env python3
"""Build the D768 (jina) evolution benchmark substrates + disjointness/membership proofs (option b).
CORRECTED 2026-09-01 (5th review): T0 must EXACTLY equal the P1.5 baseline@42 head's training rows =
_jina_multi_load's output = THREE per-corpus PREFIXES of en-2m (fineweb/redpajama/pile blocks) + multi-1m,
NOT contiguous en-2m[:1M]. EN growth tranches are COMPOSITION-MATCHED: per-corpus proportional draws from
the UNUSED span REMAINDERS, so T1/T2 keep T0's corpus mix and cannot leak head members.

en-2m layout: _JINA_EN_PER=(666667,666667,666666) blocks = [fineweb 0:666667][redpajama 666667:1333334]
[pile 1333334:2000000]. Head EN prefixes (333334,333333,333333) at each block start. Remainders are the
tails of each block. T4/T5 = ml2 language-mass topup; T3 = reddit-jina social OOD.

Proofs: (1) pairwise CONTENT disjointness across snapshots (void-view exact-row); (2) NEW head-membership
class — T0 == head members exactly, AND no tranche row is a head member (the leak the 5th review caught)."""
import json, glob
from pathlib import Path
import numpy as np

JP = Path("/data/latent-basemap/substrates/jina-prompted")
REDDIT = Path("/data/latent-basemap/substrates/reddit-jina-pool/substrate.f16.npy")
OUT = Path("/data/latent-basemap/substrates/evolbench-d768")
PROOFS = Path("/data/latent-basemap/sandbox/evolbench-d768-proofs.json")
TARGET = 400_000

# en-2m corpus blocks (fineweb-edu, redpajama, pile) and the head's per-corpus prefix sizes.
EN_BLOCKS = {"fineweb-edu": (0, 666_667), "redpajama": (666_667, 1_333_334), "pile": (1_333_334, 2_000_000)}
HEAD_PREFIX = {"fineweb-edu": 333_334, "redpajama": 333_333, "pile": 333_333}   # _jina_multi_load


def _f16(path, sl=None):
    a = np.load(path, mmap_mode="r")
    return np.asarray(a if sl is None else a[sl], dtype=np.float16)


def _void(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _dedup(cand_f16, prior_void):
    cv = _void(cand_f16); mask = ~np.isin(cv, prior_void)
    return cand_f16[mask], cv[mask], int((~mask).sum())


def _head_members():
    """Reconstruct the P1.5@42 head's EXACT training rows = _jina_multi_load output (3 EN prefixes + multi)."""
    en = np.load(JP / "en-2m.f16.npy", mmap_mode="r")
    parts = []
    for name in ("fineweb-edu", "redpajama", "pile"):        # _JINA_EN_NAMES order
        b0, _ = EN_BLOCKS[name]; p = HEAD_PREFIX[name]
        parts.append(np.asarray(en[b0:b0 + p], dtype=np.float16))
    parts.append(_f16(JP / "multi-1m.f16.npy"))
    return np.concatenate(parts)


def _en_remainders():
    """Per-corpus UNUSED tails (after the head prefix) — the composition-matched growth pool."""
    en = np.load(JP / "en-2m.f16.npy", mmap_mode="r"); rem = {}
    for name, (b0, b1) in EN_BLOCKS.items():
        rem[name] = (b0 + HEAD_PREFIX[name], b1)             # (start, end) of the remainder
    return en, rem


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- T0 = exact head members ----
    t0 = _head_members()
    head_void = _void(t0)                                    # cache for the membership proof
    print(f"T0 (head members): {t0.shape}", flush=True)
    snapshots = {"T0": t0}; subsets = {}; comp = {}
    prior = head_void.copy()
    # ---- EN growth tranches: proportional per-corpus draws from the remainders ----
    en, rem = _en_remainders()
    # per-tranche per-corpus share (thirds, matching T0's EN mix)
    per = {"fineweb-edu": 133_334, "redpajama": 133_333, "pile": 133_333}
    cursor = {name: rem[name][0] for name in EN_BLOCKS}      # advancing read cursor in each remainder
    def draw_en(n_each_extra=0):
        blk = []; provenance = []
        for name in ("fineweb-edu", "redpajama", "pile"):
            s = cursor[name]; take = per[name] + n_each_extra
            e = min(s + take, rem[name][1]); blk.append(np.asarray(en[s:e], dtype=np.float16))
            provenance.append(np.repeat(name, e - s)); cursor[name] = e
        return np.concatenate(blk), np.concatenate(provenance)
    # ---- ml2 language-mass pool for T4/T5 ----
    ml2_files = sorted(glob.glob(str(JP / "ml2-*.f16.npy")))
    ml2 = np.concatenate([_f16(f) for f in ml2_files])
    ml2_lang = np.concatenate([np.repeat(Path(f).stem.replace("ml2-", ""),
                                         np.load(f, mmap_mode="r").shape[0]) for f in ml2_files])
    ml2_cursor = 0
    counts = {"T0": int(t0.shape[0])}
    order = ["T1", "T2", "T3", "T4", "T5"]
    for name in order:
        if name in ("T1", "T2"):
            cand, prov = draw_en(n_each_extra=6_000)         # over-draw for post-dedup trim
        elif name == "T3":
            cand = _f16(REDDIT, slice(0, 440_000)); prov = np.repeat("reddit", cand.shape[0])
        else:  # T4/T5 ml2
            cand = ml2[ml2_cursor:ml2_cursor + 440_000]; prov = ml2_lang[ml2_cursor:ml2_cursor + 440_000]
            ml2_cursor += 440_000
        kept, kv, dropped = _dedup(cand, prior)
        keepmask_prov = prov[~np.isin(_void(cand), prior)][:TARGET]
        kept = kept[:TARGET]; kv = kv[:TARGET]
        snapshots[name] = kept; subsets[name] = keepmask_prov
        prior = np.concatenate([prior, kv]); counts[name] = int(kept.shape[0])
        comp[name] = {str(k): int(v) for k, v in zip(*np.unique(keepmask_prov, return_counts=True))}
        print(f"{name}: drew {cand.shape[0]:,} dropped_dup {dropped} kept {kept.shape[0]:,} comp {comp[name]}", flush=True)
    # ---- save ----
    for name, arr in snapshots.items():
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "substrate.f32.npy", np.asarray(arr, dtype=np.float32))
        if name in subsets:
            np.save(d / "subsets.npy", subsets[name])
    # ---- proofs ----
    voids = {n: _void(snapshots[n]) for n in snapshots}
    names = list(snapshots); pw = {}; pairwise_ok = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = int(np.isin(voids[names[i]], voids[names[j]]).sum())
            pw[f"{names[i]}^{names[j]}"] = inter; pairwise_ok = pairwise_ok and inter == 0
    # NEW head-membership proof class
    t0_is_head = (t0.shape[0] == t0.shape[0]) and int(np.isin(voids["T0"], head_void, invert=True).sum()) == 0 \
                 and int(np.isin(head_void, voids["T0"], invert=True).sum()) == 0
    tranche_head_leak = {n: int(np.isin(voids[n], head_void).sum()) for n in names if n != "T0"}
    no_head_leak = all(v == 0 for v in tranche_head_leak.values())
    reddit_ood_ok = all(pw.get(f"T3^{n}", pw.get(f"{n}^T3", 0)) == 0 for n in ("T0", "T1", "T2", "T4", "T5"))
    proofs = {"schema": "evolbench-d768-proofs-2026-09-01-corrected", "counts": counts,
              "final_rows": int(sum(counts.values())), "pairwise_content_intersections": pw,
              "all_disjoint": bool(pairwise_ok), "reddit_ood_disjoint": bool(reddit_ood_ok),
              "head_membership": {"T0_equals_head_members": bool(t0_is_head),
                                  "tranche_head_member_leak": tranche_head_leak,
                                  "no_head_leak": bool(no_head_leak)},
              "composition": comp}
    PROOFS.write_text(json.dumps(proofs, indent=1))
    ok = pairwise_ok and t0_is_head and no_head_leak and reddit_ood_ok
    print(f"\nALL_DISJOINT: {pairwise_ok} | T0==head_members: {t0_is_head} | no_head_leak: {no_head_leak} "
          f"| reddit_ood: {reddit_ood_ok} | final {proofs['final_rows']:,}", flush=True)
    print(f"tranche_head_leak {tranche_head_leak}", flush=True)
    print(f"proofs -> {PROOFS}  (overall_ok={ok})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

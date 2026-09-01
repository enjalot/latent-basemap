#!/usr/bin/env python3
"""Build the D768 (jina) evolution benchmark substrates + disjointness proofs (overseer option b, 2026-09-01).
T0=2M (jina-multi-2m = en-2m[:1M] + multi-1m, the P1.5 baseline@42 head's training set) + 5×400K tranches
-> final 4M. T1/T2 = en-2m[1M:2M] (in-dist EN growth); T4/T5 = ml2-<lang> (multilingual growth, enables
language-register floors); T3 = reddit-jina-pool (social OOD, drift-trigger positive control).

Disjointness = CONTENT (void-view exact-row identity) against T0 ∪ earlier tranches — the empirical scan
found ~0.03% exact-duplicate embeddings (dup text->dup vector) across index-disjoint spans, so index math
alone is not enough. Each tranche is drawn slightly over 400K, content-deduped, trimmed to <=400K; actual
counts recorded. Saves substrate.f32.npy per Tk (+ subsets.npy for T4/T5 langs) and a proofs JSON."""
import json
from pathlib import Path
import numpy as np

JP = Path("/data/latent-basemap/substrates/jina-prompted")
REDDIT = Path("/data/latent-basemap/substrates/reddit-jina-pool/substrate.f16.npy")
OUT = Path("/data/latent-basemap/substrates/evolbench-d768")
PROOFS = Path("/data/latent-basemap/sandbox/evolbench-d768-proofs.json")
TARGET = 400_000
DRAW = 440_000          # over-draw, then content-dedup down to <=TARGET
import glob

_JINA_EN_PER = (333_334, 333_333, 333_333)   # en-2m[:1M] structure (matches image_map_pipeline)


def _f16(path, sl=None):
    a = np.load(path, mmap_mode="r")
    return np.asarray(a if sl is None else a[sl], dtype=np.float16)


def _void(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _dedup(cand_f16, prior_void):
    """Drop rows of cand whose exact bytes appear in prior_void; return (kept_f16, kept_void, n_dropped)."""
    cv = _void(cand_f16)
    mask = ~np.isin(cv, prior_void)
    return cand_f16[mask], cv[mask], int((~mask).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    en2 = np.load(JP / "en-2m.f16.npy", mmap_mode="r")
    # ---- T0 = en-2m[:1M] + multi-1m (the head's training set) ----
    t0 = np.concatenate([_f16(JP / "en-2m.f16.npy", slice(0, 1_000_000)),
                         _f16(JP / "multi-1m.f16.npy")])
    print(f"T0: {t0.shape}", flush=True)
    prior = _void(t0)
    snapshots = {"T0": t0}
    subsets = {}
    # tranche source draws (over-draw DRAW rows each; T4/T5 track lang provenance)
    ml2_files = sorted(glob.glob(str(JP / "ml2-*.f16.npy")))
    ml2 = np.concatenate([_f16(f) for f in ml2_files])
    ml2_lang = np.concatenate([np.repeat(Path(f).stem.replace("ml2-", ""), np.load(f, mmap_mode="r").shape[0])
                               for f in ml2_files])
    draws = {
        "T1": (_f16(JP / "en-2m.f16.npy", slice(1_000_000, 1_000_000 + DRAW)), None),
        "T2": (_f16(JP / "en-2m.f16.npy", slice(1_440_000, 1_440_000 + DRAW)), None),
        "T3": (_f16(REDDIT, slice(0, DRAW)), None),                       # OOD
        "T4": (ml2[:DRAW], ml2_lang[:DRAW]),                             # multilingual
        "T5": (ml2[DRAW:2 * DRAW], ml2_lang[DRAW:2 * DRAW]),
    }
    counts = {"T0": int(t0.shape[0])}
    for name in ["T1", "T2", "T3", "T4", "T5"]:
        cand, lang = draws[name]
        kept, kv, dropped = _dedup(cand, prior)
        kept = kept[:TARGET]; kv = kv[:TARGET]
        snapshots[name] = kept
        prior = np.concatenate([prior, kv])
        counts[name] = int(kept.shape[0])
        print(f"{name}: drew {cand.shape[0]:,} dropped_dup {dropped} kept {kept.shape[0]:,}", flush=True)
    # realign T4/T5 lang subsets to the kept rows (recompute mask cleanly)
    for name in ("T4", "T5"):
        cand, lang = draws[name]
        cv = _void(cand); m = ~np.isin(cv, _void(np.concatenate([snapshots[s] for s in
              (["T0"] + [f"T{j}" for j in range(1, int(name[1]))])])))
        subsets[name] = lang[m][:counts[name]]
    # ---- save substrates (f32 for _load_Sk/knn) + T4/T5 subsets ----
    for name, arr in snapshots.items():
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "substrate.f32.npy", np.asarray(arr, dtype=np.float32))
    for name in ("T4", "T5"):
        np.save(OUT / name / "subsets.npy", subsets[name])
    # ---- proofs: pairwise CONTENT disjointness (exact-row) across all snapshots ----
    voids = {n: _void(snapshots[n]) for n in snapshots}
    names = list(snapshots)
    pairwise_ok = True; pw = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = int(np.isin(voids[names[i]], voids[names[j]]).sum())
            pw[f"{names[i]}∩{names[j]}"] = inter
            pairwise_ok = pairwise_ok and (inter == 0)
    reddit_ood_ok = all(pw[f"T3∩{n}"] == 0 for n in ("T4", "T5")) and \
                    all(pw.get(f"{a}∩T3", pw.get(f"T3∩{a}", 0)) == 0 for a in ("T0", "T1", "T2"))
    proofs = {"schema": "evolbench-d768-proofs-2026-09-01", "counts": counts,
              "final_rows": int(sum(counts.values())), "pairwise_content_intersections": pw,
              "all_disjoint": bool(pairwise_ok), "reddit_ood_disjoint": bool(reddit_ood_ok),
              "t4_langs": {str(k): int(v) for k, v in zip(*np.unique(subsets["T4"], return_counts=True))},
              "t5_langs": {str(k): int(v) for k, v in zip(*np.unique(subsets["T5"], return_counts=True))}}
    PROOFS.write_text(json.dumps(proofs, indent=1))
    print(f"\nALL_DISJOINT: {pairwise_ok} | final {proofs['final_rows']:,} | counts {counts}", flush=True)
    print(f"reddit_ood_disjoint: {reddit_ood_ok}", flush=True)
    print(f"proofs -> {PROOFS}", flush=True)
    return 0 if pairwise_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

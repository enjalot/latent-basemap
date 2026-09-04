"""NeoMME exp-2 matched-partner-rank probe (owner NeoMME probe, overseer 2026-09-04). Turns "modalities remain
separate manifolds" from qualitative into measured. For a sample of image rows, in the 1024-d SOURCE space
(cosine), measures for RAW and CENTERED substrates:
  - matched-partner rank: where does the image's OWN caption (row N+i) fall in its cosine-ranked neighbor list?
    median rank + fraction within k=15 / 100 / 1000. (Small => pairs resolved; huge => pairs buried.)
  - first-cross-modal-neighbor rank: rank of the NEAREST text neighbor of ANY kind. This is the discriminator:
    small (~1-15) => modality is NOT a hard partition, pairs are just subordinate to within-image similarity;
    large (~thousands) => modality is a HARD partition (no text is near any image at all).

CPU-only, cosine over unit-norm rows. Loads each 2GB substrate once into RAM (repeated full scans; workstation
RAM is ample) and scans in query batches. Usage: neomme_pair_rank_probe.py [SAMPLE=4000]."""
import sys, json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
ARMS = {"raw": Path("/data2/monet/neomme-pairs-500k"),
        "centered": Path("/data2/monet/neomme-pairs-500k-centered")}
KS = (15, 100, 1000)


def probe(dat, sample, seed=42):
    sub = np.array(np.load(dat / "substrate.f32.npy"), dtype=np.float32)   # 2GB into RAM for repeated scans
    mod = np.load(dat / "modality.npy")
    n_rows = sub.shape[0]; N = n_rows // 2
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(N, min(sample, N), replace=False))              # sample IMAGE rows [0,N)
    text_mask = (mod == 1)
    matched_rank = np.empty(len(qi), np.int64)
    first_text_rank = np.empty(len(qi), np.int64)
    B = 128
    for s in range(0, len(qi), B):
        idx = qi[s:s + B]
        sims = np.array(sub[idx], dtype=np.float32) @ sub.T                 # (b, n_rows)
        for r, i in enumerate(idx):
            row = sims[r]; row[i] = -2.0                                    # drop self
            mcos = row[N + i]                                               # matched caption cosine
            matched_rank[s + r] = int((row > mcos).sum())                   # 0-indexed rank
            best_text = row[text_mask].max()
            first_text_rank[s + r] = int((row > best_text).sum())
    def stats(a):
        return {"median": int(np.median(a)), "mean": round(float(a.mean()), 1),
                **{f"frac_within_{k}": round(float((a < k).mean()), 4) for k in KS}}
    return {"n_sample": len(qi), "matched_partner_rank": stats(matched_rank),
            "first_crossmodal_neighbor_rank": stats(first_text_rank)}


def main():
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    out = {arm: probe(dat, sample) for arm, dat in ARMS.items() if (dat / "substrate.f32.npy").exists()}
    out["interpretation"] = ("matched rank small => pairs resolved; first-crossmodal rank small (~<15) => modality "
                             "NOT a hard partition (pairs subordinate); first-crossmodal rank huge => hard partition")
    (SB / "monet-neomme-pairs-500k-centered" / "champion-bs16k" / "pair_rank_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

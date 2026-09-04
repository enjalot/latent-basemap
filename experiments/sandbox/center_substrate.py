"""Mean-center + renormalize a NeoMME substrate to remove the anisotropic cone (owner NeoMME probe, overseer
2026-09-04). exp-2 showed NeoMME's pooled space is a near-degenerate cone (mean-norm ~0.84-0.95 on UNIT vectors)
that swamps pairwise structure in raw cosine; the champion's cosine kNN graph maps the cone, not the content.
This produces a centered substrate for a re-map: subtract the mean (per-substrate, or per-modality when a
modality.npy is given) then RENORMALIZE to unit length, so cosine/kNN see the residual content.

Memory-safe: chunked mean + chunked transform over the memmap (never materializes the >2GB substrate).
Records the centering mode + cone_stats (the removed mean norms) in the output manifest as a substrate-prep
annotation. Usage: center_substrate.py <in_substrate.npy> <out_dir> [--modality <modality.npy>]"""
import sys, json
from pathlib import Path
import numpy as np

C = 100_000


def _chunk_mean(sub, idx=None):
    d = sub.shape[1]; acc = np.zeros(d, np.float64); n = 0
    rows = range(0, sub.shape[0], C) if idx is None else None
    if idx is None:
        for c in range(0, sub.shape[0], C):
            b = np.array(sub[c:c + C], dtype=np.float64); acc += b.sum(0); n += b.shape[0]
    else:
        for c in range(0, len(idx), C):
            b = np.array(sub[idx[c:c + C]], dtype=np.float64); acc += b.sum(0); n += b.shape[0]
    return (acc / n).astype(np.float32)


def main():
    infile = Path(sys.argv[1]); outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    modf = None
    if "--modality" in sys.argv:
        modf = sys.argv[sys.argv.index("--modality") + 1]
    sub = np.load(infile, mmap_mode="r"); N, D = sub.shape
    if modf:
        mod = np.load(modf); assert mod.shape[0] == N, "modality/substrate mismatch"
        means = {int(m): _chunk_mean(sub, np.where(mod == m)[0]) for m in np.unique(mod)}
        cone = {f"mean_norm_mod{m}": round(float(np.linalg.norm(v)), 4) for m, v in means.items()}
        mode = "per-modality"
    else:
        m0 = _chunk_mean(sub); means = {0: m0}; cone = {"mean_norm": round(float(np.linalg.norm(m0)), 4)}
        mode = "per-substrate"
    out = np.lib.format.open_memmap(outdir / "substrate.f32.npy", mode="w+", dtype=np.float32, shape=(N, D))
    for c in range(0, N, C):
        b = np.array(sub[c:c + C], dtype=np.float32)
        if modf:
            mb = mod[c:c + C]
            for mkey, mvec in means.items():
                sel = mb == mkey
                if sel.any():
                    b[sel] -= mvec
        else:
            b -= means[0]
        b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)   # renorm to unit length
        out[c:c + C] = b
    out.flush()
    (outdir / "manifest.json").write_text(json.dumps({
        "schema": "neomme-centered-substrate-2026-09-04", "source": str(infile), "n": int(N), "dim": int(D),
        "substrate_prep": f"mean-centered ({mode}) + renormalized to remove anisotropic cone",
        "cone_stats_removed": cone,
        "note": "cone_stats should be a standard diagnostic before ANY new-encoder map (overseer 2026-09-04); "
                "NeoMME pooled space needs this or the cosine kNN graph maps the shared cone not the content"}, indent=1))
    print(json.dumps({"centered": str(outdir), "mode": mode, "n": int(N), "cone_stats_removed": cone}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

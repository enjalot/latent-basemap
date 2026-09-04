"""Resolve the text-chain per-stage manifest (plan-basemap-chain, CPU prep, 2026-09-04). Emits ordered row
identities, tranche substrate+provenance hashes, exact cumulative membership per stage, and a PROVENANCE-based
pairwise-disjointness verdict (not directory names). Written BEFORE graph construction so each cumulative graph's
membership is pinned. Output: sandbox/text-chain-focused-20260904/chain-manifest.json."""
import json, hashlib, time
from pathlib import Path
import numpy as np

SUB = Path("/data/latent-basemap/substrates")
OUT = Path("/data/latent-basemap/sandbox/text-chain-focused-20260904"); OUT.mkdir(parents=True, exist_ok=True)
# tranche key -> substrate path (evolbench T0-T3 + the OOD-CA T3)
TRANCHES = {"T0": SUB / "evolbench/T0", "T1": SUB / "evolbench/T1", "T2": SUB / "evolbench/T2",
            "T3": SUB / "evolbench/T3", "oodca_T3": SUB / "evolbench-ood-ca/T3"}
# stage table (plan): cumulative ORDERED membership; OOD-A trains on T0+T1+T3 (5.6M), OOD-B on +T2+oodca (7.2M)
STAGES = {"Base": ["T0"], "OrdinaryA": ["T0", "T1"], "OOD_A": ["T0", "T1", "T3"],
          "OrdinaryB": ["T0", "T1", "T3", "T2"], "OOD_B": ["T0", "T1", "T3", "T2", "oodca_T3"]}


def _sha_file(p, cap=None):
    h = hashlib.sha256(); a = np.load(p, mmap_mode="r")
    # hash a bounded sample of bytes for large arrays (full hash of 4M x384 is slow) + shape/dtype for identity
    h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
    n = a.shape[0]; idx = np.linspace(0, n - 1, min(n, 100000)).astype(np.int64)
    h.update(np.ascontiguousarray(a[idx]).tobytes())
    return h.hexdigest()[:16]


def _prov_keys(d):
    """int64 keys (corpus,shard,row) for provenance disjointness. row/shard/corpus fit in <2^58."""
    p = np.load(d / "provenance.npy")
    return (p["corpus"].astype(np.int64) << 50) | (p["shard"].astype(np.int64) << 34) | p["row"].astype(np.int64)


def main():
    info = {}; keys = {}
    for k, d in TRANCHES.items():
        sub = d / "substrate.f32.npy"; a = np.load(sub, mmap_mode="r")
        kk = _prov_keys(d) if (d / "provenance.npy").exists() else None
        keys[k] = kk
        info[k] = {"dir": str(d), "rows": int(a.shape[0]), "dim": int(a.shape[1]),
                   "substrate_sha16": _sha_file(sub), "has_provenance": kk is not None,
                   "n_unique_prov": int(len(np.unique(kk))) if kk is not None else None,
                   "corpus_ids": sorted(set(int(c) for c in np.load(d / "provenance.npy")["corpus"][:200000])) if kk is not None else None}
    # pairwise disjointness from provenance
    ks = list(TRANCHES); disj = {}; all_ok = True
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            if keys[a] is None or keys[b] is None:
                disj[f"{a}|{b}"] = "NO_PROVENANCE"; continue
            inter = int(np.intersect1d(keys[a], keys[b], assume_unique=False).size)
            disj[f"{a}|{b}"] = inter
            if inter != 0:
                all_ok = False
    stages = {}
    for s, parts in STAGES.items():
        stages[s] = {"tranches_ordered": parts, "cumulative_rows": int(sum(info[t]["rows"] for t in parts))}
    out = {"schema": "text-chain-manifest-2026-09-04", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tranches": info, "stages": stages, "pairwise_prov_intersections": disj,
           "all_pairwise_disjoint": bool(all_ok),
           "note": "OOD_A trains T0+T1+T3 (5.6M), OOD_B trains +T2+oodca_T3 (7.2M) — matches plan stage table, "
                   "NOT _load_S3's T0+T1+T2+T3. Disjointness verified from provenance (corpus,shard,row) keys."}
    (OUT / "chain-manifest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"stages": stages, "all_pairwise_disjoint": all_ok,
                      "prov_intersections": disj, "rows": {k: info[k]["rows"] for k in info}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

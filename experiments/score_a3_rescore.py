"""Durable A3 recipe rescore (P0-F): the nomn-vs-mn4 recipe decision, recomputed
from a CHECKED-IN, config-pinned scorer instead of the machine-local
`/data/latent-basemap/track1/score_a3_rescore.py` with its "latest glob" run
selection.

All six run directories are pinned explicitly below (no globbing). FFR uses the
canonical `basemap.panel_v2` fixed-fraction formula over all anchors; purity uses
FineWeb-only anchors (the mixed-corpus protocol) via a per-metric mask. Every
run records coord/data/label/centroid/anchor hashes, the scorer hash, the repo
commit + dirty status, the exact formula version, and the output path/command.

Run:
  python experiments/score_a3_rescore.py --out experiments/evidence/a3_rescore.json
"""
from __future__ import annotations
import argparse, os, sys, json, time, hashlib, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (score_panel, PanelV2Config, load_embeddings,
                              sample_anchors, _ids_hash)
from basemap.round0005_retirement import refuse_retired_launcher

R = os.path.join(os.path.dirname(__file__), "results")
# PINNED run dirs — one per (recipe, seed). No globbing (P0-F).
RUNS = {
    ("nomn", 42): f"{R}/jina_en_8m_nested-fuzzyW_pr05_nomn_20260709_043029_b63ca1a8",
    ("nomn", 43): f"{R}/jina_en_8m_track1_a3_nomn_s43_20260711_144049_49e9d5a4",
    ("nomn", 44): f"{R}/jina_en_8m_track1_a3_nomn_s44_20260711_215833_1b948e64",
    ("mn4", 42):  f"{R}/jina_en_8m_nested-mn4_20260709_132310_8f48fd08",
    ("mn4", 43):  f"{R}/jina_en_8m_track1_a3_mn4_s43_20260711_174359_ba7ec10d",
    ("mn4", 44):  f"{R}/jina_en_8m_track1_a3_mn4_s44_20260712_010137_078f5ef0",
}
TRAIN8M = "/data/latent-basemap/jina-en-8M-nested/train"
NESTED_LABELS = "/data/latent-basemap/jina-en-8m/corpus_labels.npy"
C256_PATH = "/data/latent-basemap/track1/audit_centroids_k256_s0.npy"
C1024_PATH = "/data/latent-basemap/track1/centroids_fineweb_k1024.npy"


def _sha_file(p, cap=1 << 20):
    try:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read(cap))
        return h.hexdigest()[:16]
    except Exception:
        return None


def _git():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()[:12]
        d = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
        return c, d
    except Exception:
        return None, None


def main():
    refuse_retired_launcher("experiments/score_a3_rescore.py")
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/evidence/a3_rescore.json")
    ap.add_argument("--frac", type=float, default=0.001)
    ap.add_argument("--n-anchors", type=int, default=10000)
    args = ap.parse_args()
    # S1: GPU scorer must run under a held/inherited lease (controller or in-process).
    import torch as _t
    if _t.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()

    for (r, s), d in RUNS.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"pinned run dir missing: {r} s{s} -> {d}")

    cfg = PanelV2Config(frac=args.frac, n_anchors=args.n_anchors, k_clust=(256, 1024))
    X = load_embeddings(TRAIN8M, dim=768)
    n = len(X)
    labels = np.asarray(np.load(NESTED_LABELS, mmap_mode="r")[:n])   # corpus id per row
    C256 = np.load(C256_PATH); C1024 = np.load(C1024_PATH)
    centroids = {256: C256, 1024: C1024}

    aidx = sample_anchors(n, cfg)
    fineweb_mask = (labels[aidx] == 0)          # purity over FineWeb anchors only
    commit, dirty = _git()
    base_prov = {
        "scorer": "experiments/score_a3_rescore.py",
        "scorer_sha": _sha_file(os.path.abspath(__file__)),
        "train": TRAIN8M, "labels": NESTED_LABELS, "labels_sha": _sha_file(NESTED_LABELS),
        "centroid_sha": {"256": _sha_file(C256_PATH), "1024": _sha_file(C1024_PATH)},
        "anchor_hash": _ids_hash(aidx), "n_fineweb_anchors": int(fineweb_mask.sum()),
    }

    per_run = {}
    for (recipe, seed), rd in sorted(RUNS.items()):
        cpath = os.path.join(rd, "coords.parquet")
        df = pd.read_parquet(cpath)
        Z = df[["x", "y"]].values.astype("float32")
        res = score_panel(
            X, Z, config=cfg, centroids_by_k=centroids,
            anchor_masks={"ffr": None, "purity": fineweb_mask, "density": None},
            provenance={**base_prov, "run_dir": os.path.basename(rd),
                        "coords_sha": _sha_file(cpath)})
        per_run[f"{recipe}_s{seed}"] = {
            "recipe": recipe, "seed": seed, "ffr": res["ffr"], "recall@k": res["recall@k"],
            "purity_k256": res["purity"]["k256"], "purity_k1024": res["purity"]["k1024"],
            "density": res["density"], "k_frac": res["k_frac"],
            "coords_sha": _sha_file(cpath), "run_dir": os.path.basename(rd)}
        print(f"{recipe}_s{seed}: ffr={res['ffr']} purity1024={res['purity']['k1024']} "
              f"density={res['density']}", flush=True)

    means = {}
    for recipe in ("nomn", "mn4"):
        rows = [v for k, v in per_run.items() if v["recipe"] == recipe]
        means[recipe] = {m: round(float(np.mean([x[m] for x in rows])), 4)
                         for m in ("ffr", "purity_k256", "purity_k1024", "density")}
        means[recipe]["n_seeds"] = len(rows)
    decision = ("nomn" if means["nomn"]["ffr"] > means["mn4"]["ffr"]
                and means["nomn"]["purity_k1024"] > means["mn4"]["purity_k1024"] else "REVIEW")

    out = {"gate": "a3_recipe_rescore", "formula_version": cfg.formula_version,
           "metric": "fixed_fraction_recall", "n": n, "config": {"frac": cfg.frac,
           "n_anchors": cfg.n_anchors}, "provenance": {**base_prov, "code_commit": commit,
           "code_dirty": dirty, "generated_cmd": "python " + " ".join(sys.argv)},
           "per_run": per_run, "seed_means": means, "decision": decision}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n=== 3-seed means ===\n  nomn: {means['nomn']}\n  mn4:  {means['mn4']}")
    print(f"decision (purity>=ffr priority): {decision} -> {args.out}")


if __name__ == "__main__":
    main()

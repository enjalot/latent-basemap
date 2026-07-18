"""Complete-panel decision scorer for the R1 kernel comparison (plan §R1).

The kernel A/B (run_r1_kernel.py) reports transductive ffr/recall/density. Before
any kernel DECISION the plan requires the FULL canonical panel:
  - purity at k∈{256,1024} against FROZEN centroids (computed once per substrate);
  - held-out PROJECTION fidelity (queries provably outside the training rows) +
    a random floor;
  - a kNN-regressor OOS baseline — the NUMAP trigger: if the neural map does not
    clearly beat non-parametric kNN regression on held-out projection, we do NOT
    claim method superiority.

Everything routes through basemap.panel_v2 for the transductive metrics and reuses
its exact ffr formula for projection, so the numbers are comparable to the A/B.

Usage:
  python experiments/score_complete_panel.py \
     --runs legacy=<dir> umap=<dir> stdcurve=<dir> \
     --testbed /data/latent-basemap/jina-en-200k \
     --source /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train \
     --out /data/latent-basemap/r1_kernel/complete_panel.json
"""
from __future__ import annotations
import argparse, hashlib, os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (score_panel, PanelV2Config, load_embeddings, load_coords,
                              ffr_from_neighbors, recall_at_k_from_neighbors, _ids_hash,
                              cross_knn, sample_anchors, build_hiD_reference,
                              save_hiD_reference, load_hiD_reference, QueryTruthCache,
                              process_cuda_peak, reset_process_cuda_peak,
                              _torch_scoring_device)
from basemap.artifact_identity import expected_input_signature
from basemap.query_artifact import load_query_artifact, validate_convention
from basemap.output_safety import (atomic_save_new_npy, atomic_write_new_json,
                                   require_empty_directory)
from basemap.round0005_retirement import refuse_retired_launcher
from basemap.round0005_staging import MAP_EXPECTATIONS


def _sha_file(path):
    import hashlib
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def frozen_centroids(X, ks, cache_dir, seed=0, iters=25):
    """GPU k-means (random init + Lloyd) once per substrate; cached to disk so the
    purity labels are a frozen artifact, never silently regenerated."""
    # This is itself a device-selection boundary: direct library callers do not
    # get to reach CUDA merely by acquiring the public lease or exposing a GPU.
    torch, dev = _torch_scoring_device()
    out = {}
    n = len(X)
    for k in ks:
        path = os.path.join(cache_dir, f"centroids_k{k}.npy")
        if os.path.exists(path):
            if os.path.islink(path):
                raise RuntimeError(f"frozen centroid cache may not be a symlink: {path}")
            existing = np.load(path, mmap_mode="r")
            if (existing.shape != (k, X.shape[1]) or
                    existing.dtype != np.dtype("float32") or
                    not np.isfinite(existing).all()):
                raise RuntimeError(
                    f"existing frozen centroid cache is corrupt/mismatched and will not be "
                    f"overwritten: {path} shape={existing.shape} dtype={existing.dtype}")
            out[k] = existing
            continue
        rng = np.random.RandomState(seed)
        C = torch.from_numpy(np.asarray(X[np.sort(rng.choice(n, k, replace=False))],
                                        dtype=np.float32)).to(dev)
        for _ in range(iters):
            sums = torch.zeros_like(C); counts = torch.zeros(k, device=dev)
            for i in range(0, n, 100000):
                xb = torch.from_numpy(np.asarray(X[i:i + 100000], dtype=np.float32)).to(dev)
                lab = torch.cdist(xb, C).argmin(1)
                sums.index_add_(0, lab, xb); counts.index_add_(0, lab, torch.ones(len(xb), device=dev))
            nz = counts > 0
            C[nz] = sums[nz] / counts[nz, None]
        out[k] = C.cpu().numpy()
        atomic_save_new_npy(path, out[k], immutable=True)
    return out


def projection_ffr(X, Z, Xq, Zq, cfg, *, hi_truth=None):
    """Held-out FFR: hi-D query→corpus top-k_hit vs projected-query→map top-k_frac,
    via the canonical panel_v2.cross_knn + ffr formula (P0-4). Returns (ffr, r@k)."""
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))
    hi = (cross_knn(Xq, X, cfg.k_hit, cfg, hi_dim=True)
          if hi_truth is None else np.asarray(hi_truth)[:, :cfg.k_hit])
    lo = cross_knn(Zq, Z, kf, cfg, hi_dim=False)
    return (round(ffr_from_neighbors(hi, lo, cfg.k_hit), 4),
            round(recall_at_k_from_neighbors(hi, lo, cfg.k_hit), 5))


def knn_regress_coords(Xq, X, Z, cfg, k=15, *, hi_truth=None):
    """Non-parametric OOS map: each held-out query's 2D = mean of the map coords of
    its k nearest TRAIN rows in high-D. The baseline the neural map must beat."""
    nb = (cross_knn(Xq, X, k, cfg, hi_dim=True) if hi_truth is None
          else np.asarray(hi_truth)[:, :k])             # (nq, k) train-row ids
    return Z[nb].mean(axis=1)


def load_sample_indices(testbed, *, no_model):
    """Load the source-row map only when projection semantics require it.

    Coord-only/no-model maps are transductive and may legitimately describe an
    ordered corpus that has no source sampling artifact.
    """
    path = os.path.join(testbed, "sample_indices.npy")
    if os.path.exists(path):
        return np.load(path)
    if no_model:
        return None
    raise FileNotFoundError(f"{path} required for held-out projection (drop --no-model "
                            f"or provide sample_indices.npy)")


def align_coords_to_corpus(Z, z_ids, n: int, *, label: str):
    """Put every map in canonical corpus-row order before shared query truth use."""
    if z_ids is None:
        raise ValueError(f"map {label} has no semantic coordinate IDs; complete-panel "
                         "query truth cannot be aligned by row position")
    ids = np.asarray(z_ids, dtype=np.int64)
    order = np.argsort(ids, kind="mergesort")
    expected = np.arange(n, dtype=np.int64)
    if not np.array_equal(ids[order], expected):
        missing = np.setdiff1d(expected, ids, assume_unique=False)
        extra = np.setdiff1d(ids, expected, assume_unique=False)
        raise ValueError(f"map {label} semantic ID universe mismatch: "
                         f"missing={missing[:5].tolist()} extra={extra[:5].tolist()}")
    return np.asarray(Z[order], dtype=np.float32), expected, order


def parse_run_pairs(values: list[str]) -> dict[str, str]:
    """Parse labels without the silent duplicate collapse of ``dict(...)``."""
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run must be label=path: {value!r}")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError(f"run must contain a nonempty label and path: {value!r}")
        if label in runs:
            raise ValueError(f"duplicate map label: {label}")
        runs[label] = path
    return runs


def expected_query_consumers(labels, *, k_hit: int) -> list[dict]:
    suffixes = (
        ("parametric", k_hit),
        ("knn-regressor-coordinates", 15),
        ("knn-regressor-score", k_hit),
        ("random-floor", k_hit),
    )
    return [{"consumer": f"{label}:{suffix}", "k": int(k)}
            for label in sorted(labels) for suffix, k in suffixes]


def validate_query_truth_consumers(telemetry: dict, labels, *, k_hit: int,
                                   expected_builds: int = 1) -> dict:
    expected = expected_query_consumers(labels, k_hit=k_hit)
    observed = telemetry.get("consumers") if isinstance(telemetry, dict) else None
    checks = {
        "build_count": telemetry.get("build_count") == expected_builds,
        "maximum_k_15": telemetry.get("maximum_k") == 15,
        "consumer_count": telemetry.get("consumer_count") == len(expected),
        "consumer_multiset": isinstance(observed, list) and
            sorted(observed, key=lambda item: (item.get("consumer", ""), item.get("k", -1))) ==
            sorted(expected, key=lambda item: (item["consumer"], item["k"])),
        "consumer_names_unique": isinstance(observed, list) and
            len({item.get("consumer") for item in observed}) == len(observed),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "expected_consumers": expected, "observed_consumers": observed}


def persisted_scalars(run: dict) -> dict:
    return {key: run.get(key) for key in (
        "ffr", "recall@k", "purity_k256", "purity_k1024", "density",
        "proj_ffr", "proj_recall@k", "proj_knn_regressor_ffr",
        "proj_random_floor_ffr", "proj_beats_knn", "proj_margin_over_knn")}


def score_query_bundle(*, X, Z, Xq, Zq, cfg: PanelV2Config,
                       truth_cache: QueryTruthCache, label: str, random_seed: int,
                       phase_delay_s: float = 0.0) -> dict:
    """Run the real held-out scorer/baselines from one max-k truth build.

    ``phase_delay_s`` exists only for the synthetic admission regression: it
    sleeps in each real scoring phase, so the launcher observes actual wall
    time instead of multiplying a report field.
    """
    if phase_delay_s < 0:
        raise ValueError("phase_delay_s must be nonnegative")

    def delayed():
        if phase_delay_s:
            time.sleep(phase_delay_s)

    delayed()
    hi_neural = truth_cache.use(f"{label}:parametric", k=cfg.k_hit)
    proj_ffr, proj_rk = projection_ffr(X, Z, Xq, Zq, cfg, hi_truth=hi_neural)
    delayed()
    hi_knn = truth_cache.use(f"{label}:knn-regressor-coordinates", k=15)
    Zq_knn = knn_regress_coords(Xq, X, Z, cfg, hi_truth=hi_knn)
    knn_ffr, _ = projection_ffr(
        X, Z, Xq, Zq_knn, cfg,
        hi_truth=truth_cache.use(f"{label}:knn-regressor-score", k=cfg.k_hit))
    delayed()
    lo, hi = Z.min(0), Z.max(0)
    label_seed = int.from_bytes(hashlib.sha256(
        f"{random_seed}:{label}".encode("utf-8")).digest()[:4], "big")
    label_rng = np.random.RandomState(label_seed)
    Zq_rand = (label_rng.rand(len(Xq), Z.shape[1]).astype(np.float32)
               * (hi - lo) + lo)
    floor_ffr, _ = projection_ffr(
        X, Z, Xq, Zq_rand, cfg,
        hi_truth=truth_cache.use(f"{label}:random-floor", k=cfg.k_hit))
    return {
        "proj_ffr": proj_ffr,
        "proj_recall@k": proj_rk,
        "proj_knn_regressor_ffr": knn_ffr,
        "proj_random_floor_ffr": floor_ffr,
        "proj_beats_knn": bool(proj_ffr > knn_ffr),
        "proj_margin_over_knn": round(proj_ffr - knn_ffr, 4),
    }


def main():
    main_started = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label=run_dir pairs (or label=coords.parquet with --no-model)")
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--no-model", action="store_true",
                    help="coord-only references (cuML/umap-learn): transductive panel only, "
                         "no projection/kNN-regressor (run values are coords.parquet paths)")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--frac", type=float, default=0.001)
    ap.add_argument("--n-anchors", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--query-artifact",
                    help="required held-out query manifest for model scoring")
    ap.add_argument("--query-expectation",
                    help="required JSON convention expected by this scoring cell")
    ap.add_argument("--query-cache-mode", choices=("on", "off"), default="on")
    ap.add_argument("--require-round0005-nine-maps", action="store_true",
                    help="require the exact registered nine labels and 36 truth consumers")
    ap.add_argument("--expected-highd-builds", type=int)
    ap.add_argument("--wall-max", type=float)
    ap.add_argument("--peak-max", type=float)
    ap.add_argument("--out-root", required=True,
                    help="pre-created empty private parent; fixed sibling outputs are created within it")
    args = ap.parse_args()
    out_root = require_empty_directory(args.out_root, label="complete panel private output parent")
    if not out_root.startswith("/data/"):
        raise ValueError("complete panel output parent must be under /data")
    report_path = os.path.join(out_root, "report.json")
    reference_path = os.path.join(out_root, "hiD-reference.npz")
    reference_receipt_path = os.path.join(out_root, "hiD-reference-receipt.json")
    query_cache_dir = os.path.join(out_root, "query-truth-cache")

    from basemap.pumap.parametric_umap.core import ParametricUMAP
    # cap corpus_chunk so query-tile × corpus-chunk cross matrices stay bounded
    # (4096 × 500k ≈ 2 GB fp32) even when the corpus is 2M+.
    cfg = PanelV2Config(frac=args.frac, n_anchors=args.n_anchors, corpus_chunk=500_000)
    runs = parse_run_pairs(args.runs)
    if args.require_round0005_nine_maps and set(runs) != set(MAP_EXPECTATIONS):
        missing = sorted(set(MAP_EXPECTATIONS) - set(runs))
        extra = sorted(set(runs) - set(MAP_EXPECTATIONS))
        raise ValueError(f"Round 0005 scorer needs exact nine-map labels: "
                         f"missing={missing} extra={extra}")

    X = load_embeddings(os.path.join(args.testbed, "train"), dim=args.dim)
    if len(X) >= 8_000_000:
        refuse_retired_launcher("experiments/score_complete_panel.py")

    # The actual reopened row count is known before the first CUDA API call.
    # Scale scoring uses score_8m_bridge.py, whose release/certificate inputs
    # are checked before CUDA. This generic path remains subscale.
    _torch, _scoring_device = _torch_scoring_device()
    reset_process_cuda_peak(_torch.cuda)
    # sample_indices maps testbed rows back to the source corpus; required for the
    # held-out projection (model path), optional for a --no-model transductive score
    # (e.g. the 4M-nested corpus has none).
    si = load_sample_indices(args.testbed, no_model=args.no_model)
    missing = [os.path.join(args.testbed, f"centroids_k{k}.npy")
               for k in (256, 1024)
               if not os.path.isfile(os.path.join(args.testbed, f"centroids_k{k}.npy"))]
    if missing:
        raise FileNotFoundError(f"Round 0005 frozen centroids must already exist: {missing}")
    centroids = frozen_centroids(X, (256, 1024), args.testbed)

    # The scorer owns its reference.  It is never a pre-gate input: build once in
    # this node's private fresh output set, publish atomically, then strictly reload
    # before any map consumes it.  The receipt binds both the source identity key
    # and every computed reference payload.
    aidx0 = sample_anchors(len(X), cfg)
    built_ref = build_hiD_reference(X, aidx0, cfg, centroids)
    save_hiD_reference(built_ref, reference_path)
    ref = load_hiD_reference(
        reference_path, expected_key=built_ref["key"],
        expected_key_parts=built_ref["key_parts"])
    reference_receipt = {
        "schema": "round0005_private_hiD_reference_receipt.v1",
        "reference": expected_input_signature(reference_path),
        "identity_key": ref["key"],
        "content_sha256": ref["content_sha256"],
        "key_parts": ref["key_parts"],
        "built_and_reloaded_in_same_scorer": True,
        "pre_gate_reference_consumed": False,
    }
    atomic_write_new_json(reference_receipt_path, reference_receipt, immutable=True)
    print(f"[panel] private reference key={ref['key']} content={ref['content_sha256']}",
          flush=True)

    # Held-out vectors now come only from an explicit artifact.  There is no raw
    # source fallback: model/prompt/pooling/dtype/normalization and query selection
    # must match a separately content-bound expectation before any map is loaded.
    held = np.array([], dtype=np.int64); Xq = None; query_artifact = None
    truth_cache = None
    if not args.no_model:
        if not args.query_artifact or not args.query_expectation:
            raise ValueError("model scoring requires --query-artifact and --query-expectation")
        with open(args.query_expectation, encoding="utf-8") as handle:
            expected_convention = validate_convention(json.load(handle))
        query_artifact = load_query_artifact(
            args.query_artifact, testbed=args.testbed,
            expected_convention=expected_convention)
        held = query_artifact["query_ids"]
        Xq = query_artifact["Xq"]
        truth_cache = QueryTruthCache(
            cache_dir=(query_cache_dir if args.query_cache_mode == "on" else None),
            enabled=args.query_cache_mode == "on")
        truth_cache.get_or_build(
            Xq, X, cfg=cfg,
            corpus_identity=query_artifact["manifest"]["corpus"],
            query_identity={
                "artifact_identity_sha256": query_artifact["identity_sha256"],
                "ordered_query_ids_sha256": query_artifact["manifest"]["ordered_ids_sha256"],
                "ordered_query_embeddings_sha256":
                    query_artifact["manifest"]["ordered_embeddings_sha256"],
            },
            k=max(15, cfg.k_hit))
    query_truth_peak = process_cuda_peak(_torch.cuda)

    import subprocess
    try:
        _commit = subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                           "rev-parse", "HEAD"], text=True).strip()[:12]
        # scorer_dirty = TRACKED source modifications only. An untracked output
        # (e.g. a sibling rescore's just-written evidence file) is not a scorer-code
        # change and must not dirty a later scorer in the same batch (--untracked-files=no).
        _dirty = bool(subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except Exception:
        _commit = _dirty = None
    summary = {"testbed": args.testbed, "n": int(len(X)), "n_holdout": int(len(held)),
               "n_holdout_unique": int(len(np.unique(held))),
               "held_disjoint_from_train": True, "held_hash": _ids_hash(held),
               "query_artifact": (None if query_artifact is None else {
                   "manifest_path": query_artifact["manifest_path"],
                   "manifest_sha256": query_artifact["manifest_sha256"],
                   "identity_sha256": query_artifact["identity_sha256"],
                   "convention": query_artifact["manifest"]["convention"],
                   "query_selection": query_artifact["manifest"]["query_selection"],
               }),
               "sample_indices_hash": (_ids_hash(np.asarray(si, np.int64)) if si is not None else None),
               "frac": cfg.frac, "n_anchors": cfg.n_anchors, "seed": args.seed,
               "scorer_commit": _commit, "scorer_dirty": _dirty,
               "formula_version": cfg.formula_version,
               "private_outputs": {
                   "report": report_path,
                   "hiD_reference": reference_path,
                   "hiD_reference_receipt": reference_receipt_path,
                   "query_truth_cache": (query_cache_dir if args.query_cache_mode == "on" else None),
               },
               "hiD_reference_path": reference_path,
               "hiD_reference_key": ref["key"],
               "hiD_reference_content_sha256": ref["content_sha256"],
               "hiD_reference_receipt": expected_input_signature(reference_receipt_path),
               "pre_gate_reference_consumed": False,
               "runs": {},
               "query_truth_peak": query_truth_peak,
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for label, rd in runs.items():
        t0 = time.time()
        coords_path = rd if args.no_model else os.path.join(rd, "coords.parquet")
        Z, z_ids = load_coords(coords_path)
        if not args.no_model:
            Z, z_ids, gathered_rows = align_coords_to_corpus(Z, z_ids, len(X), label=label)
        else:
            gathered_rows = None
        # pass z_ids through — the exact-alignment contract must hold for the
        # decision scorer too (P0-4), not just the transductive panel.
        panel = score_panel(X, Z, config=cfg, z_ids=z_ids, centroids_by_k=centroids,
                            hiD_reference=ref,
                            provenance={"scorer": "complete_panel", "run": os.path.basename(rd),
                                        "coords_sha": _sha_file(coords_path),
                                        "no_model_reference": bool(args.no_model)})
        if not panel["provenance"].get("hiD_reference_reused"):
            raise ValueError(f"map {label} did not reuse the shared reference "
                             f"(key drift?) — refuse to certify (L0.4).")
        if args.no_model:
            # coord-only reference (cuML/umap-learn): transductive metrics only —
            # no model to project held-out queries, so proj/kNN-regressor are N/A.
            proj_ffr = proj_rk = knn_ffr = floor_ffr = None
        else:
            Xa = X
            model = ParametricUMAP.load(
                os.path.join(rd, "model.pt"),
                # Revalidate the genuine child capability at this later device
                # boundary too; a gate may have changed since panel startup.
                device=_torch_scoring_device()[1])
            Zq = np.asarray(model.transform(Xq), dtype=np.float32)
            query_scores = score_query_bundle(
                X=Xa, Z=Z, Xq=Xq, Zq=Zq, cfg=cfg, truth_cache=truth_cache,
                label=label, random_seed=args.seed)
            proj_ffr = query_scores["proj_ffr"]
            proj_rk = query_scores["proj_recall@k"]
            knn_ffr = query_scores["proj_knn_regressor_ffr"]
            floor_ffr = query_scores["proj_random_floor_ffr"]
        summary["runs"][label] = {
            "run_dir": os.path.basename(rd), "wall_s": round(time.time() - t0, 1),
            "no_model_reference": bool(args.no_model),
            "ffr": panel["ffr"], "recall@k": panel["recall@k"],
            "purity_k256": (panel.get("purity") or {}).get("k256"),
            "purity_k1024": (panel.get("purity") or {}).get("k1024"),
            "density": panel["density"],
            "proj_ffr": proj_ffr, "proj_recall@k": proj_rk,
            "proj_knn_regressor_ffr": knn_ffr, "proj_random_floor_ffr": floor_ffr,
            "proj_beats_knn": (None if args.no_model else query_scores["proj_beats_knn"]),
            "proj_margin_over_knn": (None if args.no_model else
                                     query_scores["proj_margin_over_knn"]),
            "coordinate_alignment": (None if gathered_rows is None else {
                "policy": "semantic IDs gathered into canonical corpus order",
                "gathered_rows_sha256": hashlib.sha256(
                    np.ascontiguousarray(gathered_rows).tobytes()).hexdigest(),
            }),
            # L0.4: shared-reference provenance per map.
            "hiD_reference_key": panel["provenance"].get("hiD_reference_key"),
            "hiD_reference_reused": panel["provenance"].get("hiD_reference_reused"),
            # P0-4: retain the full panel audit trail, not just scalar leaves.
            "panel_full": panel}
        r = summary["runs"][label]
        print(f"[panel] {label:10s} ffr={r['ffr']} purity1024={r['purity_k1024']} dens={r['density']} "
              f"| proj={r['proj_ffr']} knnReg={r['proj_knn_regressor_ffr']} floor={r['proj_random_floor_ffr']} "
              f"beats_knn={r['proj_beats_knn']}", flush=True)

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["total_wall_s"] = round(time.time() - main_started, 3)
    if truth_cache is not None:
        summary["query_truth_cache"] = truth_cache.telemetry()
        summary["query_truth_contract"] = validate_query_truth_consumers(
            summary["query_truth_cache"], runs, k_hit=cfg.k_hit,
            expected_builds=(args.expected_highd_builds
                             if args.expected_highd_builds is not None else 1))
    cuda_peak = process_cuda_peak(_torch.cuda)
    summary["process_cuda_peak"] = cuda_peak
    summary["peak_gpu_allocated_gb"] = cuda_peak["allocated_gib"]
    summary["peak_gpu_reserved_gb"] = cuda_peak["reserved_gib"]
    summary["peak_gpu_gb"] = cuda_peak["maximum_gib"]
    expected_builds = (args.expected_highd_builds
                       if args.expected_highd_builds is not None else 1)
    checks = {
        "highd_build_count": (args.no_model or
                              summary.get("query_truth_cache", {}).get("build_count") ==
                              expected_builds),
        "exact_query_consumers": (args.no_model or
                                  summary.get("query_truth_contract", {}).get("passed") is True),
        "exact_round0005_map_labels": (not args.require_round0005_nine_maps or
                                       set(summary["runs"]) == set(MAP_EXPECTATIONS)),
        "wall": args.wall_max is None or summary["total_wall_s"] <= args.wall_max,
        "peak_gpu": (args.peak_max is None or
                     (summary["peak_gpu_gb"] is not None and
                      summary["peak_gpu_gb"] <= args.peak_max)),
    }
    summary["performance_gate"] = {
        "passed": all(checks.values()), "checks": checks,
        "wall_max_s": args.wall_max, "peak_gpu_max_gb": args.peak_max,
        "expected_highd_builds": expected_builds,
    }
    atomic_write_new_json(report_path, summary, immutable=True, indent=1)
    print(f"[panel] -> {report_path}", flush=True)
    if not summary["performance_gate"]["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

"""P4 — score the retrained WEIGHTED 8M bridge maps under panel v2.2 and bind the
ACTUAL training pipeline/semantics into the evidence (closure item 4). Scoring
runs in this fresh process (one GPU phase per process); it reads each seed's
training results.json for the stamped pipeline + positive-LR update count, then
scores the coords transductively (ffr/purity/density, FineWeb purity mask).

Usage (under a held lease):
  python experiments/score_8m_bridge.py --performance-gate <round0005-gate.json> \
      --release-sha <40-hex-release> \
      --out-root <fresh-private-parent>
"""
from __future__ import annotations
import argparse, os, sys, json, glob, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.panel_v2 import (score_panel, PanelV2Config, load_embeddings, load_coords,
                              sample_anchors, FORMULA_VERSION, build_hiD_reference,
                              save_hiD_reference, load_hiD_reference,
                              process_cuda_peak, reset_process_cuda_peak)
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, require_empty_directory
from experiments.round0005_performance_gate import (derive_scale_rows,
                                                     require_current_release_sha,
                                                     require_scale_performance_gate)
from experiments.score_complete_panel import parse_run_pairs

TRAIN8M = "/data/latent-basemap/jina-en-8M-nested/train"
LABELS8M = "/data/latent-basemap/jina-en-8m/corpus_labels.npy"
C256 = "/data/latent-basemap/track1/audit_centroids_k256_s0.npy"
C1024 = "/data/latent-basemap/track1/centroids_fineweb_k1024.npy"


def main():
    from basemap.round0005_retirement import refuse_retired_launcher
    refuse_retired_launcher("experiments/score_8m_bridge.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchors", type=int, default=2000)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="optional label=run_dir pairs (e.g. legacy=<dir> stdcurve=<dir>); "
                         "default = the two weighted bridge seeds")
    ap.add_argument("--performance-gate", required=True,
                    help="accepted Round 0005 content-bound scale certificate")
    ap.add_argument("--release-sha", required=True,
                    help="exact queue release bound by the scale certificate")
    ap.add_argument("--out-root", required=True,
                    help="pre-created empty private scorer output parent")
    args = ap.parse_args()
    require_current_release_sha(args.release_sha)
    out_root = require_empty_directory(args.out_root, label="8M scorer private output parent")
    if not out_root.startswith("/data/"):
        raise ValueError("8M scorer output parent must be under /data")
    report_path = os.path.join(out_root, "report.json")
    reference_path = os.path.join(out_root, "hiD-reference.npz")
    reference_receipt_path = os.path.join(out_root, "hiD-reference-receipt.json")
    X = load_embeddings(TRAIN8M, dim=768); n = len(X)
    row_derivation = derive_scale_rows(TRAIN8M, dimensions=768, loaded_matrix=X)
    if n != 8_000_000:
        raise ValueError(f"8M scorer input derived {n:,} rows, expected 8,000,000")
    certificate = require_scale_performance_gate(
        args.performance_gate, scientific_rows=n, row_derivation=row_derivation,
        release_sha=args.release_sha)
    # Gate and row derivation both pass before any CUDA allocation or lease work.
    import torch as _t
    if _t.cuda.is_available():
        from basemap.run_controller import require_active_lease
        require_active_lease()
    reset_process_cuda_peak(_t.cuda)
    labels = np.asarray(np.load(LABELS8M, mmap_mode="r")[:n])
    cents = {256: np.load(C256), 1024: np.load(C1024)}
    cfg = PanelV2Config(frac=0.001, n_anchors=args.n_anchors, corpus_chunk=500_000, k_clust=(256, 1024))
    aidx = sample_anchors(n, cfg); fw = (labels[aidx] == 0)
    # One private reference is produced and strictly reloaded by this scorer.
    built_ref = build_hiD_reference(X, aidx, cfg, cents)
    save_hiD_reference(built_ref, reference_path)
    ref = load_hiD_reference(
        reference_path, expected_key=built_ref["key"],
        expected_key_parts=built_ref["key_parts"])
    reference_receipt = {
        "schema": "round0005_private_hiD_reference_receipt.v1",
        "reference": expected_input_signature(reference_path),
        "identity_key": ref["key"], "content_sha256": ref["content_sha256"],
        "key_parts": ref["key_parts"], "pre_gate_reference_consumed": False,
        "built_and_reloaded_in_same_scorer": True,
    }
    atomic_write_new_json(reference_receipt_path, reference_receipt, immutable=True)
    print(f"[bridge] private reference key={ref['key']}", flush=True)
    out = {"substrate": "mixed-8M (fineweb+rpj+pile)", "recipe": "legacy nomn+weighted",
           "budget": "500k updates v3", "n_anchors": args.n_anchors,
           "formula_version": FORMULA_VERSION,
           "hiD_reference_key": ref["key"],
           "hiD_reference_content_sha256": ref["content_sha256"],
           "hiD_reference_path": reference_path,
           "hiD_reference_receipt": expected_input_signature(reference_receipt_path),
           "pre_gate_reference_consumed": False,
           "scale_row_derivation": row_derivation,
           "release_sha": args.release_sha,
           "performance_gate": expected_input_signature(args.performance_gate),
           "seeds": {}}
    # default: the two weighted bridge seeds; optional --runs label=dir pairs (G1 pair).
    run_map = (parse_run_pairs(args.runs) if args.runs
               else {f"s{s}": None for s in (42, 43)})
    for label, rd_arg in run_map.items():
        seed = label
        if rd_arg is None:
            sd = label[1:] if label.startswith("s") else label
            rds = sorted(glob.glob(f"experiments/results/r1_8m_bridge_weighted_s{sd}_*"))
            if not rds:
                out["seeds"][label] = {"error": "no run dir"}; continue
            rd = rds[-1]
        else:
            rd = rd_arg
        tr = json.load(open(os.path.join(rd, "results.json")))
        acct = (tr.get("run_manifest") or {}).get("train_accounting", {})
        Z, zid = load_coords(os.path.join(rd, "coords.parquet"))
        t0 = time.time()
        p = score_panel(X, Z, config=cfg, z_ids=zid,
                        centroids_by_k=cents, hiD_reference=ref,
                        anchor_masks={"ffr": None, "purity": fw, "density": None},
                        scale_admission={
                            "performance_gate": args.performance_gate,
                            "release_sha": args.release_sha,
                            "row_derivation": row_derivation,
                            "scale_policy": None,
                        },
                        provenance={
                            "gate": "weighted_8m_bridge",
                            "scale_certificate_identity_sha256":
                                certificate["identity_sha256"],
                            "run": os.path.basename(rd),
                        })
        if not p["provenance"].get("hiD_reference_reused"):
            raise ValueError(f"map {label} did not reuse the shared 8M reference (L0.4).")
        out["seeds"][label] = {
            "run_dir": os.path.basename(rd), "score_wall_s": round(time.time() - t0, 1),
            # ACTUAL execution pipeline/semantics (P1 stamp) — the review's core requirement
            "pipeline": acct.get("pipeline_pipeline"),
            "positive_sampling": acct.get("pipeline_positive_sampling"),
            "sampler_class": acct.get("pipeline_sampler_class"),
            "x_residency": acct.get("pipeline_x_residency"),
            "positive_lr_updates": acct.get("positive_lr_optimizer_steps"),
            "budget_satisfied": acct.get("budget_satisfied"),
            "schedule_version": acct.get("schedule_version"),
            "updates_per_s": acct.get("updates_per_s"),
            "formula_version": p["formula_version"],
            "hiD_reference_key": p["provenance"].get("hiD_reference_key"),
            "hiD_reference_reused": p["provenance"].get("hiD_reference_reused"),
            "ffr": p["ffr"], "recall@k": p["recall@k"], "purity": p.get("purity"), "density": p["density"],
            "exactness": p["provenance"].get("exactness"),
        }
        s = out["seeds"][label]
        print(f"[bridge] {label}: pipeline={s['pipeline']}/{s['positive_sampling']} "
              f"updates={s['positive_lr_updates']} | ffr={s['ffr']} purity={s['purity']} "
              f"density={s['density']} ({s['updates_per_s']} upd/s)", flush=True)
    out["process_cuda_peak"] = process_cuda_peak(_t.cuda)
    out["peak_gpu_gb"] = out["process_cuda_peak"]["maximum_gib"]
    atomic_write_new_json(report_path, out, immutable=True, indent=1)
    print(f"[bridge] -> {report_path}", flush=True)


if __name__ == "__main__":
    main()

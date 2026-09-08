"""C1 production-path gradient check (follow-up review 2026-09-08). Verifies the endpoint-reuse wiring in the
PRODUCTION path (DeviceEdgeSampler.endpoint_reuse + core.py reuse branch) is gradient-neutral: two short fits on
the SAME data/seed, fp32 (use_amp off, deterministic), device_int8 + rank-window negatives + the production
optimizer — one with ENDPOINT_REUSE=0, one with =1 — must produce matching model weights. exp_c_verify already
proved the unique-endpoint math vs the production LOSS (6.66e-16); this proves the production IMPLEMENTATION
(loader unique/gather + core scatter) reproduces the direct forward. PASS if max|Δweight| is at fp32 round-off.
Usage: exp_c1_gradcheck.py [STEPS=25]
"""
import os, sys, json
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
X_PATH = "/data2/monet/eval-common/train_hd.f16.npy"; EDGES = SB / "eval-common-train" / "edges-k15-fuzzy.npz"
A, B_ = 1.9328, 0.7905


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    X = np.asarray(np.load(X_PATH, mmap_mode="r"), np.float32)              # 500K x1536 (fit quantizes to int8)

    def run(reuse):
        os.environ["ENDPOINT_REUSE"] = str(reuse); os.environ["GROUPED_NEGATIVES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        pu = ParametricUMAP(n_components=2, hidden_dim=2048, n_layers=3, architecture="residual_bottleneck",
                            low_dim_kernel="umap", a=A, b=B_, use_amp=False, device="cuda",
                            fneg_weight=1.0, neg_tanh_gamma=4.0, pos_ratio=0.10, rankneg_window=125_000,
                            correlation_weight=0.0, batch_size=16384, x_residency="device_int8",
                            n_epochs=1, total_steps_estimate=steps)
        pu._max_train_steps = steps                                        # bench hook (attr, no model needed); fit() inits the model after edge-list admission
        pu.fit(X, precomputed_edges_path=str(EDGES), random_state=42)
        return {k: v.detach().float().cpu().clone() for k, v in pu.model.state_dict().items()}

    wA = run(0); torch.cuda.empty_cache(); wB = run(1)
    maxd = max((wA[k] - wB[k]).abs().max().item() for k in wA)
    meand = float(np.mean([(wA[k] - wB[k]).abs().mean().item() for k in wA]))
    out = {"schema": "exp-c1-gradcheck-2026-09-08", "steps": steps, "precision": "fp32 (use_amp off)",
           "setup": "device_int8 + rank-window negs + production optimizer, eval-common-train 500K + its full-D graph",
           "max_abs_weight_diff": maxd, "mean_abs_weight_diff": meand, "tol": 1e-4,
           "PASS": bool(maxd < 1e-4),
           "note": "reuse-off vs reuse-on weights after %d identical fp32 steps; gradient-neutral endpoint reuse -> "
                   "fp32 round-off. exp_c_verify proved the math (6.66e-16); this proves the production wiring." % steps}
    (SB / "exp-c1-gradcheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

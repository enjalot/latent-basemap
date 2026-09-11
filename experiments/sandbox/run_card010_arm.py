"""Card010 EXPLORATORY arm trainer — from-scratch matched parametric-UMAP on the 300K DINO draw.
Per OC/card010-exploratory-release.json + card010-floor-ruling.md: the three heads (fixed15 /
adaptive[5,60] / fixed12) differ ONLY in the precomputed edge set. Everything else is the validated
DINO champion recipe, taken VERBATIM by loading the champion head's config, then:
  - re-initialized to a FRESH shared random init (NOT the champion's trained weights) so all arms
    start from bitwise-identical weights (shared init sha asserted across arms) — matched init;
  - rankneg_window = 75000 (25% of N=300K) — reproduces the champion's (2*W/N)^0.75 repulsion regime
    exactly (2M champion used W=500000 -> (0.5)^0.75); matched across all arms;
  - _max_train_steps = STEPS (the preregistered common dose), batch 16384, LR 0.001, seed 42.
No replay/deriv/anchor hooks. Fail-closed on edge/substrate/init identity. Writes admission BEFORE any
optimizer step. Usage: run_card010_arm.py <TAG> <STEPS>   (EDGES via env EVOLBENCH_LAMBDA_EDGES)
"""
import os, sys, time, json, hashlib
from pathlib import Path
import numpy as np

CHAMPION = Path("/data/latent-basemap/sandbox/dino-arrival-t0/champion-bs16k/model.pt")
SUB = Path("/data/latent-basemap/substrates/card010-adaptive/substrate.f16.npy")
OUTD = Path(os.environ.get("EVOLBENCH_LAMBDA_OUTD", "/data/latent-basemap/sandbox/card010-train"))
EDGES = Path(os.environ["EVOLBENCH_LAMBDA_EDGES"])
INIT_PATH = Path(os.environ.get("CARD010_INIT", "/data/latent-basemap/substrates/card010-adaptive/init-card010.pt"))
INIT_SHA_FILE = INIT_PATH.with_suffix(".sha")
RELEASE = Path("/data/latent-basemap/sandbox/overseer-codex/card010-exploratory-release.json")
EDGES_MANIFEST = Path("/data/latent-basemap/sandbox/overseer-codex/card010-edges-manifest.json")
SEED = int(os.environ.get("EVOLBENCH_LAMBDA_SEED", "42"))
RANKNEG_WINDOW = 75000  # 25% of N=300K -> matches champion (2*W/N)^0.75 repulsion regime


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def ensure_shared_init(ParametricUMAP, torch):
    """Build the FRESH shared random init once (seeded, deterministic), reuse across arms.
    Config comes from the champion head; weights are re-initialized fresh (not the trained head)."""
    if INIT_PATH.exists():
        return json.loads(INIT_SHA_FILE.read_text())
    p = ParametricUMAP.load(str(CHAMPION), device="cpu")
    p.model = None
    torch.manual_seed(SEED); np.random.seed(SEED)
    p._init_model(1536)
    sd = {k: v.detach().cpu().clone() for k, v in p.model.state_dict().items()}
    torch.save({"model_state": sd, "init_state_sha256": p.init_state_sha256}, INIT_PATH)
    rec = {"init_state_sha256": p.init_state_sha256, "seed": SEED, "arch": p.architecture,
           "hidden_dim": p.hidden_dim, "n_layers": p.n_layers, "neck_fraction": p.neck_fraction, "input_dim": 1536}
    INIT_SHA_FILE.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    tag = sys.argv[1]; steps = int(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    # ── fail-closed identity: edges + substrate must match the frozen manifests/release ──
    em = json.loads(EDGES_MANIFEST.read_text()); rel = json.loads(RELEASE.read_text())
    arm_key = {"fixed15": "fixed15", "adaptive": "adaptive", "fixed_mean": "fixed_mean"}[tag]
    assert _sha(EDGES) == em["edges"][arm_key]["sha256"], f"{tag}: EDGES sha != frozen manifest"
    assert _sha(SUB) == rel["input_hashes"]["substrate.f16.npy"]["sha256"], "substrate sha != release"
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()

    init_rec = ensure_shared_init(ParametricUMAP, torch)
    init_obj = torch.load(str(INIT_PATH), map_location="cpu", weights_only=False)
    init_state = init_obj["model_state"]

    pumap = ParametricUMAP.load(str(CHAMPION), device="cuda")   # champion config VERBATIM
    pumap.model = None                                          # discard trained weights -> fresh init below
    pumap.rankneg_window = RANKNEG_WINDOW
    pumap.learning_rate = 0.001; pumap.batch_size = 16384; pumap.warmup_steps = 0
    # The frozen card specifies constant LR. Plateau can reduce only one graph
    # arm's rate because edge count/loss history changes its epoch boundaries.
    pumap.lr_schedule = "constant"
    pumap.n_epochs = 10000; pumap._max_train_steps = steps
    # ensure NO hooks
    for attr, val in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0),
                      ("replay_bank_path", ""), ("replay_weight", 0.0),
                      ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, attr):
            setattr(pumap, attr, val)

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32)     # already unit-normalized (PRENORMED)
    n = X.shape[0]
    cfg = {k: getattr(pumap, k, None) for k in ("architecture", "hidden_dim", "n_layers", "neck_fraction",
           "n_components", "learning_rate", "a", "b", "low_dim_kernel", "kernel_alpha", "clip_grad_norm",
           "fneg_lo", "fneg_hi", "fneg_weight", "neg_tanh_gamma", "pos_ratio", "positive_target_mode",
           "rankneg_window", "rankneg_exclude_neighbors", "density_weight", "x_residency", "batch_size", "lr_schedule")}
    admission = {"schema": "card010-exploratory-admission-2026-09-11", "tag": tag, "written_before_steps": True,
                 "exploratory": True, "original_prereg_viability": False,
                 "ruling_sha256": rel.get("ruling_sha256"), "release_arms": rel.get("arms"),
                 "edges": str(EDGES), "edges_sha256": _sha(EDGES), "edges_mean_k": em["edges"][arm_key]["mean_k"],
                 "substrate_sha256": rel["input_hashes"]["substrate.f16.npy"]["sha256"],
                 "shared_init_sha256": init_rec["init_state_sha256"], "seed": SEED,
                 "max_train_steps": steps, "n_rows": int(n), "rankneg_window": RANKNEG_WINDOW,
                 "rankneg_window_note": "25% of N=300K; reproduces champion (2W/N)^0.75 repulsion regime",
                 "config": cfg, "prenormed": True}
    (OUTD / f"admission-{tag}.json").write_text(json.dumps(admission, indent=1))

    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False, warm_start_state=init_state)
    wall = time.time() - t0
    warm_sha = getattr(pumap, "warm_start_sha256", None)
    assert warm_sha == init_rec["init_state_sha256"], f"warm-start sha {warm_sha} != shared init {init_rec['init_state_sha256']}"
    trained_sha = _state_sha(pumap.model)
    exec_steps = int(pumap._train_stats.get("executed_iters", 0)) if hasattr(pumap, "_train_stats") else 0
    its = round(exec_steps / wall, 2) if wall > 0 else None
    peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32)
    np.save(OUTD / f"coords-{tag}.npy", coords); pumap.save(str(OUTD / f"model-{tag}.pt"))
    ts = dict(getattr(pumap, "_train_stats", {}) or {})
    assert ts["positive_lr_optimizer_steps"] == steps
    assert ts["lr_used_count"] == steps
    assert ts["lr_used_min"] == ts["lr_used_max"] == 0.001
    assert np.isfinite(coords).all()
    r = np.linalg.norm(coords.astype(np.float64) - np.median(coords, 0), axis=1)
    man = {"schema": "card010-exploratory-arm-2026-09-11", "tag": tag, "n": int(n), "exploratory": True,
           "edges": str(EDGES), "edges_sha256": admission["edges_sha256"], "edges_mean_k": admission["edges_mean_k"],
           "shared_init_sha256": init_rec["init_state_sha256"], "warm_start_sha256": warm_sha,
           "trained_sha256": trained_sha, "warm_start_changed": bool(warm_sha != trained_sha),
           "max_train_steps": steps, "executed_steps": exec_steps, "train_wall_s": round(wall, 1),
           "it_per_s": its, "peak_vram_gb": peak_vram, "rankneg_window": RANKNEG_WINDOW,
           "coords_p50_radius": round(float(np.percentile(r, 50)), 3), "stop_reason": ts.get("stop_reason"),
           "train_stats": ts, "config": cfg}
    (OUTD / f"manifest-{tag}.json").write_text(json.dumps(man, indent=1))
    print(f"[card010 {tag}] n={n:,} steps={exec_steps}/{steps} {wall:.0f}s it/s={its} "
          f"vram={peak_vram}GB changed={warm_sha != trained_sha} init={init_rec['init_state_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

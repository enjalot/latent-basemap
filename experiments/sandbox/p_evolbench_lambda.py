"""Evolution λ-frontier — anchored fine-tune sweep (5th review, PROMOTED). GPU (.venv/torch).

The decisive experiment (owner 2026-09-01): the drift-triggered FULL retrain buys +0.20 OOD reception at a
one-time churn 0.369. How much of that gain can an ANCHORED FINE-TUNE buy at what fraction of the churn?

Per cell (weight w = λ):
  - WARM-START from the S0 head (ParametricUMAP.load -> fit skips _init_model when self.model is set),
  - sparse-anchor the S2 rows (first n_2) to their S2-snapshot layout (the LIVE map at the moment of drift,
    not T0-only) via anchor_ids_path=s2_anchor.npz, anchor_hold_weight=w, anchor_holdout_fraction=0.1,
  - FINE-TUNE a few epochs on S3-cumulative (incl. the reddit OOD tranche) using the precomputed S3 edges,
  - transform S3 -> coords; save coords + manifest (w, epochs, warm-start/trained state hashes, gen-key).
Endpoints already exist: w=inf (frozen) = armA-frozen S3; w=0 (full retrain) = armA-triggered S3. So the
sweep runs the interior w in {100,50,20,10,5,2}. All cells SAME seed (frontier internally exact); acceptance
band attaches when the deferred validation batch lands the MiniLM S0-head seed-43 noise floor.
Usage: p_evolbench_lambda.py <w> <n_epochs>. PROVISIONAL-PENDING-VALIDATION.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox")
SUBDIR = os.environ.get("EVOLBENCH_SUBDIR", "/data/latent-basemap/substrates/evolbench")
S0_HEAD = SB / "evolbench-S0/champion-bs16k/model.pt"
S3_EDGES = SB / "evolbench-S3/edges-k15-fuzzy.npz"
ANCHOR = SB / "lambda/s2_anchor.npz"
OUTD = SB / "lambda"
SEED = 42


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _load_S3():
    parts = ["T0", "T1", "T2", "T3"]
    return _norm(np.concatenate([np.asarray(np.load(f"{SUBDIR}/{t}/substrate.f32.npy", mmap_mode="r"),
                                            dtype=np.float32) for t in parts]))


def _state_hash(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    w = float(sys.argv[1]); n_epochs = int(sys.argv[2])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import ensure_paths; ensure_paths()
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import gen_key
    OUTD.mkdir(parents=True, exist_ok=True)
    for f in (S0_HEAD, S3_EDGES, ANCHOR):
        if not Path(f).exists():
            raise SystemExit(f"missing prerequisite: {f}")

    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    pumap = ParametricUMAP.load(str(S0_HEAD), device="cuda")   # config (arch/hparams) + S0 weights
    warm_hash = _state_hash(pumap.model)          # the warm-start point (loaded S0 head)
    warm_state = {k: v.detach().clone() for k, v in pumap.model.state_dict().items()}
    pumap.model = None                            # so the edge-list admission guard passes; _init_model
    # re-allocates the SAME arch (from pumap's stored config) and fit() then injects warm_state post-admission
    # configure the anchored fine-tune on the loaded instance
    pumap.anchor_ids_path = str(ANCHOR)
    pumap.anchor_hold_weight = w
    pumap.anchor_hold_fraction = 0.05
    pumap.anchor_holdout_fraction = 0.1
    pumap.anchored_init = "none"; pumap.anchored_init_path = ""
    pumap.batch_size = 16384                       # loaded head saved batch_size=32 (default) -> 52M
    # batches/epoch (96h); the champion's effective batch is 16k. Fine-tune uses the same large batch.
    pumap.n_epochs = max(n_epochs, 50)             # keep epochs high; the STEP CAP is the real control
    pumap._max_train_steps = int(os.environ.get("EVOLBENCH_LAMBDA_MAXSTEPS", "10000"))
    pumap.warmup_steps = 0                        # short fine-tune -> no long warmup vs the tiny horizon

    X = _load_S3(); n = X.shape[0]
    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(S3_EDGES), random_state=SEED, verbose=False,
              warm_start_state=warm_state)
    train_wall = time.time() - t0
    trained_hash = _state_hash(pumap.model)
    coords = np.asarray(pumap.transform(X, batch_size=8192), dtype=np.float32)
    tag = "inf" if w == float("inf") else (f"{w:g}")
    np.save(OUTD / f"coords-w{tag}.npy", coords)
    key = gen_key.artifact_key({"kind": "lambda-cell", "w": w, "n_epochs": n_epochs, "seed": SEED,
                                "s0_head": warm_hash, "anchor": gen_key.file_digest(ANCHOR),
                                "s3_edges": gen_key.file_digest(S3_EDGES)})
    manifest = {"schema": "evolbench-lambda-cell-2026-09-01",
                "_PROVISIONAL": "PROVISIONAL-PENDING-VALIDATION (single seed; noise floor in the deferred batch)",
                "w": w, "max_train_steps": pumap._max_train_steps, "batch_size": pumap.batch_size,
                "n_epochs_cap": n_epochs, "seed": SEED, "n": int(n), "train_wall_s": round(train_wall, 1),
                "warm_start_state_hash": warm_hash, "trained_state_hash": trained_hash,
                "warm_start_changed": bool(warm_hash != trained_hash), "gen_key": key}
    (OUTD / f"manifest-w{tag}.json").write_text(json.dumps(manifest, indent=1))
    gen_key.write_manifest(OUTD / f"coords-w{tag}.npy", key, {"w": w, "n_epochs": n_epochs})
    r = np.linalg.norm(coords.astype(np.float64) - np.median(coords, 0), axis=1)
    print(f"lambda w={tag} epochs={n_epochs}: n={n:,} train {train_wall:.0f}s warm={warm_hash} "
          f"trained={trained_hash} (changed={warm_hash!=trained_hash}) radius p50={np.percentile(r,50):.1f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

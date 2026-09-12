"""Card015 arm trainer (per card015-scale-3d.md). 3D output (n_components=3) from the shared 3D init
(5544a31160054bcc, hidden+first-2-rows copied from 2D 589895f0, fresh 3rd row). Same card013 recipe (300K
draw, fixed15 graph, 60K, LR .001, seed 42, bs16384, rankwindow 75000) + the UNCHANGED local-scale core
(Hook C is dim-agnostic). Three arms reuse card013 radii: baseline3d (none) / actual_full3d (r_actual) /
shuffled_full3d (r_shuffled). Executes from an isolated checkout derived from c0cd715; loaded source hashes are checked before fitting. Output card015-train.
Usage: run_card015_arm.py <baseline3d|actual_full3d|shuffled_full3d> <STEPS>
"""
import os, sys, time, json, hashlib, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

SB = Path("/data/latent-basemap/sandbox")
CHAMPION = SB / "dino-arrival-t0/champion-bs16k/model.pt"
SUBD = Path("/data/latent-basemap/substrates/card010-adaptive")
SUB = SUBD / "substrate.f16.npy"; EDGES = SUBD / "edges-fixed15.npz"
INIT3D = SB / "card015-init/init-card015-3d.pt"; R13 = SB / "card013-radii"
OUTD = SB / "card015-train"; OC = SB / "overseer-codex"
SEED = 42; LR = 0.001; SNAPS = (20000, 30000, 40000, 60000); PINNED = "c0cd715"
RMAP = {"baseline3d": None, "actual_full3d": "r_actual.npy", "shuffled_full3d": "r_shuffled.npy"}


def _sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
def _arr_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _state_sha(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode()); h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    arm = sys.argv[1]; steps = int(sys.argv[2]); assert arm in RMAP
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    import basemap.pumap.parametric_umap.core as _core
    import basemap.pumap.parametric_umap.datasets.edge_list_dataset as _eld
    import torch
    OUTD.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "Card015 training requires admitted GPU"
    dev = "cuda"
    assert _sha(EDGES) == "d214a839b07113df", f"fixed15 edges sha: {_sha(EDGES)}"
    init_obj = torch.load(str(INIT3D), map_location="cpu", weights_only=False)
    init_state = init_obj["model_state"]; init_sha = init_obj["init_state_sha256"]
    assert init_sha == "5544a31160054bcc" and init_obj["n_components"] == 3, "3D init identity mismatch"
    radii = None; radii_sha = None
    if RMAP[arm] is not None:
        radii = np.load(R13 / RMAP[arm]).astype(np.float32); radii_sha = _arr_sha(radii)
        assert radii.shape[0] == 300000 and np.isfinite(radii).all() and (radii > 0).all()
    githead = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loaded = {"core": {"path": _core.__file__, "sha": _sha(_core.__file__)},
              "edge_list_dataset": {"path": _eld.__file__, "sha": _sha(_eld.__file__)},
              "run_card015_arm": {"path": __file__, "sha": _sha(__file__)}, "git_head": githead,
              "pinned_commit_prefix": PINNED, "at_pinned": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "merge-base", "--is-ancestor", PINNED, "HEAD"]).returncode == 0}
    expected = json.loads((Path(__file__).resolve().parents[2] / "card015-runtime-sha.json").read_text())
    root = Path(__file__).resolve().parents[2]
    for rel, digest in expected.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, f"frozen source changed: {rel}"
    for name, record in loaded.items():
        if not isinstance(record, dict) or "path" not in record: continue
        path = Path(record["path"]).resolve()
        assert path.is_relative_to(root), f"live import rejected: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[str(path.relative_to(root))], f"runtime mismatch: {path}"
    for name, module in list(sys.modules.items()):
        if name.startswith("basemap.") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(root), f"module escaped checkout: {name}"
    loaded["verified_frozen_runtime"] = True
    torch.manual_seed(SEED); np.random.seed(SEED)
    if dev == "cuda": torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()

    pumap = ParametricUMAP.load(str(CHAMPION), device=dev); pumap.model = None
    pumap.n_components = 3     # 3D output; kernel/pair_scale are dim-agnostic
    pumap.learning_rate = LR; pumap.lr_schedule = "constant"; pumap.batch_size = 16384; pumap.warmup_steps = 0
    pumap.n_epochs = 10000; pumap._max_train_steps = steps; pumap.rankneg_window = 75000
    for a, v in (("anchor_ids_path", ""), ("anchor_hold_weight", 0.0), ("replay_bank_path", ""),
                 ("replay_weight", 0.0), ("deriv_bank_path", ""), ("deriv_weight", 0.0)):
        if hasattr(pumap, a): setattr(pumap, a, v)
    if radii is not None:
        pumap._card013_radii = radii   # same Hook C attr; dim-agnostic

    X = np.asarray(np.load(SUB, mmap_mode="r"), np.float32); n = X.shape[0]
    SNAPDIR = OUTD / arm; SNAPDIR.mkdir(exist_ok=True)
    admission = {"schema": "card015-admission-2026-09-12", "arm": arm, "written_before_steps": True,
                 "n_components": 3, "shared_init_sha256": init_sha, "derived_from_2d": "589895f037d406ae", "init_hash_format": "sorted named state", "expected_parameter_order_warm_sha": "b261492f84aa24bd",
                 "edges_sha": _sha(EDGES), "substrate_sha": _sha(SUB), "lr": LR, "lr_schedule": "constant",
                 "batch_size": 16384, "seed": SEED, "steps": steps, "rankneg_window": 75000,
                 "radii_file": (str(R13 / RMAP[arm]) if radii is not None else None), "radii_sha": radii_sha,
                 "kernel": ("d2/(r_i*r_j) detached 3D" if radii is not None else "baseline umap 3D"),
                 "snapshots": list(SNAPS), "loaded_modules": loaded, "n_rows": int(n)}
    (OUTD / f"admission-{arm}.json").write_text(json.dumps(admission, indent=1))

    t0 = time.time()
    pumap.fit(X, precomputed_edges_path=str(EDGES), random_state=SEED, verbose=False,
              warm_start_state=init_state, snapshot_steps=SNAPS, snapshot_dir=str(SNAPDIR))
    wall = time.time() - t0
    ts = dict(getattr(pumap, "_train_stats", {}) or {})
    exec_steps = int(ts.get("executed_iters", 0)); assert exec_steps == steps, f"{exec_steps} != {steps}"
    assert ts.get("positive_lr_optimizer_steps") == steps, "successful dose mismatch"
    assert abs(ts.get("lr_used_min", 0) - LR) < 1e-12 and abs(ts.get("lr_used_max", 0) - LR) < 1e-12, "LR not .001"
    assert getattr(pumap, "warm_start_sha256", None) == "b261492f84aa24bd", "parameter-order warm digest mismatch"
    coords = np.asarray(pumap.transform(X, batch_size=8192), np.float32)
    assert np.isfinite(coords).all(), "non-finite endpoint coordinates"
    assert coords.shape == (n, 3), f"expected 3D coords, got {coords.shape}"
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None
    np.save(OUTD / f"coords-{arm}.npy", coords); pumap.save(str(OUTD / f"model-{arm}.pt"))
    man = {"schema": "card015-arm-2026-09-12", "arm": arm, "n": int(n), "n_components": 3, "executed_steps": exec_steps,
           "shared_init_sha256": init_sha, "warm_parameter_order_sha": pumap.warm_start_sha256, "trained_sha256": _state_sha(pumap.model),
           "lr_used_min": ts.get("lr_used_min"), "lr_used_max": ts.get("lr_used_max"), "radii_sha": radii_sha,
           "kernel": admission["kernel"], "snapshots": list(SNAPS), "loaded_modules": loaded,
           "train_wall_s": round(wall, 1), "it_per_s": round(exec_steps / wall, 2) if wall > 0 else None,
           "peak_vram_gb": peak, "stop_reason": ts.get("stop_reason"), "train_stats": ts}
    (OUTD / f"manifest-{arm}.json").write_text(json.dumps(man, indent=1))
    print(f"[card015 {arm}] steps={exec_steps} {wall:.0f}s 3D vram={peak}GB kernel={admission['kernel']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

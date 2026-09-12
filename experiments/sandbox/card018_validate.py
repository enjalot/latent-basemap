"""Card018 canonical strict validator + identity (per root review groups 2/3). ONE source of truth reused by
the trainer (idempotent-skip), chain, and scorer. CPU-only.

- expected_identity(arm): the immutable admission identity bound into every checkpoint. FULL (64-char) content
  hashes of substrate/graph/init/radii, the code (runtime-manifest) hash, the data-manifest hash, and
  seed/LR/dose/precision. Deterministic ⇒ a resume rebuilds the same dict and the core's admission guard
  fails closed on any drift.
- runtime_manifest_check / loaded_basemap_under_root: freeze ALL basemap source + sandbox scripts and require
  every loaded basemap module to resolve UNDER the isolated worktree (not the live tree).
- strict_validate_arm: full completion validation (exact dose, mandatory/forbidden radius by arm, warm
  provenance, precision, finite endpoints, endpoint==400K snapshot, snapshot progress+freshness vs ORIGINAL
  admission, every resumable step checkpoint loadable/finite/correct-step/identity-bound with snapshot payload
  equality, model+coords file hashes). Raises AssertionError with a clear message; corrupt checkpoints fail
  closed. Idempotent completion is decided ONLY by this validator, never manifest dose alone.
"""
import hashlib, json, re
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
DATA = Path("/data/latent-basemap/substrates/card018-scale2m")
INIT3D = SB / "card015-init/init-card015-3d.pt"
DATA_MANIFEST = OC / "card018-data-manifest.json"
TD_DEFAULT = SB / "card018-train"
# FULL content hashes (from the admitted card018 data manifest; the trainer re-verifies the on-disk files)
SUB_SHA256 = "6e6894c2135057c89fe2ece3949c91845e6a5d3c076b51057dc88ffdee5a49be"
EDGES_SHA256 = "baba0034683490b7a092957709dbd3868e39a9f3a9232467c4f62304ef73b059"
R_ACTUAL_SHA256 = "56cefbf473ec0063d32df0d119c9b6dbdf9b78782c40e0e0f6850055455d019a"
DRAW_IDS_SHA256 = "f067f3a844c389c81cb37e44aa30080c565c82acce80850dfc4964bb8ba4eb3a"
INIT_NAMED_SHA = "5544a31160054bcc"; WARM_PARAM_SHA = "b261492f84aa24bd"
N = 2000000; DOSE = 400000; RANKNEG = 500000; BATCH = 16384; POS_RATIO = 0.10; SEED = 42; LR = 0.001
SNAP_STEPS = [50000, 100000, 200000, 300000, 400000]
STEP_CKPTS = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]  # 400K resumable too
ARM_RADII = {"ordinary3d": None, "actual3d": "r_actual.npy"}
KERNEL = {"ordinary3d": "baseline umap 3D", "actual3d": "d2/(r_i*r_j) detached 3D"}


def full_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def state_sha(sd):
    import numpy as np
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def runtime_manifest_path(ROOT): return Path(ROOT) / "card018-runtime-sha.json"


def runtime_manifest_check(ROOT):
    """(ok, mismatches) — every frozen source file hashes to its pinned value."""
    exp = json.loads(runtime_manifest_path(ROOT).read_text())
    bad = [n for n, hsh in exp.items() if full_sha(Path(ROOT) / n) != hsh]
    return (len(bad) == 0, bad)


def loaded_basemap_under_root(ROOT):
    """{module: {path, sha16}} for every imported basemap.* module; assert each resolves under ROOT."""
    import sys
    ROOT = Path(ROOT).resolve(); out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("basemap"): continue
        f = getattr(mod, "__file__", None)
        if not f: continue
        rp = Path(f).resolve()
        assert rp.is_relative_to(ROOT), f"loaded basemap module {name} resolves OUTSIDE worktree: {rp}"
        out[name] = {"path": str(rp), "sha16": full_sha(rp)[:16]}
    return out


def expected_identity(arm, ROOT):
    """The immutable, resume-stable admission identity. FULL content hashes + code + data-manifest + LR/seed/
    dose/precision. Big-file hashes use the admitted-manifest constants (trainer re-verifies on-disk files)."""
    assert arm in ARM_RADII, arm
    radii_file = (str(DATA / ARM_RADII[arm]) if ARM_RADII[arm] else None)
    radii_sha = (R_ACTUAL_SHA256 if arm == "actual3d" else None)
    return {
        "card": "card018", "arm": arm, "kernel": KERNEL[arm],
        "radii_file": radii_file, "radii_sha256": radii_sha,
        "init_named_sha": INIT_NAMED_SHA, "init_full_sha256": full_sha(INIT3D),
        "warm_param_sha256": WARM_PARAM_SHA,
        "substrate_sha256": SUB_SHA256, "edges_sha256": EDGES_SHA256,
        "data_manifest_sha256": full_sha(DATA_MANIFEST),
        "runtime_manifest_sha256": full_sha(runtime_manifest_path(ROOT)),
        "seed": SEED, "lr": LR, "lr_schedule": "constant", "dose": DOSE,
        "rankneg_window": RANKNEG, "batch_size": BATCH, "pos_ratio": POS_RATIO,
        "precision": "device_fp16", "x_residency": "auto", "required_input_pipeline": "device",
    }


def validate_resume_payload(ck, step, n_nodes=None):
    """Validate the actual PERM/device checkpoint schema observed in real Card019 step20K.
    CPU RNG is load-tested; CUDA Philox states are checked against this runtime's observed16-byte schema.
    This does not substitute for the mandatory real GPU resume twin.
    """
    import torch, numpy as np
    n_nodes = N if n_nodes is None else n_nodes
    assert ck.get("schema") == "pumap-ckpt-2026-08-30", "checkpoint schema"
    assert isinstance(ck.get("epoch"), int) and ck["epoch"] >= 0, "missing epoch"
    ts = ck.get("train_stats", {})
    assert ts.get("executed_iters") == ts.get("positive_lr_optimizer_steps") == ts.get("optimizer_steps_succeeded") == step, "checkpoint successful counters"
    cfg = ck["config"]
    assert cfg.get("batch_size") == BATCH and cfg.get("random_state") == SEED and cfg.get("architecture") == "residual_bottleneck", "checkpoint sampler/model config"
    assert cfg.get("replay_enabled") is False and cfg.get("deriv_enabled") is False, "unexpected preservation treatment"
    opt = ck["optimizer"]; groups = opt.get("param_groups", []); states = opt.get("state", {})
    assert groups and states, "empty optimizer state/groups"
    pids = [i for g in groups for i in g.get("params", [])]
    assert len(pids) == len(set(pids)) == len(ck["model"]) and set(pids) == set(states), "optimizer parameter identity"
    for g in groups:
        assert all(k in g for k in ["lr", "betas", "eps", "weight_decay", "amsgrad", "params"]), "incomplete Adam group"
        assert g["lr"] == LR and g["eps"] > 0 and len(g["betas"]) == 2, "Adam group configuration"
    for pid, parameter in zip(pids, ck["model"].values()):
        state = states[pid]
        assert all(k in state for k in ["step", "exp_avg", "exp_avg_sq"]), "missing Adam moments"
        assert torch.is_tensor(state["step"]) and state["step"].numel() == 1 and float(state["step"]) == step, "Adam successful dose"
        for key in ["exp_avg", "exp_avg_sq"]:
            assert state[key].shape == parameter.shape and bool(torch.isfinite(state[key]).all()), "Adam moment shape/nonfinite"
    sched = ck["scheduler"]
    assert all(k in sched for k in ["base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"]), "incomplete constant scheduler"
    assert len(sched["base_lrs"]) == len(groups) == len(sched["_last_lr"]) and all(x == LR for x in sched["base_lrs"] + sched["_last_lr"]), "scheduler LR mismatch"
    assert isinstance(sched["last_epoch"], int) and isinstance(sched["_step_count"], int) and sched["_step_count"] >= 1, "scheduler counters"
    cpu = ck["torch_rng"]
    assert torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.ndim == 1, "CPU RNG dtype/shape"
    torch.Generator(device="cpu").set_state(cpu.cpu())
    def cuda_state(v):
        return torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() == 16
    assert isinstance(ck["cuda_rng"], (list, tuple)) and len(ck["cuda_rng"]) >= 1 and all(cuda_state(v) for v in ck["cuda_rng"]), "CUDA RNG schema"
    assert cuda_state(ck["loader_gen"]), "device loader RNG schema"
    for k in ["loader_perm", "loader_pos_idx", "loader_batch_no", "loader_rank_of_node", "loader_node_at_rank", "rankneg_scale"]:
        assert k in ck and ck[k] is not None, "missing PERM/rank continuation field: " + k
    perm = ck["loader_perm"]; n_edges = 15 * n_nodes
    assert torch.is_tensor(perm) and perm.dtype == torch.int64 and perm.shape == (n_edges,), "PERM shape/dtype"
    assert int(perm.min()) >= 0 and int(perm.max()) < n_edges, "PERM bounds"
    assert isinstance(ck["loader_pos_idx"], int) and 0 <= ck["loader_pos_idx"] < n_edges + int(BATCH * POS_RATIO), "PERM cursor"
    assert isinstance(ck["loader_batch_no"], int) and ck["loader_batch_no"] >= 0, "loader batch cursor"
    rank, node = ck["loader_rank_of_node"], ck["loader_node_at_rank"]
    assert all(torch.is_tensor(v) and v.dtype == torch.int64 and v.shape == (n_nodes,) for v in [rank, node]), "rank-order shape/dtype"
    assert int(rank.min()) >= 0 and int(rank.max()) < n_nodes and int(node.min()) >= 0 and int(node.max()) < n_nodes, "rank-order bounds"
    assert torch.equal(rank[node], torch.arange(n_nodes)), "rank-order inverse mismatch"
    assert np.isfinite(ck["rankneg_scale"]) and ck["rankneg_scale"] > 0, "rank scale invalid"


def strict_validate_arm(arm, ROOT, base=None, recompute_substrate=False):
    """Full completion validation. Returns a receipt dict on success; raises AssertionError otherwise.
    Corrupt/malformed checkpoints raise (never silently pass). Set recompute_substrate=True for an
    independent data audit (CPU-cheap for edges/radii; hashes the 12GB substrate too)."""
    import torch, numpy as np
    base = Path(base) if base else TD_DEFAULT
    ad = base / f"admission-{arm}.json"; mp = base / f"manifest-{arm}.json"
    assert ad.exists() and mp.exists(), f"missing admission/manifest for {arm}"
    adm = json.loads(ad.read_text()); man = json.loads(mp.read_text())
    exp = expected_identity(arm, ROOT)

    # (a) immutable identity: canonical == admission == manifest
    assert adm.get("card012_identity") == exp, f"{arm}: admission identity != canonical"
    assert man.get("card012_identity") == exp, f"{arm}: manifest identity != canonical"
    # (b) radius mandatory for actual3d, forbidden for ordinary3d (unconditional)
    if arm == "actual3d":
        assert exp["radii_sha256"] == R_ACTUAL_SHA256 and exp["kernel"].startswith("d2/"), "actual3d must apply r_actual"
        assert man.get("radii_sha") is not None, "actual3d manifest missing radii_sha"
    else:
        assert exp["radii_file"] is None and exp["kernel"].startswith("baseline"), "ordinary3d must NOT apply radii"
        assert man.get("radii_sha") is None, "ordinary3d manifest carries radii_sha"
    # (c) warm-start provenance bound to original init (recorded even when a resume does not reapply it)
    assert man.get("warm_param_sha256") == WARM_PARAM_SHA, f"{arm}: warm-start provenance != original init"
    # (d) exact positive + executed dose
    ts = man.get("train_stats", {})
    assert man.get("executed_steps") == ts.get("executed_iters") == ts.get("positive_lr_optimizer_steps") == DOSE, f"{arm}: dose != {DOSE}"
    assert ts.get("lr_used_min") == ts.get("lr_used_max") == LR, "actual LR mismatch"
    assert man.get("proc_peak_vram_gb", float("inf")) < 30 and man.get("global_vram_used_gb", float("inf")) < 30, "VRAM cap"
    assert adm.get("kernel") == man.get("kernel") == exp["kernel"], "kernel intervention mismatch"
    if arm == "actual3d":
        expected_arr = hashlib.sha256(np.ascontiguousarray(np.load(DATA/"r_actual.npy"), dtype=np.float32).tobytes()).hexdigest()[:16]
        assert man["radii_sha"] == adm["radii_sha"] == expected_arr, "actual radius array mismatch"
    # (e) precision actually device_fp16 (distinguishes int8)
    assert man.get("pipeline_info", {}).get("x_residency") == "device_fp16", f"{arm}: pipeline not device_fp16"
    # (f) frozen isolated runtime + all basemap modules resolved under ROOT (recorded at train time)
    lm = man.get("loaded_modules", {})
    assert lm.get("verified_frozen_runtime") and lm.get("all_basemap_under_root"), f"{arm}: runtime not frozen/under-root"
    ok, bad = runtime_manifest_check(ROOT); assert ok, f"frozen runtime drifted at validate: {bad}"

    # (g) endpoint finite + == manifest sha; snapshots progress + freshness vs ORIGINAL admission
    ep = torch.load(base / f"model-{arm}.pt", map_location="cpu", weights_only=False)
    assert ep["n_components"] == 3 and ep["learning_rate"] == LR and ep["lr_schedule"] == "constant", "endpoint config"
    ep_sd = ep["model_state_dict"]; assert all(bool(torch.isfinite(t).all()) for t in ep_sd.values()), "endpoint non-finite"
    ep_sha = state_sha(ep_sd); assert ep_sha == man.get("trained_sha256"), f"{arm}: endpoint sha != manifest"
    ad_mtime = ad.stat().st_mtime; snap_sha = {}
    for s in SNAP_STEPS:
        p = base / arm / f"model-step{s}.pt"; assert p.exists(), f"missing snapshot {s}"
        o = torch.load(p, map_location="cpu", weights_only=False); sd = o["model_state_dict"]
        assert o["n_components"] == 3 and o["learning_rate"] == LR and o["lr_schedule"] == "constant", f"snapshot {s} config"
        assert all(bool(torch.isfinite(t).all()) for t in sd.values()), f"snapshot {s} non-finite"
        assert p.stat().st_mtime >= ad_mtime, f"snapshot {s} older than admission (stale)"
        snap_sha[s] = state_sha(sd)
    assert len(set(snap_sha.values())) == len(SNAP_STEPS), f"{arm}: snapshots did not progress"
    assert snap_sha[400000] == ep_sha, f"{arm}: endpoint != 400K snapshot"

    # (h) every resumable step checkpoint: loadable/finite/correct-step/identity-bound + snapshot payload equality
    ck_receipt = {}
    for s in STEP_CKPTS:
        p = base / arm / "ckpts" / f"ckpt-step{s}.pt"; assert p.exists(), f"missing resumable step ckpt {s}"
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            raise AssertionError(f"corrupt step checkpoint {s}: {e!r}")
        assert int(ck.get("global_step", -1)) == s, f"step ckpt {s} global_step mismatch"
        assert bool(ck.get("step_checkpoint")), f"ckpt {s} not marked step_checkpoint (not resumable-continue)"
        assert ck.get("card012_identity") == exp, f"step ckpt {s} identity != canonical"
        assert p.stat().st_mtime >= ad_mtime, "stale resumable checkpoint"
        assert all(k in ck for k in ["optimizer","scheduler","torch_rng","cuda_rng","loader_gen","config"]), "incomplete resumable state"
        assert ck["config"]["learning_rate"] == LR and ck["config"]["lr_schedule"] == "constant", "checkpoint LR"
        assert ck["torch_rng"] is not None and ck["cuda_rng"] is not None and ck["loader_gen"] is not None, "missing RNG"
        for state in ck["optimizer"]["state"].values():
            assert all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)), "optimizer nonfinite"
        validate_resume_payload(ck, s)
        m = ck["model"]; assert all(bool(torch.isfinite(t).all()) for t in m.values()), f"step ckpt {s} non-finite"
        msha = state_sha(m)
        if s in snap_sha:
            assert msha == snap_sha[s], f"step ckpt {s} model payload != inference snapshot"
        ck_receipt[s] = msha

    # (i) coords identity + finite
    cf = base / f"coords-{arm}.npy"; coords = np.load(cf, mmap_mode="r")
    assert coords.shape == (N, 3) and bool(np.isfinite(coords).all()), f"{arm}: coords bad"

    assert full_sha(base / f"model-{arm}.pt") == man.get("model_file_sha256"), "endpoint file identity"
    assert full_sha(cf) == man.get("coords_file_sha256"), "coordinate file identity"
    # (j) optional independent data audit
    if recompute_substrate:
        assert full_sha(DATA / "substrate.f16.npy") == SUB_SHA256, "substrate re-hash != admitted"
        assert full_sha(DATA / "edges-fixed15.npz") == EDGES_SHA256, "edges re-hash != admitted"
        assert full_sha(DATA / "r_actual.npy") == R_ACTUAL_SHA256, "r_actual re-hash != admitted"

    return {"arm": arm, "PASS": True, "model_state_sha": ep_sha, "model_file_sha": full_sha(base / f"model-{arm}.pt"),
            "coords_file_sha": full_sha(cf), "snapshot_sha": {str(k): v for k, v in snap_sha.items()},
            "step_ckpt_sha": {str(k): v for k, v in ck_receipt.items()}, "identity": exp}

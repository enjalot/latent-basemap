"""Merge-validation suite for the fog-targeted-negatives (fneg) treatment.

fneg was ported into round-0208's core.py from the round-0115 SANDBOX core
(Track 4C, PLAN4). This suite proves three things about that merge, all CPU-only
(CUDA hidden), and adds a construct→train→checkpoint smoke:

  1. EQUIVALENCE  — the merged round-0208 core reproduces the SANDBOX core's loss
     and fneg telemetry on a fixed seeded batch. Comparison is against a FROZEN
     ORACLE captured from the sandbox core at branch basemap-100m/round-0115
     (importing two same-named `basemap` packages in one interpreter is
     impractical, so the sanctioned frozen-oracle route is used; observed match
     when the oracle was captured was BIT-EXACT). If the sandbox worktree is
     present, an optional live cross-check re-derives the oracle in a subprocess.

  2. INERTNESS   — with fneg off (default), the merged core is byte-identical to
     the PRE-MERGE round-0208 core. The pre-merge core is materialised at test
     time via `git show HEAD:.../core.py` into a temp copy of the package, and
     both cores are run through the same seeded scenario with NO fneg args.

  3. ACTIVITY    — fneg_weight=1.0 vs 0.0 on the merged core produce measurably
     different loss/coords (proving the merge is not a silent no-op).

Each cross-core comparison runs in a fresh subprocess so a repo's own `basemap`
(selected by PYTHONPATH) is the one imported; determinism is seed-pinned and was
observed bit-exact across independent processes.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# CPU-only: hide any CUDA device from this process and every subprocess.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_REL = "basemap/pumap/parametric_umap/core.py"
SANDBOX_REPO = Path("/home/enjalot/code/latent-basemap")

# ── Frozen oracle: SANDBOX core (round-0115) outputs for the fixed fneg=1.0
# scenario in RUNNER_SRC. Captured 2026-08-14; the merged core matched these
# bit-exactly at capture time. Tolerances below are deliberately tight.
SANDBOX_ORACLE = {
    "fneg_telemetry": {
        "per_epoch": [
            {"epoch": 1, "band_hit_frac_of_neg": 0.1297668038408779,
             "fneg_loss_share": 0.11052028456583815,
             "batch_p90_radius_mean": 4.3375270513840665,
             "batch_p90_radius_std": 1.81848136157204,
             "batch_p90_radius_min": 0.13868071138858795,
             "batch_p90_radius_max": 6.380281925201416, "n_batches": 81},
            {"epoch": 2, "band_hit_frac_of_neg": 0.1281207133058985,
             "fneg_loss_share": 0.0884650930504431,
             "batch_p90_radius_mean": 4.984856122805748,
             "batch_p90_radius_std": 0.2810766649187469,
             "batch_p90_radius_min": 4.345287322998047,
             "batch_p90_radius_max": 5.7923054695129395, "n_batches": 81},
            {"epoch": 3, "band_hit_frac_of_neg": 0.12016460905349795,
             "fneg_loss_share": 0.0849013952682841,
             "batch_p90_radius_mean": 4.810582290461034,
             "batch_p90_radius_std": 0.2783505710406586,
             "batch_p90_radius_min": 4.321859359741211,
             "batch_p90_radius_max": 5.758299827575684, "n_batches": 81},
        ],
        "final_band_hit_frac_of_neg": 0.12016460905349795,
        "final_fneg_loss_share": 0.0849013952682841,
        "final_batch_p90_radius_mean": 4.810582290461034,
    },
    "coords_sha": -239.30754648149014,
    "coords_first8": [-3.0507328510284424, 1.8488290309906006,
                      -2.224525213241577, -2.914393663406372,
                      0.07801540195941925, -6.197103977203369,
                      -1.4919207096099854, 1.2718932628631592],
    "probe_bce": 0.25207582116127014,
}

# ── The deterministic scenario, as a standalone runner executed in a fresh
# interpreter. It imports `basemap` from sys.path (set by PYTHONPATH), so the
# SAME source drives whichever core a given repo ships. When fneg_weight==0 it
# passes NO fneg kwargs, so it also runs the pre-merge core (which lacks them).
RUNNER_SRC = r'''
import json, sys, os, tempfile
import numpy as np
import torch

def build_scenario(seed=1234, n=256, dim=16, k=6):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]
    sources = np.repeat(np.arange(n, dtype=np.int32), k)
    targets = idx.reshape(-1).astype(np.int32)
    weights = np.full(len(sources), 1.0 / k, dtype=np.float32)
    return X, sources, targets, weights, n

def main():
    fneg_weight = float(sys.argv[1]); out_path = sys.argv[2]
    seed = 1234
    X, sources, targets, weights, n_nodes = build_scenario(seed)
    tmpd = tempfile.mkdtemp()
    edges_path = os.path.join(tmpd, "edges.npz")
    np.savez_compressed(edges_path, sources=sources, targets=targets,
                        weights=weights, n_nodes=n_nodes, k=6)
    from basemap.pumap.parametric_umap import ParametricUMAP
    torch.manual_seed(seed); np.random.seed(seed)
    kw = dict(architecture="mlp", hidden_dim=32, n_layers=2, n_components=2,
              a=1.0, b=1.0, correlation_weight=0.0, learning_rate=0.01,
              n_epochs=3, batch_size=64, pos_ratio=0.3, device="cpu",
              use_amp=False, positive_target_mode="binary",
              lr_schedule="plateau", require_full_budget=False)
    if fneg_weight > 0:
        kw.update(fneg_weight=fneg_weight, fneg_lo=0.1, fneg_hi=0.4)
    pumap = ParametricUMAP(**kw)
    pumap.fit(X, precomputed_edges_path=edges_path, random_state=42, verbose=False)
    Z = pumap.transform(X)
    torch.manual_seed(99)
    with torch.no_grad():
        Xt = torch.from_numpy(X); src = Xt[:128]
        dst = Xt[torch.randperm(len(Xt))[:128]]
        se = pumap.model(src); de = pumap.model(dst)
        qs, _ = pumap._low_dim_qs(se, de)
        qs = torch.clamp(torch.nan_to_num(qs, nan=1e-7, posinf=1 - 1e-7,
                                          neginf=1e-7), 1e-7, 1 - 1e-7)
        tgt = torch.zeros_like(qs)
        probe_bce = torch.nn.functional.binary_cross_entropy(qs.float(), tgt).item()
    import basemap.pumap.parametric_umap.core as _c
    result = {
        "fneg_weight": fneg_weight,
        "fneg_telemetry": getattr(pumap, "fneg_telemetry", None),
        "coords_sha": float(np.asarray(Z, dtype=np.float64).sum()),
        "coords_first8": np.asarray(Z, dtype=np.float64).ravel()[:8].tolist(),
        "coords_finite": bool(np.isfinite(Z).all()),
        "coords_shape": list(Z.shape),
        "probe_bce": probe_bce,
        "core_path": _c.__file__,
    }
    with open(out_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def _run_scenario(pythonpath: Path, fneg_weight: float, tmp_path: Path,
                  tag: str) -> dict:
    """Run RUNNER_SRC in a fresh interpreter with `pythonpath` first on
    sys.path, so that repo's `basemap` is imported. Returns the parsed JSON."""
    runner = tmp_path / f"runner_{tag}.py"
    runner.write_text(RUNNER_SRC)
    out = tmp_path / f"out_{tag}.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(runner), str(fneg_weight), str(out)],
        cwd=str(pythonpath), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"[{tag}] runner failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    res = json.loads(out.read_text())
    # Confirm the intended core.py was the one imported.
    assert str(pythonpath) in res["core_path"], (
        f"[{tag}] imported wrong core: {res['core_path']}")
    return res


def _materialize_premerge(tmp_path: Path) -> Path:
    """Copy round-0208's basemap package to a temp tree and overwrite core.py
    with the PRE-MERGE version from `git show HEAD` (no fneg). Returns the tree
    root to place on PYTHONPATH."""
    tree = tmp_path / "premerge_tree"
    tree.mkdir()
    shutil.copytree(REPO_ROOT / "basemap", tree / "basemap",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    pre_core = subprocess.run(
        ["git", "show", f"HEAD:{CORE_REL}"], cwd=str(REPO_ROOT),
        capture_output=True, text=True)
    assert pre_core.returncode == 0, f"git show HEAD failed: {pre_core.stderr}"
    assert "fneg" not in pre_core.stdout, (
        "PRE-MERGE HEAD core.py already contains fneg — the merge appears "
        "committed; the inertness baseline must be the pre-fneg core.")
    (tree / "basemap/pumap/parametric_umap/core.py").write_text(pre_core.stdout)
    return tree


# ────────────────────────────────────────────────────────────────────────────
# 1. EQUIVALENCE
# ────────────────────────────────────────────────────────────────────────────
def test_equivalence_merged_matches_sandbox_oracle(tmp_path):
    """Merged round-0208 core == SANDBOX core outputs on the fixed fneg=1.0
    batch, compared against a frozen oracle (bit-exact at capture; asserted to
    tight tolerance here)."""
    merged = _run_scenario(REPO_ROOT, 1.0, tmp_path, "merged_f1")

    # Loss surrogate + coordinates.
    assert merged["coords_finite"]
    np.testing.assert_allclose(
        merged["coords_sha"], SANDBOX_ORACLE["coords_sha"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        merged["coords_first8"], SANDBOX_ORACLE["coords_first8"],
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        merged["probe_bce"], SANDBOX_ORACLE["probe_bce"], rtol=0, atol=1e-6)

    # fneg telemetry — the three metrics the task names, per epoch + final.
    mt = merged["fneg_telemetry"]
    ot = SANDBOX_ORACLE["fneg_telemetry"]
    assert mt is not None
    assert len(mt["per_epoch"]) == len(ot["per_epoch"])
    for me, oe in zip(mt["per_epoch"], ot["per_epoch"]):
        for key in ("band_hit_frac_of_neg", "fneg_loss_share",
                    "batch_p90_radius_mean"):
            np.testing.assert_allclose(me[key], oe[key], rtol=1e-9, atol=1e-9,
                                       err_msg=f"epoch {oe['epoch']} {key}")
    for key in ("final_band_hit_frac_of_neg", "final_fneg_loss_share",
                "final_batch_p90_radius_mean"):
        np.testing.assert_allclose(mt[key], ot[key], rtol=1e-9, atol=1e-9,
                                   err_msg=key)


@pytest.mark.skipif(not (SANDBOX_REPO / CORE_REL).exists(),
                    reason="sandbox worktree not present")
def test_equivalence_live_sandbox_cross_check(tmp_path):
    """Optional: re-derive the oracle live from the sandbox core and assert it
    equals the merged core (documents the frozen oracle's provenance)."""
    sandbox = _run_scenario(SANDBOX_REPO, 1.0, tmp_path, "sandbox_f1")
    merged = _run_scenario(REPO_ROOT, 1.0, tmp_path, "merged_f1b")
    np.testing.assert_allclose(sandbox["coords_sha"], merged["coords_sha"],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(sandbox["probe_bce"], merged["probe_bce"],
                               rtol=0, atol=1e-6)
    for me, se in zip(merged["fneg_telemetry"]["per_epoch"],
                      sandbox["fneg_telemetry"]["per_epoch"]):
        for key in ("band_hit_frac_of_neg", "fneg_loss_share",
                    "batch_p90_radius_mean"):
            np.testing.assert_allclose(me[key], se[key], rtol=1e-9, atol=1e-9)


# ────────────────────────────────────────────────────────────────────────────
# 2. INERTNESS
# ────────────────────────────────────────────────────────────────────────────
def test_inertness_fneg_off_matches_premerge(tmp_path):
    """fneg off (default) => merged core byte-identical to PRE-MERGE round-0208
    core. Proves the merge's mere existence cannot perturb legacy gate families."""
    premerge_tree = _materialize_premerge(tmp_path)
    pre = _run_scenario(premerge_tree, 0.0, tmp_path, "premerge_f0")
    merged = _run_scenario(REPO_ROOT, 0.0, tmp_path, "merged_f0")

    assert pre["fneg_telemetry"] is None and merged["fneg_telemetry"] is None
    # Byte-identical loss trajectory surrogate + output coordinates.
    assert merged["coords_sha"] == pre["coords_sha"]
    assert merged["probe_bce"] == pre["probe_bce"]
    assert merged["coords_first8"] == pre["coords_first8"]
    assert merged["coords_shape"] == pre["coords_shape"]


# ────────────────────────────────────────────────────────────────────────────
# 3. ACTIVITY
# ────────────────────────────────────────────────────────────────────────────
def test_activity_fneg_changes_training(tmp_path):
    """fneg_weight=1.0 vs 0.0 on the merged core must differ measurably — a
    merge that silently no-ops fneg would be untested."""
    off = _run_scenario(REPO_ROOT, 0.0, tmp_path, "act_f0")
    on = _run_scenario(REPO_ROOT, 1.0, tmp_path, "act_f1")

    assert off["fneg_telemetry"] is None
    assert on["fneg_telemetry"] is not None

    coords_delta = abs(on["coords_sha"] - off["coords_sha"])
    probe_delta = abs(on["probe_bce"] - off["probe_bce"])
    per_coord = np.abs(np.array(on["coords_first8"])
                       - np.array(off["coords_first8"])).mean()
    # Observed at authoring: coords_sha Δ≈329, probe_bce Δ≈0.061, per-coord Δ≈2.
    assert coords_delta > 1.0, f"coords_sha delta {coords_delta} too small"
    assert probe_delta > 1e-3, f"probe_bce delta {probe_delta} too small"
    assert per_coord > 1e-2, f"per-coord mean delta {per_coord} too small"


# ────────────────────────────────────────────────────────────────────────────
# CPU smoke: construct → train → checkpoint round-trip with fneg on.
# ────────────────────────────────────────────────────────────────────────────
def test_cpu_smoke_fneg_train_and_checkpoint(tmp_path):
    """CUDA-hidden: build the merged core with fneg on, train a couple epochs on
    tiny synthetic data + a tiny edge npz, assert finite output, populated fneg
    telemetry, and a checkpoint that round-trips the fneg params."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from basemap.pumap.parametric_umap import ParametricUMAP

    rng = np.random.RandomState(0)
    X = rng.randn(200, 12).astype(np.float32)
    from sklearn.neighbors import NearestNeighbors
    k = 5
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]
    sources = np.repeat(np.arange(len(X), dtype=np.int32), k)
    targets = idx.reshape(-1).astype(np.int32)
    weights = np.full(len(sources), 1.0 / k, dtype=np.float32)
    edges_path = tmp_path / "edges.npz"
    np.savez_compressed(edges_path, sources=sources, targets=targets,
                        weights=weights, n_nodes=len(X), k=k)

    pumap = ParametricUMAP(
        architecture="mlp", hidden_dim=24, n_layers=2, n_components=2,
        a=1.0, b=1.0, correlation_weight=0.0, learning_rate=0.01,
        n_epochs=2, batch_size=64, pos_ratio=0.3, device="cpu",
        use_amp=False, positive_target_mode="binary", lr_schedule="plateau",
        require_full_budget=False,
        fneg_weight=1.0, fneg_lo=0.1, fneg_hi=0.4,
    )
    pumap.fit(X, precomputed_edges_path=str(edges_path), random_state=1,
              verbose=False)

    Z = pumap.transform(X)
    assert Z.shape == (200, 2)
    assert np.isfinite(Z).all()

    # fneg telemetry populated.
    assert pumap.fneg_telemetry is not None
    assert "per_epoch" in pumap.fneg_telemetry
    assert len(pumap.fneg_telemetry["per_epoch"]) >= 1
    for key in ("final_band_hit_frac_of_neg", "final_fneg_loss_share",
                "final_batch_p90_radius_mean"):
        assert np.isfinite(pumap.fneg_telemetry[key])

    # Checkpoint round-trips fneg params.
    ckpt = tmp_path / "model.pt"
    pumap.save(str(ckpt))
    reloaded = ParametricUMAP.load(str(ckpt), device="cpu")
    assert reloaded.fneg_weight == 1.0
    assert reloaded.fneg_lo == 0.1
    assert reloaded.fneg_hi == 0.4

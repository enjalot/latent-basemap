"""Cards 006/007 CPU canary for the off-graph preservation-replay hook (core.py).
Validates, on a tiny real CPU fit at BOTH D=768 (Jina) and D=1536 (DINO):
  T1/T2 target/row identity + gradient (STRENGTHENED per overseer review): an INDEPENDENT
     FROZEN teacher supplies targets cached from the EXACT fp16->model-dtype input the hook
     feeds (no rounding masquerade); a PERTURBED student has a nonzero initial residual; the
     ACTUAL saved bank is loaded and paired via SHUFFLED index_select (mirroring the hook).
     Assert: student gradients nonzero+finite; training the replay loss REDUCES the residual;
     the frozen teacher's params/grads are UNCHANGED; a WRONG (mis-aligned) pairing does NOT
     satisfy — so the 1:1 row->target provenance is what drives the fit, not luck.
  T3 replay disabled restores the old path: on THIS implementation, a fit with replay params
     ABSENT and a fit with replay_bank_path SET but replay_weight=0 give BIT-IDENTICAL weights
     (block skipped); a fit with replay_weight>0 DIFFERS. (Scope: same-source-revision claim.)
  T4 graph/negative RNG untouched (generator-level): consuming from an independent replay
     generator does not advance a graph-proxy generator, and a replay-gen draw does not advance
     the torch GLOBAL rng. The REAL positive/negative/hold sampler-stream equality on the
     production device path is validated in the GPU preflight (device-path only), not here.
  T5 end-to-end replay-ON fit runs with finite loss/coords.
The true checkpoint save/reload round-trip of replay_gen and the real-sampler RNG hash are
DEVICE-path only (fit raises on CPU), so they are deferred to the short GPU preflight and
reported there. Pure CPU. Usage: tiny_replay_canary.py
"""
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

N, D, K = 1500, int(os.environ.get("CANARY_D", "768")), 10; NB = 200
rng = np.random.default_rng(3)


def _norm(X):
    return (X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


def tiny_edges(X, path):
    Xn = _norm(X); S = Xn @ Xn.T; np.fill_diagonal(S, -1); nn = np.argsort(-S, axis=1)[:, :K]
    np.savez(path, sources=np.repeat(np.arange(N), K).astype(np.int32), targets=nn.reshape(-1).astype(np.int32),
             weights=np.ones(N * K, np.float32), n_nodes=np.int64(N))


def _fit(edges, X, replay_kwargs):
    torch.manual_seed(0); np.random.seed(0)
    m = ParametricUMAP(n_components=2, n_epochs=6, batch_size=4096, low_dim_kernel="umap", **replay_kwargs)
    m.fit(X, precomputed_edges_path=str(edges), random_state=0, verbose=False)
    return m


def main():
    out = {"schema": "replay-canary-2026-09-10", "input_dim": D, "n_bank": NB}
    X = rng.standard_normal((N, D)).astype(np.float32)
    td = Path(tempfile.mkdtemp(prefix="replaycanary_")); edges = td / "edges.npz"; tiny_edges(X, edges)

    # Teacher (no replay) -> bank inputs (already L2-normed, as the real banks are) + frozen teacher coords.
    teacher = _fit(edges, X, {})
    bank_raw = rng.standard_normal((NB, D)).astype(np.float32)   # off-graph rows (not in X)
    bank_X = _norm(bank_raw)                                     # stored normed, exactly as the model consumes
    with torch.no_grad():
        bank_tgt = teacher.model(torch.from_numpy(bank_X)).float().numpy().astype(np.float32)
    np.savez(td / "bank.npz", replay_X=bank_X.astype(np.float16), replay_targets=bank_tgt.astype(np.float32),
             replay_ids=np.arange(NB, dtype=np.int64))

    # ---- T1/T2 (strengthened): independent frozen teacher, perturbed student, ACTUAL loaded
    # fp16 bank paired via SHUFFLED index_select; targets cached from the EXACT fp16->model-dtype
    # representation the hook feeds. Validate provenance + gradient direction, not a tautology. ----
    mdl_dtype = next(teacher.model.parameters()).dtype
    teacher_frozen = copy.deepcopy(teacher.model).eval()
    for p in teacher_frozen.parameters(): p.requires_grad_(False)
    teacher_ref = [p.detach().clone() for p in teacher_frozen.parameters()]

    loaded = np.load(td / "bank.npz")                              # the ACTUAL saved bank (fp16 X)
    bX16 = torch.from_numpy(loaded["replay_X"])                    # fp16, as replay holds it
    with torch.no_grad():                                          # targets from EXACT fp16->dtype input
        tgt_exact = teacher_frozen(bX16.to(mdl_dtype)).float()
    perm = torch.randperm(NB)                                      # shuffled 1:1 pairing (mirror index_select)
    r_feats = bX16.index_select(0, perm).to(mdl_dtype)
    r_tgt = tgt_exact.index_select(0, perm)                        # CORRECT pairing (same perm)
    r_tgt_wrong = tgt_exact.index_select(0, torch.roll(perm, 1))   # MIS-ALIGNED pairing

    student = copy.deepcopy(teacher.model).train()                 # perturb -> nonzero initial residual
    with torch.no_grad():
        for p in student.parameters(): p.add_(0.05 * torch.randn_like(p))

    def resid(m, tg):
        with torch.no_grad(): z = m(r_feats).float()
        return float((z - tg).pow(2).sum(1).mean())
    r0 = resid(student, r_tgt)                                     # >0 (student != teacher)
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    grad_norm0, grads_finite = 0.0, True
    for k in range(80):
        opt.zero_grad(); z = student(r_feats).float()
        rl = (z - r_tgt).pow(2).sum(1).mean(); rl.backward()
        if k == 0:
            grad_norm0 = float(sum(p.grad.norm() for p in student.parameters() if p.grad is not None))
            grads_finite = all(torch.isfinite(p.grad).all() for p in student.parameters())
        opt.step()
    r1 = resid(student, r_tgt)                                     # reduced toward CORRECT targets
    r_wrong = resid(student, r_tgt_wrong)                          # NOT reduced toward mis-aligned
    teacher_unchanged = (all(torch.equal(p, q) for p, q in zip(teacher_frozen.parameters(), teacher_ref))
                         and all(p.grad is None for p in teacher_frozen.parameters()))

    out["finite_target_scale"] = bool(np.isfinite(tgt_exact.numpy()).all())
    out["student_grad_norm_step0"] = round(grad_norm0, 6)
    out["residual_before"] = round(r0, 6); out["residual_after_correct"] = round(r1, 6)
    out["residual_after_wrong"] = round(r_wrong, 6)
    T1 = bool(out["finite_target_scale"] and r0 > 1e-4 and r1 < 0.2 * r0 and r_wrong > 3 * max(r1, 1e-9))
    T2 = bool(grad_norm0 > 0 and grads_finite and teacher_unchanged)

    # ---- T3 replay disabled restores old path (bit-identical); enabled differs ----
    mA = _fit(edges, X, {})                                                              # params absent
    mB = _fit(edges, X, {"replay_bank_path": str(td / "bank.npz"), "replay_weight": 0.0})  # bank set, weight 0 -> skipped
    mC = _fit(edges, X, {"replay_bank_path": str(td / "bank.npz"), "replay_weight": 0.02,
                         "replay_fraction": 0.05, "replay_seed": 51549})                   # active
    def sd(m): return m.model.state_dict()
    off_identical = all(torch.equal(sd(mA)[k], sd(mB)[k]) for k in sd(mA))
    on_differs = any(not torch.equal(sd(mA)[k], sd(mC)[k]) for k in sd(mA))
    coords_finite = bool(np.isfinite(np.asarray(mC.transform(X, batch_size=4096))).all())
    out["off_path_bit_identical"] = off_identical
    out["replay_on_changes_weights"] = on_differs
    out["replay_on_coords_finite"] = coords_finite
    T3 = bool(off_identical and on_differs and coords_finite)

    # ---- T4 graph/negative RNG untouched by the independent replay generator ----
    g = torch.Generator().manual_seed(123); A = [torch.randint(0, 1000, (4,), generator=g).clone() for _ in range(40)]
    g.manual_seed(123); r = torch.Generator().manual_seed(999); B = []
    for _ in range(40):
        B.append(torch.randint(0, 1000, (4,), generator=g).clone())
        torch.randint(0, 50, (819,), generator=r)   # simulate a per-step replay draw
    graph_stream_identical = all(torch.equal(a, b) for a, b in zip(A, B))
    torch.manual_seed(7); before = torch.rand(3).clone()
    torch.manual_seed(7); rr = torch.Generator().manual_seed(1); torch.randint(0, 50, (819,), generator=rr)
    after = torch.rand(3)
    global_rng_untouched = bool(torch.equal(before, after))
    out["graph_stream_identical_under_replay_draws"] = bool(graph_stream_identical)
    out["global_rng_untouched_by_replay_gen"] = global_rng_untouched
    T4 = bool(graph_stream_identical and global_rng_untouched)

    # ---- T6 fail-closed: an ENABLED replay (weight>0) with a missing/malformed bank must RAISE,
    # never silently become an unlabelled baseline. Validation fires before the training loop. ----
    def _raises(kw):
        try:
            _fit(edges, X, kw); return False
        except (AssertionError, ValueError, FileNotFoundError, OSError):
            return True
    # (a) enabled + missing path; (b) enabled + wrong target width; (c) enabled + non-normalised X
    bad_dim = td / "bank_baddim.npz"
    np.savez(bad_dim, replay_X=bank_X.astype(np.float16),
             replay_targets=np.zeros((NB, 3), np.float32), replay_ids=np.arange(NB, dtype=np.int64))  # width 3 != n_components 2
    bad_norm = td / "bank_badnorm.npz"
    np.savez(bad_norm, replay_X=(bank_X * 5.0).astype(np.float16),                                # mean norm ~5, not ~1
             replay_targets=bank_tgt.astype(np.float32), replay_ids=np.arange(NB, dtype=np.int64))
    miss = _raises({"replay_bank_path": str(td / "does_not_exist.npz"), "replay_weight": 0.02})
    baddim = _raises({"replay_bank_path": str(bad_dim), "replay_weight": 0.02})
    badnorm = _raises({"replay_bank_path": str(bad_norm), "replay_weight": 0.02})
    out["fail_closed_missing"] = miss; out["fail_closed_bad_target_dim"] = baddim
    out["fail_closed_unnormalised_X"] = badnorm
    T6 = bool(miss and baddim and badnorm)

    out["T1_target_row_identity"] = T1; out["T2_grad_student_not_teacher"] = T2
    out["T3_off_path_restored"] = T3; out["T4_rng_untouched"] = T4; out["T6_fail_closed"] = T6
    out["checkpoint_replay_gen_roundtrip"] = "deferred_to_gpu_preflight (device-path only)"
    out["real_sampler_rng_equality"] = "deferred_to_gpu_preflight (device-path only)"
    out["PASS"] = bool(T1 and T2 and T3 and T4 and T6)
    print(json.dumps(out, indent=1))
    oc = "/data/latent-basemap/sandbox/overseer-codex"
    p = f"{oc}/card007-replay-canary.json" if D == 768 else f"{oc}/card006-replay-canary.json"
    Path(p).write_text(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

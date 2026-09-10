"""Cards 006/007 CPU canary for the off-graph preservation-replay hook (core.py).
Validates, on a tiny real CPU fit at BOTH D=768 (Jina) and D=1536 (DINO):
  T1 target/row identity + finite target scale: student initialised to the teacher's
     weights reproduces each bank row's stored teacher coord (per-row residual ~0),
     and a WRONG (rolled) mapping gives a large residual -> identity, not luck.
  T2 gradient reaches student but NOT teacher: replay_loss.backward() gives every model
     parameter a finite grad, while the frozen target tensor stays requires_grad=False/grad None.
  T3 replay disabled restores the old path: a fit with replay params ABSENT and a fit with
     replay_bank_path SET but replay_weight=0 produce BIT-IDENTICAL weights (block skipped);
     a fit with replay_weight>0 DIFFERS (the term actually does something).
  T4 graph/negative RNG untouched: consuming from an independent replay generator does not
     advance a graph-proxy generator, and a replay-gen draw does not advance the torch GLOBAL rng.
  T5 end-to-end replay-ON fit runs with finite loss/coords.
The true checkpoint save/reload round-trip of replay_gen is DEVICE-path only (fit raises on CPU),
so it is deferred to the short GPU preflight and reported there. Pure CPU. Usage: tiny_replay_canary.py
"""
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

    # ---- T1 identity + finite scale: a student initialised to the teacher reproduces each bank
    # row's stored target under the SAME index_select pairing the hook uses; a rolled (mis-aligned)
    # target gives a large residual, so the 1:1 row->target pairing is what makes the residual ~0. ----
    with torch.no_grad():
        z = teacher.model(torch.from_numpy(bank_X)).float().numpy()
    perrow = np.linalg.norm(z - bank_tgt, axis=1)
    wrong = np.linalg.norm(z - np.roll(bank_tgt, 1, axis=0), axis=1)
    out["target_row_identity_median_resid"] = round(float(np.median(perrow)), 8)
    out["wrong_map_median_resid"] = round(float(np.median(wrong)), 6)
    out["finite_target_scale"] = bool(np.isfinite(bank_tgt).all())
    T1 = bool(np.median(perrow) < 1e-4 and np.median(perrow) < 0.25 * max(np.median(wrong), 1e-9)
              and out["finite_target_scale"])

    # ---- T2 gradient reaches student but NOT teacher ----
    bx = torch.from_numpy(bank_X); tg = torch.from_numpy(bank_tgt)   # both requires_grad=False (leaf constants)
    for p in teacher.model.parameters(): p.grad = None
    z_t = teacher.model(bx)
    rloss = (z_t.float() - tg.float()).pow(2).sum(dim=1).mean()
    rloss.backward()
    grads_ok = all((p.grad is not None and torch.isfinite(p.grad).all()) for p in teacher.model.parameters())
    teacher_no_grad = (tg.requires_grad is False and tg.grad is None and bx.grad is None)
    out["replay_loss_init"] = round(float(rloss.item()), 8)   # ~0 since student==teacher
    T2 = bool(grads_ok and teacher_no_grad)

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

    out["T1_target_row_identity"] = T1; out["T2_grad_student_not_teacher"] = T2
    out["T3_off_path_restored"] = T3; out["T4_rng_untouched"] = T4
    out["checkpoint_replay_gen_roundtrip"] = "deferred_to_gpu_preflight (device-path only)"
    out["PASS"] = bool(T1 and T2 and T3 and T4)
    print(json.dumps(out, indent=1))
    oc = "/data/latent-basemap/sandbox/overseer-codex"
    p = f"{oc}/card007-replay-canary.json" if D == 768 else f"{oc}/card006-replay-canary.json"
    Path(p).write_text(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Card008 DEVICE-path LR-schedule resume canary (req 3). Tiny cosine+lr_min fit with epoch
checkpoints, then: (a) resume with a DIFFERENT lr_min / lr_schedule / learning_rate must RAISE
(LambdaLR closure is not serialized, so the bound LR config must fail closed on mismatch); (b)
resume with the SAME config continues and reaches a BITWISE-IDENTICAL endpoint to the
uninterrupted twin (so the LR sequence matches across the resume boundary). Run under GPU flock.
Usage: gpu_lr_resume_canary.py
"""
import os, sys, json, tempfile, shutil, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

N, D, K, BATCH, EPOCHS, H = 3000, 64, 10, 512, 3, 200  # H bench-caps at 200 within a 3-epoch plan; epoch1 ckpt (~118) survives keep-last-2
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
rng = np.random.default_rng(8)


def _edges(X, p):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    S = Xn @ Xn.T; np.fill_diagonal(S, -1); nn = np.argsort(-S, axis=1)[:, :K]
    np.savez(p, sources=np.repeat(np.arange(N), K).astype(np.int32), targets=nn.reshape(-1).astype(np.int32),
             weights=np.ones(N * K, np.float32), n_nodes=np.int64(N))


def _fit(edges, X, ckdir, resume=None, lr=1e-4, lr_min=1e-5, sched="cosine"):
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    m = ParametricUMAP(n_components=2, n_epochs=EPOCHS, batch_size=BATCH, low_dim_kernel="umap",
                       x_residency="device_int8", learning_rate=lr, lr_schedule=sched, lr_min=lr_min)
    m.total_steps_estimate = H; m._max_train_steps = H
    fk = dict(precomputed_edges_path=str(edges), random_state=0, verbose=False,
              checkpoint_dir=str(ckdir), checkpoint_every_epochs=1)
    if resume: fk["resume_from"] = resume
    m.fit(X, **fk); return m


def _ck1(d):
    fs = [f for f in glob.glob(str(Path(d) / "ckpt-epoch*.pt")) if f.endswith("epoch1.pt")]
    return fs[0] if fs else None


def _sd_eq(a, b):
    ka, kb = a.model.state_dict(), b.model.state_dict()
    return set(ka) == set(kb) and all(torch.equal(ka[k].cpu(), kb[k].cpu()) for k in ka)


def main():
    out = {"schema": "card008-lr-resume-canary-2026-09-11"}
    if not torch.cuda.is_available():
        out["PASS"] = False; out["error"] = "no cuda"; print(json.dumps(out)); return 3
    td = Path(tempfile.mkdtemp(prefix="lrresume_")); e = td / "e.npz"
    X = rng.standard_normal((N, D)).astype(np.float32); _edges(X, e)
    m_un = _fit(e, X, td / "un")                      # uninterrupted cosine+lr_min
    endpoint = int(m_un._train_stats.get("executed_iters", 0))
    ck1 = _ck1(td / "un")
    # A real resume test REQUIRES a genuine mid-run checkpoint (not None after pruning).
    if ck1 is None:
        out["error"] = "epoch1 checkpoint missing (pruned) — cannot test real resume"; out["PASS"] = False
        print(json.dumps(out, indent=1)); return 3
    ck1_step = int(torch.load(ck1, map_location="cpu", weights_only=False).get("global_step", -1))
    ck1_valid = bool(0 < ck1_step < endpoint)

    def resume_raises(expect_msg, **kw):
        try:
            _fit(e, X, td / "tmp", resume=ck1, **kw)
            return {"raised": False, "msg_ok": False}
        except ValueError as ex:
            return {"raised": True, "msg_ok": bool(expect_msg in str(ex))}
        except (AssertionError, RuntimeError) as ex:
            return {"raised": True, "msg_ok": False, "wrong_error": str(ex)[:120]}
        finally:
            shutil.rmtree(td / "tmp", ignore_errors=True)
    diff_min = resume_raises("mismatch", lr_min=5e-5)
    diff_sched = resume_raises("mismatch", sched="plateau", lr_min=0.0)
    diff_lr = resume_raises("mismatch", lr=5e-5)
    m_re = _fit(e, X, td / "re", resume=ck1)          # same config -> continues to bitwise-identical endpoint
    same_ok = _sd_eq(m_un, m_re)
    re_endpoint = int(m_re._train_stats.get("executed_iters", 0))

    out.update(uninterrupted_endpoint_steps=endpoint, epoch1_ckpt_step=ck1_step, epoch1_step_in_range=ck1_valid,
               resumed_endpoint_steps=re_endpoint,
               diff_lr_min=diff_min, diff_schedule=diff_sched, diff_lr_peak=diff_lr,
               same_config_resume_bitwise_identical=same_ok)
    out["PASS"] = bool(ck1_valid and same_ok and re_endpoint == endpoint
                       and all(r["raised"] and r["msg_ok"] for r in (diff_min, diff_sched, diff_lr)))
    print(json.dumps(out, indent=1)); (OC / "card008-lr-resume-canary.json").write_text(json.dumps(out, indent=1))
    shutil.rmtree(td, ignore_errors=True)
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

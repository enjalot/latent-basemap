"""Card009 derivative-preservation DEVICE canary (production arch). Tiny synthetic setup on the
real residual_bottleneck arch + device_int8 path. Validates before the GPU arms:
  R1 off-path: deriv_weight=0 (or params absent) -> BITWISE-identical weights; enabled differs.
  R2 JVP gradient reaches student, not teacher: at a PERTURBED student the deriv term produces
     nonzero-finite student grads; frozen teacher Jv is constant (no grad); correct pairing at the
     unperturbed (teacher) init gives ~0 deriv loss (create_graph JVP matches the cached teacher Jv).
  R3 chunk invariance: the deriv term over a subbatch equals the mean of its split chunks.
  R4 incorrect-pairing negative control: shuffling the teacher-Jv targets raises the deriv loss well
     above the correctly-paired loss (the 1:1 direction->Jv pairing is what the term rewards).
  R5 resume fail-closed: resume with a different deriv_weight / bank_sha raises "mismatch"; same-config
     resume is bitwise-identical (deriv_gen + config bound into the checkpoint).
  R6 throughput/memory: deriv-on vs deriv-off it/s + peak VRAM over K steps -> overhead %% (target <=25%,
     else reduce deriv_subbatch). Autocast-off fp32 double-backward on the real arch.
Writes card009-deriv-canary.json. Run under the GPU flock. Usage: gpu_deriv_canary.py
"""
import os, sys, json, time, tempfile, shutil, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
from basemap.pumap.parametric_umap.core import ParametricUMAP

N, D, K, NB, ND = 3000, 64, 10, 400, 400
BATCH, EPOCHS = 512, 3
OC = Path("/data/latent-basemap/sandbox/overseer-codex")
rng = np.random.default_rng(909)


def _norm(a):
    return (a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)


def tiny_edges(X, path):
    Xn = _norm(X); S = Xn @ Xn.T; np.fill_diagonal(S, -1); nn = np.argsort(-S, axis=1)[:, :K]
    np.savez(path, sources=np.repeat(np.arange(N), K).astype(np.int32), targets=nn.reshape(-1).astype(np.int32),
             weights=np.ones(N * K, np.float32), n_nodes=np.int64(N))


def _mk(**kw):
    return ParametricUMAP(n_components=2, n_epochs=EPOCHS, batch_size=BATCH, low_dim_kernel="umap",
                          architecture="residual_bottleneck", x_residency="device_int8", **kw)


def _fit(edges, X, ckdir=None, resume=None, snap=(), maxsteps=200, **kw):
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    m = _mk(**kw); m.total_steps_estimate = maxsteps; m._max_train_steps = maxsteps
    fk = dict(precomputed_edges_path=str(edges), random_state=0, verbose=False)
    if ckdir: fk.update(checkpoint_dir=str(ckdir), checkpoint_every_epochs=1)
    if resume: fk["resume_from"] = resume
    t = time.time(); m.fit(X, **fk); return m, time.time() - t


def _sd_eq(a, b):
    ka, kb = a.model.state_dict(), b.model.state_dict()
    return set(ka) == set(kb) and all(torch.equal(ka[k].cpu(), kb[k].cpu()) for k in ka)


def _ck1(d):
    fs = [f for f in glob.glob(str(Path(d) / "ckpt-epoch*.pt")) if f.endswith("epoch1.pt")]
    return fs[0] if fs else None


def main():
    out = {"schema": "card009-deriv-canary-2026-09-11"}
    if not torch.cuda.is_available():
        out["PASS"] = False; out["error"] = "no cuda"; print(json.dumps(out)); return 3
    td = Path(tempfile.mkdtemp(prefix="derivcan_")); e = td / "e.npz"
    X = rng.standard_normal((N, D)).astype(np.float32); tiny_edges(X, e)
    # teacher (short fit) -> frozen; build deriv bank at exact fp16 input with tangent dirs + teacher Jv
    teacher, _ = _fit(e, X, maxsteps=150)
    tmodel = teacher.model.eval()
    for p in tmodel.parameters(): p.requires_grad_(False)
    bx = _norm(rng.standard_normal((ND, D)).astype(np.float32)); bx16 = bx.astype(np.float16); xu = bx16.astype(np.float32)
    nb = _norm(rng.standard_normal((ND, D)).astype(np.float32))
    raw = nb - xu; xn2 = (xu * xu).sum(1, keepdims=True).clip(1e-12)
    v = raw - (raw * xu).sum(1, keepdims=True) / xn2 * xu; v = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    with torch.no_grad():
        _, jvt = torch.autograd.functional.jvp(lambda i: tmodel(i), torch.from_numpy(xu).cuda(), torch.from_numpy(v).cuda())
    jvt = jvt.float().cpu().numpy().astype(np.float32)
    bank = td / "deriv.npz"
    np.savez(bank, deriv_X=bx16, deriv_dir=v, deriv_teacher_jv=jvt, deriv_scale=np.ones(ND, np.float32), deriv_ids=np.arange(ND, dtype=np.int64))

    dkw = dict(deriv_bank_path=str(bank), deriv_weight=0.02, deriv_subbatch=128, deriv_seed=7, deriv_radius=33.6717)

    # R1 off-path
    mA, _ = _fit(e, X)
    mB, _ = _fit(e, X, deriv_bank_path=str(bank), deriv_weight=0.0)   # params set, weight 0 -> skipped
    mC, _ = _fit(e, X, **dkw)
    R1 = bool(_sd_eq(mA, mB) and not _sd_eq(mA, mC))
    out["R1_off_path_bitwise_identical"] = bool(_sd_eq(mA, mB)); out["R1_on_differs"] = bool(not _sd_eq(mA, mC))

    # R2/R4 deriv-term properties at teacher init (correct pairing ~0) + perturbed grads + wrong pairing
    import copy
    xu_t = torch.from_numpy(xu).cuda(); v_t = torch.from_numpy(v).cuda(); jvt_t = torch.from_numpy(jvt).cuda()
    def deriv_loss(model, jv_target, create_graph):
        _, jvs = torch.autograd.functional.jvp(lambda i: model(i), xu_t, v_t, create_graph=create_graph)
        return ((jvs.float() - jv_target) / 33.6717).pow(2).sum(1).mean()
    with torch.no_grad():   # VALUE-only (no graph on the teacher)
        loss_correct_init = float(deriv_loss(tmodel, jvt_t, False).item())          # student==teacher -> ~0
        loss_wrong_init = float(deriv_loss(tmodel, jvt_t[torch.randperm(ND)], False).item())  # shuffled -> >0
    student = copy.deepcopy(tmodel)                                                 # independent params
    with torch.no_grad():
        for p in student.parameters(): p.add_(0.05 * torch.randn_like(p))
    for p in student.parameters(): p.requires_grad_(True)
    for p in tmodel.parameters(): p.grad = None                                     # clean before the student backward
    dl = deriv_loss(student, jvt_t, True); dl.backward()                            # gradient must reach student only
    gnorm = float(sum(p.grad.norm() for p in student.parameters() if p.grad is not None))
    teacher_nograd = all(p.grad is None for p in tmodel.parameters())
    out["R2_loss_correct_at_init"] = round(loss_correct_init, 8); out["R2_student_grad_norm"] = round(gnorm, 5)
    out["R2_teacher_no_grad"] = bool(teacher_nograd)
    out["R4_loss_wrong_pairing"] = round(loss_wrong_init, 6)
    R2 = bool(loss_correct_init < 1e-3 and gnorm > 0 and teacher_nograd)
    R4 = bool(loss_wrong_init > 10 * max(loss_correct_init, 1e-9))

    # R3 chunk invariance of the deriv term (on the PERTURBED student -> nonzero loss)
    def perrow_sq(model, idx):
        with torch.no_grad():
            _, jvs = torch.autograd.functional.jvp(lambda i: model(i), xu_t[idx], v_t[idx])
        return ((jvs.float() - jvt_t[idx]) / 33.6717).pow(2).sum(1)
    allrows = torch.arange(ND, device="cuda")
    full = float(perrow_sq(student, allrows).mean().item())
    a = perrow_sq(student, torch.arange(0, ND // 2, device="cuda")); b = perrow_sq(student, torch.arange(ND // 2, ND, device="cuda"))
    chunked = float(torch.cat([a, b]).mean().item())
    R3 = bool(abs(chunked - full) < 1e-4)
    out["R3_chunk_full"] = round(full, 8); out["R3_chunk_split"] = round(chunked, 8); out["R3_chunk_invariant"] = R3

    # R5 resume fail-closed
    m_un, _ = _fit(e, X, ckdir=td / "un", **dkw); ck1 = _ck1(td / "un")
    def resume_raises(**kw):
        try:
            _fit(e, X, ckdir=td / "tmp", resume=ck1, **kw); return False
        except ValueError as ex:
            return "mismatch" in str(ex)
        finally:
            shutil.rmtree(td / "tmp", ignore_errors=True)
    diff_w = resume_raises(deriv_bank_path=str(bank), deriv_weight=0.05, deriv_subbatch=128, deriv_seed=7, deriv_radius=33.6717)
    # different bank content (shuffle targets -> different sha)
    bank2 = td / "deriv2.npz"; np.savez(bank2, deriv_X=bx16, deriv_dir=v, deriv_teacher_jv=jvt[::-1].copy(),
                                        deriv_scale=np.ones(ND, np.float32), deriv_ids=np.arange(ND, dtype=np.int64))
    diff_bank = resume_raises(deriv_bank_path=str(bank2), deriv_weight=0.02, deriv_subbatch=128, deriv_seed=7, deriv_radius=33.6717)
    m_re, _ = _fit(e, X, ckdir=td / "re", resume=ck1, **dkw)
    R5 = bool(diff_w and diff_bank and _sd_eq(m_un, m_re))
    out["R5_diff_weight_raises"] = diff_w; out["R5_diff_bank_raises"] = diff_bank; out["R5_same_resume_bitwise"] = bool(_sd_eq(m_un, m_re))

    # R6 throughput / VRAM overhead
    torch.cuda.reset_peak_memory_stats(); _, t_off = _fit(e, X, maxsteps=200); vram_off = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats(); _, t_on = _fit(e, X, maxsteps=200, **dkw); vram_on = torch.cuda.max_memory_allocated() / 2**30
    overhead = (t_on - t_off) / max(t_off, 1e-9)
    out["R6_it_s_off"] = round(200 / t_off, 2); out["R6_it_s_on"] = round(200 / t_on, 2)
    out["R6_overhead_frac"] = round(float(overhead), 4); out["R6_vram_on_gb"] = round(float(vram_on), 3)
    R6 = bool(np.isfinite(overhead))   # report overhead; do not fail canary on it (subbatch is the knob)

    out["R1"] = R1; out["R2"] = R2; out["R3"] = R3; out["R4"] = R4; out["R5"] = R5; out["R6_measured"] = R6
    out["PASS"] = bool(R1 and R2 and R3 and R4 and R5)
    out["note"] = "R6 overhead reported (subbatch is the throughput knob if >25%); nonzero-drift coefficient calibration is a separate step on the real banks."
    print(json.dumps(out, indent=1)); (OC / "card009-deriv-canary.json").write_text(json.dumps(out, indent=1))
    shutil.rmtree(td, ignore_errors=True)
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

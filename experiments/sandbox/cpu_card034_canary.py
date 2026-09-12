"""Card034 CPU checks (per card034 + foundation review item 3/4). No GPU. Exercises the guarded radial +
logphi + three losses + sampler:
  loss/gradient vs explicit scalar math; beta0 NCE == fixed-logit; InfoNCE additive-logit-offset + group-
  permutation invariance; wrong positive index changes loss; radial guard at EXACT zero (forward 0, gradient
  0) / near-zero / distance 1e5 (finite); NaN/Inf inputs are NOT concealed (nonfinite loss, no nan_to_num);
  finite-difference head gradients for all three objectives; NCE/InfoNCE far-tail gradient NONZERO (not a
  clamped plateau); shared-head == repeated-head accumulated gradient for EVERY objective; sampler 4/4/2 over
  a 10-edge epoch with exact positive/noise counts and no duplicated/skipped positive edge; constructor +
  resume-state validation rejects empty graph / n<2 / bad endpoints / malformed permutation/cursor;
  identical streams at the same seed. Exit 0 = PASS. Usage: cpu_card034_canary.py
"""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
import card034_grouped as G

OC = Path("/data/latent-basemap/sandbox/overseer-codex")
LOSSES = {"umap": lambda r, b=None: G.grouped_umap_loss(r), "nce": lambda r, b: G.grouped_nce_loss(r, b),
          "infonce": lambda r, b=None: G.grouped_infonce_loss(r)}


def _loss(name, radial, beta):
    return G.grouped_umap_loss(radial) if name == "umap" else (G.grouped_nce_loss(radial, beta) if name == "nce" else G.grouped_infonce_loss(radial))


def main():
    torch.set_default_dtype(torch.float64)
    R = {"schema": "card034-cpu-canary-2026-09-12", "a": G.A, "b": G.B, "n_noise": G.N_NOISE}
    g = torch.Generator().manual_seed(34)
    m = 5; head = torch.randn(m, 2, generator=g); tails = torch.randn(m, G.GROUP, 2, generator=g)
    radial = G.radial_from_emb(head, tails)

    # (1) explicit-scalar agreement
    r = radial.numpy(); p = np.clip(1.0 / (1.0 + G.A * r), G.CLAMP_LO, G.CLAMP_HI)
    umap_ref = np.mean((-np.log(p[:, 0]) - np.log1p(-p[:, 1:]).sum(1)) / G.GROUP)
    logp = -np.log1p(G.A * r); sp = np.log1p(np.exp(-np.abs(logp))) + np.maximum(logp, 0)
    t = np.zeros_like(logp); t[:, 0] = 1.0; nce_ref = np.mean(sp - t * logp)
    info_ref = np.mean(np.log(np.exp(logp).sum(1)) - logp[:, 0])
    R["umap_matches_explicit"] = abs(float(G.grouped_umap_loss(radial)) - umap_ref) < 1e-9
    R["nce_matches_explicit"] = abs(float(G.grouped_nce_loss(radial, torch.tensor(0.0))) - nce_ref) < 1e-9
    R["infonce_matches_explicit"] = abs(float(G.grouped_infonce_loss(radial)) - info_ref) < 1e-9
    R["beta0_nce_equals_fixed"] = abs(float(G.grouped_nce_loss(radial, torch.tensor(0.0))) - nce_ref) < 1e-12

    # (2) InfoNCE offset + permutation invariance
    lp = G.logphi_from_radial(radial); c = 2.71828
    R["infonce_offset_invariant"] = abs(float(G._infonce_from_logphi(lp)) - float(G._infonce_from_logphi(lp + c))) < 1e-9
    perm = torch.cat([torch.tensor([0]), 1 + torch.randperm(G.N_NOISE, generator=g)])
    R["infonce_group_perm_invariant"] = abs(float(G.grouped_infonce_loss(radial)) - float(G.grouped_infonce_loss(radial[:, perm]))) < 1e-9
    sw = torch.arange(G.GROUP); sw[0], sw[3] = 3, 0
    R["wrong_positive_changes_loss"] = abs(float(G.grouped_infonce_loss(radial)) - float(G.grouped_infonce_loss(radial[:, sw]))) > 1e-6

    # (3a) radial guard: EXACT zero forward 0 + gradient 0; near-zero + far 1e5 finite
    hz = torch.zeros(1, 2, requires_grad=True); tz = torch.zeros(1, G.GROUP, 2)      # tail slot 0 coincides with head
    rz = G.radial_from_emb(hz, tz); rz.sum().backward()
    R["radial_exact_zero_forward"] = float(rz.detach()[0, 0]) == 0.0
    R["radial_exact_zero_grad_zero"] = bool(torch.isfinite(hz.grad).all()) and float(hz.grad.abs().sum()) == 0.0
    hn = torch.zeros(1, 2, requires_grad=True); tn = torch.full((1, G.GROUP, 2), 1e-4)
    rn = G.radial_from_emb(hn, tn); rn.sum().backward(); R["radial_near_zero_finite"] = bool(torch.isfinite(hn.grad).all())
    hf = torch.zeros(1, 2, requires_grad=True); tf = torch.full((1, G.GROUP, 2), 1e5)
    rf = G.radial_from_emb(hf, tf); R["radial_far_finite"] = bool(torch.isfinite(rf).all() and (rf > 0).all())
    # (1b) production mixed-precision edge (root evidence): fp16 +60000/-60000 -> cast-before-subtract finite,
    # while the naive fp16 subtract-then-upcast overflows to Inf (the bug this guards against).
    hh = torch.tensor([[60000.0, 0.0]], dtype=torch.float16); tt = torch.full((1, G.GROUP, 2), -60000.0, dtype=torch.float16)
    R["fp16_cast_before_subtract_finite"] = bool(torch.isfinite(G.radial_from_emb(hh, tt)).all())
    R["fp16_naive_subtract_overflows"] = not bool(torch.isfinite(tt - hh.unsqueeze(1)).all())

    # (3b) NaN/Inf inputs are NOT concealed (nonfinite loss)
    hbad = head.clone(); hbad[0, 0] = float("nan")
    R["nan_input_not_concealed"] = not bool(torch.isfinite(G.grouped_infonce_loss(G.radial_from_emb(hbad, tails))))
    hinf = head.clone(); hinf[0, 0] = float("inf")   # unclipped infonce: inf head -> nonfinite (umap's prob clamp would mask it)
    R["inf_input_not_concealed"] = not bool(torch.isfinite(G.grouped_infonce_loss(G.radial_from_emb(hinf, tails))))

    # (3c) finite-difference head gradient for ALL three objectives; (3d) shared==repeated accumulated grad
    fd_ok = True; sr_ok = True
    for name in ("umap", "nce", "infonce"):
        h1 = head.clone().requires_grad_(True); beta = torch.zeros((), requires_grad=(name == "nce"))
        L = _loss(name, G.radial_from_emb(h1, tails), beta if name == "nce" else None); L.backward()
        ana = float(h1.grad[2, 0]); eps = 1e-6
        hp = head.clone(); hp[2, 0] += eps; hm = head.clone(); hm[2, 0] -= eps
        num = (float(_loss(name, G.radial_from_emb(hp, tails), torch.tensor(0.0))) - float(_loss(name, G.radial_from_emb(hm, tails), torch.tensor(0.0)))) / (2 * eps)
        fd_ok = fd_ok and abs(ana - num) < 1e-4
        hrep = head.clone().unsqueeze(1).repeat(1, G.GROUP, 1).requires_grad_(True)
        d2 = ((tails - hrep) ** 2).sum(-1); tiny = torch.finfo(d2.dtype).tiny
        rrep = torch.where(d2 == 0, torch.zeros_like(d2), d2.clamp_min(tiny).pow(G.B))
        Lr = _loss(name, rrep, torch.zeros((), requires_grad=(name == "nce"))); Lr.backward()
        sr_ok = sr_ok and torch.allclose(hrep.grad.sum(1), h1.grad, atol=1e-9)
    R["finite_difference_grad_all_objectives"] = fd_ok
    R["shared_equals_repeated_head_grad_all"] = sr_ok

    # (3e) NCE/InfoNCE far-tail POSITIVE gradient nonzero (not a clamped plateau)
    hp2 = torch.zeros(1, 2, requires_grad=True); tp2 = torch.zeros(1, G.GROUP, 2); tp2[0, 0] = 1e3   # far positive
    Ln = G.grouped_nce_loss(G.radial_from_emb(hp2, tp2), torch.tensor(0.0)); Ln.backward()
    R["nce_far_tail_grad_nonzero"] = float(hp2.grad.abs().sum()) > 0
    hp3 = torch.zeros(1, 2, requires_grad=True); Li = G.grouped_infonce_loss(G.radial_from_emb(hp3, tp2)); Li.backward()
    R["infonce_far_tail_grad_nonzero"] = float(hp3.grad.abs().sum()) > 0

    # (4) sampler: 10-edge epoch -> blocks 4,4,2; exact counts; no dup/skip; identical streams
    N = 8; src = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 3], np.int64); tgt = np.array([1, 2, 3, 4, 5, 6, 7, 0, 2, 5], np.int64)
    s = G.GroupedSampler(N, src, tgt, seed=5, block_pos=4, n_noise=9)
    sizes = []; edges_seen = []
    for _ in range(3):
        h, tl = s.next_block(); sizes.append(h.shape[0])
        assert tl.shape[1] == G.GROUP and tl.shape[0] == h.shape[0]
        for rr in range(h.shape[0]): edges_seen.append((int(h[rr]), int(tl[rr, 0])))
    R["sampler_block_sizes_442"] = sizes == [4, 4, 2]
    R["sampler_exact_9to1"] = True   # every tl had GROUP=10 cols (9 noise + 1 pos), asserted above
    R["sampler_no_dup_skip_epoch"] = sorted(edges_seen) == sorted(zip(src.tolist(), tgt.tolist()))
    a = G.GroupedSampler(N, src, tgt, seed=42, block_pos=4); b = G.GroupedSampler(N, src, tgt, seed=42, block_pos=4)
    R["identical_streams_same_seed"] = all(np.array_equal(a.next_block()[i2], b.next_block()[i2]) for _ in range(3) for i2 in (0,)) or True
    # redo identical-streams cleanly
    a = G.GroupedSampler(N, src, tgt, seed=42, block_pos=4); b = G.GroupedSampler(N, src, tgt, seed=42, block_pos=4); ok = True
    for _ in range(4):
        ha, ta = a.next_block(); hb, tb = b.next_block(); ok = ok and np.array_equal(ha, hb) and np.array_equal(ta, tb)
    R["identical_streams_same_seed"] = ok

    # constructor + resume-state validation
    def _raises(fn):
        try: fn(); return False
        except (AssertionError, ValueError): return True
        except Exception: return False
    R["reject_empty_graph"] = _raises(lambda: G.GroupedSampler(N, np.array([], np.int64), np.array([], np.int64), seed=1))
    R["reject_n_lt_2"] = _raises(lambda: G.GroupedSampler(1, src, tgt, seed=1))
    R["reject_bad_endpoint"] = _raises(lambda: G.GroupedSampler(5, src, tgt, seed=1))     # ids up to 7 >= n=5
    def _bad_state():
        ss = G.GroupedSampler(N, src, tgt, seed=1); st = ss.state(); st["cursor"] = 999; ss.load_state(st)
    R["reject_malformed_cursor"] = _raises(_bad_state)
    def _bad_perm():
        ss = G.GroupedSampler(N, src, tgt, seed=1); st = ss.state(); st["perm"] = np.zeros(len(src), np.int64); ss.load_state(st)
    R["reject_malformed_perm"] = _raises(_bad_perm)

    R = {k: (bool(v) if isinstance(v, (np.bool_, bool)) else v) for k, v in R.items()}
    keys = [k for k in R if isinstance(R[k], bool) and k != "PASS"]
    R["PASS"] = bool(all(R[k] for k in keys))
    (OC / "card034-cpu-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

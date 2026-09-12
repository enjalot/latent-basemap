"""Card034 CPU checks (per card034 "Independent CPU checks"). No GPU. Exercises the shared grouped losses +
sampler (card034_grouped): loss/gradient vs explicit scalar math; beta0 NCE == fixed-logit; InfoNCE
additive-logit-offset invariance + group-permutation invariance; wrong positive index changes loss; gradients
include the shared head; repeated-head-forward == shared-head-forward accumulated gradients; noise uniform
nonself on a small exhaustively enumerated fixture; the short final group obeys 9:1; three arms use identical
sample streams. Exit 0 = PASS. Usage: cpu_card034_canary.py
"""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import torch
import card034_grouped as G

OC = Path("/data/latent-basemap/sandbox/overseer-codex")


def main():
    torch.set_default_dtype(torch.float64)
    R = {"schema": "card034-cpu-canary-2026-09-12", "a": G.A, "b": G.B, "n_noise": G.N_NOISE}
    g = torch.Generator().manual_seed(34)
    m = 5; head = torch.randn(m, 2, generator=g); tails = torch.randn(m, G.GROUP, 2, generator=g)
    phi = G.phi_from_emb(head, tails)

    # (1) explicit-scalar agreement (numpy) for all three losses on the same phi
    p = np.clip(np.nan_to_num(phi.numpy(), nan=1e-7), G.CLAMP_LO, G.CLAMP_HI)
    umap_ref = np.mean((-np.log(p[:, 0]) - np.log1p(-p[:, 1:]).sum(1)) / G.GROUP)
    nce_ref = np.mean(-(np.log(p[:, 0]) - np.log1p(np.exp(np.log(p[:, 0])))) )  # slot0 positive term only? compute full below
    # full NCE (BCE-with-logits) explicit: t0=1 rest 0; logit=log p
    logit = np.log(p); softplus = np.log1p(np.exp(-np.abs(logit))) + np.maximum(logit, 0)
    t = np.zeros_like(logit); t[:, 0] = 1.0
    nce_ref = np.mean(softplus - t * logit)
    infonce_ref = np.mean(np.log(np.exp(np.log(p)).sum(1)) - np.log(p[:, 0]))
    R["umap_matches_explicit"] = bool(abs(float(G.grouped_umap_loss(phi)) - umap_ref) < 1e-9)
    R["nce_matches_explicit"] = bool(abs(float(G.grouped_nce_loss(phi, torch.tensor(0.0))) - nce_ref) < 1e-9)
    R["infonce_matches_explicit"] = bool(abs(float(G.grouped_infonce_loss(phi)) - infonce_ref) < 1e-9)

    # (2) beta0 NCE == fixed-logit (beta=0 is the fixed-logit BCE)
    R["beta0_nce_equals_fixed"] = bool(abs(float(G.grouped_nce_loss(phi, torch.tensor(0.0))) - nce_ref) < 1e-12)

    # (3) InfoNCE additive-logit-offset invariance: adding c to every log phi leaves it unchanged
    logp = torch.log(G._clamp(phi)); c = 3.14159
    R["infonce_offset_invariant"] = bool(abs(float(G._infonce_from_logphi(logp)) - float(G._infonce_from_logphi(logp + c))) < 1e-9)
    # (4) group-permutation invariance: permuting the 9 NOISE slots leaves InfoNCE unchanged
    perm = torch.cat([torch.tensor([0]), 1 + torch.randperm(G.N_NOISE, generator=g)])
    R["infonce_group_perm_invariant"] = bool(abs(float(G.grouped_infonce_loss(phi)) - float(G.grouped_infonce_loss(phi[:, perm]))) < 1e-9)
    # (5) wrong positive index changes loss (swap slot 0 with a noise slot)
    sw = torch.arange(G.GROUP); sw[0], sw[3] = 3, 0
    R["wrong_positive_changes_loss"] = bool(abs(float(G.grouped_infonce_loss(phi)) - float(G.grouped_infonce_loss(phi[:, sw]))) > 1e-6)

    # (6) gradients include the shared head; (7) repeated-head == shared-head accumulated gradient
    h1 = head.clone().requires_grad_(True); t1 = tails.clone()
    L = G.grouped_infonce_loss(G.phi_from_emb(h1, t1)); L.backward()
    R["shared_head_receives_gradient"] = bool(h1.grad is not None and h1.grad.abs().sum().item() > 0)
    hrep = head.clone().unsqueeze(1).repeat(1, G.GROUP, 1).requires_grad_(True)   # independent per-slot head copies
    d2 = ((tails - hrep) ** 2).sum(-1); phirep = 1.0 / (1.0 + G.A * d2 ** G.B)
    Lr = G._infonce_from_logphi(torch.log(G._clamp(phirep))); Lr.backward()
    R["repeated_equals_shared_head_grad"] = bool(torch.allclose(hrep.grad.sum(1), h1.grad, atol=1e-9))

    # (8) uniform nonself on a small exhaustively enumerated fixture (N=8)
    N = 8; src = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 3], np.int64); tgt = np.array([1, 2, 3, 4, 5, 6, 7, 0, 2, 5], np.int64)
    s = G.GroupedSampler(N, src, tgt, seed=1, block_pos=4, n_noise=9)
    seen = {i: set() for i in range(N)}; self_hit = False
    for _ in range(400):
        h, tl = s.next_block()
        for r in range(h.shape[0]):
            for k in range(1, G.GROUP):
                seen[int(h[r])].add(int(tl[r, k]))
                if int(tl[r, k]) == int(h[r]): self_hit = True
    R["noise_never_self"] = (not self_hit)
    R["noise_covers_all_nonself"] = all(seen[i] == (set(range(N)) - {i}) for i in range(N))

    # (9) short final group obeys 9:1 (every block, incl. the short tail, has exactly GROUP columns)
    s2 = G.GroupedSampler(N, src, tgt, seed=2, block_pos=4, n_noise=9)
    shapes = [s2.next_block()[1].shape[1] for _ in range(6)]     # 10 edges, block 4 -> 4,4,2 then reshuffle...
    R["short_tail_9to1"] = all(w == G.GROUP for w in shapes)

    # (10) three arms identical sample streams (same seed -> identical blocks)
    a = G.GroupedSampler(N, src, tgt, seed=42, block_pos=4); b = G.GroupedSampler(N, src, tgt, seed=42, block_pos=4)
    ok = True
    for _ in range(6):
        ha, ta = a.next_block(); hb, tb = b.next_block()
        ok = ok and np.array_equal(ha, hb) and np.array_equal(ta, tb)
    R["identical_streams_same_seed"] = ok

    keys = ["umap_matches_explicit", "nce_matches_explicit", "infonce_matches_explicit", "beta0_nce_equals_fixed",
            "infonce_offset_invariant", "infonce_group_perm_invariant", "wrong_positive_changes_loss",
            "shared_head_receives_gradient", "repeated_equals_shared_head_grad", "noise_never_self",
            "noise_covers_all_nonself", "short_tail_9to1", "identical_streams_same_seed"]
    R["PASS"] = bool(all(R[k] for k in keys))
    (OC / "card034-cpu-canary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1), flush=True)
    return 0 if R["PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

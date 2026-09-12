"""Card034 grouped relative-neighbor objective — shared core (guarded radial + logphi, three losses, grouped
sampler). Standalone (no core edits): the trainer reuses the existing model architecture + checkpoint export
and calls these functions.

Numerical contract (foundation review items 1 & 2):
  - radial is computed in FP32 with the verified `_low_dim_qs` guard: radial = where(r2==0, 0,
    r2.clamp_min(finfo.tiny)**b). Exact-zero forward is 0 (phi=1) with zero finite gradient; near/far gradients
    are finite nonzero. NO nan_to_num (nonfinite is never concealed).
  - grouped_umap keeps UMAP's [1e-7, 1-1e-7] PROBABILITY clamp (only in its branch).
  - grouped_nce / grouped_infonce use the stable UNCLIPPED logphi = -log1p(a*radial) (Card024's contrastive
    helper) so a distant-positive gradient is never erased by the UMAP probability floor.

Grouped recipe: shared deterministic PERM positive stream; for EACH positive (head i, tail j) draw exactly
N_NOISE=9 iid noise tails uniformly from all N-1 IDs except i (with replacement; graph neighbors and j remain
eligible). Group = [j, noise_1..9] (slot 0 the positive). All arms consume IDENTICAL groups. Head forwarded
once + shared across its 10 candidates; gradients flow through the shared head.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

A, B = 1.9328, 0.7905
CLAMP_LO, CLAMP_HI = 1e-7, 1 - 1e-7      # UMAP probability clamp — grouped_umap ONLY
BLOCK_POS = 1638; N_NOISE = 9; GROUP = N_NOISE + 1


def radial_from_emb(head, tails):
    """head (m,2), tails (m,GROUP,2) -> guarded radial (m,GROUP), FP32. radial = where(r2==0, 0,
    r2.clamp_min(tiny)**b): singular-free at exact zero (forward 0, gradient 0); finite near/far gradients."""
    delta = tails - head.unsqueeze(1)
    if delta.dtype in (torch.float16, torch.bfloat16):    # upcast AMP fp16 to FP32 (never downcast fp64)
        delta = delta.float()
    r2 = (delta * delta).sum(-1)                          # squared Euclidean, >= FP32
    tiny = torch.finfo(r2.dtype).tiny
    radial_nz = r2.clamp_min(tiny).pow(B)                 # flat below tiny -> zero grad at r2==0
    return torch.where(r2 == 0, torch.zeros_like(r2), radial_nz)


def phi_from_radial(radial): return 1.0 / (1.0 + A * radial)
def logphi_from_radial(radial): return -torch.log1p(A * radial)     # stable, UNCLIPPED log phi
def phi_from_emb(head, tails): return phi_from_radial(radial_from_emb(head, tails))   # convenience


def grouped_umap_loss(radial):
    """mean BCE over the 1 positive (slot 0) + 9 noise slots; UMAP probability clamp applied HERE ONLY."""
    p = torch.clamp(phi_from_radial(radial), CLAMP_LO, CLAMP_HI)
    per = -torch.log(p[:, 0]) - torch.log1p(-p[:, 1:]).sum(1)
    return (per / GROUP).mean()


def grouped_nce_loss(radial, beta):
    """mean stable binary-logit loss, logit = logphi - beta (beta fixed 0 or learned). No probability clamp."""
    logit = logphi_from_radial(radial) - beta
    t = torch.zeros_like(logit); t[:, 0] = 1.0
    return Fn.binary_cross_entropy_with_logits(logit, t, reduction="mean")


def _infonce_from_logphi(logp):
    return (torch.logsumexp(logp, dim=1) - logp[:, 0]).mean()


def grouped_infonce_loss(radial):
    """mean over groups of logsumexp(logphi over all 10) - logphi_positive. No scalar/temperature, no
    false-negative filtering, positive kept in the denominator. No probability clamp."""
    return _infonce_from_logphi(logphi_from_radial(radial))


class GroupedSampler:
    """Shared deterministic PERM positive stream + exactly-9 uniform nonself noise per positive. Arm-agnostic.
    Constructor + resume state are validated (empty graph, n<2, out-of-range endpoints, malformed
    permutation/cursor) BEFORE any resampling loop or resume."""

    def __init__(self, n_nodes, sources, targets, seed, block_pos=BLOCK_POS, n_noise=N_NOISE):
        self.n = int(n_nodes); self.src = np.asarray(sources).astype(np.int64); self.tgt = np.asarray(targets).astype(np.int64)
        self.E = int(self.src.shape[0]); self.block_pos = int(block_pos); self.n_noise = int(n_noise); self.seed = int(seed)
        assert self.n >= 2, "grouped sampler needs n>=2 nodes"
        assert self.E > 0 and self.tgt.shape[0] == self.E, "empty / mismatched graph"
        assert self.block_pos >= 1 and self.n_noise >= 1, "invalid block/noise size"
        assert int(self.src.min()) >= 0 and int(self.src.max()) < self.n, "source id out of range"
        assert int(self.tgt.min()) >= 0 and int(self.tgt.max()) < self.n, "target id out of range"
        self.perm_gen = np.random.default_rng(self.seed)
        self.noise_gen = np.random.default_rng(self.seed + 777)
        self.perm = self.perm_gen.permutation(self.E); self.cursor = 0; self.epoch = 0

    def state(self):
        return {"perm": self.perm.copy(), "cursor": int(self.cursor), "epoch": int(self.epoch),
                "perm_gen": self.perm_gen.bit_generator.state, "noise_gen": self.noise_gen.bit_generator.state}

    def load_state(self, st):
        perm = np.asarray(st["perm"]).astype(np.int64); cur = int(st["cursor"])
        assert perm.shape == (self.E,) and np.array_equal(np.sort(perm), np.arange(self.E)), "malformed permutation"
        assert 0 <= cur <= self.E, "malformed cursor"
        self.perm = perm; self.cursor = cur; self.epoch = int(st["epoch"])
        self.perm_gen.bit_generator.state = st["perm_gen"]; self.noise_gen.bit_generator.state = st["noise_gen"]

    def _draw_noise(self, heads):
        P = heads.shape[0]; noise = self.noise_gen.integers(0, self.n, size=(P, self.n_noise))
        clash = noise == heads[:, None]
        while clash.any():
            noise[clash] = self.noise_gen.integers(0, self.n, size=int(clash.sum()))
            clash = noise == heads[:, None]
        return noise

    def next_block(self):
        """(head_ids (P,), tail_ids (P,GROUP)); P<=block_pos. Reshuffles at epoch boundaries; the short final
        block still gets exactly n_noise noise per positive. Each positive edge appears exactly once/epoch."""
        if self.cursor >= self.E:
            self.perm = self.perm_gen.permutation(self.E); self.cursor = 0; self.epoch += 1
        end = min(self.cursor + self.block_pos, self.E)
        idx = self.perm[self.cursor:end]; self.cursor = end
        heads = self.src[idx]; pos = self.tgt[idx]; noise = self._draw_noise(heads).astype(np.int64)
        return heads, np.concatenate([pos[:, None], noise], axis=1)

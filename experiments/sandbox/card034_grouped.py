"""Card034 grouped relative-neighbor objective — shared core (sampler + three losses). Standalone (no core
edits): the trainer reuses the existing model architecture + checkpoint export and calls these functions.

Grouped recipe (per card034-grouped-relative-objective.md): each update consumes the next block of up to
BLOCK_POS graph-positive edges from a shared deterministic PERM stream; for EACH positive (head i, tail j)
draw exactly N_NOISE=9 iid noise tails uniformly from all N-1 IDs except i (with replacement; graph neighbors
and j remain eligible). Group = [j, noise_1..9] (10 candidates), slot 0 the positive. All three arms consume
IDENTICAL groups (same PERM + same independent noise generator) ⇒ identical 9:1 exposure incl. the short
final epoch block. phi(d)=1/(1+a*(d^2)^b) with the shared finite/zero-gradient clamp. The head is forwarded
ONCE and shared across its 10 candidates; gradients flow through the shared head.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

A, B = 1.9328, 0.7905
CLAMP_LO, CLAMP_HI = 1e-7, 1 - 1e-7
BLOCK_POS = 1638          # positives per update (16384 * 0.1, floor)
N_NOISE = 9               # exactly 9 noise tails per positive (9:1)
GROUP = N_NOISE + 1       # 10 candidates, slot 0 = positive


def phi_from_emb(head, tails):
    """head (m,2), tails (m,GROUP,2) -> phi (m,GROUP). phi=1/(1+a*(d^2)^b), d^2 = squared Euclidean."""
    d2 = ((tails - head.unsqueeze(1)) ** 2).sum(-1)
    return 1.0 / (1.0 + A * d2 ** B)


def _clamp(phi):
    return torch.clamp(torch.nan_to_num(phi, nan=1e-7, posinf=1 - 1e-7, neginf=1e-7), CLAMP_LO, CLAMP_HI)


def grouped_umap_loss(phi):
    """mean BCE over the 1 positive (slot 0) + 9 noise slots, using phi with the UMAP probability clamp."""
    p = _clamp(phi)
    per = -torch.log(p[:, 0]) - torch.log1p(-p[:, 1:]).sum(1)     # -log p0 - sum log(1-p_k)
    return (per / GROUP).mean()


def grouped_nce_loss(phi, beta):
    """mean stable binary-logit loss, logit = log(phi) - beta (beta fixed 0 or learned scalar)."""
    logit = torch.log(_clamp(phi)) - beta
    t = torch.zeros_like(logit); t[:, 0] = 1.0
    return Fn.binary_cross_entropy_with_logits(logit, t, reduction="mean")


def _infonce_from_logphi(logp):
    """mean over groups of logsumexp(log phi over all 10 candidates) - log phi_positive. No scalar/temperature,
    no false-negative filtering, positive kept in the denominator."""
    return (torch.logsumexp(logp, dim=1) - logp[:, 0]).mean()


def grouped_infonce_loss(phi):
    return _infonce_from_logphi(torch.log(_clamp(phi)))


class GroupedSampler:
    """Shared deterministic PERM positive stream + exactly-9 uniform nonself noise per positive. Arm-agnostic
    (identical groups for every arm at a given seed). Saves/restores PERM + cursor + noise-generator state for
    genuine resume. Noise: uniform in [0,N) resampled to exclude the head i (nonself); tail j and graph
    neighbors stay eligible."""

    def __init__(self, n_nodes, sources, targets, seed, block_pos=BLOCK_POS, n_noise=N_NOISE):
        self.n = int(n_nodes); self.src = np.asarray(sources); self.tgt = np.asarray(targets)
        self.E = self.src.shape[0]; self.block_pos = int(block_pos); self.n_noise = int(n_noise)
        self.seed = int(seed)
        self.perm_gen = np.random.default_rng(self.seed)
        self.noise_gen = np.random.default_rng(self.seed + 777)
        self.perm = self.perm_gen.permutation(self.E)
        self.cursor = 0; self.epoch = 0

    def state(self):
        return {"perm": self.perm.copy(), "cursor": int(self.cursor), "epoch": int(self.epoch),
                "perm_gen": self.perm_gen.bit_generator.state, "noise_gen": self.noise_gen.bit_generator.state}

    def load_state(self, st):
        self.perm = np.asarray(st["perm"]); self.cursor = int(st["cursor"]); self.epoch = int(st["epoch"])
        self.perm_gen.bit_generator.state = st["perm_gen"]; self.noise_gen.bit_generator.state = st["noise_gen"]

    def _draw_noise(self, heads):
        """heads (P,) -> noise (P, n_noise) uniform nonself (resample any draw equal to its head)."""
        P = heads.shape[0]; noise = self.noise_gen.integers(0, self.n, size=(P, self.n_noise))
        # resample collisions with the head (nonself); loops until clean (prob of repeated collision ~ (1/N)^k)
        clash = noise == heads[:, None]
        while clash.any():
            noise[clash] = self.noise_gen.integers(0, self.n, size=int(clash.sum()))
            clash = noise == heads[:, None]
        return noise

    def next_block(self):
        """Return (head_ids (P,), tail_ids (P,GROUP)) for the next block; P<=block_pos (short final block).
        Reshuffles PERM at epoch boundaries (new perm_gen draw) — the short tail still gets exactly 9 noise."""
        if self.cursor >= self.E:
            self.perm = self.perm_gen.permutation(self.E); self.cursor = 0; self.epoch += 1
        end = min(self.cursor + self.block_pos, self.E)
        idx = self.perm[self.cursor:end]; self.cursor = end
        heads = self.src[idx].astype(np.int64); pos = self.tgt[idx].astype(np.int64)
        noise = self._draw_noise(heads).astype(np.int64)
        tails = np.concatenate([pos[:, None], noise], axis=1)     # slot 0 = positive
        return heads, tails

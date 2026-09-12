"""Card034 ONE production training/checkpoint/restore engine (per production review item 2). The trainer,
device canary and throughput preflight all drive THIS engine so they exercise identical production behavior:
same optimizer groups (model wd .01, LR .001; grouped_nce scalar in its own group LR .001 wd 0), same
GRAD_CLIP (V.GRAD_CLIP — not the nonexistent V.CLIP), the EXACT frozen InfoNCE coefficient, the same
nonfinite handling (reject a nonfinite forward BEFORE any objective; count AMP/nonfinite skips), live stats,
and coherent POST-STEP checkpoints (model + sampler both advanced past the same block — no skipped/duplicated
batch) with a full payload (model/optimizer/scaler/scalar/sampler/RNG + up-to-date stats + ID-digest probes).
Restore deep-validates the payload BEFORE loading (card034_validate.validate_ckpt_payload) and rejects a
missing/nonfinite scalar rather than keeping a fresh 0.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.optim import AdamW
import card034_validate as V
import card034_grouped as G

PROBE_SUCCESS = {1, V.DOSE // 2, V.DOSE}
PROBE_ATTEMPTED = {1, V.DOSE // 2, V.DOSE}
MAX_CONSEC_BAD = 300     # bound persistent nonfinite/AMP-skip failure instead of looping until timeout


def _id_digest(heads, tails):
    h = hashlib.sha256(); h.update(np.ascontiguousarray(heads).tobytes()); h.update(np.ascontiguousarray(tails).tobytes())
    return h.hexdigest()[:16]


def pipeline_receipt(Xt):
    """Explicit experimental pipeline receipt bound to the ACTUAL device bank observations — NOT a borrowed
    core fit receipt. Asserts the bank is CUDA + fp16 + (N,1536)."""
    assert Xt.is_cuda and Xt.dtype == torch.float16 and Xt.ndim == 2 and Xt.shape[1] == 1536, "device bank not CUDA/fp16/(N,1536)"
    return {"x_residency": "device_fp16", "device": str(Xt.device), "dtype": str(Xt.dtype), "shape": list(Xt.shape),
            "amp_dtype": "float16", "source": "card034_engine explicit observation", "verified": True}


class GroupedEngine:
    def __init__(self, arm, identity, coeff, dev, champion, init_state):
        assert arm in V.ARMS; self.arm = arm; self.identity = identity; self.coeff = float(coeff); self.dev = dev
        from basemap.pumap.parametric_umap.core import ParametricUMAP
        self.p = ParametricUMAP.load(str(champion), device=dev); self.p.model = None; self.p.n_components = V.NC
        self.p.learning_rate = V.LR; self.p.lr_schedule = "constant"
        self.p._init_model(1536); self.p.model.load_state_dict(init_state); self.model = self.p.model
        self.beta = None
        groups = [{"params": list(self.model.parameters()), "lr": V.LR, "weight_decay": V.WEIGHT_DECAY}]
        if arm == "grouped_nce":
            self.beta = nn.Parameter(torch.zeros((), device=dev)); groups.append({"params": [self.beta], "lr": V.LR, "weight_decay": 0.0})
        self.opt = AdamW(groups); self.scaler = torch.amp.GradScaler(dev, enabled=True)
        self.success = 0; self._consec_bad = 0
        self.stats = {"attempted_steps": 0, "positive_lr_optimizer_steps": 0, "amp_skips": 0, "nonfinite_skips": 0,
                      "attempted_positive": 0, "attempted_noise": 0, "successful_positive": 0, "successful_noise": 0,
                      "lr_used_min": V.LR, "lr_used_max": V.LR, "exposure_probes": [], "attempted_probes": []}

    def _loss(self, radial):
        if self.arm == "grouped_umap": return G.grouped_umap_loss(radial)
        if self.arm == "grouped_nce": return G.grouped_nce_loss(radial, self.beta)
        return self.coeff * G.grouped_infonce_loss(radial)

    def step(self, Xt, heads, tails):
        """One production step on a grouped block. Returns True iff a successful positive-LR update was applied.
        Updates stats live. Rejects a nonfinite forward/loss before applying an update (charged as a skip)."""
        self.stats["attempted_steps"] += 1
        self.stats["attempted_positive"] += int(heads.shape[0]); self.stats["attempted_noise"] += int(heads.shape[0]) * V.N_NOISE
        if self.stats["attempted_steps"] in PROBE_ATTEMPTED:     # attempted-step probe (honest re AMP skips)
            self.stats["attempted_probes"].append({"attempted_step": int(self.stats["attempted_steps"]), "n_pos": int(heads.shape[0]),
                                                   "noise_per_pos": int(tails.shape[1] - 1), "id_digest": _id_digest(heads, tails)})
        h = torch.as_tensor(heads, device=self.dev); tl = torch.as_tensor(tails, device=self.dev)
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            he = self.model(Xt.index_select(0, h))
            te = self.model(Xt.index_select(0, tl.reshape(-1))).reshape(h.shape[0], G.GROUP, V.NC)
        if not G.embeddings_finite(he, te):                       # reject nonfinite forward BEFORE any objective
            self.stats["nonfinite_skips"] += 1; return self._bad()
        loss = self._loss(G.radial_from_emb(he, te))
        if not bool(torch.isfinite(loss)):
            self.stats["nonfinite_skips"] += 1; return self._bad()
        self.scaler.scale(loss).backward(); self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), V.GRAD_CLIP)   # MODEL grad clip
        prev = self.scaler.get_scale(); self.scaler.step(self.opt); self.scaler.update()
        if self.scaler.get_scale() >= prev:
            self._consec_bad = 0; self.success += 1
            self.stats["positive_lr_optimizer_steps"] = self.success
            self.stats["successful_positive"] += int(heads.shape[0]); self.stats["successful_noise"] += int(heads.shape[0]) * V.N_NOISE
            if self.success in PROBE_SUCCESS:
                self.stats["exposure_probes"].append({"step": int(self.success), "n_pos": int(heads.shape[0]),
                                                       "noise_per_pos": int(tails.shape[1] - 1), "id_digest": _id_digest(heads, tails)})
            return True
        self.stats["amp_skips"] += 1; return self._bad()

    def _bad(self):
        """A skipped (nonfinite/AMP-overflow) step. Bound persistent failure instead of looping to timeout."""
        self._consec_bad += 1
        if self._consec_bad >= MAX_CONSEC_BAD:
            raise RuntimeError(f"card034 engine: {self._consec_bad} consecutive nonfinite/AMP-skipped steps — bounded failure")
        return False

    def final_beta(self):
        return float(self.beta.detach()) if self.beta is not None else None

    def ckpt_dict(self, sampler, step_checkpoint):
        return {"schema": "card034-ckpt-2026-09-12", "arm": self.arm, "mode": V.MODE[self.arm],
                "global_step": int(self.success), "epoch": int(sampler.epoch), "step_checkpoint": bool(step_checkpoint),
                "identity": self.identity, "coeff": self.coeff, "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                "beta": (self.beta.detach().cpu() if self.beta is not None else None),
                "sampler_state": sampler.state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "train_stats": dict(self.stats), "consecutive_bad":self._consec_bad}

    def restore(self, ck, sampler, ROOT, n_nodes):
        """Deep-validate BEFORE restore (identity + Adam groups/moments/counter + scalar + scaler + RNG +
        sampler); then load model/opt/scaler/scalar/sampler/RNG. Rejects a missing/nonfinite scalar."""
        V.validate_ckpt_payload(ck, self.arm, ROOT, self.identity, n_nodes, expect_beta=(self.beta is not None),
                                model_sd=self.model.state_dict(), expect_perm_len=sampler.E)
        self.model.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["optimizer"]); self.scaler.load_state_dict(ck["scaler"])
        if self.beta is not None:
            assert ck.get("beta") is not None and bool(torch.isfinite(ck["beta"]).all()), "missing/nonfinite scalar on resume"
            with torch.no_grad(): self.beta.copy_(ck["beta"].to(self.dev))
        sampler.load_state(ck["sampler_state"])
        torch.set_rng_state(ck["torch_rng"].to("cpu", torch.uint8))
        torch.cuda.set_rng_state_all([s.to("cpu", torch.uint8) for s in ck["cuda_rng"]])
        self.success = int(ck["global_step"]); self.stats = dict(ck["train_stats"]);self._consec_bad=int(ck.get("consecutive_bad",0))

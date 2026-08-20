#!/usr/bin/env python3
"""R11 REHEARSAL harness — validate the WHOLE post-training pipe before attempt-5.

The lesson of R0268 attempts 1-4 is that each fix validated the LAST failure while a NEW one waited
one step downstream in a code path that had never executed to completion: transform SIGINT (attempt 1),
CellWatchdog swap-abort (attempt 3), then the latent `int8_full` use-after-del in the train-receipt
assembly (attempt 4). The only gate that validates the whole pipe at once is a rehearsal that runs the
REAL node — receipt assembly, seal, the fixed transform poll, the 100M transform, the transform-receipt,
the done-marker, AND the runner's own validation layer — with EXACTLY ONE substitution: `model.fit` is
stubbed to load attempt-4's preserved BYTE-IDENTICAL model instead of training for 24h. Because attempt-3
and attempt-4 produced byte-identical weights (sha 5bc2cd58), the rehearsal transform produces the very
coordinates attempt-5 will produce, so the tripwire + finiteness checks are meaningful too.

**This is a PLUMBING REHEARSAL. Its outputs are NON-EVIDENCE** — written to a throwaway dir, never a
panel/gate input. attempt-5 produces the round evidence. Run via the REAL runner (`runner.py --node`
against the rehearsal queue) so the runner's done-marker validation / declared-output checks execute
exactly as attempt-5 will see them — NOT by calling run_train() in-process (that would leave the runner
validation layer as the next never-executed path — the delegate's rider).
"""
from __future__ import annotations

import os
from typing import Any, Mapping

PRESERVED_MODEL = "/data/latent-basemap/runs/round-0268/salvage/attempt4-preserved/model.pt"

#: The synthetic post-fit sampler stamp — the REAL host-int8 uniform values run_train's fail-closed
#: tripwire requires (weighted_effective False / positive_sampling uniform / x_residency host_int8),
#: mirrored from a real attempt's `_pipeline_info` log line. The rehearsal does not train, so this is
#: representative, not measured; the rehearsal receipt is NON-EVIDENCE and says so.
_REHEARSAL_PIPELINE_INFO = {
    "pipeline": "host_int8",
    "sampler_class": "DeviceEdgeSampler",
    "positive_sampling": "uniform",
    "x_residency": "host_int8",
    "weighted_requested": False,
    "weighted_effective": False,
    "uniform_with_replacement": True,
    "positive_with_replacement": True,
    "path_reason": "REHEARSAL stub (host_int8 uniform); not a measured train",
    "multiplicity_policy": "row_multiplicity_uncapped",
}


def apply_rehearsal_fit(model: Any, preserved_model_path: str, expected_updates: int) -> Any:
    """Make `model` look like a completed fit WITHOUT training: transplant attempt-4's initialised,
    weight-loaded encoder (via the TESTED ``ParametricUMAP.load`` classmethod — NOT a hand-rolled
    ``load_state_dict`` on ``self.model``, which is ``None`` until fit()/load() runs), and synthesize
    the exact post-fit telemetry run_train reads. A module-level function so the load path is
    UNIT-TESTED on CPU (the stub's load path was itself a never-executed path — the class this whole
    effort exists to catch)."""
    from basemap.pumap.parametric_umap import ParametricUMAP

    loaded = ParametricUMAP.load(preserved_model_path, device=model.device)
    # transplant the initialised, weight-loaded encoder; the transform runs self.model(batch).
    model.model = loaded.model
    model.input_dim = loaded.input_dim
    model.is_fitted = True
    model._train_stats = {
        "successful_positive_lr_updates": int(expected_updates),
        "executed_positive_lr_updates": int(expected_updates),
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "rehearsal_stub": True,
    }
    model._pipeline_info = dict(_REHEARSAL_PIPELINE_INFO)
    # fneg_telemetry must be non-None (train_checks.fneg_reweighting_was_active).
    if not getattr(model, "fneg_telemetry", None):
        model.fneg_telemetry = {"rehearsal_stub": True, "fneg_weight": float(model.fneg_weight)}
    return model


def run_rehearsal_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    from basemap.pumap.parametric_umap import ParametricUMAP
    from experiments import round0268_nodes as N

    preserved = str(job.get("preserved_model_path") or PRESERVED_MODEL)
    if not os.path.exists(preserved):
        raise RuntimeError(f"R11 rehearsal: preserved model missing at {preserved}")
    # hard refusal: the rehearsal must write to a clearly-marked throwaway dir, never round artifacts.
    out0 = str((job.get("outputs") or [""])[0])
    if "rehearsal" not in out0.lower():
        raise RuntimeError(
            f"R11 rehearsal REFUSES to run: output dir {out0!r} is not a rehearsal/NON-EVIDENCE dir"
        )

    expected_updates = int(job.get("base_horizon", 0)) * int(N.DOSE_MULTIPLIER)

    real_fit = ParametricUMAP.fit

    def _stub_fit(self, X, *args, **kwargs):
        return apply_rehearsal_fit(self, preserved, expected_updates)

    ParametricUMAP.fit = _stub_fit
    try:
        # the REAL node — receipt assembly, seal, fixed transform poll, 100M transform, transform
        # receipt, done-marker all execute for real, under the real runner that invoked us.
        N.run_train(active, job)
    finally:
        ParametricUMAP.fit = real_fit


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == "rehearse_transform_pipe_minilm_fneg_100m_x2_hostint8":
        run_rehearsal_job(active, job)
        return
    raise RuntimeError(f"R11 rehearsal: unknown action {action!r}")

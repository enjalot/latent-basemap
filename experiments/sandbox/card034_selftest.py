"""Card034 canonical-validator self-test (CPU, no GPU). Builds structurally-valid grouped_umap/grouped_nce/
grouped_infonce fixtures whose checkpoints carry the FULL deep payload (Adam groups + per-parameter finite
moments + step counter, scalar param, scaler state, CPU/CUDA RNG, valid sampler permutation/cursor/epoch,
live stats), asserts strict_validate_arm PASSES them, then applies independent single-fault mutations that
must each FAIL closed — including the deep-state faults (stale ckpt stats, wrong Adam step counter, nonfinite
Adam moment, missing scaler, malformed sampler permutation, missing scalar, missing init-payload hash).
Usage: card034_selftest.py
"""
import os, sys, json, shutil, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()
import card034_validate as V
import torch

ROOT = Path(__file__).resolve().parents[2]
V.N = 64
# Explicit reduced-shape fixture; production uses real bound init tensors.
V.expected_model_state=lambda:_sd(0)
_GEN = np.random.default_rng(0).bit_generator.state


def _sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {"proj_out.weight": torch.randn(2, 4, generator=g), "proj_out.bias": torch.randn(2, generator=g)}


def _sampler_state():
    return {"perm": np.arange(V.N * 15), "cursor": 3, "epoch": 1, "perm_gen": _GEN, "noise_gen": _GEN}


def _adam_state(model_sd, nce):
    ids = list(range(len(model_sd))); state = {}
    for i, v in enumerate(model_sd.values()):
        state[i] = {"step": torch.tensor(float(GS_HOLDER[0])), "exp_avg": torch.zeros_like(v), "exp_avg_sq": torch.zeros_like(v)}
    groups = [{"params": ids, "lr": V.LR, "weight_decay": V.WEIGHT_DECAY, "betas": (.9, .999), "eps": 1e-8, "amsgrad": False}]
    if nce:
        state[len(ids)] = {"step": torch.tensor(float(GS_HOLDER[0])), "exp_avg": torch.zeros(()), "exp_avg_sq": torch.zeros(())}
        groups.append({"params": [len(ids)], "lr": V.LR, "weight_decay": 0.0, "betas": (.9, .999), "eps": 1e-8, "amsgrad": False})
    return {"param_groups": groups, "state": state}


GS_HOLDER = [0]


def _ckpt(arm, s, ident, model_sd):
    GS_HOLDER[0] = s; nce = (arm == "grouped_nce")
    return {"schema": "card034-ckpt-2026-09-12", "arm": arm, "mode": V.MODE[arm], "global_step": s, "epoch": 1,
            "step_checkpoint": True, "identity": ident, "coeff": ident.get("infonce_coeff") or 1.0, "model": model_sd,
            "optimizer": _adam_state(model_sd, nce), "scaler": torch.amp.GradScaler("cpu",enabled=True).state_dict(),
            "beta": (torch.tensor(0.03) if nce else None), "sampler_state": _sampler_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": [torch.zeros(16, dtype=torch.uint8)],
            "train_stats": {"positive_lr_optimizer_steps": s}}


def build(base, arm):
    td = Path(base); (td / arm / "ckpts").mkdir(parents=True, exist_ok=True)
    ident = V.expected_identity(arm, ROOT)
    snap = {s: _sd(1000 + i) for i, s in enumerate(V.SNAP_STEPS)}
    for s in V.SNAP_STEPS: torch.save({"model_state_dict": snap[s], "n_components": V.NC}, td / arm / f"model-step{s}.pt")
    torch.save({"model_state_dict": snap[60000]}, td / f"model-{arm}.pt")
    for s in V.STEP_CKPTS: torch.save(_ckpt(arm, s, ident, snap[s] if s in snap else _sd(9000 + s)), td / arm / "ckpts" / f"ckpt-step{s}.pt")
    torch.save(_ckpt(arm, 2747, ident, _sd(5)), td / arm / "ckpts" / "ckpt-epoch1.pt")
    np.save(td / f"coords-{arm}.npy", np.zeros((V.N, V.NC), "f4"))
    fb = (0.03 if arm == "grouped_nce" else None)
    _prb = lambda key: [{key: s, "noise_per_pos": 9, "id_digest": f"d{s}"} for s in (1, 30000, 60000)]
    man = {"arm": arm, "mode": V.MODE[arm], "identity": ident, "warm_init_sha256": V.INIT_SHA, "init_payload_sha256": V.INIT_SHA,
           "executed_steps": V.DOSE, "train_stats": {"positive_lr_optimizer_steps": V.DOSE, "exposure_probes": _prb("step"), "attempted_probes": _prb("attempted_step")},
           "lr_used_min": V.LR, "lr_used_max": V.LR, "weight_decay": V.WEIGHT_DECAY, "grad_clip": V.GRAD_CLIP,
           "final_beta": fb, "infonce_coeff": ident.get("infonce_coeff"),
           "pipeline_receipt": {"x_residency": "device_fp16", "device": "cuda:0", "dtype": "torch.float16", "shape": [V.N, 1536], "verified": True},
           "trained_sha256": V.state_sha(snap[60000]), "loaded_modules": {"verified_frozen_runtime": True, "all_basemap_under_root": True}}
    ts=man['train_stats'];ts.update(attempted_steps=V.DOSE,amp_skips=0,nonfinite_skips=0,successful_positive=V.DOSE*100,successful_noise=V.DOSE*900,attempted_positive=V.DOSE*100,attempted_noise=V.DOSE*900)
    final_path=td/arm/'ckpts/ckpt-step60000.pt';final=torch.load(final_path,weights_only=False);final['train_stats']=dict(ts);torch.save(final,final_path)
    (td / f"admission-{arm}.json").write_text(json.dumps({"arm": arm, "identity": ident, "warm_init_sha256": V.INIT_SHA, "init_payload_sha256": V.INIT_SHA}))
    os.utime(td / f"admission-{arm}.json", (0, 0)); (td / f"manifest-{arm}.json").write_text(json.dumps(man))
    return td


def expect_fail(base, arm, mutate, label):
    td = Path(tempfile.mkdtemp(dir=base)); shutil.copytree(Path(base) / arm, td / arm)
    for f in (f"coords-{arm}.npy", f"model-{arm}.pt", f"manifest-{arm}.json", f"admission-{arm}.json"):
        shutil.copy(Path(base) / f, td / f)
    os.utime(td / f"admission-{arm}.json", (0, 0)); mutate(td)
    try:
        V.strict_validate_arm(arm, ROOT, base=td)
    except Exception as e:
        return f"FAIL-CLOSED ok: {label} -> {type(e).__name__}"
    raise SystemExit(f"GUARD DID NOT FIRE: {label}")


def _man(td, arm, **kw):
    m = json.loads((td / f"manifest-{arm}.json").read_text()); m.update(kw); (td / f"manifest-{arm}.json").write_text(json.dumps(m))
def _edit_ck(td, arm, name, fn):
    p = td / arm / "ckpts" / name; ck = torch.load(p, map_location="cpu", weights_only=False); fn(ck); torch.save(ck, p)


def main():
    base = tempfile.mkdtemp(prefix="card034-selftest-", dir="/tmp"); results = []
    for arm in V.ARMS:
        build(base, arm); assert V.strict_validate_arm(arm, ROOT, base=base)["PASS"]; results.append(f"PASS {arm} valid")
    A = "grouped_umap"
    results.append(expect_fail(base, A, lambda td: _man(td, A, executed_steps=59999), "wrong dose"))
    results.append(expect_fail(base, A, lambda td: _man(td, A, warm_init_sha256="0" * 16), "warm-init drift"))
    results.append(expect_fail(base, A, lambda td: _man(td, A, init_payload_sha256="0" * 16), "init-payload hash drift"))
    results.append(expect_fail(base, A, lambda td: _man(td, A, weight_decay=0.0), "wrong weight decay"))
    def _idmut(td):
        m = json.loads((td / f"manifest-{A}.json").read_text()); m["identity"] = dict(m["identity"]); m["identity"]["dose"] = 1; (td / f"manifest-{A}.json").write_text(json.dumps(m))
    results.append(expect_fail(base, A, _idmut, "identity drift"))
    results.append(expect_fail(base, A, lambda td: torch.save({"model_state_dict": _sd(7)}, td / f"model-{A}.pt"), "endpoint != 60K snapshot"))
    results.append(expect_fail(base, A, lambda td: os.utime(td / f"admission-{A}.json", None), "stale snapshot"))
    results.append(expect_fail(base, A, lambda td: (td / f"{A}/ckpts/ckpt-step40000.pt").write_bytes(b"garbage"), "corrupt step ckpt"))
    results.append(expect_fail(base, A, lambda td: (td / f"{A}/ckpts/ckpt-step60000.pt").unlink(), "missing 60K ckpt"))
    results.append(expect_fail(base, A, lambda td: (td / f"{A}/ckpts/ckpt-epoch1.pt").unlink(), "missing epoch ckpt"))
    def _exp(td):
        m = json.loads((td / f"manifest-{A}.json").read_text()); m["train_stats"]["exposure_probes"][1]["noise_per_pos"] = 8; (td / f"manifest-{A}.json").write_text(json.dumps(m))
    results.append(expect_fail(base, A, _exp, "successful-probe 9:1 violated"))
    def _att(td):
        m = json.loads((td / f"manifest-{A}.json").read_text()); m["train_stats"]["attempted_probes"][1]["noise_per_pos"] = 8; (td / f"manifest-{A}.json").write_text(json.dumps(m))
    results.append(expect_fail(base, A, _att, "attempted-probe 9:1 violated"))
    results.append(expect_fail(base, A, lambda td: _man(td, A, pipeline_receipt={"x_residency": "host_int8", "device": "cpu", "dtype": "torch.float32", "shape": [V.N, 1536], "verified": True}), "wrong pipeline receipt"))
    # DEEP-STATE faults
    results.append(expect_fail(base, A, lambda td: _edit_ck(td, A, "ckpt-step40000.pt", lambda c: c["train_stats"].__setitem__("positive_lr_optimizer_steps", 0)), "stale ckpt stats"))
    results.append(expect_fail(base, A, lambda td: _edit_ck(td, A, "ckpt-step40000.pt", lambda c: c["optimizer"]["state"][0].__setitem__("step", torch.tensor(123.0))), "wrong Adam step counter"))
    results.append(expect_fail(base, A, lambda td: _edit_ck(td, A, "ckpt-step40000.pt", lambda c: c["optimizer"]["state"][0].__setitem__("exp_avg", torch.tensor([float("inf")] * 4))), "nonfinite Adam moment"))
    results.append(expect_fail(base, A, lambda td: _edit_ck(td, A, "ckpt-step40000.pt", lambda c: c.pop("scaler")), "missing scaler"))
    results.append(expect_fail(base, A, lambda td: _edit_ck(td, A, "ckpt-step40000.pt", lambda c: c["sampler_state"].__setitem__("perm", np.zeros(10, np.int64))), "malformed sampler perm"))
    results.append(expect_fail(base, A, lambda td: _edit_ck(td, A, "ckpt-step40000.pt", lambda c: c["optimizer"]["param_groups"][0].__setitem__("weight_decay", 0.0)), "ckpt Adam wd drift"))
    NC = "grouped_nce"
    results.append(expect_fail(base, NC, lambda td: _man(td, NC, final_beta=0.0), "nce scalar never moved"))
    results.append(expect_fail(base, NC, lambda td: _edit_ck(td, NC, "ckpt-step40000.pt", lambda c: c.__setitem__("beta", None)), "nce ckpt missing scalar"))

    for label,mut in [
        ('wrong moment shape',lambda c:c['optimizer']['state'][0].__setitem__('exp_avg',torch.zeros(1))),
        ('malformed CPU RNG',lambda c:c.__setitem__('torch_rng',torch.zeros(16,dtype=torch.uint8))),
        ('wrong permutation length',lambda c:c['sampler_state'].__setitem__('perm',np.arange(63))),
        ('invalid generator content',lambda c:c['sampler_state']['noise_gen'].__setitem__('state',{})),
        ('old invented scaler key',lambda c:c['scaler'].__setitem__('growth_tracker',c['scaler'].pop('_growth_tracker'))),
        ('payload coefficient drift',lambda c:c.__setitem__('coeff',3.0)),
    ]:
        results.append(expect_fail(base,A,lambda td,m=mut:_edit_ck(td,A,'ckpt-step40000.pt',m),label))
    shutil.rmtree(base, ignore_errors=True)
    out = {"schema": "card034-selftest-2026-09-12", "PASS": True, "n_cases": len(results), "results": results}
    (V.OC / "card034-selftest.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

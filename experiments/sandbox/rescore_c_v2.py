"""C grouped-quality RE-VERDICT on seal-v2 (owner via overseer 2026-09-08). CPU. Instrument correction, SAME gates.
Projects the persisted C heads (uniform-neg, grouped-neg) through the common evaluator against BOTH seal-v1 (to
validate the heads reproduce the original scores) and seal-v2 (the corrected instrument: every cohort ≥1060 val).
Applies the UNCHANGED gates (Δrecall within 0.005 AND no cohort loss >0.01). Records the full honesty trail.
Usage: rescore_c_v2.py
"""
import json, os
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); CDIR = SB / "exp-c-quality"


def score_on(seal_dir, head, label):
    os.environ["EVAL_SEAL_DIR"] = seal_dir
    import importlib, eval_common; importlib.reload(eval_common)             # re-read EVAL_SEAL_DIR
    seal = eval_common._load_seal()
    rc = eval_common._project(str(head), seal["ref_hd"], device="cpu")
    vc = eval_common._project(str(head), seal["val_hd"], device="cpu")
    return eval_common.score(rc, vc, seal, label)


def main():
    heads = {"uniform-neg": CDIR / "uniform-neg" / "model.pt", "grouped-neg": CDIR / "grouped-neg" / "model.pt"}
    for a, h in heads.items():
        if not h.exists(): raise SystemExit(f"missing head {h} — C re-run must persist heads first")
    v1prev = json.loads((CDIR / "result.json").read_text()) if (CDIR / "result.json").exists() else None

    out = {"schema": "exp-c-quality-reverdict-v2-2026-09-08",
           "trail": "C mechanical-STOP on v1 (worst-cohort 0.0357 @ diffusion n=99 ≈1 bootstrap SD) -> instrument "
                    "incapacity (99-query cohort can't resolve a 0.01 gate) -> seal-v2 (val ≥1060/source, gates UNCHANGED) "
                    "-> re-run C with head persistence (deterministic, forced by v1 non-persistence) -> this rescore."}
    for seal_dir, tag in [("/data2/monet/eval-common", "v1_validation"), ("/data2/monet/eval-common-v2", "v2_reverdict")]:
        uni = score_on(seal_dir, heads["uniform-neg"], f"uniform-{tag}"); grp = score_on(seal_dir, heads["grouped-neg"], f"grouped-{tag}")
        d2000 = round(grp["recall@k15_B2000"]["micro"] - uni["recall@k15_B2000"]["micro"], 4)
        cohort_losses = {r: round(uni["recall@k15_B2000"]["per_source"][r] - grp["recall@k15_B2000"]["per_source"][r], 4)
                         for r in uni["recall@k15_B2000"]["per_source"]}
        worst = max(cohort_losses.values()); worst_c = max(cohort_losses, key=cohort_losses.get)
        passes = abs(d2000) <= 0.005 and worst <= 0.01
        out[tag] = {"uniform_B2000": uni["recall@k15_B2000"]["micro"], "grouped_B2000": grp["recall@k15_B2000"]["micro"],
                    "delta_recall_B2000": d2000, "worst_cohort_loss": worst, "worst_cohort": worst_c,
                    "cohort_losses": cohort_losses, "gate_pass": passes}
        print(f"[{tag}] uni {uni['recall@k15_B2000']['micro']} grp {grp['recall@k15_B2000']['micro']} Δ{d2000} worst-loss {worst}@{worst_c} pass {passes}", flush=True)
    v2 = out["v2_reverdict"]
    out["verdict"] = ("PROMOTE: grouped preserves quality on the corrected instrument (Δ %.4f within 0.005, worst-cohort "
                      "loss %.4f ≤0.01 on ≥1060-query cohorts) — the 1.42x throughput compounds into every run"
                      % (v2["delta_recall_B2000"], v2["worst_cohort_loss"])) if v2["gate_pass"] else \
                     ("STOP (final): grouped trips a ≥1060-query cohort by >0.01 (worst-cohort loss %.4f @ %s) — real harm, no further appeals"
                      % (v2["worst_cohort_loss"], v2["worst_cohort"]))
    if v1prev: out["v1_original_note"] = "first C run (in-process, v1): Δ +0.0029, worst-cohort 0.0357 @ diffusion n=99"
    (CDIR / "reverdict-v2.json").write_text(json.dumps(out, indent=1)); print("VERDICT:", out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

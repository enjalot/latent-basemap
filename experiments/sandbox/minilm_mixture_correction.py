"""Mixture-pilot verdict CORRECTION (owner plan item 5, 2026-09-08). The original pilot scored each arm against
its OWN reference population; the plan requires a FIXED common reference + IDENTICAL query IDs before the verdict
is quoted. This builds ONE common base reference + query set (original MiniLM-384, fixed seed) and projects the
SAME rows through BOTH the mix head and the base head, scoring each with the common evaluator's recall@k15 (fixed
B) against IDENTICAL original-384 truth. base_displacement is then a fair per-register comparison. CPU (GPU untouched).

Reference + queries are random base rows (fineweb/redpajama/pile, equal thirds) — drawn from ~400M base rows so
they are ~certainly held out of both 2M training draws (out-of-sample for both heads). register = provenance cohort.
Writes SANDBOX/minilm-mixture-correction.json. Usage: minilm_mixture_correction.py [N_REF=50000] [N_Q=5000].
"""
import json, sys, time
from pathlib import Path
import numpy as np

SB = Path("/data/latent-basemap/sandbox"); DIM = 384; K = 15


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    n_ref = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from minilm_mix_draw import Corpus, BASE
    import eval_common, faiss
    rng = np.random.default_rng(2026)

    bases = list(BASE); per_ref = n_ref // len(bases); per_q = n_q // len(bases)
    ref_parts, q_parts, ref_reg, q_reg = [], [], [], []
    for b in bases:
        c = Corpus(b, True); perm = rng.permutation(c.n)
        rsel = np.sort(perm[:per_ref]); qsel = np.sort(perm[per_ref:per_ref + per_q])   # disjoint ref/query per corpus
        ref_parts.append(_norm(c.gather(rsel))); q_parts.append(_norm(c.gather(qsel)))
        ref_reg += [b] * rsel.size; q_reg += [b] * qsel.size
        print(f"[mix-corr] {b}: ref {rsel.size} query {qsel.size} (of {c.n:,})", flush=True)
    ref_hd = np.concatenate(ref_parts); val_hd = np.concatenate(q_parts)
    val_source = np.array(q_reg, dtype=object)

    t0 = time.time(); hdx = faiss.IndexFlatIP(DIM); hdx.add(np.ascontiguousarray(ref_hd))
    _, truth_val = hdx.search(np.ascontiguousarray(val_hd), K)                # identical original-384 truth for BOTH heads
    diag_idx = np.sort(rng.choice(ref_hd.shape[0], min(5000, ref_hd.shape[0]), replace=False)).astype(np.int32)
    seal = {"ref_hd": ref_hd, "val_hd": val_hd, "truth_val": truth_val.astype(np.int32),
            "val_source": val_source, "diag_idx": diag_idx}
    print(f"[mix-corr] common ref {ref_hd.shape} query {val_hd.shape} truth {time.time()-t0:.0f}s", flush=True)

    scores = {}
    for arm, ckpt in [("mix", SB / "minilm-mixpilot-2m/champion-bs16k/model.pt"),
                      ("base", SB / "minilm-base-2m/champion-bs16k/model.pt")]:
        rc = eval_common._project(str(ckpt), ref_hd, device="cpu"); vc = eval_common._project(str(ckpt), val_hd, device="cpu")
        scores[arm] = eval_common.score(rc, vc, seal, label=f"minilm-{arm}")
        print(f"[mix-corr] {arm}: B250 {scores[arm]['recall@k15_B250']['micro']} B2000 {scores[arm]['recall@k15_B2000']['micro']}", flush=True)

    disp = {}
    for B in ("recall@k15_B250", "recall@k15_B2000"):
        bc = scores["base"][B]["per_source"]; mc = scores["mix"][B]["per_source"]
        disp[B] = {r: round(bc[r] - mc[r], 4) for r in bc}                    # base − mix on IDENTICAL ref+query+truth
    allv = [v for d in disp.values() for v in d.values()]
    harmful = max(allv)                                                      # base − mix > 0 = mixing HURT base reception
    beneficial = min(allv)                                                   # < 0 = mixing HELPED
    verdict = ("base NOT harmfully displaced under a fixed common reference (max harmful Δ %.4f ≤ 0.02; largest "
               "swing %.4f is beneficial — mixing improved that register)" % (harmful, beneficial)) if harmful <= 0.02 \
              else "base DISPLACED (max harmful base−mix Δ %.4f > 0.02)" % harmful
    out = {"schema": "minilm-mixture-correction-2026-09-08", "n_ref": int(ref_hd.shape[0]), "n_query": int(val_hd.shape[0]),
           "protocol": "FIXED common base reference + IDENTICAL query IDs (original MiniLM-384 truth), both heads scored "
                       "by the common evaluator; supersedes the per-arm-reference 0.0041 verdict",
           "base_displacement_base_minus_mix": disp, "note_sign": "positive = mixing HURT base reception (displacement); negative = mixing HELPED",
           "max_harmful_displacement": round(harmful, 4), "max_beneficial_swing": round(beneficial, 4),
           "verdict": verdict, "mix": scores["mix"], "base": scores["base"]}
    (SB / "minilm-mixture-correction.json").write_text(json.dumps(out, indent=1))
    print(f"[mix-corr] DONE max harmful Δ {harmful:.4f} / beneficial {beneficial:.4f} -> {out['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

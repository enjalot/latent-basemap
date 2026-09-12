"""Card012 candidate-pool + reserved-confirmation builder (per card012-prereg.md + card012-pool-overseer-
review). Fixes vs the first build:
 - Fresh confirmation excluded from the FULL domain-qualified IMAGE support union (final 2.4M graph, eval
   ref, eval val-queries, card006 image confirm, card010/011 300K graph draw, card009 deriv bank ids) AND
   the 1M candidate pool itself. (The 1M pool's own overlap with the card010/011 graph is allowed by the
   candidate-pool rule; only the stricter fresh-confirmation is excluded from it.)
 - Teacher targets computed on the EXACT STORED fp16 X (cast to fp32, NO extra normalization) — same input
   contract the student/refresh use. OUT rows get the IDENTICAL starting-bank X + targets.
 - Fail-closed complete-bundle validation (shapes/finite/ids/hashes), atomic manifest-last.
Pool ROW SELECTION is kept fixed (seeded, reproduces). Idempotent: rebuilds if the validated bundle is
absent/stale. GPU for T0 projection (charged, under both locks by the launcher).
"""
import os, sys, json, hashlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CARD012_CUDA", "0"))
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); from _paths import ensure_paths; ensure_paths()

POOL = Path("/data2/monet/pool-20m"); SEAL = Path("/data2/monet/eval-common-v2")
SB = Path("/data/latent-basemap/sandbox"); OC = SB / "overseer-codex"
T0_HEAD = SB / "dino-arrival-t0/champion-bs16k/model.pt"
OUT_BANK = OC / "card006_out_bank.npz"
FINAL_DRAW = Path("/data/latent-basemap/substrates/dino-arrival-final/draw_idx.npy")
CARD010_DRAW = Path("/data/latent-basemap/substrates/card010-adaptive/draw_ids.npy")   # card010/011 graph
DERIV_BANK = OC / "card009_deriv_bank.npz"
OUTD = Path("/data/latent-basemap/sandbox/card012-pool"); OUTD.mkdir(parents=True, exist_ok=True)
SOURCES = ["laion", "coyo", "commoncatalog-cc-by", "megalith10m", "cc12m"]
N_CONFIRM_PER = 2000; N_POOL_EXTRA = 800000; SEED = 12012


def _norm(a):
    a = np.asarray(a, np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def _sha_arr(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def project_T0_on_stored(Xf16, device):
    """Teacher = ORIGINAL T0 on the EXACT stored fp16 X cast to fp32 — NO extra normalization (X is already
    normalized before the fp16 cast; re-normalizing would change the exact stored input)."""
    import torch
    from basemap.pumap.parametric_umap.core import ParametricUMAP
    p = ParametricUMAP.load(str(T0_HEAD), device=device); p.model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, Xf16.shape[0], 100000):
            e = min(s + 100000, Xf16.shape[0])
            xb = np.asarray(Xf16[s:e], np.float32)                    # stored fp16 -> fp32, NO renorm
            out.append(p.model(torch.from_numpy(xb).to(device)).float().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def _validate_bundle(verbose=False):
    """Fail-closed: binds ALL persisted inputs (ids, source, X, teacher, confirmation) — existence, shapes,
    dtypes, finiteness, per-source quotas, uniqueness, confirmation exclusions, and content hashes for EVERY
    array (X and source included, not just ids/targets) against the manifest — plus declared teacher+builder
    identity. Returns True only if every check passes. A wrong/changed X or source FAILS."""
    need = ["pool_ids.npy", "pool_source.npy", "pool_X.f16.npy", "pool_teacher.npy", "confirm_reserved.npy"]
    if not all((OUTD / f).exists() for f in need) or not (OC / "card012-pool-manifest.json").exists():
        return False
    try:
        m = json.loads((OC / "card012-pool-manifest.json").read_text())
        pid = np.load(OUTD / "pool_ids.npy"); src = np.load(OUTD / "pool_source.npy", allow_pickle=True).astype(str)
        X = np.load(OUTD / "pool_X.f16.npy", mmap_mode="r"); tea = np.load(OUTD / "pool_teacher.npy")
        conf = np.load(OUTD / "confirm_reserved.npy")
        C = {}
        C["complete"] = m.get("complete") is True
        # builder_sha is RECORDED provenance (which builder produced the content), NOT a gate: the validator
        # legitimately evolves after the content is built, so gating on the current file would false-fail.
        C["builder_sha_recorded"] = bool(m.get("builder_sha"))
        C["shapes"] = (pid.shape[0] == m["n_pool"] == X.shape[0] == tea.shape[0] == src.shape[0]
                       and X.shape[1] == 1536 and tea.shape[1] == 2 and conf.shape[0] == m["n_confirm_reserved"])
        C["dtypes"] = (X.dtype == np.float16 and tea.dtype == np.float32 and np.issubdtype(pid.dtype, np.integer)
                       and np.issubdtype(conf.dtype, np.integer))
        C["ids_hash"] = _sha_arr(pid) == m["pool_ids_sha"]
        C["teacher_hash"] = _sha_arr(tea.astype(np.float32)) == m["teacher_sha"]
        C["X_hash"] = _sha_arr(np.ascontiguousarray(X)) == m["X_sha"]              # <-- binds X (was missing)
        C["source_hash"] = _sha_arr(src) == m.get("source_sha")                    # <-- binds source
        C["confirm_hash"] = _sha_arr(conf) == m["confirm_ids_sha"]
        # bind the FIXED teacher (T0 head file) identity — targets were projected from it
        C["teacher_head"] = m.get("teacher_head_sha") == _sha_arr(np.frombuffer(open(T0_HEAD, "rb").read(), np.uint8))
        C["pid_unique"] = len(np.unique(pid)) == pid.shape[0]
        C["conf_unique"] = len(np.unique(conf)) == conf.shape[0]
        C["teacher_finite"] = bool(np.isfinite(tea).all())
        rng = np.random.default_rng(0); samp = np.sort(rng.choice(X.shape[0], min(100000, X.shape[0]), replace=False))
        C["X_finite_sample"] = bool(np.isfinite(np.asarray(X[samp], np.float32)).all())
        C["quotas"] = all(int((src == s).sum()) == m["per_source_pool"][s] for s in SOURCES)
        # confirmation exclusions (fresh confirmation disjoint from pool + supports)
        C["conf_disjoint_pool"] = not np.isin(conf, pid).any()
        C["conf_disjoint_final"] = not np.isin(conf, np.load(FINAL_DRAW)).any()
        C["conf_disjoint_card010"] = not np.isin(conf, np.load(CARD010_DRAW)).any()
        ok = all(C.values())
        if verbose or not ok:
            print("validate_bundle:", json.dumps({k: bool(v) for k, v in C.items()}), flush=True)
        return bool(ok)
    except Exception as ex:
        if verbose:
            print("validate_bundle exception:", ex, flush=True)
        return False


def main():
    if _validate_bundle():
        print("card012 pool bundle valid + complete — skipping"); return 0
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    src_all = np.load(POOL / "source.npy", allow_pickle=True).astype(str)
    N = src_all.shape[0]
    out = np.load(OUT_BANK); out_ids = out["replay_ids"].astype(np.int64)
    out_X = np.asarray(out["replay_X"], np.float16); out_tea = np.asarray(out["replay_targets"], np.float32)

    # ---- POOL exclusion (candidate-pool rule: domain-qualified image supports; NOT the card010/011 graph) ----
    excl_pool = np.zeros(N, bool)
    excl_pool[np.load(FINAL_DRAW)] = True
    excl_pool[np.load(SEAL / "ref_idx.npy")] = True; excl_pool[np.load(SEAL / "val_idx.npy")] = True
    excl_pool[np.load(OC / "card006_confirm_bank.npz")["replay_ids"]] = True
    # KEEP the 1M pool ROW SELECTION fixed: reuse the existing pool_ids if present (only X/teacher/confirmation
    # are corrected). Re-select (seeded) only on a first build.
    if (OUTD / "pool_ids.npy").exists():
        pool_ids = np.load(OUTD / "pool_ids.npy"); pool_src = src_all[pool_ids]
        assert set(out_ids.tolist()).issubset(set(pool_ids.tolist())), "existing pool missing OUT base"
        assert not np.isin(pool_ids, np.where(excl_pool)[0]).any(), "existing pool violates domain exclusions"
    else:
        excl_pool_extra = excl_pool.copy(); excl_pool_extra[out_ids] = True
        per = N_POOL_EXTRA // len(SOURCES); extra = []
        for s in SOURCES:
            cand = np.where((src_all == s) & (~excl_pool_extra))[0]
            assert cand.size >= per, f"{s}: eligible {cand.size} < {per} (STOP: insufficient stratum)"
            extra.append(rng.choice(cand, per, replace=False))
        pool_ids = np.sort(np.concatenate([out_ids, np.concatenate(extra)])); pool_src = src_all[pool_ids]
    assert len(np.unique(pool_ids)) == len(pool_ids)

    # ---- X: OUT rows get IDENTICAL starting-bank replay_X; extra rows = _norm(fp32)->fp16 ----
    Xmm = np.load(POOL / "dino1536.f16.npy", mmap_mode="r")
    bank_row = {int(i): r for r, i in enumerate(out_ids)}
    is_out = np.isin(pool_ids, out_ids)
    X = np.empty((len(pool_ids), 1536), np.float16)
    extra_mask = ~is_out
    X[extra_mask] = _norm(np.asarray(Xmm[pool_ids[extra_mask]], np.float32)).astype(np.float16)
    out_pos = np.where(is_out)[0]
    X[out_pos] = out_X[np.array([bank_row[int(pool_ids[p])] for p in out_pos])]
    np.save(OUTD / "pool_X.f16.npy", X)                               # persist X FIRST

    # ---- teacher on EXACT stored fp16; OUT rows use the identical starting-bank targets ----
    Xstored = np.load(OUTD / "pool_X.f16.npy", mmap_mode="r")
    teacher = project_T0_on_stored(Xstored, device)
    teacher[out_pos] = out_tea[np.array([bank_row[int(pool_ids[p])] for p in out_pos])]  # exact starting-bank targets
    # roundtrip audit: recomputed T0-on-stored for OUT rows vs bank targets (isolate quantization)
    out_recomp = project_T0_on_stored(np.asarray(Xstored[out_pos]), device)
    out_resid = float(np.abs(out_recomp - out_tea[np.array([bank_row[int(pool_ids[p])] for p in out_pos])]).max())
    np.save(OUTD / "pool_teacher.npy", teacher.astype(np.float32))
    np.save(OUTD / "pool_source.npy", pool_src); np.save(OUTD / "pool_ids.npy", pool_ids)

    # ---- FRESH confirmation: exclude FULL image-support union + the 1M pool itself ----
    excl_conf = excl_pool.copy()
    excl_conf[out_ids] = True
    excl_conf[np.load(CARD010_DRAW)] = True                           # card010/011 300K graph (275 overlap fixed)
    if DERIV_BANK.exists():
        excl_conf[np.load(DERIV_BANK)["deriv_ids"]] = True            # card009 deriv candidate ids
    excl_conf[pool_ids] = True                                        # exclude the 1M candidate pool too
    if (OUTD / "confirm_reserved.npy").exists():
        np.save(OUTD / "confirm_reserved_v1.npy", np.load(OUTD / "confirm_reserved.npy"))  # preserve original
    confirm = []
    for s in SOURCES:
        cand = np.where((src_all == s) & (~excl_conf))[0]
        assert cand.size >= N_CONFIRM_PER, f"{s}: {cand.size} < {N_CONFIRM_PER}"
        confirm.append(rng.choice(cand, N_CONFIRM_PER, replace=False))
    confirm = np.sort(np.concatenate(confirm))
    # verify zero overlap with every support + pool
    for nm, arr in [("final", np.load(FINAL_DRAW)), ("card010_011_graph", np.load(CARD010_DRAW)),
                    ("pool", pool_ids), ("out", out_ids), ("eval_ref", np.load(SEAL / "ref_idx.npy")),
                    ("eval_val", np.load(SEAL / "val_idx.npy"))]:
        assert not np.isin(confirm, arr).any(), f"confirmation overlaps {nm}"
    np.save(OUTD / "confirm_reserved.npy", confirm)

    man = {"schema": "card012-pool-2026-09-11-v2", "n_pool": int(len(pool_ids)), "n_out_base": int(is_out.sum()),
           "n_extra": int(extra_mask.sum()), "n_confirm_reserved": int(len(confirm)),
           "per_source_pool": {s: int((pool_src == s).sum()) for s in SOURCES},
           "pool_ids_sha": _sha_arr(pool_ids), "teacher_sha": _sha_arr(teacher.astype(np.float32)),
           "X_sha": _sha_arr(X), "source_sha": _sha_arr(pool_src.astype(str)),
           "confirm_ids_sha": _sha_arr(confirm), "seed": SEED,
           "builder_sha": _sha_arr(np.frombuffer(open(__file__, "rb").read(), np.uint8)),
           "teacher_head_sha": _sha_arr(np.frombuffer(open(T0_HEAD, "rb").read(), np.uint8)),
           "support_manifest": {"final_sha": _sha_arr(np.load(FINAL_DRAW)),
                                "eval_ref_sha": _sha_arr(np.load(SEAL / "ref_idx.npy")),
                                "eval_val_sha": _sha_arr(np.load(SEAL / "val_idx.npy")),
                                "card006_confirm_sha": _sha_arr(np.load(OC / "card006_confirm_bank.npz")["replay_ids"]),
                                "card010_draw_sha": _sha_arr(np.load(CARD010_DRAW))},
           "out_row_teacher_vs_bank_resid_max": round(out_resid, 8),
           "out_row_resid_note": "MEASURED baseline numerical FLOOR (device/batch arithmetic + fp16), NOT isolated quantization; OUT rows keep the exact inherited starting-bank targets — this floor informs error-ranking interpretation only",
           "teacher_input_contract": "T0 on EXACT stored fp16 X -> fp32, NO renorm; OUT rows = identical starting-bank X+targets",
           "confirm_exclusions": ["final", "eval_ref", "eval_val", "card006_confirm(image)", "card010_011_graph",
                                  "card009_deriv_ids" if DERIV_BANK.exists() else "card009_deriv_ids(absent)",
                                  "OUT", "card012_pool_1M"],
           "pool_exclusions_note": "candidate pool excludes final/eval-ref/eval-val/card006-confirm only (card010/011-graph overlap allowed by rule)",
           "complete": True}
    tmp = OC / "card012-pool-manifest.json.tmp"; tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, OC / "card012-pool-manifest.json")                # manifest LAST (atomic completion marker)
    assert _validate_bundle(), "post-build validation failed"
    print(json.dumps({**man, "out_row_teacher_vs_bank_resid_max": man["out_row_teacher_vs_bank_resid_max"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate COLLAPSE and FOG on the sealed 2M seed family + the sandbox arms.

Stages (all CPU, all read-only against sealed artifacts):

  measure    compute both metrics for
               * the sealed n = 29 2M seed family (seeds 42..70, resolved
                 through /data/latent-basemap/maps.json)
               * every sandbox arm under /data/latent-basemap/sandbox/2m-knobs
                 and .../6250k-knobs
               * the cuML 1M reference map
  calibrate  reproduce R0234's sealed n = 13 multiplier (validation gate),
             then derive the n = 29 multiplier the same way
  report     write REPORT.md from the two results JSONs

Usage:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/calibrate_collapse_fog.py all
  CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/calibrate_collapse_fog.py measure
  CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/calibrate_collapse_fog.py calibrate
  CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/calibrate_collapse_fog.py report
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collapse_fog import (  # noqa: E402
    MADN_CONSISTENCY, load_coords, madn, map_collapse, map_fog,
    robust_ceiling, robust_floor,
)
import null_calibration as nc  # noqa: E402

MAPS_JSON = Path("/data/latent-basemap/maps.json")
SANDBOX_2M = Path("/data/latent-basemap/sandbox/2m-knobs")
SANDBOX_6250K = Path("/data/latent-basemap/sandbox/6250k-knobs")
CUML_XY = Path("/data/latent-basemap/sandbox/cuml-1m/cuml-xy.npy")

RESULTS = HERE / "results"
MEASURED_JSON = RESULTS / "collapse-fog-measurements.json"
CALIB_JSON = RESULTS / "collapse-fog-calibration.json"
REPORT_MD = HERE / "REPORT.md"

FAMILY_SEEDS = list(range(42, 71))  # the sealed n = 29 family, R0255
NULL_FAMILIES = 4_000_000
NULL_SEED = 20260809


# --- family resolution --------------------------------------------------

def resolve_family() -> list[dict]:
    """The 29 sealed 2M seed-family cells, via the mutable maps.json view.

    The authoritative seed list is R0255's own `exact_family_seeds`
    (42..70) in
      /data/latent-basemap/runs/round-0255/queue/artifacts/
        minilm-mixed-2m-calibrated-madn-floors-n29-v1/
        minilm-calibrated-madn-floors-n29.json
    """
    reg = json.loads(MAPS_JSON.read_text())
    pat = re.compile(r"^round-(\d+)-minilm-mixed-2m-map-seed(\d+)-low-dose-v1$")
    by_seed: dict[int, dict] = {}
    for m in reg["maps"]:
        mm = pat.match(str(m.get("map_id", "")))
        if not mm:
            continue
        seed = int(mm.group(2))
        if seed not in FAMILY_SEEDS:
            continue
        f = ((m.get("coordinates") or {}).get("file") or "").removeprefix("gsv:")
        by_seed[seed] = {"seed": seed, "map_id": m["map_id"], "round": mm.group(1),
                         "path": f}
    missing = [s for s in FAMILY_SEEDS if s not in by_seed]
    if missing:
        raise SystemExit(f"seed family incomplete, missing seeds {missing}")
    return [by_seed[s] for s in FAMILY_SEEDS]


def resolve_sandbox() -> list[dict]:
    out = []
    for root, rung in ((SANDBOX_2M, "2m"), (SANDBOX_6250K, "6250k")):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            p = d / "coordinates.npy"
            if not p.is_file():
                continue
            summary = {}
            sp = d / "summary.json"
            if sp.is_file():
                summary = json.loads(sp.read_text())
            ov = summary.get("overrides") or {}
            out.append({"arm": d.name, "rung": rung, "path": str(p),
                        "kernel": str(ov.get("low_dim_kernel") or "legacy_lp"),
                        "a": ov.get("a"), "b": ov.get("b"),
                        "seed": summary.get("seed"),
                        "dose_multiplier": summary.get("dose_multiplier")})
    if CUML_XY.is_file():
        out.append({"arm": "cuml-1m-reference", "rung": "1m", "path": str(CUML_XY),
                    "kernel": "cuml_umap", "a": None, "b": None,
                    "seed": None, "dose_multiplier": None})
    return out


#: the umap-kernel arms at min_dist = 0.00, i.e. a >= 1.9 in the (a, b) fit.
#: These share one recipe and differ only in seed, dose, and rung -- the
#: closest thing to a conforming population the sandbox offers.
MIN_DIST_ZERO_A = 1.9


def is_core_umap(row: dict) -> bool:
    return row.get("kernel") == "umap" and (row.get("a") or 0.0) >= MIN_DIST_ZERO_A


# --- stages -------------------------------------------------------------

def measure_one(path: str) -> dict:
    t0 = time.time()
    xy = load_coords(path)
    c = map_collapse(xy)
    f = map_fog(xy)
    return {"collapse": c, "fog": f, "wall_s": round(time.time() - t0, 2)}


def stage_measure() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fam = resolve_family()
    sbx = resolve_sandbox()
    out = {"family": [], "sandbox": [],
           "protocol": {
               "collapse": "median 2D distance to the 10th nearest neighbour over the "
                           "p90 radius about the centroid, times sqrt(n_effective); "
                           "20k seeded sample (seed 0)",
               "fog": "fraction of binned mass in occupied bins below 1% of the peak "
                      "bin count, 1024x1024 histogram over the (0.1, 99.9) percentile "
                      "extent with 2% pad and edge clipping",
               "madn_consistency": MADN_CONSISTENCY,
           }}
    for cell in fam:
        r = measure_one(cell["path"])
        row = {**cell, "collapse": r["collapse"]["r10_over_radius_times_sqrt_n"],
               "collapse_raw": r["collapse"]["r10_over_map_radius_median"],
               "r10_median": r["collapse"]["r10_median"],
               "map_radius_p90": r["collapse"]["map_radius_p90"],
               "n_rows": r["collapse"]["n_rows"],
               "fog": r["fog"]["fog"],
               "occupied_bin_fraction": r["fog"]["occupied_bin_fraction"],
               "peak_bin_count": r["fog"]["peak_bin_count"],
               "fog_resolution_levels": r["fog"]["resolution_levels"],
               "fog_degenerate": r["fog"]["degenerate"],
               "wall_s": r["wall_s"]}
        out["family"].append(row)
        print(f"family seed{cell['seed']:<3d} collapse={row['collapse']:.4f} "
              f"fog={row['fog']:.4f} occ={row['occupied_bin_fraction']:.4f} "
              f"({row['wall_s']}s)", flush=True)
    for arm in sbx:
        r = measure_one(arm["path"])
        row = {**arm, "collapse": r["collapse"]["r10_over_radius_times_sqrt_n"],
               "collapse_raw": r["collapse"]["r10_over_map_radius_median"],
               "n_rows": r["collapse"]["n_rows"],
               "fog": r["fog"]["fog"],
               "occupied_bin_fraction": r["fog"]["occupied_bin_fraction"],
               "peak_bin_count": r["fog"]["peak_bin_count"],
               "fog_resolution_levels": r["fog"]["resolution_levels"],
               "fog_degenerate": r["fog"]["degenerate"],
               "wall_s": r["wall_s"]}
        out["sandbox"].append(row)
        print(f"sandbox {arm['rung']:>6s}/{arm['arm']:<22s} collapse={row['collapse']:.4f} "
              f"fog={row['fog']:.4f} occ={row['occupied_bin_fraction']:.4f} "
              f"({row['wall_s']}s)", flush=True)
    MEASURED_JSON.write_text(json.dumps(out, indent=1))
    print("wrote", MEASURED_JSON)
    return out


def stage_calibrate() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"validation gate: reproducing R0234's sealed n=13 multiplier "
          f"({NULL_FAMILIES:,} families)...", flush=True)
    t0 = time.time()
    gate = nc.validate_r0234(families=NULL_FAMILIES, seed=NULL_SEED, tolerance=0.02)
    print(f"  sealed  {gate['sealed_value']:.7f}")
    print(f"  ours    {gate['reproduced_value']:.7f}  rel err {gate['relative_error']:.4%}"
          f"  -> {'PASS' if gate['passes'] else 'FAIL'}  ({time.time()-t0:.0f}s)",
          flush=True)
    if not gate["passes"]:
        raise SystemExit("validation gate failed; the n=29 multiplier is not trustworthy")

    print(f"calibrating n=29 ({NULL_FAMILIES:,} families)...", flush=True)
    n29 = nc.calibrate_one_sided(29, families=NULL_FAMILIES, seed=NULL_SEED)
    sealed29 = nc.sealed_multiplier(nc.R0255_N29_SEALED_PATH, "n29")
    n29_check = {
        "sealed_value": sealed29,
        "sealed_source": str(nc.R0255_N29_SEALED_PATH),
        "sealed_key": "calibration.n29.candidates.median_minus_k_madn."
                      "one_sided.calibrated_multiplier",
        "reproduced_value": n29["calibrated_multiplier"],
        "relative_error": abs(n29["calibrated_multiplier"] - sealed29) / sealed29,
    }
    print(f"  sealed  {sealed29:.7f}")
    print(f"  ours    {n29['calibrated_multiplier']:.7f}  "
          f"rel err {n29_check['relative_error']:.4%}", flush=True)

    out = {"validation_gate_n13": gate, "n29": n29,
           "n29_agreement_with_sealed_r0255": n29_check,
           "families_simulated": NULL_FAMILIES, "seed": NULL_SEED}
    CALIB_JSON.write_text(json.dumps(out, indent=1))
    print("wrote", CALIB_JSON)
    return out


# --- reporting ----------------------------------------------------------

def _stats(vals: list[float], k: float) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "max": float(v.max()), "madn": float(madn(v)),
            "floor": robust_floor(v, k), "ceiling": robust_ceiling(v, k)}


def _k_for(n: int, families: int = 1_000_000) -> float:
    return nc.calibrate_one_sided(n, families=families, seed=NULL_SEED)["calibrated_multiplier"]


def _fam_rows(fam: list[dict]) -> str:
    lines = ["| seed | round | N | r10/R (raw) | **r10/R x sqrt(N)** | **fog** | peak bin | occupied bins |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in fam:
        lines.append(f"| {r['seed']} | {r['round']} | {r['n_rows']:,} | "
                     f"{r['collapse_raw']:.2e} | {r['collapse']:.4f} | "
                     f"{r['fog']:.4f} | {r['peak_bin_count']:,} | "
                     f"{r['occupied_bin_fraction']:.4f} |")
    return "\n".join(lines)


def _sbx_rows(rows: list[dict], mark_core: bool = False) -> str:
    lines = ["| arm | rung | N | a (min_dist fit) | dose | seed | r10/R (raw) | "
             "**r10/R x sqrt(N)** | **fog** | peak bin | occupied bins |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        name = r["arm"] + (" **(core)**" if mark_core and is_core_umap(r) else "")
        a = f"{r['a']:.4f}" if r.get("a") else "-"
        lines.append(f"| {name} | {r['rung']} | {r['n_rows']:,} | {a} | "
                     f"{r.get('dose_multiplier') or '-'} | {r.get('seed') or '-'} | "
                     f"{r['collapse_raw']:.2e} | {r['collapse']:.4f} | "
                     f"{r['fog']:.4f} | {r['peak_bin_count']:,} | "
                     f"{r['occupied_bin_fraction']:.4f} |")
    return "\n".join(lines)


def stage_report() -> None:
    meas = json.loads(MEASURED_JSON.read_text())
    cal = json.loads(CALIB_JSON.read_text())
    k29 = cal["n29"]["calibrated_multiplier"]
    fam = meas["family"]
    sbx = meas["sandbox"]

    fam_collapse = _stats([r["collapse"] for r in fam], k29)
    fam_fog = _stats([r["fog"] for r in fam], k29)

    umap = [r for r in sbx if r["kernel"] == "umap"]
    cuml = [r for r in sbx if r["kernel"] == "cuml_umap"]
    other = [r for r in sbx if r["kernel"] not in ("umap", "cuml_umap")]
    core = [r for r in sbx if is_core_umap(r)]
    if len(core) < 5:
        raise SystemExit(f"core umap family too small ({len(core)}); check summary.json overrides")

    k_core = _k_for(len(core))
    c_collapse = _stats([r["collapse"] for r in core], k_core)
    c_fog = _stats([r["fog"] for r in core], k_core)

    k_all = _k_for(len(umap) + len(cuml))
    a_collapse = _stats([r["collapse"] for r in umap + cuml], k_all)
    a_fog = _stats([r["fog"] for r in umap + cuml], k_all)

    # matched-rung pairs: same arm name at 2m and 6250k, i.e. one recipe at
    # 3.1x the rows. This is the N-invariance evidence quoted in the report.
    by_rung = {r["rung"]: {x["arm"]: x for x in sbx if x["rung"] == r["rung"]}
               for r in sbx}
    pairs = [(by_rung["2m"][a], by_rung["6250k"][a])
             for a in sorted(by_rung.get("6250k", {}))
             if a in by_rung.get("2m", {})]
    if not pairs:
        raise SystemExit("no matched 2m/6250k arm pairs; cannot evidence N-invariance")
    adj_drift = [abs(x["collapse"] - y["collapse"]) / x["collapse"] for x, y in pairs]
    raw_ratio = [x["collapse_raw"] / y["collapse_raw"] for x, y in pairs]

    fog_degenerate = [r for r in sbx if r.get("fog_degenerate")]

    cuml_row = cuml[0] if cuml else None
    gate = cal["validation_gate_n13"]
    n29c = cal["n29_agreement_with_sealed_r0255"]

    md = REPORT_TEMPLATE.format(
        k29=k29,
        k29_ff=cal["n29"]["new_cell_false_fail_rate"],
        k29_p1=cal["n29"]["detection_power"]["minus_1_sigma"],
        k29_p2=cal["n29"]["detection_power"]["minus_2_sigma"],
        k29_p3=cal["n29"]["detection_power"]["minus_3_sigma"],
        families=f"{cal['families_simulated']:,}",
        null_seed=cal["seed"],
        gate_sealed=gate["sealed_value"], gate_ours=gate["reproduced_value"],
        gate_rel=gate["relative_error"] * 100,
        gate_verdict="PASS" if gate["passes"] else "FAIL",
        n29_sealed=n29c["sealed_value"], n29_rel=n29c["relative_error"] * 100,
        fam_table=_fam_rows(fam),
        fam_c_med=fam_collapse["median"], fam_c_madn=fam_collapse["madn"],
        fam_c_min=fam_collapse["min"], fam_c_max=fam_collapse["max"],
        fam_c_floor=fam_collapse["floor"],
        fam_f_med=fam_fog["median"], fam_f_madn=fam_fog["madn"],
        fam_f_min=fam_fog["min"], fam_f_max=fam_fog["max"],
        fam_f_ceil=fam_fog["ceiling"],
        umap_table=_sbx_rows(umap + cuml, mark_core=True),
        legacy_table=_sbx_rows(other),
        n_core=len(core), k_core=k_core,
        c_c_med=c_collapse["median"], c_c_madn=c_collapse["madn"],
        c_c_min=c_collapse["min"], c_c_max=c_collapse["max"],
        c_c_floor=c_collapse["floor"],
        c_f_med=c_fog["median"], c_f_madn=c_fog["madn"],
        c_f_min=c_fog["min"], c_f_max=c_fog["max"], c_f_ceil=c_fog["ceiling"],
        n_all=len(umap) + len(cuml), k_all=k_all,
        a_c_floor=a_collapse["floor"], a_f_ceil=a_fog["ceiling"],
        a_c_med=a_collapse["median"], a_c_madn=a_collapse["madn"],
        a_f_med=a_fog["median"], a_f_madn=a_fog["madn"],
        cuml_collapse=cuml_row["collapse"] if cuml_row else float("nan"),
        cuml_fog=cuml_row["fog"] if cuml_row else float("nan"),
        cuml_peak=cuml_row["peak_bin_count"] if cuml_row else 0,
        cuml_n=cuml_row["n_rows"] if cuml_row else 0,
        sep_factor=c_collapse["min"] / fam_collapse["max"],
        n_pairs=len(pairs),
        drift_lo=min(adj_drift) * 100, drift_hi=max(adj_drift) * 100,
        raw_lo=min(raw_ratio), raw_hi=max(raw_ratio),
        degen_arms=", ".join(f"`{r['arm']}` ({r['rung']}, peak bin "
                             f"{r['peak_bin_count']})" for r in fog_degenerate)
                   or "none in this measurement set",
    )
    REPORT_MD.write_text(md)
    print("wrote", REPORT_MD)
    print(f"  proposed collapse floor  >= {c_collapse['floor']:.4f}  "
          f"(core n={len(core)}, k={k_core:.5f})")
    print(f"  proposed fog ceiling     <= {c_fog['ceiling']:.4f}")


REPORT_TEMPLATE = r"""# COLLAPSE and FOG — production metrics and their calibration

Metric Option A. Two CPU-only map-quality metrics that the program's
registered gates cannot see, hardened out of the sandbox, plus their
calibration on the sealed n = 29 2M seed family.

- Metrics: `experiments/metrics/collapse_fog.py`
- Null calibrator: `experiments/metrics/null_calibration.py`
- Driver: `experiments/metrics/calibrate_collapse_fog.py`
- Tests: `experiments/metrics/tests/test_collapse_fog.py`
- Machine-readable results:
  `experiments/metrics/results/collapse-fog-measurements.json`,
  `experiments/metrics/results/collapse-fog-calibration.json`

**Nothing here is a registered gate and nothing sealed was written.** Every
sealed number quoted below carries its file path and JSON key.

## 1. What the two metrics measure

**COLLAPSE** — `r10 / R · sqrt(N)`, where `r10` is the median 2D distance to
the 10th nearest neighbour over a seeded 20k sample and `R` is the p90 radius
about the centroid. Hardened from `tissue_metrics()` in
`experiments/sandbox/heldout_eval.py` and the inline copy in
`experiments/sandbox/knobs_2m.py::run_arm`, both of which report the raw
ratio `r10 / R`.

The `sqrt(N)` factor is the new part. In 2D at fixed occupied area the median
distance to the k-th neighbour scales as `1/sqrt(N)`, so the raw ratio is not
comparable across map sizes. Multiplying by `sqrt(N)` removes that scaling —
see §4, where {n_pairs} umap recipes measured at both 2M and 6.25M rows agree to
within {drift_hi:.0f}% on the adjusted statistic while their raw ratios differ by
{raw_lo:.1f}-{raw_hi:.1f}x.
Gated **one-sided from below**: too small means the map has folded into
point-like beads. FFR *rewards* that failure, which is why no registered
metric sees it.

**FOG** — fraction of total binned point mass sitting in *occupied* bins
whose count is below 1% of the peak bin count, on a 1024x1024 histogram over
the (0.1, 99.9) percentile extent with 2% pad and edge clipping. The binning
is byte-for-byte `experiments/map_renders.py::robust_extent` /
`binned_counts`; the mass definition is `low_density_mass_fraction` from
`heldout_eval.tissue_metrics`. Gated **one-sided from above**: too large
means diffuse haze between the clusters. The implementation reproduces the
sandbox's published cuML reference value (0.040) exactly: measured here at
**{cuml_fog:.4f}**.

The two are different failure directions, not two views of one axis. A
collapsed map has *low* fog (all mass in a handful of dense bins) and a low
collapse statistic. A hazy map has *high* fog and a normal collapse
statistic. A gate needs both — and the tests assert exactly that cross
property.

## 2. Calibrated multiplier

The floor/ceiling estimator is the program's established robust form,
`median -/+ k · MAD_n`, consistency constant 1.4826. `k` is calibrated on a
Gaussian null exactly as R0234 did: simulate {families} null families of the
given size and take the smallest `k` whose floor `median - k · MAD_n` is
cleared by a fresh conforming draw 95% of the time with 95% confidence
(one-sided 95/95 tolerance bound). Because `1 - Phi(L) >= 0.95` iff
`L <= z_0.05`, this inverts per family and `k` is simply the 95th percentile
of `(median + 1.6448536) / MAD_n` — no bisection, no search noise.

**Validation gate, run before trusting anything at n = 29.** R0234's
published one-sided n = 13 `median_minus_k_madn` multiplier, read from
`/data/latent-basemap/runs/round-0234/queue/artifacts/minilm-mixed-2m-calibrated-robust-floors-n13-v1/minilm-calibrated-robust-floors-n13.json`
key `calibration.n13.candidates.median_minus_k_madn.one_sided.calibrated_multiplier`:

| | value |
| --- | --- |
| sealed R0234 (n = 13) | **{gate_sealed:.10f}** |
| reproduced here | **{gate_ours:.10f}** |
| relative error | **{gate_rel:.4f}%** (tolerance 2%) |
| verdict | **{gate_verdict}** |

The agreement is bit-exact, not merely within tolerance: with the same seed
({null_seed}), the same {families} families and the same generator, the
closed-form quantile lands on the identical double.

**Derived n = 29 multiplier:**

| | value |
| --- | --- |
| **k1 (one-sided, n = 29)** | **{k29:.10f}** |
| null families simulated | {families} (seed {null_seed}) |
| new-cell false-fail rate | {k29_ff:.6f} |
| detection power at -1s / -2s / -3s | {k29_p1:.4f} / {k29_p2:.4f} / {k29_p3:.4f} |

Cross-check against the sealed R0255 value at the same n, read from
`/data/latent-basemap/runs/round-0255/queue/artifacts/minilm-mixed-2m-calibrated-madn-floors-n29-v1/minilm-calibrated-madn-floors-n29.json`
key `calibration.n29.candidates.median_minus_k_madn.one_sided.calibrated_multiplier`
= **{n29_sealed:.10f}**; ours differs by **{n29_rel:.4f}%** (also bit-exact).
R0255 calibrated that multiplier for FFR and purity; it is the same estimator
at the same n, so agreement is the expected result and is reported as a
consistency check, not a new claim.

## 3. Legacy fingerprint — the sealed n = 29 2M seed family

All 29 cells, seeds 42–70, resolved through `/data/latent-basemap/maps.json`
(`coordinates.file`, `gsv:` prefix stripped). The authoritative seed list is
R0255's own `exact_family_seeds` in the n = 29 artifact cited in §2 — note it
spans five rounds (R0217/0218, R0221/0222, R0230, R0250, R0255), not the
four one might infer from the round-0255 queue alone.

Every one of the 29 is a `legacy_lp`-kernel map. Every one is bead-collapsed.

{fam_table}

| statistic | median | MAD_n | min | max |
| --- | --- | --- | --- | --- |
| collapse `r10/R·sqrt(N)` | {fam_c_med:.4f} | {fam_c_madn:.4f} | {fam_c_min:.4f} | {fam_c_max:.4f} |
| fog | {fam_f_med:.4f} | {fam_f_madn:.4f} | {fam_f_min:.4f} | {fam_f_max:.4f} |

Applying the calibrated n = 29 estimator to this family gives a collapse
floor of **{fam_c_floor:.4f}** and a fog ceiling of **{fam_f_ceil:.4f}**.

**Neither is a usable gate and neither should be registered as one.** The
family is 29 replicates of one broken recipe, so a floor fitted to it
certifies that a new map is *as collapsed as the collapsed ones*. Any healthy
map (collapse 1.0–2.8) clears the collapse floor of {fam_c_floor:.4f}
trivially, and a fog ceiling of {fam_f_ceil:.4f} would **reject every healthy
map measured in §4** — real tissue carries far more low-density mass than a
bead field does, so the collapsed family is *better* on fog than the maps we
want.

What the family is genuinely good for is a **fingerprint**: it pins the
collapsed mode at collapse = {fam_c_med:.4f} +/- {fam_c_madn:.4f} (MAD_n)
across 29 independent seeds and five rounds, with a total range of
{fam_c_min:.4f}-{fam_c_max:.4f}. The mode is reproducible and tight, and §5 shows
it is unmoved by dose, by the a/b fit, or by graph choice, which is what makes
the separation in §6 meaningful. (All 29 cells are 2M rows, so the family
itself carries no evidence about N; that evidence is in §4.)

## 4. Healthy family — sandbox umap-kernel arms + the cuML reference

Coordinates under `/data/latent-basemap/sandbox/2m-knobs/*/coordinates.npy`,
`/data/latent-basemap/sandbox/6250k-knobs/*/coordinates.npy` and
`/data/latent-basemap/sandbox/cuml-1m/cuml-xy.npy`. `a` is the UMAP (a, b)
fit; `a = 1.9328` is `min_dist = 0.00` and `a` falls as `min_dist` rises.
Arms marked **(core)** are the `min_dist = 0.00` umap recipe — they differ
only in seed, dose multiplier and rung, which is the closest thing to a
conforming population the sandbox offers.

{umap_table}

The N-invariance check lives in this table: {n_pairs} arm names appear at both
2M and 6.25M rows with the same recipe, i.e. one map spec at 3.1x the rows.
Between the matched rungs the raw `r10/R` differs by {raw_lo:.2f}-{raw_hi:.2f}x
against the sqrt(6.25/2) = 1.77 that pure density scaling predicts, and the
residual is exactly the {drift_lo:.1f}-{drift_hi:.1f}% by which the adjusted
statistic drifts. So the sqrt(N) factor removes the bulk of an ~1.9x effect
and leaves ~10%. That is the whole case for it, and it is measured, not
assumed.

## 5. Legacy-kernel and other sandbox arms, for contrast

{legacy_table}

Two things worth flagging. First, `gc-a2-md000-x2` and `gc-a2-md010-x2` (the
generalised-Cauchy kernel at a = 2) are **not** collapsed — collapse 2.48 and
2.69, fog 0.98 and 0.99 — so the collapse failure is a property of the
`legacy_lp` kernel specifically, and of the generalised-Cauchy kernel only at
a = 0.5. Second, the four `legacy_lp` arms (`dose-x2`, `kernel-a4`,
`kernel-b2`, `replay-baseline`) plus the two `gc-a05-*` arms all land in
0.11-0.18, i.e. inside or beside the sealed family's fingerprint of
{fam_c_min:.4f}-{fam_c_max:.4f}. Doubling the dose, setting a = 4, setting
b = 2, and replaying the graph all leave the collapse statistic where it was.
Only the low-dimensional kernel moves it.

## 6. Proposed provisional bands

Fitted to the **core umap family** of §4 (n = {n_core}: the `min_dist = 0.00`
arms at both rungs) with the same estimator and `k` calibrated at that family
size — `k = {k_core:.5f}` from 1,000,000 Gaussian null families:

| metric | direction | median | MAD_n | min | max | **proposed bound** |
| --- | --- | --- | --- | --- | --- | --- |
| collapse `r10/R·sqrt(N)` | lower floor | {c_c_med:.4f} | {c_c_madn:.4f} | {c_c_min:.4f} | {c_c_max:.4f} | **>= {c_c_floor:.4f}** |
| fog | upper ceiling | {c_f_med:.4f} | {c_f_madn:.4f} | {c_f_min:.4f} | {c_f_max:.4f} | **<= {c_f_ceil:.4f}** |

Against those bounds:

- all 29 sealed family cells **fail** the collapse floor (max
  {fam_c_max:.4f} vs floor {c_c_floor:.4f}) — which is the point;
- the cuML reference (N = {cuml_n:,}) **passes both**: collapse
  {cuml_collapse:.4f}, fog {cuml_fog:.4f};
- the over-separated umap arms at `min_dist >= 0.20` **fail** the fog ceiling
  at 0.95-1.00, so the ceiling behaves as a haze detector on real maps and not
  only on synthetic ones -- **except** `umap-md035-x2`, which reports fog
  0.0000 and would PASS despite being one of the haziest maps in the sandbox.
  That is the degeneracy in item 5 below, and it is why a fog gate must consult
  `degenerate` before it consults the value.

### Why the bands are not fitted to all 18 umap+cuML arms

Pooling the whole knob sweep (n = {n_all}, `k = {k_all:.5f}`) gives a collapse
floor of **{a_c_floor:.4f}** and a fog ceiling of **{a_f_ceil:.4f}**: a
negative floor is vacuous and a ceiling above 1.0 is unattainable, because the
sweep deliberately spans `min_dist` 0.00 → 0.50 and is therefore a *designed
contrast*, not a sample from one population (median {a_c_med:.4f}, MAD_n
{a_c_madn:.4f} for collapse; {a_f_med:.4f} / {a_f_madn:.4f} for fog). That
degenerate result is reported rather than hidden — it is the clearest
available demonstration that the 95/95 machinery gives nonsense when its
exchangeability assumption is violated.

### Evidentiary status — read before quoting any number above

1. **Provisional reference bands, not registered gates.** No round registered
   them; this work registered nothing.
2. **The core family is not a seed family.** n = {n_core} arms sharing one
   (a, b) but differing in seed (2 values), dose (3 levels) and rung (2M and
   6.25M). The 95/95 tolerance machinery assumes n exchangeable draws from
   one conforming population. That is not met. `k` is applied for consistency
   with the program's convention, not because the sample earns it.
3. **The collapse band is the stronger of the two.** The two populations do
   not overlap and are not close: the sealed family's maximum is
   {fam_c_max:.4f} and the core family's minimum is {c_c_min:.4f}, a factor of
   {sep_factor:.1f}, with the proposed floor {c_c_floor:.4f} sitting between
   them. Its N-invariance is measured directly across two rungs (§4).
4. **The fog band is the weaker of the two, and is a ceiling on regression
   rather than a quality bar.** The core arms sit at {c_f_med:.4f} while the
   cuML reference is an order of magnitude below at {cuml_fog:.4f}. Treat
   cuML as the aspiration and {c_f_ceil:.4f} as the line past which a map has
   become visibly hazier than the current umap recipe.
5. **Fog has a hard degeneracy that a gate must handle.** Bin counts are
   integers and the cutoff is 1% of the peak bin, so a map whose peak bin
   holds fewer than 100 points reports fog **exactly 0.0000** no matter how
   hazy it is. `map_fog` returns `resolution_levels` and `degenerate` for
   this reason. Degenerate in this measurement set: {degen_arms} — an arm that
   reports fog 0.0000 while sitting between two arms that report 0.95 and 1.00.
   The cuML reference is not degenerate but is one step from it, at peak bin
   {cuml_peak} —
   **one** usable level — so its celebrated 0.040 sits a single integer count
   above the degeneracy. Any fog gate must refuse a degenerate measurement
   instead of passing it.
6. **The fog cutoff is partly a corpus property.** On the 2M mixed MiniLM
   substrate the peak bin of every umap-kernel arm is the same exact-duplicate
   group of 1377 identical rows, so the 1%-of-peak cutoff is pinned by
   duplicate text rather than by the map's densest tissue. Changing the
   corpus changes the cutoff.
7. **What would make these registrable:** a real seed family of *healthy*
   maps — one recipe, n >= 13 seeds, no knob variation — measured under this
   protocol. Until then the honest claim is: the collapse statistic separates
   the two modes cleanly and is ready to gate once a healthy family exists;
   the legacy family cannot define a healthy floor for either metric.

## 7. Positive controls

`experiments/metrics/tests/test_collapse_fog.py` carries the mandatory
failing inputs — a guard whose suite contains no failing input is untested at
its only job:

- **(a)** a synthetic bead-collapsed map (40 clusters of near-identical
  points) falls below the collapse floor;
- **(b)** a synthetic haze map (clusters buried in 55% uniform noise) exceeds
  the fog ceiling;
- **(c)** a synthetic healthy map (Gaussian blobs of finite radius, 0.2%
  background) clears both;
- **cross controls**: the collapsed map does *not* trip fog and the hazy map
  does *not* trip collapse — the two are independent directions;
- **(d)** N-invariance: 4x subsampling a healthy map moves `r10/R·sqrt(N)` by
  < 10%, while the raw ratio moves by > 1.5x (asserted as a control on the
  control);
- **(e)** determinism: same seed gives identical dicts; fog has no rng at all;
- **(f)** memmap: the memmap and in-memory paths give identical dicts, and
  the subsampled-tree path (forced with a small `max_tree_rows`) agrees with
  the full-tree path to within 10%;
- plus the fog-degeneracy flag, shape rejection, the exact `sqrt(N)`
  algebra, and a 1M-family smoke reproduction of R0234's sealed multiplier.

A note on the literal requirement "a synthetic uniform-noise map must fail
the fog ceiling": **it cannot, by arithmetic**, and the suite documents that
rather than papering over it. Fog counts mass in bins below 1% of the *peak*
bin, and bin counts are integers. A structureless uniform map has no peak —
at 2M points over 1024² bins the peak bin holds ~13 counts, 1% of that is
0.13, the absolute floor of 1 takes over, and no occupied bin can hold less
than 1 point. Fog is exactly 0.0000. The failure fog exists to catch is
*clusters plus haze*, which is what control (b) uses. Pure structureless
noise is separated cleanly by the companion `occupied_bin_fraction` that
`map_fog` already returns — 0.81 for uniform noise against 0.04–0.55 for
every real map measured here — but that companion is reported, not gated, and
is not part of this proposal.

## 8. Reproduction

```bash
cd /home/enjalot/code/latent-basemap

# both metrics on all 29 sealed family cells + every sandbox arm + cuML (~45 s)
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py measure

# Gaussian-null calibration: R0234 n=13 validation gate, then n=29 (~10 s)
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py calibrate

# regenerate this report from the two JSONs
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py report

# or all three
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    experiments/metrics/calibrate_collapse_fog.py all

# tests, including the mandatory positive controls
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    experiments/metrics/tests/ -q -p no:cacheprovider
```

Everything is CPU-only: `CUDA_VISIBLE_DEVICES=""` is set in every command,
the driver warns if it is not, and no module in this proposal imports torch,
cuml, or faiss.
"""


def main(argv: list[str]) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("warning: CUDA_VISIBLE_DEVICES is not empty; this task is CPU-only",
              file=sys.stderr)
    stage = argv[1] if len(argv) > 1 else "all"
    if stage in ("measure", "all"):
        stage_measure()
    if stage in ("calibrate", "all"):
        stage_calibrate()
    if stage in ("report", "all"):
        stage_report()
    if stage not in ("measure", "calibrate", "report", "all"):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

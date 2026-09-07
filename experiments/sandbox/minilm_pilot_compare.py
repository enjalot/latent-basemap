"""MiniLM mix-pilot verdict — CORRECTED compare (owner gate 2026-09-06). Standalone so it does NOT edit the running
driver. Reads the ARM-LEVEL register-score.json that minilm_mix_score.py actually writes (SB/<ds>/register-score.json),
not the driver's buggy SB/<ds>/champion-bs16k/register-score.json path. Emits base-displacement (base reception drop
from mixing) + social-sink (per-social top-1%-cell mass + entropy) -> SB/minilm-pilot-verdict.json.
"""
import json
from pathlib import Path
SB = Path("/data/latent-basemap/sandbox")
BASE = ["fineweb", "redpajama", "pile"]; SOC = ["reddit", "communityarchive", "bluesky", "twitter100m"]


def load(ds):
    p = SB / ds / "register-score.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    mix = load("minilm-mixpilot-2m"); base = load("minilm-base-2m")
    out = {"schema": "minilm-mix-pilot-verdict-2026-09-06-corrected",
           "note": "base_displacement = base_reception - mix_reception per base register (>0 = mixing DEGRADED base "
                   "reception, a pathology). social_sink: high top1%-cell-mass + low entropy = a register collapsed "
                   "into few cells."}
    if not (mix and base):
        out["error"] = f"missing register-score (mix={mix is not None}, base={base is not None})"
        (SB / "minilm-pilot-verdict.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); return 0
    mp, bp = mix["per_register"], base["per_register"]
    out["mix_v2ffr"] = mix["v2_ffr_own_truth"]; out["base_v2ffr"] = base["v2_ffr_own_truth"]
    out["base_displacement"] = {r: round(bp.get(r, {}).get("reception@15", 0) - mp.get(r, {}).get("reception@15", 0), 4)
                                for r in BASE if r in mp and r in bp}
    out["social_sink"] = {r: {"reception@15": mp.get(r, {}).get("reception@15"),
                              "top1pct_cell_mass": mp.get(r, {}).get("top1pct_cell_mass"),
                              "entropy_norm": mp.get(r, {}).get("entropy_norm")} for r in SOC if r in mp}
    md = out["base_displacement"]
    out["verdict"] = ("base NOT displaced (max |Δ| %.4f)" % max((abs(v) for v in md.values()), default=0)
                      if md and max(md.values(), default=0) <= 0.02 else "base DISPLACED — mixing degraded base reception")
    (SB / "minilm-pilot-verdict.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

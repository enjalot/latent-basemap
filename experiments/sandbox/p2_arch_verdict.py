#!/usr/bin/env python3
"""P2 arch two-seed verdict (delegate 2026-08-30). Compares the h3072-neck625 vs h2048 maximin delta
across seeds 42 + 43. Sign agreement -> seal width (flagship = x8-h3072-neck625); split -> seed-scale,
flagship = h2048 (mean-winner, cheaper). Flagship LAUNCH stays owner-gated regardless."""
import json, sys
from pathlib import Path
SB = Path("/data/latent-basemap/sandbox")
def _wd(fn):
    p = SB / fn
    if not p.exists(): return None
    return json.loads(p.read_text()).get("worst_register_delta_mix_minus_baseline")
def main():
    s42 = _wd("p2-arch-jina-confirm-results.json")
    s43 = _wd("p2-arch-s43-jina-confirm-results.json")
    out = {"schema": "p2-arch-two-seed-verdict", "s42_maximin_delta": s42, "s43_maximin_delta": s43}
    if s42 is None or s43 is None:
        out["verdict"] = f"INCOMPLETE (s42={s42}, s43={s43})"
    else:
        sign = lambda x: (1 if x > 0 else (-1 if x < 0 else 0))
        agree = sign(s42) == sign(s43) and sign(s42) != 0
        out["sign_agreement"] = bool(agree)
        if agree and sign(s42) == 1:
            out["verdict"] = "SEALED WIN — h3072-neck625 wins maximin both seeds; flagship spec = x8-h3072-neck625"
        elif agree and sign(s42) == -1:
            out["verdict"] = "SEALED — h2048 wins maximin both seeds; flagship = h2048"
        else:
            out["verdict"] = "SPLIT (seed-scale) — width question closes; flagship = h2048 (mean-winner, cheaper)"
    (SB / "p2-arch-verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)); return 0
if __name__ == "__main__": raise SystemExit(main())

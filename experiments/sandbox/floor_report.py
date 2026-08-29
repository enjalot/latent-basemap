#!/usr/bin/env python3
"""P0.1 resident-floor reporter (4th-review, delegate 2026-08-29).

Reads the same-seed(42) determinism-twin summaries and reports the resident floor
as a PATH property. PRIMARY readout: trained_state_sha256 EQUALITY within a space —
identical => the resident floor is literally ZERO (every same-seed cross-config
comparison is exact, no statistical banding). init_state_sha256 MUST match across a
pair regardless (the seeding proof); a mismatch there means the seeding fix itself
failed. If trained hashes DIFFER, the space needs the |Δmaximin| fallback floor
(flagged here, not computed — that's a follow-on scoring decision).

Writes /data/latent-basemap/sandbox/floor-result.json.
"""
import json
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox")
PAIRS = {
    "resident-D384": ("minilm-bmix10cp-2m/floor-resident-a",
                      "minilm-bmix10cp-2m/floor-resident-b"),
    "resident-D768": ("jina-multi-2m/floor-resident-h200k-a",
                      "jina-multi-2m/floor-resident-h200k-b"),
}


def _load(rel):
    p = SB / rel / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    out = {"schema": "resident-floor-2026-08-29", "spaces": {}}
    all_deterministic = True
    for space, (ra, rb) in PAIRS.items():
        a, b = _load(ra), _load(rb)
        rec = {"arm_a": ra, "arm_b": rb}
        if a is None or b is None:
            rec["status"] = f"INCOMPLETE (missing {'a' if a is None else ''}{'b' if b is None else ''})"
            out["spaces"][space] = rec
            all_deterministic = False
            continue
        ia, ib = a.get("init_state_sha256"), b.get("init_state_sha256")
        ta, tb = a.get("trained_state_sha256"), b.get("trained_state_sha256")
        rec.update({
            "seed_a": a.get("seed"), "seed_b": b.get("seed"),
            "init_sha_a": ia, "init_sha_b": ib, "init_match": bool(ia and ia == ib),
            "trained_sha_a": ta, "trained_sha_b": tb, "trained_match": bool(ta and ta == tb),
        })
        if not rec["init_match"]:
            rec["verdict"] = "SEEDING BROKEN — init hashes differ at same seed (fix failed)"
            all_deterministic = False
        elif rec["trained_match"]:
            rec["verdict"] = "DETERMINISTIC — resident floor is ZERO for this space; same-seed comparisons are exact"
        else:
            rec["verdict"] = "NONDETERMINISTIC — init matches but trained differs; needs |Δmaximin| fallback floor"
            all_deterministic = False
        out["spaces"][space] = rec
        print(f"[{space}] init {'MATCH' if rec['init_match'] else 'MISMATCH'} "
              f"({ia} vs {ib}) | trained {'MATCH' if rec['trained_match'] else 'DIFFER'} "
              f"({ta} vs {tb}) -> {rec['verdict']}", flush=True)
    out["all_resident_deterministic"] = all_deterministic
    out["implication"] = (
        "both spaces deterministic -> resident floors are 0; measure int8 same-seed x2 "
        "to band the device-int8 parity question. Any space nondeterministic -> one "
        "|Δmaximin| floor protocol covers everything." )
    (SB / "floor-result.json").write_text(json.dumps(out, indent=1))
    print(f"\nall_resident_deterministic={all_deterministic}; wrote {SB/'floor-result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

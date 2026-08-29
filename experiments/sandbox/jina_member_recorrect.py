#!/usr/bin/env python3
"""Recorrect the jina projector member/unseen SPLIT in previously-written *-jina-confirm-results.json
(4th-review P1.6, 2026-08-29).

Those files split the jina_6m_transform FFR by member_cutoff=2M (first 2M CONTIGUOUS rows), which is
WRONG — the 2M head's members are the OLD-block PREFIXES scattered at each 6.25M span start. The
OVERALL FFR is UNAFFECTED (same queries/truth/disc); only the member/unseen split was mis-bucketed.

For each result file, per head: if its saved projector coords exist, RE-EMIT the split with the exact
member_mask_2m (cheap: quick_ffr_v2_split on saved 2D coords, CPU) and assert overall is unchanged;
else just STAMP the correction note. Writes the file back with member_split_corrected metadata.
CPU-only, safe alongside GPU training.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

SB = Path("/data/latent-basemap/sandbox")


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr_v2_split
    from jina_head_membership import member_mask_2m
    import confirm_jina_mixture as J
    mask = member_mask_2m()
    n6 = int(mask.shape[0])
    truth = J.JINA6M_TRUTH
    knn6 = J.JINA6M_KNN if J.JINA6M_KNN.exists() else None

    files = sorted(glob.glob(str(SB / "*-jina-confirm-results.json")))
    print(f"found {len(files)} jina-confirm result files", flush=True)
    summary = []
    for fp in files:
        fp = Path(fp)
        d = json.loads(fp.read_text())
        if d.get("ABORTED"):
            continue
        arm = d.get("arm", fp.stem.replace("-jina-confirm-results", ""))
        coords_dir = SB / f"{arm}-jina-confirm-coords"
        projs = d.get("projectors", {})
        changed = False
        for key, pinfo in projs.items():
            j6 = (pinfo or {}).get("jina_6m_transform")
            if not isinstance(j6, dict) or "ffr_v2_member" not in j6:
                continue
            pfx = "baseline" if key.startswith("baseline") else arm
            coords = coords_dir / f"{pfx}__proj-jina6m.npy"
            old = {"member": j6.get("ffr_v2_member"), "unseen": j6.get("ffr_v2_unseen"),
                   "member_frac": j6.get("member_frac")}
            if coords.exists():
                xy = np.asarray(np.load(coords), dtype=np.float32)
                sp = quick_ffr_v2_split(xy, truth, n6, member_mask=mask,
                                        knn_indices_path=(knn6 if knn6 else None))
                # overall must be unchanged (sanity): compare to recorded ffr_v2
                rec_overall = j6.get("ffr_v2")
                ok = (rec_overall is None or abs(float(sp["overall"]) - float(rec_overall)) < 1e-6)
                j6["ffr_v2_member"] = sp["member"]
                j6["ffr_v2_unseen"] = sp["unseen"]
                j6["member_frac"] = sp["member_frac"]
                j6["n_member_queries"] = sp["n_member"]
                j6["n_unseen_queries"] = sp["n_unseen"]
                j6["member_split_corrected"] = {
                    "reemitted_from_coords": str(coords), "overall_unchanged": bool(ok),
                    "prior_wrong_split": old,
                    "note": "member = exact old-block-prefix mask; prior split used contiguous 2M cutoff"}
                changed = True
                print(f"  {fp.name} [{key}]: re-emitted member {old['member']}->{sp['member']} "
                      f"unseen {old['unseen']}->{sp['unseen']} (overall unchanged={ok})", flush=True)
            else:
                j6["member_split_corrected"] = {
                    "reemitted_from_coords": None, "prior_wrong_split": old,
                    "note": ("member/unseen SPLIT used the WRONG contiguous 2M cutoff; coords absent "
                             "so not re-emitted. OVERALL ffr_v2 is unaffected (same queries/truth).")}
                changed = True
                print(f"  {fp.name} [{key}]: coords absent, stamped note only", flush=True)
        if changed:
            d["member_split_correction_2026_08_29"] = (
                "jina_6m_transform member/unseen re-bucketed with the exact old-block-prefix mask "
                "(jina_head_membership.member_mask_2m). Overall FFR unaffected; do not over-correct.")
            fp.write_text(json.dumps(d, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
            summary.append(fp.name)
    print(f"\ncorrected {len(summary)} files: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

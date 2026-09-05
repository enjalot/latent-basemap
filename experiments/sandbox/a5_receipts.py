"""A5: gen-key receipts for the new image-phase artifacts (owner D-block, 2026-09-05). CPU.

Stamps each headline artifact of the image phase with a gen_key manifest (git commit + input digests + config) so
the article numbers are audit-traceable to the exact code + inputs that produced them. Idempotent: re-writes the
.manifest.json next to each artifact.

Covers: the 5 projection coords (full-pool 2D/3D, full-corpus 2D/3D/4M), the 4M gate verdict, the D2 rescore
table, the 4M draw manifest, and the BL OOD scores.
"""
import json, sys
from pathlib import Path

SB = Path("/data/latent-basemap/sandbox")
HERE = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(HERE))
    import gen_key
    commit = gen_key.git_commit()

    # (artifact_path, config-inputs dict) — key captures what the artifact depends on
    items = []
    projs = {
        "monet-clip-fullpool-proj-20260905": "monet-random-clip-2m/champion-bs16k/model.pt",
        "monet-clip-fullpool-proj-3d-20260905": "monet-random-clip-2m-3d/champion-bs16k/model.pt",
        "monet-clip-fullcorpus-proj-2d-20260905": "monet-random-clip-2m/champion-bs16k/model.pt",
        "monet-clip-fullcorpus-proj-3d-20260905": "monet-random-clip-2m-3d/champion-bs16k/model.pt",
        "monet-clip-fullcorpus-proj-4m-20260905": "monet-random-clip-4m/champion-bs16k/model.pt",
    }
    for d, head in projs.items():
        coords = SB / d / "coords.f32.npy"
        if coords.exists():
            hp = SB / head
            items.append((coords, {"kind": "projection", "head": head,
                                   "head_digest": gen_key.file_digest(hp) if hp.exists() else None}))
    for p, kind in ((SB / "monet-random-clip-4m/gate-vs-2m.json", "4m-gate"),
                    (SB / "d2-rescore-20260905.json", "d2-rescore"),
                    (Path("/data2/monet/random-clip-4m/manifest.json"), "4m-draw"),
                    (SB / "bl-ood-20260905/score-2m-2d.json", "bl-ood-2m-2d"),
                    (SB / "bl-ood-20260905/score-2m-3d.json", "bl-ood-2m-3d")):
        if p.exists():
            items.append((p, {"kind": kind, "self_digest": gen_key.file_digest(p)}))

    stamped = []
    for path, cfg in items:
        cfg = {**cfg, "git_commit": commit}
        key = gen_key.artifact_key(cfg)
        gen_key.write_manifest(path, key, {"a5_receipt": True, "git_commit": commit, **cfg})
        stamped.append({"artifact": str(path), "key": key, "kind": cfg["kind"]})
        print(f"[a5] {cfg['kind']}: {path.name} -> key {key}", flush=True)

    (SB / "a5-receipts-20260905.json").write_text(json.dumps(
        {"schema": "a5-image-phase-receipts-2026-09-05", "git_commit": commit,
         "n_stamped": len(stamped), "receipts": stamped}, indent=1))
    print(f"[a5] DONE {len(stamped)} artifacts stamped @ commit {commit}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

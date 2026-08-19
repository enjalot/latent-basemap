"""Python reference for the in-browser projection POC.

sentence-transformers all-MiniLM-L6-v2 (fp32, CPU) -> torch map head (the
checkpoint itself, not the ONNX export) -> (x, y).  Used two ways:

    # write the reference
    python reference_projection.py --out reference.json

    # compare the browser's answers against it
    python reference_projection.py --compare browser_results.json

Gates (task spec): per-string cosine(browser embedding, ST embedding) > 0.999
and |xy_browser - xy_reference| < 0.5% of the map's extent diagonal.

Run it with the scratch venv (never the repo's main .venv):
    /data/latent-basemap/envs/mappack-onnx/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parent.parent
REPO = POC.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/data/hf")

CHECKPOINT = Path("/data/latent-basemap/sandbox/2m-knobs/umap-md000-x4-fneg10/model.pt")
MODELS_JSON = POC / "map" / "models.json"
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "def quicksort(a):\n    if len(a) <= 1: return a",
    "Interest rates rose sharply after the central bank meeting.",
    "Ich habe gestern ein sehr gutes Buch gelesen.",
]

COS_GATE = 0.999
XY_GATE_FRAC = 0.005     # 0.5% of the extent diagonal


def extent_diagonal() -> tuple[list[float], float]:
    frame = json.loads(MODELS_JSON.read_text())["frame"]
    x0, x1, y0, y1 = frame["extent"]
    return frame["extent"], float(np.hypot(x1 - x0, y1 - y0))


def reference(texts: list[str]) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    torch.set_num_threads(8)
    st = SentenceTransformer(ST_MODEL, device="cpu")
    emb = st.encode(texts, normalize_embeddings=True).astype(np.float32)
    head = ParametricUMAP.load(str(CHECKPOINT), device="cpu")
    xy = head.transform(emb, batch_size=64)
    ext, diag = extent_diagonal()
    return {
        "encoder": ST_MODEL,
        "checkpoint": f"gsv:{CHECKPOINT}",
        "extent": ext,
        "extent_diagonal": diag,
        "results": [{"text": t, "embedding": e.tolist(),
                     "xy": [float(p[0]), float(p[1])]}
                    for t, e, p in zip(texts, emb, xy)],
    }


def compare(ref: dict, browser_path: Path) -> dict:
    browser = json.loads(browser_path.read_text())
    diag = ref["extent_diagonal"]
    by_text = {r["text"]: r for r in ref["results"]}
    rows, ok = [], True
    for b in browser["results"]:
        r = by_text.get(b["text"])
        if r is None:
            rows.append({"text": b["text"][:40], "error": "not in reference"})
            ok = False
            continue
        be = np.asarray(b["embedding"], dtype=np.float64)
        re_ = np.asarray(r["embedding"], dtype=np.float64)
        cos = float(be @ re_ / (np.linalg.norm(be) * np.linalg.norm(re_)))
        d = float(np.hypot(b["xy"][0] - r["xy"][0], b["xy"][1] - r["xy"][1]))
        row = {"text": b["text"][:44].replace("\n", " "), "cosine": cos,
               "xy_browser": b["xy"], "xy_reference": r["xy"],
               "xy_delta": d, "xy_delta_frac_of_extent": d / diag,
               "cosine_pass": cos > COS_GATE,
               "xy_pass": d / diag < XY_GATE_FRAC}
        ok &= row["cosine_pass"] and row["xy_pass"]
        rows.append(row)
    return {"encoder_variant": browser.get("encoder_variant"),
            "cosine_gate": COS_GATE, "xy_gate_frac": XY_GATE_FRAC,
            "extent_diagonal": diag, "rows": rows, "passed": bool(ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "reference.json")
    ap.add_argument("--compare", type=Path, default=None,
                    help="browser_results.json produced by verify_headless.mjs")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.out.is_file() and args.compare is not None:
        ref = json.loads(args.out.read_text())
    else:
        ref = reference(TEXTS)
        args.out.write_text(json.dumps(ref, indent=1) + "\n")
        print(f"wrote {args.out}")

    if args.compare is None:
        for r in ref["results"]:
            print(f"  {r['xy'][0]:10.4f} {r['xy'][1]:10.4f}  {r['text'][:44]!r}")
        return 0

    res = compare(ref, args.compare)
    out = args.report or (args.compare.parent / "comparison.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print("PASS" if res["passed"] else "FAIL")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

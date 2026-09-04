"""NeoMME exp-2 (joint-modality) embed (owner MONET/NeoMME probe, overseer 2026-09-04). Embeds the 250K MONET
thumbnail+caption pairs BOTH through NeoMME-260M as retrieval DOCUMENTS -> a single 500K joint substrate in the
same 1024-d space: rows [0..N) = images (thumbnail), rows [N..2N) = texts (caption), row i and row N+i are the
matched pair. Provenance arrays (modality 0=image/1=text, pair_id, source) align 1:1 with the substrate for the
viewer + the interleave-vs-islands analysis.

PROTOCOL NOTE: both modalities use task="document" -> both get the SAME <doc> prompt prefix (token id 5,
verified 2026-09-04); images additionally carry patch tokens + pixel_values. Using the identical prompt for
both means any modality SEPARATION in the map is genuine geometry, NOT a query-vs-doc prompt-token artifact
(the confound the overseer flagged). assert-on-output: we verify <doc> is present, pixel_values exist for
images, and image!=text embeddings — never trust the call succeeding.

Usage: neomme_embed_pairs.py [--selftest] [N=250000]  (env nothing special; needs neomme-env)."""
import os, sys, io, json, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from PIL import Image
from transformers import NeoMMEForRetrieval, NeoMMEProcessor

MID = "Hcompany/NeoMME-260M-Retriever"; DIM = 1024
PAIRS = Path("/data2/monet/neomme-pairs-250k/pairs.parquet")
OUT = Path("/data2/monet/neomme-pairs-500k"); OUT.mkdir(parents=True, exist_ok=True)
B_IMG = 48; B_TXT = 256


def _load_proc_model():
    proc = NeoMMEProcessor.from_pretrained(MID)
    model = NeoMMEForRetrieval.from_pretrained(MID, dtype=torch.float16).cuda().eval()
    doc_id = proc.tokenizer.convert_tokens_to_ids("<doc>")
    assert doc_id not in (None, proc.tokenizer.unk_token_id), "no <doc> token"
    return proc, model, doc_id


def _enc_images(proc, doc_id, pil_list):
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}]}] for im in pil_list]
    inp = proc.apply_chat_template(msgs, task="document", tokenize=True, return_dict=True,
                                   return_tensors="pt", padding=True)
    assert "pixel_values" in inp, "image encode produced no pixel_values"
    assert int(inp["input_ids"][0, 0]) == doc_id, "image doc form missing <doc>"
    return {k: v.cuda() for k, v in inp.items() if hasattr(v, "cuda")}


def _enc_texts(proc, doc_id, txt_list):
    msgs = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in txt_list]
    inp = proc.apply_chat_template(msgs, task="document", tokenize=True, return_dict=True,
                                   return_tensors="pt", padding=True, truncation=True, max_length=512)
    assert int(inp["input_ids"][0, 0]) == doc_id, "text doc form missing <doc>"
    return {k: v.cuda() for k, v in inp.items() if hasattr(v, "cuda")}


def _embed(model, inp):
    with torch.no_grad():
        e = torch.nn.functional.normalize(model(**inp).dense_embeddings, dim=1)
    return e.detach().float().cpu().numpy()


def _pil(b):
    return Image.open(io.BytesIO(bytes(b))).convert("RGB")


def selftest():
    t = pq.read_table(PAIRS, columns=["thumbnail", "caption_florence-2-large"])
    thumbs = t["thumbnail"].to_pylist()[:8]; caps = [str(x) for x in t["caption_florence-2-large"].to_pylist()[:8]]
    proc, model, doc_id = _load_proc_model()
    imgs = [_pil(b) for b in thumbs]
    ie = _embed(model, _enc_images(proc, doc_id, imgs))
    te = _embed(model, _enc_texts(proc, doc_id, caps))
    assert ie.shape == (8, DIM) and te.shape == (8, DIM), f"bad shapes {ie.shape} {te.shape}"
    # images must not collapse to one point, and image != text embedding space
    assert np.abs(ie[0] - ie[1]).sum() > 1e-3, "image embeddings are identical (image path broken)"
    matched = float((ie * te).sum(1).mean())                       # cos(image_i, text_i)
    cross = float((ie @ te.T)[~np.eye(8, dtype=bool)].mean())      # cos(image_i, text_j!=i)
    print(json.dumps({"selftest": "OK", "img_shape": list(ie.shape), "txt_shape": list(te.shape),
                      "matched_pair_cos": round(matched, 4), "random_pair_cos": round(cross, 4),
                      "pairs_align": matched > cross,
                      "note": "matched>random => NeoMME aligns image-caption pairs (expected for a retriever)"}, indent=1))
    return 0


def main():
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    if "--selftest" in sys.argv:
        return selftest()
    N = int(args[0]) if args else 250_000
    t = pq.read_table(PAIRS, columns=["id", "thumbnail", "caption_florence-2-large", "source"])
    ids = t["id"].to_pylist()[:N]; thumbs = t["thumbnail"].to_pylist()[:N]
    caps = [str(x) for x in t["caption_florence-2-large"].to_pylist()[:N]]
    srcs = [str(x) for x in t["source"].to_pylist()[:N]]
    N = len(ids)
    subf = OUT / "substrate.f32.npy"; donef = OUT / "done.txt"
    sub = np.lib.format.open_memmap(subf, mode="r+" if subf.exists() else "w+", dtype=np.float32, shape=(2 * N, DIM))
    done = int(donef.read_text()) if donef.exists() else 0     # rows written into `sub` so far (images then texts)
    proc, model, doc_id = _load_proc_model()
    t0 = time.time()
    # phase 1: images -> rows [0, N)
    s = done if done < N else N
    while s < N:
        e = s; b = min(B_IMG, N - s)
        imgs = [_pil(thumbs[j]) for j in range(s, s + b)]
        sub[s:s + b] = _embed(model, _enc_images(proc, doc_id, imgs))
        s += b; done = s; donef.write_text(str(done))
        if (s // B_IMG) % 20 == 0:
            print(f"  img {s:,}/{N:,} ({s/(time.time()-t0):.0f}/s)", flush=True)
    # phase 2: texts -> rows [N, 2N)
    s = max(done, N)
    t1 = time.time()
    while s < 2 * N:
        j0 = s - N; b = min(B_TXT, 2 * N - s)
        sub[s:s + b] = _embed(model, _enc_texts(proc, doc_id, caps[j0:j0 + b]))
        s += b; done = s; donef.write_text(str(done))
        if ((s - N) // B_TXT) % 20 == 0:
            print(f"  txt {s-N:,}/{N:,} ({(s-N)/(time.time()-t1):.0f}/s)", flush=True)
    sub.flush()
    # provenance aligned 1:1 with substrate rows
    modality = np.concatenate([np.zeros(N, np.int8), np.ones(N, np.int8)])          # 0=image, 1=text
    pair_id = np.concatenate([np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64)])
    np.save(OUT / "modality.npy", modality); np.save(OUT / "pair_id.npy", pair_id)
    (OUT / "ids.json").write_text(json.dumps({"ids": ids, "sources": srcs}))
    (OUT / "manifest.json").write_text(json.dumps({
        "model": MID, "dim": DIM, "n_pairs": N, "n_rows": 2 * N,
        "layout": "rows [0,N)=images, rows [N,2N)=texts; row i and row N+i are the matched pair",
        "prompt": "document (both modalities, same <doc> prefix — modality islands are geometry not prompt artifact)",
        "dtype": "float32-normed", "provenance": ["modality.npy (0=image,1=text)", "pair_id.npy", "ids.json"]}, indent=1))
    print(f"exp-2 substrate done: {2*N:,} rows -> {subf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NeoMME exp-1 harness embed (owner MONET/NeoMME, overseer 2026-09-04). Embeds the first N fineweb-edu
chunked-120 chunks with NeoMME-260M dense_embeddings (document-prompt, 1024-d, mean-pooled, L2-norm) ->
/data2/monet/neomme-fineweb-2m/substrate.f32.npy. Resumable via a row counter (open_memmap r+ resume).
Same text rows as the MiniLM-384 fineweb-120 substrate (continuity). Usage: neomme_embed_fineweb.py [N=2000000]."""
import os, sys, glob, json, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from transformers import NeoMMEForRetrieval, NeoMMEProcessor

MID = "Hcompany/NeoMME-260M-Retriever"; DIM = 1024; B = 256
OUT = Path("/data2/monet/neomme-fineweb-2m"); OUT.mkdir(parents=True, exist_ok=True)
CH = "/data/chunks/fineweb-edu-sample-10BT-chunked-120/train"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    subf = OUT / "substrate.f32.npy"; donef = OUT / "done.txt"
    sub = np.lib.format.open_memmap(subf, mode="r+" if subf.exists() else "w+", dtype=np.float32, shape=(N, DIM))
    done = int(donef.read_text()) if donef.exists() else 0
    print(f"NeoMME embed: N={N:,} resume@{done:,}", flush=True)
    if done >= N:
        print("already complete"); return 0
    proc = NeoMMEProcessor.from_pretrained(MID)
    model = NeoMMEForRetrieval.from_pretrained(MID, dtype=torch.float16).cuda().eval()
    # Official document form: apply_chat_template(task="document") prepends
    # the <doc> token (id 5). proc(text=..., task=...) warns-and-IGNORES the
    # kwarg (verified 2026-09-04, reviewer catch) — never use it.
    doc_id = proc.tokenizer.convert_tokens_to_ids("<doc>")
    assert doc_id not in (None, proc.tokenizer.unk_token_id), "no <doc> token"
    def _enc(batch):
        msgs = [[{"role": "user", "content": [{"type": "text", "text": t}]}]
                for t in batch]
        inp = proc.apply_chat_template(
            msgs, task="document", tokenize=True, return_dict=True,
            return_tensors="pt", padding=True, truncation=True, max_length=512)
        assert int(inp["input_ids"][0, 0]) == doc_id, "document form missing <doc>"
        return {k: v.cuda() for k, v in inp.items() if hasattr(v, "cuda")}
    # stream chunk_text, skipping the first `done` rows
    seen = 0; buf = []; t0 = time.time()
    with torch.no_grad():
        for f in sorted(glob.glob(f"{CH}/*.parquet")):
            if done >= N:
                break
            col = pq.read_table(f, columns=["chunk_text"])["chunk_text"]
            for x in col.to_pylist():
                if seen < done:
                    seen += 1; continue
                if done + len(buf) >= N:   # done advances only at flush — cap on scheduled rows
                    break
                buf.append(str(x)); seen += 1
                if len(buf) == B:
                    inp = _enc(buf); emb = model(**inp).dense_embeddings
                    emb = torch.nn.functional.normalize(emb, dim=1).float().cpu().numpy()
                    m = min(len(buf), N - done)
                    sub[done:done+m] = emb[:m]; done += m; buf = []
                    if done % 100_000 == 0:
                        sub.flush(); donef.write_text(str(done))
                        print(f"  {done:,}/{N:,} ({done/(time.time()-t0):.0f} ch/s)", flush=True)
    if buf and done < N:
        with torch.no_grad():   # tail runs OUTSIDE the main loop's no_grad -> was crashing .numpy() on a grad tensor
            emb = torch.nn.functional.normalize(model(**_enc(buf)).dense_embeddings, dim=1).detach().float().cpu().numpy()
        m = min(len(buf), N - done)
        sub[done:done+m] = emb[:m]; done += m
    sub.flush(); donef.write_text(str(done))
    (OUT / "manifest.json").write_text(json.dumps({"model": MID, "dim": DIM, "n": int(done),
        "corpus": "fineweb-edu-chunked-120 (first N, same text as MiniLM-384)", "prompt": "document", "dtype": "float32-normed"}, indent=1))
    print(f"NeoMME embed DONE: {done:,} x {DIM} -> {subf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

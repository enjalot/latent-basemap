"""MiniLM 2M mix-pilot draw (owner via overseer 2026-09-06). CPU. Builds TWO 2M draws from local MiniLM-L6-v2
(384-d) embeddings for the mixture-pathology gate:
  MIX      = base (fineweb/redpajama/pile, ~equal thirds, ~68%) + socials (reddit-tldr17/communityarchive/
             bluesky-5m/twitter100m, ~8% each, ~32%) — with per-register labels.
  BASE     = pure base 2M (equal thirds) — the baseline the mix is read against.
Plus per-register val/test held-out (for reception). Deterministic (seed 42).

LOAD GOTCHA (verified 2026-09-07): base corpora are HEADERLESS raw f32 memmaps (dim 384; np.load FAILS — magic is
float data; rows = filesize/4/384). Socials are proper npy f16 (500K×384). Two load paths below.

Output /data2/monet/minilm-mix-2m/ + minilm-base-2m/: {substrate.f32.npy, register.npy (int per corpus),
val_idx/test_idx per register, manifest.json}. Usage: minilm_mix_draw.py [N=2000000] [SEED=42].
"""
import json, sys, glob, os
from pathlib import Path
import numpy as np

EMB = "/data/embeddings"; DIM = 384
BASE = {"fineweb": "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2",
        "redpajama": "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2",
        "pile": "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2"}
SOCIAL = {"reddit": "reddit-tldr17-chunked-120-all-MiniLM-L6-v2",
          "communityarchive": "communityarchive-tweets-all-MiniLM-L6-v2",
          "bluesky": "bluesky-5m-chunked-120-all-MiniLM-L6-v2",
          "twitter100m": "twitter100m-chunked-120-all-MiniLM-L6-v2"}
REGISTERS = list(BASE) + list(SOCIAL)  # label index = position here
CHUNK = 200_000


class Corpus:
    """Unifies base (raw f32 memmap) + social (npy f16) row access -> f32 rows on demand."""
    def __init__(self, name, is_base):
        self.name = name; self.is_base = is_base
        d = f"{EMB}/{BASE[name] if is_base else SOCIAL[name]}/train"
        self.shards = sorted(glob.glob(f"{d}/*.npy"))
        self.offsets = [0]
        for s in self.shards:
            if is_base:
                nrows = os.path.getsize(s) // (4 * DIM)                # headerless raw f32
            else:
                nrows = np.load(s, mmap_mode="r").shape[0]             # proper npy
            self.offsets.append(self.offsets[-1] + nrows)
        self.n = self.offsets[-1]

    def gather(self, global_idx):
        """global_idx: sorted array of row positions in [0,n). Returns (len,384) f32."""
        out = np.empty((global_idx.shape[0], DIM), np.float32)
        for si, s in enumerate(self.shards):
            lo, hi = self.offsets[si], self.offsets[si + 1]
            m = (global_idx >= lo) & (global_idx < hi)
            if not m.any():
                continue
            local = global_idx[m] - lo
            if self.is_base:
                arr = np.memmap(s, dtype=np.float32, mode="r").reshape(-1, DIM)
            else:
                arr = np.load(s, mmap_mode="r")
            out[m] = np.asarray(arr[local], np.float32)
        return out


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_val = n_test = 20_000                                            # per register, held-out for reception
    # composition: socials ~8% each (32%), base equal thirds of the remaining 68%
    soc_each = int(round(0.08 * N)); base_each = (N - soc_each * len(SOCIAL)) // len(BASE)
    plan = {**{k: base_each for k in BASE}, **{k: soc_each for k in SOCIAL}}
    plan[list(BASE)[0]] += N - sum(plan.values())                      # absorb rounding into fineweb
    rng = np.random.default_rng(seed)

    corp = {k: Corpus(k, k in BASE) for k in REGISTERS}
    for k in REGISTERS:
        print(f"[mix] {k}: {corp[k].n:,} rows, plan {plan[k]:,}", flush=True)
        assert plan[k] + n_val + n_test <= corp[k].n, f"{k}: not enough rows"

    def draw_one(counts, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        sub = np.lib.format.open_memmap(f"{out_dir}/substrate.f32.npy", mode="w+", dtype=np.float32,
                                        shape=(sum(counts.values()), DIM))
        reg = np.empty(sum(counts.values()), np.int16)
        val_idx, test_idx = {}, {}
        base_row = 0
        for k, cnt in counts.items():
            perm = rng.permutation(corp[k].n)
            sel = np.sort(perm[:cnt])
            for s in range(0, cnt, CHUNK):
                j = min(s + CHUNK, cnt)
                g = corp[k].gather(sel[s:j])
                g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)  # L2-norm (MiniLM cosine convention)
                sub[base_row + s:base_row + j] = g
            reg[base_row:base_row + cnt] = REGISTERS.index(k)
            val_idx[k] = perm[cnt:cnt + n_val].tolist(); test_idx[k] = perm[cnt + n_val:cnt + n_val + n_test].tolist()
            base_row += cnt
        sub.flush()
        np.save(f"{out_dir}/register.npy", reg)
        (Path(out_dir) / "held_out.json").write_text(json.dumps({"val_idx": val_idx, "test_idx": test_idx}))
        (Path(out_dir) / "manifest.json").write_text(json.dumps({
            "schema": "minilm-mix-pilot-2026-09-06", "n_rows": int(sum(counts.values())), "dim": DIM,
            "composition": counts, "registers": REGISTERS, "seed": seed,
            "note": "L2-normed MiniLM-L6-v2; register.npy labels each row's source corpus; val/test held-out per register"}, indent=1))
        print(f"[mix] {out_dir}: {sum(counts.values()):,} rows, composition {counts}", flush=True)

    draw_one(plan, "/data2/monet/minilm-mix-2m")
    base_plan = {k: N // len(BASE) for k in BASE}; base_plan[list(BASE)[0]] += N - sum(base_plan.values())
    draw_one(base_plan, "/data2/monet/minilm-base-2m")
    print("[mix] DONE (mix + pure-base 2M draws)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

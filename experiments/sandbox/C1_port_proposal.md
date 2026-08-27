# C1 + C2-wiring Port Proposal — high-throughput host-int8 input machinery → generic rankneg path

**Status:** PROPOSAL ONLY. Nothing here is applied. A GPU pipeline is mid-training and
re-imports `core.py` / `edge_list_dataset.py`; applying any of these diffs now would break
production trains. See §6 for the safe apply-window.

**Scope:** Port round0034's fast input machinery — two pinned host slots + combined endpoint gather,
cached binary labels, fused forward (M1/M3/M5/M6) — into the generic
`HostInt8ArrayDataset` + `DeviceEdgeSampler` resident path, **default-off**, with rankneg
(`configure_rank_negatives` / `set_rank_order`) preserved; plus a `sealed_int8_path` knob so a
pre-sealed int8 substrate is consumed without re-encode and without re-normalize. **Cut 1 deliberately
excludes round0034's background producer thread (M4)** — it is a stale-feature correctness hazard on
the generic rankneg path and is deferred to a follow-on (see §2a). Cut 1 also **never pins the full
int8 matrix**, so it holds at 30M where a full-matrix pin would hit the memlock ceiling.

**Files that would change (do NOT edit until §6 window):**
`basemap/pumap/parametric_umap/datasets/edge_list_dataset.py`,
`basemap/pumap/parametric_umap/core.py`,
`experiments/sandbox/image_map_pipeline.py`,
`experiments/sandbox/knobs_2m.py`.
`round0034_pipeline.py` is READ-ONLY reference; it is not touched.

---

## 1. Mechanism inventory (file:line)

### 1a. What round0034 has that makes it fast (`basemap/round0034_pipeline.py`)

| # | Mechanism | Location | What it does |
|---|-----------|----------|--------------|
| M1 | **Two bounded pinned host slots** | `HostInt8MaterializedArray.__init__` `round0034_pipeline.py:771-789` | Allocates exactly 2 slots, each holding `pin_memory` `source_i8`/`destination_i8` (int8, `buffer_rows×dim`) + `source_scale`/`destination_scale` (fp16). A `torch.cuda.Event` per slot (`_wait_slot`, `:829-832`) guards reuse. The full int8 matrix stays **plain numpy** (`self.encoded`, `:758`); only the two small slots are pinned. |
| M2 | **Host-side fancy-index fill (background-fillable)** | `fill_pair_slot` `:853-876` | Copies `self.encoded[source_rows]` / `[destination_rows]` (+ scales) into the pinned slot via `slot["source_i8"].numpy()[:count] = ...`. **Pure host/numpy work, issues zero CUDA** — so it can run in a producer thread. |
| M3 | **COMBINED endpoint gather (one slot, one transfer, split)** | `transfer_pair_slot` `:878-915` | ONE H2D of BOTH endpoints from the **pinned** slot (`.to(device, non_blocking=True)`), then dequant `int8.float()*scale.float()` and returns `(source, destination)` separately. Records the slot's CUDA event (`:908-910`). |
| M4 | **Double-buffered producer thread** | `HostInt8CanonicalSampler._start_prefetch/_prefetch_one/_next_prefetched` `:1015-1110` | Single-worker `ThreadPoolExecutor` pre-fills the *alternate* slot for batch t+1 while the GPU consumes batch t; slot rotation + event sync (`:1101-1107`). |
| M5 | **Cached binary labels** | `HostInt8CanonicalSampler.__init__` `:996-999`, returned in `__next__` `:1124` | `self._labels = cat(ones(num_pos), zeros(num_neg))` built **once** on device; every batch returns the same tensor. No per-batch `ones/zeros/cat`. |
| M6 | **Fused endpoint forward** | flag set `:983` (`self.fused_endpoint_forward = True`); consumed in `core.py:1839-1848` | Signals the loop to run `model(cat(src,dst))` once and split, instead of `model(src)`+`model(dst)`. |

### 1b. What is round0034-SPECIFIC and must NOT be ported

- `HostInt8CanonicalSampler` positive/negative **draw** logic (`_draw_positive_pairs`
  `:1042-1061`, `_draw_negative_pairs` `:1063-1070`, degree/targets/excluded-rows) is the sealed
  R0033/R0034 **canonical-graph** semantics — pure host numpy draws over a bespoke graph, not
  generic NPZ `sources`/`targets`. The generic path draws positives from the NPZ arrays and
  negatives on-device (incl. rankneg). **Only M1–M6 (the *transport*), not the draw, ports.**
- `Round0034TrainingInput` / `prepare_round0034_training` `:1172-1271` — sealed manifest/eligibility
  admission. Untouched; the generic path already has its own admission in `core.py`.
- R0034 is **binary uniform, no weighted sampling, no rankneg** (`:1233-1241` fails closed
  otherwise). Its producer can prefetch the *entire* batch (draw included) because negatives are
  host-drawn. The generic rankneg path **cannot** — see §2c.

### 1c. What the generic resident path LACKS (`edge_list_dataset.py`)

`HostInt8ArrayDataset.index_select` (`edge_list_dataset.py:239-253`) is called **twice per batch**
(once for `all_src`, once for `all_dst`, from `DeviceEdgeSampler.__next__` `:595-596`). Each call:
`idx.detach().to("cpu")` → `self._i8.index_select(0, idx_cpu)` (CPU gather into a **freshly
allocated, non-pinned** tensor) → `.to(device, non_blocking=self._pin)`. Because the gather result
is **pageable, not pinned**, the `non_blocking=True` is a no-op and the H2D is effectively
synchronous. So vs round0034 the generic path is missing: combined single-transfer (M3),
pinned-source async transfer (M1/M3), overlapped background fill (M2/M4), cached labels (M5),
fused forward (M6). `DeviceEdgeSampler` never sets `fused_endpoint_forward`, so `core.py:1839`
takes the two-forward branch.

`DeviceEdgeSampler.__next__` (`:582-593`) also rebuilds `ones`/`zeros`/`cat` for labels every batch
(lacks M5).

---

## 2. The port diffs (default-off; rankneg preserved)

**Design:** one new opt-in constructor flag `host_int8_fast_input` (default `False`). When off, every
path is byte-identical. When on **and** `x_residency="host_int8"`, `HostInt8ArrayDataset` grows the two
pinned slots + a synchronous combined `gather_pairs` (M1/M3) and keeps the codes as plain numpy (no
full-matrix pin), and `DeviceEdgeSampler` uses the combined gather + cached labels (M5) + sets
`fused_endpoint_forward` (M6). **The negative draw —
including rankneg `_sample_negatives` / `_rank_window_dst` — is unchanged**; only the *feature
transport given the already-drawn indices* changes. That is why `configure_rank_negatives` /
`set_rank_order` remain fully functional (proof in §2c).

**Cut 1 ships M1/M3/M5/M6 only — no producer thread (M4), no full-matrix pin.** Two design decisions
that the coordinator flagged as required:

1. **M4 (background producer thread) is DROPPED from cut 1** — it is a genuine correctness hazard as
   naively pipelined (would pair batch *t*'s labels/rankneg draw with batch *t−1*'s features; see the
   M4 follow-on note after §2a for why, and what a correct version requires). Cut 1 uses a
   **synchronous** combined gather. Overlap still happens — the host-side numpy fancy-index of the
   current batch runs on the main thread *while the GPU is still executing the previous batch's queued
   forward/backward kernels* (ordinary async-CUDA overlap), and the two pinned slots + per-slot CUDA
   events let each combined H2D be genuinely non-blocking without clobbering an in-flight transfer.
   No thread, no reordering, no stale-feature risk.
2. **`fast_input=True` NEVER pins (or materializes) the full int8 matrix** — that full-matrix pin is
   the 30M memlock blocker (C3: ~21.5 GiB at 30M×768 > the 15.4 GiB memlock ceiling), which is exactly
   what C1 exists to avoid. In fast mode the codes/scales stay **plain numpy** (page-cache-backed
   memmap on the sealed path, zero-copy) and only the two `buffer_rows`-sized slots are pinned. The
   legacy `index_select` (used only by `_refresh_rank_negatives` at epoch boundaries) falls back to a
   pageable numpy gather — throughput-irrelevant there.

### 2a. `edge_list_dataset.py` — grow `HostInt8ArrayDataset` (fast_input adds slots + combined gather; NO full-matrix pin)

```diff
--- a/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
+++ b/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
@@ class HostInt8ArrayDataset:
     def __init__(self, X, device, *, encoded=None, scales=None,
-                 encode_chunk=1_000_000):
+                 encode_chunk=1_000_000, fast_input=False):
@@
         encoded = np.ascontiguousarray(encoded)
         scales = np.ascontiguousarray(scales)
@@ (invariant checks unchanged) @@
         self.device = device
         self._pin = "cuda" in str(device)
-        self._i8 = torch.from_numpy(encoded)
-        self._scales = torch.from_numpy(scales)
-        if self._pin:
-            self._i8 = self._i8.pin_memory()
-            self._scales = self._scales.pin_memory()
+        self._fast_input = bool(fast_input)
+        if self._fast_input:
+            # C1: DO NOT pin or copy the full matrix — that full-matrix pin is
+            # the 30M memlock blocker (~21.5 GiB @ 30M×768 > 15.4 GiB ceiling).
+            # Keep codes/scales as plain numpy (page-cache-backed memmap on the
+            # sealed path, zero-copy) and gather per batch into two small pinned
+            # slots. `np.ascontiguousarray` above is a NO-OP for the already
+            # C-order sealed memmap, so no 21.5 GiB copy happens here.
+            self._enc_np = encoded          # int8 (N, D), C-order (ndarray or memmap)
+            self._sc_np = scales            # fp16 (N,)
+            self._i8 = None                 # no resident full-matrix tensor
+            self._scales = None
+            self._slots = None              # two pinned slots, sized on first gather
+            self._slot_index = 0
+            self._buffer_rows = 0
+        else:
+            self._i8 = torch.from_numpy(encoded)
+            self._scales = torch.from_numpy(scales)
+            if self._pin:
+                self._i8 = self._i8.pin_memory()
+                self._scales = self._scales.pin_memory()
         self.shape = (int(encoded.shape[0]), int(encoded.shape[1]))
         self.storage_dtype = torch.int8
         self._n = int(encoded.shape[0])
+
+    def _ensure_slots(self, rows):
+        """Allocate the two pinned host slots (M1) once, sized to the batch.
+        `rows` is bounded by batch_size (num_pos+num_neg) and constant across a
+        run, so this allocates exactly once — no growth path, no leak."""
+        if self._slots is not None:
+            if rows > self._buffer_rows:
+                # batch never grows mid-run; treat as a contract violation.
+                raise RuntimeError(
+                    f"gather batch {rows} exceeds slot buffer {self._buffer_rows}")
+            return
+        self._buffer_rows = int(rows)
+        d = self.shape[1]
+        self._slots = []
+        for _ in range(2):
+            si8 = torch.empty((rows, d), dtype=torch.int8, pin_memory=True)
+            di8 = torch.empty_like(si8, pin_memory=True)
+            ssc = torch.empty((rows,), dtype=torch.float16, pin_memory=True)
+            dsc = torch.empty_like(ssc, pin_memory=True)
+            self._slots.append({"src_i8": si8, "dst_i8": di8,
+                                "src_sc": ssc, "dst_sc": dsc, "event": None})
+
+    def gather_pairs(self, src_idx, dst_idx):
+        """Combined endpoint gather for two device LongTensors of equal length.
+        Synchronous (no producer thread): host fancy-index BOTH endpoints into
+        one pinned slot (M1/M2), ONE non-blocking H2D from the pinned slot (M3),
+        dequant + split. Two slots + per-slot CUDA events let the current fill
+        proceed while the previous slot's H2D is still in flight, and overlap the
+        host gather with the GPU compute already queued for the prior batch.
+        Identical dequant arithmetic to two index_select calls. rankneg-agnostic:
+        the caller supplies indices that already include rank-window negatives."""
+        src_np = src_idx.detach().to("cpu", dtype=torch.long).numpy()
+        dst_np = dst_idx.detach().to("cpu", dtype=torch.long).numpy()
+        c = len(src_np)
+        self._ensure_slots(c)
+        si = self._slot_index
+        self._slot_index ^= 1                       # round-robin the two slots
+        slot = self._slots[si]
+        ev = slot["event"]
+        if ev is not None:
+            ev.synchronize()                        # guard: prior H2D on this slot done
+        slot["src_i8"].numpy()[:c] = self._enc_np[src_np]
+        slot["dst_i8"].numpy()[:c] = self._enc_np[dst_np]
+        slot["src_sc"].numpy()[:c] = self._sc_np[src_np]
+        slot["dst_sc"].numpy()[:c] = self._sc_np[dst_np]
+        s_i8 = slot["src_i8"][:c].to(self.device, non_blocking=True)
+        d_i8 = slot["dst_i8"][:c].to(self.device, non_blocking=True)
+        s_sc = slot["src_sc"][:c].to(self.device, non_blocking=True)
+        d_sc = slot["dst_sc"][:c].to(self.device, non_blocking=True)
+        src = s_i8.float() * s_sc.float().view(-1, 1)
+        dst = d_i8.float() * d_sc.float().view(-1, 1)
+        nev = torch.cuda.Event()
+        nev.record(torch.cuda.current_stream(self.device))
+        slot["event"] = nev
+        return src, dst
```

The legacy `index_select` (`edge_list_dataset.py:239-253`) needs a fast-mode fallback because in
fast mode there is no resident `self._i8` tensor. It is called **only** by `_refresh_rank_negatives`
(`core.py:324`) at epoch boundaries and by the optional density/mid-near gathers — never on the
throughput hot path — so a pageable numpy gather is fine:

```diff
@@ class HostInt8ArrayDataset:
     def index_select(self, idx):
+        if self._fast_input:
+            # Epoch-boundary path (rank-order projection / density). No resident
+            # pinned matrix in fast mode: pageable numpy gather -> device.
+            if torch.is_tensor(idx):
+                rows = idx.detach().to("cpu", dtype=torch.long).numpy()
+            else:
+                rows = np.asarray(idx, dtype=np.int64)
+            enc = np.ascontiguousarray(self._enc_np[rows])
+            sc = np.asarray(self._sc_np[rows], dtype=np.float32)
+            rt = torch.from_numpy(enc).to(self.device).float()
+            st = torch.from_numpy(sc).to(self.device).unsqueeze(1)
+            return rt * st
         if not torch.is_tensor(idx):
             idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
         idx_cpu = idx.detach().to("cpu", dtype=torch.long)
         rows_i8 = self._i8.index_select(0, idx_cpu)
         ...
```

The `to()` method (`edge_list_dataset.py:231-237`) re-pins the full matrix; guard it so fast mode
never re-pins (there is nothing resident to pin):

```diff
@@ class HostInt8ArrayDataset:
     def to(self, device):  # int8 rows stay on host; only the dequant target moves
         self.device = device
         self._pin = "cuda" in str(device)
+        if self._fast_input:
+            return self          # no resident matrix to (re-)pin in fast mode
         if self._pin and not self._i8.is_pinned():
             self._i8 = self._i8.pin_memory()
             self._scales = self._scales.pin_memory()
         return self
```

> **M4 (background producer thread) — DROPPED from cut 1; follow-on work.** A naive pipelined producer
> is a **stale-feature correctness bug**, not just a missed optimization. Trace the reuse-current-indices
> design: call 1 (indices I1) fills slot0 with I1, transfers slot0 (correct), then fills slot1 with I1
> **again** and stores it pending; call 2 (indices I2) transfers the pending slot1 = **I1's features**,
> returned for I2's labels and rankneg draw. So every batch *t ≥ 2* pairs batch *t*'s labels/negatives
> with batch *t−1*'s FEATURES — silently wrong, and FFR would drift. Resolving the future in-place
> (`.submit(...).result()`) buys **zero** overlap anyway. A *correct* producer must fill slot *t+1* with
> batch *t+1*'s **own** indices, which requires the SAMPLER to **pre-draw batch t+1's indices** (the
> on-device positive+rankneg draw is cheap, and the rank order is epoch-stable, so drawing one batch
> ahead is sound) and hand those future indices to the fill thread — a `DeviceEdgeSampler` change, not a
> dataset one. That is deferred to a follow-on cut with its own A/B; **cut 1 is synchronous only.** The
> synchronous path already overlaps the host gather with the previous batch's in-flight GPU kernels, so
> most of the transport win is captured without the thread.

### 2b. `edge_list_dataset.py` — `DeviceEdgeSampler`: cached labels (M5), combined gather, fused flag (M6)

```diff
--- a/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
+++ b/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
@@ class DeviceEdgeSampler:
     def __init__(self, dataset, sources, targets, weights, n_nodes,
                  pos_ratio=0.2, batch_size=4096, shuffle=True,
                  random_state=0, positive_target_mode="binary",
                  weighted_edge_sampling=False,
                  uniform_with_replacement=False,
                  positive_source_rows=None,
                  fixed_edges_per_source=None,
-                 device="cpu"):
+                 device="cpu", fast_host_int8_input=False):
         self.dataset = dataset          # DeviceArrayDataset
         self.device = device
@@
         self.num_pos = max(1, int(batch_size * pos_ratio))
         self.num_neg = batch_size - self.num_pos
+        # C1 fast host-int8 transport: only when the dataset supports the
+        # combined gather (HostInt8ArrayDataset with fast_input=True) AND targets
+        # are binary (cached-label path). Off => byte-identical legacy behavior.
+        self._fast_gather = bool(fast_host_int8_input) and hasattr(
+            dataset, "gather_pairs") and positive_target_mode == "binary"
+        self.fused_endpoint_forward = self._fast_gather   # M6; core guards bn/dropout
+        self._cached_labels = None                        # M5, lazily sized
```

```diff
@@ class DeviceEdgeSampler:
     def __next__(self):
@@
         neg_src, neg_dst = self._sample_negatives(self.num_neg)
-        neg_labels = torch.zeros(self.num_neg, dtype=torch.float32,
-                                 device=self.device)
-
-        all_src = torch.cat([p_src, neg_src])
-        all_dst = torch.cat([p_dst, neg_dst])
-        targets = torch.cat([p_labels, neg_labels])
-
-        src_feats = self.dataset.index_select(all_src)
-        dst_feats = self.dataset.index_select(all_dst)
-        return src_feats, dst_feats, targets
+        all_src = torch.cat([p_src, neg_src])
+        all_dst = torch.cat([p_dst, neg_dst])
+
+        if self._fast_gather:
+            # M5: cache the constant [ones(num_pos)|zeros(num_neg)] label vector;
+            # rebuild ONLY on a short final batch (perm mode last chunk).
+            n_pos_b = p_src.shape[0]
+            if n_pos_b == self.num_pos:
+                if self._cached_labels is None:
+                    self._cached_labels = torch.cat([
+                        torch.ones(self.num_pos, dtype=torch.float32, device=self.device),
+                        torch.zeros(self.num_neg, dtype=torch.float32, device=self.device)])
+                targets = self._cached_labels
+            else:
+                targets = torch.cat([
+                    torch.ones(n_pos_b, dtype=torch.float32, device=self.device),
+                    torch.zeros(self.num_neg, dtype=torch.float32, device=self.device)])
+            # M3: one combined gather instead of two index_select calls.
+            src_feats, dst_feats = self.dataset.gather_pairs(all_src, all_dst)
+            return src_feats, dst_feats, targets
+
+        neg_labels = torch.zeros(self.num_neg, dtype=torch.float32,
+                                 device=self.device)
+        targets = torch.cat([p_labels, neg_labels])
+        src_feats = self.dataset.index_select(all_src)
+        dst_feats = self.dataset.index_select(all_dst)
+        return src_feats, dst_feats, targets
```

> `p_labels` (binary → `ones`) is not read in the fast branch; harmless. The `probability` mode never
> enters `_fast_gather` (guarded), so weighted-target semantics are unchanged.

### 2c. Why rankneg still works (the required proof)

`configure_rank_negatives` (`edge_list_dataset.py:468-502`) and `set_rank_order` (`:504-516`) install
`self._rank_window` / `self._rank_of_node` / `self._node_at_rank`. They are consumed **only** inside
`_sample_negatives` (`:545-557` → `_rank_window_dst` `:518-527`), which produces `neg_src` / `neg_dst`
on-device. The diff in §2b leaves `_sample_negatives` and the `all_src = cat(p_src, neg_src)` /
`all_dst = cat(p_dst, neg_dst)` construction **byte-for-byte unchanged**. The fast path only swaps the
*feature transport* (`gather_pairs` instead of two `index_select`) and the *label construction*
(cached constant). Therefore:

- `core.py:1415-1424` still finds `configure_rank_negatives` on the loader (unchanged method).
- `core.py:1787-1793` still calls `_refresh_rank_negatives` → `loader.set_rank_order(...)` each epoch;
  `_refresh_rank_negatives` (`core.py:303-331`) uses `loader.dataset.index_select(idx)` (the legacy
  method, still present) to project the current embedding — untouched.
- The rank-window scale `self._rankneg_scale` and the per-element BCE reweight (`core.py:1883-1888`)
  are downstream of the labels/features and unaffected.

**Net:** rankneg negatives are identical; only how their features reach the GPU changed. FFR-neutral.

---

## 3. `sealed_int8_path` wiring diffs (consume pre-sealed int8 without re-encode / re-normalize)

The C2 builder (`experiments/sandbox/build_int8_substrate.py`) writes a RAW, headerless, C-order
directory: `embeddings.i8` (int8 `(rows,dim)`), `scales.f16` (`<f2 (rows,)`), and `manifest.json`
with `rows`, `dim`, `normalized_before_quant: True` (`build_int8_substrate.py:164-201`). It applies
`_norm` **then** `quantize_int8_rows` at build time (`:139-143`), so the sealed rows are **already
normalized**. Consuming them must (a) skip the loader's re-encode and (b) skip the pipeline's `_norm`.

### 3a. `edge_list_dataset.py` — a thin sealed-substrate loader (loader-compatible with `encoded=`/`scales=`)

```diff
--- a/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
+++ b/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
@@ (module level, near HostInt8ArrayDataset)
+class SealedInt8Substrate:
+    """Memmap view over a C2 sealed int8 substrate directory (embeddings.i8 +
+    scales.f16 [+ manifest.json]). Presents `.shape`, `__len__`, and a
+    dequantizing `__getitem__` so it works as X for BOTH `fit` (host_int8 branch
+    reads `.encoded`/`.scales` directly, no re-encode) and `transform` (slices
+    return dequantized fp32 rows, never materializing the whole array).
+
+    Marker attributes let callers assert the no-double-norm contract:
+      * ``sealed_int8`` is True   -> _norm() must refuse it (fail closed).
+      * ``_prenormalized`` is True -> the sealed branch asserts this before use.
+    """
+    sealed_int8 = True
+    _prenormalized = True
+
+    def __init__(self, path, dim=None, verify_hashes=True):
+        import json, os, hashlib
+        path = str(path)
+        i8p = os.path.join(path, "embeddings.i8")
+        scp = os.path.join(path, "scales.f16")
+        manp = os.path.join(path, "manifest.json")
+        rows = None; man = None
+        if os.path.isfile(manp):
+            man = json.loads(open(manp).read())
+            rows, dim = int(man["rows"]), int(man["dim"])
+            if not man.get("normalized_before_quant", False):
+                raise ValueError("sealed substrate manifest is not normalized_before_quant")
+        if dim is None:
+            raise ValueError("SealedInt8Substrate needs dim (manifest.json absent)")
+        i8_bytes = os.path.getsize(i8p); sc_bytes = os.path.getsize(scp)
+        if rows is None:
+            rows = i8_bytes // dim
+        if i8_bytes != rows * dim or sc_bytes != rows * 2:
+            raise ValueError(f"sealed byte geometry mismatch: i8={i8_bytes} sc={sc_bytes} "
+                             f"rows={rows} dim={dim}")
+        # review #9: LOADER VERIFIES HASHES. A sealed substrate built by the
+        # hardened build_int8_substrate.py carries manifest["sha256"]; re-stream
+        # both files (1 MiB chunks, never materialised) and fail closed on any
+        # mismatch before the 30M train trusts a corrupt/tampered/truncated seal.
+        # One-time ~i8_bytes read at open; set verify_hashes=False only for a
+        # throwaway re-open of an already-verified seal in the same process.
+        want = (man or {}).get("sha256")
+        if verify_hashes and want:
+            for fname, want_hex in want.items():
+                h = hashlib.sha256()
+                with open(os.path.join(path, fname), "rb") as f:
+                    for chunk in iter(lambda: f.read(1 << 20), b""):
+                        h.update(chunk)
+                if h.hexdigest() != want_hex:
+                    raise ValueError(f"sealed {fname}: SHA-256 mismatch vs manifest "
+                                     f"(corrupt/tampered seal) at {path}")
+        elif verify_hashes and man is not None:
+            raise ValueError("sealed manifest has no sha256 block — rebuild with the "
+                             "hardened build_int8_substrate.py before training on it")
+        self.encoded = np.memmap(i8p, dtype=np.int8, mode="r", shape=(rows, dim))
+        self.scales = np.memmap(scp, dtype="<f2", mode="r", shape=(rows,))
+        self.shape = (rows, dim)
+
+    def __len__(self):
+        return self.shape[0]
+
+    def __getitem__(self, sl):
+        enc = np.asarray(self.encoded[sl], dtype=np.float32)
+        sc = np.asarray(self.scales[sl], dtype=np.float32)
+        return enc * sc[..., None] if enc.ndim == 2 else enc * sc
+
+
+def load_sealed_int8_substrate(path, dim=None):
+    return SealedInt8Substrate(path, dim=dim)
```

### 3b. `core.py` — host_int8 branch consumes sealed encoded/scales without re-encoding

```diff
--- a/basemap/pumap/parametric_umap/core.py
+++ b/basemap/pumap/parametric_umap/core.py
@@ def __init__(self, ...):
         x_residency="auto",
+        # C1: opt-in high-throughput host-int8 transport (pinned slots, combined
+        # gather, cached labels, fused forward). "auto"/off => byte-identical.
+        host_int8_fast_input=False,
         graph_manifest_path=None,
@@
         self.x_residency = x_residency
+        self.host_int8_fast_input = bool(host_int8_fast_input)
```

```diff
@@ (host_int8 branch, core.py:695-731)
         if str(self.x_residency).lower() == "host_int8":
             if edge_set is not None:
                 raise RuntimeError(
                     "x_residency='host_int8' is incompatible with reject_neighbors ...")
@@
             reason = "x_residency=host_int8 (int8 X on host, dequant per batch)"
             logging.info("Edge-list mode: HOST-INT8 residency (%s).", reason)
             _stamp_pipeline(
                 "host_int8", "DeviceEdgeSampler", weighted_ok=True,
                 x_residency="host_int8",
                 uniform_with_replacement=device_uniform_replacement)
-            hi8 = HostInt8ArrayDataset(X, self.device)
+            if getattr(X, "sealed_int8", False):
+                # No-double-norm guard (see §4): the sealed substrate is already
+                # _norm'd + quantized; assert it and consume its codes/scales
+                # directly (encoded=/scales=), bypassing HostInt8ArrayDataset's
+                # re-encode path entirely. No _norm, no quantize here.
+                assert getattr(X, "_prenormalized", False) is True, (
+                    "sealed_int8 substrate must be pre-normalized; refusing to "
+                    "feed a non-prenormalized sealed X to the host-int8 loader")
+                # Feed the memmaps DIRECTLY (no np.asarray): with fast_input=True
+                # the loader keeps them as plain numpy (page-cache-backed,
+                # zero-copy) and never pins the full matrix — the 30M path.
+                hi8 = HostInt8ArrayDataset(
+                    None, self.device,
+                    encoded=X.encoded, scales=X.scales,
+                    fast_input=self.host_int8_fast_input)
+            else:
+                hi8 = HostInt8ArrayDataset(
+                    X, self.device, fast_input=self.host_int8_fast_input)
             self._X_dev = hi8
             self._fast_device_path = True
             loader = DeviceEdgeSampler(
                 hi8, sources, targets, weights, n_nodes=n_train,
                 pos_ratio=self.pos_ratio, batch_size=self.batch_size,
                 shuffle=True, random_state=random_state,
                 positive_target_mode=self.positive_target_mode,
                 weighted_edge_sampling=self.weighted_edge_sampling,
                 uniform_with_replacement=uwr,
                 positive_source_rows=positive_source_rows,
                 fixed_edges_per_source=fixed_edges_per_source,
-                device=self.device,
+                device=self.device,
+                fast_host_int8_input=self.host_int8_fast_input,
             )
             return hi8, loader, n_pos_edges
```

> **Zero-copy on the fast path.** `X.encoded`/`X.scales` are memmaps passed straight through; with
> `fast_input=True` the loader stores them as `_enc_np`/`_sc_np` without pinning or copying the full
> matrix (`np.ascontiguousarray` is a no-op on the already-C-order memmap), so resident/pinned bytes
> stay O(2 slots) regardless of N — the whole point of C1 at 30M. `fit` reads `X.shape[1]`/`[0]` from
> the wrapper (`.shape`); `transform` slices the wrapper (dequantized fp32) — no full fp32 array is ever
> built. Save/load and stamps are unchanged.
>
> **Sealed WITHOUT fast_input (small-scale only):** if `host_int8_fast_input=False` on a sealed X, the
> loader takes the legacy branch — `torch.from_numpy(memmap).pin_memory()` materializes + pins the full
> int8 payload (fine at ≤2M, the C3 memlock blocker at 30M). Sealed substrates at scale MUST pair with
> `host_int8_fast_input=True`; the §5 checklist asserts the no-full-pin invariant.

### 3c. `image_map_pipeline.py` — DATASETS knob + branch the substrate load (no `_norm` on sealed)

```diff
--- a/experiments/sandbox/image_map_pipeline.py
+++ b/experiments/sandbox/image_map_pipeline.py
@@ def train(ds: str) -> int:
     from basemap.pumap.parametric_umap.core import ParametricUMAP
+    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
+        load_sealed_int8_substrate)
@@
-    x = _norm(DATASETS[ds]["load"]())
+    sealed = DATASETS[ds].get("sealed_int8_path")
+    if sealed:
+        # Pre-sealed int8 substrate: already _norm'd + quantized at build time.
+        # Load a dequantizing memmap view and DO NOT call _norm (see §4 guard).
+        x = load_sealed_int8_substrate(sealed, dim=DATASETS[ds].get("int8_dim"))
+        assert getattr(x, "_prenormalized", False) is True, (
+            "sealed_int8_path substrate is not pre-normalized; refusing to train")
+        # NB: _norm(...) is intentionally NOT applied on this branch.
+    else:
+        x = _norm(DATASETS[ds]["load"]())
```

A sealed dataset entry then looks like (arms unchanged; opt into fast transport via `extra`):

```python
"minilm-mix-500k-sealed": {
    "load": None,                       # unused on the sealed branch
    "sealed_int8_path": "/data/latent-basemap/substrates/minilm-mix-500k-int8",
    "int8_dim": 384,                    # fallback if manifest.json absent
    "subsets": None,
    "arms": {"int8fac-hostint8-fast": {"md": "000", "dose": 4, "extra": {
        "fneg_weight": 1.0, "neg_tanh_gamma": 4.0, "pos_ratio": 0.10,
        "rankneg_window": 125_000, "batch_size": 16384,
        "x_residency": "host_int8", "host_int8_fast_input": True}}},
},
```

> `knobs_2m.run_arm` (`knobs_2m.py:539`) loads `np.load(rung["substrate"], mmap_mode="r")` and **never
> calls `_norm`** (its f32 substrates are pre-normalized). To use a sealed substrate there, add the same
> `if rung.get("sealed_int8_path"): X = load_sealed_int8_substrate(...)` branch at `:539`; there is no
> `_norm` to guard on that file, so only the loader swap + the `host_int8_fast_input` kwarg (already in
> `BASE_KWARGS`/`overrides`) are needed.

---

## 4. Explicit no-double-norm guard (assert, not convention)

The hazard: the sealed substrate rows are already unit-norm (up to int8 quant error, ~1e-2 per element
per `manifest.max_abs_quant_error_vs_normalized`). Dequantized rows have `‖row‖ ≈ 0.998–1.002`, so a
second `_norm` would divide each row by a near-1 but **not-exactly-1** factor — a silent per-row shift,
not a no-op. Three asserts fail closed on every route to `_norm`:

1. **In `image_map_pipeline._norm` itself (primary, defense-in-depth):** the function refuses any
   sealed substrate, so *no* code path can normalize one, even a future caller.

   ```diff
   --- a/experiments/sandbox/image_map_pipeline.py
   +++ b/experiments/sandbox/image_map_pipeline.py
    def _norm(x: np.ndarray) -> np.ndarray:
   +    assert not getattr(x, "sealed_int8", False), (
   +        "_norm() called on a sealed pre-normalized int8 substrate — "
   +        "double-normalize guard (build_int8_substrate already _norm'd it)")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n
   ```

2. **At the sealed load site in `train()` (§3c):** `assert getattr(x, "_prenormalized", False) is True`
   — positively confirms the object on the sealed branch is the pre-normalized wrapper, and the branch
   structurally does not reach `_norm(...)`.

3. **At the core host_int8 consumption site (§3b):** `assert getattr(X, "_prenormalized", False) is
   True` before reading `X.encoded`/`X.scales` — the loader refuses to consume a sealed X that does not
   carry the pre-normalized marker.

Together: the sealed branch **asserts** `_prenormalized is True` at both the pipeline and loader, and
`_norm` **asserts** it is never handed a `sealed_int8` object. A double-normalize is impossible without
tripping an assert.

---

## 5. ≥85%-throughput 2M A/B validation plan (+ FFR parity)

**Control (current resident path, rankneg on):** the existing arm
`umap-md000-x4bs16k-winner-hostint8` (`knobs_2m.py:385-387`) — 2M rung, `x_residency="host_int8"`,
`rankneg_window=500_000`, `batch_size=16384`, dose x4. Runs `HostInt8ArrayDataset` (legacy two
`index_select`) + `DeviceEdgeSampler`.

**Treatment (ported path):** same arm + `host_int8_fast_input=True` (the ONLY delta). Add as a sibling
arm so `receipt_diff` records the one-knob departure:

```python
# knobs_2m.py ARMS (proposal — add alongside the control)
"umap-md000-x4bs16k-winner-hostint8-fast": _umap(
    "000", dose=4, fneg_weight=1.0, neg_tanh_gamma=4.0, pos_ratio=0.10,
    rankneg_window=500_000, batch_size=16384, x_residency="host_int8",
    host_int8_fast_input=True),
```

**Commands (CPU-launch, single GPU, run only in the §6 window):**

```bash
cd /home/enjalot/code/latent-basemap
# 1) dry-run receipt check both arms (no GPU)
.venv/bin/python experiments/sandbox/knobs_2m.py 2m umap-md000-x4bs16k-winner-hostint8      --dry-run
.venv/bin/python experiments/sandbox/knobs_2m.py 2m umap-md000-x4bs16k-winner-hostint8-fast --dry-run
# 2) control then treatment (write-once out dirs; same seed 42)
.venv/bin/python experiments/sandbox/knobs_2m.py 2m umap-md000-x4bs16k-winner-hostint8
.venv/bin/python experiments/sandbox/knobs_2m.py 2m umap-md000-x4bs16k-winner-hostint8-fast
```

**Throughput metric:** updates/s = `positive_lr_optimizer_steps / wall_s` from each arm's
`summary.json` (`realized_updates`, `wall_s`). Both arms run identical horizons, so compare wall clock
directly.

**Pass bars (hard):**
- **Throughput:** **treatment updates/s ≥ control updates/s** — the port must not regress and should
  show the combined-gather/pinned-transfer win. This is the binding bar.
- **FFR parity:** `quick_ffr_at_0.1pct` (control) vs (treatment) within **±0.005** (noise band already
  used for the int8 parity arm, `knobs_2m.py:383`). Because the fast path is a transport/label/forward
  refactor with **bit-identical dequant, identical rankneg draws, and MLP-fused forward equal to the
  two-forward split** (no batchnorm/dropout), FFR should match to well within noise. A drift beyond
  ±0.005 signals a real bug (e.g., slot reuse race, label misalignment) → do not ship.

**Directional only (not a pass/fail bar):** round0034 hit **122.6 updates/s at 30M**, a *different*
batch geometry / model / graph than the 2M knobs arm — so it is a sanity reference for "are we in the
right order of magnitude for the transport," NOT a threshold the 2M arm must clear. Record the 2M
treatment updates/s and note it alongside 122.6/s for context; do not gate on it.

**30M-shaped memory assertion (REQUIRED — the whole reason for `fast_input`):** confirm that with
`host_int8_fast_input=True` the resident/pinned footprint is **O(2 slots), never O(N)**. Cheap CPU-only
checks (no 30M run needed):
- Static: in the fast branch there is no `self._i8` / `.pin_memory()` on the full matrix (grep the
  applied diff); only the two `buffer_rows`-sized slots are pinned. `buffer_rows == batch_size`
  (16384), so pinned host bytes ≈ `2 slots × 2 endpoints × 16384 × (dim×1 B int8 + 2 B fp16)` ≈ a few
  MiB — independent of N.
- Runtime spot-check (2M arm is enough to prove the invariant): during the treatment run, sample
  `VmLck` / pinned bytes (e.g. `grep VmLck /proc/<pid>/status`, or CUDA pinned-alloc counters) and
  assert it does **not** scale with the substrate row count. Extrapolation: at 30M×768 the *dropped*
  full-matrix pin would have been ~21.5 GiB > the 15.4 GiB memlock ceiling (C3); the fast path must
  show none of that growth.
- Sealed zero-copy: confirm RSS does not jump by the full int8 payload at load (memmap is page-cache
  backed, faulted lazily by the per-batch fancy-index), i.e. no eager 21.5 GiB materialization.

**Sealed-path A/B (separate, optional):** build a 2M sealed substrate with
`build_int8_substrate.py`, add a `sealed_int8_path` twin of the 500k int8 dataset in
`image_map_pipeline.py`, and confirm `int8fac-hostint8` (re-encode) vs `...-sealed` (no re-encode)
produce **identical** FFR (the sealed codes are bit-identical to the in-loader encode per
`build_int8_substrate.validate`), with the sealed run skipping the encode wall-time.

---

## 5b. Three admission gates for 30M (fail-closed) — review #10

30M admission on the host-int8 path is gated by **three independent checks**. All
three must PASS (or gate 3 must be explicitly owner-accepted) before a 30M run is
launched. They test *different* things and must not be conflated:

**Gate 1 — transport equivalence (bit-identical).** The `host_int8_fast_input`
port must be bit-identical to the resident-int8 control it replaces: same dequant
bytes, same rankneg draws, fused forward == two-forward split. Measured by the §5
2M A/B (`umap-md000-x4bs16k-winner-hostint8` vs `...-hostint8-fast`, same seed):
FFR within ±0.005 AND the sealed-path bitwise check (`build_int8_substrate
--validate`, now PASS). This gates the CODE PORT only — it says the fast transport
did not corrupt anything. **Status: PENDING** (runs in the §6 quiescent window;
sealed bitwise sub-check already PASS).

**Gate 2 — throughput ≥ 85% of RESIDENT at the true jina shape.** The int8 path's
updates/s must be ≥ 85% of the fp16/fp32-RESIDENT path measured at the *actual*
jina D768 batch geometry (not a MiniLM D384 proxy, not the 30M round0034 D-shape).
This is what makes int8 admission worth its quality cost. **Status: UNMEASURED at
the jina shape** — the §5 A/B measures transport (int8-fast vs int8-control), not
int8-vs-RESIDENT throughput; a jina-shape resident-vs-int8 bench is required
(folds into the #12 perf-tail rebuild at jina D768 h2048/h3072).

**Gate 3 — quality within ±0.005 of RESIDENT (the int8 tax).** The int8-QUANTIZED
substrate's FFR must be within ±0.005 of the fp16-RESIDENT substrate at the same
recipe/shape. This is the *quantization* cost (fp16 → int8 per-row), NOT the
transport port (gate 1, bit-identical). **The tax GROWS with N**, so a small-N pass
does not license 30M. Two measured points on disk (both computed as resident −
hostint8, sole delta `x_residency`):

| shape | resident (v2) | hostint8 (v2) | tax v1 / v2 | vs ±0.005 |
|---|---|---|---|---|
| 500K MiniLM D384 (`int8fac`) | 0.4429 | 0.4391 | +0.0034 / +0.0038 | **PASS** |
| 2M MiniLM D384 (`2m-knobs/umap-md000-x4bs16k-winner[-hostint8]`) | 0.47954 | 0.46621 | +0.01227 / +0.01333 | **FAIL** |
| jina D768 (true shape) | — | — | — | **UNMEASURED** |

The 2M point is the binding evidence: **+0.0133 FAILS ±0.005** under the corrected
v2 metric, and the tax roughly quadrupled from 500K→2M — the quantization scheme
carries ≈70% of it (int8fac qdq-vs-hostint8 attribution), which is the fix lever.
Do NOT read the 500K PASS as the state of the world. (My first search reported
−0.0123 "unreproducible" — that was a scope error: it is a computed difference of
two persisted summaries, not a stored literal; verified 2026-08-27.) **RULE: 30M is
BLOCKED until gate 3 is measured-and-passed at the true jina shape, OR the owner
explicitly accepts the measured tax** (which at 2M is already ~2.7× the band).

**Measurement methods (no proxies).** Gate 2 (throughput) is measured by the
jina-shape within-run segment bench (`perf_bench` rebuilt at D768 h2048+h3072,
int8 vs fp16-resident updates/s; don't-revisit variants dropped; the segment
machinery is reused later for the B2 levers — BF16, combined-gather, cached
labels, strided fneg/clip). Gate 3 (the int8 tax) is measured by a FULL parity
twin ARM at the true shape — `jina-multi-2m champion-bs16k` (fp16 resident, on
disk) vs a `champion-bs16k-hostint8` twin whose sole delta is `x_residency`
(~1.5 h) — NOT a truncated-horizon quality segment (a short quality readout is a
proxy, and the campaign's standing lesson is that proxies don't measure the real
thing). Both re-queue after Phase A.

Gate-check discipline: a `c1_admission_gates.py` reads each arm's `summary.json`
and reports PASS/FAIL/UNMEASURED per gate, defaulting UNMEASURED→BLOCK (fail-
closed). It must never infer a gate from a proxy shape (MiniLM D384 does not
satisfy a jina D768 gate) and must refuse to emit ADMIT while any gate is
UNMEASURED. (Script deferred to the §6 apply-window with the A/B it scores.)

---

## 6. Risks + safe apply-window

**Apply-window (hard gate):** `core.py` and `edge_list_dataset.py` are re-imported by the live GPU
pipeline (round runner + Phase A trains). **Do NOT apply any diff until the current sweep + Phase A
trains have finished** — verify the GPU is idle and no `ParametricUMAP.fit` process is live
(`nvidia-smi` + `ps` lineage, per the workspace liveness rule) before editing. All changes are
default-off, but an in-flight process holds the OLD module in memory and a *new* process spawned after
the edit would import the changed module mid-sweep — inconsistent. Apply in a quiescent window, run the
§5 A/B, then let production resume.

**Risks:**
- **Pinned-slot reuse race (M1/M3):** before overwriting a slot the GPU may still be transferring,
  `gather_pairs` calls `ev.synchronize()` on that slot's CUDA event (recorded after the prior H2D on
  that slot). Cut 1 is single-threaded, so fill and transfer are serialized on the main thread and the
  only concurrency is the async H2D itself — fully covered by the per-slot event (mirrors round0034
  `:829-832`). No producer thread, so no cross-thread race.
- **Slot allocation is one-shot (no leak):** `_ensure_slots` allocates exactly once (batch size is
  constant across a run) and raises if a later batch exceeds `buffer_rows` rather than reallocating —
  so there is no per-batch pinned-buffer churn and no growth leak. (The earlier draft's
  `ThreadPoolExecutor` in `_ensure_slots` is gone with M4; nothing to leak.)
- **No full-matrix pin at scale:** the fast branch never builds/pins `self._i8`; the §5 memory
  assertion gates this. A regression that reintroduces the full-matrix pin would pass 2M but hit the
  15.4 GiB memlock ceiling at 30M — the assertion catches it at 2M.
- **Fast-mode teardown:** the two pinned slots are freed with the dataset at GC; no thread to join.
  If an explicit release is wanted, add it to the **`fit`-end / `DeviceEdgeSampler` teardown** path
  (near `core.py:2438`, where `HostStreamEdgeSampler` producer threads are stopped) — NOT inside
  `HostStreamEdgeSampler.close()` (a different sampler that cut 1 does not use). Cut 1 has no producer
  thread, so this is optional hygiene, not a stall risk.
- **Partial final batch (M5):** perm-mode (non-uwr) last chunk has `n_pos_b < num_pos`; the diff
  rebuilds labels for that batch. The 2M rankneg arms use non-uwr perm mode, so this path is exercised
  — the guard is required, not theoretical.
- **Fused forward vs batchnorm/dropout (M6):** `core.py:1840-1842` already raises if bn/dropout are on.
  The knobs/image arms use a plain MLP (no bn/dropout), so fused == two-forward exactly. Any arm that
  turns bn/dropout on will fail closed rather than silently diverge.
- **Sealed double-norm:** covered by the three asserts in §4; impossible to normalize a sealed
  substrate without an AssertionError.
- **Sealed geometry mismatch:** `SealedInt8Substrate.__init__` validates
  `i8_bytes == rows*dim and sc_bytes == rows*2` (same contract as
  `build_int8_substrate.py:161-162` and `int8_eligibility.py:303-304`) → a truncated/rewritten file
  fails at load, not mid-train.
- **`prepare_round0034_training` / `prepare_round0042_training` unaffected:** those adapters return
  before the host_int8 branch (`core.py:412-459`); the new flag never reaches them.

---

## Appendix — one-flag summary (cut 1)

| Flag / knob | Default | Effect when set |
|---|---|---|
| `ParametricUMAP(host_int8_fast_input=True)` | `False` | On the `x_residency="host_int8"` path only: combined pinned-slot gather (M1/M3), cached labels (M5), fused forward (M6), and **no full-matrix pin**. Off ⇒ byte-identical legacy. |
| DATASETS `sealed_int8_path` (+`int8_dim`) | absent | `train()` loads a `SealedInt8Substrate` (no `_norm`, no re-encode) instead of `_norm(load())`. Pair with `host_int8_fast_input=True` for the zero-copy / no-pin benefit at scale. |

**Not in cut 1 (follow-on):** M4 background producer thread — requires a `DeviceEdgeSampler` change to
pre-draw batch *t+1*'s indices (so the fill has genuine future indices, not stale current ones); it has
its own A/B and is deferred. Cut 1 is synchronous and already captures the transport win via
async-CUDA overlap of the host gather with the prior batch's queued GPU kernels.
```

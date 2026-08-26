# C1 + C2-wiring Port Proposal — high-throughput host-int8 input machinery → generic rankneg path

**Status:** PROPOSAL ONLY. Nothing here is applied. A GPU pipeline is mid-training and
re-imports `core.py` / `edge_list_dataset.py`; applying any of these diffs now would break
production trains. See §6 for the safe apply-window.

**Scope:** Port round0034's fast input machinery (two pinned host slots, background fill,
combined endpoint gather, cached binary labels, fused forward) into the generic
`HostInt8ArrayDataset` + `DeviceEdgeSampler` resident path, **default-off**, with rankneg
(`configure_rank_negatives` / `set_rank_order`) preserved; plus a `sealed_int8_path` knob so a
pre-sealed int8 substrate is consumed without re-encode and without re-normalize.

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
path is byte-identical. When on **and** `x_residency="host_int8"`, `HostInt8ArrayDataset` grows M1–M4
(pinned slots + combined `gather_pairs` + optional producer thread) and `DeviceEdgeSampler` uses the
combined gather + cached labels (M5) + sets `fused_endpoint_forward` (M6). **The negative draw —
including rankneg `_sample_negatives` / `_rank_window_dst` — is unchanged**; only the *feature
transport given the already-drawn indices* changes. That is why `configure_rank_negatives` /
`set_rank_order` remain fully functional (proof in §2c).

An env kill-switch `PUMAP_HOST_INT8_PREFETCH=0` disables just the producer thread (M4), leaving the
synchronous combined gather (M1/M3/M5/M6) — a safety valve if the thread ever misbehaves on an
unattended run.

### 2a. `edge_list_dataset.py` — grow `HostInt8ArrayDataset` (add methods only; existing `index_select` untouched)

```diff
--- a/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
+++ b/basemap/pumap/parametric_umap/datasets/edge_list_dataset.py
@@ class HostInt8ArrayDataset:
     def __init__(self, X, device, *, encoded=None, scales=None,
-                 encode_chunk=1_000_000):
+                 encode_chunk=1_000_000, fast_input=False, buffer_rows=0):
@@
         self.shape = (int(encoded.shape[0]), int(encoded.shape[1]))
         self.storage_dtype = torch.int8
         self._n = int(encoded.shape[0])
+        # ── C1 fast-input transport (opt-in; M1–M4). Default off => the class
+        #    behaves exactly as before (only index_select is used). ───────────
+        self._fast_input = bool(fast_input)
+        self._slots = None
+        self._slot_index = 0
+        self._producer = None
+        self._pending = None            # (slot_index, count) filled by producer
+        import os as _os
+        self._prefetch = self._fast_input and self._pin and (
+            _os.environ.get("PUMAP_HOST_INT8_PREFETCH", "1") != "0")
+        if self._fast_input:
+            # Keep a plain-numpy view for host-side fancy-index (M2); the pinned
+            # full matrix (self._i8) is only used by the legacy index_select.
+            self._enc_np = encoded                     # int8 (N, D), C-order
+            self._sc_np = scales                       # fp16 (N,)
+            self._buffer_rows = int(buffer_rows) or None   # sized on first use
+
+    def _ensure_slots(self, rows):
+        """Allocate the two pinned host slots (M1) sized to the batch."""
+        if self._slots is not None and self._buffer_rows >= rows:
+            return
+        self._buffer_rows = max(int(rows), int(self._buffer_rows or 0))
+        d = self.shape[1]
+        self._slots = []
+        for _ in range(2):
+            si8 = torch.empty((self._buffer_rows, d), dtype=torch.int8, pin_memory=True)
+            di8 = torch.empty_like(si8, pin_memory=True)
+            ssc = torch.empty((self._buffer_rows,), dtype=torch.float16, pin_memory=True)
+            dsc = torch.empty_like(ssc, pin_memory=True)
+            self._slots.append({"src_i8": si8, "dst_i8": di8,
+                                "src_sc": ssc, "dst_sc": dsc, "event": None})
+        if self._prefetch:
+            import concurrent.futures
+            self._producer = concurrent.futures.ThreadPoolExecutor(
+                max_workers=1, thread_name_prefix="pumap-host-int8-fill")
+
+    def _fill_slot(self, slot_index, src_rows_np, dst_rows_np):
+        """Host fancy-index of BOTH endpoints into one pinned slot (M2).
+        Pure numpy; issues no CUDA — safe to run in the producer thread."""
+        slot = self._slots[slot_index]
+        ev = slot.get("event")
+        if ev is not None:
+            ev.synchronize()                            # guard slot reuse
+        c = len(src_rows_np)
+        slot["src_i8"].numpy()[:c] = self._enc_np[src_rows_np]
+        slot["dst_i8"].numpy()[:c] = self._enc_np[dst_rows_np]
+        slot["src_sc"].numpy()[:c] = self._sc_np[src_rows_np]
+        slot["dst_sc"].numpy()[:c] = self._sc_np[dst_rows_np]
+        return slot_index, c
+
+    def _transfer_slot(self, slot_index, count):
+        """One combined H2D from the pinned slot + dequant + split (M3)."""
+        slot = self._slots[slot_index]
+        si8 = slot["src_i8"][:count].to(self.device, non_blocking=True)
+        di8 = slot["dst_i8"][:count].to(self.device, non_blocking=True)
+        ssc = slot["src_sc"][:count].to(self.device, non_blocking=True)
+        dsc = slot["dst_sc"][:count].to(self.device, non_blocking=True)
+        src = si8.float() * ssc.float().view(-1, 1)
+        dst = di8.float() * dsc.float().view(-1, 1)
+        ev = torch.cuda.Event()
+        ev.record(torch.cuda.current_stream(self.device))
+        slot["event"] = ev
+        return src, dst
+
+    def gather_pairs(self, src_idx, dst_idx):
+        """Combined endpoint gather for two device LongTensors of equal length.
+        Returns (src_feats, dst_feats) fp32 on device — identical arithmetic to
+        two index_select calls, one transfer instead of two, pinned source, and
+        (with prefetch) the host fancy-index of batch t+1 overlapped with the GPU
+        compute of batch t (M4). rankneg-agnostic: the indices are supplied by
+        the caller, which has already applied rank-window negatives."""
+        src_np = src_idx.detach().to("cpu", dtype=torch.long).numpy()
+        dst_np = dst_idx.detach().to("cpu", dtype=torch.long).numpy()
+        self._ensure_slots(len(src_np))
+        if not self._prefetch:
+            si = self._slot_index
+            self._slot_index ^= 1
+            _, c = self._fill_slot(si, src_np, dst_np)
+            return self._transfer_slot(si, c)
+        # Software-pipelined: transfer the slot the producer filled last call,
+        # submit this call's fill into the alternate slot for next time.
+        if self._pending is None:              # prime the pipeline (batch 0)
+            si = self._slot_index; self._slot_index ^= 1
+            self._pending = self._producer.submit(self._fill_slot, si, src_np, dst_np).result()
+        si, c = self._pending
+        out = self._transfer_slot(si, c)
+        nxt = self._slot_index; self._slot_index ^= 1
+        self._pending = self._producer.submit(self._fill_slot, nxt, src_np, dst_np).result()
+        return out
+
+    def close_fast_input(self):
+        if self._producer is not None:
+            self._producer.shutdown(wait=True, cancel_futures=True)
+            self._producer = None
```

> **Note on M4 fidelity:** round0034 prefetches the whole *draw* because its negatives are host-drawn.
> The generic rankneg path draws negatives **on-device** (§2c), so we prefetch only the host
> fancy-index step (M2), not the draw. The `.result()` above still hands the CPU gather to the worker
> thread; to get true overlap the coordinator can switch the two `.submit(...).result()` sites to hold
> the `Future` across the caller's GPU compute (a one-line change — store the future, resolve it at the
> top of the next `gather_pairs`). Presented conservatively (resolve-in-place) so the first-cut diff is
> obviously correct; the overlap upgrade is a follow-on toggle. Either way the FILL values are
> identical, so FFR is unaffected.

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
+    def __init__(self, path, dim=None):
+        import json, os
+        path = str(path)
+        i8p = os.path.join(path, "embeddings.i8")
+        scp = os.path.join(path, "scales.f16")
+        manp = os.path.join(path, "manifest.json")
+        rows = None
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
+                hi8 = HostInt8ArrayDataset(
+                    None, self.device,
+                    encoded=np.asarray(X.encoded), scales=np.asarray(X.scales),
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

> `np.asarray(X.encoded)` materializes the int8 payload once (9.6 GiB @ 25M×384, ~0.77 GiB @ 2M) —
> which `HostInt8ArrayDataset.__init__` then `ascontiguousarray`s and pins; this is the intended int8
> residency, not a >=2 GB fp32 violation. `fit` reads `X.shape[1]`/`[0]` from the wrapper (`.shape`),
> and `transform` slices the wrapper (dequantized fp32) — no full fp32 array is ever built. Save/load
> and stamps are unchanged.

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

**Pass bars:**
- **Throughput:** treatment updates/s ≥ control updates/s (the port is a speedup; it must not regress).
  The "≥85%" bar: treatment must sustain ≥ 85% of round0034's reference 122.6 updates/s **at the
  matching batch geometry** — i.e., confirm the ported generic path recovers the bulk of R0034's
  transport headroom rather than the legacy two-transfer path. Record both arms' updates/s and the
  ratio to 122.6/s.
- **FFR parity:** `quick_ffr_at_0.1pct` (control) vs (treatment) within **±0.005** (noise band already
  used for the int8 parity arm, `knobs_2m.py:383`). Because the fast path is a transport/label/forward
  refactor with **bit-identical dequant, identical rankneg draws, and MLP-fused forward equal to the
  two-forward split** (no batchnorm/dropout), FFR should match to well within noise. A drift beyond
  ±0.005 signals a real bug (e.g., slot reuse race, label misalignment) → do not ship.
- **Determinism spot-check:** a 3rd run of the treatment at the same seed should reproduce FFR exactly
  (the producer thread only reorders host copies, never RNG; if FFR wobbles run-to-run, the M4 overlap
  upgrade reordered a draw — fall back to `PUMAP_HOST_INT8_PREFETCH=0`).

**Sealed-path A/B (separate, optional):** build a 2M sealed substrate with
`build_int8_substrate.py`, add a `sealed_int8_path` twin of the 500k int8 dataset in
`image_map_pipeline.py`, and confirm `int8fac-hostint8` (re-encode) vs `...-sealed` (no re-encode)
produce **identical** FFR (the sealed codes are bit-identical to the in-loader encode per
`build_int8_substrate.validate`), with the sealed run skipping the encode wall-time.

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
- **Pinned-slot reuse race (M1/M4):** the CUDA event guard in `_fill_slot` (`ev.synchronize()`) must
  precede overwriting a slot the GPU may still be transferring. Mitigation: 2 slots + event-per-slot
  exactly mirrors round0034 (`:829-832`, `:1101-1107`); the conservative resolve-in-place variant in
  §2a serializes fill↔transfer so no race is possible on the first cut. Only the follow-on overlap
  toggle introduces concurrency, guarded by the same event.
- **Producer thread on unattended runs (M4):** a dead worker could stall. Mitigation:
  `PUMAP_HOST_INT8_PREFETCH=0` disables just the thread (keeps M1/M3/M5/M6); `close_fast_input()`
  should be called at train end (wire into the existing `HostStreamEdgeSampler.close()` teardown at
  `core.py:2438`).
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

## Appendix — one-flag summary

| Flag / knob | Default | Effect when set |
|---|---|---|
| `ParametricUMAP(host_int8_fast_input=True)` | `False` | Enables M1–M6 on the `x_residency="host_int8"` path only. Off ⇒ byte-identical legacy. |
| env `PUMAP_HOST_INT8_PREFETCH=0` | `1` | Disables the producer thread (M4); keeps combined pinned gather (M1/M3), cached labels (M5), fused forward (M6). |
| DATASETS `sealed_int8_path` (+`int8_dim`) | absent | `train()` loads a `SealedInt8Substrate` (no `_norm`, no re-encode) instead of `_norm(load())`. |
```

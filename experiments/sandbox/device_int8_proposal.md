# device_int8 residency — proposal (owner 2026-08-28, apply with C1 in one quiescent window)

Zero-transport int8: keep the full int8 X + fp16 scales RESIDENT on the GPU and
gather+dequant per-batch ON DEVICE — no per-batch H2D copy. host_int8 pays that
H2D copy every batch (`edge_list_dataset.py:251` `rows_i8.to(self.device)`), which
is the measured gate-2 penalty (int8 0.639 h2048 / 0.769 h3072 of resident). If the
data never moves, the int8 path should reach resident-class throughput while keeping
int8's favorable jina quality (+0.0134) and ~30% VRAM saving. Resident-set math says
it FITS at 30M (21.5 GB X + 5.0 GB full edges + 2.5 GB model/opt/act = 29.0 GB < 32).

Same discipline as C1: DEFAULT-OFF (new x_residency value), fail-closed, reviewed,
applied in the SAME quiescent window as C1. When x_residency != "device_int8" every
path is byte-identical.

## Diffs

**1. `core.py:216` — accept the new value.**
```python
-        if str(x_residency).lower() not in ("auto", "host_int8"):
+        if str(x_residency).lower() not in ("auto", "host_int8", "device_int8"):
             raise ValueError(
-                f"x_residency must be 'auto' or 'host_int8', got {x_residency!r}")
+                f"x_residency must be 'auto', 'host_int8' or 'device_int8', got {x_residency!r}")
```

**2. `core.py` — device_int8 branch (mirror the host_int8 branch `:695-731`), with a
fail-closed resident-set pre-check.** Inserted just before the `host_int8` branch:
```python
+        if str(self.x_residency).lower() == "device_int8":
+            if edge_set is not None:
+                raise RuntimeError("x_residency='device_int8' incompatible with reject_neighbors")
+            # Fail-closed VRAM pre-check (FIXED per review 2026-08-28): the resident set
+            # is int8 X + the edge arrays the sampler is about to upload + a real margin
+            # for model/opt/activations/rankneg. NEVER silently fall back to host_int8.
+            # (Earlier draft was fail-OPEN: `headroom = budget or (free-need)` — a nonzero
+            # env var made headroom always >0 so the check was DISABLED; unset, it only
+            # asked "does X alone fit", reserving nothing for what OOMs two lines later.)
+            free_b, total_b = torch.cuda.mem_get_info(self.device)
+            need_x = n_train * n_features * 1 + n_train * 2          # int8 codes + fp16 scales
+            edges_bytes = int(len(sources)) * (4 + 4 + 4)            # src+dst i32 + wt f32 on device
+            # margin covers model+optimizer+activations+rankneg rank-order arrays at champion
+            # shapes; the env var RAISES it (stricter), it can NOT bypass the check.
+            margin = max(3.5, float(os.environ.get("DEVICE_INT8_VRAM_MARGIN_GB", "3.5"))) * (1024**3)
+            need_total = need_x + edges_bytes + margin
+            if need_total > free_b:
+                raise RuntimeError(
+                    "device_int8: resident set exceeds free VRAM — X "
+                    f"{need_x/1e9:.1f} GB + edges {edges_bytes/1e9:.1f} GB + margin "
+                    f"{margin/1e9:.1f} GB = {need_total/1e9:.1f} GB > {free_b/1e9:.1f} GB free. "
+                    "Compact edges (CSR/streamed) or use host_int8. "
+                    "[breakdown printed so a 30M attempt shows which term to compact]")
+            uwr = positive_source_rows is not None
+            device_uniform_replacement = bool(
+                uwr or (not self.weighted_edge_sampling
+                        and n_pos_edges > int(os.environ.get("PER_BATCH_EDGE_THRESHOLD", 400_000_000))))
+            logging.info("Edge-list mode: DEVICE-INT8 residency (zero-transport int8 X on device).")
+            _stamp_pipeline("device_int8", "DeviceEdgeSampler", weighted_ok=True,
+                            x_residency="device_int8",
+                            uniform_with_replacement=device_uniform_replacement)
+            di8 = HostInt8ArrayDataset(X, self.device, resident="device")   # int8 X ON device
+            self._X_dev = di8
+            self._fast_device_path = True
+            loader = DeviceEdgeSampler(
+                di8, sources, targets, weights, n_nodes=n_train,
+                pos_ratio=self.pos_ratio, batch_size=self.batch_size, shuffle=True,
+                random_state=random_state, positive_target_mode=self.positive_target_mode,
+                weighted_edge_sampling=self.weighted_edge_sampling,
+                uniform_with_replacement=uwr, positive_source_rows=positive_source_rows,
+                fixed_edges_per_source=fixed_edges_per_source, device=self.device)
+            return di8, loader, n_pos_edges
```

**3. `edge_list_dataset.py` — `HostInt8ArrayDataset` gains a `resident="host"|"device"`
knob** (default "host" ⇒ byte-identical). When "device", `_i8`/`_scales` live on the
GPU and `index_select` gathers + dequants on-device (no `idx→cpu`, no H2D):
```python
     def __init__(self, X, device, *, encoded=None, scales=None,
-                 ...):
+                 resident="host", fast_input=False, ...):
+        # GUARD (review 2026-08-28): device residency and C1's fast_input are mutually
+        # exclusive — gather_pairs reads the HOST-side _enc_np arrays, whose semantics
+        # diverge when the int8 X lives on the device. Fail closed rather than admit an
+        # undefined combined path from a future arm spec.
+        if resident == "device" and fast_input:
+            raise ValueError("HostInt8ArrayDataset: resident='device' and fast_input are "
+                             "mutually exclusive (C1 gather_pairs reads host-side _enc_np).")
+        self._resident = resident
         ...
         self._i8 = torch.from_numpy(encoded)
         self._scales = torch.from_numpy(scales)
-        if self._pin:
+        if resident == "device":
+            self._i8 = self._i8.to(device)          # full int8 X resident on GPU (one upload)
+            self._scales = self._scales.to(device)
+        elif self._pin:
             self._i8 = self._i8.pin_memory(); self._scales = self._scales.pin_memory()
     def index_select(self, idx):
+        if self._resident == "device":
+            idx_dev = idx.to(device=self.device, dtype=torch.long)   # stays on device
+            rows = self._i8.index_select(0, idx_dev).float()          # device gather
+            sc = self._scales.index_select(0, idx_dev).float().unsqueeze(1)
+            return rows * sc                                          # on-device dequant, no H2D
         idx_cpu = idx.detach().to("cpu", dtype=torch.long)           # host path unchanged
         ...
```
The dequant arithmetic (`int8.float() * scale.float()`) is IDENTICAL to host_int8 — the
ONLY difference is where the gather happens — so quality must match host_int8 bit-for-bit
modulo gather order (asserted below).

## 4-way transport A/B (2M jina champion-bs16k, seed 42) — settles gate-2 with data

One table, four arms, sole delta = transport:
| arm | x_residency | flag | transport |
|---|---|---|---|
| resident | auto | — | full fp16 on device |
| legacy-int8 | host_int8 | — | int8 host → H2D int8 slice per batch |
| fast-int8 | host_int8 | `host_int8_fast_input=True` (C1) | pinned combined-gather |
| **device-int8** | device_int8 | — | int8 resident on device, on-device gather |

Report updates/s + ratio-vs-resident for each. **Quality assert (free):** device-int8
shares the exact quant scheme + sampler draws with legacy-int8, so its seed-42 FFR must
reproduce the hostint8 arm's **0.6964 (v2)** to numerical noise — assert `|Δ| < 0.002`
in the A/B rather than training extra twins; a larger deviation is a real bug (gather
misorder / dequant divergence), fail the arm.

## Per-phase VRAM caveats (carry — 30M is NOT unconditionally "fits")
- The 29.0 GB total is the TRAINING step at **h2048** with a generous 2.5 GB for
  model+opt+activations. **h3072-class** runs likely need **CSR / streamed-edge
  compaction** (drops the 5.0 GB full-edge term toward 3.4 GB / streamed ~2 GB).
  Keep edge compaction on the roadmap; do not declare 30M-fits unconditionally.
- **rankneg** allocates rank-order arrays (`_refresh_rank_negatives`), and the
  **transform / knn** phases have their own peaks (a full 30M transform materializes
  the coords + a KD/gather working set). The feasibility claim must be **per-phase**
  (train step, rankneg refresh, transform), each ≤ the device budget — not just the
  training step. The A/B measures the train step; a 2M transform-phase VRAM probe
  should accompany before a 30M device-int8 commit.
- Composes with batch-scaling (task 1): device-int8 + bs32K may stack (bigger batch
  amortizes the on-device gather launch overhead too) — the batchscan int8 segments
  become device-int8 segments once this lands.

## Apply window + rollback
Apply in the C1 quiescent window (GPU idle, no live `fit`). Two default-off features,
one window. Rollback = the value is rejected at `core.py:216` if reverted; no data
migration. If the 4-way A/B shows device-int8 at resident-class throughput, gate-2
DISSOLVES and C1's transport machinery becomes the 100M-era fallback, not the 30M
critical path.

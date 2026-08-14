"""CPU tests for the opt-in host-int8 X residency path (x_residency="host_int8").

Run with CUDA hidden (the harness sets CUDA_VISIBLE_DEVICES=""). Covers:

  T1  encoder fidelity: round-trip dequant error bounded per row; scales are
      fp16 > 0; encoded is int8 in [-127, 127]. PLUS the strongest check —
      bit-for-bit reproduction of the sealed R0262 100M int8 substrate.
  T2  parity (positive control): with a losslessly-int8-representable X and the
      SAME DeviceEdgeSampler + seed, host_int8 and the fp32 device path produce
      floating-point-identical maps. Tolerance is near-exact BECAUSE dequant is
      a no-op for this X and the only thing that differs is X's storage.
  T3  quantization-actually-happens PLANT: with a continuous (lossy) X, the
      int8 map measurably DIFFERS from fp32 (mean delta > 0) but within a bound,
      AND the model actually used HostInt8ArrayDataset (x_residency stamped
      host_int8). This test FAILS if the code secretly kept fp32.
  T4  fp32 byte-untouched control: x_residency="auto" never builds the int8
      dataset and never stamps the host_int8 pipeline.

CPU-only. Does not launch any GPU job.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/enjalot/code/latent-basemap")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import torch  # noqa: E402
from basemap.pumap.parametric_umap.core import ParametricUMAP  # noqa: E402
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (  # noqa: E402
    HostInt8ArrayDataset, quantize_int8_rows, INT8_DIVISOR,
)

# Sealed R0262 100M substrate + its int8 artifact (for bit-for-bit verification).
R0262_SRC = ("/data/latent-basemap/runs/round-0238/queue/artifacts/"
             "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy")
R0262_I8 = ("/data/latent-basemap/runs/round-0262/artifacts/"
            "minilm-mixed-100m-int8-v1/substrate.i8")
R0262_SC = ("/data/latent-basemap/runs/round-0262/artifacts/"
            "minilm-mixed-100m-int8-v1/substrate-scales.f16")
DIM_100M = 384

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)
    return cond


def tiny_edges(N, rng, k=4):
    srcs, dsts, wts = [], [], []
    for i in range(N):
        for j in rng.choice(N, size=k, replace=False):
            if j == i:
                continue
            srcs.append(i); dsts.append(int(j)); wts.append(1.0)
    p = Path(tempfile.mkdtemp()) / "edges.npz"
    np.savez(p, sources=np.asarray(srcs, np.int64),
             targets=np.asarray(dsts, np.int64),
             weights=np.asarray(wts, np.float32))
    return p


def lossless_int8_X(N, D, rng):
    """X that the R0262 encoder round-trips with ZERO error: each row has an
    entry at ±127 so the recomputed scale is exactly 2**-7 (fp16-exact), and
    every element is an integer multiple of that scale."""
    scale = np.float32(2.0 ** -7)  # 0.0078125, exact in fp16 and fp32
    q = rng.integers(-126, 127, size=(N, D)).astype(np.int32)
    # force one ±127 per row so absmax == 127*scale and scale recomputes exactly
    cols = rng.integers(0, D, size=N)
    signs = rng.choice([-127, 127], size=N)
    q[np.arange(N), cols] = signs
    return (q.astype(np.float32) * scale), scale


BASE = dict(n_components=2, hidden_dim=32, n_layers=2, n_neighbors=4,
            a=1.0, b=1.0, low_dim_kernel="umap", learning_rate=1e-3,
            n_epochs=3, batch_size=64, pos_ratio=0.25,
            architecture="residual_bottleneck", positive_target_mode="binary",
            use_amp=False, use_batchnorm=False, use_dropout=False,
            correlation_weight=0.0, require_full_budget=False, device="cpu")


def fit_map(X, edges, *, x_residency="auto", gpu_resident_data="auto", seed=7,
            **extra):
    torch.manual_seed(seed)
    m = ParametricUMAP(**{**BASE, "x_residency": x_residency,
                          "gpu_resident_data": gpu_resident_data, **extra})
    m.fit(X, precomputed_edges_path=str(edges), random_state=seed)
    xy = m.transform(X, batch_size=64).astype(np.float64)
    return m, xy


# ── T1: encoder fidelity + R0262 bit-for-bit ──────────────────────────────────
def t1_encoder():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 48)).astype(np.float32)
    X[7] = 0.0  # an all-zero row must still encode with scale > 0
    enc, sc = quantize_int8_rows(X)
    check("T1.dtype_int8", enc.dtype == np.int8)
    check("T1.range_127", int(np.abs(enc).max()) <= 127,
          f"max|q|={int(np.abs(enc).max())}")
    check("T1.scales_fp16", sc.dtype == np.float16)
    check("T1.scales_positive_finite",
          bool(np.all(sc > 0) and np.all(np.isfinite(sc))))
    deq = enc.astype(np.float32) * sc.astype(np.float32)[:, None]
    per_row_max_err = np.abs(deq - X).max(axis=1)
    # symmetric rounding bound: |x - round(x/s)*s| <= s/2 (fp32 slack 1e-4)
    bound = sc.astype(np.float32) / 2.0 * (1.0 + 1e-4)
    check("T1.roundtrip_within_scale_over_2",
          bool(np.all(per_row_max_err <= bound)),
          f"max ratio={float((per_row_max_err / (sc.astype(np.float32)/2)).max()):.4f}")
    check("T1.zero_row_dequant_zero", bool(np.all(deq[7] == 0.0)))

    # Bit-for-bit against the sealed R0262 100M substrate.
    if all(os.path.exists(p) for p in (R0262_SRC, R0262_I8, R0262_SC)):
        src = np.load(R0262_SRC, mmap_mode="r")
        ok = True
        detail = []
        for lo, hi in [(0, 8), (50_000_000, 50_000_004),
                       (99_999_996, 100_000_000)]:
            blk = np.array(src[lo:hi], dtype=np.float32)
            e, s = quantize_int8_rows(blk)
            with open(R0262_I8, "rb") as f:
                f.seek(lo * DIM_100M)
                ref_i8 = np.frombuffer(f.read((hi - lo) * DIM_100M),
                                       dtype=np.int8).reshape(hi - lo, DIM_100M)
            with open(R0262_SC, "rb") as f:
                f.seek(lo * 2)
                ref_sc = np.frombuffer(f.read((hi - lo) * 2), dtype="<f2")
            i8_ok = np.array_equal(e, ref_i8)
            sc_ok = np.array_equal(s.view(np.uint16), ref_sc.view(np.uint16))
            ok = ok and i8_ok and sc_ok
            detail.append(f"[{lo}:{hi}] i8={i8_ok} sc={sc_ok}")
        check("T1.R0262_bit_for_bit", ok, "; ".join(detail))
    else:
        print("[SKIP] T1.R0262_bit_for_bit — sealed artifact not on disk")


# ── T2: parity positive control (lossless X ⇒ identical maps) ─────────────────
def t2_parity():
    rng = np.random.default_rng(1)
    N, D = 512, 24
    X, scale = lossless_int8_X(N, D, rng)
    edges = tiny_edges(N, rng)

    # dataset-level exactness: dequant must equal X bit-for-bit here.
    ds = HostInt8ArrayDataset(X, "cpu")
    idx = torch.arange(N)
    deq = ds.index_select(idx).numpy()
    check("T2.dataset_lossless", np.array_equal(deq.astype(np.float32),
                                                X.astype(np.float32)),
          f"max|deq-X|={float(np.abs(deq - X).max()):.2e}")

    # map-level: same DeviceEdgeSampler + seed, only X residency differs.
    _, xy_fp32 = fit_map(X, edges, x_residency="auto", gpu_resident_data=True)
    _, xy_i8 = fit_map(X, edges, x_residency="host_int8")
    delta = np.abs(xy_fp32 - xy_i8).max()
    # near-exact: dequant is a no-op, both share the sampler + seed. Allow a
    # small fp slack for non-associative reductions.
    check("T2.map_parity_near_exact", float(delta) <= 1e-4,
          f"max|Δxy|={float(delta):.2e} (tol 1e-4)")


# ── T3: quantization-actually-happens PLANT ───────────────────────────────────
def t3_plant():
    rng = np.random.default_rng(2)
    N, D = 512, 24
    X = rng.standard_normal((N, D)).astype(np.float32)  # continuous ⇒ lossy
    edges = tiny_edges(N, rng)

    # dataset-level: dequant must DIFFER from X (quant happened) but stay bounded.
    ds = HostInt8ArrayDataset(X, "cpu")
    deq = ds.index_select(torch.arange(N)).numpy()
    mean_feat_delta = float(np.abs(deq - X).mean())
    _, sc = quantize_int8_rows(X)
    max_bound = float((sc.astype(np.float32) / 2 * (1 + 1e-4)).max())
    max_feat_delta = float(np.abs(deq - X).max())
    check("T3.features_differ", mean_feat_delta > 0.0,
          f"mean|deq-X|={mean_feat_delta:.2e}")
    check("T3.features_bounded", max_feat_delta <= max_bound,
          f"max|deq-X|={max_feat_delta:.2e} <= max(scale/2)={max_bound:.2e}")

    # map-level: int8 map measurably differs from fp32, but within a cap.
    m_fp32, xy_fp32 = fit_map(X, edges, x_residency="auto", gpu_resident_data=True)
    m_i8, xy_i8 = fit_map(X, edges, x_residency="host_int8")
    mean_delta = float(np.abs(xy_fp32 - xy_i8).mean())
    coord_scale = float(np.abs(xy_fp32).mean()) + 1e-9
    check("T3.map_differs", mean_delta > 1e-4,
          f"mean|Δxy|={mean_delta:.3e} (must be > 1e-4; ==0 ⇒ secretly fp32)")
    check("T3.map_bounded", mean_delta < 2.0 * coord_scale,
          f"mean|Δxy|={mean_delta:.3e} < {2.0 * coord_scale:.3e}")

    # residency proof: the int8 model actually used the int8 dataset/pipeline.
    check("T3.used_int8_dataset",
          isinstance(m_i8._X_dev, HostInt8ArrayDataset)
          and m_i8._X_dev.storage_dtype == torch.int8)
    check("T3.pipeline_stamped_host_int8",
          m_i8._pipeline_info.get("x_residency") == "host_int8"
          and m_i8._pipeline_info.get("pipeline") == "host_int8")


# ── T4: fp32 path byte-untouched when off ─────────────────────────────────────
def t4_fp32_untouched():
    rng = np.random.default_rng(3)
    N, D = 256, 24
    X = rng.standard_normal((N, D)).astype(np.float32)
    edges = tiny_edges(N, rng)
    m_auto, _ = fit_map(X, edges, x_residency="auto", gpu_resident_data=True)
    check("T4.auto_not_int8_dataset",
          not isinstance(m_auto._X_dev, HostInt8ArrayDataset))
    check("T4.auto_pipeline_not_host_int8",
          m_auto._pipeline_info.get("x_residency") != "host_int8")


if __name__ == "__main__":
    t1_encoder()
    t2_parity()
    t3_plant()
    t4_fp32_untouched()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({', '.join(FAILS)})")
    sys.exit(0 if not FAILS else 1)

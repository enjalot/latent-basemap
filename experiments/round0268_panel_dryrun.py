#!/usr/bin/env python3
"""R0268 pre-flight: END-TO-END 100M panel dry-run on a NON-EVIDENCE throwaway map.

Plan §3 (plan-100m-flagship-2026-08-17.md): exercise `score_one_map`'s 100M
collapse/fog/FFR path (100M-unexercised) + the reserve-projection FFR + the
prefix-purity build, on a THROWAWAY map (a real promoted-recipe ParametricUMAP
whose output we DISCARD — it is non-evidence for R0268), BEFORE any real seed.
Measures wall + peak-RSS + peak-VRAM per stage; STOP on any scale error. This
would have caught trips 5,6,8,9,10 of the 50M saga for pennies.

The throwaway model is R0267's SEALED 50M seed42 promoted-recipe map — its encoder
is 384->2 regardless of N, so transforming the 100M substrate through it is a valid
non-evidence stress of the exact panel code path. We measure resources, not values.

Reuses the EXACT node helpers (no reimplementation): `_transform_100m_in_chunks`,
`_build_prefix_purity_centroids`, `score_one_map`, `score_panel`, `panel_config`.
"""
import os, sys, time, json, gc, resource, threading

os.environ.setdefault("HF_HOME", "/data/hf")
import numpy as np

THROWAWAY_MODEL = (
    "/data/latent-basemap/runs/round-0267/queue-correction-3/artifacts/"
    "minilm-mixed-50000k-fneg-x2-md000-hostint8-seed42-r0267-v1/model.pt"
)
SUBSTRATE_NPY = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/substrate.f32.npy"
)
RESERVE_NPY = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/reserve.f32.npy"
)
QROWS_NPY = (
    "/data/latent-basemap/runs/round-0238/queue/artifacts/"
    "minilm-mixed-100000k-nested-substrate-and-reserves-v1/reserve-query-rows.i64.npy"
)
TRUTH_NPY = "/data/latent-basemap/runs/round-0268/ffr/reserve-truth-100m/truth-top10.npy"
CACHE_DIR = "/data/latent-basemap/runs/round-0268/dryrun/prefix-centroids"
OUT_JSON = "/data/latent-basemap/runs/round-0268/dryrun/panel-dryrun-report.json"


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


class RssSampler(threading.Thread):
    """Poll /proc/self/statm RSS every 2s so we see the steady-state page-cache
    footprint, not just ru_maxrss's monotone high-water."""
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.peak_rss_gib = 0.0
        self.pagesize = os.sysconf("SC_PAGE_SIZE")

    def run(self):
        while not self.stop:
            try:
                with open("/proc/self/statm") as f:
                    rss_pages = int(f.read().split()[1])
                g = rss_pages * self.pagesize / (1024 ** 3)
                if g > self.peak_rss_gib:
                    self.peak_rss_gib = g
            except Exception:
                pass
            time.sleep(2.0)


def main():
    t_start = time.monotonic()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    for p in (THROWAWAY_MODEL, SUBSTRATE_NPY, RESERVE_NPY, QROWS_NPY, TRUTH_NPY):
        if not os.path.exists(p):
            print(f"MISSING input: {p}", flush=True)
            sys.exit(2)

    import torch
    from experiments.round0268_nodes import (
        ROWS, DIMENSION, PREFIX_ROWS, FULL_TRANSFORM_BATCH,
        _transform_100m_in_chunks, _build_prefix_purity_centroids,
        score_one_map,
    )
    from basemap.panel_v2 import score_panel, reset_process_cuda_peak
    from basemap.pumap.parametric_umap import ParametricUMAP
    from experiments.round0268_nodes import prompt_contract

    sampler = RssSampler(); sampler.start()
    stages = {}

    def poll(msg):
        # mirror the node's poll callback; print sparse progress
        if not hasattr(poll, "_n"):
            poll._n = 0
        poll._n += 1
        if poll._n % 20 == 0:
            print(f"  {msg}  rss~{sampler.peak_rss_gib:.1f}GiB", flush=True)

    print(f"[dryrun] ROWS={ROWS:,} DIM={DIMENSION} PREFIX_ROWS={PREFIX_ROWS:,} "
          f"batch={FULL_TRANSFORM_BATCH} disc={int(ROWS*0.001):,}", flush=True)

    reset_process_cuda_peak()
    torch.cuda.reset_peak_memory_stats()

    # --- open the 153.6 GB substrate lazily (never materialize) ---
    source = np.load(SUBSTRATE_NPY, mmap_mode="r", allow_pickle=False)
    assert source.shape == (ROWS, DIMENSION) and source.dtype == np.float32, source.shape
    reserve_all = np.load(RESERVE_NPY, mmap_mode="r", allow_pickle=False)
    qrows = np.load(QROWS_NPY, allow_pickle=False).astype(np.int64, copy=False)
    reserve_embeddings = np.asarray(reserve_all[qrows], dtype=np.float32)
    reserve_truth = np.load(TRUTH_NPY, allow_pickle=False).astype(np.int64, copy=False)
    reserve_disc = int(ROWS * 0.001)
    print(f"[dryrun] reserve queries={reserve_embeddings.shape[0]} truth={reserve_truth.shape}",
          flush=True)

    cfg = prompt_contract.panel_config()
    centroid_ks = [256, 1024]

    # --- STAGE 1: prefix purity centroids (GPU k-means on the first 2M rows) ---
    t = time.monotonic()
    centroids, _sig = _build_prefix_purity_centroids(
        source[:PREFIX_ROWS], centroid_ks, cache_dir=CACHE_DIR)
    stages["prefix_centroids_s"] = round(time.monotonic() - t, 1)
    print(f"[dryrun] STAGE1 prefix centroids: {stages['prefix_centroids_s']}s "
          f"rss~{sampler.peak_rss_gib:.1f}GiB", flush=True)

    # --- STAGE 2: transform 100M substrate through the throwaway model ---
    proj_model = ParametricUMAP.load(THROWAWAY_MODEL, device="cuda")
    t = time.monotonic()
    coordinates = _transform_100m_in_chunks(proj_model, source, poll)
    stages["transform_100m_s"] = round(time.monotonic() - t, 1)
    assert coordinates.shape == (ROWS, 2) and np.isfinite(coordinates).all(), coordinates.shape
    print(f"[dryrun] STAGE2 transform 100M: {stages['transform_100m_s']}s "
          f"coords={coordinates.shape} rss~{sampler.peak_rss_gib:.1f}GiB", flush=True)

    # --- STAGE 3: descriptive purity score_panel on the 2M prefix ---
    t = time.monotonic()
    purity_panel = score_panel(
        source[:PREFIX_ROWS], coordinates[:PREFIX_ROWS], config=cfg,
        centroids_by_k=centroids, hiD_reference=None,
        provenance={"round_id": "0268", "seed": 42, "pass": "dryrun-descriptive",
                    "descriptive": True, "gated": False},
    )
    purity_ratios = {"k256": float(purity_panel["purity"]["k256"]),
                     "k1024": float(purity_panel["purity"]["k1024"])}
    stages["purity_prefix_s"] = round(time.monotonic() - t, 1)
    print(f"[dryrun] STAGE3 purity prefix: {stages['purity_prefix_s']}s "
          f"k256={purity_ratios['k256']:.4f} k1024={purity_ratios['k1024']:.4f}", flush=True)

    # --- STAGE 4: reserve-projection + score_one_map on the FULL 100M coords ---
    placed = np.asarray(proj_model.transform(reserve_embeddings, batch_size=FULL_TRANSFORM_BATCH),
                        dtype=np.float32)
    t = time.monotonic()
    scored = score_one_map(coordinates=coordinates, probes_placed=placed,
                           truth_top10=reserve_truth, purity_ratios=purity_ratios,
                           disc=reserve_disc)
    stages["score_one_map_s"] = round(time.monotonic() - t, 1)
    print(f"[dryrun] STAGE4 score_one_map: {stages['score_one_map_s']}s "
          f"collapse={scored['collapse']:.4f} fog={scored['fog']:.4f} "
          f"heldout_ffr={scored['heldout_ffr']:.4f} rss~{sampler.peak_rss_gib:.1f}GiB", flush=True)

    del proj_model, coordinates, placed
    torch.cuda.empty_cache(); gc.collect()

    sampler.stop = True
    time.sleep(0.1)
    vram_alloc_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    vram_resv_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
    report = {
        "role": "R0268 100M panel END-TO-END dry-run (non-evidence throwaway map)",
        "throwaway_model": THROWAWAY_MODEL,
        "rows": ROWS, "prefix_rows": PREFIX_ROWS, "reserve_disc": reserve_disc,
        "reserve_queries": int(reserve_embeddings.shape[0]),
        "stages_s": stages,
        "total_wall_s": round(time.monotonic() - t_start, 1),
        "peak_rss_gib_rusage": round(rss_gib(), 2),
        "peak_rss_gib_statm_sampled": round(sampler.peak_rss_gib, 2),
        "peak_vram_gib_allocated": round(vram_alloc_gib, 2),
        "peak_vram_gib_reserved": round(vram_resv_gib, 2),
        "throwaway_metrics_DISCARD": {
            "collapse": scored["collapse"], "fog": scored["fog"],
            "heldout_ffr": scored["heldout_ffr"],
            "purity_k256": purity_ratios["k256"], "purity_k1024": purity_ratios["k1024"],
        },
        "note": ("metrics are from a 50M-trained model projecting a 100M substrate — "
                 "NON-EVIDENCE; this run measures RESOURCES + scale-path correctness only."),
    }
    json.dump(report, open(OUT_JSON, "w"), indent=2)
    print(f"\n[dryrun] DONE total={report['total_wall_s']}s", flush=True)
    print(f"[dryrun] PEAK RSS rusage={report['peak_rss_gib_rusage']}GiB "
          f"statm-sampled={report['peak_rss_gib_statm_sampled']}GiB", flush=True)
    print(f"[dryrun] PEAK VRAM alloc={report['peak_vram_gib_allocated']}GiB "
          f"reserved={report['peak_vram_gib_reserved']}GiB", flush=True)
    print(f"[dryrun] report -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

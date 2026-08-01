"""map_metrics_extract.py — Component C of the basemap map-viz build.

Reads the frozen metric artifacts (core-panel-arrays.npz for the R0108 atlas
family, per-probe OOD coordinate npz, and the R0102 150M high-D reference.npz)
and emits the two binary/JSON packets the viewer consumes:

    metrics-anchors.bin   — u32 magic 0x414E4331, u32 count, f32 (x, y, score)*
    metrics-queries.json   — {"probes": [{"key","label","recall50","queries":[...]}]}

Everything here is post-hoc tooling; nothing on the launch path, no GPU.

Verified metric semantics (binding — see the design doc appendix):

R0108-family (core-panel-arrays.npz, 5632 anchors = 22 groups x 256):
  * neighbor ids (high_neighbors_top15, low_neighbors_top50) are COMPACT ids.
  * anchor_coordinates are already in map space.
  * recall@10          = mean over anchors of |hi_top10 & lo_top10| / 10
  * recall@50-of-high10 = mean over anchors of |hi_top10 & lo_top50| / 10
    (both reproduce core-geometry.json exactly: 0.002450284..., 0.011044034...)
  * PRIMARY anchor score = local expansion:
        ratio_i = low_radius_i / high_radius_i
        score_i = log2(ratio_i / median(ratio)), clipped to [-2, 2]
        score01 = (clipped + 2) / 4               (mapped to [0,1] for the bin)
    labelled "local expansion (log2 vs median)".
  * TRUE per-anchor FFR is NOT computable from this npz (needs the full
    fraction_k low-D pool); the global FFR (0.6386...) is carried as a summary
    stat from core-geometry.json, never as the per-anchor score.

R0108-family OOD npz (dadabase / fineweb-heldout / pol_Latn / trec-covid):
  * exact_high_d_top10 and low_d_top50 are POSITIONS into probe_corpus_coords.
  * per-query packet:
        truth      = exact_high_d_top10[q]           (10 corpus positions)
        neighbors  = probe_corpus_coords[truth]       (10 xy in map space)
        hits       = isin(truth, low_d_top50[q])      (10 booleans)
        recall     = mean(hits)
    mean recall over queries reproduces 0.2278 for pol_Latn exactly.
  * Query packets are emitted ONLY when the npz embeds exact_high_d_top10 +
    low_d_top50; otherwise the probe is skipped (recorded in the manifest).

R0102 150M (high-d-reference-150m/reference.npz):
  * CORRECTION vs the design appendix: hi_frac (10000,147222) is the APPROXIMATE
    HIGH-D top-k_frac *membership* pool used for centroid purity — NOT the low-D
    FFR pool.  isin(hi_hit, hi_frac) is therefore ~1.0 for every anchor and is
    NOT the FFR (verified: full-10k mean = 1.0000).  The real panel FFR is
    ffr_from_neighbors(hi_hit, lo_kf, k_hit) where lo_kf is the EXACT LOW-D
    top-k_frac neighbours from the *map* coordinates (score-time pass); the
    published full-150M value is 0.5075 (panel.json transductive = 0.5011).  A
    sampled low-D recompute (compute_r0102_true_ffr, 40 anchors) reproduces
    0.5125 — matching ~0.50, confirming the correction.  hi_frac is still handled
    safely: it is ~11.8 GB, memmapped in place (stored uncompressed) and never
    np.load'd whole; a compressed future copy would be extracted to
    /data/latent-basemap/tmp/ and memmapped there.
  * SHIPPED anchor score = local expansion (log2 low/high radius vs median) from
    the precomputed density-v2 radii (full_150m) — cheap, per-anchor, and the
    same primary signal as the R0108 core panel.  The global FFR is a summary
    stat only.
  * Anchor display coords = coordinates[anchor_substrate_rows] — the GLOBAL
    (identity-order) 150M coordinate rows.  hi_hit ids are COMPACT; mapping
    compact->global (from the eligibility selector's excluded_rows) is verified:
    global_rows[anchor_ids] == anchor_substrate_rows exactly.
"""

from __future__ import annotations

import io
import json
import os
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.lib.format as npformat

# ---------------------------------------------------------------------------
# Binary format constants (frozen data contract).
# ---------------------------------------------------------------------------

ANCHORS_MAGIC = 0x414E4331  # "ANC1"

# Published reference values (core-geometry.json for R0108-a3 core panel).
R0108_RECALL_AT_10 = 0.002450284090909091
R0108_RECALL_AT_50_OF_HIGH10 = 0.011044034090909092
R0108_GLOBAL_FFR = 0.6386363636363636
R0102_PUBLISHED_FULL_FFR = 0.5075

# A texts resolver maps (probe_key, id_array) -> list[str] or None.  It is an
# optional standalone hook; callers may leave it None and no text is emitted.
TextsResolver = Callable[[str, np.ndarray], Optional[Sequence[str]]]


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------

def write_anchors_bin(path: os.PathLike | str, xy: np.ndarray, score01: np.ndarray) -> int:
    """Write metrics-anchors.bin: u32 magic, u32 count, f32 (x, y, score)*.

    xy is (N, 2), score01 is (N,) in [0, 1].  Returns the anchor count.
    """
    xy = np.ascontiguousarray(xy, dtype="<f4")
    score01 = np.ascontiguousarray(score01, dtype="<f4")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N,2), got {xy.shape}")
    if score01.shape[0] != xy.shape[0]:
        raise ValueError("xy and score01 length mismatch")
    n = xy.shape[0]
    triples = np.empty((n, 3), dtype="<f4")
    triples[:, 0] = xy[:, 0]
    triples[:, 1] = xy[:, 1]
    triples[:, 2] = score01
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", ANCHORS_MAGIC, n))
        f.write(triples.tobytes())
    return n


def read_anchors_bin(path: os.PathLike | str) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of write_anchors_bin — returns (xy (N,2), score01 (N,)). For tests."""
    with open(path, "rb") as f:
        magic, n = struct.unpack("<II", f.read(8))
        if magic != ANCHORS_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08X}")
        data = np.frombuffer(f.read(n * 3 * 4), dtype="<f4").reshape(n, 3)
    return data[:, :2].copy(), data[:, 2].copy()


def write_queries_json(path: os.PathLike | str, probes: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"probes": probes}, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# R0108-family core-panel anchor score (local expansion).
# ---------------------------------------------------------------------------

def _row_isin_fraction(a_rows: np.ndarray, b_rows: np.ndarray, k: int) -> float:
    """Mean over rows of |a_row & b_row| / k (a_rows already sliced to top-k)."""
    n = a_rows.shape[0]
    acc = 0.0
    for i in range(n):
        acc += np.isin(a_rows[i], b_rows[i]).sum()
    return acc / (n * k)


def recompute_core_panel_recalls(npz) -> tuple[float, float]:
    """(recall@10, recall@50-of-high10) from the core-panel arrays.

    recall@10          : hi_top10 vs lo_top10
    recall@50-of-high10: hi_top10 vs lo_top50
    """
    hi15 = np.asarray(npz["high_neighbors_top15"])
    lo50 = np.asarray(npz["low_neighbors_top50"])
    hi10 = hi15[:, :10]
    r10 = _row_isin_fraction(hi10, lo50[:, :10], 10)
    r50 = _row_isin_fraction(hi10, lo50[:, :50], 10)
    return r10, r50


def local_expansion_score(low_radius: np.ndarray, high_radius: np.ndarray):
    """Primary R0108 anchor score.

    Returns (score01, log2_norm, median_ratio) where
      ratio      = low_radius / high_radius
      log2_norm  = log2(ratio / median(ratio))   (unclipped)
      score01    = (clip(log2_norm, -2, 2) + 2) / 4
    """
    low = np.asarray(low_radius, dtype=np.float64)
    high = np.asarray(high_radius, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = low / high
    # Guard degenerate anchors (high_radius == 0 or nan): treat as median.
    median_ratio = float(np.median(ratio[np.isfinite(ratio) & (ratio > 0)]))
    safe = np.where(np.isfinite(ratio) & (ratio > 0), ratio, median_ratio)
    log2_norm = np.log2(safe / median_ratio)
    clipped = np.clip(log2_norm, -2.0, 2.0)
    score01 = (clipped + 2.0) / 4.0
    return score01.astype(np.float64), log2_norm.astype(np.float64), median_ratio


def _find_sibling_core_geometry(npz_path: Path) -> Optional[dict]:
    cand = npz_path.parent / "core-geometry.json"
    if cand.exists():
        try:
            return json.loads(cand.read_text())
        except Exception:
            return None
    return None


def extract_core_panel_anchors(
    npz_path: os.PathLike | str,
    *,
    assert_published: bool = True,
    tol: float = 1e-9,
) -> dict:
    """Extract R0108-family anchor packet from core-panel-arrays.npz.

    Returns a dict with:
      xy (N,2) f32, score01 (N,) f64, group_ids (N,) uint8,
      recall_at_10, recall_at_50_of_high10 (recomputed),
      score_label, summary (stats + carried global FFR).

    If assert_published, verifies the recomputed recalls against
    core-geometry.json (if a sibling exists) else the frozen constants, to `tol`.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as z:
        xy = np.asarray(z["anchor_coordinates"], dtype="<f4")
        group_ids = np.asarray(z["group_ids"], dtype=np.uint8)
        low_radius = np.asarray(z["low_radius"])
        high_radius = np.asarray(z["high_radius"])
        r10, r50 = recompute_core_panel_recalls(z)

    geom = _find_sibling_core_geometry(npz_path)
    exp10, exp50, global_ffr = (
        R0108_RECALL_AT_10,
        R0108_RECALL_AT_50_OF_HIGH10,
        R0108_GLOBAL_FFR,
    )
    if geom is not None:
        try:
            g = geom["metrics"]["global"]
            exp10 = float(g["recall_at_10"])
            exp50 = float(g["recall_at_50_of_high10"])
            global_ffr = float(g["ffr"])
        except Exception:
            pass

    if assert_published:
        assert abs(r10 - exp10) <= tol, (
            f"recall@10 mismatch: recomputed {r10!r} vs published {exp10!r}"
        )
        assert abs(r50 - exp50) <= tol, (
            f"recall@50-of-high10 mismatch: recomputed {r50!r} vs published {exp50!r}"
        )

    score01, log2_norm, median_ratio = local_expansion_score(low_radius, high_radius)

    summary = {
        "score_label": "local expansion (log2 vs median)",
        "median_radius_ratio": median_ratio,
        "log2_ratio_min": float(np.min(log2_norm)),
        "log2_ratio_p10": float(np.percentile(log2_norm, 10)),
        "log2_ratio_median": float(np.median(log2_norm)),
        "log2_ratio_p90": float(np.percentile(log2_norm, 90)),
        "log2_ratio_max": float(np.max(log2_norm)),
        "expanded_frac": float(np.mean(log2_norm > 0)),  # low > high radius
        "ffr": global_ffr,  # carried from core-geometry.json (not per-anchor)
        "recall_at_10": r10,
        "recall_at_50_of_high10": r50,
    }
    return {
        "xy": xy,
        "score01": score01,
        "group_ids": group_ids,
        "recall_at_10": r10,
        "recall_at_50_of_high10": r50,
        "score_label": "local expansion (log2 vs median)",
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# OOD query packets (R0108-family probes).
# ---------------------------------------------------------------------------

_OOD_REQUIRED_KEYS = ("exact_high_d_top10", "low_d_top50", "probe_corpus_coords",
                      "probe_query_coords")


def _probe_has_embedded_truth(files: Sequence[str]) -> bool:
    return all(k in files for k in _OOD_REQUIRED_KEYS)


def _in_extent(xy: np.ndarray, extent: Sequence[float], eps: float = 1e-3) -> bool:
    xmin, ymin, xmax, ymax = extent
    return bool(
        np.all(xy[:, 0] >= xmin - eps)
        and np.all(xy[:, 0] <= xmax + eps)
        and np.all(xy[:, 1] >= ymin - eps)
        and np.all(xy[:, 1] <= ymax + eps)
    )


@dataclass
class ProbeResult:
    key: str
    label: str
    n_queries: int
    recall50: float
    packet: dict
    skipped: bool = False
    reason: str = ""


def build_probe_packet(
    npz_path: os.PathLike | str,
    *,
    key: Optional[str] = None,
    label: Optional[str] = None,
    texts_resolver: Optional[TextsResolver] = None,
    max_text_chars: int = 200,
) -> ProbeResult:
    """Build one probe's query packet from an OOD coordinate npz.

    Only emits a packet when the npz embeds exact_high_d_top10 + low_d_top50;
    otherwise returns a skipped ProbeResult (packet is None).
    """
    npz_path = Path(npz_path)
    if key is None:
        key = npz_path.stem.replace("-coordinates", "")
    if label is None:
        label = key

    with np.load(npz_path, allow_pickle=True) as z:
        files = list(z.files)
        if not _probe_has_embedded_truth(files):
            return ProbeResult(
                key=key, label=label, n_queries=0, recall50=float("nan"),
                packet=None, skipped=True,
                reason="npz lacks embedded exact_high_d_top10 + low_d_top50",
            )
        truth = np.asarray(z["exact_high_d_top10"])          # (Q, 10) positions
        low50 = np.asarray(z["low_d_top50"])                 # (Q, 50) positions
        corpus = np.asarray(z["probe_corpus_coords"], dtype=np.float64)  # (C, 2)
        qcoords = np.asarray(z["probe_query_coords"], dtype=np.float64)  # (Q, 2)
        query_ids = np.asarray(z["probe_query_ids"]) if "probe_query_ids" in files else None

    q = truth.shape[0]

    # Optional query-text resolution (standalone hook; texts absent is fine).
    texts = None
    if texts_resolver is not None and query_ids is not None:
        try:
            resolved = texts_resolver(key, query_ids)
            if resolved is not None and len(resolved) == q:
                texts = [
                    (str(t)[:max_text_chars] if t is not None else None)
                    for t in resolved
                ]
        except Exception:
            texts = None

    queries = []
    recalls = np.empty(q, dtype=np.float64)
    for i in range(q):
        t = truth[i]
        hits = np.isin(t, low50[i])
        rec = float(hits.mean())
        recalls[i] = rec
        neigh = corpus[t]  # (10, 2) in map space
        entry = {
            "xy": [float(qcoords[i, 0]), float(qcoords[i, 1])],
            "neighbors": [[float(x), float(y)] for x, y in neigh],
            "hits": [bool(h) for h in hits],
            "recall": rec,
        }
        if texts is not None and texts[i] is not None:
            entry["text"] = texts[i]
        queries.append(entry)

    recall50 = float(recalls.mean())
    packet = {"key": key, "label": label, "recall50": recall50, "queries": queries}
    return ProbeResult(
        key=key, label=label, n_queries=q, recall50=recall50, packet=packet,
    )


def build_ood_query_packets(
    npz_paths: Sequence[os.PathLike | str],
    *,
    labels: Optional[dict] = None,
    texts_resolver: Optional[TextsResolver] = None,
    extent: Optional[Sequence[float]] = None,
) -> tuple[list[dict], list[dict]]:
    """Build query packets for a set of OOD probe npz files.

    Returns (probe_packets, probe_manifest_entries).  probe_packets go into
    metrics-queries.json; probe_manifest_entries are the compact manifest
    fragment (key/label/queries/recall50, plus a `skipped` note for probes
    without embedded truth).

    If `extent` is given, asserts every emitted probe's neighbor coords fall
    within it (the frozen "packet neighbor coords within the map extent" gate).
    """
    labels = labels or {}
    packets: list[dict] = []
    manifest: list[dict] = []
    for p in npz_paths:
        p = Path(p)
        key = p.stem.replace("-coordinates", "")
        res = build_probe_packet(
            p, key=key, label=labels.get(key, key), texts_resolver=texts_resolver
        )
        if res.skipped:
            manifest.append({"key": res.key, "label": res.label, "skipped": True,
                             "reason": res.reason})
            continue
        if extent is not None:
            alln = np.array(
                [n for qentry in res.packet["queries"] for n in qentry["neighbors"]],
                dtype=np.float64,
            )
            if alln.size:
                assert _in_extent(alln, extent), (
                    f"probe {res.key}: neighbor coords fall outside map extent {extent}"
                )
        packets.append(res.packet)
        manifest.append({"key": res.key, "label": res.label,
                         "queries": res.n_queries, "recall50": res.recall50})
    return packets, manifest


# ---------------------------------------------------------------------------
# R0102 150M per-anchor FFR (large-member handling).
# ---------------------------------------------------------------------------

def _memmap_npy_member_in_zip(zip_path: Path, member: str):
    """Memmap a STORED (uncompressed) .npy member directly from inside a zip.

    Returns a read-only np.memmap, or None when the member is deflated (caller
    must extract instead).  Never reads the array body into memory.
    """
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        header_offset = info.header_offset
    with open(zip_path, "rb") as f:
        f.seek(header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError("bad local file header")
        name_len = struct.unpack("<H", local[26:28])[0]
        extra_len = struct.unpack("<H", local[28:30])[0]
        data_start = header_offset + 30 + name_len + extra_len
        f.seek(data_start)
        version = npformat.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = npformat.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = npformat.read_array_header_2_0(f)
        else:
            raise ValueError(f"unsupported .npy version {version}")
        body_offset = f.tell()
    return np.memmap(
        zip_path, dtype=dtype, mode="r", offset=body_offset, shape=shape,
        order="F" if fortran else "C",
    )


def _extract_and_memmap_member(zip_path: Path, member: str, tmp_dir: Path):
    """Extract a .npy member to tmp_dir and open it with mmap_mode='r'.

    Idempotent: reuses an existing extraction of the right byte size.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{zip_path.stem}__{member}"
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        if not (out.exists() and out.stat().st_size == info.file_size):
            with zf.open(member) as src, open(out, "wb") as dst:
                # stream copy — never materialize the whole member in RAM
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
    return np.load(out, mmap_mode="r")


def open_reference_member(
    zip_path: os.PathLike | str,
    member: str,
    tmp_dir: os.PathLike | str = "/data/latent-basemap/tmp",
):
    """Open a (possibly huge) .npy member of reference.npz as a read-only memmap.

    Prefers a zero-copy in-place memmap when the member is stored uncompressed;
    otherwise extracts it to tmp_dir and memmaps from there.  Never np.load's
    the member whole.
    """
    zip_path = Path(zip_path)
    mm = _memmap_npy_member_in_zip(zip_path, member)
    if mm is not None:
        return mm
    return _extract_and_memmap_member(zip_path, member, Path(tmp_dir))


def _load_small_member(zip_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        return np.load(io.BytesIO(zf.read(member)), allow_pickle=True)


def compute_r0102_hi_frac_membership(
    reference_npz: os.PathLike | str,
    *,
    tmp_dir: os.PathLike | str = "/data/latent-basemap/tmp",
    progress_every: int = 0,
) -> np.ndarray:
    """membership_i = mean(isin(hi_hit[i], hi_frac[i])) over the 10 hi_hit neighbors.

    IMPORTANT — this is NOT the FFR.  In reference.npz, ``hi_frac`` is the
    APPROXIMATE HIGH-D top-k_frac membership pool (used for centroid *purity*,
    per basemap/panel_v2.py line ~1736).  The true high-D top-10 (``hi_hit``) is
    trivially a subset of the high-D top-147222, so this membership is ~1.0 for
    every anchor and carries no map-quality signal.

    The real panel FFR is ``ffr_from_neighbors(hi_hit, lo_kf, k_hit)`` where
    ``lo_kf`` is the exact LOW-D top-k_frac neighbours computed from the *map*
    coordinates (score-time pass) — see compute_r0102_true_ffr.  This helper
    exists only to make the distinction explicit / auditable.
    """
    reference_npz = Path(reference_npz)
    hi_hit = np.asarray(_load_small_member(reference_npz, "hi_hit.npy"))
    frac = open_reference_member(reference_npz, "hi_frac.npy", tmp_dir=tmp_dir)
    n, k = hi_hit.shape
    if frac.shape[0] != n:
        raise ValueError(f"hi_frac rows {frac.shape[0]} != hi_hit rows {n}")
    membership = np.empty(n, dtype=np.float64)
    for i in range(n):
        membership[i] = np.isin(hi_hit[i], frac[i]).sum() / k
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  membership {i + 1}/{n}  running mean={membership[:i + 1].mean():.4f}",
                  flush=True)
    return membership


def _load_compact_to_global(eligibility_npz: Path, n_rows: int = 150_000_000,
                            cache_dir: Path = Path("/data/latent-basemap/tmp")) -> np.ndarray:
    """compact retained-row position -> global identity row (150M map order).

    Built from the eligibility selector's ``excluded_rows`` (the non-retained
    rows): global_rows = arange(n_rows) with excluded removed.  Cached to disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"compact_to_global_{n_rows}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    el = np.load(eligibility_npz)
    excluded = np.asarray(el["excluded_rows"], dtype=np.int64)
    mask = np.ones(n_rows, dtype=bool)
    mask[excluded] = False
    global_rows = np.flatnonzero(mask).astype(np.int64)
    np.save(cache, global_rows)
    return global_rows


def compute_r0102_true_ffr(
    reference_npz: os.PathLike | str,
    coords_dir: os.PathLike | str,
    eligibility_npz: os.PathLike | str,
    *,
    sample: Optional[int] = None,
    seed: int = 0,
    block: int = 2_000_000,
    tmp_dir: os.PathLike | str = "/data/latent-basemap/tmp",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """TRUE per-anchor FFR for the R0102 150M map (report path — expensive).

    ffr_i = |hi_hit[i] ∩ lo_kf[i]| / k_hit, where lo_kf[i] is the exact LOW-D
    top-k_frac neighbours of anchor i over ALL 150M map coordinates.  Rather
    than materialize lo_kf, we count, for each true neighbour j, how many map
    points are strictly closer to the anchor than j is; j is a hit iff that
    count < k_frac.

    ``sample`` limits to a random subset of anchors (full 10k ≈ hours). Returns
    (ffr_per_sampled_anchor, sampled_anchor_indices).  hi_hit ids are compact,
    so both anchor and neighbour rows are mapped compact->global to index the
    identity-order coordinate chunks.
    """
    reference_npz = Path(reference_npz)
    coords_dir = Path(coords_dir)
    hi_hit = np.asarray(_load_small_member(reference_npz, "hi_hit.npy"))
    kf = int(np.asarray(_load_small_member(reference_npz, "kf.npy")))
    k_hit = hi_hit.shape[1]
    sib = reference_npz.parent / "anchor-substrate-rows.npy"
    substrate = np.load(sib)  # global rows for anchors
    g2 = _load_compact_to_global(Path(eligibility_npz), cache_dir=Path(tmp_dir))

    n_anchors = hi_hit.shape[0]
    if sample is not None and sample < n_anchors:
        idx = np.random.default_rng(seed).choice(n_anchors, sample, replace=False)
    else:
        idx = np.arange(n_anchors)

    mmaps, offsets = _concat_coordinate_chunks(coords_dir)

    def _coords(global_rows):
        return anchor_coords_from_chunks(coords_dir, np.asarray(global_rows)).astype(np.float32)

    anc_xy = _coords(substrate[idx])                                   # (S, 2)
    neigh_global = np.asarray(g2)[hi_hit[idx]]                          # (S, k_hit)
    neigh_xy = _coords(neigh_global.ravel()).reshape(len(idx), k_hit, 2)
    thr = ((neigh_xy - anc_xy[:, None, :]) ** 2).sum(-1)               # (S, k_hit)

    ncloser = np.zeros((len(idx), k_hit), dtype=np.int64)
    for m in mmaps:
        B = m.shape[0]
        for s in range(0, B, block):
            blk = np.asarray(m[s:s + block], dtype=np.float32)
            d2 = ((blk[:, None, :] - anc_xy[None, :, :]) ** 2).sum(-1)  # (b, S)
            for a in range(len(idx)):
                ncloser[a] += (d2[:, a][:, None] < thr[a][None, :]).sum(0)
        if verbose:
            print(f"  scanned chunk of {B} rows", flush=True)
    # each anchor's own row sits at distance 0 (< any positive threshold): drop 1
    ncloser -= (thr > 0).astype(np.int64)
    ffr = (ncloser < kf).mean(axis=1)
    return ffr, idx


def _concat_coordinate_chunks(coords_dir: Path) -> np.ndarray:
    """Return an object exposing __getitem__ over the ordered coordinate chunks.

    We only index a small set of anchor rows, so we resolve each requested row
    from the appropriate chunk memmap without concatenating 150M rows in RAM.
    """
    chunks = sorted(coords_dir.glob("chunk-*/coordinates.npy"))
    if not chunks:
        raise FileNotFoundError(f"no chunk-*/coordinates.npy under {coords_dir}")
    mmaps = [np.load(c, mmap_mode="r") for c in chunks]
    offsets = np.cumsum([0] + [m.shape[0] for m in mmaps])
    return mmaps, offsets


def anchor_coords_from_chunks(coords_dir: Path, rows: np.ndarray) -> np.ndarray:
    """Gather coordinates[rows] from ordered coordinate chunks (identity order)."""
    mmaps, offsets = _concat_coordinate_chunks(coords_dir)
    rows = np.asarray(rows)
    out = np.empty((rows.shape[0], 2), dtype=np.float64)
    # bucket rows by chunk
    chunk_idx = np.searchsorted(offsets, rows, side="right") - 1
    for ci in np.unique(chunk_idx):
        sel = chunk_idx == ci
        local = rows[sel] - offsets[ci]
        out[sel] = np.asarray(mmaps[ci][local], dtype=np.float64)
    return out


def local_expansion_from_density_v2(
    density_v2_npz: os.PathLike | str, key_prefix: str = "full_150m"
):
    """Local-expansion score from the precomputed density-v2 per-anchor radii.

    density-v2-radii.npz stores ``<prefix>__high_radius`` / ``<prefix>__low_radius``
    (10000 anchors for full_150m).  Score is identical in form to the R0108
    core-panel score: log2(low/high ratio vs median), clipped [-2,2] -> [0,1].
    """
    with np.load(density_v2_npz) as z:
        low = np.asarray(z[f"{key_prefix}__low_radius"])
        high = np.asarray(z[f"{key_prefix}__high_radius"])
    return local_expansion_score(low, high)


def build_r0102_anchor_score(
    reference_npz: os.PathLike | str,
    coords_dir: os.PathLike | str,
    density_v2_npz: os.PathLike | str,
    *,
    key_prefix: str = "full_150m",
    published_ffr: float = R0102_PUBLISHED_FULL_FFR,
    true_ffr_summary: Optional[dict] = None,
) -> dict:
    """Shippable R0102 anchor packet.

    Per-anchor score = local expansion (log2 low/high radius vs median), taken
    from the precomputed density-v2 radii — cheap, per-anchor, map-derived, and
    the same primary signal as the R0108 core panel.  Anchor display coords use
    the GLOBAL substrate rows to index the identity-order 150M coordinate chunks.

    The global FFR (0.5075) is carried as a SUMMARY stat only.  Per-anchor FFR
    is NOT shipped as the score: it is not derivable from reference.npz alone
    (``hi_frac`` is a high-D purity pool, not the low-D FFR pool) and the true
    value requires the expensive low-D top-k_frac map pass (see
    compute_r0102_true_ffr).  When available, pass ``true_ffr_summary`` (from a
    sampled run) to record it honestly in the manifest.
    """
    reference_npz = Path(reference_npz)
    coords_dir = Path(coords_dir)
    score01, log2_norm, median_ratio = local_expansion_from_density_v2(
        density_v2_npz, key_prefix
    )

    sib = reference_npz.parent / "anchor-substrate-rows.npy"
    if sib.exists():
        substrate_rows = np.load(sib)
    else:
        substrate_rows = np.asarray(_load_small_member(reference_npz, "anchor_ids.npy"))
    xy = anchor_coords_from_chunks(coords_dir, substrate_rows)
    if xy.shape[0] != score01.shape[0]:
        raise ValueError(
            f"anchor coords ({xy.shape[0]}) vs radii ({score01.shape[0]}) mismatch"
        )

    summary = {
        "score_label": "local expansion (log2 vs median)",
        "median_radius_ratio": median_ratio,
        "log2_ratio_min": float(np.min(log2_norm)),
        "log2_ratio_median": float(np.median(log2_norm)),
        "log2_ratio_max": float(np.max(log2_norm)),
        "expanded_frac": float(np.mean(log2_norm > 0)),
        "ffr": published_ffr,  # global panel FFR, carried as a summary stat
        "n_anchors": int(score01.shape[0]),
    }
    if true_ffr_summary is not None:
        summary["true_ffr_sample"] = true_ffr_summary

    return {
        "xy": xy.astype("<f4"),
        "score01": score01,
        "score_label": "local expansion (log2 vs median)",
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration helpers (used by builder D; standalone-safe).
# ---------------------------------------------------------------------------

def build_r0108_metrics(
    core_panel_npz: os.PathLike | str,
    ood_npz_paths: Sequence[os.PathLike | str],
    out_dir: os.PathLike | str,
    *,
    labels: Optional[dict] = None,
    texts_resolver: Optional[TextsResolver] = None,
    extent: Optional[Sequence[float]] = None,
) -> dict:
    """Write metrics-anchors.bin + metrics-queries.json for an R0108-family map.

    Returns the `metrics` manifest fragment.
    """
    out_dir = Path(out_dir)
    anc = extract_core_panel_anchors(core_panel_npz)
    count = write_anchors_bin(out_dir / "metrics-anchors.bin", anc["xy"], anc["score01"])

    packets, probe_manifest = build_ood_query_packets(
        ood_npz_paths, labels=labels, texts_resolver=texts_resolver, extent=extent
    )
    write_queries_json(out_dir / "metrics-queries.json", packets)

    return {
        "anchors": {
            "file": "metrics-anchors.bin",
            "count": count,
            "score": anc["score_label"],
            "summary": anc["summary"],
        },
        "probes": probe_manifest,
    }


def build_r0102_metrics(
    reference_npz: os.PathLike | str,
    coords_dir: os.PathLike | str,
    density_v2_npz: os.PathLike | str,
    ood_npz_paths: Sequence[os.PathLike | str],
    out_dir: os.PathLike | str,
    *,
    key_prefix: str = "full_150m",
    labels: Optional[dict] = None,
    texts_resolver: Optional[TextsResolver] = None,
    extent: Optional[Sequence[float]] = None,
    true_ffr_summary: Optional[dict] = None,
) -> dict:
    """Write metrics for the R0102 150M map.

    Anchor score = local expansion (precomputed density-v2 radii); global FFR
    (0.5075) is a summary stat.  Query packets are emitted only for OOD npz that
    embed exact_high_d_top10 + low_d_top50 (R0102 common-corpus / universality
    probes carry external truth, so they are skipped and logged).
    """
    out_dir = Path(out_dir)
    anc = build_r0102_anchor_score(
        reference_npz, coords_dir, density_v2_npz,
        key_prefix=key_prefix, true_ffr_summary=true_ffr_summary,
    )
    count = write_anchors_bin(out_dir / "metrics-anchors.bin", anc["xy"], anc["score01"])

    packets, probe_manifest = build_ood_query_packets(
        ood_npz_paths, labels=labels, texts_resolver=texts_resolver, extent=extent
    )
    write_queries_json(out_dir / "metrics-queries.json", packets)

    return {
        "anchors": {
            "file": "metrics-anchors.bin",
            "count": count,
            "score": anc["score_label"],
            "summary": anc["summary"],
        },
        "probes": probe_manifest,
    }


# ---------------------------------------------------------------------------
# CLI (manual runs; builder D is the main caller).
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Extract map-viz metric packets")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p8 = sub.add_parser("r0108", help="R0108-family core-panel + OOD packets")
    p8.add_argument("--core-panel", required=True)
    p8.add_argument("--ood", nargs="*", default=[])
    p8.add_argument("--out", required=True)

    p2 = sub.add_parser("r0102", help="R0102 150M anchor score + OOD packets")
    p2.add_argument("--reference", required=True)
    p2.add_argument("--coords", required=True)
    p2.add_argument("--density-v2", required=True)
    p2.add_argument("--ood", nargs="*", default=[])
    p2.add_argument("--out", required=True)
    p2.add_argument("--key-prefix", default="full_150m")

    pf = sub.add_parser("true-ffr", help="report TRUE R0102 per-anchor FFR (low-D map pass)")
    pf.add_argument("--reference", required=True)
    pf.add_argument("--coords", required=True)
    pf.add_argument("--eligibility", required=True)
    pf.add_argument("--sample", type=int, default=None)
    pf.add_argument("--seed", type=int, default=0)
    pf.add_argument("--tmp", default="/data/latent-basemap/tmp")

    pm = sub.add_parser("membership", help="report hi_frac membership (=~1.0, NOT FFR)")
    pm.add_argument("--reference", required=True)
    pm.add_argument("--tmp", default="/data/latent-basemap/tmp")

    args = ap.parse_args(argv)
    if args.cmd == "r0108":
        frag = build_r0108_metrics(args.core_panel, args.ood, args.out)
        print(json.dumps(frag, indent=2))
    elif args.cmd == "r0102":
        frag = build_r0102_metrics(
            args.reference, args.coords, args.density_v2, args.ood, args.out,
            key_prefix=args.key_prefix,
        )
        print(json.dumps(frag, indent=2))
    elif args.cmd == "true-ffr":
        ffr, idx = compute_r0102_true_ffr(
            args.reference, args.coords, args.eligibility,
            sample=args.sample, seed=args.seed, tmp_dir=args.tmp, verbose=True,
        )
        print(json.dumps({
            "mean_ffr": float(ffr.mean()),
            "published_full_150m": R0102_PUBLISHED_FULL_FFR,
            "delta": float(ffr.mean()) - R0102_PUBLISHED_FULL_FFR,
            "n_sampled": int(ffr.shape[0]),
            "min": float(ffr.min()), "median": float(np.median(ffr)),
            "max": float(ffr.max()),
        }, indent=2))
    elif args.cmd == "membership":
        mem = compute_r0102_hi_frac_membership(args.reference, tmp_dir=args.tmp)
        print(json.dumps({
            "mean_hi_frac_membership": float(mem.mean()),
            "note": "this is high-D purity-pool membership (~1.0), NOT the FFR",
            "n": int(mem.shape[0]),
        }, indent=2))


if __name__ == "__main__":
    _main()

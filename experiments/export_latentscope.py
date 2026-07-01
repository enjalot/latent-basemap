#!/usr/bin/env python
"""export_latentscope.py — Work Package 5 inspection stack.

Bridges basemap runs into two viewers:

  1. latent-scope Compare page  — writes ``input.parquet`` / ``meta.json`` /
     ``umaps/umap-NNN.{parquet,json}`` / ``embeddings/*.json`` directly onto
     disk (no latent-scope job needed; the Compare page reads these files).
  2. the basemap small-multiples gallery — converts a set of coords parquets +
     per-point diagnostics into ``data/<id>.bin`` float32 blobs + a
     ``manifest.json`` for ``~/.agent/moonshine/basemap-gallery/``.

On-disk schemas were read from latent-scope's ``scripts/ingest.py``,
``scripts/umapper.py`` and ``scripts/scope.py`` (all read-only reference).

Subcommands
-----------
  export-dataset   build a ~100k-row inspection dataset (fineweb-edu chunks +
                   jina-v5-nano embeddings) as a latent-scope dataset.
  export-umap      write a coords parquet as the next umap-NNN entry, with
                   optional Procrustes ``--align-to`` and ``--metrics`` merge.
  merge-metrics    merge per-point metric columns into a dataset's input.parquet.
  fix-squad-demo   repair ~/latent-scope-demo/ls-squad and add PCA + umap-learn
                   projections.
  export-gallery   build the basemap gallery from a spec (or --seed-squad).

Chunk<->embedding alignment (verified 2026-07-01)
-------------------------------------------------
  /data/chunks/fineweb-edu-sample-10BT-chunked-500/train/data-NNNNN-of-00099.parquet
  /data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train/data-NNNNN-of-00099.npy
Same shard basenames; per-shard row counts match exactly (shard 0: 265419 ==
265419, shard 1: 260422 == 260422). => parquet row i of shard N is the text for
npy row i of shard N. export-dataset re-asserts this per shard and aborts if a
count mismatches (the "unambiguous offsets" safety net from the WP5 spec).
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.path.expanduser("~/latent-scope-demo")
CHUNKS_DIR = "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train"
EMB_DIR = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"
JINA_MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
DEFAULT_GALLERY = os.path.expanduser("~/.agent/moonshine/basemap-gallery")
LS_VERSION = "wp5-export"


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def get_data_dir(cli=None):
    return cli or os.environ.get("LATENT_SCOPE_DATA") or DEFAULT_DATA_DIR


def next_numbered(directory, prefix):
    """Next NNN for files like ``<prefix>-NNN.json`` in ``directory``."""
    if not os.path.isdir(directory):
        return 1
    pat = re.compile(rf"{re.escape(prefix)}-(\d+)\.json")
    nums = [int(m.group(1)) for f in os.listdir(directory) if (m := pat.match(f))]
    return max(nums) + 1 if nums else 1


def build_column_metadata(df):
    """Mirror ingest.py's column typing (string/number/date/array) + extents."""
    column_metadata = {}
    for column in df.columns:
        non_null = df[column].dropna()
        if pd.api.types.is_datetime64_any_dtype(non_null):
            ctype = "date"
        elif pd.api.types.is_numeric_dtype(non_null):
            ctype = "number"
        elif pd.api.types.is_string_dtype(non_null):
            ctype = "string"
        elif len(non_null) and isinstance(non_null.iloc[0], (list, np.ndarray)):
            ctype = "array"
        else:
            ctype = "string"
            df[column] = df[column].astype(str)
        try:
            uniq = int(df[column].nunique(dropna=True))
        except Exception:
            uniq = -1
        meta = {"type": ctype, "unique_values_count": uniq}
        if ctype == "number":
            ext = df[column].agg(["min", "max"]).replace([np.inf, -np.inf], np.nan)
            meta["extent"] = [None if pd.isna(x) else float(x) for x in ext.tolist()]
        if ctype == "string" and 0 < uniq <= 100:
            vc = df[column].value_counts()
            meta["categories"] = vc.index.tolist()
            meta["counts"] = {str(k): int(v) for k, v in vc.to_dict().items()}
        column_metadata[column] = meta
    return column_metadata


def write_meta(directory, dataset_id, df, text_column):
    column_metadata = build_column_metadata(df)
    if text_column is None:
        text_column = "text" if "text" in df.columns else next(
            (c for c, m in column_metadata.items() if m["type"] == "string"), None)
    potential = [c for c, m in column_metadata.items() if m["type"] == "array"]
    meta = {
        "id": dataset_id,
        "length": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "text_column": text_column,
        "column_metadata": column_metadata,
        "potential_embeddings": potential,
        "ls_version": LS_VERSION,
    }
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def ensure_dataset_dirs(directory):
    for sub in ("embeddings", "saes", "umaps", "clusters", "scopes", "tags"):
        os.makedirs(os.path.join(directory, sub), exist_ok=True)
    for name in ("\U0001F44D.indices", "\U0001F44E.indices"):
        p = os.path.join(directory, "tags", name)
        if not os.path.exists(p):
            open(p, "w").close()


# --------------------------------------------------------------------------- #
# alignment / normalization (WP5 §4.1 canonical alignment)
# --------------------------------------------------------------------------- #
def similarity_transform(src, ref):
    """Return a callable mapping ``src`` onto ``ref`` by orthogonal Procrustes
    with a single global scale + translation (rotation/reflection/scale). Fit on
    the shared rows passed in; the returned closure applies to any Nx2 array."""
    from scipy.linalg import orthogonal_procrustes

    src = np.asarray(src, float)
    ref = np.asarray(ref, float)
    mu_s = src.mean(0)
    mu_r = ref.mean(0)
    S = src - mu_s
    R = ref - mu_r
    Q, _ = orthogonal_procrustes(S, R)  # min || S Q - R ||, Q orthogonal (2x2)
    denom = float((S * S).sum())
    scale = float((S @ Q * R).sum() / denom) if denom > 0 else 1.0

    def apply(pts):
        pts = np.asarray(pts, float)
        return scale * ((pts - mu_s) @ Q) + mu_r

    return apply


def minmax_unit(coords):
    """Per-axis min-max to [-1, 1] (latent-scope's umapper convention)."""
    coords = np.asarray(coords, float)
    lo = coords.min(0)
    hi = coords.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return 2 * (coords - lo) / span - 1, lo, hi


def uniform_unit(coords, pad=0.02, center=None, half=None):
    """Uniform (aspect-preserving) fit into [-1,1]. If ``center``/``half`` are
    given, apply that fixed frame instead of computing a fresh one — used to put
    every gallery panel in the *same* frame so aligned maps superpose."""
    coords = np.asarray(coords, float)
    lo = coords.min(0)
    hi = coords.max(0)
    if center is None:
        center = (lo + hi) / 2
    if half is None:
        half = float((hi - lo).max()) / 2 or 1.0
    return (coords - center) / half * (1 - pad), center, half


def load_coords(path, length=None):
    """Read x,y (and optional ls_index/index/id) from a coords parquet.
    Returns an (N,2) array. If an index column is present and ``length`` given,
    scatter rows into a full-length array (missing rows -> error)."""
    df = pd.read_parquet(path)
    lower = {c.lower(): c for c in df.columns}
    if "x" not in lower or "y" not in lower:
        raise ValueError(f"{path} needs 'x' and 'y' columns, got {list(df.columns)}")
    coords = df[[lower["x"], lower["y"]]].to_numpy(float)
    idx = None
    for cand in ("ls_index", "index", "id"):
        if cand in lower:
            idx = df[lower[cand]].to_numpy()
            break
    if length is not None and idx is not None and len(coords) != length:
        full = np.full((length, 2), np.nan)
        full[idx.astype(int)] = coords
        if np.isnan(full).any():
            raise ValueError(
                f"{path}: ls_index does not cover all {length} rows "
                f"({len(coords)} provided)")
        coords = full
    return coords


def _write_umap_files(umap_dir, umap_id, coords_unit, lo, hi, embedding_id,
                      neighbors, min_dist, extra=None):
    os.makedirs(umap_dir, exist_ok=True)
    pd.DataFrame(coords_unit, columns=["x", "y"]).to_parquet(
        os.path.join(umap_dir, f"{umap_id}.parquet"))
    meta = {
        "id": umap_id,
        "embedding_id": embedding_id,
        "neighbors": neighbors,
        "min_dist": min_dist,
        "min_values": [float(v) for v in np.asarray(lo).tolist()],
        "max_values": [float(v) for v in np.asarray(hi).tolist()],
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(umap_dir, f"{umap_id}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def write_umap(data_dir, dataset_id, embedding_id, coords, neighbors=25,
               min_dist=0.1, align_to=None, extra=None, umap_id=None):
    """Normalize + write coords as the next umap-NNN entry. With ``align_to``
    (an existing umap id) the coords are Procrustes-aligned first, then fit
    uniformly into [-1,1] (aspect-preserving so alignment survives)."""
    ddir = os.path.join(data_dir, dataset_id)
    umap_dir = os.path.join(ddir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)
    extra = dict(extra or {})
    coords = np.asarray(coords, float)

    if align_to:
        ref_path = os.path.join(umap_dir, f"{align_to}.parquet")
        ref = pd.read_parquet(ref_path)[["x", "y"]].to_numpy(float)
        if len(ref) != len(coords):
            raise ValueError(
                f"align-to {align_to} has {len(ref)} rows, coords have {len(coords)}")
        coords = similarity_transform(coords, ref)(coords)
        unit, _, _ = uniform_unit(coords)
        lo, hi = coords.min(0), coords.max(0)
        extra["align"] = align_to
        extra["align_method"] = "orthogonal_procrustes(scale+rot+refl)+uniform_fit"
    else:
        unit, lo, hi = minmax_unit(coords)

    if umap_id is None:
        umap_id = f"umap-{next_numbered(umap_dir, 'umap'):03d}"
    return umap_id, _write_umap_files(
        umap_dir, umap_id, unit, lo, hi, embedding_id, neighbors, min_dist, extra)


# --------------------------------------------------------------------------- #
# metrics merge (feeds latent-scope issue #131 color-by-metric)
# --------------------------------------------------------------------------- #
def merge_metrics(data_dir, dataset_id, metrics_path):
    ddir = os.path.join(data_dir, dataset_id)
    input_path = os.path.join(ddir, "input.parquet")
    df = pd.read_parquet(input_path)
    m = pd.read_parquet(metrics_path)

    lower = {c.lower(): c for c in m.columns}
    idx_col = next((lower[c] for c in ("ls_index", "index", "id") if c in lower), None)
    if idx_col is not None:
        idx = m[idx_col].to_numpy().astype(int)
        metric_cols = [c for c in m.columns if c != idx_col]
        aligned = pd.DataFrame(index=df.index)
        for c in metric_cols:
            col = np.full(len(df), np.nan)
            col[idx] = m[c].to_numpy()
            aligned[c] = col
        m = aligned
    else:
        if len(m) != len(df):
            raise ValueError(
                f"metrics has {len(m)} rows, dataset has {len(df)}; add an "
                f"ls_index column to merge by id")
        metric_cols = list(m.columns)

    added = []
    for c in metric_cols:
        col = c if c not in df.columns else f"metric_{c}"
        df[col] = pd.to_numeric(m[c].to_numpy(), errors="coerce")
        added.append(col)

    df.to_parquet(input_path)
    meta_path = os.path.join(ddir, "meta.json")
    meta = json.load(open(meta_path))
    cm = meta.get("column_metadata", {})
    for c in added:
        ext = df[c].agg(["min", "max"]).replace([np.inf, -np.inf], np.nan)
        cm[c] = {
            "type": "number",
            "unique_values_count": int(df[c].nunique(dropna=True)),
            "extent": [None if pd.isna(x) else float(x) for x in ext.tolist()],
        }
    meta["column_metadata"] = cm
    meta["columns"] = df.columns.tolist()
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"merged {len(added)} metric column(s) into {input_path}: {added}")
    return added


# --------------------------------------------------------------------------- #
# export-dataset
# --------------------------------------------------------------------------- #
def export_dataset(dataset_id, n, data_dir, chunks_dir, emb_dir, write_embeddings=True):
    data_dir = get_data_dir(data_dir)
    directory = os.path.join(data_dir, dataset_id)
    os.makedirs(directory, exist_ok=True)
    ensure_dataset_dirs(directory)

    cshards = sorted(glob.glob(os.path.join(chunks_dir, "*.parquet")))
    eshards = sorted(glob.glob(os.path.join(emb_dir, "*.npy")))
    if not cshards or not eshards:
        raise SystemExit(f"no shards found: {chunks_dir} / {emb_dir}")

    # match by shard basename stem (data-NNNNN-of-00099)
    estem = {os.path.splitext(os.path.basename(p))[0]: p for p in eshards}
    pairs = [(cp, estem[os.path.splitext(os.path.basename(cp))[0]])
             for cp in cshards
             if os.path.splitext(os.path.basename(cp))[0] in estem]
    if not pairs:
        raise SystemExit("no chunk/embedding shards share a basename stem")
    per_shard = max(1, n // len(pairs))

    text_cols = ["chunk_text", "chunk_token_count", "id", "url", "score",
                 "dump", "chunk_index"]
    frames, embs = [], []
    taken = 0
    for cp, ep in pairs:
        if taken >= n:
            break
        df = pd.read_parquet(cp, columns=text_cols)
        arr = np.load(ep, mmap_mode="r")
        if len(df) != arr.shape[0]:
            raise SystemExit(
                f"ALIGNMENT FAIL: {cp} has {len(df)} rows but {ep} has "
                f"{arr.shape[0]}; refusing to build a misaligned dataset")
        take = min(per_shard, len(df), n - taken)
        sel = np.linspace(0, len(df) - 1, take).astype(int)
        sel = np.unique(sel)
        sub = df.iloc[sel].copy()
        sub["source_shard"] = os.path.basename(cp)
        sub["source_row"] = sel
        frames.append(sub)
        if write_embeddings:
            embs.append(np.asarray(arr[sel], dtype=np.float16))
        taken += len(sel)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"chunk_text": "text"})
    # keep chunk_tokens OUT (huge array column); text_column is the rendered text
    out.to_parquet(os.path.join(directory, "input.parquet"))
    write_meta(directory, dataset_id, out, text_column="text")
    print(f"wrote {directory}/input.parquet + meta.json ({len(out)} rows)")

    if write_embeddings:
        import h5py

        E = np.concatenate(embs, axis=0)
        emb_id = "embedding-001"
        h5_path = os.path.join(directory, "embeddings", f"{emb_id}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=E)
        Ef = E.astype(np.float32)
        json.dump({
            "id": emb_id,
            "model_id": JINA_MODEL_ID,
            "dataset_id": dataset_id,
            "text_column": "text",
            "dimensions": int(E.shape[1]),
            "late_interaction": False,
            "min_values": E.min(0).astype(float).tolist(),
            "max_values": E.max(0).astype(float).tolist(),
            "note": "fp16 jina-v5-nano pooled; row-aligned to input.parquet "
                    "via verified per-shard chunk<->embedding correspondence",
        }, open(os.path.join(directory, "embeddings", f"{emb_id}.json"), "w"), indent=2)
        del Ef
        print(f"wrote embeddings/{emb_id}.h5 ({E.shape[0]}x{E.shape[1]} fp16)")

    print(f"dataset '{dataset_id}' ready at {directory}")
    return directory


# --------------------------------------------------------------------------- #
# fix-squad-demo
# --------------------------------------------------------------------------- #
def fix_squad_demo(data_dir, run_umap=True):
    import h5py

    data_dir = get_data_dir(data_dir)
    dataset_id = "ls-squad"
    directory = os.path.join(data_dir, dataset_id)
    ensure_dataset_dirs(directory)

    emb_h5 = os.path.join(directory, "embeddings", "embedding-003.h5")
    if not os.path.exists(emb_h5):
        raise SystemExit(f"missing {emb_h5} (expected symlink to /data/latent-basemap)")
    with h5py.File(emb_h5, "r") as f:
        E = np.array(f["embeddings"], dtype=np.float32)
    n = E.shape[0]
    print(f"ls-squad embeddings: {E.shape}")

    # input.parquet — placeholder text (real SQuAD text not on this machine)
    df = pd.DataFrame({
        "id": np.arange(n),
        "text": [f"squad row {i}" for i in range(n)],
    })
    df.to_parquet(os.path.join(directory, "input.parquet"))
    write_meta(directory, dataset_id, df, text_column="text")
    print("wrote input.parquet + meta.json (PLACEHOLDER text: 'squad row N' — "
          "real SQuAD passages are not available locally)")

    # embedding-003.json (nomic-embed-text-v1.5, 768d)
    json.dump({
        "id": "embedding-003",
        "model_id": "transformers-nomic-ai___nomic-embed-text-v1.5",
        "dataset_id": dataset_id,
        "text_column": "text",
        "dimensions": int(E.shape[1]),
        "late_interaction": False,
        "min_values": E.min(0).astype(float).tolist(),
        "max_values": E.max(0).astype(float).tolist(),
        "note": "embeddings recovered from /data/latent-basemap/ls-squad; "
                "text unavailable so input.parquet uses placeholder rows",
    }, open(os.path.join(directory, "embeddings", "embedding-003.json"), "w"), indent=2)

    # umap-001.json — reference layout (parquet already present, already [-1,1])
    ref = pd.read_parquet(os.path.join(directory, "umaps", "umap-001.parquet"))
    json.dump({
        "id": "umap-001",
        "embedding_id": "embedding-003",
        "neighbors": 25,
        "min_dist": 0.1,
        "min_values": ref[["x", "y"]].min().astype(float).tolist(),
        "max_values": ref[["x", "y"]].max().astype(float).tolist(),
        "note": "reference layout recovered from disk; original UMAP "
                "hyperparameters unknown (neighbors/min_dist are placeholders)",
    }, open(os.path.join(directory, "umaps", "umap-001.json"), "w"), indent=2)
    print("wrote embedding-003.json + umap-001.json")

    if not run_umap:
        return directory

    # umap-002: PCA-2D floor, aligned to umap-001
    from sklearn.decomposition import PCA

    pca_xy = PCA(n_components=2, random_state=0).fit_transform(E)
    uid, _ = write_umap(data_dir, dataset_id, "embedding-003", pca_xy,
                        neighbors=0, min_dist=0.0, align_to="umap-001",
                        extra={"method": "PCA-2D (linear floor)"},
                        umap_id="umap-002")
    print(f"wrote {uid} (PCA-2D, aligned to umap-001)")

    # umap-003: fresh umap-learn run, aligned to umap-001
    import umap

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="cosine",
                        random_state=42, n_components=2, verbose=True)
    umap_xy = reducer.fit_transform(E)
    uid2, _ = write_umap(data_dir, dataset_id, "embedding-003", umap_xy,
                         neighbors=25, min_dist=0.1, align_to="umap-001",
                         extra={"method": "umap-learn cosine n_neighbors=25 "
                                          "min_dist=0.1 seed=42"},
                         umap_id="umap-003")
    print(f"wrote {uid2} (umap-learn, aligned to umap-001)")
    return directory


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def _local_density(xy, k=15):
    """Mean distance to k nearest neighbours in 2D (small -> dense)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=min(k + 1, len(xy)))
    return d[:, 1:].mean(1)


def build_gallery(out_dir, maps, reference_id=None, max_points=20000,
                  title="basemap gallery", lede="", seed=0):
    """maps: list of dicts {id,label,group,dataset,coords(Nx2 or path),
    metrics(dict name->array or path)}. All maps must share row order/ids."""
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    # load coords
    loaded = []
    for m in maps:
        coords = m["coords"]
        if isinstance(coords, str):
            coords = load_coords(coords)
        loaded.append(np.asarray(coords, float))
    N = len(loaded[0])
    for m, c in zip(maps, loaded):
        if len(c) != N:
            raise ValueError(f"map {m['id']} has {len(c)} rows, expected {N}")

    if reference_id is None:
        reference_id = maps[0]["id"]
    ref_idx = next(i for i, m in enumerate(maps) if m["id"] == reference_id)
    ref_coords = loaded[ref_idx]

    # shared frame from the aligned reference (uniform, aspect-preserving)
    _, center, half = uniform_unit(ref_coords)

    # common subsample (same rows across all panels -> linked hover by index)
    if N > max_points:
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(N, size=max_points, replace=False))
    else:
        sub = np.arange(N)

    manifest_maps = []
    for m, coords in zip(maps, loaded):
        aligned = similarity_transform(coords, ref_coords)(coords)
        unit, _, _ = uniform_unit(aligned, center=center, half=half)
        pts = unit[sub].astype(np.float32)
        bin_name = f"data/{m['id']}.bin"
        np.ascontiguousarray(pts).tofile(os.path.join(out_dir, bin_name))

        # metrics: per-map drift vs reference + local density, plus any provided
        metric_cols = {}
        provided = m.get("metrics", {}) or {}
        prov = {}
        for name, val in provided.items():
            arr = load_coords(val)[:, 0] if isinstance(val, str) else np.asarray(val, float)
            prov[name] = arr
        drift = np.linalg.norm(
            similarity_transform(coords, ref_coords)(coords) - ref_coords, axis=1)
        prov.setdefault("drift", drift)
        prov.setdefault("density", _local_density(unit))
        scalar = {"mean_drift": float(drift[sub].mean()),
                  "n": int(len(sub))}
        scalar.update({k: float(v) for k, v in m.get("scalar_metrics", {}).items()})
        for name, arr in prov.items():
            mb = f"data/{m['id']}_{name}.bin"
            np.ascontiguousarray(arr[sub].astype(np.float32)).tofile(
                os.path.join(out_dir, mb))
            metric_cols[name] = mb

        manifest_maps.append({
            "id": m["id"],
            "label": m.get("label", m["id"]),
            "group": m.get("group", ""),
            "dataset": m.get("dataset", ""),
            "points": bin_name,
            "n": int(len(sub)),
            "metrics": scalar,
            "metricColumns": metric_cols,
        })

    manifest = {
        "title": title,
        "lede": lede,
        "reference": reference_id,
        "extent": [-1, 1],
        "total_rows": int(N),
        "maps": manifest_maps,
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    write_gallery_html(out_dir)
    print(f"wrote gallery: {out_dir}/manifest.json + index.html "
          f"({len(manifest_maps)} panels, {len(sub)} pts each)")
    return out_dir


def gallery_from_squad(data_dir, out_dir, max_points=20000):
    data_dir = get_data_dir(data_dir)
    ddir = os.path.join(data_dir, "ls-squad", "umaps")
    specs = [
        ("umap-001", "reference layout", "reference"),
        ("umap-002", "PCA-2D (floor)", "floor"),
        ("umap-003", "umap-learn (cosine)", "umap"),
    ]
    maps = []
    for uid, label, group in specs:
        p = os.path.join(ddir, f"{uid}.parquet")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run fix-squad-demo first")
        maps.append({"id": uid, "label": label, "group": group,
                     "dataset": "ls-squad", "coords": p})
    return build_gallery(
        out_dir, maps, reference_id="umap-001", max_points=max_points,
        title="basemap gallery · ls-squad",
        lede="Three projections of the same 20,958 nomic-embed SQuAD vectors, "
             "Procrustes-aligned to the reference layout so panels superpose. "
             "Hover a point to highlight the same row across all panels; click "
             "to pin. Color-by drift (distance from the reference layout) or "
             "local density.")


def export_gallery_from_spec(spec_path, out_dir, reference_id, max_points):
    spec = json.load(open(spec_path))
    maps = spec["maps"] if isinstance(spec, dict) else spec
    reference_id = reference_id or (spec.get("reference") if isinstance(spec, dict) else None)
    return build_gallery(
        out_dir, maps, reference_id=reference_id, max_points=max_points,
        title=(spec.get("title") if isinstance(spec, dict) else None) or "basemap gallery",
        lede=(spec.get("lede") if isinstance(spec, dict) else "") or "")


# --------------------------------------------------------------------------- #
# gallery HTML (self-contained, D3 v7 CDN, canvas panels; paper/ink palette)
# --------------------------------------------------------------------------- #
def write_gallery_html(out_dir):
    html = GALLERY_HTML
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


GALLERY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>basemap gallery · small multiples</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Sans+3:wght@400;600;700&family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --body-font: 'Source Serif 4', Georgia, serif;
  --heading-font: 'Source Sans 3', system-ui, sans-serif;
  --mono-font: 'Source Code Pro', monospace;
  --ink: #1a1613; --ink-2: #5a554e; --ink-3: #8a867f;
  --paper: #fbf8f3; --paper-2: #f2ede3; --rule: #d9d3c5; --cell: #ffffff;
  --accent: #2b628a; --pt: #5a554e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e9e4d8; --ink-2: #aaa49a; --ink-3: #807b72;
    --paper: #1a1814; --paper-2: #252118; --rule: #3a3528; --cell: #22201a;
    --accent: #7fb0e0; --pt: #9a948a;
  }
}
body { font-family: var(--body-font); color: var(--ink); background: var(--paper);
  -webkit-font-smoothing: antialiased; line-height: 1.45; }
.app { padding: 1rem 1.25rem 3rem; max-width: 1600px; margin: 0 auto; }
header { font-family: var(--heading-font); margin-bottom: 0.6rem; }
h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.2rem; }
.lede { font-size: 0.9rem; color: var(--ink-2); margin-bottom: 0.6rem; max-width: 95ch; }
.controls { display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;
  font-family: var(--heading-font); font-size: 0.85rem; padding: 0.4rem 0.6rem;
  background: var(--cell); border: 1px solid var(--rule); margin-bottom: 0.5rem; }
.controls .group { display: flex; gap: 0.4rem; align-items: center; }
.controls label { color: var(--ink-3); margin-right: 0.3rem; }
.controls select { font-family: var(--heading-font); font-size: 0.82rem;
  padding: 0.2em 0.4em; border: 1px solid var(--rule); background: var(--paper-2);
  color: var(--ink); border-radius: 2px; }
#status { color: var(--ink-3); font-size: 0.82rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 0.75rem; }
.panel { background: var(--cell); border: 1px solid var(--rule);
  padding: 0.5rem 0.6rem 0.55rem; position: relative; }
.panel h3 { font-family: var(--heading-font); font-size: 0.92rem; font-weight: 600;
  color: var(--ink-2); margin-bottom: 0.1rem; }
.panel .subtitle { font-family: var(--heading-font); font-size: 0.72rem;
  color: var(--ink-3); margin-bottom: 0.35rem; }
.panel .caption { font-family: var(--mono-font); font-size: 0.72rem;
  color: var(--ink-3); margin-top: 0.3rem; }
canvas.scatter { display: block; width: 100%; aspect-ratio: 1 / 1; cursor: crosshair;
  background: var(--paper-2); border: 1px solid var(--rule); }
.group-hdr { grid-column: 1 / -1; font-family: var(--heading-font); font-size: 0.8rem;
  color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.06em;
  border-bottom: 1px solid var(--rule); padding: 0.5rem 0 0.15rem; margin-top: 0.3rem; }
.detail { margin-top: 0.7rem; padding: 0.6rem 0.8rem; background: var(--cell);
  border: 1px solid var(--rule); font-family: var(--mono-font); font-size: 0.82rem;
  color: var(--ink-2); min-height: 2.6rem; }
.swatchbar { display: inline-flex; align-items: center; gap: 0.3rem; margin-left: 0.4rem; }
.swatchbar .bar { width: 90px; height: 10px; border: 1px solid var(--rule);
  background: linear-gradient(to right, #2b628a, #d8b070, #b06252); }
</style>
</head>
<body>
<div class="app">
<header>
  <h1 id="title">basemap gallery</h1>
  <p class="lede" id="lede"></p>
</header>

<div class="controls">
  <div class="group"><label>group by</label>
    <select id="groupBy"><option value="">— none —</option><option value="group">group</option><option value="dataset">dataset</option></select></div>
  <div class="group"><label>sort by</label>
    <select id="sortBy"><option value="">manifest order</option><option value="label">label</option><option value="group">group</option><option value="mean_drift">mean drift</option></select></div>
  <div class="group"><label>color by</label>
    <select id="colorBy"><option value="">— none —</option></select>
    <span class="swatchbar" id="swatch" style="display:none"><span class="bar"></span></span></div>
  <div class="group"><span id="status">loading…</span></div>
</div>

<div class="grid" id="grid"></div>

<div class="detail" id="detail">hover a point to highlight the same row across every panel · click to pin</div>

<script>
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const EXTENT = 1;              // coords are pre-aligned into [-1,1]
const PAD = 0.03;
let MANIFEST = null;
const MAPS = {};               // id -> {pts:Float32Array, metrics:{name:Float32Array}, meta}
let colorField = "";
let hovered = -1, pinned = -1;
const canvases = {};           // id -> {cnv, ctx, w, h, dpr}

async function loadBin(url) {
  const buf = await fetch(url).then(r => r.arrayBuffer());
  return new Float32Array(buf);
}

async function boot() {
  MANIFEST = await fetch("manifest.json").then(r => r.json());
  document.getElementById("title").textContent = MANIFEST.title || "basemap gallery";
  document.getElementById("lede").textContent = MANIFEST.lede || "";

  // union of metric column names for color-by dropdown
  const metricNames = new Set();
  for (const m of MANIFEST.maps) {
    const pts = await loadBin(m.points);
    const metrics = {};
    for (const [name, url] of Object.entries(m.metricColumns || {})) {
      metrics[name] = await loadBin(url);
      metricNames.add(name);
    }
    MAPS[m.id] = { pts, metrics, meta: m };
  }
  const cb = document.getElementById("colorBy");
  [...metricNames].sort().forEach(n => {
    const o = document.createElement("option"); o.value = n; o.textContent = n; cb.appendChild(o);
  });

  document.getElementById("status").textContent =
    `${MANIFEST.maps.length} maps · ${MANIFEST.maps[0]?.n?.toLocaleString?.() || "?"} pts each · ` +
    `${MANIFEST.total_rows?.toLocaleString?.() || "?"} rows total`;

  document.getElementById("groupBy").addEventListener("change", render);
  document.getElementById("sortBy").addEventListener("change", render);
  cb.addEventListener("change", (e) => { colorField = e.target.value; drawAll(); updateSwatch(); });
  render();
}

function orderedMaps() {
  let ms = MANIFEST.maps.slice();
  const sortBy = document.getElementById("sortBy").value;
  if (sortBy === "mean_drift") ms.sort((a,b) => (a.metrics?.mean_drift||0) - (b.metrics?.mean_drift||0));
  else if (sortBy) ms.sort((a,b) => String(a[sortBy]).localeCompare(String(b[sortBy])));
  return ms;
}

function render() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const k of Object.keys(canvases)) delete canvases[k];
  const groupBy = document.getElementById("groupBy").value;
  const ms = orderedMaps();

  const groups = new Map();
  for (const m of ms) {
    const g = groupBy ? String(m[groupBy] ?? "") : "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(m);
  }
  for (const [g, list] of groups) {
    if (groupBy && g !== "") {
      const hdr = document.createElement("div");
      hdr.className = "group-hdr"; hdr.textContent = `${groupBy}: ${g}`;
      grid.appendChild(hdr);
    }
    for (const m of list) grid.appendChild(makePanel(m));
  }
  requestAnimationFrame(() => { setupCanvases(); drawAll(); });
  updateSwatch();
}

function makePanel(m) {
  const div = document.createElement("div");
  div.className = "panel";
  const cap = captionFor(m);
  div.innerHTML =
    `<h3>${m.label}</h3><div class="subtitle">${m.id} · ${m.group||""} · ${m.dataset||""}</div>` +
    `<canvas class="scatter" id="cv-${m.id}"></canvas>` +
    `<div class="caption">${cap}</div>`;
  return div;
}

function captionFor(m) {
  const parts = [];
  for (const [k, v] of Object.entries(m.metrics || {})) {
    parts.push(`${k}=${typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}`);
  }
  return parts.join(" · ");
}

function setupCanvases() {
  for (const m of MANIFEST.maps) {
    const cnv = document.getElementById(`cv-${m.id}`);
    if (!cnv) continue;
    const dpr = window.devicePixelRatio || 1;
    const rect = cnv.getBoundingClientRect();
    cnv.width = Math.max(1, rect.width * dpr);
    cnv.height = Math.max(1, rect.height * dpr);
    const ctx = cnv.getContext("2d");
    ctx.scale(dpr, dpr);
    canvases[m.id] = { cnv, ctx, w: rect.width, h: rect.height, dpr };
    cnv.onmousemove = (e) => onMouse(m.id, e);
    cnv.onmouseleave = () => { if (hovered !== -1) { hovered = -1; drawAll(); renderDetail(); } };
    cnv.onclick = (e) => onClick(m.id, e);
  }
}

// identical extent for every panel so aligned maps superpose
function proj(id, i) {
  const c = canvases[id], p = MAPS[id].pts;
  const s = (1 - 2*PAD);
  const x = (p[2*i] / EXTENT * 0.5 + 0.5);       // [-1,1] -> [0,1]
  const y = (p[2*i+1] / EXTENT * 0.5 + 0.5);
  return [(PAD + x*s) * c.w, (PAD + (1-y)*s) * c.h];
}

let colorScale = null;
function rebuildColorScale() {
  colorScale = null;
  if (!colorField) return;
  let lo = Infinity, hi = -Infinity;
  for (const m of MANIFEST.maps) {
    const arr = MAPS[m.id].metrics[colorField];
    if (!arr) continue;
    for (let i = 0; i < arr.length; i++) { const v = arr[i];
      if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (lo < hi) colorScale = d3.scaleSequential(d3.interpolateInferno).domain([lo, hi]);
}

function drawAll() {
  rebuildColorScale();
  for (const m of MANIFEST.maps) drawPanel(m.id);
}

function drawPanel(id) {
  const c = canvases[id]; if (!c) return;
  const { ctx, w, h } = c;
  const p = MAPS[id].pts;
  const n = p.length / 2;
  ctx.clearRect(0, 0, w, h);
  const metric = colorField ? MAPS[id].metrics[colorField] : null;
  const base = css('--pt');
  ctx.globalAlpha = 0.55;
  for (let i = 0; i < n; i++) {
    const [cx, cy] = proj(id, i);
    ctx.fillStyle = (metric && colorScale) ? colorScale(metric[i]) : base;
    ctx.beginPath(); ctx.arc(cx, cy, 1.5, 0, 6.283); ctx.fill();
  }
  ctx.globalAlpha = 1;
  drawMark(id, pinned, css('--accent'), 5);
  drawMark(id, hovered, css('--ink'), 4);
}

function drawMark(id, i, color, r) {
  if (i < 0) return;
  const c = canvases[id]; if (!c) return;
  const [cx, cy] = proj(id, i);
  const ctx = c.ctx;
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 6.283); ctx.fill();
  ctx.strokeStyle = css('--paper'); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx, cy, r + 1.5, 0, 6.283); ctx.stroke();
}

function nearest(id, mx, my, maxD = 12) {
  const p = MAPS[id].pts, n = p.length / 2;
  let best = -1, bd = maxD * maxD;
  for (let i = 0; i < n; i++) {
    const [cx, cy] = proj(id, i);
    const dx = cx - mx, dy = cy - my, d = dx*dx + dy*dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function onMouse(id, e) {
  const c = canvases[id]; const rect = c.cnv.getBoundingClientRect();
  const i = nearest(id, e.clientX - rect.left, e.clientY - rect.top);
  if (i !== hovered) { hovered = i; drawAll(); renderDetail(); }
}
function onClick(id, e) {
  const c = canvases[id]; const rect = c.cnv.getBoundingClientRect();
  const i = nearest(id, e.clientX - rect.left, e.clientY - rect.top);
  pinned = (pinned === i) ? -1 : i;
  drawAll(); renderDetail();
}

function renderDetail() {
  const el = document.getElementById("detail");
  const i = hovered !== -1 ? hovered : pinned;
  if (i < 0) { el.textContent = "hover a point to highlight the same row across every panel · click to pin"; return; }
  const parts = [`row #${i}`];
  if (colorField) {
    for (const m of MANIFEST.maps) {
      const arr = MAPS[m.id].metrics[colorField];
      if (arr) parts.push(`${m.id}:${colorField}=${arr[i].toFixed(3)}`);
    }
  }
  el.textContent = parts.join("  ·  ") + (pinned === i ? "   [pinned]" : "");
}

function updateSwatch() {
  document.getElementById("swatch").style.display = colorField ? "inline-flex" : "none";
}

window.addEventListener("resize", () => { setupCanvases(); drawAll(); });
boot();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("export-dataset", help="build ~100k-row inspection dataset")
    d.add_argument("--dataset-id", default="fineweb-edu-jina-100k")
    d.add_argument("--n", type=int, default=100000)
    d.add_argument("--data-dir", default=None)
    d.add_argument("--chunks-dir", default=CHUNKS_DIR)
    d.add_argument("--emb-dir", default=EMB_DIR)
    d.add_argument("--no-embeddings", action="store_true")

    u = sub.add_parser("export-umap", help="write a coords parquet as umap-NNN")
    u.add_argument("--dataset-id", required=True)
    u.add_argument("--embedding-id", required=True)
    u.add_argument("--coords", required=True, help="parquet with x,y[,ls_index]")
    u.add_argument("--data-dir", default=None)
    u.add_argument("--neighbors", type=int, default=25)
    u.add_argument("--min-dist", type=float, default=0.1)
    u.add_argument("--align-to", default=None, help="existing umap id to Procrustes-align to")
    u.add_argument("--metrics", default=None, help="diagnostics parquet to merge into input.parquet")
    u.add_argument("--label", default=None)

    mm = sub.add_parser("merge-metrics", help="merge per-point metrics into input.parquet")
    mm.add_argument("--dataset-id", required=True)
    mm.add_argument("--metrics", required=True)
    mm.add_argument("--data-dir", default=None)

    s = sub.add_parser("fix-squad-demo", help="repair ls-squad + add PCA/umap-learn")
    s.add_argument("--data-dir", default=None)
    s.add_argument("--no-umap", action="store_true", help="skip PCA/umap-learn generation")

    g = sub.add_parser("export-gallery", help="build the basemap gallery")
    g.add_argument("--out", default=DEFAULT_GALLERY)
    g.add_argument("--spec", default=None, help="JSON spec: {maps:[...],reference,title,lede}")
    g.add_argument("--seed-squad", action="store_true", help="seed from ls-squad projections")
    g.add_argument("--reference", default=None)
    g.add_argument("--max-points", type=int, default=20000)
    g.add_argument("--data-dir", default=None)

    a = p.parse_args()

    if a.cmd == "export-dataset":
        export_dataset(a.dataset_id, a.n, a.data_dir, a.chunks_dir, a.emb_dir,
                       write_embeddings=not a.no_embeddings)
    elif a.cmd == "export-umap":
        data_dir = get_data_dir(a.data_dir)
        length = json.load(open(os.path.join(data_dir, a.dataset_id, "meta.json")))["length"]
        coords = load_coords(a.coords, length)
        extra = {"label": a.label} if a.label else None
        uid, _ = write_umap(data_dir, a.dataset_id, a.embedding_id, coords,
                            neighbors=a.neighbors, min_dist=a.min_dist,
                            align_to=a.align_to, extra=extra)
        print(f"wrote {uid} for dataset {a.dataset_id}")
        if a.metrics:
            merge_metrics(data_dir, a.dataset_id, a.metrics)
    elif a.cmd == "merge-metrics":
        merge_metrics(get_data_dir(a.data_dir), a.dataset_id, a.metrics)
    elif a.cmd == "fix-squad-demo":
        fix_squad_demo(a.data_dir, run_umap=not a.no_umap)
    elif a.cmd == "export-gallery":
        if a.seed_squad:
            gallery_from_squad(a.data_dir, a.out, max_points=a.max_points)
        elif a.spec:
            export_gallery_from_spec(a.spec, a.out, a.reference, a.max_points)
        else:
            raise SystemExit("export-gallery needs --seed-squad or --spec")


if __name__ == "__main__":
    main()

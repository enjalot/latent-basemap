"""Assemble the MiniLM 2M mixed substrate and build its exact k15 fuzzy graph."""
from __future__ import annotations

import glob, os, time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy, atomic_save_new_npz, atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0216_minilm_2m_substrate import (
    CAPABILITY, COMPOSITION, DIMENSION, GRAPH_K, GRAPH_SCHEMA,
    MEAN_RECALL_FLOOR, P10_RECALL_FLOOR, RAW_FORMAT, RECALL_PROBE_ROWS,
    RECALL_PROBE_SEED, ROUND_ID, ROWS, ROW_POLICY, Round0216Error,
    SELECTION_SEED, SUBSTRATE_SCHEMA, TRAILING_FRAGMENT_POLICY,
    resolve_shard_rows, validate_composition, validate_graph,
)
from basemap import round0113_prompt_contrast as prompt_contract

EMB = "/data/embeddings"
SEARCH_BLOCK = 100_000
QUERY_BLOCK = 16_384


def _verify_sizes(job: Mapping[str, Any]) -> None:
    """Re-verify every base-corpus shard byte size against the bound manifest."""
    sig = job.get("source_size_manifest")
    if sig is None:
        raise Round0216Error("R0216 requires the bound source size manifest")
    path = prompt_contract.verify_signature(sig, label="R0216 source size manifest")
    import json as _json
    manifest = _json.load(open(path))
    drift = []
    for corpus, entry in (manifest.get("corpora") or {}).items():
        for rel, want in (entry.get("shard_sizes") or {}).items():
            actual = os.path.getsize(os.path.join(EMB, rel))
            if actual != int(want):
                drift.append(f"{rel}: {actual} != {want}")
    if drift:
        raise Round0216Error(
            f"R0216 source shards changed size since preparation: {drift[:4]}"
        )


def _shards(corpus: str) -> list[tuple[str, int, bool]]:
    """(path, complete_rows, is_real_npy) honouring the R0025 loading contract."""
    out = []
    for path in sorted(glob.glob(os.path.join(EMB, corpus, "train", "*.npy"))):
        if path.endswith(".tmp.npy"):
            continue
        with open(path, "rb") as h:
            real_npy = h.read(6) == b"\x93NUMPY"
        if real_npy:
            rows = int(np.load(path, mmap_mode="r").shape[0])
        else:
            rel = os.path.relpath(path, EMB)
            rows = resolve_shard_rows(relative_path=rel, size_bytes=os.path.getsize(path))
        out.append((path, rows, real_npy))
    if not out:
        raise Round0216Error(f"no shards for {corpus}")
    return out


def _open(path: str, rows: int, real_npy: bool) -> np.ndarray:
    if real_npy:
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, DIMENSION))


def run_assemble_and_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0216Error("R0216 handler received another queue")
    started = time.monotonic()
    _verify_sizes(job)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0216 substrate+graph")

    X = np.empty((ROWS, DIMENSION), dtype=np.float32)
    provenance = np.empty(ROWS, dtype=np.dtype([
        ("corpus", "u1"), ("shard", "u2"), ("row", "i8")]))
    counts: dict[str, int] = {}
    sources: dict[str, Any] = {}
    at = 0
    for ci, (corpus, want) in enumerate(COMPOSITION):
        shards = _shards(corpus)
        total = sum(r for _p, r, _n in shards)
        if total < want:
            raise Round0216Error(f"{corpus}: need {want} rows, corpus has {total}")
        rng = np.random.RandomState(SELECTION_SEED + ci)
        picks = np.sort(rng.choice(total, want, replace=False)).astype(np.int64)
        offs, acc = [], 0
        for _p, r, _n in shards:
            offs.append(acc); acc += r
        offs = np.asarray(offs, dtype=np.int64)
        shard_of = np.searchsorted(offs, picks, side="right") - 1
        sig_list = []
        for si, (path, rows, real_npy) in enumerate(shards):
            local = picks[shard_of == si] - offs[si]
            if local.size == 0:
                continue
            arr = _open(path, rows, real_npy)
            block = np.asarray(arr[local], dtype=np.float32)
            X[at:at + len(local)] = block
            provenance["corpus"][at:at + len(local)] = ci
            provenance["shard"][at:at + len(local)] = si
            provenance["row"][at:at + len(local)] = local
            at += len(local)
            del arr, block
        counts[corpus] = want
        sources[corpus] = {
            "shards": len(shards), "corpus_rows": int(total),
            "selected_rows": int(want),
            "format": "npy" if shards[0][2] else RAW_FORMAT,
            "first_shard": expected_input_signature(shards[0][0]),
        }
    if at != ROWS:
        raise Round0216Error(f"assembled {at} rows, expected {ROWS}")
    composition = validate_composition(counts)

    norms = np.linalg.norm(X, axis=1)
    if not np.isfinite(X).all() or float(norms.min()) <= 0:
        raise Round0216Error("substrate contains nonfinite or zero rows")
    X /= norms[:, None]

    sub_path = atomic_save_new_npy(os.path.join(output, "substrate.f32.npy"), X, immutable=True)
    prov_path = atomic_save_new_npy(os.path.join(output, "provenance.npy"), provenance, immutable=True)

    # ---- exact k15 graph: brute force, blocked ----
    device = torch.device("cuda")
    Xt = torch.from_numpy(X).to(device)
    nbr = np.empty((ROWS, GRAPH_K), dtype=np.int32)
    dist = np.empty((ROWS, GRAPH_K), dtype=np.float32)
    search_started = time.monotonic()
    for qs in range(0, ROWS, QUERY_BLOCK):
        qe = min(qs + QUERY_BLOCK, ROWS)
        q = Xt[qs:qe]
        best_s = torch.full((qe - qs, GRAPH_K + 1), -float("inf"), device=device)
        best_i = torch.full((qe - qs, GRAPH_K + 1), -1, device=device, dtype=torch.int64)
        for cs in range(0, ROWS, SEARCH_BLOCK):
            ce = min(cs + SEARCH_BLOCK, ROWS)
            sims = q @ Xt[cs:ce].T
            k = min(GRAPH_K + 1, ce - cs)
            ts, ti = torch.topk(sims, k, dim=1)
            cs_ = torch.cat([best_s, ts], 1); ci_ = torch.cat([best_i, ti.to(torch.int64) + cs], 1)
            order = torch.argsort(cs_, dim=1, descending=True)[:, : GRAPH_K + 1]
            best_s = torch.gather(cs_, 1, order); best_i = torch.gather(ci_, 1, order)
            del sims, ts, ti, cs_, ci_, order
        ids = best_i.cpu().numpy(); sims = best_s.cpu().numpy()
        for n in range(qe - qs):
            row = qs + n
            keep = [(int(i), float(s)) for i, s in zip(ids[n], sims[n]) if int(i) != row][:GRAPH_K]
            if len(keep) < GRAPH_K:
                raise Round0216Error(f"row {row} got {len(keep)} neighbours")
            nbr[row] = [i for i, _ in keep]
            dist[row] = [1.0 - s for _, s in keep]
    search_s = time.monotonic() - search_started
    del Xt
    torch.cuda.empty_cache()

    # recall probe against an independent brute-force pass
    rng = np.random.RandomState(RECALL_PROBE_SEED)
    probe = np.sort(rng.choice(ROWS, RECALL_PROBE_ROWS, replace=False))
    Xt = torch.from_numpy(X).to(device)
    qp = Xt[torch.from_numpy(probe).to(device)]
    truth_s = torch.full((len(probe), GRAPH_K + 1), -float("inf"), device=device)
    truth_i = torch.full((len(probe), GRAPH_K + 1), -1, device=device, dtype=torch.int64)
    for cs in range(0, ROWS, SEARCH_BLOCK):
        ce = min(cs + SEARCH_BLOCK, ROWS)
        sims = qp @ Xt[cs:ce].T
        k = min(GRAPH_K + 1, ce - cs)
        ts, ti = torch.topk(sims, k, dim=1)
        cs_ = torch.cat([truth_s, ts], 1); ci_ = torch.cat([truth_i, ti.to(torch.int64) + cs], 1)
        order = torch.argsort(cs_, dim=1, descending=True)[:, : GRAPH_K + 1]
        truth_s = torch.gather(cs_, 1, order); truth_i = torch.gather(ci_, 1, order)
        del sims, ts, ti, cs_, ci_, order
    ti_np = truth_i.cpu().numpy()
    recalls = []
    for n, row in enumerate(probe):
        t = [int(i) for i in ti_np[n] if int(i) != int(row)][:GRAPH_K]
        recalls.append(len(set(t) & set(nbr[row].tolist())) / GRAPH_K)
    recalls = np.asarray(recalls)
    del Xt, qp
    torch.cuda.empty_cache()

    # ---- fuzzy simplicial set ----
    import umap.umap_ as umap_api
    fuzzy_started = time.monotonic()
    graph, _s, _r = umap_api.fuzzy_simplicial_set(
        X, n_neighbors=GRAPH_K, random_state=np.random.RandomState(42),
        metric="cosine", knn_indices=nbr, knn_dists=dist,
    )
    coo = graph.tocoo()
    src = np.asarray(coo.row, dtype=np.int32)
    dst = np.asarray(coo.col, dtype=np.int32)
    wts = np.asarray(coo.data, dtype=np.float32)
    fuzzy_s = time.monotonic() - fuzzy_started
    if not np.isfinite(wts).all() or wts.min() <= 0 or wts.max() > 1:
        raise Round0216Error("fuzzy weights are invalid")
    deg = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((deg == 0).sum()),
        "min": int(deg.min()), "median": float(np.median(deg)),
        "mean": float(deg.mean()), "max": int(deg.max()),
    }
    checks = validate_graph(
        degrees=degrees,
        recall={"mean_recall_at_k": float(recalls.mean()),
                "p10_recall_at_k": float(np.percentile(recalls, 10))},
        edges=len(src),
    )
    graph_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy.npz"), immutable=True, compressed=False,
        sources=src, targets=dst, weights=wts,
        n_nodes=np.asarray(ROWS, dtype=np.int64), k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    receipt = prompt_contract.seal({
        "schema": GRAPH_SCHEMA, "substrate_schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID, "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY, "capabilities": [CAPABILITY],
        "rows": ROWS, "dimension": DIMENSION, "k": GRAPH_K,
        "composition": composition,
        "sources": sources,
        "loading_contract": {"raw_format": RAW_FORMAT, "row_policy": ROW_POLICY,
                             "trailing_fragment_policy": TRAILING_FRAGMENT_POLICY},
        "selection": {"seed": SELECTION_SEED,
                      "law": "per-corpus uniform without replacement over all complete rows, ascending"},
        "substrate": expected_input_signature(sub_path),
        "provenance": expected_input_signature(prov_path),
        "ordered_substrate_sha256": ordered_array_sha256(X),
        "graph": expected_input_signature(graph_path),
        "graph_checks": checks, "degrees": degrees,
        "performance": {"exact_search_s": search_s, "fuzzy_s": fuzzy_s,
                        "total_wall_s": time.monotonic() - started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda"))},
        "training_performed": False,
    })
    atomic_write_new_json(os.path.join(output, "substrate-graph.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "assemble_2m_substrate_and_graph":
        raise Round0216Error("R0216 authorizes only the 2M substrate + exact graph")
    run_assemble_and_graph(active, job)


__all__ = ["run_job"]

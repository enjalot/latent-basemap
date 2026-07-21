"""
Edge-list training path for ParametricUMAP.

Consumes the ``.npz`` edge lists produced by ``build_150m_index_modal.py`` /
``build_*_index_modal.py`` (int32 ``sources`` / ``targets``, float32 ``weights``,
scalars ``n_nodes`` / ``k``) and streams balanced positive/negative batches with
on-the-fly negative sampling — the scale path that unblocks training above
~300k rows and the 150M asset.

Design notes
------------
* Positive edges are read directly from the (optionally memmapped) int arrays;
  they are never converted into Python objects en masse. Only the per-batch
  ``(src, dst)`` tuples are materialised, matching the existing
  ``OnTheFlyBalancedIterator`` contract expected by ``DataPrefetcher``.
* Negatives are sampled per batch with ``rng.randint`` over the training node
  range. Self-pairs are always rejected. Rejecting actual graph neighbours
  (positive edges) is supported via an optional ``edge_set`` but is off by
  default because the rejection set costs memory at 150M scale.
* ``X`` may be a lazy ``np.memmap`` / ``MemmapArrayConcatenator``; the loader
  only ever indexes it per batch, so nothing >=2 GB is materialised.
"""

import logging
import numpy as np
import torch


def load_edge_arrays(path, load_weights=True):
    """Load the ``.npz`` edge list.

    Returns ``(sources, targets, weights_or_None, n_nodes)``. The arrays are
    read from the ``NpzFile`` lazily (per-array); ``savez_compressed`` output
    cannot be true-memmapped, so each accessed array is decompressed into RAM
    once (~8.8 GB per 2.2B-edge int32 array — within the 123 GB budget). When
    ``load_weights`` is False (binary-target mode) the weights array is skipped
    entirely to save memory.
    """
    npz = np.load(path, mmap_mode="r")
    sources = npz["sources"]
    targets = npz["targets"]
    weights = npz["weights"] if load_weights and "weights" in npz.files else None
    if "n_nodes" in npz.files:
        n_nodes = int(np.asarray(npz["n_nodes"]))
    else:
        n_nodes = int(max(int(sources.max()), int(targets.max())) + 1)
    return sources, targets, weights, n_nodes


def build_edge_key_set(sources, targets, n_nodes):
    """Build a Python ``set`` of encoded positive-edge keys for neighbour
    rejection. Costs ~1 entry per directed edge — only feasible at small/medium
    scale, hence gated behind an explicit flag by the caller.
    """
    keys = sources.astype(np.int64) * np.int64(n_nodes) + targets.astype(np.int64)
    return set(keys.tolist())


class LazyArrayDataset:
    """Minimal dataset wrapper that indexes ``X`` lazily and returns CPU torch
    tensors. Keeps memmap-backed inputs off the device until ``DataPrefetcher``
    moves each batch, so nothing large is ever materialised.
    """

    def __init__(self, X):
        self.X = X
        self._n = int(len(X))

    def __len__(self):
        return self._n

    def to(self, device):  # no-op: data stays lazy on CPU
        return self

    def __getitem__(self, idx):
        arr = np.asarray(self.X[idx], dtype=np.float32)
        return torch.from_numpy(arr)


class DeviceArrayDataset:
    """Feature matrix held resident on the training device as a single tensor.

    Used by the GPU-resident fast path: ``X`` is uploaded **once** (typically as
    fp16 on CUDA to halve VRAM; fp32 on CPU where fp16 matmul is slow) and every
    per-batch gather is a device-side ``index_select`` -- zero host-to-device
    transfers per step. Indexing returns fp32 rows so the downstream model /
    correlation loss see the same dtype as the legacy path (the fp16 storage is
    the only precision change, and it is intentional per the scale plan).
    """

    def __init__(self, X, device, storage_dtype=None, upload_chunk=1_000_000):
        if storage_dtype is None:
            storage_dtype = torch.float16 if "cuda" in str(device) else torch.float32
        if torch.is_tensor(X):
            self.tensor = X.to(device=device, dtype=storage_dtype).contiguous()
        else:
            # Upload row-chunks so a memmap-backed X never materialises a full
            # host fp32 copy (respects the >=2 GB no-materialise rule); only one
            # ``upload_chunk``-row slice is transiently held on the host.
            n = int(X.shape[0]); d = int(X.shape[1])
            self.tensor = torch.empty((n, d), dtype=storage_dtype, device=device)
            for i in range(0, n, upload_chunk):
                end = min(i + upload_chunk, n)
                chunk = np.asarray(X[i:end], dtype=np.float32)
                self.tensor[i:end].copy_(torch.from_numpy(chunk).to(storage_dtype))
        self.device = device
        self.storage_dtype = storage_dtype
        self._n = int(self.tensor.shape[0])

    def __len__(self):
        return self._n

    def to(self, device):  # already resident; keep the contract
        if str(device) != str(self.device):
            self.tensor = self.tensor.to(device)
            self.device = device
        return self

    def index_select(self, idx):
        """Gather rows for a device LongTensor ``idx`` as fp32 device rows."""
        return self.tensor.index_select(0, idx).float()

    def __getitem__(self, idx):
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(np.asarray(idx), dtype=torch.long,
                                  device=self.device)
        else:
            idx = idx.to(device=self.device, dtype=torch.long)
        return self.index_select(idx)


class DeviceEdgeSampler:
    """Fully device-resident balanced edge sampler (GPU-resident fast path).

    Drop-in replacement for ``EdgeListBalancedIterator`` + ``DataPrefetcher``:
    each ``__next__`` yields ``(src_feats, dst_feats, targets)`` already on the
    training device, gathered via ``index_select`` from a resident feature
    matrix. No Python list-of-tuples round-trip, no per-batch host-to-device
    feature copy, no rejection-loop sync.

    Semantics vs the legacy path (documented distribution-level equivalence):

    * Positive edges: same directed-edge universe, normally streamed in a
      per-epoch random permutation (``torch.randperm``).  Large graphs may draw
      bounded batches with replacement instead.  The explicit
      ``uniform_with_replacement`` mode makes that bounded uniform distribution
      independent of the process threshold (the authenticated Round-0014 path).
      In-batch shuffling is dropped because the BCE + correlation losses are
      permutation-invariant over the batch, so it never affected the gradient.
    * Negatives: ``neg_src`` uniform over ``[0, n_nodes)``; ``neg_dst`` drawn as
      ``(neg_src + offset) mod n_nodes`` with ``offset`` uniform over
      ``[1, n_nodes)``. This is *exactly* the conditional distribution of
      independent-uniform sampling with self-pair rejection (dst uniform over the
      ``n_nodes-1`` non-self nodes), but needs no rejection loop / sync.
    * RNG stream differs from the numpy path (torch generator, per-seed
      deterministic). Distribution is equivalent; bitwise values are not.
    * ``edge_set`` (graph-neighbour rejection) is **not** supported here -- the
      caller must fall back to the legacy path when ``reject_neighbors`` is set.
    """

    def __init__(self, dataset, sources, targets, weights, n_nodes,
                 pos_ratio=0.2, batch_size=4096, shuffle=True,
                 random_state=0, positive_target_mode="binary",
                 weighted_edge_sampling=False,
                 uniform_with_replacement=False,
                 device="cpu"):
        self.dataset = dataset          # DeviceArrayDataset
        self.device = device
        self.n_nodes = int(n_nodes)
        self.pos_ratio = pos_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.positive_target_mode = positive_target_mode
        self.n_pos = int(len(sources))
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = batch_size - self.num_pos

        if positive_target_mode not in ("binary", "probability"):
            raise ValueError(f"Unknown positive_target_mode: {positive_target_mode}")
        if positive_target_mode == "probability" and weights is None:
            raise ValueError(
                "positive_target_mode='probability' requires edge weights, "
                "but none were loaded."
            )
        if self.n_nodes < 2:
            raise ValueError("DeviceEdgeSampler requires n_nodes >= 2.")

        # Directed edges resident on the device as int32 (half the memory of
        # int64; matters at >=8M edges). The per-batch selection is cast to long
        # in __next__ for the feature gather — result-identical, far leaner.
        self.sources_t = torch.as_tensor(np.asarray(sources), dtype=torch.int32,
                                         device=device)
        self.targets_t = torch.as_tensor(np.asarray(targets), dtype=torch.int32,
                                         device=device)
        if positive_target_mode == "probability":
            self.weights_t = torch.as_tensor(np.asarray(weights),
                                             dtype=torch.float32, device=device)
        else:
            self.weights_t = None

        # Weighted edge sampling (reference-UMAP behaviour): draw positive edges
        # with frequency proportional to their fuzzy membership strength, instead
        # of a uniform once-per-epoch permutation. Strong local edges get
        # attracted many more times; weak/long-range edges rarely. Implemented as
        # inverse-CDF sampling (searchsorted over the normalised cumulative
        # weights) so it is O(n_pos log n_pos)/epoch and has no category cap.
        self.weighted_edge_sampling = bool(weighted_edge_sampling)
        self.uniform_with_replacement = bool(uniform_with_replacement)
        if self.weighted_edge_sampling and self.uniform_with_replacement:
            raise ValueError(
                "uniform_with_replacement cannot be combined with "
                "weighted_edge_sampling")
        self.sample_cdf = None
        if self.weighted_edge_sampling:
            if weights is None:
                raise ValueError(
                    "weighted_edge_sampling=True requires edge weights, "
                    "but none were loaded."
                )
            w = torch.as_tensor(np.asarray(weights), dtype=torch.float64,
                                 device=device)
            # S0: fail closed on degenerate weights rather than collapsing every
            # draw onto one edge. Non-finite or negative weights are invalid; a
            # non-positive total is unsamplable. Constant positive weights are
            # explicitly uniform-equivalent (linear CDF) — allowed, logged.
            if not bool(torch.isfinite(w).all()):
                raise ValueError("weighted_edge_sampling: non-finite edge weights (S0).")
            if bool((w < 0).any()):
                raise ValueError("weighted_edge_sampling: negative edge weights (S0).")
            total = float(w.sum())
            if not (total > 0):
                raise ValueError(f"weighted_edge_sampling: non-positive weight total {total} — "
                                 f"unsamplable; refuse rather than collapse all draws (S0).")
            wmin, wmax = float(w.min()), float(w.max())
            if wmax - wmin <= 1e-12 * max(1.0, abs(wmax)):
                logging.info("weighted_edge_sampling: constant weights → uniform-equivalent (S0).")
            cdf = torch.cumsum(w, dim=0)
            cdf = cdf / total
            self.sample_cdf = cdf  # float64, monotonic in (0, 1]

        # Persistent generator: state advances across batches AND epochs, so
        # negatives + permutations differ every epoch (matches the numpy path's
        # "rng advances across epochs" behaviour without an explicit resample).
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(random_state))
        self.perm = None
        self.pos_idx = 0
        self.batch_no = 0
        # Per-batch sampling: for very large edge lists, building a full-epoch
        # permutation is prohibitive — torch.randperm(n_pos) needs ~3x the array
        # in transient sort workspace (e.g. ~14 GB at 605M edges), which OOMs the
        # fast path. Above the threshold, draw each batch's indices on the fly
        # (with replacement) — O(batch) memory, statistically equivalent for SGD.
        import os
        self._per_batch = self.uniform_with_replacement or self.n_pos > int(
            os.environ.get("PER_BATCH_EDGE_THRESHOLD", 400_000_000))

    def __len__(self):
        return int(np.ceil(self.n_pos / self.num_pos))

    def _draw_idx(self, m):
        """Draw m positive-edge indices (per-batch mode)."""
        if self.weighted_edge_sampling:
            u = torch.rand(m, generator=self.gen, device=self.device,
                           dtype=torch.float64)
            return torch.searchsorted(self.sample_cdf, u).clamp_(max=self.n_pos - 1)
        return torch.randint(0, self.n_pos, (m,), generator=self.gen,
                             device=self.device)

    def __iter__(self):
        self.pos_idx = 0
        self.batch_no = 0
        if self._per_batch:
            self.perm = None            # sampled per batch in __next__
        elif self.weighted_edge_sampling:
            # Inverse-CDF draw of n_pos edges ∝ membership strength (with
            # replacement). Keeps the epoch length identical (n_pos positive
            # slots) but concentrates attraction on strong local edges.
            u = torch.rand(self.n_pos, generator=self.gen, device=self.device,
                           dtype=torch.float64)
            self.perm = torch.searchsorted(self.sample_cdf, u).clamp_(
                max=self.n_pos - 1)
        elif self.shuffle:
            self.perm = torch.randperm(self.n_pos, generator=self.gen,
                                       device=self.device)
        else:
            self.perm = torch.arange(self.n_pos, device=self.device)
        return self

    def _sample_negatives(self, n):
        neg_src = torch.randint(0, self.n_nodes, (n,), generator=self.gen,
                                device=self.device)
        # offset in [1, n_nodes-1] -> dst != src, uniform over non-self nodes.
        offset = torch.randint(1, self.n_nodes, (n,), generator=self.gen,
                               device=self.device)
        neg_dst = (neg_src + offset) % self.n_nodes
        return neg_src, neg_dst

    def __next__(self):
        if self._per_batch:
            if self.batch_no >= len(self):
                raise StopIteration
            self.batch_no += 1
            idx = self._draw_idx(self.num_pos)
        else:
            if self.perm is None:
                iter(self)
            if self.pos_idx >= self.n_pos:
                raise StopIteration
            end = min(self.pos_idx + self.num_pos, self.n_pos)
            idx = self.perm[self.pos_idx:end]
            self.pos_idx += self.num_pos

        p_src = self.sources_t.index_select(0, idx).long()
        p_dst = self.targets_t.index_select(0, idx).long()
        if self.positive_target_mode == "binary":
            p_labels = torch.ones(p_src.shape[0], dtype=torch.float32,
                                  device=self.device)
        else:
            p_labels = self.weights_t.index_select(0, idx)

        neg_src, neg_dst = self._sample_negatives(self.num_neg)
        neg_labels = torch.zeros(self.num_neg, dtype=torch.float32,
                                 device=self.device)

        all_src = torch.cat([p_src, neg_src])
        all_dst = torch.cat([p_dst, neg_dst])
        targets = torch.cat([p_labels, neg_labels])

        src_feats = self.dataset.index_select(all_src)
        dst_feats = self.dataset.index_select(all_dst)
        return src_feats, dst_feats, targets


class EdgeListBalancedIterator:
    """Balanced batch iterator over a precomputed edge list with on-the-fly
    negatives.

    Yields ``(edge_batch, labels)`` where ``edge_batch`` is a list of
    ``(src, dst)`` int tuples and ``labels`` is a float32 numpy array — the same
    contract as ``BalancedEdgeBatchIterator`` / ``OnTheFlyBalancedIterator``.

    Parameters
    ----------
    sources, targets : np.ndarray (int)
        Directed positive edges. May be memmap-backed.
    weights : np.ndarray (float32) or None
        Per-edge P_sym weight. Ignored (and may be None) when
        ``positive_target_mode == "binary"``.
    n_nodes : int
        Number of training rows; negatives are drawn from ``[0, n_nodes)``.
    edge_set : set[int] or None
        Optional set of encoded positive-edge keys (``src*n_nodes+dst``) for
        neighbour rejection. When None only self-pairs are rejected.
    """

    def __init__(self, sources, targets, weights, n_nodes,
                 pos_ratio=0.2, batch_size=4096, shuffle=True,
                 random_state=0, positive_target_mode="binary",
                 edge_set=None):
        self.sources = sources
        self.targets = targets
        self.weights = weights
        self.n_pos = int(len(sources))
        self.n_nodes = int(n_nodes)
        self.pos_ratio = pos_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.positive_target_mode = positive_target_mode
        self.edge_set = edge_set
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = batch_size - self.num_pos
        self.perm = None
        self.pos_idx = 0

        if positive_target_mode == "probability" and weights is None:
            raise ValueError(
                "positive_target_mode='probability' requires edge weights, "
                "but none were loaded."
            )
        if positive_target_mode not in ("binary", "probability"):
            raise ValueError(
                f"Unknown positive_target_mode: {positive_target_mode}"
            )

    def __iter__(self):
        # Fresh shuffle each epoch. rng advances across epochs so negatives
        # differ too (no explicit resample needed).
        if self.shuffle:
            self.perm = self.rng.permutation(self.n_pos)
        else:
            self.perm = np.arange(self.n_pos)
        self.pos_idx = 0
        return self

    def __len__(self):
        return int(np.ceil(self.n_pos / self.num_pos))

    def _neg_invalid(self, neg_src, neg_dst):
        invalid = neg_src == neg_dst
        if self.edge_set is not None:
            keys = neg_src.astype(np.int64) * np.int64(self.n_nodes) + neg_dst.astype(np.int64)
            adj = np.fromiter((int(k) in self.edge_set for k in keys),
                              dtype=bool, count=len(keys))
            invalid = invalid | adj
        return invalid

    def _sample_negatives(self, n):
        neg_src = self.rng.randint(0, self.n_nodes, size=n).astype(np.int64)
        neg_dst = self.rng.randint(0, self.n_nodes, size=n).astype(np.int64)
        invalid = self._neg_invalid(neg_src, neg_dst)
        # Resample only the rejected candidates. Bounded retries — random
        # collisions are O(k/n_nodes), so this converges immediately in practice.
        tries = 0
        while invalid.any() and tries < 100:
            bad = np.flatnonzero(invalid)
            neg_src[bad] = self.rng.randint(0, self.n_nodes, size=len(bad))
            neg_dst[bad] = self.rng.randint(0, self.n_nodes, size=len(bad))
            invalid = self._neg_invalid(neg_src, neg_dst)
            tries += 1
        return neg_src, neg_dst

    def __next__(self):
        if self.perm is None:
            iter(self)
        if self.pos_idx >= self.n_pos:
            raise StopIteration

        end = min(self.pos_idx + self.num_pos, self.n_pos)
        idx = self.perm[self.pos_idx:end]
        self.pos_idx += self.num_pos

        p_src = np.asarray(self.sources[idx], dtype=np.int64)
        p_dst = np.asarray(self.targets[idx], dtype=np.int64)
        if self.positive_target_mode == "binary":
            p_labels = np.ones(len(p_src), dtype=np.float32)
        else:
            p_labels = np.asarray(self.weights[idx], dtype=np.float32)

        neg_src, neg_dst = self._sample_negatives(self.num_neg)

        all_src = np.concatenate([p_src, neg_src])
        all_dst = np.concatenate([p_dst, neg_dst])
        labels = np.concatenate([p_labels, np.zeros(len(neg_src), dtype=np.float32)])

        edge_batch = list(zip(all_src.tolist(), all_dst.tolist()))
        if self.shuffle:
            perm = self.rng.permutation(len(edge_batch))
            edge_batch = [edge_batch[i] for i in perm]
            labels = labels[perm]
        return edge_batch, labels


class HostStreamEdgeSampler:
    """Hybrid edge sampler (plan-100m B1): X resident on GPU, positive edges +
    weighted-sampling CDF resident on the HOST. A background daemon thread
    pre-draws positive-edge batches (weighted searchsorted or uniform) and
    pushes pinned (src_node, dst_node) index tensors into a bounded queue;
    ``__next__`` copies them H2D (non_blocking), draws negatives on-device, and
    gathers features from the resident X via ``DeviceArrayDataset.index_select``.

    Motivation: at k=50 fuzzy the edge arrays + weighted CDF (~23 GB at 1.157B
    edges) don't fit GPU alongside X, but X alone does. Keeping edges/CDF on
    host (plentiful RAM) and prefetching batches keeps the GPU fed without the
    fully-host legacy path's input-bound stall. Same on-device output contract
    as ``DeviceEdgeSampler`` -> plugs into ``_fast_device_path``.
    """

    def __init__(self, dataset, sources, targets, weights, n_nodes,
                 pos_ratio=0.2, batch_size=4096, random_state=0,
                 positive_target_mode="binary", weighted_edge_sampling=False,
                 device="cuda", n_workers=2, queue_size=8,
                 retained_node_rows=None):
        import threading, queue
        self.dataset = dataset            # DeviceArrayDataset (X resident)
        self.device = device
        self.n_nodes = int(n_nodes)
        self.batch_size = batch_size
        self.positive_target_mode = positive_target_mode
        self.weighted_edge_sampling = weighted_edge_sampling
        self.source_n_pos = int(len(sources))
        self.n_pos = self.source_n_pos
        self.num_pos = max(1, int(batch_size * pos_ratio))
        self.num_neg = batch_size - self.num_pos
        if self.n_nodes < 2:
            raise ValueError("HostStreamEdgeSampler requires n_nodes >= 2.")

        # Host-resident edges (int32) + f64 CDF (precision-safe on host RAM).
        self._src_h = np.ascontiguousarray(np.asarray(sources), dtype=np.int32)
        self._dst_h = np.ascontiguousarray(np.asarray(targets), dtype=np.int32)
        self._retained_node_mask_h = None
        self._retained_node_rows_t = None
        self.excluded_positive_edges = 0
        if retained_node_rows is not None:
            previous = None
            for start in range(0, self.source_n_pos, 5_000_000):
                block = self._src_h[start:start + 5_000_000]
                if (len(block) and previous is not None and int(block[0]) < previous) or (
                        len(block) > 1 and np.any(block[1:] < block[:-1])):
                    raise ValueError(
                        "retained-node sampling requires a source-sorted edge list")
                if len(block):
                    previous = int(block[-1])
            retained = np.asarray(retained_node_rows, dtype=np.int64)
            if (retained.ndim != 1 or len(retained) < 2
                    or not np.array_equal(retained, np.unique(retained))
                    or retained[0] < 0 or retained[-1] >= self.n_nodes):
                raise ValueError(
                    "retained_node_rows must be sorted, unique, in-range, and contain "
                    "at least two rows")
            allowed = np.zeros(self.n_nodes, dtype=bool)
            allowed[retained] = True
            excluded = np.flatnonzero(~allowed).astype(np.int64, copy=False)
            left = np.searchsorted(self._src_h, excluded, side="left")
            right = np.searchsorted(self._src_h, excluded, side="right")
            self.excluded_positive_edges = int(np.sum(right - left, dtype=np.int64))
            self.n_pos = self.source_n_pos - self.excluded_positive_edges
            if self.n_pos <= 0:
                raise ValueError("retained node cap excludes every positive source edge")
            self._excluded_source_ranges = tuple(
                (int(lo), int(hi)) for lo, hi in zip(left, right) if hi > lo)
            self._retained_node_mask_h = allowed
            self._retained_node_rows_t = torch.as_tensor(
                retained, dtype=torch.int32, device=device)
        else:
            self._excluded_source_ranges = ()
        self._cdf_h = None
        if weighted_edge_sampling:
            if weights is None:
                raise ValueError("weighted_edge_sampling=True requires edge weights (S0).")
            # One in-place f64 work array becomes the CDF. The prior
            # np.cumsum(w) + cdf/total sequence transiently held two 5.9 GB
            # arrays at 738M edges.
            w = np.array(weights, dtype=np.float64, copy=True)
            # L0.5: HostStream degenerate-weight guards, matching DeviceEdgeSampler.
            # A non-finite/negative weight or a non-positive total must FAIL CLOSED,
            # not silently collapse every draw. Constant weights are uniform-equivalent.
            if not np.all(np.isfinite(w)):
                raise ValueError("weighted_edge_sampling: non-finite edge weights (S0).")
            if np.any(w < 0):
                raise ValueError("weighted_edge_sampling: negative edge weights (S0).")
            full_wmin, full_wmax = float(w.min()), float(w.max())
            for lo, hi in self._excluded_source_ranges:
                w[lo:hi] = 0.0
            total = float(w.sum())
            if not (total > 0):
                raise ValueError(f"weighted_edge_sampling: non-positive weight total {total} — "
                                 f"all-zero/degenerate weights; refuse to sample (S0).")
            if full_wmax == full_wmin:
                logging.info("weighted_edge_sampling: constant weights → uniform-equivalent (S0).")
            np.cumsum(w, out=w)
            w /= total
            self._cdf_h = w                           # f64 host, monotonic (0,1]
        # binary targets use a constant ones vector; probability mode gathers weights
        self._w_h = (np.asarray(weights, dtype=np.float32)
                     if positive_target_mode == "probability" else None)

        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._worker_err = None   # first worker exception, surfaced by __next__
        self._threads = []
        for wi in range(n_workers):
            t = threading.Thread(target=self._produce, args=(random_state + 1 + wi,),
                                 daemon=True)
            t.start()
            self._threads.append(t)
        # device-side RNG for negatives (matches DeviceEdgeSampler semantics)
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(int(random_state))

    def _draw_positive_indices(self, rng, count):
        """Draw exact retained-source edge indices without a materialized index."""
        count = int(count)
        out = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            remaining = count - filled
            if self.weighted_edge_sampling:
                proposal = np.searchsorted(
                    self._cdf_h, rng.random(remaining), side="right")
                np.clip(proposal, 0, self.source_n_pos - 1, out=proposal)
            else:
                # A small oversample keeps the common cap-one case to one pass;
                # rejection is exact uniform conditioning on retained sources.
                keep_fraction = self.n_pos / self.source_n_pos
                proposal_n = max(remaining, int(np.ceil(
                    remaining / keep_fraction * 1.02)))
                proposal = rng.integers(0, self.source_n_pos, size=proposal_n)
            if self._retained_node_mask_h is not None:
                proposal = proposal[
                    self._retained_node_mask_h[self._src_h[proposal]]]
            take = min(remaining, len(proposal))
            if take:
                out[filled:filled + take] = proposal[:take]
                filled += take
        return out

    def _produce(self, seed):
        """Background: draw positive-edge node batches, pin, enqueue. Captures
        any exception into ``self._worker_err`` so ``__next__`` can re-raise it
        instead of hanging on a silently-dead worker pool."""
        import queue as _queue
        rng = np.random.default_rng(seed)
        m = self.num_pos
        try:
            while not self._stop.is_set():
                idx = self._draw_positive_indices(rng, m)
                ps = torch.from_numpy(self._src_h[idx].astype(np.int64))
                pd = torch.from_numpy(self._dst_h[idx].astype(np.int64))
                item = (ps.pin_memory(), pd.pin_memory())
                if self._w_h is not None:
                    pw = torch.from_numpy(self._w_h[idx]).pin_memory()
                    item = item + (pw,)
                # Retry the SAME drawn batch until enqueued (no-discard) or stop.
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=1.0)
                        break
                    except _queue.Full:
                        continue
        except BaseException as e:   # noqa: BLE001 — surface to the main thread
            if self._worker_err is None:
                self._worker_err = e

    def __len__(self):
        return int(np.ceil(self.n_pos / self.num_pos))

    def _sample_negatives(self, n):
        if self._retained_node_rows_t is not None:
            universe = len(self._retained_node_rows_t)
            src_pos = torch.randint(0, universe, (n,), generator=self._gen,
                                    device=self.device)
            offset = torch.randint(1, universe, (n,), generator=self._gen,
                                   device=self.device)
            dst_pos = (src_pos + offset) % universe
            return (
                self._retained_node_rows_t.index_select(0, src_pos).long(),
                self._retained_node_rows_t.index_select(0, dst_pos).long(),
            )
        neg_src = torch.randint(0, self.n_nodes, (n,), generator=self._gen,
                                device=self.device)
        offset = torch.randint(1, self.n_nodes, (n,), generator=self._gen,
                               device=self.device)
        neg_dst = (neg_src + offset) % self.n_nodes
        return neg_src, neg_dst

    def __iter__(self):
        self._batch_no = 0
        return self

    def __next__(self):
        import queue as _queue
        if self._batch_no >= len(self):
            raise StopIteration
        self._batch_no += 1
        # Hang-safe pull: bounded wait + worker-liveness check. Never blocks
        # forever if the producer pool dies mid-run (unattended multi-hour runs).
        while True:
            try:
                item = self._q.get(timeout=60)
                break
            except _queue.Empty:
                if self._worker_err is not None:
                    raise RuntimeError(
                        "HostStreamEdgeSampler producer died") from self._worker_err
                if not any(t.is_alive() for t in self._threads):
                    raise RuntimeError(
                        "HostStreamEdgeSampler: all producer threads dead, no batch in 60s")
                # workers alive but slow — keep waiting
        p_src = item[0].to(self.device, non_blocking=True)
        p_dst = item[1].to(self.device, non_blocking=True)
        if self.positive_target_mode == "binary":
            p_labels = torch.ones(p_src.shape[0], dtype=torch.float32, device=self.device)
        else:
            p_labels = item[2].to(self.device, non_blocking=True)
        neg_src, neg_dst = self._sample_negatives(self.num_neg)
        neg_labels = torch.zeros(self.num_neg, dtype=torch.float32, device=self.device)
        all_src = torch.cat([p_src, neg_src])
        all_dst = torch.cat([p_dst, neg_dst])
        targets = torch.cat([p_labels, neg_labels])
        src_feats = self.dataset.index_select(all_src)
        dst_feats = self.dataset.index_select(all_dst)
        return src_feats, dst_feats, targets

    def close(self):
        """Stop producer threads (call when training ends so daemons don't keep
        drawing-and-discarding batches during scoring/transform)."""
        self._stop.set()
        for t in getattr(self, "_threads", []):
            t.join(timeout=2.0)

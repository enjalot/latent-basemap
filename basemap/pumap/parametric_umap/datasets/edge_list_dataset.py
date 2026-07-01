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

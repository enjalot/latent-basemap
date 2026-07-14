from .models.mlp import MLP, ResidualBottleneckMLP
from .datasets.edge_dataset import EdgeDataset
from .datasets.covariates_datasets import VariableDataset
from .utils.losses import compute_correlation_loss
from .utils.graph import compute_all_p_umap
from .utils.data_prefetcher import DataPrefetcher

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm.auto import tqdm
import pickle
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

class ParametricUMAP:
    def __init__(
        self,
        n_components=2,
        hidden_dim=1024,
        n_layers=3,
        n_neighbors=15,
        a=0.1,
        b=1.0,
        low_dim_kernel="legacy_lp",
        correlation_weight=0.1,
        learning_rate=1e-4,
        n_epochs=10,
        batch_size=32,
        device=None,  # Auto-detect: cuda > mps > cpu
        use_batchnorm=False,
        use_dropout=False,
        clip_grad_norm=1.0,
        clip_grad_value=None,
        pos_ratio=0.5,
        architecture="mlp",
        correlation_distance_transform="raw",
        lr_schedule="plateau",
        warmup_steps=0,
        total_steps_estimate=0,
        use_amp=True,
        positive_target_mode="probability",
        reject_neighbors=False,
        anchored_init="none",
        anchored_init_epochs=2,
        anchored_init_lr=1e-3,
        anchored_init_path="",
        anchor_hold_weight=0.0,
        anchor_hold_fraction=0.05,
        midnear_enabled=False,
        mn_pairs_per_batch=0,
        mn_weight_scale=1.0,
        weighted_edge_sampling=False,
        gpu_resident_data="auto",
        gpu_resident_vram_budget_gb=10.0,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.a = a
        self.b = b
        # Low-D similarity kernel (P0.1). "legacy_lp" = the historically-shipped
        # 1/(1+a·‖Δ‖_{2b}) (an Lp/quasi-norm curve, NOT UMAP). "umap" = the
        # standard 1/(1+a·‖Δ‖²^b). Default legacy_lp so old checkpoints keep
        # their trained semantics; new runs opt into "umap" explicitly.
        if low_dim_kernel not in ("legacy_lp", "umap"):
            raise ValueError(f"low_dim_kernel must be 'legacy_lp' or 'umap', got {low_dim_kernel!r}")
        self.low_dim_kernel = low_dim_kernel
        # P0.8: only allow prefix-filtering a larger graph onto X for a verified
        # literal prefix (never for balanced/sampled matrices).
        self.allow_prefix_edge_filter = False
        self.correlation_weight = correlation_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.pos_ratio = pos_ratio
        self.architecture = architecture
        self.correlation_distance_transform = correlation_distance_transform
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.total_steps_estimate = total_steps_estimate
        self.use_amp = use_amp
        self.positive_target_mode = positive_target_mode
        self.reject_neighbors = reject_neighbors
        self.anchored_init = anchored_init
        self.anchored_init_epochs = anchored_init_epochs
        self.anchored_init_lr = anchored_init_lr
        self.anchored_init_path = anchored_init_path
        self.anchor_hold_weight = anchor_hold_weight
        self.anchor_hold_fraction = anchor_hold_fraction
        self.midnear_enabled = midnear_enabled
        self.mn_pairs_per_batch = mn_pairs_per_batch
        self.mn_weight_scale = mn_weight_scale
        self.weighted_edge_sampling = weighted_edge_sampling
        # GPU-resident fast path (input-pipeline optimisation). "auto" enables it
        # when X fits in VRAM within the budget on CUDA; True forces it on any
        # device (fp16 storage on CUDA, fp32 on CPU); False keeps the legacy path.
        self.gpu_resident_data = gpu_resident_data
        self.gpu_resident_vram_budget_gb = gpu_resident_vram_budget_gb
        self._fast_device_path = False   # set True by _prepare_edge_list_training
        self._X_dev = None               # DeviceArrayDataset when fast path active
        self._max_train_steps = None     # benchmark hook: stop after N global steps
        self._bench_warmup = 0           # benchmark hook: steps to exclude from timing
        self._bench_t0 = 0.0
        self._bench_seconds = None       # measured steady-state loop seconds
        self.model = None
        self.input_dim = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False
        # Populated by anchored initialization; persisted by the runner so that
        # stability analysis can reuse the exact deterministic targets used.
        self.anchor_targets_ = None
        self.anchor_scale_ = None        # RMS-radius scale factor (manifest)
        self._anchor_targets_dev = None  # resident targets for the hold term

    def _init_model(self, input_dim):
        """Initialize the configured parametric model."""
        self.input_dim = input_dim
        if self.architecture == "mlp":
            self.model = MLP(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_components,
                num_layers=self.n_layers,
                use_batchnorm=self.use_batchnorm,
                use_dropout=self.use_dropout
            ).to(self.device)
        elif self.architecture == "residual_bottleneck":
            self.model = ResidualBottleneckMLP(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_components,
                num_layers=self.n_layers,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _low_dim_qs(self, src, dst):
        """Low-D similarity q_ij for an edge batch, per ``self.low_dim_kernel``.

        - ``legacy_lp`` (shipped historically): ``1 / (1 + a·‖Δ‖_{2b})`` where
          ``‖Δ‖_{2b}`` is the p=2b vector norm — an Lp/quasi-norm radial curve,
          NOT the UMAP kernel. Kept so old checkpoints retain their semantics.
        - ``umap`` (standard): ``1 / (1 + a·(‖Δ‖²)^b)`` = ``1/(1+a·r²^b)``.
          At b=1 these differ: legacy 1/(1+a·r) vs umap 1/(1+a·r²).
        """
        delta = src - dst
        if self.low_dim_kernel == "umap":
            r2 = delta.square().sum(dim=1)
            radial = r2.pow(self.b)
        else:  # legacy_lp
            radial = torch.norm(delta, dim=1, p=2 * self.b)
        return torch.pow(1 + self.a * radial, -1), radial

    def _decide_gpu_resident(self, n_train, n_features, n_pos_edges,
                             edge_set, low_memory):
        """Decide whether to use the GPU-resident fast path.

        Returns ``(use_fast, reason)``. ``gpu_resident_data`` may be:
          * ``False`` / ``"false"`` -> always legacy.
          * ``True`` / ``"true"``   -> force resident on any device.
          * ``"auto"`` (default)    -> resident only on CUDA when the resident
            footprint (X fp16 + edge index int64 [+ weights]) fits within
            ``min(gpu_resident_vram_budget_gb, 0.9 * free_vram)``.

        Neighbour-rejection (``edge_set``) is unsupported on the fast path, so it
        forces the legacy sampler.
        """
        mode = str(self.gpu_resident_data).lower()
        if mode in ("false", "0", "off", "none"):
            return False, "gpu_resident_data=false"
        if edge_set is not None:
            return False, "reject_neighbors set -> legacy path"
        is_cuda = "cuda" in str(self.device)
        storage_bytes = 2 if is_cuda else 4  # fp16 on cuda, fp32 elsewhere
        x_bytes = n_train * n_features * storage_bytes
        # DeviceEdgeSampler holds src+dst as int32 pairs (8 B/edge). Weighted
        # sampling adds an f64 CDF (8 B/edge); probability mode adds f32 weights
        # (4 B/edge). The old `2*8` int64 figure was only accidentally right for
        # the binary+weighted case (8+8=16 B/edge).
        edge_bytes = n_pos_edges * 2 * 4      # src + dst int32 resident
        if self.weighted_edge_sampling:
            edge_bytes += n_pos_edges * 8     # f64 sampling CDF
        if self.positive_target_mode == "probability":
            edge_bytes += n_pos_edges * 4     # f32 weights
        need = x_bytes + edge_bytes
        if mode in ("true", "1", "on", "force"):
            return True, f"forced (need ~{need/1e9:.2f} GB on device)"
        if not is_cuda:
            return False, "auto: non-CUDA device -> legacy path"
        budget = float(self.gpu_resident_vram_budget_gb) * 1e9
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            free_bytes = budget
        avail = min(budget, free_bytes * 0.9)  # 10% headroom for model+activations
        if need <= avail:
            return True, (f"auto: fits (need ~{need/1e9:.2f} GB <= "
                          f"avail ~{avail/1e9:.2f} GB)")
        return False, (f"auto: too large (need ~{need/1e9:.2f} GB > "
                       f"avail ~{avail/1e9:.2f} GB) -> legacy path")

    def _prepare_edge_list_training(self, X, edges_path, n_train, low_memory, random_state):
        """Build the (dataset, loader) pair for the precomputed edge-list path.

        Loads the .npz edge list (lazily), filters edges to the training node
        range when the index covers more nodes than X (e.g. a 15M slice of the
        150M index), optionally builds a neighbour-rejection set, and returns a
        memmap-safe dataset plus an on-the-fly balanced iterator.
        """
        from .datasets.edge_list_dataset import (
            EdgeListBalancedIterator, LazyArrayDataset,
            DeviceArrayDataset, DeviceEdgeSampler, HostStreamEdgeSampler,
            load_edge_arrays, build_edge_key_set,
        )
        from .datasets.covariates_datasets import VariableDataset

        load_weights = (self.positive_target_mode == "probability"
                        or self.weighted_edge_sampling)
        sources, targets, weights, n_nodes = load_edge_arrays(
            edges_path, load_weights=load_weights)
        logging.info(
            "Loaded edge list from %s: %d directed edges, n_nodes=%d",
            edges_path, len(sources), n_nodes,
        )

        # P0.8: validate the graph/data pairing. Rejects n_nodes<n_train, ANN
        # sentinel (-1) endpoints, out-of-range ids, and — critically — the
        # prefix-filter of a larger graph onto a balanced/sampled matrix (which
        # silently connects unrelated cross-corpus rows) unless explicitly opted
        # into for a verified literal prefix.
        from ...graph_validation import validate_graph_data_pair
        mask = validate_graph_data_pair(
            sources, targets, n_nodes, n_train,
            allow_prefix_filter=getattr(self, "allow_prefix_edge_filter", False))
        if mask is not None:
            logging.warning(
                "P0.8: prefix-filtering a larger graph (n_nodes=%d > n_train=%d) — "
                "valid ONLY if X is the graph's literal first %d rows.",
                n_nodes, n_train, n_train)
            sources = np.ascontiguousarray(np.asarray(sources)[mask])
            targets = np.ascontiguousarray(np.asarray(targets)[mask])
            if weights is not None:
                weights = np.ascontiguousarray(np.asarray(weights)[mask])
            logging.info("Kept %d / %d edges after filtering.", len(sources), len(mask))

        n_pos_edges = int(len(sources))

        edge_set = None
        if self.reject_neighbors:
            logging.info("Building positive-edge rejection set (reject_neighbors=True)...")
            edge_set = build_edge_key_set(np.asarray(sources), np.asarray(targets), n_train)

        # GPU-resident fast path: upload X once (fp16 on CUDA) and do all
        # gathers + negative sampling on-device. See _decide_gpu_resident.
        n_features = int(X.shape[1])
        use_fast, reason = self._decide_gpu_resident(
            n_train, n_features, n_pos_edges, edge_set, low_memory)
        if use_fast:
            logging.info("Edge-list mode: GPU-resident fast path (%s).", reason)
            ddataset = DeviceArrayDataset(X, self.device)
            self._X_dev = ddataset
            self._fast_device_path = True
            loader = DeviceEdgeSampler(
                ddataset, sources, targets, weights, n_nodes=n_train,
                pos_ratio=self.pos_ratio, batch_size=self.batch_size,
                shuffle=True, random_state=random_state,
                positive_target_mode=self.positive_target_mode,
                weighted_edge_sampling=self.weighted_edge_sampling,
                device=self.device,
            )
            return ddataset, loader, n_pos_edges

        # Hybrid path (B1): X alone fits resident but X+edges+CDF don't (the
        # k=50 fuzzy edge wall). Keep X on GPU; stream positive edges + weighted
        # CDF from host via background workers with pinned prefetch. Negatives
        # + gathers stay on-device. Enabled when gpu_resident_data is not False,
        # device is CUDA, edge_set is unused, and X fits the resident budget.
        is_cuda = "cuda" in str(self.device)
        want_resident = str(self.gpu_resident_data).lower() not in ("false", "0", "off", "none")
        if want_resident and is_cuda and edge_set is None:
            x_bytes = n_train * n_features * 2  # fp16 on cuda
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
            except Exception:
                free_bytes = float(self.gpu_resident_vram_budget_gb) * 1e9
            x_budget = min(float(self.gpu_resident_vram_budget_gb) * 1e9, free_bytes * 0.9)
            if x_bytes <= x_budget:
                logging.info("Edge-list mode: HYBRID (X resident %.1f GB + host-streamed "
                             "edges/CDF; %s).", x_bytes / 1e9, reason)
                ddataset = DeviceArrayDataset(X, self.device)
                self._X_dev = ddataset
                self._fast_device_path = True
                loader = HostStreamEdgeSampler(
                    ddataset, sources, targets, weights, n_nodes=n_train,
                    pos_ratio=self.pos_ratio, batch_size=self.batch_size,
                    random_state=random_state,
                    positive_target_mode=self.positive_target_mode,
                    weighted_edge_sampling=self.weighted_edge_sampling,
                    device=self.device,
                )
                return ddataset, loader, n_pos_edges

        logging.info("Edge-list mode: legacy sampler path (%s).", reason)
        loader = EdgeListBalancedIterator(
            sources, targets, weights, n_nodes=n_train,
            pos_ratio=self.pos_ratio, batch_size=self.batch_size,
            shuffle=True, random_state=random_state,
            positive_target_mode=self.positive_target_mode,
            edge_set=edge_set,
        )

        # In-memory ndarray on the GPU is fastest when it fits; otherwise keep
        # indexing lazy so memmap-backed inputs never materialise.
        if isinstance(X, np.ndarray) and not low_memory:
            logging.info("Edge-list mode: feature matrix resident on %s.", self.device)
            dataset = VariableDataset(X).to(self.device)
        else:
            logging.info("Edge-list mode: lazy per-batch feature indexing (memmap-safe).")
            dataset = LazyArrayDataset(X)

        return dataset, loader, n_pos_edges

    # ── Anchored initialization (plan §4.2 / §6 Phase 1) ────────────────────
    def _compute_pca_anchor_targets(self, X, n_train, subsample=100_000,
                                    target_rms_radius=5.0, random_state=0,
                                    batch_size=50_000):
        """Deterministic PCA-2D target coordinates for anchored initialization.

        Fit a 2-component PCA on a ``<=subsample`` row sample of ``X`` (which may
        be a lazy memmap / MemmapArrayConcatenator — only the sampled rows and
        per-batch slices are ever materialised), then project every training row.

        Determinism: PCA components are sign-fixed so that the largest-|loading|
        entry of each component is positive, removing PCA's arbitrary sign
        freedom. The resulting cloud is isotropically scaled so its RMS radius
        (sqrt(mean(x^2 + y^2))) equals ``target_rms_radius`` (~5, matching
        typical UMAP layout extents).

        Returns a float32 array of shape ``(n_train, n_components)``.
        """
        from sklearn.decomposition import PCA

        rng = np.random.RandomState(random_state)
        n_sub = min(subsample, n_train)
        if n_sub < n_train:
            sub_idx = np.sort(rng.choice(n_train, n_sub, replace=False))
            X_sub = np.asarray(X[sub_idx], dtype=np.float32)
        else:
            X_sub = np.asarray(X[0:n_train], dtype=np.float32)

        pca = PCA(n_components=self.n_components, random_state=random_state)
        pca.fit(X_sub)
        mean = pca.mean_.astype(np.float32)
        components = pca.components_.astype(np.float32)  # (n_components, n_features)

        # Fix the sign of each component deterministically.
        for c in range(components.shape[0]):
            j = int(np.argmax(np.abs(components[c])))
            if components[c, j] < 0:
                components[c] = -components[c]

        # Project all rows in batches (memmap-safe).
        targets = np.empty((n_train, self.n_components), dtype=np.float32)
        comp_t = components.T  # (n_features, n_components)
        for i in range(0, n_train, batch_size):
            end = min(i + batch_size, n_train)
            rows = np.asarray(X[i:end], dtype=np.float32)
            targets[i:end] = (rows - mean) @ comp_t

        # Isotropic scale to the requested RMS radius.
        rms = float(np.sqrt(np.mean(np.sum(targets.astype(np.float64) ** 2, axis=1))))
        scale = (target_rms_radius / rms) if rms > 0 else 1.0
        targets *= np.float32(scale)
        self.anchor_scale_ = float(scale)
        return targets

    def _compute_anchor_targets(self, X, n_train, random_state=0):
        """Dispatch to the configured anchored-init target source (pca / file)."""
        if self.anchored_init == "pca":
            return self._compute_pca_anchor_targets(
                X, n_train, random_state=random_state)
        if self.anchored_init == "file":
            return self._load_file_anchor_targets(
                n_train, random_state=random_state)
        raise ValueError(f"Unknown anchored_init: {self.anchored_init}")

    def _load_file_anchor_targets(self, n_train, target_rms_radius=5.0,
                                  random_state=0):
        """Load reference-atlas teacher coordinates for distillation (plan §4.3).

        Reads ``anchored_init_path`` (a coords parquet with columns ``x, y`` and
        optionally ``ls_index``) into a ``(n_train, n_components)`` float32 target
        array. Rows align by ``ls_index`` when that column is present (training
        row ``i`` takes the teacher row whose ``ls_index == i``), otherwise
        positionally (the first ``n_train`` rows in file order). Targets are
        isotropically scaled to RMS radius ``target_rms_radius`` (~5), matching
        the PCA path; the scale factor is recorded on ``self.anchor_scale_`` so
        the runner can persist it in the run manifest.
        """
        import pyarrow.parquet as pq

        path = os.path.expanduser(str(self.anchored_init_path or ""))
        if not path:
            raise ValueError(
                "anchored_init='file' requires train.anchored_init_path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"anchor target parquet not found: {path}")

        df = pq.read_table(path).to_pandas()
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError(
                f"anchor target parquet {path} must have 'x' and 'y' columns; "
                f"got {list(df.columns)}")
        xy = df[["x", "y"]].to_numpy().astype(np.float32)

        if "ls_index" in df.columns:
            ls = df["ls_index"].to_numpy().astype(np.int64)
            targets = np.full((n_train, self.n_components), np.nan, dtype=np.float32)
            in_range = (ls >= 0) & (ls < n_train)
            targets[ls[in_range]] = xy[in_range]
            missing = int(np.isnan(targets[:, 0]).sum())
            if missing:
                raise ValueError(
                    f"anchor target parquet {path} covers only "
                    f"{n_train - missing}/{n_train} training rows by ls_index "
                    f"(missing {missing}); cannot align teacher layout.")
        else:
            if xy.shape[0] < n_train:
                raise ValueError(
                    f"anchor target parquet {path} has {xy.shape[0]} rows < "
                    f"n_train {n_train}; cannot align positionally.")
            targets = np.ascontiguousarray(xy[:n_train])

        rms = float(np.sqrt(np.mean(np.sum(targets.astype(np.float64) ** 2, axis=1))))
        scale = (target_rms_radius / rms) if rms > 0 else 1.0
        targets = (targets * np.float32(scale)).astype(np.float32)
        self.anchor_scale_ = float(scale)
        logging.info(
            "File anchor targets from %s: %d rows, source RMS %.3f -> scale "
            "%.4f (target RMS %.1f).", path, n_train, rms, scale, target_rms_radius)
        return targets

    def _anchored_pretrain(self, dataset, targets, n_train, random_state):
        """Pretrain the encoder to regress deterministic 2D anchor targets (MSE).

        Plain minibatch MSE over the training rows for ``anchored_init_epochs``
        epochs with a fresh AdamW optimizer (``anchored_init_lr``). Runs in fp32
        (no AMP/scaler) — it is only a couple of epochs and numerically simple.
        The optimizer/scheduler for the main UMAP phase are created afterwards,
        so their state is reset relative to this pretraining phase.
        """
        rng = np.random.RandomState(random_state + 7919)
        opt = AdamW(self.model.parameters(), lr=self.anchored_init_lr)
        mse = nn.MSELoss()
        targets_t = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.model.train()

        n_batches = int(np.ceil(n_train / self.batch_size))
        for epoch in range(self.anchored_init_epochs):
            perm = rng.permutation(n_train)
            running = 0.0
            for bi in range(n_batches):
                idx = perm[bi * self.batch_size:(bi + 1) * self.batch_size]
                if len(idx) == 0:
                    continue
                feats = self._gather_feature_rows(dataset, idx)
                tgt = targets_t[idx].to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = self.model(feats)
                loss = mse(pred, tgt)
                loss.backward()
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.clip_grad_norm)
                opt.step()
                running += loss.item()
            logging.info(
                "Anchored pretrain epoch %d/%d — MSE: %.4f",
                epoch + 1, self.anchored_init_epochs, running / max(n_batches, 1),
            )

    def _gather_feature_rows(self, dataset, idx):
        """Gather feature rows for ``idx`` from ``dataset`` as a device tensor.

        Works for both the device-resident ``VariableDataset`` (indexing returns
        a tensor already on ``self.device``) and the lazy ``LazyArrayDataset``
        (returns a CPU tensor gathered per-batch from a memmap). Only the
        requested rows are ever materialised.
        """
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        vals = dataset[idx_list]
        if not torch.is_tensor(vals):
            vals = torch.as_tensor(np.asarray(vals, dtype=np.float32))
        return vals.to(self.device, non_blocking=True)

    def _sample_midnear_features(self, dataset, n_anchors, n_train, rng,
                                 n_candidates=6, rank=1):
        """Sample mid-near anchor/partner feature pairs (PaCMAP-style).

        For each of ``n_anchors`` random anchors, draw ``n_candidates`` random
        rows, compute high-D distances anchor→candidates on the training device,
        and keep the ``rank``-th nearest (rank=1 → 2nd nearest) as the mid-near
        partner. Returns ``(anchor_feats, partner_feats)`` device tensors of
        shape ``(n_anchors, n_features)`` each. Fully vectorized; only the
        sampled rows are gathered from ``X`` (memmap-safe).
        """
        anchors = rng.randint(0, n_train, size=n_anchors)
        cands = rng.randint(0, n_train, size=n_anchors * n_candidates)
        anchor_feats = self._gather_feature_rows(dataset, anchors)          # (m, D)
        cand_feats = self._gather_feature_rows(dataset, cands)              # (m*C, D)
        cand_feats = cand_feats.view(n_anchors, n_candidates, -1)           # (m, C, D)

        with torch.no_grad():
            d = torch.linalg.vector_norm(
                anchor_feats.unsqueeze(1).float() - cand_feats.float(), dim=2
            )  # (m, C)
            k = min(rank + 1, n_candidates)
            # topk smallest: partner is the `rank`-th nearest candidate.
            near_idx = torch.topk(d, k=k, dim=1, largest=False).indices[:, rank]
        partner_feats = cand_feats[torch.arange(n_anchors, device=cand_feats.device),
                                   near_idx]  # (m, D)
        return anchor_feats, partner_feats

    def _sample_midnear_features_device(self, n_anchors, n_train, gen,
                                        n_candidates=6, rank=1):
        """Device-resident mid-near sampling (GPU-resident fast path).

        Same PaCMAP mid-near candidate selection as ``_sample_midnear_features``
        but anchors/candidates are drawn with ``torch.randint`` on-device and
        gathered via ``index_select`` from the resident feature matrix. The
        distance ranking stays in fp32 (matching the legacy ``.float()`` path) so
        the mid-near partner selection is distribution-equivalent.
        """
        X_dev = self._X_dev
        anchors = torch.randint(0, n_train, (n_anchors,), generator=gen,
                                device=self.device)
        cands = torch.randint(0, n_train, (n_anchors * n_candidates,),
                              generator=gen, device=self.device)
        anchor_feats = X_dev.index_select(anchors)                 # (m, D) fp32
        cand_feats = X_dev.index_select(cands).view(n_anchors, n_candidates, -1)
        with torch.no_grad():
            d = torch.linalg.vector_norm(
                anchor_feats.unsqueeze(1) - cand_feats, dim=2)     # (m, C)
            k = min(rank + 1, n_candidates)
            near_idx = torch.topk(d, k=k, dim=1, largest=False).indices[:, rank]
        partner_feats = cand_feats[torch.arange(n_anchors, device=self.device),
                                   near_idx]                       # (m, D)
        return anchor_feats, partner_feats

    def fit(self, X, y=None,
            resample_negatives=False,
            n_processes=6,
            low_memory=False,
            random_state=0,
            verbose=True,
            precomputed_p_sym_path=None,
            precomputed_negatives_path=None,
            precomputed_edges_path=None,
            cache_p_sym_path=None,
            cache_negatives_path=None,
            use_wandb=False,
            wandb_project=None,
            wandb_run_name=None):
        """
        Fit the model using X as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
        resample_negatives : bool
            Whether to resample negatives at the beginning of each epoch
        n_processes : int
            Number of processes for batching utilities
        low_memory : bool
            Whether to use low memory mode (keeps data on CPU)
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print detailed logs during training
        precomputed_p_sym_path : str or None
            Path to pickled precomputed P_sym
        precomputed_negatives_path : str or None
            Path to pickled negative edges
        cache_p_sym_path : str or None
            Path where computed P_sym should be loaded from or written to
        cache_negatives_path : str or None
            Path where sampled negative edges should be loaded from or written to
        use_wandb : bool
            If True, log metrics to wandb
        wandb_project : str or None
            wandb project name
        wandb_run_name : str or None
            wandb run name

        Returns
        -------
        self
        """
        if use_wandb:
            import wandb
            config = {
                "n_components": self.n_components,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "n_neighbors": self.n_neighbors,
                "a": self.a,
                "b": self.b,
                "correlation_weight": self.correlation_weight,
                "learning_rate": self.learning_rate,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "device": self.device,
                "use_batchnorm": self.use_batchnorm,
                "use_dropout": self.use_dropout,
                "clip_grad_norm": self.clip_grad_norm,
                "clip_grad_value": self.clip_grad_value,
                "pos_ratio": self.pos_ratio,
                "architecture": self.architecture,
                "correlation_distance_transform": self.correlation_distance_transform,
                "lr_schedule": self.lr_schedule,
                "warmup_steps": self.warmup_steps,
                "total_steps_estimate": self.total_steps_estimate,
                "use_amp": self.use_amp,
                "anchored_init": self.anchored_init,
                "anchored_init_epochs": self.anchored_init_epochs,
                "anchored_init_lr": self.anchored_init_lr,
                "anchored_init_path": self.anchored_init_path,
                "anchor_hold_weight": self.anchor_hold_weight,
                "anchor_hold_fraction": self.anchor_hold_fraction,
                "midnear_enabled": self.midnear_enabled,
                "mn_pairs_per_batch": self.mn_pairs_per_batch,
                "mn_weight_scale": self.mn_weight_scale,
                "gpu_resident_data": str(self.gpu_resident_data),
                "gpu_resident_vram_budget_gb": self.gpu_resident_vram_budget_gb,
            }
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            self.wandb_run = wandb.run
            self.wandb_run_id = wandb.run.id

        edge_list_mode = precomputed_edges_path is not None

        if edge_list_mode:
            # Scale path: stream positive edges from the precomputed .npz and
            # sample negatives on the fly. X is kept lazy (may be a memmap /
            # MemmapArrayConcatenator) — never materialise >=2 GB.
            n_features = int(X.shape[1])
            n_train = int(X.shape[0])
            logging.info(f"Input shape (lazy, edge-list mode): ({n_train}, {n_features})")
            if self.model is None:
                self._init_model(n_features)
            ed = None
            dataset, loader, n_pos_edges = self._prepare_edge_list_training(
                X, precomputed_edges_path, n_train, low_memory, random_state)
            neg_desc = "on-the-fly"
        else:
            logging.info("Casting input array to float32...")
            X = np.asarray(X).astype(np.float32)
            logging.info(f"Input shape: {X.shape}")

            if self.model is None:
                self._init_model(X.shape[1])

            # Load or compute P_sym
            if precomputed_p_sym_path is not None:
                logging.info("Loading precomputed P_sym from %s", precomputed_p_sym_path)
                with open(precomputed_p_sym_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict) and "P_sym" in loaded:
                        P_sym = loaded["P_sym"]
                    else:
                        P_sym = loaded
            elif cache_p_sym_path is not None and Path(cache_p_sym_path).exists():
                logging.info("Loading cached P_sym from %s", cache_p_sym_path)
                with open(cache_p_sym_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict) and "P_sym" in loaded:
                        P_sym = loaded["P_sym"]
                    else:
                        P_sym = loaded
            else:
                logging.info("Computing p_sym using compute_all_p_umap...")
                P_sym = compute_all_p_umap(X, k=self.n_neighbors)
                if cache_p_sym_path is not None:
                    cache_path = Path(cache_p_sym_path)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    logging.info("Saving cached P_sym to %s", cache_p_sym_path)
                    with cache_path.open("wb") as f:
                        pickle.dump({"P_sym": P_sym}, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Create the EdgeDataset
            logging.info("Creating EdgeDataset...")
            ed = EdgeDataset(P_sym)

            # Load precomputed negative edges if available
            if precomputed_negatives_path is not None:
                logging.info("Loading precomputed negative edges from %s", precomputed_negatives_path)
                with open(precomputed_negatives_path, "rb") as f:
                    neg_edges = pickle.load(f)
                ed.neg_edges = neg_edges
            elif cache_negatives_path is not None and Path(cache_negatives_path).exists():
                logging.info("Loading cached negative edges from %s", cache_negatives_path)
                with open(cache_negatives_path, "rb") as f:
                    ed.neg_edges = pickle.load(f)

            # Create feature dataset (labels come from the balanced loader now)
            if low_memory:
                logging.info("Using low memory mode (data on CPU, async transfer to GPU).")
                dataset = VariableDataset(X)
            else:
                logging.info("Using high memory mode (data moved to device).")
                dataset = VariableDataset(X).to(self.device)

            loader = ed.get_balanced_loader(
                batch_size=self.batch_size,
                pos_ratio=self.pos_ratio,
                shuffle=True,
                random_state=random_state,
                n_processes=n_processes,
                verbose=verbose,
                positive_target_mode=self.positive_target_mode)
            if cache_negatives_path is not None and precomputed_negatives_path is None and not Path(cache_negatives_path).exists():
                cache_path = Path(cache_negatives_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                logging.info("Saving cached negative edges to %s", cache_negatives_path)
                with cache_path.open("wb") as f:
                    pickle.dump(ed.neg_edges, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_pos_edges = len(ed.pos_edges)
            neg_desc = str(len(ed.neg_edges))

        n_train_rows = int(len(dataset))

        # ── Anchored initialization (plan §4.2) + hold distillation (§4.3) ──
        # Pretrain the encoder to regress deterministic 2D targets before the
        # UMAP-loss phase, so every seed starts in the same basin. The main
        # optimizer/scheduler are created *after* this block, resetting their
        # state relative to pretraining. When anchor_hold_weight>0, the same
        # targets are kept resident and an ongoing MSE term is added throughout
        # the main phase (reference-atlas distillation proper).
        self.anchor_targets_ = None
        self.anchor_scale_ = None
        self._anchor_targets_dev = None
        want_hold = bool(self.anchor_hold_weight and self.anchor_hold_weight > 0)
        has_init = bool(self.anchored_init and self.anchored_init != "none")
        if want_hold and not has_init:
            raise ValueError(
                "anchor_hold_weight>0 requires anchored_init in {pca,file} to "
                "define the target source (use anchored_init_epochs=0 for a "
                "hold-only run with no pretrain).")
        if has_init:
            targets = self._compute_anchor_targets(
                X, n_train_rows, random_state=random_state)
            self.anchor_targets_ = targets
            if self.anchored_init_epochs and self.anchored_init_epochs > 0:
                logging.info(
                    "Anchored init (%s): pretraining encoder to regress 2D "
                    "targets (%d epochs)...",
                    self.anchored_init, self.anchored_init_epochs)
                self._anchored_pretrain(dataset, targets, n_train_rows, random_state)
            if want_hold:
                logging.info(
                    "Anchor-hold distillation enabled (w=%.3g, frac=%.3g): "
                    "ongoing MSE to targets during the main phase.",
                    self.anchor_hold_weight, self.anchor_hold_fraction)
                self._anchor_targets_dev = torch.from_numpy(
                    np.asarray(targets, dtype=np.float32)).to(self.device)

        # ── Mid-near pair loss setup (plan §6 Phase 1, PaCMAP-style) ──
        mn_rng = np.random.RandomState(random_state + 104729)
        mn_gen = None
        if self._fast_device_path:
            mn_gen = torch.Generator(device=self.device)
            mn_gen.manual_seed(int(random_state) + 104729)

        # ── Anchor-hold sampler setup (reference-atlas distillation, §4.3) ──
        hold_rng = np.random.RandomState(random_state + 92821)
        hold_gen = None
        if self._fast_device_path:
            hold_gen = torch.Generator(device=self.device)
            hold_gen.manual_seed(int(random_state) + 92821)
        mn_total_steps = (
            self.total_steps_estimate if self.total_steps_estimate > 0
            else max(len(loader) * self.n_epochs, 1)
        )
        mn_t_phase1 = max(0.1 * mn_total_steps, 1.0)

        if 'cuda' in str(self.device):
            torch.backends.cudnn.benchmark = True

        # Mixed precision: only on CUDA (MPS and CPU don't support GradScaler)
        use_amp = self.use_amp and 'cuda' in str(self.device)
        scaler = torch.amp.GradScaler(self.device, enabled=use_amp) if use_amp else None

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)
        elif self.lr_schedule == "cosine":
            total_steps = max(self.total_steps_estimate, 1)

            def lr_lambda(step):
                if self.warmup_steps > 0 and step < self.warmup_steps:
                    return step / self.warmup_steps
                progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        # P0.2: stop at the effective LR horizon. Past total_steps the cosine
        # schedule is clamped to LR=0, so further iterations run forward/backward/
        # step but never change weights (82–90% of the loop in past configs).
        # Stop there — identical weights, ~5.6× less wall time — unless the caller
        # forces a step cap. Persist honest step accounting.
        planned_loop = int(len(loader)) * int(self.n_epochs)
        lr_horizon = (max(int(self.total_steps_estimate), 1)
                      if self.lr_schedule == "cosine" else planned_loop)
        stop_step = lr_horizon
        if self._max_train_steps is not None:
            stop_step = min(stop_step, int(self._max_train_steps))
        if self.lr_schedule == "cosine" and planned_loop > int(lr_horizon * 1.05):
            logging.warning(
                "P0.2: planned loop %d >> LR horizon %d — training stops at the "
                "horizon (the extra %.0f%% would be LR=0). Set total_steps_estimate "
                "to the intended optimizer-step budget.",
                planned_loop, lr_horizon, 100.0 * (planned_loop - lr_horizon) / planned_loop)
        self._train_stats = {"planned_loop_iters": planned_loop, "lr_horizon": lr_horizon,
                             "stop_step": stop_step, "optimizer_steps": 0,
                             "nonfinite_skips": 0, "final_lr": None}
        stop_training = False

        self.model.train()
        losses = []

        logging.info("Starting training with balanced batches...")
        logging.info(f"Batch size: {self.batch_size}, Pos ratio: {self.pos_ratio}")
        logging.info(f"Batches per epoch: {len(loader)}")
        logging.info(f"Positive edges: {n_pos_edges}, Negative edges: {neg_desc}")
        logging.info(f"Mixed precision: {use_amp}")
        logging.info(f"Positive target mode: {self.positive_target_mode}")

        # Per-batch GPU->CPU syncs (grad-norm diagnostics + per-class stats) are
        # only needed by wandb / the tqdm postfix. Skipping them when nothing
        # consumes them keeps the GPU from stalling on .item() every step.
        collect_diag = bool(use_wandb)
        collect_stats = bool(use_wandb or verbose)

        global_step = 0
        self._train_t0 = time.perf_counter()
        consecutive_nonfinite_losses = 0
        for epoch in range(self.n_epochs):
            epoch_loss_t = torch.zeros((), device=self.device)
            epoch_umap_t = torch.zeros((), device=self.device)
            epoch_corr_t = torch.zeros((), device=self.device)
            num_batches = 0
            pbar = tqdm(
                range(len(loader)),
                desc=f'Epoch {epoch+1}/{self.n_epochs}',
                position=0,
                disable=not verbose,
            )

            if self._fast_device_path:
                # DeviceEdgeSampler yields (src, dst, targets) already on-device.
                _batch_iter = iter(loader)

                def _get_next(_it=_batch_iter):
                    try:
                        return next(_it)
                    except StopIteration:
                        return None
            else:
                prefetcher = DataPrefetcher(loader, dataset, self.device)
                _get_next = prefetcher.next

            batch = _get_next()
            while batch is not None:
                optimizer.zero_grad(set_to_none=True)
                src_values, dst_values, targets = batch

                with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                    # Forward pass
                    src_embeddings = self.model(src_values)
                    dst_embeddings = self.model(dst_values)

                    # Low-D similarity kernel (P0.1 switch). `dists` is the
                    # kernel's radial term (‖Δ‖_{2b} legacy / ‖Δ‖²^b umap), used
                    # only for optional per-batch stats below.
                    qs, dists = self._low_dim_qs(src_embeddings, dst_embeddings)

                    # Keep BCE inputs in its valid probability domain. Larger
                    # scale pilots can occasionally produce non-finite low-dim
                    # distances early in training.
                    qs = torch.nan_to_num(qs, nan=1e-7, posinf=1 - 1e-7, neginf=1e-7)
                    qs = torch.clamp(qs, min=1e-7, max=1 - 1e-7)

                # BCELoss is unsafe under autocast. Keep the forward pass mixed
                # precision, then compute BCE in fp32.
                targets_for_loss = torch.nan_to_num(targets.float(), nan=0.0, posinf=1.0, neginf=0.0)
                targets_for_loss = torch.clamp(targets_for_loss, min=0.0, max=1.0)
                umap_loss = self.loss_fn(qs.float(), targets_for_loss)
                # P0.3: skip the correlation branch entirely when its weight is 0
                # (the frozen recipe). It computed two 384-D distance vectors +
                # a sync-forcing Pearson every batch, and `0 * NaN = NaN` could
                # spuriously skip a whole batch. Use an on-device scalar 0 for
                # aggregate logging only.
                if self.correlation_weight != 0.0:
                    # Correlation loss in full precision (numerically sensitive)
                    corr_loss = compute_correlation_loss(
                        self._transform_correlation_distances(torch.norm(src_values - dst_values, dim=1)),
                        self._transform_correlation_distances(
                            torch.norm(src_embeddings.float() - dst_embeddings.float(), dim=1)
                        )
                    )
                    loss = umap_loss + self.correlation_weight * corr_loss
                else:
                    corr_loss = torch.zeros((), device=umap_loss.device)
                    loss = umap_loss

                # ── Mid-near attractive term (PaCMAP-style global structure) ──
                mn_loss_val = 0.0
                w_mn = 0.0
                if self.midnear_enabled:
                    if self.mn_pairs_per_batch > 0:
                        mn_m = self.mn_pairs_per_batch
                    else:
                        # auto: match the number of positive edges in this batch.
                        mn_m = int((targets > 0.5).sum().item())
                    if mn_m > 0:
                        if self._fast_device_path:
                            anchor_feats, partner_feats = \
                                self._sample_midnear_features_device(
                                    mn_m, n_train_rows, mn_gen)
                        else:
                            anchor_feats, partner_feats = \
                                self._sample_midnear_features(
                                    dataset, mn_m, n_train_rows, mn_rng)
                        with torch.autocast(device_type='cuda' if use_amp else 'cpu',
                                            enabled=use_amp):
                            z_a = self.model(anchor_feats)
                            z_b = self.model(partner_feats)
                        # PaCMAP mid-near loss: d~ = ||z_a - z_b||^2 + 1;
                        # L_mn = mean( d~ / (10000 + d~) ). Bounded per pair.
                        d_mn = (z_a.float() - z_b.float()).pow(2).sum(dim=1) + 1.0
                        mn_loss = (d_mn / (10000.0 + d_mn)).mean()
                        # PaCMAP weight schedule: 1000 -> 3 over the first 10% of
                        # steps, then hold at 3, scaled by mn_weight_scale.
                        if global_step < mn_t_phase1:
                            frac = global_step / mn_t_phase1
                            w_mn = 1000.0 * (1.0 - frac) + 3.0 * frac
                        else:
                            w_mn = 3.0
                        w_mn *= self.mn_weight_scale
                        loss = loss + w_mn * mn_loss
                        mn_loss_val = mn_loss.item() if use_wandb else 0.0

                # ── Anchor-hold term (reference-atlas distillation, §4.3) ──
                # Draw anchor_hold_fraction of the batch as random anchors and
                # pull their projections toward the frozen teacher targets. Tiny
                # n x 2 target tensor is resident on the training device; gathers
                # reuse the fast-path feature matrix when active, else the lazy
                # dataset (memmap-safe). Composes with midnear + the fast path.
                hold_loss_val = 0.0
                if self._anchor_targets_dev is not None:
                    h_m = max(1, int(self.batch_size * self.anchor_hold_fraction))
                    if self._fast_device_path:
                        h_idx = torch.randint(0, n_train_rows, (h_m,),
                                              generator=hold_gen, device=self.device)
                        h_feats = self._X_dev.index_select(h_idx)
                    else:
                        h_np = hold_rng.randint(0, n_train_rows, size=h_m)
                        h_feats = self._gather_feature_rows(dataset, h_np)
                        h_idx = torch.as_tensor(h_np, dtype=torch.long,
                                                device=self.device)
                    h_tgt = self._anchor_targets_dev.index_select(0, h_idx)
                    with torch.autocast(device_type='cuda' if use_amp else 'cpu',
                                        enabled=use_amp):
                        z_h = self.model(h_feats)
                    hold_loss = (z_h.float() - h_tgt.float()).pow(2).sum(dim=1).mean()
                    loss = loss + self.anchor_hold_weight * hold_loss
                    hold_loss_val = hold_loss.item() if use_wandb else 0.0

                if not torch.isfinite(loss):
                    consecutive_nonfinite_losses += 1
                    self._train_stats["nonfinite_skips"] += 1
                    logging.warning(
                        "Non-finite loss detected at step %d, skipping batch (%d consecutive)",
                        global_step,
                        consecutive_nonfinite_losses,
                    )
                    if consecutive_nonfinite_losses >= 100:
                        raise RuntimeError(
                            f"Aborting training after {consecutive_nonfinite_losses} "
                            "consecutive non-finite losses"
                        )
                    batch = _get_next()
                    pbar.update(1)
                    continue
                consecutive_nonfinite_losses = 0

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # Gradient clipping is always applied; the norm read-backs it
                # needs for logging are gated so we don't sync every step.
                if collect_diag:
                    pre_clip_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                    grad_norm_pre = float(np.mean(pre_clip_norms)) if pre_clip_norms else 0.0
                else:
                    grad_norm_pre = 0.0

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                if self.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad_value)

                if collect_diag:
                    post_clip_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                    grad_norm_post = float(np.mean(post_clip_norms)) if post_clip_norms else 0.0
                    clipping_ratio = grad_norm_post / grad_norm_pre if grad_norm_pre > 0 else 1.0
                else:
                    grad_norm_post = 0.0
                    clipping_ratio = 1.0

                optimizer_was_run = True
                if scaler is not None:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_was_run = scaler.get_scale() >= scale_before
                else:
                    optimizer.step()

                if self.lr_schedule == "cosine" and optimizer_was_run:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']

                # Per-class stats only when a consumer (wandb/pbar) needs them.
                if collect_stats:
                    pos_mask = targets > 0.5
                    neg_mask = ~pos_mask
                    pos_qs_mean = qs[pos_mask].mean().item() if pos_mask.any() else 0.0
                    neg_qs_mean = qs[neg_mask].mean().item() if neg_mask.any() else 0.0
                    mean_pos_dist = dists[pos_mask].mean().item() if pos_mask.any() else 0.0
                    mean_neg_dist = dists[neg_mask].mean().item() if neg_mask.any() else 0.0
                else:
                    pos_qs_mean = neg_qs_mean = mean_pos_dist = mean_neg_dist = 0.0

                # Accumulate epoch losses on-device; sync once per epoch below.
                epoch_loss_t += loss.detach()
                epoch_umap_t += umap_loss.detach()
                epoch_corr_t += corr_loss.detach()
                num_batches += 1
                global_step += 1
                if optimizer_was_run and current_lr > 0:
                    self._train_stats["optimizer_steps"] += 1
                self._train_stats["final_lr"] = current_lr

                # P0.2: stop at the effective horizon (LR=0 past here → no weight
                # change). Ends the run in ~1/5.6 the wall time with identical
                # weights.
                if global_step >= stop_step:
                    stop_training = True
                    logging.info("P0.2: reached LR horizon at step %d/%d (planned loop %d) — "
                                 "stopping; %d optimizer steps.", global_step, stop_step,
                                 planned_loop, self._train_stats["optimizer_steps"])
                    break

                # Step-rate logging every 10k steps (plan B1: never fly blind on
                # a long run again). Reports steps/s over the last window + ETA.
                if global_step % 10000 == 0:
                    now = time.perf_counter()
                    if not hasattr(self, "_rate_t0"):
                        self._rate_t0 = self._train_t0 if hasattr(self, "_train_t0") else now
                        self._rate_s0 = 0
                    dt = now - self._rate_t0
                    dsteps = global_step - self._rate_s0
                    rate = dsteps / dt if dt > 0 else 0.0
                    total_planned = self.n_epochs * len(loader)
                    remaining = max(0, total_planned - global_step)
                    eta_h = (remaining / rate / 3600.0) if rate > 0 else float("nan")
                    logging.info("  step %d/%d | %.0f steps/s | ETA %.1fh",
                                 global_step, total_planned, rate, eta_h)
                    self._rate_t0 = now
                    self._rate_s0 = global_step

                if self._max_train_steps is not None and global_step == self._bench_warmup:
                    if 'cuda' in str(self.device):
                        torch.cuda.synchronize(self.device)
                    self._bench_t0 = time.perf_counter()

                if use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'umap_loss': umap_loss.item(),
                        'corr_loss': corr_loss.item(),
                        'mn_loss': mn_loss_val,
                        'mn_weight': w_mn,
                        'anchor_hold_loss': hold_loss_val,
                        'learning_rate': current_lr,
                        'mean_distance': dists.mean().item(),
                        'std_distance': dists.std().item(),
                        'mean_pos_distance': mean_pos_dist,
                        'mean_neg_distance': mean_neg_dist,
                        'pos_qs_mean': pos_qs_mean,
                        'neg_qs_mean': neg_qs_mean,
                        'grad_norm_pre': grad_norm_pre,
                        'grad_norm_post': grad_norm_post,
                        'clipping_ratio': clipping_ratio,
                        'global_step': global_step,
                    })

                batch = _get_next()
                pbar.update(1)
                if verbose:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'umap': f'{umap_loss.item():.4f}',
                        'pos_q': f'{pos_qs_mean:.3f}',
                        'neg_q': f'{neg_qs_mean:.3f}',
                    })

                if self._max_train_steps is not None and global_step >= self._max_train_steps:
                    batch = None  # benchmark hook: stop this epoch early

            pbar.close()

            if stop_training:
                break  # P0.2: LR horizon reached — end the run, not just the epoch

            if self._max_train_steps is not None and global_step >= self._max_train_steps:
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize(self.device)
                self._bench_seconds = time.perf_counter() - self._bench_t0
                break  # benchmark hook: stop the whole run, not just the epoch

            if resample_negatives and ed is not None:
                ed.neg_edges = None  # Force re-sampling
                loader = ed.get_balanced_loader(
                    batch_size=self.batch_size,
                    pos_ratio=self.pos_ratio,
                    shuffle=True,
                    random_state=random_state + epoch + 1,
                    n_processes=n_processes,
                    verbose=verbose)

            epoch_loss = float(epoch_loss_t.item())
            epoch_umap_loss = float(epoch_umap_t.item())
            epoch_corr_loss = float(epoch_corr_t.item())
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_umap = epoch_umap_loss / max(num_batches, 1)
            avg_corr = epoch_corr_loss / max(num_batches, 1)
            losses.append(avg_loss)

            # Step plateau scheduler per-epoch; cosine schedule is stepped per-batch.
            if self.lr_schedule == "plateau":
                scheduler.step(avg_loss)

            if verbose:
                logging.info(
                    'Epoch %d/%d — Loss: %.4f (umap: %.4f, corr: %.4f) — LR: %.2e',
                    epoch+1, self.n_epochs, avg_loss, avg_umap, avg_corr, current_lr)
            if use_wandb:
                log_dict = {'epoch_loss': avg_loss, 'epoch_umap_loss': avg_umap, 'epoch_corr_loss': avg_corr}
                if 'cuda' in str(self.device):
                    peak_memory = torch.cuda.max_memory_allocated(self.device) / 1e9
                    log_dict['epoch_peak_gpu_memory'] = peak_memory
                    logging.info("Peak GPU Memory: %.2f GB", peak_memory)
                wandb.log(log_dict)

        self.is_fitted = True
        # P0.2: honest step accounting (loop iters vs effective optimizer steps).
        s = self._train_stats
        logging.info("Step accounting: %d optimizer steps (LR>0) / %d planned loop iters "
                     "(horizon %d); %d non-finite skips; final LR %.2e.",
                     s["optimizer_steps"], s["planned_loop_iters"], s["lr_horizon"],
                     s["nonfinite_skips"], s["final_lr"] or 0.0)
        # Stop HostStreamEdgeSampler producer threads so they don't keep drawing
        # and discarding batches during the final transform / downstream scoring.
        if hasattr(loader, "close"):
            loader.close()
        if use_wandb:
            wandb.finish()
        return self

    def _transform_correlation_distances(self, distances):
        if self.correlation_distance_transform == "raw":
            return distances
        if self.correlation_distance_transform == "log1p":
            return torch.log1p(distances)
        raise ValueError(
            f"Unknown correlation_distance_transform: {self.correlation_distance_transform}"
        )

    def transform(self, X, batch_size=4096):
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        batch_size : int
            Process in batches to avoid OOM on large inputs

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        self.model.eval()
        # Do not materialise the whole array — index per batch so lazy memmaps
        # / MemmapArrayConcatenator inputs stay off-RAM (>=2 GB rule).
        n = len(X)

        results = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                # Clamp the slice end: MemmapArrayConcatenator's slice handler
                # does not clamp `stop` to the array length.
                chunk = np.asarray(X[i:min(i + batch_size, n)], dtype=np.float32)
                batch = torch.from_numpy(chunk).to(self.device)
                results.append(self.model(batch).cpu().numpy())

        return np.concatenate(results, axis=0)

    def fit_transform(self, X, **kwargs):
        """Fit and transform in one call."""
        self.fit(X, **kwargs)
        return self.transform(X)

    def save(self, path):
        """Save the model to a file."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'architecture': self.architecture,
            'input_dim': self.input_dim,
            'n_components': self.n_components,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'n_neighbors': self.n_neighbors,
            'a': self.a,
            'b': self.b,
            'low_dim_kernel': self.low_dim_kernel,
            'correlation_weight': self.correlation_weight,
            'learning_rate': self.learning_rate,
            'use_batchnorm': self.use_batchnorm,
            'use_dropout': self.use_dropout,
            'clip_grad_norm': self.clip_grad_norm,
            'clip_grad_value': self.clip_grad_value,
            'pos_ratio': self.pos_ratio,
            'positive_target_mode': self.positive_target_mode,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path, device=None):
        """Load a saved model."""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        save_dict = torch.load(path, map_location=device, weights_only=False)

        instance = cls(
            n_components=save_dict['n_components'],
            hidden_dim=save_dict['hidden_dim'],
            n_layers=save_dict['n_layers'],
            n_neighbors=save_dict.get('n_neighbors', 15),
            a=save_dict['a'],
            b=save_dict['b'],
            # Checkpoints predating the P0.1 switch were trained with the
            # legacy Lp kernel; load them as such (never silently upgrade).
            low_dim_kernel=save_dict.get('low_dim_kernel', 'legacy_lp'),
            correlation_weight=save_dict['correlation_weight'],
            device=device,
            use_batchnorm=save_dict['use_batchnorm'],
            use_dropout=save_dict['use_dropout'],
            clip_grad_norm=save_dict['clip_grad_norm'],
            clip_grad_value=save_dict['clip_grad_value'],
            pos_ratio=save_dict.get('pos_ratio', 0.5),
            architecture=save_dict.get('architecture', 'mlp'),
            positive_target_mode=save_dict.get('positive_target_mode', 'probability'),
        )

        state_dict = save_dict['model_state_dict']
        input_dim = save_dict.get('input_dim')
        if input_dim is None:
            # Backward compat: infer from the first linear layer of either arch.
            if 'model.0.weight' in state_dict:            # MLP
                input_dim = state_dict['model.0.weight'].shape[1]
            elif 'proj_in.weight' in state_dict:          # ResidualBottleneckMLP
                input_dim = state_dict['proj_in.weight'].shape[1]
            else:
                raise KeyError("Cannot infer input_dim from checkpoint state_dict")
        instance._init_model(input_dim=int(input_dim))
        instance.model.load_state_dict(state_dict)
        instance.is_fitted = True

        return instance

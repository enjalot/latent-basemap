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
        self.model = None
        self.input_dim = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False

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

    def _prepare_edge_list_training(self, X, edges_path, n_train, low_memory, random_state):
        """Build the (dataset, loader) pair for the precomputed edge-list path.

        Loads the .npz edge list (lazily), filters edges to the training node
        range when the index covers more nodes than X (e.g. a 15M slice of the
        150M index), optionally builds a neighbour-rejection set, and returns a
        memmap-safe dataset plus an on-the-fly balanced iterator.
        """
        from .datasets.edge_list_dataset import (
            EdgeListBalancedIterator, LazyArrayDataset,
            load_edge_arrays, build_edge_key_set,
        )
        from .datasets.covariates_datasets import VariableDataset

        load_weights = self.positive_target_mode == "probability"
        sources, targets, weights, n_nodes = load_edge_arrays(
            edges_path, load_weights=load_weights)
        logging.info(
            "Loaded edge list from %s: %d directed edges, n_nodes=%d",
            edges_path, len(sources), n_nodes,
        )

        # Filter to the training range if the index spans more nodes than X.
        if n_nodes > n_train:
            logging.info(
                "Filtering edges to training range (n_nodes=%d > n_train=%d)...",
                n_nodes, n_train,
            )
            mask = (np.asarray(sources) < n_train) & (np.asarray(targets) < n_train)
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

        self.model.train()
        losses = []

        logging.info("Starting training with balanced batches...")
        logging.info(f"Batch size: {self.batch_size}, Pos ratio: {self.pos_ratio}")
        logging.info(f"Batches per epoch: {len(loader)}")
        logging.info(f"Positive edges: {n_pos_edges}, Negative edges: {neg_desc}")
        logging.info(f"Mixed precision: {use_amp}")
        logging.info(f"Positive target mode: {self.positive_target_mode}")

        global_step = 0
        consecutive_nonfinite_losses = 0
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            epoch_umap_loss = 0.0
            epoch_corr_loss = 0.0
            num_batches = 0
            pbar = tqdm(
                range(len(loader)),
                desc=f'Epoch {epoch+1}/{self.n_epochs}',
                position=0,
                disable=not verbose,
            )

            prefetcher = DataPrefetcher(loader, dataset, self.device)
            batch = prefetcher.next()
            while batch is not None:
                optimizer.zero_grad(set_to_none=True)
                src_values, dst_values, targets = batch

                with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                    # Forward pass
                    src_embeddings = self.model(src_values)
                    dst_embeddings = self.model(dst_values)

                    # Compute pairwise distances in embedding space
                    dists = torch.norm(src_embeddings - dst_embeddings, dim=1, p=2*self.b)

                    # UMAP similarity in low-dim: q_ij = (1 + a * ||z_i - z_j||^{2b})^{-1}
                    qs = torch.pow(1 + self.a * dists, -1)

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
                # Correlation loss in full precision (distance correlation is numerically sensitive)
                corr_loss = compute_correlation_loss(
                    self._transform_correlation_distances(torch.norm(src_values - dst_values, dim=1)),
                    self._transform_correlation_distances(
                        torch.norm(src_embeddings.float() - dst_embeddings.float(), dim=1)
                    )
                )

                loss = umap_loss + self.correlation_weight * corr_loss

                if not torch.isfinite(loss):
                    consecutive_nonfinite_losses += 1
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
                    batch = prefetcher.next()
                    pbar.update(1)
                    continue
                consecutive_nonfinite_losses = 0

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # Gradient diagnostics
                pre_clip_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                grad_norm_pre = float(np.mean(pre_clip_norms)) if pre_clip_norms else 0.0

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                if self.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_grad_value)

                post_clip_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                grad_norm_post = float(np.mean(post_clip_norms)) if post_clip_norms else 0.0
                clipping_ratio = grad_norm_post / grad_norm_pre if grad_norm_pre > 0 else 1.0

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

                # Per-class stats
                pos_mask = targets > 0.5
                neg_mask = ~pos_mask
                pos_qs_mean = qs[pos_mask].mean().item() if pos_mask.any() else 0.0
                neg_qs_mean = qs[neg_mask].mean().item() if neg_mask.any() else 0.0
                mean_pos_dist = dists[pos_mask].mean().item() if pos_mask.any() else 0.0
                mean_neg_dist = dists[neg_mask].mean().item() if neg_mask.any() else 0.0

                epoch_loss += loss.item()
                epoch_umap_loss += umap_loss.item()
                epoch_corr_loss += corr_loss.item()
                num_batches += 1
                global_step += 1

                if use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'umap_loss': umap_loss.item(),
                        'corr_loss': corr_loss.item(),
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

                batch = prefetcher.next()
                pbar.update(1)
                if verbose:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'umap': f'{umap_loss.item():.4f}',
                        'pos_q': f'{pos_qs_mean:.3f}',
                        'neg_q': f'{neg_qs_mean:.3f}',
                    })

            pbar.close()

            if resample_negatives and ed is not None:
                ed.neg_edges = None  # Force re-sampling
                loader = ed.get_balanced_loader(
                    batch_size=self.batch_size,
                    pos_ratio=self.pos_ratio,
                    shuffle=True,
                    random_state=random_state + epoch + 1,
                    n_processes=n_processes,
                    verbose=verbose)

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

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
        self.model = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False

    def _init_model(self, input_dim):
        """Initialize the configured parametric model."""
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

    def fit(self, X, y=None,
            resample_negatives=False,
            n_processes=6,
            low_memory=False,
            random_state=0,
            verbose=True,
            precomputed_p_sym_path=None,
            precomputed_negatives_path=None,
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
        else:
            logging.info("Computing p_sym using compute_all_p_umap...")
            P_sym = compute_all_p_umap(X, k=self.n_neighbors)

        # Create the EdgeDataset
        logging.info("Creating EdgeDataset...")
        ed = EdgeDataset(P_sym)

        # Load precomputed negative edges if available
        if precomputed_negatives_path is not None:
            logging.info("Loading precomputed negative edges from %s", precomputed_negatives_path)
            with open(precomputed_negatives_path, "rb") as f:
                neg_edges = pickle.load(f)
            ed.neg_edges = neg_edges

        # Create feature dataset (labels come from the balanced loader now)
        if low_memory:
            logging.info("Using low memory mode (data on CPU, async transfer to GPU).")
            dataset = VariableDataset(X)
        else:
            logging.info("Using high memory mode (data moved to device).")
            dataset = VariableDataset(X).to(self.device)

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

        loader = ed.get_balanced_loader(
            batch_size=self.batch_size,
            pos_ratio=self.pos_ratio,
            shuffle=True,
            random_state=random_state)

        logging.info("Starting training with balanced batches...")
        logging.info(f"Batch size: {self.batch_size}, Pos ratio: {self.pos_ratio}")
        logging.info(f"Batches per epoch: {len(loader)}")
        logging.info(f"Positive edges: {len(ed.pos_edges)}, Negative edges: {len(ed.neg_edges)}")
        logging.info(f"Mixed precision: {use_amp}")

        global_step = 0
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

                    # Clamp qs to avoid log(0) in BCE
                    qs = torch.clamp(qs, min=1e-7, max=1 - 1e-7)

                # BCELoss is unsafe under autocast. Keep the forward pass mixed
                # precision, then compute BCE in fp32.
                umap_loss = self.loss_fn(qs.float(), targets.float())
                # Correlation loss in full precision (distance correlation is numerically sensitive)
                corr_loss = compute_correlation_loss(
                    self._transform_correlation_distances(torch.norm(src_values - dst_values, dim=1)),
                    self._transform_correlation_distances(
                        torch.norm(src_embeddings.float() - dst_embeddings.float(), dim=1)
                    )
                )

                loss = umap_loss + self.correlation_weight * corr_loss

                # Check for NaN loss
                if torch.isnan(loss):
                    logging.warning("NaN loss detected at step %d, skipping batch", global_step)
                    batch = prefetcher.next()
                    pbar.update(1)
                    continue

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

            if resample_negatives:
                ed.neg_edges = None  # Force re-sampling
                loader = ed.get_balanced_loader(
                    batch_size=self.batch_size,
                    pos_ratio=self.pos_ratio,
                    shuffle=True,
                    random_state=random_state + epoch + 1)

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
        X = np.asarray(X, dtype=np.float32)

        results = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
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
            'n_components': self.n_components,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'n_neighbors': self.n_neighbors,
            'a': self.a,
            'b': self.b,
            'correlation_weight': self.correlation_weight,
            'use_batchnorm': self.use_batchnorm,
            'use_dropout': self.use_dropout,
            'clip_grad_norm': self.clip_grad_norm,
            'clip_grad_value': self.clip_grad_value,
            'pos_ratio': self.pos_ratio,
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
        save_dict = torch.load(path, map_location=device)

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
        )

        input_dim = save_dict['model_state_dict']['model.0.weight'].shape[1]
        instance._init_model(input_dim=input_dim)
        instance.model.load_state_dict(save_dict['model_state_dict'])
        instance.is_fitted = True

        return instance

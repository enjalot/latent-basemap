from .models.mlp import MLP
from .datasets.edge_dataset import EdgeDataset
from .datasets.covariates_datasets import VariableDataset, TorchSparseDataset
from .utils.losses import compute_correlation_loss
from .utils.graph import compute_all_p_umap
from .utils.data_prefetcher import DataPrefetcher

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm.auto import tqdm
import pickle  # Added for loading precomputed files
import logging

# Setup logging (similar to edge_dataset.py)
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
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_batchnorm=False,
        use_dropout=False,
        clip_grad_norm=1.0,  # Maximum norm of gradients
        clip_grad_value=None,  # Maximum value of gradients (optional)
    ):
        """
        Initialize ParametricUMAP.
        
        Parameters:
        -----------
        n_components : int
            Number of dimensions in the output embedding
        hidden_dim : int
            Dimension of hidden layers in the MLP
        n_layers : int
            Number of hidden layers in the MLP
        n_neighbors : int
            Number of nearest neighbors to consider for UMAP probabilities
        a, b : float
            UMAP parameters for the optimization
        correlation_weight : float
            Weight of the correlation loss term
        learning_rate : float
            Learning rate for the optimizer
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        device : str
            Device to use for computations ('cpu' or 'cuda')
        use_batchnorm : bool
            Whether to use batch normalization in the MLP
        use_dropout : bool
            Whether to use dropout in the MLP
        clip_grad_norm : float
            Maximum norm of gradients. If None, gradient norm clipping is disabled.
        clip_grad_value : float or None
            Maximum value of gradients. If None, gradient value clipping is disabled.
        """
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
        
        self.model = None
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False
        
    def _init_model(self, input_dim):
        """Initialize the MLP model"""
        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_components,
            num_layers=self.n_layers,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout
        ).to(self.device)
        
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
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        y : Ignored
            Not used, present for API consistency
            
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
            If provided, path to a file containing pickled precomputed P_sym
            
        precomputed_negatives_path : str or None
            If provided, path to a file containing pickled negative edges
            
        use_wandb : bool
            If True, initialize wandb and log metrics
            
        wandb_project : str or None
            The wandb project name (if use_wandb is True)
            
        wandb_run_name : str or None
            The wandb run name (if use_wandb is True)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Initialize wandb if enabled; also log model hyperparameters.
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
                "clip_grad_value": self.clip_grad_value
            }
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        
        logging.info("Casting input array to float32...")
        X = np.asarray(X).astype(np.float32)
        logging.info(f"Input shape: {X.shape}")
    
        # Initialize model if not already done
        if self.model is None:
            self._init_model(X.shape[1])
            
        # Create datasets - load precomputed P_sym if available
        if precomputed_p_sym_path is not None:
            logging.info("Loading precomputed P_sym from %s", precomputed_p_sym_path)
            with open(precomputed_p_sym_path, "rb") as f:
                P_sym = pickle.load(f)["P_sym"]
        else:
            logging.info("Computing p_sym using compute_all_p_umap...")
            P_sym = compute_all_p_umap(X, k=self.n_neighbors)
            
        # Create the EdgeDataset instance
        logging.info("Creating EdgeDataset...")
        ed = EdgeDataset(P_sym)
        
        # Load negative edges if a precomputed file is provided
        if precomputed_negatives_path is not None:
            logging.info("Loading precomputed negative edges from %s", precomputed_negatives_path)
            with open(precomputed_negatives_path, "rb") as f:
                neg_edges = pickle.load(f)
            ed.neg_edges = neg_edges  # Assumes EdgeDataset uses this attribute internally
        
        # Create appropriate datasets for training
        if low_memory:
            logging.info("Using low memory mode.")
            dataset = VariableDataset(X)
            target_dataset = TorchSparseDataset(P_sym)
        else:
            logging.info("Using high memory mode (data moved to device).")
            dataset = VariableDataset(X).to(self.device)
            target_dataset = TorchSparseDataset(P_sym).to(self.device)
        
        # Enable cudnn benchmark for optimized performance on fixed input sizes
        torch.backends.cudnn.benchmark = True

        # Initialize mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        
        self.model.train()
        losses = []
        
        loader = ed.get_loader(batch_size=self.batch_size, 
                               sample_first=True,
                               random_state=random_state,
                               n_processes=n_processes,
                               verbose=verbose)
        
        logging.info("Starting training...")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Batches per epoch: {len(loader)}")

        
            
        # pbar = tqdm(range(self.n_epochs), desc='Epochs', position=0)
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            num_batches = 0
            pbar2 = tqdm(range(len(loader)), desc=f'Epoch {epoch+1} batches:', position=0)
            
            prefetcher = DataPrefetcher(loader, dataset, target_dataset, self.device)
            batch_idx = 0
            batch = prefetcher.next()
            while batch is not None:
                optimizer.zero_grad()
                src_values, dst_values, targets = batch

                # forward pass
                src_embeddings = self.model(src_values)
                dst_embeddings = self.model(dst_values)
                
                # Compute distances and logits
                dists = torch.norm(src_embeddings - dst_embeddings, dim=1, p=2*self.b)
                qs = torch.pow(1 + self.a * dists, -1)
                # logits = -self.a * torch.pow(dists, self.b)
                umap_loss = self.loss_fn(qs, targets)
                corr_loss = compute_correlation_loss(
                    torch.norm(src_values - dst_values, dim=1),
                    torch.norm(src_embeddings - dst_embeddings, dim=1)
                )
                loss = umap_loss + self.correlation_weight * corr_loss
                
                # Compute scaled gradients
                scaler.scale(loss).backward()
                
                # Unscale gradients to prepare for clipping
                scaler.unscale_(optimizer)
                # Log pre-clipping gradient norms
                pre_clip_grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                grad_norm_pre_avg = float(np.mean(pre_clip_grad_norms)) if pre_clip_grad_norms else 0.0
                
                # Apply gradient clipping if configured
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_grad_norm
                    )
                if self.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        clip_value=self.clip_grad_value
                    )
                post_clip_grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                grad_norm_post_avg = float(np.mean(post_clip_grad_norms)) if post_clip_grad_norms else 0.0
                clipping_ratio = grad_norm_post_avg / grad_norm_pre_avg if grad_norm_pre_avg > 0 else 0.0
                
                # Optimizer and scheduler step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(loss.item())
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log pos vs neg qs statistics
                if (targets == 1).sum() > 0:
                    pos_qs_mean = qs[targets==1].mean().item()
                    pos_qs_std = qs[targets==1].std().item()
                else:
                    pos_qs_mean, pos_qs_std = 0, 0
                if (targets == 0).sum() > 0:
                    neg_qs_mean = qs[targets==0].mean().item()
                    neg_qs_std = qs[targets==0].std().item()
                else:
                    neg_qs_mean, neg_qs_std = 0, 0

                epoch_loss += loss.item()
                num_batches += 1

                # Log to wandb including grad norms and qs statistics
                if use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'umap_loss': umap_loss.item(),
                        'corr_loss': corr_loss.item(),
                        'learning_rate': current_lr,
                        'mean_distance': dists.mean().item(),
                        'std_distance': dists.std().item(),
                        'mean_qs': qs.mean().item(),
                        'std_qs': qs.std().item(),
                        'pos_qs_mean': pos_qs_mean,
                        'pos_qs_std': pos_qs_std,
                        'neg_qs_mean': neg_qs_mean,
                        'neg_qs_std': neg_qs_std,
                        'grad_norm_pre': grad_norm_pre_avg,
                        'grad_norm_post': grad_norm_post_avg,
                        'clipping_ratio': clipping_ratio,
                        # 'step': batch_idx  # useful to chart progress over steps
                    })
                    
                batch = prefetcher.next()
                batch_idx += 1
                pbar2.update(1)
                pbar2.set_postfix({'loss': f'{loss.item():.4f}'})


            if resample_negatives:
                loader = ed.get_loader(batch_size=self.batch_size, sample_first=True)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            # pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            if verbose:
                logging.info('Epoch %d/%d, Loss: %.4f', epoch+1, self.n_epochs, avg_loss)
            if use_wandb:
                if torch.cuda.is_available():
                    memory = torch.cuda.memory_allocated(self.device)/1e9
                    peak_memory = torch.cuda.max_memory_allocated(self.device)/1e9
                    wandb.log({
                        'epoch_loss': avg_loss,
                        'epoch_peak_gpu_memory': peak_memory
                    })
                    logging.info("GPU Memory: %.2f GB (Peak: %.2f GB)", memory, peak_memory)
                    
        self.is_fitted = True
        if use_wandb:
            wandb.finish()
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to transform
            
        Returns:
        --------
        X_new : array-like of shape (n_samples, n_components)
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")
            
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            X_reduced = self.model(X)
            
        return X_reduced.cpu().numpy()
    
    def fit_transform(self, X, verbose=True, low_memory=False):
        """
        Fit the model with X and apply dimensionality reduction on X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_new : array-like of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X, verbose=verbose, low_memory=low_memory)
        return self.transform(X)
    
    def save(self, path):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
            
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'n_components': self.n_components,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'a': self.a,
            'b': self.b,
            'correlation_weight': self.correlation_weight,
            'use_batchnorm': self.use_batchnorm,
            'use_dropout': self.use_dropout,
            'clip_grad_norm': self.clip_grad_norm,
            'clip_grad_value': self.clip_grad_value
        }
        
        torch.save(save_dict, path)
        
    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load a saved model.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        device : str
            Device to load the model to
            
        Returns:
        --------
        model : ParametricUMAP
            Loaded model
        """
        save_dict = torch.load(path, map_location=device)
        
        # Create instance with saved parameters
        instance = cls(
            n_components=save_dict['n_components'],
            hidden_dim=save_dict['hidden_dim'],
            n_layers=save_dict['n_layers'],
            a=save_dict['a'],
            b=save_dict['b'],
            correlation_weight=save_dict['correlation_weight'],
            device=device,
            use_batchnorm=save_dict['use_batchnorm'],
            use_dropout=save_dict['use_dropout'],
            clip_grad_norm=save_dict['clip_grad_norm'],
            clip_grad_value=save_dict['clip_grad_value']
        )
        
        # Initialize model architecture
        instance._init_model(input_dim=save_dict['model_state_dict']['model.0.weight'].shape[1])
        
        # Load state dict
        instance.model.load_state_dict(save_dict['model_state_dict'])
        instance.is_fitted = True
        
        return instance

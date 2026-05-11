"""
Experiment configuration system for Parametric UMAP.

Supports:
- YAML config files for reproducible experiments
- CLI overrides for quick iteration
- Wandb integration for tracking
- File-based result logging alongside wandb
- Architecture sweeps for scaling law exploration
"""

import yaml
import json
import os
import copy
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime


def _coerce_override_value(value: str, current: Any) -> Any:
    """Coerce CLI override strings to the type of the existing config value."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean override value: {value!r}")

    if current is not None:
        return type(current)(value)

    return value


@dataclass
class DataConfig:
    """Where the data comes from and how to subsample it."""
    source: str = "synthetic"       # "synthetic", "h5", "memmap", "lancedb"
    # For h5
    h5_path: str = ""
    h5_dataset: str = "embeddings"
    reference_umap_path: str = ""
    # For memmap
    memmap_dirs: List[str] = field(default_factory=list)
    input_dim: int = 384
    # For lancedb
    lancedb_path: str = ""
    lancedb_table: str = "scopes-001"
    lancedb_columns: List[str] = field(default_factory=lambda: ["vector"])
    # Subsampling
    n_samples: Optional[int] = None  # None = use all data
    random_seed: int = 42
    # Precomputed graph
    precomputed_p_sym_path: Optional[str] = None
    precomputed_negatives_path: Optional[str] = None
    precomputed_edges_path: Optional[str] = None
    precomputed_index_path: Optional[str] = None
    n_neighbors: int = 15


@dataclass
class ModelConfig:
    """Architecture and UMAP hyperparameters."""
    architecture: str = "mlp"
    n_components: int = 2
    hidden_dim: int = 512
    n_layers: int = 3
    use_batchnorm: bool = False
    use_dropout: bool = False
    dropout_prob: float = 0.5
    # UMAP curve parameters
    a: float = 1.0
    b: float = 1.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    n_epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    correlation_weight: float = 0.1
    pos_ratio: float = 0.5
    clip_grad_norm: float = 1.0
    clip_grad_value: Optional[float] = None
    low_memory: bool = False
    resample_negatives: bool = False
    n_processes: int = 6
    device: Optional[str] = None  # Auto-detect
    verbose: bool = True
    correlation_distance_transform: str = "raw"
    lr_schedule: str = "plateau"
    warmup_steps: int = 0
    total_steps_estimate: int = 0
    use_amp: bool = True
    positive_target_mode: str = "probability"  # "probability" or "binary"


@dataclass
class EvalConfig:
    """What to measure and how."""
    eval_every_n_epochs: int = 0         # 0 = only at end
    metrics: List[str] = field(default_factory=lambda: [
        "trustworthiness", "distance_correlation", "knn_preservation"
    ])
    knn_k: int = 10
    metric_sample_size: int = 5000       # Subsample for fast metric computation
    compare_umap: bool = False           # Run standard UMAP as baseline
    umap_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"
    })


@dataclass
class LoggingConfig:
    """Where results go."""
    use_wandb: bool = False
    wandb_project: str = "parametric-umap"
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_group: Optional[str] = None     # Group related runs (e.g. a sweep)
    # File-based logging
    results_dir: str = "experiments/results"
    save_model: bool = True
    save_embeddings: bool = False         # Save transformed embeddings


@dataclass
class ExperimentConfig:
    """Top-level config combining all sub-configs."""
    name: str = "unnamed"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        return cls(
            name=d.get('name', 'unnamed'),
            description=d.get('description', ''),
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            train=TrainConfig(**d.get('train', {})),
            eval=EvalConfig(**d.get('eval', {})),
            logging=LoggingConfig(**d.get('logging', {})),
        )

    def config_hash(self) -> str:
        """Short hash of the config for deduplication."""
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:8]

    def run_dir(self) -> str:
        """Directory for this run's results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.logging.results_dir, f"{self.name}_{timestamp}_{self.config_hash()}")

    def apply_overrides(self, overrides: dict):
        """Apply dot-notation overrides like {'train.batch_size': 1024}."""
        for key, value in overrides.items():
            parts = key.split('.')
            if len(parts) == 2:
                section, param = parts
                sub = getattr(self, section, None)
                if sub is not None and hasattr(sub, param):
                    # Coerce types
                    current = getattr(sub, param)
                    setattr(sub, param, _coerce_override_value(value, current))
                else:
                    raise ValueError(f"Unknown config key: {key}")
            else:
                raise ValueError(f"Override key must be section.param, got: {key}")


def load_config(path: str, overrides: Optional[dict] = None) -> ExperimentConfig:
    """Load a YAML config file with optional CLI overrides."""
    config = ExperimentConfig.from_yaml(path)
    if overrides:
        config.apply_overrides(overrides)
    return config


def generate_sweep_configs(base_config: ExperimentConfig, sweep: Dict[str, list]) -> List[ExperimentConfig]:
    """
    Generate configs for a parameter sweep.

    Example:
        sweep = {
            'model.hidden_dim': [256, 512, 1024, 2048],
            'model.n_layers': [2, 3, 4, 6],
        }
    Produces the cartesian product of all parameter values.
    """
    import itertools

    keys = list(sweep.keys())
    values = list(sweep.values())
    configs = []

    for combo in itertools.product(*values):
        cfg = ExperimentConfig.from_dict(base_config.to_dict())
        overrides = dict(zip(keys, combo))
        cfg.apply_overrides(overrides)
        # Name the run after the swept parameters
        param_str = "_".join(f"{k.split('.')[-1]}{v}" for k, v in overrides.items())
        cfg.name = f"{base_config.name}__{param_str}"
        configs.append(cfg)

    return configs

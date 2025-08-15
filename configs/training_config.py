"""
Training configuration for Metamon replication
Critical parameters based on research paper values
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Transformer model configuration"""
    # Architecture (200M parameters to match Metamon)
    hidden_dim: int = 1024
    num_layers: int = 24  
    num_heads: int = 16
    sequence_length: int = 2048
    vocab_size: int = 50000  # Action + observation vocabulary
    action_vocab_size: int = 10  # 4 moves + 6 switches for Pokemon battles
    
    # Dropout and regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    use_bias: bool = True
    
    # Activation function
    activation: str = "gelu"
    
    def get_param_count(self) -> int:
        """Estimate parameter count"""
        # Rough calculation for transformer parameters
        embed_params = self.vocab_size * self.hidden_dim
        layer_params = self.num_layers * (
            4 * self.hidden_dim * self.hidden_dim +  # Attention
            8 * self.hidden_dim * self.hidden_dim    # FFN
        )
        return embed_params + layer_params

@dataclass  
class TrainingConfig:
    """Training hyperparameters"""
    # Batch sizes
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 1024  # batch_size * grad_accum
    
    # Learning rates
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10000
    
    # Training steps
    total_steps: int = 1000000
    bc_steps: int = 200000      # Behavior cloning phase
    rl_steps: int = 800000      # RL fine-tuning phase
    
    # Evaluation
    eval_every: int = 5000
    save_every: int = 10000
    log_every: int = 100
    
    # Optimization
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

@dataclass
class OfflineRLConfig:
    """Offline RL specific parameters"""
    # Algorithm type
    algorithm: str = "filtered_bc_then_ppo"
    
    # PPO parameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # KL penalty (to prevent forgetting BC policy)
    kl_weight: float = 0.1
    target_kl: float = 0.05
    
    # Advantage estimation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Value function
    value_clip: bool = True
    normalize_advantages: bool = True
    
    # Data filtering
    filter_low_win_rate: bool = True
    min_win_rate_threshold: float = 0.3
    
@dataclass
class DataConfig:
    """Dataset configuration"""
    # Paths
    dataset_name: str = "milkkarten/pokechamp"
    raw_data_path: str = "data/pokechamp_raw"
    processed_data_path: str = "data/processed_replays"
    self_play_data_path: str = "data/self_play"
    
    # Processing
    max_sequence_length: int = 2048
    validation_split: float = 0.05
    test_split: float = 0.02
    
    # Data loading
    num_workers: int = 32
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Filtering
    min_turns: int = 10
    max_turns: int = 300
    filter_timeouts: bool = True
    filter_forfeits: bool = True

@dataclass
class InfrastructureConfig:
    """Infrastructure and hardware configuration"""
    # Hardware
    num_gpus: int = 8
    gpu_memory_fraction: float = 0.9
    
    # Distributed training
    backend: str = "nccl"
    find_unused_parameters: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 5
    monitor_metric: str = "val_win_rate"
    
    # Logging
    log_dir: str = "logs"
    wandb_project: str = "pokeagent-2000"
    wandb_entity: str = None  # Set your wandb username
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False  # Set True for debugging

@dataclass
class InferenceConfig:
    """Inference optimization configuration"""
    # Speed optimizations
    use_torchscript: bool = True
    use_half_precision: bool = True
    use_cache: bool = True
    cache_size: int = 10000
    
    # Sampling
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # Time constraints
    max_time_per_move: float = 9.0  # seconds
    early_stop_threshold: float = 0.95  # confidence
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.offline_rl = OfflineRLConfig()
        self.data = DataConfig()
        self.infrastructure = InfrastructureConfig()
        self.inference = InferenceConfig()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "offline_rl": self.offline_rl.__dict__,
            "data": self.data.__dict__,
            "infrastructure": self.infrastructure.__dict__,
            "inference": self.inference.__dict__,
        }
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        # Update configurations
        for key, value in data.items():
            if hasattr(config, key):
                for k, v in value.items():
                    setattr(getattr(config, key), k, v)
        return config

# Global configuration instance
CONFIG = Config()

# Print model size for verification
if __name__ == "__main__":
    print(f"Model parameter count: {CONFIG.model.get_param_count():,}")
    print(f"Effective batch size: {CONFIG.training.effective_batch_size}")
    print(f"Total training steps: {CONFIG.training.total_steps:,}")
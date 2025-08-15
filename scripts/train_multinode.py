#!/usr/bin/env python3
"""
Multi-Node Distributed Training for Pokemon Agent
Scalable training across arbitrary nodes and GPUs
"""

import os
import sys
import json
import time
import signal
import socket
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from models.metamon_transformer import MetamonTransformer, ModelConfig
from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig, collate_real_data
from configs.training_config import TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s'
)

class RankFilter(logging.Filter):
    """Add rank information to log records"""
    def filter(self, record):
        record.rank = dist.get_rank() if dist.is_initialized() else 0
        return True

logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Node/GPU configuration
    nodes: int = 1
    gpus_per_node: int = 4
    
    # Training configuration
    batch_size_per_gpu: int = 256
    gradient_accumulation_steps: int = 1
    max_steps: int = 200000
    time_limit_hours: float = 6.0
    checkpoint_minutes: int = 15
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Data
    data_dir: str = "data/pokechamp/processed"
    max_samples: Optional[int] = None
    
    # Output
    output_dir: str = "checkpoints"
    log_every: int = 100
    eval_every: int = 5000
    
    # Resume
    resume: bool = False
    resume_checkpoint: Optional[str] = None
    
    # Debugging
    dry_run: bool = False
    mock_data: bool = False
    debug_mode: bool = False
    
    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node
    
    @property 
    def total_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.world_size * self.gradient_accumulation_steps
    
    def scale_learning_rate(self) -> float:
        """Scale learning rate based on total batch size"""
        # Linear scaling rule with sqrt adjustment for very large batches
        base_batch_size = 256
        scale_factor = self.total_batch_size / base_batch_size
        
        if scale_factor > 16:
            # Use sqrt scaling for very large batches to avoid instability
            scale_factor = math.sqrt(scale_factor)
        
        return self.learning_rate * scale_factor

class GracefulShutdown:
    """Handle graceful shutdown for distributed training"""
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Shutdown signal {signum} received")
        self.shutdown_requested = True
    
    def should_stop(self) -> bool:
        return self.shutdown_requested

class DistributedTrainer:
    """Distributed trainer for Pokemon agent"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.global_rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.start_time = time.time()
        
        # Graceful shutdown
        self.shutdown_handler = GracefulShutdown()
        
        # Initialize distributed training
        self._init_distributed()
        
        # Setup device
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.device)
        
        # Create directories
        self.output_dir = Path(config.output_dir)
        if self.is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        
        # Synchronize all processes
        dist.barrier()
    
    def _init_distributed(self):
        """Initialize distributed training"""
        if not dist.is_initialized():
            # Get distributed configuration from environment
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '29500')
            
            logger.info(f"Initializing distributed training: {master_addr}:{master_port}")
            logger.info(f"World size: {self.world_size}, Global rank: {self.global_rank}, Local rank: {self.local_rank}")
            
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_addr}:{master_port}',
                world_size=self.world_size,
                rank=self.global_rank
            )
            
            logger.info("Distributed training initialized successfully")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.global_rank == 0
    
    def setup_model(self):
        """Setup model for distributed training"""
        logger.info("Setting up model...")
        
        # Create model
        model_config = ModelConfig()
        self.model = MetamonTransformer(model_config)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.scale_learning_rate(),
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )
        
        # Setup mixed precision
        if self.config.use_amp:
            self.scaler = GradScaler()
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model setup complete: {param_count:,} parameters")
        logger.info(f"Scaled learning rate: {self.config.scale_learning_rate():.2e}")
    
    def setup_data(self):
        """Setup data loaders for distributed training"""
        logger.info("Setting up data loaders...")
        
        # Create dataset config
        dataset_config = DatasetConfig(
            data_dir=self.config.data_dir,
            max_samples=self.config.max_samples,
            validation_split=0.05,
            test_split=0.02,
            cache_size=5000,  # Smaller cache per GPU
            use_memory_mapping=True
        )
        
        # Use mock data if specified
        if self.config.mock_data or self.config.dry_run:
            dataset_config.data_dir = "mock_data"  # Will trigger mock data creation
        
        # Create datasets
        train_dataset = RealPokemonDataset(dataset_config, split='train')
        val_dataset = RealPokemonDataset(dataset_config, split='val')
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False
        )
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=min(4, os.cpu_count() // self.config.gpus_per_node),
            pin_memory=True,
            collate_fn=collate_real_data,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=val_sampler,
            num_workers=min(2, os.cpu_count() // self.config.gpus_per_node),
            pin_memory=True,
            collate_fn=collate_real_data,
            persistent_workers=True
        )
        
        logger.info(f"Data setup complete:")
        logger.info(f"  Train samples: {len(train_dataset)} ({len(self.train_loader)} batches)")
        logger.info(f"  Val samples: {len(val_dataset)} ({len(self.val_loader)} batches)")
        logger.info(f"  Total batch size: {self.config.total_batch_size}")
    
    def save_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Save training checkpoint (only main process)"""
        if not self.is_main_process():
            return
        
        # Get model state dict (unwrap DDP)
        model_state_dict = self.model.module.state_dict()
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': asdict(self.config),
            'world_size': self.world_size,
            'training_time': time.time() - self.start_time,
            'timestamp': time.time()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint"""
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            # Load scaler if available
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            logger.info(f"Checkpoint loaded: step={self.step}, epoch={self.epoch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint to resume from"""
        if self.config.resume_checkpoint:
            checkpoint_path = Path(self.config.resume_checkpoint)
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        # Look for latest checkpoint in output directory
        checkpoint_pattern = "checkpoint_step_*.pt"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))
        
        if checkpoints:
            # Sort by step number
            latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            return str(latest)
        
        return None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = {k: v.to(self.device, non_blocking=True) 
                    for k, v in batch['input_ids'].items()}
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Forward pass
        if self.config.use_amp:
            with autocast():
                outputs = self.model(input_ids)
                loss = nn.functional.cross_entropy(outputs['policy_logits'], labels)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(input_ids)
            loss = nn.functional.cross_entropy(outputs['policy_logits'], labels)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(outputs['policy_logits'], dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'accuracy': accuracy.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping"""
        if self.config.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = {k: v.to(self.device, non_blocking=True) 
                           for k, v in batch['input_ids'].items()}
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(input_ids)
                loss = nn.functional.cross_entropy(outputs['policy_logits'], labels)
                
                predictions = torch.argmax(outputs['policy_logits'], dim=-1)
                accuracy = (predictions == labels).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
                
                # Limit validation time
                if num_batches >= 100:
                    break
        
        # Average across all processes
        if dist.is_initialized():
            metrics = torch.tensor([total_loss, total_accuracy, num_batches], 
                                 device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_accuracy, num_batches = metrics.cpu().numpy()
        
        return {
            'val_loss': total_loss / max(num_batches, 1),
            'val_accuracy': total_accuracy / max(num_batches, 1)
        }
    
    def get_time_remaining(self) -> float:
        """Get remaining time in seconds"""
        elapsed = time.time() - self.start_time
        remaining = self.config.time_limit_hours * 3600 - elapsed
        return max(0, remaining)
    
    def should_checkpoint(self, last_checkpoint_time: float) -> bool:
        """Check if we should save a checkpoint"""
        time_since_checkpoint = time.time() - last_checkpoint_time
        checkpoint_interval = self.config.checkpoint_minutes * 60
        
        return (time_since_checkpoint >= checkpoint_interval or 
                self.shutdown_handler.should_stop() or
                self.get_time_remaining() < 300)  # 5 minutes remaining
    
    def train(self):
        """Main training loop"""
        logger.info("Starting distributed training...")
        
        # Setup model and data
        self.setup_model()
        self.setup_data()
        
        # Resume from checkpoint if available
        if self.config.resume:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
        
        # Initialize wandb (main process only)
        if self.is_main_process() and not self.config.dry_run:
            wandb.init(
                project="pokeagent-2000-distributed",
                name=f"multinode_{self.world_size}gpu_{int(time.time())}",
                config=asdict(self.config)
            )
        
        # Training loop
        last_checkpoint_time = time.time()
        accumulated_metrics = {'loss': 0.0, 'accuracy': 0.0, 'count': 0}
        
        logger.info(f"Training from step {self.step} to {self.config.max_steps}")
        logger.info(f"Time limit: {self.config.time_limit_hours} hours")
        
        while self.step < self.config.max_steps:
            # Check time and shutdown conditions
            if (self.get_time_remaining() < 600 or  # 10 minutes remaining
                self.shutdown_handler.should_stop()):
                logger.info("Stopping training due to time limit or shutdown signal")
                break
            
            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(self.epoch)
            
            for batch_idx, batch in enumerate(self.train_loader):
                if self.step >= self.config.max_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                accumulated_metrics['loss'] += metrics['loss']
                accumulated_metrics['accuracy'] += metrics['accuracy']
                accumulated_metrics['count'] += 1
                
                # Optimizer step
                if ((self.step + 1) % self.config.gradient_accumulation_steps == 0 or 
                    self.step + 1 == self.config.max_steps):
                    self.optimizer_step()
                
                self.step += 1
                
                # Logging
                if self.step % self.config.log_every == 0 and self.is_main_process():
                    avg_loss = accumulated_metrics['loss'] / accumulated_metrics['count']
                    avg_acc = accumulated_metrics['accuracy'] / accumulated_metrics['count']
                    
                    elapsed = time.time() - self.start_time
                    remaining = self.get_time_remaining() / 3600
                    
                    logger.info(
                        f"Step {self.step}/{self.config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Acc: {avg_acc:.4f} | "
                        f"LR: {metrics['learning_rate']:.2e} | "
                        f"Time: {elapsed/3600:.1f}h | "
                        f"Remaining: {remaining:.1f}h"
                    )
                    
                    if not self.config.dry_run:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/accuracy': avg_acc,
                            'train/learning_rate': metrics['learning_rate'],
                            'step': self.step,
                            'epoch': self.epoch,
                            'time_remaining_hours': remaining
                        })
                    
                    # Reset accumulated metrics
                    accumulated_metrics = {'loss': 0.0, 'accuracy': 0.0, 'count': 0}
                
                # Validation and checkpointing
                if (self.step % self.config.eval_every == 0 or 
                    self.should_checkpoint(last_checkpoint_time)):
                    
                    # Validation
                    val_metrics = self.validate()
                    
                    if self.is_main_process():
                        logger.info(
                            f"Validation | "
                            f"Loss: {val_metrics['val_loss']:.4f} | "
                            f"Acc: {val_metrics['val_accuracy']:.4f}"
                        )
                        
                        if not self.config.dry_run:
                            wandb.log({
                                'val/loss': val_metrics['val_loss'],
                                'val/accuracy': val_metrics['val_accuracy'],
                                'step': self.step
                            })
                    
                    # Checkpoint
                    is_best = val_metrics['val_accuracy'] > self.best_val_acc
                    if is_best:
                        self.best_val_acc = val_metrics['val_accuracy']
                        if self.is_main_process():
                            logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
                    
                    if self.is_main_process():
                        checkpoint_path = self.output_dir / f"checkpoint_step_{self.step}.pt"
                        self.save_checkpoint(str(checkpoint_path), is_best)
                        last_checkpoint_time = time.time()
                    
                    # Synchronize all processes
                    dist.barrier()
                
                # Check for early stopping conditions
                if self.shutdown_handler.should_stop():
                    logger.info("Shutdown requested, stopping training")
                    break
            
            self.epoch += 1
        
        # Final checkpoint
        if self.is_main_process():
            final_checkpoint = self.output_dir / f"final_checkpoint_step_{self.step}.pt"
            self.save_checkpoint(str(final_checkpoint))
        
        # Cleanup
        if self.is_main_process() and not self.config.dry_run:
            wandb.finish()
        
        dist.destroy_process_group()
        
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time/3600:.1f} hours")

def main():
    parser = argparse.ArgumentParser(description="Multi-node distributed training")
    
    # Node/GPU configuration
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node")
    
    # Training configuration
    parser.add_argument("--batch-size-per-gpu", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--max-steps", type=int, default=200000, help="Maximum training steps")
    parser.add_argument("--time-limit-hours", type=float, default=6.0, help="Time limit in hours")
    parser.add_argument("--checkpoint-minutes", type=int, default=15, help="Checkpoint frequency")
    
    # Optimization
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=10000, help="Warmup steps")
    
    # Data
    parser.add_argument("--data-dir", default="data/pokechamp/processed", help="Data directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use")
    
    # Output
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--log-every", type=int, default=100, help="Log frequency")
    parser.add_argument("--eval-every", type=int, default=5000, help="Evaluation frequency")
    
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-checkpoint", help="Specific checkpoint to resume from")
    
    # Debugging
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--mock-data", action="store_true", help="Use mock data")
    parser.add_argument("--debug-mode", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DistributedConfig(
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        time_limit_hours=args.time_limit_hours,
        checkpoint_minutes=args.checkpoint_minutes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        log_every=args.log_every,
        eval_every=args.eval_every,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        dry_run=args.dry_run,
        mock_data=args.mock_data,
        debug_mode=args.debug_mode
    )
    
    # Create trainer and run
    trainer = DistributedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Robust Behavior Cloning Training Script with Checkpoint Resume
Enhanced for interrupted training sessions
"""

import os
import sys
import json
import pickle
import argparse
import wandb
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.metamon_transformer import MetamonTransformer, ModelConfig, StateEncoder
from configs.training_config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GracefulKiller:
    """Handle SIGTERM and SIGINT gracefully"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.info(f"Received signal {signum}, will save checkpoint and exit...")
        self.kill_now = True

class PokemonBattleDataset(Dataset):
    """Simplified dataset for quick training"""
    
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.encoder = StateEncoder()
        
        # For now, create dummy data for testing
        self.num_samples = max_samples or 10000
        logger.info(f"Creating {self.num_samples} dummy training samples")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a dummy training example"""
        # Create dummy encoded state
        encoded_state = self.encoder.encode_battle_state(None)
        
        # Random target action (use model's action vocab size)
        target_action = torch.randint(0, 10, (1,)).long()  # 4 moves + 6 switches
        
        return {
            'input_ids': encoded_state,
            'labels': target_action,
            'win_label': torch.tensor(0.5, dtype=torch.float)
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    input_ids = {}
    labels = []
    win_labels = []
    
    for example in batch:
        labels.append(example['labels'])
        win_labels.append(example['win_label'])
        
        for key, value in example['input_ids'].items():
            if key not in input_ids:
                input_ids[key] = []
            input_ids[key].append(value)
    
    for key in input_ids:
        input_ids[key] = torch.stack(input_ids[key], dim=0)
    
    labels = torch.stack(labels, dim=0).squeeze(-1)
    win_labels = torch.stack(win_labels, dim=0)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'win_labels': win_labels
    }

class RobustBCTrainer:
    """Robust BC trainer with checkpoint resume"""
    
    def __init__(self, 
                 model: MetamonTransformer,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 config: CONFIG,
                 output_dir: str = "checkpoints"):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.setup_training()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Data loaders
        self.setup_dataloaders()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.start_time = time.time()
        
        # Graceful shutdown handler
        self.killer = GracefulKiller()
        
        # Resume state file
        self.state_file = self.output_dir / "training_state.json"
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.bc_steps
        )
    
    def setup_dataloaders(self):
        """Setup data loaders"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=min(8, self.config.data.num_workers),  # Limit for stability
            pin_memory=self.config.data.pin_memory,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=min(4, self.config.data.num_workers),
            pin_memory=self.config.data.pin_memory,
            collate_fn=collate_fn
        )
    
    def save_checkpoint(self, path: str, is_best: bool = False, emergency: bool = False):
        """Save comprehensive checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_acc': self.best_val_acc,
            'config': self.config.to_dict(),
            'training_time': time.time() - self.start_time,
            'timestamp': time.time(),
            'emergency_save': emergency
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"{'🚨 Emergency' if emergency else '💾'} Checkpoint saved: {path}")
        
        # Save best model
        if is_best:
            best_path = str(Path(path).parent / "best_bc_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Best model saved: {best_path}")
        
        # Save training state for resume
        training_state = {
            'step': self.step,
            'epoch': self.epoch,
            'best_val_acc': self.best_val_acc,
            'last_checkpoint': str(path),
            'training_time': time.time() - self.start_time,
            'completed': self.step >= self.config.training.bc_steps
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(training_state, f, indent=2)
    
    def load_checkpoint(self, path: str) -> bool:
        """Load checkpoint and resume training"""
        try:
            logger.info(f"Loading checkpoint: {path}")
            checkpoint = torch.load(path, map_location='cpu')
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            logger.info(f"✅ Resumed from step {self.step}, epoch {self.epoch}")
            logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint to resume from"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                last_checkpoint = state.get('last_checkpoint')
                if last_checkpoint and Path(last_checkpoint).exists():
                    return last_checkpoint
            except Exception as e:
                logger.warning(f"Error reading state file: {e}")
        
        # Fallback: find latest checkpoint by timestamp
        checkpoints = list(self.output_dir.glob("bc_checkpoint_step_*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            return str(latest)
        
        return None
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with error handling"""
        self.model.train()
        
        try:
            # Move to device
            input_ids = {k: v.to(self.device) for k, v in batch['input_ids'].items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids)
                    loss = F.cross_entropy(outputs['policy_logits'], labels)
            else:
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs['policy_logits'], labels)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(outputs['policy_logits'], dim=-1)
                accuracy = (predictions == labels).float().mean()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0, 'learning_rate': 0.0}
    
    def validate(self) -> Dict[str, float]:
        """Validation with error handling"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = {k: v.to(self.device) for k, v in batch['input_ids'].items()}
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids)
                    loss = F.cross_entropy(outputs['policy_logits'], labels)
                    
                    predictions = torch.argmax(outputs['policy_logits'], dim=-1)
                    accuracy = (predictions == labels).float().mean()
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    num_batches += 1
                    
                    # Limit validation batches for speed
                    if num_batches >= 50:
                        break
            
            return {
                'val_loss': total_loss / max(num_batches, 1),
                'val_accuracy': total_accuracy / max(num_batches, 1)
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {'val_loss': float('inf'), 'val_accuracy': 0.0}
    
    def train(self):
        """Main training loop with robust checkpoint handling"""
        logger.info("🚀 Starting robust BC training...")
        logger.info(f"Target steps: {self.config.training.bc_steps}")
        logger.info(f"Current step: {self.step}")
        
        # Calculate remaining steps
        remaining_steps = self.config.training.bc_steps - self.step
        
        if remaining_steps <= 0:
            logger.info("✅ Training already complete!")
            return
        
        logger.info(f"Remaining steps: {remaining_steps}")
        
        # Training loop
        step_count = 0
        last_save_time = time.time()
        save_interval = 300  # Save every 5 minutes
        
        try:
            while self.step < self.config.training.bc_steps and not self.killer.kill_now:
                
                for batch in self.train_loader:
                    if self.step >= self.config.training.bc_steps or self.killer.kill_now:
                        break
                    
                    # Training step
                    metrics = self.train_step(batch)
                    
                    # Log metrics
                    if self.step % self.config.training.log_every == 0:
                        elapsed = time.time() - self.start_time
                        logger.info(f"Step {self.step}/{self.config.training.bc_steps}: "
                                   f"Loss: {metrics['loss']:.4f}, "
                                   f"Acc: {metrics['accuracy']:.4f}, "
                                   f"Time: {elapsed/60:.1f}m")
                        
                        try:
                            wandb.log({
                                'train/loss': metrics['loss'],
                                'train/accuracy': metrics['accuracy'],
                                'train/learning_rate': metrics['learning_rate'],
                                'step': self.step,
                                'epoch': self.epoch,
                                'training_time': elapsed
                            })
                        except:
                            pass  # Continue if wandb fails
                    
                    # Periodic validation and checkpointing
                    current_time = time.time()
                    
                    if (self.step % self.config.training.eval_every == 0 or 
                        current_time - last_save_time > save_interval or
                        self.killer.kill_now):
                        
                        # Validation
                        val_metrics = self.validate()
                        
                        logger.info(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                                   f"Acc: {val_metrics['val_accuracy']:.4f}")
                        
                        # Check if best model
                        is_best = val_metrics['val_accuracy'] > self.best_val_acc
                        if is_best:
                            self.best_val_acc = val_metrics['val_accuracy']
                            logger.info(f"🏆 New best validation accuracy: {self.best_val_acc:.4f}")
                        
                        # Save checkpoint
                        checkpoint_path = self.output_dir / f"bc_checkpoint_step_{self.step}.pt"
                        self.save_checkpoint(str(checkpoint_path), is_best, emergency=self.killer.kill_now)
                        
                        last_save_time = current_time
                        
                        # Log validation metrics
                        try:
                            wandb.log({
                                'val/loss': val_metrics['val_loss'],
                                'val/accuracy': val_metrics['val_accuracy'],
                                'val/best_accuracy': self.best_val_acc,
                                'step': self.step
                            })
                        except:
                            pass
                        
                        # Exit if interrupted
                        if self.killer.kill_now:
                            logger.info("🛑 Graceful shutdown initiated")
                            break
                    
                    self.step += 1
                    step_count += 1
                
                self.epoch += 1
        
        except Exception as e:
            logger.error(f"💥 Training error: {e}")
            # Emergency save
            emergency_path = self.output_dir / f"emergency_checkpoint_step_{self.step}.pt"
            self.save_checkpoint(str(emergency_path), emergency=True)
            raise e
        
        # Final save
        if self.step >= self.config.training.bc_steps:
            final_path = self.output_dir / "final_bc_model.pt"
            self.save_checkpoint(str(final_path), is_best=True)
            logger.info("✅ BC training completed successfully!")
        else:
            logger.info(f"⏸️ Training paused at step {self.step}")

def main():
    parser = argparse.ArgumentParser(description="Robust BC Training")
    parser.add_argument("--data-dir", default="data/processed_replays", help="Data directory")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    try:
        wandb.init(
            project="pokeagent-2000",
            name=f"bc_robust_{int(time.time())}",
            config=CONFIG.to_dict(),
            resume="allow"
        )
    except:
        logger.warning("Failed to initialize wandb, continuing without logging")
    
    # Create model using the actual ModelConfig from metamon_transformer
    model_config = ModelConfig()
    model = MetamonTransformer(model_config)
    logger.info(f"Model created: {model.get_parameter_count():,} parameters")
    
    # Create datasets
    if args.debug:
        train_dataset = PokemonBattleDataset("dummy", max_samples=1000)
        val_dataset = PokemonBattleDataset("dummy", max_samples=100)
    else:
        train_dataset = PokemonBattleDataset(args.data_dir, max_samples=100000)
        val_dataset = PokemonBattleDataset(args.data_dir, max_samples=5000)
    
    # Create trainer
    trainer = RobustBCTrainer(model, train_dataset, val_dataset, CONFIG, args.output_dir)
    
    # Resume if requested
    if args.resume:
        latest_checkpoint = trainer.find_latest_checkpoint()
        if latest_checkpoint:
            trainer.load_checkpoint(latest_checkpoint)
        else:
            logger.info("No checkpoint found, starting fresh")
    
    # Start training
    trainer.train()
    
    try:
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()
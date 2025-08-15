#!/usr/bin/env python3
"""
Behavior Cloning Training Script
First stage: Learn to imitate human players from 3.5M replay dataset
Goal: Achieve ~60% move prediction accuracy
"""

import os
import sys
import json
import pickle
import argparse
import wandb
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

class PokemonBattleDataset(Dataset):
    """Dataset for Pokemon battle replays"""
    
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.encoder = StateEncoder()
        
        # Find all replay files
        self.replay_files = list(self.data_dir.glob("*.pkl"))
        
        if max_samples:
            self.replay_files = self.replay_files[:max_samples]
        
        logger.info(f"Found {len(self.replay_files)} replay files")
        
        # Cache for loaded data
        self.cache = {}
        
    def __len__(self):
        return len(self.replay_files)
    
    def __getitem__(self, idx):
        """Get a training example"""
        if idx in self.cache:
            return self.cache[idx]
        
        replay_file = self.replay_files[idx]
        
        try:
            with open(replay_file, 'rb') as f:
                replay_data = pickle.load(f)
            
            # Process the replay data
            # This is simplified - real implementation needs full state encoding
            example = self.process_replay(replay_data)
            
            # Cache the result
            if len(self.cache) < 1000:  # Limit cache size
                self.cache[idx] = example
            
            return example
            
        except Exception as e:
            logger.warning(f"Failed to load {replay_file}: {e}")
            # Return a dummy example
            return self.get_dummy_example()
    
    def process_replay(self, replay_data: List[Dict]) -> Dict:
        """
        Process a single replay into training format
        
        Each replay contains a sequence of states and actions
        We need to predict the action given the state
        """
        if not replay_data:
            return self.get_dummy_example()
        
        # Use the last state as our training example
        # In practice, you'd want to use multiple states from the replay
        last_example = replay_data[-1]
        
        battle_state = last_example.get('state')
        label = last_example.get('label', 0)  # Win/loss
        
        # Encode the battle state
        # This is a simplified version - needs full implementation
        encoded_state = self.encoder.encode_battle_state(battle_state)
        
        # Create target action (simplified)
        # In practice, this would be the actual action taken by the human player
        target_action = torch.randint(0, CONFIG.model.action_vocab_size, (1,)).long()
        
        return {
            'input_ids': encoded_state,
            'labels': target_action,
            'win_label': torch.tensor(label, dtype=torch.float)
        }
    
    def get_dummy_example(self) -> Dict:
        """Get a dummy example for error cases"""
        dummy_state = self.encoder.encode_battle_state(None)
        
        return {
            'input_ids': dummy_state,
            'labels': torch.tensor([0], dtype=torch.long),
            'win_label': torch.tensor(0.0, dtype=torch.float)
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    
    # Extract components
    input_ids = {}
    labels = []
    win_labels = []
    
    for example in batch:
        # Collect labels
        labels.append(example['labels'])
        win_labels.append(example['win_label'])
        
        # Collect input_ids
        for key, value in example['input_ids'].items():
            if key not in input_ids:
                input_ids[key] = []
            input_ids[key].append(value)
    
    # Stack tensors
    for key in input_ids:
        input_ids[key] = torch.stack(input_ids[key], dim=0)
    
    labels = torch.stack(labels, dim=0).squeeze(-1)
    win_labels = torch.stack(win_labels, dim=0)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'win_labels': win_labels
    }

class BehaviorCloningTrainer:
    """Trainer for behavior cloning phase"""
    
    def __init__(self, 
                 model: MetamonTransformer,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 config: CONFIG):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # Setup scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=config.training.bc_steps
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers // 4,  # Reduce for BC
            pin_memory=config.data.pin_memory,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers // 4,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_fn
        )
        
        # Metrics tracking
        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        input_ids = {k: v.to(self.device) for k, v in batch['input_ids'].items()}
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
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
    
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                input_ids = {k: v.to(self.device) for k, v in batch['input_ids'].items()}
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs['policy_logits'], labels)
                
                # Calculate accuracy
                predictions = torch.argmax(outputs['policy_logits'], dim=-1)
                accuracy = (predictions == labels).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_acc': self.best_val_acc,
            'config': self.config.to_dict()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = str(Path(path).parent / "best_bc_model.pt")
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting behavior cloning training...")
        logger.info(f"Training steps: {self.config.training.bc_steps}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        steps_per_epoch = len(self.train_loader)
        
        for epoch in range(self.config.training.bc_steps // steps_per_epoch + 1):
            self.epoch = epoch
            
            # Training loop
            epoch_metrics = {'loss': [], 'accuracy': []}
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch in progress_bar:
                if self.step >= self.config.training.bc_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                
                # Track metrics
                epoch_metrics['loss'].append(metrics['loss'])
                epoch_metrics['accuracy'].append(metrics['accuracy'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.4f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Log to wandb
                if self.step % self.config.training.log_every == 0:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/accuracy': metrics['accuracy'],
                        'train/learning_rate': metrics['learning_rate'],
                        'step': self.step
                    })
                
                # Validation
                if self.step % self.config.training.eval_every == 0:
                    val_metrics = self.validate()
                    
                    logger.info(f"Step {self.step}: Val Loss: {val_metrics['val_loss']:.4f}, "
                              f"Val Acc: {val_metrics['val_accuracy']:.4f}")
                    
                    # Log validation metrics
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'val/accuracy': val_metrics['val_accuracy'],
                        'step': self.step
                    })
                    
                    # Save best model
                    if val_metrics['val_accuracy'] > self.best_val_acc:
                        self.best_val_acc = val_metrics['val_accuracy']
                        is_best = True
                    else:
                        is_best = False
                    
                    # Save checkpoint
                    if self.step % self.config.training.save_every == 0:
                        checkpoint_path = f"checkpoints/bc_checkpoint_step_{self.step}.pt"
                        self.save_checkpoint(checkpoint_path, is_best)
                
                self.step += 1
                
                if self.step >= self.config.training.bc_steps:
                    break
            
            # End of epoch summary
            avg_loss = np.mean(epoch_metrics['loss'])
            avg_acc = np.mean(epoch_metrics['accuracy'])
            
            logger.info(f"Epoch {epoch} complete: Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
            
            if self.step >= self.config.training.bc_steps:
                break
        
        logger.info("Behavior cloning training complete!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning Training")
    parser.add_argument("--data-dir", required=True, help="Directory with processed replay data")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory for models")
    parser.add_argument("--config-path", help="Path to custom config file")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited data")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=CONFIG.infrastructure.wandb_project,
        entity=CONFIG.infrastructure.wandb_entity,
        name=f"bc_training_{CONFIG.model.hidden_size}M",
        config=CONFIG.to_dict()
    )
    
    # Create model
    model_config = ModelConfig()
    model = MetamonTransformer(model_config)
    
    logger.info(f"Model parameter count: {model.get_parameter_count():,}")
    
    # Load datasets
    logger.info(f"Loading dataset from {args.data_dir}")
    
    if args.debug:
        train_dataset = PokemonBattleDataset(args.data_dir, max_samples=1000)
        val_dataset = PokemonBattleDataset(args.data_dir, max_samples=100)
    else:
        train_dataset = PokemonBattleDataset(args.data_dir)
        val_dataset = PokemonBattleDataset(args.data_dir, max_samples=len(train_dataset) // 20)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create trainer
    trainer = BehaviorCloningTrainer(model, train_dataset, val_dataset, CONFIG)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.step = checkpoint['step']
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_acc = checkpoint['best_val_acc']
        
        if trainer.scaler and 'scaler_state_dict' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Start training
    trainer.train()
    
    # Save final model
    final_path = f"{args.output_dir}/final_bc_model.pt"
    trainer.save_checkpoint(final_path, is_best=True)
    
    wandb.finish()

if __name__ == "__main__":
    main()
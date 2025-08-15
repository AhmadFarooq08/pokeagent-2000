#!/usr/bin/env python3
"""
Offline RL Training Script
Second stage: Optimize for winning, not just imitation
Uses PPO with KL penalty to prevent forgetting behavior cloning policy
"""

import os
import sys
import json
import pickle
import argparse
import wandb
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.metamon_transformer import MetamonTransformer, ModelConfig
from scripts.train_bc import PokemonBattleDataset, collate_fn
from configs.training_config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineRLDataset(Dataset):
    """Dataset for offline RL training with full episode trajectories"""
    
    def __init__(self, data_dir: str, max_episodes: Optional[int] = None):
        self.data_dir = Path(data_dir)
        
        # Find all replay files
        self.replay_files = list(self.data_dir.glob("*.pkl"))
        
        if max_episodes:
            self.replay_files = self.replay_files[:max_episodes]
        
        logger.info(f"Found {len(self.replay_files)} episode files for RL training")
        
        # Load and process episodes
        self.episodes = []
        self.load_episodes()
    
    def load_episodes(self):
        """Load all episodes into memory for RL training"""
        logger.info("Loading episodes for RL training...")
        
        for replay_file in tqdm(self.replay_files, desc="Loading episodes"):
            try:
                with open(replay_file, 'rb') as f:
                    episode_data = pickle.load(f)
                
                # Process episode into RL format
                processed_episode = self.process_episode(episode_data)
                if processed_episode:
                    self.episodes.append(processed_episode)
                    
            except Exception as e:
                logger.warning(f"Failed to load {replay_file}: {e}")
        
        logger.info(f"Loaded {len(self.episodes)} episodes for RL training")
    
    def process_episode(self, episode_data: List[Dict]) -> Optional[Dict]:
        """
        Process episode data into RL training format
        
        Returns:
            episode: {
                'states': List of states,
                'actions': List of actions,
                'rewards': List of rewards,
                'returns': List of discounted returns,
                'advantages': List of advantages,
                'old_log_probs': List of action log probabilities
            }
        """
        if len(episode_data) < 5:  # Too short
            return None
        
        states = []
        actions = []
        rewards = []
        
        # Extract states and actions
        for i, step_data in enumerate(episode_data):
            state = step_data.get('state')
            # In real implementation, extract actual action from state transition
            action = torch.randint(0, CONFIG.model.action_vocab_size, (1,)).item()
            
            states.append(state)
            actions.append(action)
            
            # Reward is 0 for all steps except the last
            if i == len(episode_data) - 1:
                # Final reward based on win/loss
                reward = float(step_data.get('label', 0))
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        # Calculate returns (discounted cumulative rewards)
        returns = self.calculate_returns(rewards, CONFIG.offline_rl.gamma)
        
        # Calculate advantages (simplified - in practice use GAE)
        advantages = self.calculate_advantages(returns)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'returns': returns,
            'advantages': advantages,
            'episode_length': len(states)
        }
    
    def calculate_returns(self, rewards: List[float], gamma: float) -> List[float]:
        """Calculate discounted returns"""
        returns = []
        g = 0.0
        
        # Calculate backwards
        for reward in reversed(rewards):
            g = reward + gamma * g
            returns.insert(0, g)
        
        return returns
    
    def calculate_advantages(self, returns: List[float]) -> List[float]:
        """Calculate advantages (simplified)"""
        # Normalize returns to get advantages
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array) + 1e-8
        
        advantages = (returns_array - mean_return) / std_return
        return advantages.tolist()
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        return self.episodes[idx]

class PPOTrainer:
    """PPO trainer for offline RL"""
    
    def __init__(self,
                 policy_model: MetamonTransformer,
                 bc_reference_model: MetamonTransformer,
                 train_dataset: OfflineRLDataset,
                 val_dataset: OfflineRLDataset,
                 config: CONFIG):
        
        self.policy_model = policy_model
        self.bc_reference_model = bc_reference_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_model.to(self.device)
        self.bc_reference_model.to(self.device)
        
        # Freeze BC reference model
        for param in self.bc_reference_model.parameters():
            param.requires_grad = False
        self.bc_reference_model.eval()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate / 10,  # Lower LR for RL
            weight_decay=config.training.weight_decay,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Create data loader for episodes
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size // 4,  # Smaller batch for episodes
            shuffle=True,
            num_workers=config.data.num_workers // 8,
            pin_memory=False,  # Episodes are complex objects
            collate_fn=self.episode_collate_fn
        )
        
        # Metrics tracking
        self.step = 0
        self.best_win_rate = 0.0
        
    def episode_collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for episode batches"""
        # For now, return first episode (implement proper batching later)
        return batch[0] if batch else {}
    
    def compute_policy_loss(self, states, actions, advantages, old_log_probs):
        """Compute PPO policy loss"""
        
        # Get current policy distribution
        outputs = self.policy_model(states)
        logits = outputs['policy_logits']
        
        # Get log probabilities
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped loss
        clip_range = self.config.offline_rl.clip_range
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        return policy_loss, ratio
    
    def compute_value_loss(self, states, returns):
        """Compute value function loss"""
        outputs = self.policy_model(states)
        values = outputs['value']
        
        # MSE loss for value function
        value_loss = F.mse_loss(values, returns)
        
        return value_loss
    
    def compute_kl_penalty(self, states):
        """Compute KL divergence penalty from BC policy"""
        # Get current policy
        policy_outputs = self.policy_model(states)
        policy_logits = policy_outputs['policy_logits']
        
        # Get BC reference policy
        with torch.no_grad():
            bc_outputs = self.bc_reference_model(states)
            bc_logits = bc_outputs['policy_logits']
        
        # Compute KL divergence
        policy_probs = F.softmax(policy_logits, dim=-1)
        bc_probs = F.softmax(bc_logits, dim=-1)
        
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            bc_probs,
            reduction='batchmean'
        )
        
        return kl_div
    
    def compute_entropy_bonus(self, states):
        """Compute entropy bonus for exploration"""
        outputs = self.policy_model(states)
        logits = outputs['policy_logits']
        
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()
        
        return entropy
    
    def train_step(self, episode: Dict) -> Dict[str, float]:
        """Single PPO training step on an episode"""
        self.policy_model.train()
        
        # Extract episode data
        states = episode['states']
        actions = torch.tensor(episode['actions'], dtype=torch.long, device=self.device)
        advantages = torch.tensor(episode['advantages'], dtype=torch.float, device=self.device)
        returns = torch.tensor(episode['returns'], dtype=torch.float, device=self.device)
        
        # Encode states (simplified)
        from scripts.train_bc import PokemonBattleDataset
        dataset = PokemonBattleDataset("dummy")
        encoded_states = dataset.encoder.encode_battle_state(states[0])  # Use first state
        encoded_states = {k: v.to(self.device) for k, v in encoded_states.items()}
        
        # Get old log probabilities from BC model
        with torch.no_grad():
            bc_outputs = self.bc_reference_model(encoded_states)
            bc_logits = bc_outputs['policy_logits']
            bc_dist = Categorical(logits=bc_logits)
            old_log_probs = bc_dist.log_prob(actions[:1])  # First action only for now
        
        # Multiple PPO epochs per episode
        total_loss = 0.0
        for ppo_epoch in range(self.config.offline_rl.ppo_epochs):
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    # Policy loss
                    policy_loss, ratio = self.compute_policy_loss(
                        encoded_states, actions[:1], advantages[:1], old_log_probs
                    )
                    
                    # Value loss
                    value_loss = self.compute_value_loss(encoded_states, returns[:1])
                    
                    # KL penalty
                    kl_penalty = self.compute_kl_penalty(encoded_states)
                    
                    # Entropy bonus
                    entropy_bonus = self.compute_entropy_bonus(encoded_states)
                    
                    # Total loss
                    loss = (policy_loss + 
                           self.config.offline_rl.value_loss_coef * value_loss +
                           self.config.offline_rl.kl_weight * kl_penalty -
                           self.config.offline_rl.entropy_coef * entropy_bonus)
            else:
                # Policy loss
                policy_loss, ratio = self.compute_policy_loss(
                    encoded_states, actions[:1], advantages[:1], old_log_probs
                )
                
                # Value loss
                value_loss = self.compute_value_loss(encoded_states, returns[:1])
                
                # KL penalty
                kl_penalty = self.compute_kl_penalty(encoded_states)
                
                # Entropy bonus
                entropy_bonus = self.compute_entropy_bonus(encoded_states)
                
                # Total loss
                loss = (policy_loss + 
                       self.config.offline_rl.value_loss_coef * value_loss +
                       self.config.offline_rl.kl_weight * kl_penalty -
                       self.config.offline_rl.entropy_coef * entropy_bonus)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            total_loss += loss.item()
        
        return {
            'total_loss': total_loss / self.config.offline_rl.ppo_epochs,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy_bonus': entropy_bonus.item(),
            'episode_return': returns[0].item(),
            'episode_length': episode['episode_length']
        }
    
    def evaluate_win_rate(self) -> float:
        """Evaluate win rate on validation episodes"""
        self.policy_model.eval()
        
        # This is simplified - in practice, you'd run actual battles
        # For now, estimate win rate based on episode returns
        total_return = 0.0
        num_episodes = 0
        
        for episode in self.val_dataset:
            total_return += episode['returns'][0]  # Final return
            num_episodes += 1
            
            if num_episodes >= 100:  # Limit evaluation size
                break
        
        # Estimate win rate (simplified)
        avg_return = total_return / max(num_episodes, 1)
        estimated_win_rate = max(0.0, min(1.0, (avg_return + 1.0) / 2.0))  # Scale to [0, 1]
        
        return estimated_win_rate
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'best_win_rate': self.best_win_rate,
            'config': self.config.to_dict()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = str(Path(path).parent / "best_rl_model.pt")
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting offline RL training...")
        logger.info(f"Training steps: {self.config.training.rl_steps}")
        logger.info(f"Episodes per batch: {len(self.train_dataset)}")
        
        for step in range(self.config.training.rl_steps):
            self.step = step
            
            # Sample episode batch
            episode_batch = next(iter(self.train_loader))
            
            # Training step
            metrics = self.train_step(episode_batch)
            
            # Log metrics
            if step % self.config.training.log_every == 0:
                logger.info(f"Step {step}: Loss: {metrics['total_loss']:.4f}, "
                          f"Return: {metrics['episode_return']:.4f}")
                
                wandb.log({
                    'rl/total_loss': metrics['total_loss'],
                    'rl/policy_loss': metrics['policy_loss'],
                    'rl/value_loss': metrics['value_loss'],
                    'rl/kl_penalty': metrics['kl_penalty'],
                    'rl/entropy_bonus': metrics['entropy_bonus'],
                    'rl/episode_return': metrics['episode_return'],
                    'rl/episode_length': metrics['episode_length'],
                    'step': step
                })
            
            # Evaluation
            if step % self.config.training.eval_every == 0:
                win_rate = self.evaluate_win_rate()
                
                logger.info(f"Step {step}: Estimated Win Rate: {win_rate:.4f}")
                
                wandb.log({
                    'rl/win_rate': win_rate,
                    'step': step
                })
                
                # Save best model
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    is_best = True
                else:
                    is_best = False
                
                # Save checkpoint
                if step % self.config.training.save_every == 0:
                    checkpoint_path = f"checkpoints/rl_checkpoint_step_{step}.pt"
                    self.save_checkpoint(checkpoint_path, is_best)
        
        logger.info("Offline RL training complete!")
        logger.info(f"Best win rate: {self.best_win_rate:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Offline RL Training")
    parser.add_argument("--data-dir", required=True, help="Directory with processed replay data")
    parser.add_argument("--bc-checkpoint", required=True, help="Path to BC model checkpoint")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=CONFIG.infrastructure.wandb_project,
        entity=CONFIG.infrastructure.wandb_entity,
        name=f"rl_training_{CONFIG.model.hidden_size}M",
        config=CONFIG.to_dict()
    )
    
    # Load BC checkpoint
    logger.info(f"Loading BC checkpoint from {args.bc_checkpoint}")
    bc_checkpoint = torch.load(args.bc_checkpoint, map_location='cpu')
    
    # Create models
    model_config = ModelConfig()
    policy_model = MetamonTransformer(model_config)
    bc_reference_model = MetamonTransformer(model_config)
    
    # Load BC weights
    policy_model.load_state_dict(bc_checkpoint['model_state_dict'])
    bc_reference_model.load_state_dict(bc_checkpoint['model_state_dict'])
    
    logger.info(f"Loaded BC model with {policy_model.get_parameter_count():,} parameters")
    
    # Load datasets
    logger.info(f"Loading dataset from {args.data_dir}")
    
    if args.debug:
        train_dataset = OfflineRLDataset(args.data_dir, max_episodes=100)
        val_dataset = OfflineRLDataset(args.data_dir, max_episodes=20)
    else:
        train_dataset = OfflineRLDataset(args.data_dir)
        val_dataset = OfflineRLDataset(args.data_dir, max_episodes=len(train_dataset) // 10)
    
    logger.info(f"Train episodes: {len(train_dataset)}")
    logger.info(f"Validation episodes: {len(val_dataset)}")
    
    # Create trainer
    trainer = PPOTrainer(policy_model, bc_reference_model, train_dataset, val_dataset, CONFIG)
    
    # Start training
    trainer.train()
    
    # Save final model
    final_path = f"{args.output_dir}/final_rl_model.pt"
    trainer.save_checkpoint(final_path, is_best=True)
    
    wandb.finish()

if __name__ == "__main__":
    main()
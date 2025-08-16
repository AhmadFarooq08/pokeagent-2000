#!/usr/bin/env python3
"""
Mock Pokemon Dataset - ONLY for local testing
Completely separate from production dataset
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPokemonDataset(Dataset):
    """Mock dataset for local testing - NO production use"""
    
    def __init__(self, 
                 num_samples: int = 10000,
                 split: str = 'train',
                 transform: Optional[callable] = None):
        """
        Initialize mock dataset for testing only
        
        Args:
            num_samples: Number of mock samples to generate
            split: 'train', 'val', or 'test'
            transform: Optional data transformation function
        """
        self.split = split
        self.transform = transform
        self.num_samples = num_samples
        
        # Split ratios
        if split == 'train':
            self.samples = self._generate_mock_samples(int(num_samples * 0.8))
        elif split == 'val':
            self.samples = self._generate_mock_samples(int(num_samples * 0.15))
        else:  # test
            self.samples = self._generate_mock_samples(int(num_samples * 0.05))
        
        logger.info(f"Initialized MOCK {split} dataset: {len(self.samples)} samples")
    
    def _generate_mock_samples(self, count: int) -> List[Dict]:
        """Generate mock training samples"""
        samples = []
        
        for i in range(count):
            # Mock game state
            game_state = torch.randint(0, 100, (512,))  # Mock encoded state
            
            # Mock action
            action = random.randint(0, 9)  # 10 possible actions
            
            # Mock reward (binary win/loss)
            reward = random.choice([0.0, 1.0])
            
            sample = {
                'game_state': game_state,
                'action': action,
                'reward': reward,
                'battle_id': f'mock_battle_{i}',
                'turn': random.randint(1, 50)
            }
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Convert to tensors
        result = {
            'input_ids': sample['game_state'],
            'labels': torch.tensor(sample['action'], dtype=torch.long),
            'rewards': torch.tensor(sample['reward'], dtype=torch.float),
            'metadata': {
                'battle_id': sample['battle_id'],
                'turn': sample['turn']
            }
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        total_samples = len(self.samples)
        win_rate = sum(1 for s in self.samples if s['reward'] > 0.5) / total_samples
        
        return {
            'total_samples': total_samples,
            'win_rate': win_rate,
            'avg_turns': sum(s['turn'] for s in self.samples) / total_samples,
            'split': self.split,
            'type': 'MOCK_DATA_FOR_TESTING_ONLY'
        }

def collate_mock_data(batch):
    """Collate function for mock data batches"""
    if not batch:
        return {}
    
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    rewards = torch.stack([item['rewards'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels, 
        'rewards': rewards,
        'metadata': [item['metadata'] for item in batch]
    }
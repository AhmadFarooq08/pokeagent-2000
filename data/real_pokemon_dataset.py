#!/usr/bin/env python3
"""
Real Pokemon Dataset
Efficient dataset class for loading processed Pokemon battle replays
"""

import os
import sys
import json
import pickle
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import mmap
import threading
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class DatasetConfig:
    """Configuration for dataset loading"""
    data_dir: str
    max_samples: Optional[int] = None
    validation_split: float = 0.05
    test_split: float = 0.02
    min_game_length: int = 5
    max_game_length: int = 200
    filter_formats: Optional[List[str]] = None
    balance_outcomes: bool = True
    cache_size: int = 10000
    preload_batches: int = 10
    use_memory_mapping: bool = True

class SampleCache:
    """Thread-safe LRU cache for training samples"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Dict):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest = self.access_order.pop(0)
                    del self.cache[oldest]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

class RealPokemonDataset(Dataset):
    """Dataset for real Pokemon battle data"""
    
    def __init__(self, 
                 config: DatasetConfig, 
                 split: str = 'train',
                 transform: Optional[callable] = None):
        """
        Initialize dataset
        
        Args:
            config: Dataset configuration
            split: 'train', 'val', or 'test'
            transform: Optional data transformation function
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.data_dir = Path(config.data_dir)
        
        # Initialize cache
        self.cache = SampleCache(config.cache_size)
        
        # Load metadata and sample indices
        self.metadata = self._load_metadata()
        self.sample_indices = self._load_sample_indices()
        
        # Filter and split data
        self.sample_indices = self._filter_samples(self.sample_indices)
        self.sample_indices = self._split_data(self.sample_indices)[split]
        
        logger.info(f"Initialized {split} dataset: {len(self.sample_indices)} samples")
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        metadata_path = self.data_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No metadata found, creating default")
            return {
                'total_samples': 0,
                'format': 'parsed_replays',
                'version': '1.0'
            }
    
    def _load_sample_indices(self) -> List[Tuple[str, int]]:
        """Load indices of all available samples"""
        indices = []
        
        # Find all data files - support both JSON and PKL formats
        json_files = list(self.data_dir.glob("*.json"))
        pkl_files = list(self.data_dir.glob("*.pkl"))
        data_files = json_files + pkl_files
        
        if not data_files:
            logger.error(f"No data files found in {self.data_dir}")
            logger.error("Available files: " + str(list(self.data_dir.glob("*"))))
            raise FileNotFoundError(f"No training data found in {self.data_dir}")
        
        # Load sample indices from each file
        for data_file in data_files:
            try:
                if data_file.suffix == '.pkl':
                    # For pickle files, we need to load to get count
                    with open(data_file, 'rb') as f:
                        samples = pickle.load(f)
                        
                    for i, sample in enumerate(samples):
                        indices.append((str(data_file), i))
                        
                elif data_file.suffix == '.json':
                    # For JSON files, load and count
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for i, sample in enumerate(data):
                            indices.append((str(data_file), i))
                    else:
                        # Single sample file
                        indices.append((str(data_file), 0))
                    
            except Exception as e:
                logger.warning(f"Failed to load indices from {data_file}: {e}")
        
        logger.info(f"Loaded {len(indices)} sample indices")
        return indices
    
    
    def _filter_samples(self, indices: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Filter samples based on configuration"""
        if not self.config.filter_formats and not self.config.max_samples:
            return indices
        
        filtered_indices = []
        checked_samples = 0
        
        for file_path, sample_idx in indices:
            # Load sample to check filters
            sample = self._load_sample(file_path, sample_idx)
            if sample is None:
                continue
            
            checked_samples += 1
            
            # Filter by format
            if self.config.filter_formats:
                sample_format = sample.get('format', 'unknown')
                if sample_format not in self.config.filter_formats:
                    continue
            
            # Filter by game length
            game_length = sample.get('game_length', 0)
            if (game_length < self.config.min_game_length or 
                game_length > self.config.max_game_length):
                continue
            
            filtered_indices.append((file_path, sample_idx))
            
            # Limit total samples
            if (self.config.max_samples and 
                len(filtered_indices) >= self.config.max_samples):
                break
            
            # Progress logging
            if checked_samples % 10000 == 0:
                logger.info(f"Filtered {checked_samples} samples, kept {len(filtered_indices)}")
        
        logger.info(f"Filtered to {len(filtered_indices)}/{len(indices)} samples")
        return filtered_indices
    
    def _split_data(self, indices: List[Tuple[str, int]]) -> Dict[str, List[Tuple[str, int]]]:
        """Split data into train/val/test sets"""
        # Shuffle indices for random split
        indices = indices.copy()
        random.shuffle(indices)
        
        total_samples = len(indices)
        val_size = int(total_samples * self.config.validation_split)
        test_size = int(total_samples * self.config.test_split)
        train_size = total_samples - val_size - test_size
        
        splits = {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }
        
        logger.info(f"Data split: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def _load_sample(self, file_path: str, sample_idx: int) -> Optional[Dict]:
        """Load a single sample from file"""
        # Check cache first
        cache_key = f"{file_path}:{sample_idx}"
        cached_sample = self.cache.get(cache_key)
        if cached_sample is not None:
            return cached_sample
        
        try:
            file_path_obj = Path(file_path)
            
            if file_path_obj.suffix == '.pkl':
                # Load from pickle file
                with open(file_path, 'rb') as f:
                    samples = pickle.load(f)
                    
                if sample_idx < len(samples):
                    sample = samples[sample_idx]
                    self.cache.put(cache_key, sample)
                    return sample
                else:
                    logger.warning(f"Sample index {sample_idx} out of range for {file_path}")
                    return None
                    
            elif file_path_obj.suffix == '.json':
                # Load from JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    if sample_idx < len(data):
                        sample = data[sample_idx]
                        self.cache.put(cache_key, sample)
                        return sample
                    else:
                        logger.warning(f"Sample index {sample_idx} out of range for {file_path}")
                        return None
                else:
                    # Single sample file
                    if sample_idx == 0:
                        self.cache.put(cache_key, data)
                        return data
                    else:
                        logger.warning(f"Sample index {sample_idx} invalid for single-sample file {file_path}")
                        return None
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load sample {sample_idx} from {file_path}: {e}")
            return None
    
    def _convert_sample_to_tensors(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Convert sample to PyTorch tensors"""
        try:
            state = sample['state']
            
            # Convert state vectors to tensors
            input_tensors = {}
            for key, value in state.items():
                if key == 'stat_boosts':
                    # Handle 2D array for stat boosts
                    input_tensors[key] = torch.tensor(value, dtype=torch.float32)
                elif key in ['hp_values']:
                    input_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    input_tensors[key] = torch.tensor(value, dtype=torch.long)
            
            # Action and reward
            action = torch.tensor(sample['action'], dtype=torch.long)
            reward = torch.tensor(sample.get('reward', 0), dtype=torch.float32)
            
            return {
                'input_ids': input_tensors,
                'labels': action,
                'rewards': reward,
                'turn': sample.get('turn', 0),
                'game_id': sample.get('game_id', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert sample to tensors: {e}")
            return None
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        if idx >= len(self.sample_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sample_indices)}")
        
        file_path, sample_idx = self.sample_indices[idx]
        
        # Load sample
        sample = self._load_sample(file_path, sample_idx)
        if sample is None:
            # Return a dummy sample if loading fails
            logger.warning(f"Failed to load sample {idx}, returning dummy")
            return self._get_dummy_sample()
        
        # Convert to tensors
        tensor_sample = self._convert_sample_to_tensors(sample)
        if tensor_sample is None:
            return self._get_dummy_sample()
        
        # Apply transform if provided
        if self.transform:
            tensor_sample = self.transform(tensor_sample)
        
        return tensor_sample
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return dummy sample for error cases"""
        dummy_input = {
            'species_ids': torch.zeros(12, dtype=torch.long),
            'move_ids': torch.zeros(12, dtype=torch.long),
            'item_ids': torch.zeros(12, dtype=torch.long),
            'ability_ids': torch.zeros(12, dtype=torch.long),
            'hp_values': torch.ones(12, dtype=torch.float32),
            'stat_boosts': torch.zeros(12, 7, dtype=torch.float32),
            'status_ids': torch.zeros(12, dtype=torch.long),
            'weather_ids': torch.zeros(12, dtype=torch.long),
            'terrain_ids': torch.zeros(12, dtype=torch.long),
            'position_ids': torch.arange(12, dtype=torch.long)
        }
        
        return {
            'input_ids': dummy_input,
            'labels': torch.tensor(0, dtype=torch.long),
            'rewards': torch.tensor(0.0, dtype=torch.float32),
            'turn': 0,
            'game_id': 'dummy'
        }
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata about a sample without loading full data"""
        file_path, sample_idx = self.sample_indices[idx]
        
        return {
            'file_path': file_path,
            'sample_idx': sample_idx,
            'dataset_idx': idx,
            'split': self.split
        }
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.sample_indices),
            'split': self.split,
            'cache_size': len(self.cache.cache),
            'data_files': len(set(fp for fp, _ in self.sample_indices)),
        }
        
        # Sample some data for statistics
        if len(self.sample_indices) > 0:
            sample_size = min(1000, len(self.sample_indices))
            sample_indices = random.sample(self.sample_indices, sample_size)
            
            formats = defaultdict(int)
            game_lengths = []
            rewards = []
            
            for file_path, sample_idx in sample_indices:
                sample = self._load_sample(file_path, sample_idx)
                if sample:
                    formats[sample.get('format', 'unknown')] += 1
                    game_lengths.append(sample.get('game_length', 0))
                    rewards.append(sample.get('reward', 0))
            
            stats.update({
                'formats': dict(formats),
                'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
                'win_rate': np.mean(rewards) if rewards else 0,
                'sampled_from': sample_size
            })
        
        return stats

def collate_real_data(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for real Pokemon data"""
    # Separate different components
    input_ids = {}
    labels = []
    rewards = []
    turns = []
    game_ids = []
    
    for sample in batch:
        labels.append(sample['labels'])
        rewards.append(sample['rewards'])
        turns.append(sample['turn'])
        game_ids.append(sample['game_id'])
        
        # Collect input_ids
        for key, value in sample['input_ids'].items():
            if key not in input_ids:
                input_ids[key] = []
            input_ids[key].append(value)
    
    # Stack tensors
    batched_input_ids = {}
    for key, tensor_list in input_ids.items():
        batched_input_ids[key] = torch.stack(tensor_list, dim=0)
    
    return {
        'input_ids': batched_input_ids,
        'labels': torch.stack(labels, dim=0),
        'rewards': torch.stack(rewards, dim=0),
        'turns': turns,
        'game_ids': game_ids
    }

def create_real_dataloader(config: DatasetConfig, 
                          split: str = 'train',
                          batch_size: int = 32,
                          num_workers: int = 4,
                          shuffle: bool = True) -> DataLoader:
    """Create DataLoader for real Pokemon data"""
    
    dataset = RealPokemonDataset(config, split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_real_data,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader

# Test the dataset
if __name__ == "__main__":
    # Test configuration
    config = DatasetConfig(
        data_dir="data/pokechamp/processed",
        max_samples=1000,
        validation_split=0.1,
        test_split=0.1
    )
    
    print("🧪 Testing Real Pokemon Dataset...")
    
    # Test dataset creation
    train_dataset = RealPokemonDataset(config, split='train')
    print(f"✅ Created train dataset: {len(train_dataset)} samples")
    
    val_dataset = RealPokemonDataset(config, split='val')
    print(f"✅ Created val dataset: {len(val_dataset)} samples")
    
    # Test sample loading
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"✅ Loaded sample with keys: {list(sample.keys())}")
        
        # Check tensor shapes
        input_ids = sample['input_ids']
        print(f"✅ Input shapes:")
        for key, tensor in input_ids.items():
            print(f"    {key}: {tensor.shape}")
    
    # Test dataloader
    train_loader = create_real_dataloader(config, split='train', batch_size=4, num_workers=0)
    
    try:
        batch = next(iter(train_loader))
        print(f"✅ Created batch with shapes:")
        print(f"    labels: {batch['labels'].shape}")
        print(f"    rewards: {batch['rewards'].shape}")
        print(f"    input_ids keys: {list(batch['input_ids'].keys())}")
    except Exception as e:
        print(f"❌ Failed to create batch: {e}")
    
    # Test statistics
    stats = train_dataset.get_stats()
    print(f"✅ Dataset stats: {stats}")
    
    print("🎉 Real dataset tests completed!")
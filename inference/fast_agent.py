"""
Fast Metamon Inference Agent
Optimized for <10 second per move requirement
Includes caching, quantization, and other speed optimizations
"""

import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import logging

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.metamon_transformer import MetamonTransformer, ModelConfig, StateEncoder
from configs.training_config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LRUCache:
    """Simple LRU cache for move decisions"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                # Remove oldest
                self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()

class FastMetamonAgent:
    """Optimized Metamon agent for fast inference"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 use_cache: bool = True,
                 cache_size: int = 10000,
                 use_half_precision: bool = True,
                 use_torchscript: bool = True,
                 temperature: float = 0.8):
        
        self.checkpoint_path = checkpoint_path
        self.use_cache = use_cache
        self.use_half_precision = use_half_precision
        self.use_torchscript = use_torchscript
        self.temperature = temperature
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.load_model()
        
        # Setup cache
        if self.use_cache:
            self.cache = LRUCache(cache_size)
        else:
            self.cache = None
        
        # Setup state encoder
        self.encoder = StateEncoder()
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def load_model(self):
        """Load and optimize the model"""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Create model
        model_config = ModelConfig()
        self.model = MetamonTransformer(model_config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'policy_model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['policy_model_state_dict'])
        else:
            raise ValueError("No model state dict found in checkpoint")
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        # Apply optimizations
        self.optimize_model()
        
        logger.info(f"Model loaded with {self.model.get_parameter_count():,} parameters")
    
    def optimize_model(self):
        """Apply various optimizations to the model"""
        
        # Half precision
        if self.use_half_precision and self.device.type == 'cuda':
            logger.info("Converting model to half precision")
            self.model = self.model.half()
        
        # TorchScript compilation
        if self.use_torchscript:
            try:
                logger.info("Compiling model with TorchScript")
                
                # Create dummy input for tracing
                dummy_input = self.create_dummy_input()
                
                # Trace the model
                self.model = torch.jit.trace(self.model, dummy_input)
                logger.info("TorchScript compilation successful")
                
            except Exception as e:
                logger.warning(f"TorchScript compilation failed: {e}")
                logger.info("Continuing without TorchScript optimization")
        
        # Set to eval mode and disable gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def create_dummy_input(self) -> Dict[str, torch.Tensor]:
        """Create dummy input for model tracing"""
        return self.encoder.encode_battle_state(None)
    
    def create_state_key(self, battle_state) -> str:
        """Create a hashable key for the battle state"""
        try:
            # Create a simplified state representation for caching
            # This is a simplified version - in practice, create a more robust hash
            
            if hasattr(battle_state, 'turn'):
                turn = battle_state.turn
            else:
                turn = 0
            
            # Create a simple hash based on available information
            state_str = f"turn_{turn}"
            
            # Add active pokemon info if available
            if hasattr(battle_state, 'active_pokemon') and battle_state.active_pokemon:
                state_str += f"_active_{battle_state.active_pokemon.species}"
                state_str += f"_hp_{battle_state.active_pokemon.current_hp_fraction:.2f}"
            
            # Add opponent active pokemon info
            if hasattr(battle_state, 'opponent_active_pokemon') and battle_state.opponent_active_pokemon:
                state_str += f"_opp_{battle_state.opponent_active_pokemon.species}"
                state_str += f"_opp_hp_{battle_state.opponent_active_pokemon.current_hp_fraction:.2f}"
            
            return state_str
            
        except Exception as e:
            logger.warning(f"Error creating state key: {e}")
            return f"fallback_{time.time()}"
    
    def get_action(self, battle_state, time_limit: float = 9.0) -> int:
        """
        Get action for the given battle state
        
        Args:
            battle_state: Current battle state
            time_limit: Maximum time allowed for inference (seconds)
            
        Returns:
            action_index: Index of the chosen action
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                state_key = self.create_state_key(battle_state)
                cached_action = self.cache.get(state_key)
                
                if cached_action is not None:
                    self.cache_hits += 1
                    elapsed = time.time() - start_time
                    self.inference_times.append(elapsed)
                    return cached_action
                else:
                    self.cache_misses += 1
            
            # Encode battle state
            encoded_state = self.encoder.encode_battle_state(battle_state)
            
            # Move to device
            input_ids = {}
            for key, tensor in encoded_state.items():
                input_ids[key] = tensor.to(self.device)
                if self.use_half_precision and self.device.type == 'cuda':
                    if tensor.dtype in [torch.float32, torch.float64]:
                        input_ids[key] = input_ids[key].half()
            
            # Forward pass with no gradients
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['policy_logits']
                
                # Apply temperature
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                
                # Get action probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample action (with some randomness for diversity)
                if self.temperature > 0:
                    dist = Categorical(probs)
                    action_idx = dist.sample().item()
                else:
                    # Greedy selection
                    action_idx = torch.argmax(probs, dim=-1).item()
            
            # Cache the result
            if self.cache:
                self.cache.set(state_key, action_idx)
            
            # Record timing
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            
            # Check time limit
            if elapsed > time_limit:
                logger.warning(f"Inference took {elapsed:.3f}s, exceeding limit of {time_limit}s")
            
            return action_idx
            
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            
            # Fallback to random action
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            
            return np.random.randint(0, CONFIG.model.action_vocab_size)
    
    def get_action_with_confidence(self, battle_state, time_limit: float = 9.0) -> Tuple[int, float]:
        """
        Get action with confidence score
        
        Returns:
            (action_index, confidence)
        """
        start_time = time.time()
        
        try:
            # Encode and predict
            encoded_state = self.encoder.encode_battle_state(battle_state)
            input_ids = {k: v.to(self.device) for k, v in encoded_state.items()}
            
            if self.use_half_precision and self.device.type == 'cuda':
                for key in input_ids:
                    if input_ids[key].dtype in [torch.float32, torch.float64]:
                        input_ids[key] = input_ids[key].half()
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['policy_logits']
                
                # Apply temperature
                logits = logits / self.temperature
                probs = F.softmax(logits, dim=-1)
                
                # Get best action and confidence
                max_prob, action_idx = torch.max(probs, dim=-1)
                confidence = max_prob.item()
                action_idx = action_idx.item()
            
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            
            return action_idx, confidence
            
        except Exception as e:
            logger.error(f"Error in get_action_with_confidence: {e}")
            return np.random.randint(0, CONFIG.model.action_vocab_size), 0.0
    
    def get_top_k_actions(self, battle_state, k: int = 3, time_limit: float = 9.0) -> List[Tuple[int, float]]:
        """
        Get top-k actions with probabilities
        
        Returns:
            List of (action_index, probability) tuples
        """
        try:
            encoded_state = self.encoder.encode_battle_state(battle_state)
            input_ids = {k: v.to(self.device) for k, v in encoded_state.items()}
            
            if self.use_half_precision and self.device.type == 'cuda':
                for key in input_ids:
                    if input_ids[key].dtype in [torch.float32, torch.float64]:
                        input_ids[key] = input_ids[key].half()
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['policy_logits']
                
                logits = logits / self.temperature
                probs = F.softmax(logits, dim=-1)
                
                # Get top-k
                top_probs, top_indices = torch.topk(probs, k, dim=-1)
                
                results = []
                for i in range(k):
                    action_idx = top_indices[0, i].item()
                    prob = top_probs[0, i].item()
                    results.append((action_idx, prob))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in get_top_k_actions: {e}")
            return [(np.random.randint(0, CONFIG.model.action_vocab_size), 1.0/k) for _ in range(k)]
    
    def clear_cache(self):
        """Clear the action cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        stats = {
            'total_inferences': len(times),
            'avg_inference_time': float(np.mean(times)),
            'max_inference_time': float(np.max(times)),
            'min_inference_time': float(np.min(times)),
            'p95_inference_time': float(np.percentile(times, 95)),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'total_cache_hits': self.cache_hits,
            'total_cache_misses': self.cache_misses
        }
        
        return stats
    
    def warmup(self, num_warmup: int = 10):
        """Warmup the model with dummy inputs"""
        logger.info(f"Warming up model with {num_warmup} dummy inferences")
        
        for _ in range(num_warmup):
            try:
                dummy_input = self.create_dummy_input()
                input_ids = {k: v.to(self.device) for k, v in dummy_input.items()}
                
                if self.use_half_precision and self.device.type == 'cuda':
                    for key in input_ids:
                        if input_ids[key].dtype in [torch.float32, torch.float64]:
                            input_ids[key] = input_ids[key].half()
                
                with torch.no_grad():
                    _ = self.model(input_ids)
                    
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Clear warmup times from stats
        self.inference_times.clear()
        
        logger.info("Model warmup complete")
    
    def save_performance_stats(self, path: str):
        """Save performance statistics to file"""
        stats = self.get_performance_stats()
        
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Performance stats saved to {path}")

def create_fast_agent(checkpoint_path: str, **kwargs) -> FastMetamonAgent:
    """Factory function to create a fast agent"""
    return FastMetamonAgent(checkpoint_path, **kwargs)

if __name__ == "__main__":
    # Test the fast agent
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fast agent")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-tests", type=int, default=100, help="Number of test inferences")
    args = parser.parse_args()
    
    # Create agent
    agent = create_fast_agent(args.checkpoint)
    
    # Warmup
    agent.warmup(10)
    
    # Test performance
    logger.info(f"Running {args.num_tests} test inferences")
    
    for i in range(args.num_tests):
        start = time.time()
        action = agent.get_action(None)  # Dummy state
        elapsed = time.time() - start
        
        if i % 20 == 0:
            logger.info(f"Test {i}: Action {action}, Time {elapsed:.4f}s")
    
    # Print stats
    stats = agent.get_performance_stats()
    logger.info("Performance Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    if stats['avg_inference_time'] > 1.0:
        logger.warning("Average inference time > 1.0s - may need further optimization")
    else:
        logger.info("Performance looks good for competition use")
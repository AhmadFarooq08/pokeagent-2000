"""
Metamon Transformer Architecture
200M parameter model for Pokemon battling
Based on GPT-2 architecture with Pokemon-specific modifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration"""
    # Architecture
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 2048
    
    # Vocabulary sizes
    species_vocab_size: int = 1025  # ~1025 Pokemon species
    move_vocab_size: int = 850      # ~850 moves
    item_vocab_size: int = 400      # ~400 items
    ability_vocab_size: int = 300   # ~300 abilities
    action_vocab_size: int = 10     # 4 moves + 6 switches max
    
    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Other
    initializer_range: float = 0.02
    use_cache: bool = False

class PokemonEmbedding(nn.Module):
    """Embedding layer for Pokemon game state"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Core embeddings
        self.species_embed = nn.Embedding(config.species_vocab_size, 256)
        self.move_embed = nn.Embedding(config.move_vocab_size, 128)
        self.item_embed = nn.Embedding(config.item_vocab_size, 64)
        self.ability_embed = nn.Embedding(config.ability_vocab_size, 64)
        
        # Stats embeddings
        self.hp_embed = nn.Linear(1, 32)  # HP as continuous value
        self.stat_boost_embed = nn.Linear(7, 32)  # 7 stat boosts
        
        # Status embeddings
        self.status_embed = nn.Embedding(8, 32)  # Normal, burn, poison, etc.
        
        # Field embeddings
        self.weather_embed = nn.Embedding(10, 32)
        self.terrain_embed = nn.Embedding(5, 32)
        
        # Position embeddings
        self.position_embed = nn.Embedding(12, 64)  # 6 team positions x 2 sides
        
        # Projection to hidden size
        self.projection = nn.Linear(256 + 128 + 64 + 64 + 32 + 32 + 32 + 32 + 32 + 64, config.hidden_size)
        
        # Layer norm and dropout
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                species_ids: torch.Tensor,
                move_ids: torch.Tensor, 
                item_ids: torch.Tensor,
                ability_ids: torch.Tensor,
                hp_values: torch.Tensor,
                stat_boosts: torch.Tensor,
                status_ids: torch.Tensor,
                weather_ids: torch.Tensor,
                terrain_ids: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            All tensors have shape [batch_size, sequence_length, ...]
        Returns:
            embeddings: [batch_size, sequence_length, hidden_size]
        """
        
        # Embed each component
        species_emb = self.species_embed(species_ids)  # [B, S, 256]
        move_emb = self.move_embed(move_ids.long())     # [B, S, 128] 
        item_emb = self.item_embed(item_ids)           # [B, S, 64]
        ability_emb = self.ability_embed(ability_ids)  # [B, S, 64]
        
        hp_emb = self.hp_embed(hp_values.unsqueeze(-1))      # [B, S, 32]
        boost_emb = self.stat_boost_embed(stat_boosts)       # [B, S, 32]
        status_emb = self.status_embed(status_ids)           # [B, S, 32]
        weather_emb = self.weather_embed(weather_ids)        # [B, S, 32]
        terrain_emb = self.terrain_embed(terrain_ids)        # [B, S, 32]
        position_emb = self.position_embed(position_ids)     # [B, S, 64]
        
        # Concatenate all embeddings
        combined = torch.cat([
            species_emb, move_emb, item_emb, ability_emb,
            hp_emb, boost_emb, status_emb,
            weather_emb, terrain_emb, position_emb
        ], dim=-1)  # [B, S, sum of embedding dims]
        
        # Project to hidden size
        embeddings = self.projection(combined)  # [B, S, hidden_size]
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class MultiHeadAttention(nn.Module):
    """Multi-head self attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        
        assert self.hidden_size % self.num_heads == 0
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Compute Q, K, V
        q = self.query(hidden_states)  # [B, S, H]
        k = self.key(hidden_states)    # [B, S, H]
        v = self.value(hidden_states)  # [B, S, H]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)  # [B, NH, S, HS]
        k = k.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)  # [B, NH, S, HS]
        v = v.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)  # [B, NH, S, HS]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores += attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)  # [B, NH, S, HS]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        
        # Output projection
        output = self.output_projection(context)
        
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class TransformerLayer(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attention_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        
        return hidden_states

class MetamonTransformer(nn.Module):
    """Main Metamon Transformer model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embeddings = PokemonEmbedding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output heads
        self.policy_head = nn.Linear(config.hidden_size, config.action_vocab_size)
        self.value_head = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self,
                input_ids: Dict[str, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Dictionary containing all input tensors
            attention_mask: Attention mask
            return_dict: Whether to return dict or tuple
        """
        
        # Get embeddings
        hidden_states = self.embeddings(
            species_ids=input_ids['species_ids'],
            move_ids=input_ids['move_ids'],
            item_ids=input_ids['item_ids'],
            ability_ids=input_ids['ability_ids'],
            hp_values=input_ids['hp_values'],
            stat_boosts=input_ids['stat_boosts'],
            status_ids=input_ids['status_ids'],
            weather_ids=input_ids['weather_ids'],
            terrain_ids=input_ids['terrain_ids'],
            position_ids=input_ids['position_ids']
        )
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Get final hidden state (for the last token)
        final_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Compute policy and value
        policy_logits = self.policy_head(final_hidden)  # [batch_size, action_vocab_size]
        value = self.value_head(final_hidden)           # [batch_size, 1]
        
        if return_dict:
            return {
                'policy_logits': policy_logits,
                'value': value.squeeze(-1),  # [batch_size]
                'hidden_states': hidden_states
            }
        else:
            return policy_logits, value.squeeze(-1), hidden_states
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        return sum(p.numel() for p in self.parameters())

class StateEncoder:
    """Encode Pokemon battle state to model inputs"""
    
    def __init__(self):
        # Vocabulary mappings (simplified - in practice, load from data)
        self.species_to_id = {}  # Load from Pokemon data
        self.move_to_id = {}     # Load from move data
        self.item_to_id = {}     # Load from item data
        self.ability_to_id = {}  # Load from ability data
        
        # Initialize basic mappings
        self._init_vocabularies()
    
    def _init_vocabularies(self):
        """Initialize vocabulary mappings"""
        # These should be loaded from actual Pokemon data
        # For now, create dummy mappings
        
        # Status mapping
        self.status_to_id = {
            None: 0, 'par': 1, 'slp': 2, 'frz': 3, 
            'brn': 4, 'psn': 5, 'tox': 6, 'confusion': 7
        }
        
        # Weather mapping
        self.weather_to_id = {
            None: 0, 'sun': 1, 'rain': 2, 'sand': 3,
            'hail': 4, 'snow': 5, 'harsh sun': 6, 'heavy rain': 7
        }
        
        # Terrain mapping
        self.terrain_to_id = {
            None: 0, 'electric': 1, 'grassy': 2, 'misty': 3, 'psychic': 4
        }
    
    def encode_battle_state(self, battle_state) -> Dict[str, torch.Tensor]:
        """
        Encode a battle state to model inputs
        This is a simplified version - needs full implementation
        """
        # This would need to be implemented based on the actual battle state format
        # For now, return dummy tensors with correct shapes
        
        batch_size = 1
        seq_length = 12  # 6 my team + 6 opponent team
        
        return {
            'species_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'move_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'item_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'ability_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'hp_values': torch.ones(batch_size, seq_length),
            'stat_boosts': torch.zeros(batch_size, seq_length, 7),
            'status_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'weather_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'terrain_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'position_ids': torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
        }

def create_model(config: Optional[ModelConfig] = None) -> MetamonTransformer:
    """Create a Metamon model with default or custom config"""
    if config is None:
        config = ModelConfig()
    
    model = MetamonTransformer(config)
    
    # Print model info
    param_count = model.get_parameter_count()
    print(f"Created Metamon model with {param_count:,} parameters")
    print(f"Target size: ~200M parameters")
    
    return model

if __name__ == "__main__":
    # Test model creation
    config = ModelConfig()
    model = create_model(config)
    
    # Test forward pass
    encoder = StateEncoder()
    dummy_input = encoder.encode_battle_state(None)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Policy logits shape: {output['policy_logits'].shape}")
        print(f"Value shape: {output['value'].shape}")
        print("Model test passed!")
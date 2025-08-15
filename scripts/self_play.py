#!/usr/bin/env python3
"""
Self-Play Data Generation
Generate additional training data from self-play battles
Critical for discovering new strategies and improving beyond human play
"""

import os
import sys
import json
import pickle
import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.metamon_transformer import MetamonTransformer, ModelConfig, StateEncoder
from configs.training_config import CONFIG

# Import poke-env for battle simulation
from poke_env.player import Player
from poke_env.environment import Battle
from poke_env import AccountConfiguration, ServerConfiguration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation"""
    num_games: int = 100000
    num_agents: int = 5
    temperature_range: Tuple[float, float] = (0.8, 1.2)
    max_concurrent_battles: int = 10
    save_every: int = 1000
    server_url: str = "ws://127.0.0.1:8000/showdown/websocket"
    battle_format: str = "gen9ou"

class MetamonSelfPlayAgent(Player):
    """Pokemon agent using Metamon model for self-play"""
    
    def __init__(self, 
                 model: MetamonTransformer,
                 temperature: float = 1.0,
                 name_suffix: str = "",
                 **kwargs):
        
        # Generate unique name
        agent_name = f"MetamonSP_{temperature:.1f}_{name_suffix}_{random.randint(1000, 9999)}"
        
        super().__init__(name=agent_name, **kwargs)
        
        self.model = model
        self.temperature = temperature
        self.encoder = StateEncoder()
        self.device = next(model.parameters()).device
        
        # Battle history for data collection
        self.battle_data = []
        
        # Evaluation mode
        self.model.eval()
    
    def choose_move(self, battle: Battle):
        """Choose move using the Metamon model"""
        try:
            # Encode current battle state
            encoded_state = self.encoder.encode_battle_state(battle)
            
            # Move to device
            input_ids = {k: v.to(self.device) for k, v in encoded_state.items()}
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['policy_logits']
                
                # Apply temperature
                logits = logits / self.temperature
                
                # Get action probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample action
                dist = Categorical(probs)
                action_idx = dist.sample().item()
            
            # Convert to poke-env action
            action = self.convert_action_index(battle, action_idx)
            
            # Record state-action pair for training data
            self.record_state_action(battle, action_idx, probs[0])
            
            return action
            
        except Exception as e:
            logger.warning(f"Error in choose_move: {e}")
            # Fallback to random move
            return self.choose_random_move(battle)
    
    def convert_action_index(self, battle: Battle, action_idx: int):
        """Convert action index to poke-env order"""
        # Get available actions
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        
        total_moves = len(available_moves)
        
        if action_idx < total_moves:
            # It's a move
            if available_moves:
                move_idx = action_idx % len(available_moves)
                return self.create_order(available_moves[move_idx])
        else:
            # It's a switch
            switch_idx = (action_idx - total_moves) % max(1, len(available_switches))
            if available_switches:
                return self.create_order(available_switches[switch_idx])
        
        # Fallback
        return self.choose_random_move(battle)
    
    def record_state_action(self, battle: Battle, action_idx: int, action_probs: torch.Tensor):
        """Record state-action pair for training data"""
        try:
            # Create state snapshot
            state_data = {
                'turn': battle.turn,
                'my_team': self.serialize_team(battle.team),
                'opp_team': self.serialize_team(battle.opponent_team, partial=True),
                'field': self.serialize_field(battle),
                'action_taken': action_idx,
                'action_probs': action_probs.cpu().numpy().tolist(),
                'temperature': self.temperature
            }
            
            self.battle_data.append(state_data)
            
        except Exception as e:
            logger.warning(f"Error recording state-action: {e}")
    
    def serialize_team(self, team: Dict, partial: bool = False) -> List[Dict]:
        """Serialize team data"""
        team_data = []
        
        for pokemon in team.values():
            if pokemon is None:
                continue
                
            pokemon_data = {
                'species': pokemon.species,
                'hp_fraction': pokemon.current_hp_fraction,
                'status': str(pokemon.status) if pokemon.status else None,
                'active': pokemon == (self.active_pokemon if not partial else self.opponent_active_pokemon),
                'fainted': pokemon.fainted
            }
            
            # For opponent team, only include revealed information
            if not partial:
                pokemon_data.update({
                    'level': pokemon.level,
                    'moves': [move.id for move in pokemon.moves.values() if move],
                    'item': pokemon.item.id if pokemon.item else None,
                    'ability': pokemon.ability.id if pokemon.ability else None,
                    'boosts': dict(pokemon.boosts)
                })
            else:
                # Only revealed information for opponent
                pokemon_data.update({
                    'revealed_moves': [move.id for move in pokemon.moves.values() if move],
                    'revealed_item': pokemon.item.id if pokemon.item else None,
                    'revealed_ability': pokemon.ability.id if pokemon.ability else None,
                    'boosts': dict(pokemon.boosts)
                })
            
            team_data.append(pokemon_data)
        
        return team_data
    
    def serialize_field(self, battle: Battle) -> Dict:
        """Serialize field conditions"""
        return {
            'weather': str(battle.weather) if battle.weather else None,
            'terrain': str(battle.terrain) if battle.terrain else None,
            'my_side_conditions': dict(battle.side_conditions),
            'opp_side_conditions': dict(battle.opponent_side_conditions)
        }
    
    def battle_finished_callback(self, battle: Battle):
        """Called when battle finishes"""
        # Add final outcome to all recorded states
        won = battle.won
        
        for state in self.battle_data:
            state['battle_outcome'] = 1.0 if won else 0.0
            state['battle_id'] = battle.battle_tag
            state['opponent'] = battle.opponent_username
        
        # Clear for next battle
        self.battle_data = []

class SelfPlayGenerator:
    """Manages self-play data generation"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str,
                 config: SelfPlayConfig):
        
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Create agents with different temperatures
        self.create_agents()
        
        # Data collection
        self.collected_games = 0
        self.all_battle_data = []
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Create model
        model_config = ModelConfig()
        self.model = MetamonTransformer(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Model loaded with {self.model.get_parameter_count():,} parameters")
    
    def create_agents(self):
        """Create agents with different temperatures for diversity"""
        self.agents = []
        
        # Create server configuration for local battles
        server_config = ServerConfiguration(
            server_url=self.config.server_url,
            authentication_url=None
        )
        
        # Generate temperature values
        temperatures = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            self.config.num_agents
        )
        
        for i, temp in enumerate(temperatures):
            agent = MetamonSelfPlayAgent(
                model=self.model,
                temperature=temp,
                name_suffix=f"agent_{i}",
                battle_format=self.config.battle_format,
                server_configuration=server_config,
                start_timer_on_battle_start=True,
                max_concurrent_battles=self.config.max_concurrent_battles,
            )
            
            self.agents.append(agent)
        
        logger.info(f"Created {len(self.agents)} agents with temperatures: {temperatures}")
    
    async def generate_battles(self):
        """Generate self-play battles"""
        logger.info(f"Starting self-play generation: {self.config.num_games} games")
        
        games_completed = 0
        
        with tqdm(total=self.config.num_games, desc="Self-play games") as pbar:
            
            while games_completed < self.config.num_games:
                # Create battle pairs
                battle_tasks = []
                
                for _ in range(min(self.config.max_concurrent_battles // 2, 
                                 (self.config.num_games - games_completed) // 2)):
                    
                    # Randomly select two agents
                    agent1, agent2 = random.sample(self.agents, 2)
                    
                    # Start battle
                    task = asyncio.create_task(
                        self.run_single_battle(agent1, agent2)
                    )
                    battle_tasks.append(task)
                
                # Wait for battles to complete
                if battle_tasks:
                    results = await asyncio.gather(*battle_tasks, return_exceptions=True)
                    
                    # Process results
                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning(f"Battle failed: {result}")
                        else:
                            games_completed += 1
                            pbar.update(1)
                            
                            # Collect battle data
                            self.collect_battle_data(result)
                            
                            # Save periodically
                            if games_completed % self.config.save_every == 0:
                                self.save_collected_data()
                
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.1)
        
        # Save final data
        self.save_collected_data()
        logger.info(f"Self-play generation complete: {games_completed} games")
    
    async def run_single_battle(self, agent1: MetamonSelfPlayAgent, agent2: MetamonSelfPlayAgent) -> Dict:
        """Run a single battle between two agents"""
        try:
            # Start the battle
            await agent1.battle_against(agent2, n_battles=1)
            
            # Get battle results
            battle_data = {
                'agent1_data': agent1.battle_data.copy(),
                'agent2_data': agent2.battle_data.copy(),
                'agent1_temp': agent1.temperature,
                'agent2_temp': agent2.temperature,
                'winner': 'agent1' if agent1.n_won_battles > 0 else 'agent2'
            }
            
            # Clear agent data
            agent1.battle_data.clear()
            agent2.battle_data.clear()
            
            return battle_data
            
        except Exception as e:
            logger.warning(f"Battle failed: {e}")
            raise e
    
    def collect_battle_data(self, battle_result: Dict):
        """Collect and process battle data"""
        try:
            # Extract data from both agents
            agent1_data = battle_result['agent1_data']
            agent2_data = battle_result['agent2_data']
            
            # Process agent1 data
            for state in agent1_data:
                processed_state = self.process_state_for_training(state)
                self.all_battle_data.append(processed_state)
            
            # Process agent2 data
            for state in agent2_data:
                processed_state = self.process_state_for_training(state)
                self.all_battle_data.append(processed_state)
                
        except Exception as e:
            logger.warning(f"Error collecting battle data: {e}")
    
    def process_state_for_training(self, state: Dict) -> Dict:
        """Process state data for training"""
        # Add metadata
        processed = state.copy()
        processed['data_source'] = 'self_play'
        processed['timestamp'] = time.time()
        
        return processed
    
    def save_collected_data(self):
        """Save collected battle data"""
        if not self.all_battle_data:
            return
        
        # Save as pickle file
        timestamp = int(time.time())
        filename = f"self_play_data_{timestamp}.pkl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.all_battle_data, f)
        
        logger.info(f"Saved {len(self.all_battle_data)} states to {filepath}")
        
        # Clear memory
        self.all_battle_data = []
    
    def generate_summary(self):
        """Generate summary statistics"""
        summary = {
            'total_games': self.collected_games,
            'agents_used': len(self.agents),
            'temperature_range': self.config.temperature_range,
            'output_directory': str(self.output_dir),
            'generation_time': time.time()
        }
        
        summary_path = self.output_dir / "self_play_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")

async def main():
    parser = argparse.ArgumentParser(description="Self-play data generation")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated data")
    parser.add_argument("--num-games", type=int, default=10000, help="Number of games to generate")
    parser.add_argument("--num-agents", type=int, default=5, help="Number of agents with different temperatures")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000/showdown/websocket", help="Local server URL")
    parser.add_argument("--debug", action="store_true", help="Debug mode with fewer games")
    args = parser.parse_args()
    
    # Adjust for debug mode
    if args.debug:
        args.num_games = min(args.num_games, 100)
    
    # Create configuration
    config = SelfPlayConfig(
        num_games=args.num_games,
        num_agents=args.num_agents,
        server_url=args.server_url
    )
    
    # Create generator
    generator = SelfPlayGenerator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config
    )
    
    # Generate battles
    await generator.generate_battles()
    
    # Generate summary
    generator.generate_summary()

if __name__ == "__main__":
    asyncio.run(main())
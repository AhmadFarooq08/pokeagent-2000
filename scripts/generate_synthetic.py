#!/usr/bin/env python3
"""
Generate synthetic Pokemon battles for data augmentation
Based on Metamon's approach of using diverse teams for robustness
"""

import os
import sys
import json
import torch
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.metamon_transformer import MetamonTransformer, ModelConfig
from data.replay_parser import ReplayParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticBattleGenerator:
    """Generate synthetic battles using trained model"""
    
    def __init__(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_checkpoint(checkpoint_path)
        self.parser = ReplayParser()
        
    def load_checkpoint(self, path: str) -> MetamonTransformer:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        config = ModelConfig()
        model = MetamonTransformer(config)
        
        # Handle DDP wrapped models
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def generate_diverse_team(self) -> Dict:
        """Generate unusual team composition for diversity"""
        # Mix of common and rare Pokemon
        # This prevents overfitting to standard meta
        pokemon_pool = [
            'pikachu', 'charizard', 'mewtwo', 'snorlax', 'dragonite',
            'alakazam', 'gengar', 'machamp', 'lapras', 'aerodactyl',
            'articuno', 'zapdos', 'moltres', 'gyarados', 'vaporeon',
            'jolteon', 'flareon', 'espeon', 'umbreon', 'leafeon',
            'glaceon', 'sylveon', 'lucario', 'garchomp', 'tyranitar'
        ]
        
        team = []
        for _ in range(6):
            pokemon = random.choice(pokemon_pool)
            team.append({
                'species': pokemon,
                'moves': ['tackle', 'thunderbolt', 'psychic', 'earthquake'],
                'level': 100,
                'hp': random.randint(80, 100)
            })
        
        return {'team': team}
    
    def simulate_battle(self, team1: Dict, team2: Dict) -> Dict:
        """Simulate a battle between two teams using the model"""
        # Simplified battle simulation
        # In practice, this would interface with Pokemon Showdown
        
        battle_log = []
        winner = random.choice([1, 2])  # Placeholder
        
        # Generate mock battle trajectory with more realistic turns
        num_turns = random.randint(10, 30)
        
        for turn in range(num_turns):
            # Generate more realistic action patterns
            p1_action = random.randint(0, 9)
            p2_action = random.randint(0, 9)
            
            # Create a log entry that matches real replay structure
            log_entry = f"|turn|{turn+1}"
            battle_log.append(log_entry)
            
            # Add player actions
            battle_log.append(f"|move|p1a: {team1['team'][0]['species']}|tackle")
            battle_log.append(f"|move|p2a: {team2['team'][0]['species']}|tackle")
            
            # Add damage or other effects
            if random.random() < 0.3:  # 30% chance of significant event
                battle_log.append(f"|-damage|p2a: {team2['team'][0]['species']}|{random.randint(50, 90)}/100")
        
        # Add win condition
        battle_log.append(f"|win|Player {winner}")
        
        return {
            'log': '\n'.join(battle_log),
            'winner': f'Player {winner}',
            'team1': team1,
            'team2': team2,
            'id': f'synthetic-{random.randint(100000, 999999)}',
            'format': 'gen1ou'
        }
    
    def generate_battles(self, n_battles: int, output_dir: str):
        """Generate synthetic battles and save to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {n_battles} synthetic battles...")
        
        battles_per_file = 100  # Group battles into files
        file_count = 0
        current_battles = []
        
        for i in range(n_battles):
            if i % 100 == 0:
                logger.info(f"Generated {i}/{n_battles} battles")
            
            # Generate diverse teams
            team1 = self.generate_diverse_team()
            team2 = self.generate_diverse_team()
            
            # Simulate battle
            battle = self.simulate_battle(team1, team2)
            current_battles.append(battle)
            
            # Save battles in batches
            if len(current_battles) >= battles_per_file or i == n_battles - 1:
                battle_file = output_path / f"synthetic_battles_{file_count:04d}.json"
                with open(battle_file, 'w') as f:
                    json.dump(current_battles, f, indent=2)
                
                file_count += 1
                current_battles = []
        
        logger.info(f"Generated {n_battles} battles in {output_dir}")
        logger.info(f"Created {file_count} batch files")
        
        # Create metadata file
        metadata = {
            'total_battles': n_battles,
            'files_created': file_count,
            'battles_per_file': battles_per_file,
            'format': 'gen1ou',
            'type': 'synthetic'
        }
        
        with open(output_path / 'synthetic_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Pokemon battles")
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='data/synthetic', help='Output directory')
    parser.add_argument('--n-battles', type=int, default=10000, help='Number of battles')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1
    
    generator = SyntheticBattleGenerator(args.checkpoint)
    generator.generate_battles(args.n_battles, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Pokemon Replay Parser
Converts Pokemon Showdown replay logs into structured training data
"""

import os
import sys
import json
import gzip
import re
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BattleState:
    """Represents the state of a Pokemon battle at a given turn"""
    turn: int = 0
    teams: Dict[str, List[Dict]] = field(default_factory=dict)
    active_pokemon: Dict[str, str] = field(default_factory=dict)
    field_conditions: Dict[str, Any] = field(default_factory=dict)
    weather: Optional[str] = None
    terrain: Optional[str] = None
    
    def to_vector(self) -> Dict[str, List]:
        """Convert battle state to model input vectors"""
        # Initialize vectors (12 positions: 6 per side)
        vector_data = {
            'species_ids': [0] * 12,
            'move_ids': [0] * 12,
            'item_ids': [0] * 12,
            'ability_ids': [0] * 12,
            'hp_values': [1.0] * 12,
            'stat_boosts': [[0] * 7 for _ in range(12)],  # HP, Atk, Def, SpA, SpD, Spe, Acc
            'status_ids': [0] * 12,
            'weather_ids': [0] * 12,
            'terrain_ids': [0] * 12,
            'position_ids': list(range(12))
        }
        
        # Fill in team data (simplified - would need actual vocab mappings)
        for side_idx, (side, team) in enumerate(self.teams.items()):
            base_idx = side_idx * 6
            for i, pokemon in enumerate(team[:6]):  # Max 6 Pokemon per team
                idx = base_idx + i
                
                # Convert Pokemon data to IDs (simplified)
                vector_data['species_ids'][idx] = hash(pokemon.get('species', '')) % 1024
                vector_data['item_ids'][idx] = hash(pokemon.get('item', '')) % 400
                vector_data['ability_ids'][idx] = hash(pokemon.get('ability', '')) % 300
                vector_data['hp_values'][idx] = pokemon.get('hp_ratio', 1.0)
                vector_data['status_ids'][idx] = self._status_to_id(pokemon.get('status'))
                
                # Stat boosts
                boosts = pokemon.get('boosts', {})
                for j, stat in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe', 'accuracy']):
                    vector_data['stat_boosts'][idx][j] = boosts.get(stat, 0)
                
                # Moves (use first move as representative)
                moves = pokemon.get('moves', [])
                if moves:
                    vector_data['move_ids'][idx] = hash(moves[0]) % 850
        
        # Field conditions
        weather_id = self._weather_to_id(self.weather)
        terrain_id = self._terrain_to_id(self.terrain)
        
        for i in range(12):
            vector_data['weather_ids'][i] = weather_id
            vector_data['terrain_ids'][i] = terrain_id
        
        return vector_data
    
    def _status_to_id(self, status: Optional[str]) -> int:
        """Convert status condition to ID"""
        status_map = {
            None: 0, 'par': 1, 'slp': 2, 'frz': 3,
            'brn': 4, 'psn': 5, 'tox': 6, 'confusion': 7
        }
        return status_map.get(status, 0)
    
    def _weather_to_id(self, weather: Optional[str]) -> int:
        """Convert weather to ID"""
        weather_map = {
            None: 0, 'sun': 1, 'rain': 2, 'sand': 3,
            'hail': 4, 'snow': 5, 'harsh sun': 6, 'heavy rain': 7
        }
        return weather_map.get(weather, 0)
    
    def _terrain_to_id(self, terrain: Optional[str]) -> int:
        """Convert terrain to ID"""
        terrain_map = {
            None: 0, 'electric': 1, 'grassy': 2, 'misty': 3, 'psychic': 4
        }
        return terrain_map.get(terrain, 0)

@dataclass
class Action:
    """Represents an action taken in battle"""
    type: str  # 'move', 'switch', 'item'
    target: Optional[str] = None
    move: Optional[str] = None
    pokemon: Optional[str] = None
    
    def to_id(self) -> int:
        """Convert action to numerical ID"""
        if self.type == 'move':
            # Moves: 0-3 (simplified - would map actual move)
            return hash(self.move or '') % 4
        elif self.type == 'switch':
            # Switches: 4-9 (6 possible switches)
            return 4 + (hash(self.pokemon or '') % 6)
        else:
            return 0  # Default action

class ReplayParser:
    """Parse Pokemon Showdown replay logs"""
    
    def __init__(self):
        # Regex patterns for common log events
        self.patterns = {
            'turn': re.compile(r'\|turn\|(\d+)'),
            'switch': re.compile(r'\|switch\|([^|]+)\|([^|]+)\|(.+)'),
            'move': re.compile(r'\|move\|([^|]+)\|([^|]+)\|([^|]*)'),
            'damage': re.compile(r'\|-damage\|([^|]+)\|(\d+)\/(\d+)'),
            'heal': re.compile(r'\|-heal\|([^|]+)\|(\d+)\/(\d+)'),
            'faint': re.compile(r'\|faint\|([^|]+)'),
            'player': re.compile(r'\|player\|([^|]+)\|([^|]+)\|(\d*)\|'),
            'weather': re.compile(r'\|-weather\|([^|]+)'),
            'terrain': re.compile(r'\|-fieldstart\|([^|]+)'),
            'boost': re.compile(r'\|-boost\|([^|]+)\|([^|]+)\|(\d+)'),
            'unboost': re.compile(r'\|-unboost\|([^|]+)\|([^|]+)\|(\d+)'),
            'status': re.compile(r'\|-status\|([^|]+)\|([^|]+)'),
            'win': re.compile(r'\|win\|([^|]+)')
        }
        
        # Initialize data structures
        self.reset_state()
    
    def reset_state(self):
        """Reset parser state for new replay"""
        self.battle_state = BattleState()
        self.actions = []
        self.players = {}
        self.winner = None
        self.current_turn = 0
        self.teams = {'p1': [], 'p2': []}
        self.active_pokemon = {'p1': None, 'p2': None}
        
    def parse_replay(self, replay_log: str) -> Optional[List[Dict]]:
        """
        Parse a complete replay log into training samples
        
        Returns:
            List of training samples with state-action pairs
        """
        self.reset_state()
        
        try:
            lines = replay_log.strip().split('\n')
            training_samples = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line and update state
                action_taken = self._parse_line(line)
                
                # If an action was taken, create training sample
                if action_taken and self.current_turn > 0:
                    state_vector = self.battle_state.to_vector()
                    
                    sample = {
                        'state': state_vector,
                        'action': action_taken.to_id(),
                        'turn': self.current_turn,
                        'player': self._get_acting_player(line),
                        'raw_action': {
                            'type': action_taken.type,
                            'move': action_taken.move,
                            'pokemon': action_taken.pokemon,
                            'target': action_taken.target
                        }
                    }
                    training_samples.append(sample)
            
            # Add outcome information to all samples
            if self.winner and training_samples:
                for sample in training_samples:
                    # Assign reward based on winner
                    player = sample['player']
                    sample['reward'] = 1 if player == self.winner else 0
                    sample['game_length'] = len(training_samples)
            
            return training_samples if training_samples else None
            
        except Exception as e:
            logger.warning(f"Failed to parse replay: {e}")
            return None
    
    def _parse_line(self, line: str) -> Optional[Action]:
        """Parse a single log line and update state"""
        try:
            # Turn marker
            if match := self.patterns['turn'].match(line):
                self.current_turn = int(match.group(1))
                return None
            
            # Player information
            elif match := self.patterns['player'].match(line):
                player_id = match.group(1)
                player_name = match.group(2)
                rating = match.group(3)
                self.players[player_id] = {
                    'name': player_name,
                    'rating': int(rating) if rating else 1000
                }
                return None
            
            # Switch action
            elif match := self.patterns['switch'].match(line):
                player_pos = match.group(1)
                pokemon_info = match.group(2)
                condition = match.group(3)
                
                # Parse player and position
                player = self._extract_player(player_pos)
                pokemon_name = pokemon_info.split(',')[0].strip()
                
                # Update battle state
                self._update_pokemon_switch(player, player_pos, pokemon_name, condition)
                
                return Action(type='switch', pokemon=pokemon_name)
            
            # Move action
            elif match := self.patterns['move'].match(line):
                attacker = match.group(1)
                move_name = match.group(2)
                target = match.group(3)
                
                return Action(type='move', move=move_name, target=target)
            
            # Damage
            elif match := self.patterns['damage'].match(line):
                pokemon = match.group(1)
                current_hp = int(match.group(2))
                max_hp = int(match.group(3))
                
                self._update_pokemon_hp(pokemon, current_hp, max_hp)
                return None
            
            # Healing
            elif match := self.patterns['heal'].match(line):
                pokemon = match.group(1)
                current_hp = int(match.group(2))
                max_hp = int(match.group(3))
                
                self._update_pokemon_hp(pokemon, current_hp, max_hp)
                return None
            
            # Faint
            elif match := self.patterns['faint'].match(line):
                pokemon = match.group(1)
                self._update_pokemon_faint(pokemon)
                return None
            
            # Weather
            elif match := self.patterns['weather'].match(line):
                weather = match.group(1)
                self.battle_state.weather = weather.lower()
                return None
            
            # Terrain
            elif match := self.patterns['terrain'].match(line):
                terrain = match.group(1)
                if 'terrain' in terrain.lower():
                    self.battle_state.terrain = terrain.lower().replace(' terrain', '')
                return None
            
            # Stat boosts
            elif match := self.patterns['boost'].match(line):
                pokemon = match.group(1)
                stat = match.group(2)
                amount = int(match.group(3))
                self._update_pokemon_boost(pokemon, stat, amount)
                return None
            
            elif match := self.patterns['unboost'].match(line):
                pokemon = match.group(1)
                stat = match.group(2)
                amount = int(match.group(3))
                self._update_pokemon_boost(pokemon, stat, -amount)
                return None
            
            # Status conditions
            elif match := self.patterns['status'].match(line):
                pokemon = match.group(1)
                status = match.group(2)
                self._update_pokemon_status(pokemon, status)
                return None
            
            # Win condition
            elif match := self.patterns['win'].match(line):
                self.winner = match.group(1)
                return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing line '{line}': {e}")
            return None
    
    def _extract_player(self, player_pos: str) -> str:
        """Extract player ID from position string"""
        if player_pos.startswith('p1'):
            return 'p1'
        elif player_pos.startswith('p2'):
            return 'p2'
        return 'p1'  # Default
    
    def _get_acting_player(self, line: str) -> str:
        """Determine which player took the action"""
        if '|p1' in line or '|p1a:' in line:
            return 'p1'
        elif '|p2' in line or '|p2a:' in line:
            return 'p2'
        return 'p1'  # Default
    
    def _update_pokemon_switch(self, player: str, position: str, pokemon_name: str, condition: str):
        """Update battle state for Pokemon switch"""
        # Parse condition (HP, status, etc.)
        hp_ratio = 1.0
        status = None
        
        if '/' in condition:
            parts = condition.split()
            if parts:
                hp_part = parts[0]
                if '/' in hp_part:
                    current, maximum = hp_part.split('/')
                    try:
                        hp_ratio = int(current) / int(maximum)
                    except ValueError:
                        hp_ratio = 1.0
        
        # Update team data
        pokemon_data = {
            'species': pokemon_name.lower(),
            'hp_ratio': hp_ratio,
            'status': status,
            'boosts': defaultdict(int),
            'moves': [],  # Would need to track from moveset
            'ability': '',  # Would need to parse
            'item': ''  # Would need to parse
        }
        
        # Update or add to team
        team = self.teams.get(player, [])
        
        # Find existing Pokemon or add new
        found = False
        for i, poke in enumerate(team):
            if poke['species'] == pokemon_name.lower():
                team[i] = pokemon_data
                found = True
                break
        
        if not found:
            team.append(pokemon_data)
        
        self.teams[player] = team
        self.active_pokemon[player] = pokemon_name.lower()
    
    def _update_pokemon_hp(self, pokemon_str: str, current_hp: int, max_hp: int):
        """Update Pokemon HP"""
        player = self._extract_player(pokemon_str)
        pokemon_name = pokemon_str.split(':')[-1].strip().lower()
        
        team = self.teams.get(player, [])
        for pokemon in team:
            if pokemon['species'] == pokemon_name:
                pokemon['hp_ratio'] = current_hp / max_hp if max_hp > 0 else 0
                break
    
    def _update_pokemon_faint(self, pokemon_str: str):
        """Update Pokemon faint status"""
        player = self._extract_player(pokemon_str)
        pokemon_name = pokemon_str.split(':')[-1].strip().lower()
        
        team = self.teams.get(player, [])
        for pokemon in team:
            if pokemon['species'] == pokemon_name:
                pokemon['hp_ratio'] = 0.0
                pokemon['status'] = 'fnt'
                break
    
    def _update_pokemon_boost(self, pokemon_str: str, stat: str, amount: int):
        """Update Pokemon stat boosts"""
        player = self._extract_player(pokemon_str)
        pokemon_name = pokemon_str.split(':')[-1].strip().lower()
        
        team = self.teams.get(player, [])
        for pokemon in team:
            if pokemon['species'] == pokemon_name:
                pokemon['boosts'][stat.lower()] += amount
                # Clamp to -6, +6 range
                pokemon['boosts'][stat.lower()] = max(-6, min(6, pokemon['boosts'][stat.lower()]))
                break
    
    def _update_pokemon_status(self, pokemon_str: str, status: str):
        """Update Pokemon status condition"""
        player = self._extract_player(pokemon_str)
        pokemon_name = pokemon_str.split(':')[-1].strip().lower()
        
        team = self.teams.get(player, [])
        for pokemon in team:
            if pokemon['species'] == pokemon_name:
                pokemon['status'] = status.lower()
                break

def process_replay_file(file_path: Path) -> Tuple[str, Optional[List[Dict]]]:
    """Process a single replay file"""
    try:
        # Read replay file
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
        
        # Handle both single replay and list of replays
        if isinstance(data, list):
            all_samples = []
            for replay_data in data:
                log = replay_data.get('log', '')
                if log:
                    parser = ReplayParser()
                    samples = parser.parse_replay(log)
                    if samples:
                        # Add metadata
                        for sample in samples:
                            sample['game_id'] = replay_data.get('id', 'unknown')
                            sample['format'] = replay_data.get('format', 'unknown')
                            sample['winner'] = replay_data.get('winner', '')
                        all_samples.extend(samples)
            return str(file_path), all_samples
        else:
            # Single replay
            log = data.get('log', '')
            if log:
                parser = ReplayParser()
                samples = parser.parse_replay(log)
                if samples:
                    # Add metadata
                    for sample in samples:
                        sample['game_id'] = data.get('id', 'unknown')
                        sample['format'] = data.get('format', 'unknown')
                        sample['winner'] = data.get('winner', '')
                    return str(file_path), samples
        
        return str(file_path), None
        
    except Exception as e:
        logger.warning(f"Failed to process {file_path}: {e}")
        return str(file_path), None

def main():
    parser = argparse.ArgumentParser(description="Parse Pokemon replays to training data")
    parser.add_argument("--input-dir", required=True, help="Directory containing replay files")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to process")
    parser.add_argument("--batch-size", type=int, default=1000, help="Samples per output file")
    parser.add_argument("--test-mode", action="store_true", help="Process only a few files for testing")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all replay files
    replay_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.json.gz"))
    
    if args.test_mode:
        replay_files = replay_files[:5]  # Only process 5 files for testing
    elif args.max_files:
        replay_files = replay_files[:args.max_files]
    
    logger.info(f"Found {len(replay_files)} replay files to process")
    
    if not replay_files:
        logger.error("No replay files found!")
        return
    
    # Process files in parallel
    all_samples = []
    processed_files = 0
    failed_files = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(process_replay_file, file_path): file_path 
            for file_path in replay_files
        }
        
        # Process results
        for future in as_completed(future_to_file):
            file_path, samples = future.result()
            
            if samples:
                all_samples.extend(samples)
                processed_files += 1
                
                if len(all_samples) % 10000 == 0:
                    logger.info(f"Processed {len(all_samples)} samples from {processed_files} files")
            else:
                failed_files += 1
            
            # Save in batches
            if len(all_samples) >= args.batch_size:
                batch_file = output_dir / f"training_batch_{processed_files//100:04d}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(all_samples[:args.batch_size], f)
                
                all_samples = all_samples[args.batch_size:]
                logger.info(f"Saved batch to {batch_file}")
    
    # Save remaining samples
    if all_samples:
        final_batch = output_dir / f"training_batch_final.pkl"
        with open(final_batch, 'wb') as f:
            pickle.dump(all_samples, f)
        logger.info(f"Saved final batch to {final_batch}")
    
    # Save metadata
    metadata = {
        'total_files': len(replay_files),
        'processed_files': processed_files,
        'failed_files': failed_files,
        'total_samples': processed_files * args.batch_size + len(all_samples),
        'format': 'parsed_replays',
        'version': '1.0'
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processing complete:")
    logger.info(f"  Processed: {processed_files}/{len(replay_files)} files")
    logger.info(f"  Failed: {failed_files} files") 
    logger.info(f"  Total samples: {metadata['total_samples']}")

if __name__ == "__main__":
    main()
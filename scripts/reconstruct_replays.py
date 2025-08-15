#!/usr/bin/env python3
"""
Replay reconstruction pipeline
Convert spectator logs to first-person partial observations
This is THE most critical component for training competitive agents
"""

import json
import re
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PokemonState:
    """Represents what a player knows about a Pokemon"""
    species: str
    nickname: str = ""
    hp_fraction: float = 1.0
    status: Optional[str] = None
    level: int = 50
    gender: Optional[str] = None
    
    # Revealed information (initially unknown)
    revealed_moves: List[str] = field(default_factory=list)
    revealed_ability: Optional[str] = None
    revealed_item: Optional[str] = None
    
    # Stats and boosts
    boosts: Dict[str, int] = field(default_factory=lambda: {
        'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'accuracy': 0, 'evasion': 0
    })
    
    # Volatile conditions
    volatiles: List[str] = field(default_factory=list)
    
    # Whether this Pokemon is currently active
    active: bool = False
    fainted: bool = False

@dataclass
class BattleState:
    """Complete battle state from one player's perspective"""
    turn: int = 0
    my_team: List[PokemonState] = field(default_factory=list)
    opp_team: List[PokemonState] = field(default_factory=list)
    
    # Field conditions
    weather: Optional[str] = None
    terrain: Optional[str] = None
    
    # Side conditions
    my_side_conditions: Dict[str, int] = field(default_factory=dict)
    opp_side_conditions: Dict[str, int] = field(default_factory=dict)
    
    # Last action taken
    last_action: Optional[str] = None
    action_result: Optional[str] = None
    
    # Battle metadata
    format: str = "gen9ou"
    winner: Optional[str] = None
    finished: bool = False

class ReplayReconstructor:
    """Reconstructs first-person view from spectator logs"""
    
    def __init__(self):
        # Regex patterns for parsing battle events
        self.patterns = {
            'turn': re.compile(r'\|turn\|(\d+)'),
            'switch': re.compile(r'\|switch\|([^|]+)\|([^|]+)\|([^|]+)'),
            'drag': re.compile(r'\|drag\|([^|]+)\|([^|]+)\|([^|]+)'),
            'move': re.compile(r'\|move\|([^|]+)\|([^|]+)\|([^|]*)\|?([^|]*)'),
            'damage': re.compile(r'\|(-damage|-heal)\|([^|]+)\|([^|]+)'),
            'faint': re.compile(r'\|-faint\|([^|]+)'),
            'status': re.compile(r'\|-status\|([^|]+)\|([^|]+)'),
            'boost': re.compile(r'\|-boost\|([^|]+)\|([^|]+)\|([^|]+)'),
            'unboost': re.compile(r'\|-unboost\|([^|]+)\|([^|]+)\|([^|]+)'),
            'weather': re.compile(r'\|-weather\|([^|]+)'),
            'terrain': re.compile(r'\|-fieldstart\|move: ([^|]+)'),
            'sidestart': re.compile(r'\|-sidestart\|([^|]+)\|([^|]+)'),
            'win': re.compile(r'\|win\|([^|]+)'),
            'poke': re.compile(r'\|poke\|([^|]+)\|([^|]+)\|([^|]*)')
        }
    
    def parse_pokemon_ident(self, ident: str) -> Tuple[str, str]:
        """Parse pokemon identifier like 'p1a: Garchomp' -> ('p1', 'Garchomp')"""
        if ':' in ident:
            side_info, name = ident.split(':', 1)
            side = side_info.strip()[:2]  # 'p1' or 'p2'
            return side, name.strip()
        return "", ident.strip()
    
    def parse_pokemon_details(self, details: str) -> Dict[str, Any]:
        """Parse pokemon details like 'Garchomp, L50, M' -> {species: 'Garchomp', level: 50, gender: 'M'}"""
        parts = [p.strip() for p in details.split(',')]
        result = {'species': parts[0], 'level': 50, 'gender': None}
        
        for part in parts[1:]:
            if part.startswith('L'):
                result['level'] = int(part[1:])
            elif part in ['M', 'F']:
                result['gender'] = part
        
        return result
    
    def parse_hp_status(self, hp_status: str) -> Tuple[float, Optional[str]]:
        """Parse HP and status like '248/354 par' -> (0.7, 'par')"""
        parts = hp_status.split()
        
        # Parse HP fraction
        hp_fraction = 1.0
        if '/' in parts[0]:
            current, max_hp = parts[0].split('/')
            if max_hp != '0':
                hp_fraction = float(current) / float(max_hp)
        elif parts[0] == '0':
            hp_fraction = 0.0
        
        # Parse status
        status = None
        if len(parts) > 1:
            status = parts[1]
        
        return hp_fraction, status
    
    def initialize_teams(self, log_lines: List[str], player_id: str) -> Tuple[List[PokemonState], List[PokemonState]]:
        """Initialize team information from team preview"""
        my_team = []
        opp_team = []
        
        opp_id = 'p1' if player_id == 'p2' else 'p2'
        
        for line in log_lines:
            match = self.patterns['poke'].match(line)
            if match:
                side, species_details, item = match.groups()
                
                pokemon_info = self.parse_pokemon_details(species_details)
                pokemon = PokemonState(
                    species=pokemon_info['species'],
                    level=pokemon_info['level'],
                    gender=pokemon_info['gender']
                )
                
                # Item is revealed in team preview
                if item:
                    pokemon.revealed_item = item
                
                if side == player_id:
                    my_team.append(pokemon)
                elif side == opp_id:
                    opp_team.append(pokemon)
        
        return my_team, opp_team
    
    def get_pokemon_by_ident(self, teams: Dict[str, List[PokemonState]], ident: str) -> Optional[PokemonState]:
        """Find pokemon by identifier"""
        side, name = self.parse_pokemon_ident(ident)
        
        if side in teams:
            for pokemon in teams[side]:
                if pokemon.species in name or pokemon.nickname == name:
                    return pokemon
        return None
    
    def reconstruct_battle(self, spectator_log: str, player_id: str) -> List[BattleState]:
        """
        Reconstruct the battle from player's perspective
        Returns list of states (one per turn)
        """
        lines = spectator_log.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Initialize battle state
        current_state = BattleState()
        states = []
        
        # Initialize teams from team preview
        my_team, opp_team = self.initialize_teams(lines, player_id)
        current_state.my_team = my_team
        current_state.opp_team = opp_team
        
        # Create team lookup
        opp_id = 'p1' if player_id == 'p2' else 'p2'
        teams = {
            player_id: my_team,
            opp_id: opp_team
        }
        
        # Process each line
        for line in lines:
            new_state = self.process_line(line, current_state, teams, player_id)
            if new_state is not None:
                current_state = new_state
                
                # Save state at end of each turn
                if self.patterns['turn'].match(line):
                    states.append(self.copy_state(current_state))
        
        return states
    
    def process_line(self, line: str, state: BattleState, teams: Dict[str, List[PokemonState]], player_id: str) -> Optional[BattleState]:
        """Process a single log line and update state"""
        state = self.copy_state(state)  # Don't modify original
        
        # Turn counter
        turn_match = self.patterns['turn'].match(line)
        if turn_match:
            state.turn = int(turn_match.group(1))
            return state
        
        # Pokemon switches
        switch_match = self.patterns['switch'].match(line) or self.patterns['drag'].match(line)
        if switch_match:
            ident, species_details, hp_status = switch_match.groups()
            side, _ = self.parse_pokemon_ident(ident)
            
            # Set active Pokemon
            for pokemon in teams[side]:
                pokemon.active = False
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon:
                pokemon.active = True
                # Update HP and status if revealed
                if hp_status:
                    hp_fraction, status = self.parse_hp_status(hp_status)
                    pokemon.hp_fraction = hp_fraction
                    pokemon.status = status
            
            return state
        
        # Moves
        move_match = self.patterns['move'].match(line)
        if move_match:
            ident, move_name, target, details = move_match.groups()
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon:
                # Reveal move
                if move_name not in pokemon.revealed_moves:
                    pokemon.revealed_moves.append(move_name)
            
            return state
        
        # Damage/healing
        damage_match = self.patterns['damage'].match(line)
        if damage_match:
            effect, ident, hp_status = damage_match.groups()
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon:
                hp_fraction, status = self.parse_hp_status(hp_status)
                pokemon.hp_fraction = hp_fraction
                if status:
                    pokemon.status = status
            
            return state
        
        # Status conditions
        status_match = self.patterns['status'].match(line)
        if status_match:
            ident, status_name = status_match.groups()
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon:
                pokemon.status = status_name
            
            return state
        
        # Stat boosts
        boost_match = self.patterns['boost'].match(line)
        if boost_match:
            ident, stat, amount = boost_match.groups()
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon and stat in pokemon.boosts:
                pokemon.boosts[stat] += int(amount)
            
            return state
        
        # Stat drops
        unboost_match = self.patterns['unboost'].match(line)
        if unboost_match:
            ident, stat, amount = unboost_match.groups()
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon and stat in pokemon.boosts:
                pokemon.boosts[stat] -= int(amount)
            
            return state
        
        # Fainting
        faint_match = self.patterns['faint'].match(line)
        if faint_match:
            ident = faint_match.group(1)
            
            pokemon = self.get_pokemon_by_ident(teams, ident)
            if pokemon:
                pokemon.fainted = True
                pokemon.active = False
                pokemon.hp_fraction = 0.0
            
            return state
        
        # Weather
        weather_match = self.patterns['weather'].match(line)
        if weather_match:
            state.weather = weather_match.group(1)
            return state
        
        # Win condition
        win_match = self.patterns['win'].match(line)
        if win_match:
            state.winner = win_match.group(1)
            state.finished = True
            return state
        
        return None  # No state change
    
    def copy_state(self, state: BattleState) -> BattleState:
        """Deep copy a battle state"""
        import copy
        return copy.deepcopy(state)

def process_single_replay(replay_data: Dict, output_dir: str) -> bool:
    """Process a single replay and save the reconstructed states"""
    try:
        reconstructor = ReplayReconstructor()
        
        # Extract log and metadata
        log = replay_data['log']
        winner = replay_data.get('winner', '')
        format_name = replay_data.get('format', 'unknown')
        
        # Reconstruct for both players
        p1_states = reconstructor.reconstruct_battle(log, 'p1')
        p2_states = reconstructor.reconstruct_battle(log, 'p2')
        
        if not p1_states or not p2_states:
            logger.warning("Failed to reconstruct states")
            return False
        
        # Determine labels (wins/losses)
        p1_won = 1 if 'p1' in winner.lower() else 0
        p2_won = 1 if 'p2' in winner.lower() else 0
        
        # Create training examples
        p1_examples = []
        p2_examples = []
        
        for i, (p1_state, p2_state) in enumerate(zip(p1_states, p2_states)):
            # Create training example for p1
            p1_example = {
                'state': p1_state,
                'turn': i,
                'label': p1_won,
                'format': format_name
            }
            p1_examples.append(p1_example)
            
            # Create training example for p2
            p2_example = {
                'state': p2_state,
                'turn': i,
                'label': p2_won,
                'format': format_name
            }
            p2_examples.append(p2_example)
        
        # Save examples
        replay_id = replay_data.get('id', f'replay_{hash(log) % 1000000}')
        
        p1_file = Path(output_dir) / f"{replay_id}_p1.pkl"
        p2_file = Path(output_dir) / f"{replay_id}_p2.pkl"
        
        with open(p1_file, 'wb') as f:
            pickle.dump(p1_examples, f)
        
        with open(p2_file, 'wb') as f:
            pickle.dump(p2_examples, f)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing replay: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Reconstruct battle replays")
    parser.add_argument("--input-dir", required=True, help="Input directory with raw replays")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--max-replays", type=int, help="Maximum number of replays to process")
    parser.add_argument("--format-filter", default="gen9ou", help="Only process this format")
    args = parser.parse_args()
    
    # Load dataset
    from datasets import Dataset
    
    logger.info(f"Loading dataset from {args.input_dir}")
    train_dataset = Dataset.load_from_disk(f"{args.input_dir}/train")
    
    # Filter by format if specified
    if args.format_filter:
        original_size = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: args.format_filter.lower() in x['format'].lower()
        )
        logger.info(f"Filtered {original_size} -> {len(train_dataset)} battles for format {args.format_filter}")
    
    # Limit number of replays if specified
    if args.max_replays:
        train_dataset = train_dataset.select(range(min(args.max_replays, len(train_dataset))))
        logger.info(f"Processing {len(train_dataset)} replays")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process replays
    successful = 0
    failed = 0
    
    for i, replay in enumerate(tqdm(train_dataset, desc="Processing replays")):
        if process_single_replay(replay, args.output_dir):
            successful += 1
        else:
            failed += 1
        
        # Log progress
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i+1} replays, {successful} successful, {failed} failed")
    
    logger.info(f"Reconstruction complete: {successful} successful, {failed} failed")
    
    # Save summary
    summary = {
        "total_processed": len(train_dataset),
        "successful": successful,
        "failed": failed,
        "format_filter": args.format_filter,
        "output_dir": args.output_dir
    }
    
    with open(f"{args.output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
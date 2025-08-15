#!/usr/bin/env python3
"""
Competition Deployment Script
Final agent for PokeAgent server competition
Integrates FastMetamonAgent with poke-env for ladder play
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import csv
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from poke_env.player import Player
from poke_env.environment import Battle
from poke_env import AccountConfiguration, ServerConfiguration
from inference.fast_agent import FastMetamonAgent
from configs.training_config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompetitionAgent(Player):
    """
    Final competition agent for PokeAgent server
    Combines FastMetamonAgent with poke-env Player
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 team_path: str,
                 temperature: float = 0.8,
                 enable_logging: bool = True,
                 log_file: Optional[str] = None,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.checkpoint_path = checkpoint_path
        self.team_path = team_path
        self.enable_logging = enable_logging
        self.temperature = temperature
        
        # Load the fast inference model
        logger.info("Initializing FastMetamonAgent...")
        self.fast_agent = FastMetamonAgent(
            checkpoint_path=checkpoint_path,
            temperature=temperature,
            use_cache=True,
            cache_size=10000,
            use_half_precision=True,
            use_torchscript=True
        )
        
        # Warmup the model
        self.fast_agent.warmup(20)
        logger.info("Model warmup complete")
        
        # Battle statistics
        self.battle_stats = {
            'total_battles': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_turns': 0,
            'total_time': 0.0,
            'avg_time_per_move': 0.0,
            'timeouts': 0,
            'errors': 0
        }
        
        # Detailed logging
        self.battle_log = []
        self.log_file = log_file
        
        # Load team
        self.load_team()
    
    def load_team(self):
        """Load team from file"""
        try:
            team_path = Path(self.team_path)
            if team_path.exists():
                self.team = team_path.read_text().strip()
                logger.info(f"Loaded team from {team_path}")
            else:
                logger.warning(f"Team file not found: {team_path}")
                # Use a default team if file not found
                self.team = self.get_default_team()
        except Exception as e:
            logger.error(f"Error loading team: {e}")
            self.team = self.get_default_team()
    
    def get_default_team(self) -> str:
        """Get a default competitive team"""
        # This should be replaced with an actual competitive team
        return """
Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Water
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Rapid Spin
- Headlong Rush
- Close Combat
- Knock Off

Gholdengo @ Leftovers
Ability: Good as Gold
Tera Type: Flying
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Make It Rain
- Shadow Ball
- Nasty Plot
- Recover

Kingambit @ Black Glasses
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Swords Dance
- Iron Head

Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- Flamethrower
- U-turn

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 176 Def / 88 Spe
Impish Nature
- Stealth Rock
- Earthquake
- U-turn
- Protect

Raging Bolt @ Booster Energy
Ability: Protosynthesis
Tera Type: Electric
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Thunderclap
- Thunderbolt
- Dragon Pulse
- Calm Mind
"""
    
    def choose_move(self, battle: Battle):
        """Choose move using the Metamon model"""
        move_start_time = time.time()
        
        try:
            # Get action from the fast agent
            action_idx = self.fast_agent.get_action(battle, time_limit=8.0)
            
            # Convert to poke-env order
            action = self.convert_action_to_order(battle, action_idx)
            
            # Record timing
            move_time = time.time() - move_start_time
            self.battle_stats['total_time'] += move_time
            
            # Log the move decision
            if self.enable_logging:
                self.log_move_decision(battle, action_idx, action, move_time)
            
            return action
            
        except Exception as e:
            logger.error(f"Error in choose_move: {e}")
            self.battle_stats['errors'] += 1
            
            # Fallback to random move
            move_time = time.time() - move_start_time
            self.battle_stats['total_time'] += move_time
            
            return self.choose_random_move(battle)
    
    def convert_action_to_order(self, battle: Battle, action_idx: int):
        """Convert action index to poke-env order"""
        try:
            # Get available actions
            available_moves = battle.available_moves
            available_switches = battle.available_switches
            
            total_moves = len(available_moves)
            
            # Determine if it's a move or switch
            if action_idx < total_moves and available_moves:
                # It's a move
                move_idx = action_idx % len(available_moves)
                chosen_move = available_moves[move_idx]
                return self.create_order(chosen_move)
            
            elif available_switches:
                # It's a switch
                switch_idx = (action_idx - total_moves) % len(available_switches)
                chosen_pokemon = available_switches[switch_idx]
                return self.create_order(chosen_pokemon)
            
            else:
                # Fallback - should not happen
                return self.choose_random_move(battle)
                
        except Exception as e:
            logger.warning(f"Error converting action to order: {e}")
            return self.choose_random_move(battle)
    
    def log_move_decision(self, battle: Battle, action_idx: int, action, move_time: float):
        """Log the move decision for analysis"""
        try:
            log_entry = {
                'battle_tag': battle.battle_tag,
                'turn': battle.turn,
                'action_idx': action_idx,
                'action_type': 'move' if hasattr(action, 'move') else 'switch',
                'move_time': move_time,
                'my_active': battle.active_pokemon.species if battle.active_pokemon else None,
                'opp_active': battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None,
                'my_hp': battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0,
                'opp_hp': battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0,
                'timestamp': time.time()
            }
            
            self.battle_log.append(log_entry)
            
        except Exception as e:
            logger.warning(f"Error logging move decision: {e}")
    
    def battle_finished_callback(self, battle: Battle):
        """Called when a battle finishes"""
        try:
            # Update battle statistics
            self.battle_stats['total_battles'] += 1
            self.battle_stats['total_turns'] += battle.turn
            
            if battle.won:
                self.battle_stats['wins'] += 1
            elif battle.won is False:
                self.battle_stats['losses'] += 1
            else:
                self.battle_stats['draws'] += 1
            
            # Update average time per move
            total_moves = sum([len([entry for entry in self.battle_log if entry['battle_tag'] == battle.battle_tag])])
            if total_moves > 0:
                self.battle_stats['avg_time_per_move'] = self.battle_stats['total_time'] / total_moves
            
            # Log battle result
            win_rate = self.battle_stats['wins'] / max(1, self.battle_stats['total_battles'])
            
            logger.info(f"Battle {battle.battle_tag} finished: "
                       f"{'Won' if battle.won else 'Lost' if battle.won is False else 'Draw'}")
            logger.info(f"Overall record: {self.battle_stats['wins']}W-{self.battle_stats['losses']}L-{self.battle_stats['draws']}D "
                       f"({win_rate:.1%} win rate)")
            
            # Save battle log periodically
            if self.log_file and self.battle_stats['total_battles'] % 10 == 0:
                self.save_battle_log()
                
        except Exception as e:
            logger.error(f"Error in battle_finished_callback: {e}")
    
    def save_battle_log(self):
        """Save battle log to CSV file"""
        if not self.log_file or not self.battle_log:
            return
        
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to CSV
            with open(log_path, 'w', newline='') as f:
                if self.battle_log:
                    fieldnames = self.battle_log[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.battle_log)
            
            logger.info(f"Battle log saved to {log_path}")
            
        except Exception as e:
            logger.error(f"Error saving battle log: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        agent_stats = self.fast_agent.get_performance_stats()
        
        summary = {
            'battle_stats': self.battle_stats,
            'agent_performance': agent_stats,
            'total_battles': self.battle_stats['total_battles'],
            'win_rate': self.battle_stats['wins'] / max(1, self.battle_stats['total_battles']),
            'avg_time_per_move': self.battle_stats['avg_time_per_move'],
            'model_path': self.checkpoint_path,
            'team_path': self.team_path,
            'temperature': self.temperature
        }
        
        return summary

def create_pokeagent_server_config() -> ServerConfiguration:
    """Create server configuration for PokeAgent competition"""
    try:
        return ServerConfiguration(
            server_url="wss://pokeagentshowdown.com/showdown/websocket",
            authentication_url="https://play.pokemonshowdown.com/action.php"
        )
    except Exception:
        # Fallback to local server for testing
        return ServerConfiguration(
            server_url="ws://127.0.0.1:8000/showdown/websocket",
            authentication_url=None
        )

def create_account_config(username: str, password: str) -> AccountConfiguration:
    """Create account configuration"""
    return AccountConfiguration(username, password)

async def run_competition(agent: CompetitionAgent, 
                         num_battles: int = 50,
                         battle_format: str = "gen9ou"):
    """Run the competition"""
    logger.info(f"Starting competition: {num_battles} battles in {battle_format}")
    
    try:
        # Run ladder battles
        await agent.ladder(num_battles)
        
        # Get final statistics
        summary = agent.get_performance_summary()
        
        logger.info("=== COMPETITION SUMMARY ===")
        logger.info(f"Total battles: {summary['total_battles']}")
        logger.info(f"Win rate: {summary['win_rate']:.1%}")
        logger.info(f"Record: {agent.battle_stats['wins']}W-{agent.battle_stats['losses']}L-{agent.battle_stats['draws']}D")
        logger.info(f"Average time per move: {summary['avg_time_per_move']:.3f}s")
        
        if 'avg_inference_time' in summary['agent_performance']:
            logger.info(f"Average inference time: {summary['agent_performance']['avg_inference_time']:.3f}s")
            logger.info(f"Cache hit rate: {summary['agent_performance']['cache_hit_rate']:.1%}")
        
        # Save final summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"logs/competition_summary_{timestamp}.json"
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary saved to {summary_path}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Competition failed: {e}")
        raise e

async def main():
    parser = argparse.ArgumentParser(description="PokeAgent Competition")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--team", required=True, help="Path to team file")
    parser.add_argument("--username", required=True, help="Username for PokeAgent server")
    parser.add_argument("--password", required=True, help="Password for PokeAgent server")
    parser.add_argument("--num-battles", type=int, default=50, help="Number of battles to play")
    parser.add_argument("--format", default="gen9ou", help="Battle format")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--log-file", help="Path to save battle log CSV")
    parser.add_argument("--test-local", action="store_true", help="Test on local server")
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not Path(args.team).exists():
        raise FileNotFoundError(f"Team file not found: {args.team}")
    
    # Create server configuration
    if args.test_local:
        server_config = ServerConfiguration(
            server_url="ws://127.0.0.1:8000/showdown/websocket",
            authentication_url=None
        )
    else:
        server_config = create_pokeagent_server_config()
    
    # Create account configuration
    account_config = create_account_config(args.username, args.password)
    
    # Create competition agent
    agent = CompetitionAgent(
        checkpoint_path=args.checkpoint,
        team_path=args.team,
        temperature=args.temperature,
        enable_logging=True,
        log_file=args.log_file,
        battle_format=args.format,
        server_configuration=server_config,
        account_configuration=account_config,
        start_timer_on_battle_start=True,
        max_concurrent_battles=1
    )
    
    logger.info(f"Agent created: {agent.username}")
    logger.info(f"Server: {server_config.server_url}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Temperature: {args.temperature}")
    
    # Run competition
    try:
        summary = await run_competition(agent, args.num_battles, args.format)
        
        # Final save
        if args.log_file:
            agent.save_battle_log()
        
        logger.info("Competition completed successfully!")
        
        return summary
        
    except KeyboardInterrupt:
        logger.info("Competition interrupted by user")
        if args.log_file:
            agent.save_battle_log()
        return None
    
    except Exception as e:
        logger.error(f"Competition failed: {e}")
        if args.log_file:
            agent.save_battle_log()
        raise e

if __name__ == "__main__":
    asyncio.run(main())
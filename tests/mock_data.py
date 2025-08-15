#!/usr/bin/env python3
"""
Mock Data Generators for Testing
Creates realistic test data without requiring actual downloads
"""

import os
import json
import random
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import pickle

# Pokemon data for realistic mocks
POKEMON_SPECIES = [
    "pikachu", "charizard", "blastoise", "venusaur", "garchomp", "metagross",
    "salamence", "tyranitar", "dragonite", "machamp", "gengar", "alakazam",
    "lucario", "blaziken", "swampert", "sceptile", "infernape", "empoleon",
    "torterra", "dialga", "palkia", "giratina", "arceus", "mew", "mewtwo"
]

MOVES = [
    "thunderbolt", "flamethrower", "surf", "earthquake", "ice-beam", "psychic",
    "shadow-ball", "dragon-pulse", "close-combat", "stone-edge", "u-turn",
    "volt-switch", "stealth-rock", "toxic", "recover", "roost", "substitute",
    "protect", "dragon-dance", "swords-dance", "nasty-plot", "calm-mind"
]

ITEMS = [
    "leftovers", "choice-band", "choice-scarf", "choice-specs", "life-orb",
    "focus-sash", "assault-vest", "rocky-helmet", "weakness-policy", "sitrus-berry"
]

ABILITIES = [
    "intimidate", "levitate", "multiscale", "magic-guard", "speed-boost",
    "download", "adaptability", "technician", "guts", "marvel-scale"
]

class MockDataGenerator:
    """Generate mock data for testing all components"""
    
    def __init__(self, seed: int = 42):
        """Initialize with reproducible random seed"""
        random.seed(seed)
        self.temp_dir = None
    
    def create_temp_dir(self) -> Path:
        """Create temporary directory for test data"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="pokeagent_test_"))
        return self.temp_dir
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def generate_mock_team(self) -> List[Dict[str, Any]]:
        """Generate a realistic Pokemon team"""
        team = []
        used_species = set()
        
        for _ in range(6):
            # Ensure no duplicate species
            species = random.choice(POKEMON_SPECIES)
            while species in used_species:
                species = random.choice(POKEMON_SPECIES)
            used_species.add(species)
            
            pokemon = {
                "species": species,
                "level": random.choice([50, 100]),
                "ability": random.choice(ABILITIES),
                "item": random.choice(ITEMS + [None]),
                "moves": random.sample(MOVES, 4),
                "evs": {
                    "hp": random.choice([0, 4, 252]),
                    "atk": random.choice([0, 4, 252]),
                    "def": random.choice([0, 4, 252]),
                    "spa": random.choice([0, 4, 252]),
                    "spd": random.choice([0, 4, 252]),
                    "spe": random.choice([0, 4, 252])
                },
                "nature": random.choice([
                    "adamant", "modest", "jolly", "timid", "bold", "calm",
                    "impish", "careful", "hasty", "naive", "hardy"
                ])
            }
            team.append(pokemon)
        
        return team
    
    def generate_mock_replay(self, game_id: str = None) -> str:
        """Generate a realistic Pokemon Showdown replay log"""
        if game_id is None:
            game_id = f"gen9ou-{random.randint(1000000, 9999999)}"
        
        p1_team = self.generate_mock_team()
        p2_team = self.generate_mock_team()
        
        # Generate realistic battle log
        lines = [
            f"|gametype|ou",
            f"|tier|[Gen 9] OU",
            f"|id|{game_id}",
            f"|player|p1|TestPlayer1|1200|",
            f"|player|p2|TestPlayer2|1250|",
            f"|teamsize|p1|6",
            f"|teamsize|p2|6",
            f"|gen|9",
            f"|tier|[Gen 9] OU",
            f"|rule|Species Clause: Limit one of each Pokémon",
            f"|rule|OHKO Clause: OHKO moves are banned",
            f"|rule|Sleep Clause Mod: Limit one foe put to sleep",
            f"",
            f"|start",
            f"|switch|p1a: {p1_team[0]['species']}|{p1_team[0]['species']}, L{p1_team[0]['level']}, M|100/100",
            f"|switch|p2a: {p2_team[0]['species']}|{p2_team[0]['species']}, L{p2_team[0]['level']}, F|100/100",
        ]
        
        # Generate some battle turns
        num_turns = random.randint(10, 50)
        current_p1 = p1_team[0]
        current_p2 = p2_team[0]
        
        for turn in range(1, num_turns + 1):
            lines.append(f"|turn|{turn}")
            
            # Random actions
            if random.random() < 0.8:  # 80% moves, 20% switches
                move = random.choice(current_p1['moves'])
                lines.append(f"|move|p1a: {current_p1['species']}|{move}|p2a: {current_p2['species']}")
                
                # Random damage
                if random.random() < 0.7:
                    damage = random.randint(10, 40)
                    remaining = max(0, 100 - damage)
                    lines.append(f"|-damage|p2a: {current_p2['species']}|{remaining}/100")
            else:
                # Switch
                new_mon = random.choice(p1_team[1:])
                lines.append(f"|switch|p1a: {new_mon['species']}|{new_mon['species']}, L{new_mon['level']}, M|100/100")
                current_p1 = new_mon
            
            # P2 action
            if random.random() < 0.8:
                move = random.choice(current_p2['moves'])
                lines.append(f"|move|p2a: {current_p2['species']}|{move}|p1a: {current_p1['species']}")
                
                if random.random() < 0.7:
                    damage = random.randint(10, 40)
                    remaining = max(0, 100 - damage)
                    lines.append(f"|-damage|p1a: {current_p1['species']}|{remaining}/100")
            
            # Random end condition
            if turn > 5 and random.random() < 0.1:
                lines.extend([
                    f"|faint|p2a: {current_p2['species']}",
                    f"|win|TestPlayer1"
                ])
                break
        
        return "\n".join(lines)
    
    def create_mock_replay_files(self, num_files: int = 100, output_dir: Path = None) -> Path:
        """Create mock replay files for testing"""
        if output_dir is None:
            output_dir = self.create_temp_dir() / "replays"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_files):
            game_id = f"gen9ou-{1000000 + i}"
            replay_content = self.generate_mock_replay(game_id)
            
            # Save as compressed JSON (like real data)
            replay_data = {
                "id": game_id,
                "format": "gen9ou",
                "log": replay_content,
                "players": [
                    {"username": "TestPlayer1", "rating": random.randint(1000, 2000)},
                    {"username": "TestPlayer2", "rating": random.randint(1000, 2000)}
                ],
                "endType": "normal",
                "winner": random.choice(["TestPlayer1", "TestPlayer2"])
            }
            
            file_path = output_dir / f"{game_id}.json.gz"
            with gzip.open(file_path, 'wt') as f:
                json.dump(replay_data, f)
        
        return output_dir
    
    def create_mock_processed_dataset(self, num_samples: int = 1000, output_dir: Path = None) -> Path:
        """Create mock processed dataset for training"""
        if output_dir is None:
            output_dir = self.create_temp_dir() / "processed"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training samples
        samples = []
        for i in range(num_samples):
            # Mock state representation
            state = {
                'species_ids': [random.randint(0, 1024) for _ in range(12)],
                'move_ids': [random.randint(0, 849) for _ in range(12)],
                'item_ids': [random.randint(0, 399) for _ in range(12)],
                'ability_ids': [random.randint(0, 299) for _ in range(12)],
                'hp_values': [random.random() for _ in range(12)],
                'stat_boosts': [[random.randint(-6, 6) for _ in range(7)] for _ in range(12)],
                'status_ids': [random.randint(0, 7) for _ in range(12)],
                'weather_ids': [random.randint(0, 7) for _ in range(12)],
                'terrain_ids': [random.randint(0, 4) for _ in range(12)],
                'position_ids': list(range(12))
            }
            
            # Mock action
            action = random.randint(0, 9)  # 4 moves + 6 switches
            
            # Mock reward
            reward = random.choice([0, 1])  # Win/loss
            
            sample = {
                'state': state,
                'action': action,
                'reward': reward,
                'game_id': f"game_{i}",
                'turn': random.randint(1, 50)
            }
            samples.append(sample)
        
        # Save as pickle for fast loading
        with open(output_dir / "training_data.pkl", 'wb') as f:
            pickle.dump(samples, f)
        
        # Create metadata
        metadata = {
            'num_samples': num_samples,
            'format': 'processed_replays',
            'version': '1.0',
            'features': list(samples[0]['state'].keys())
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_dir
    
    def create_mock_checkpoint(self, step: int = 1000, output_dir: Path = None) -> Path:
        """Create mock training checkpoint"""
        if output_dir is None:
            output_dir = self.create_temp_dir() / "checkpoints"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock checkpoint data (without actual model weights to save space)
        checkpoint_data = {
            'step': step,
            'epoch': step // 1000,
            'best_val_acc': random.uniform(0.1, 0.8),
            'config': {
                'hidden_size': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'vocab_size': 50000
            },
            'training_time': step * 0.1,  # Mock training time
            'timestamp': 1692000000 + step
        }
        
        checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
        
        # In real implementation, this would be torch.save()
        # For testing, just save as pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return checkpoint_path
    
    def simulate_multi_node_env(self) -> Dict[str, str]:
        """Simulate multi-node environment variables"""
        return {
            'SLURM_PROCID': '0',
            'SLURM_LOCALID': '0',
            'SLURM_NODEID': '0',
            'SLURM_NNODES': '2',
            'SLURM_NTASKS': '8',
            'SLURM_GPUS_PER_NODE': '4',
            'MASTER_ADDR': '10.0.0.1',
            'MASTER_PORT': '29500',
            'WORLD_SIZE': '8',
            'LOCAL_WORLD_SIZE': '4'
        }

# Test the mock data generator
if __name__ == "__main__":
    generator = MockDataGenerator()
    
    print("🧪 Testing Mock Data Generator...")
    
    # Test team generation
    team = generator.generate_mock_team()
    print(f"✅ Generated team with {len(team)} Pokemon")
    
    # Test replay generation
    replay = generator.generate_mock_replay()
    print(f"✅ Generated replay with {len(replay.split('|'))} events")
    
    # Test file creation
    temp_dir = generator.create_temp_dir()
    
    replay_dir = generator.create_mock_replay_files(10)
    print(f"✅ Created {len(list(replay_dir.glob('*.json.gz')))} replay files")
    
    dataset_dir = generator.create_mock_processed_dataset(100)
    print(f"✅ Created processed dataset with 100 samples")
    
    checkpoint_path = generator.create_mock_checkpoint(5000)
    print(f"✅ Created mock checkpoint at {checkpoint_path}")
    
    # Test environment simulation
    env_vars = generator.simulate_multi_node_env()
    print(f"✅ Simulated multi-node environment with {len(env_vars)} variables")
    
    # Cleanup
    generator.cleanup()
    print("✅ Cleaned up temporary files")
    
    print("🎉 All mock data tests passed!")
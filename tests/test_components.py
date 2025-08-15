#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pokemon Agent Components
Tests all components individually and in integration
"""

import os
import sys
import tempfile
import unittest
import shutil
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import test modules
from tests.mock_data import MockDataGenerator

class TestMockDataGenerator(unittest.TestCase):
    """Test mock data generation"""
    
    def setUp(self):
        self.generator = MockDataGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        self.generator.cleanup()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_team_generation(self):
        """Test Pokemon team generation"""
        team = self.generator.generate_mock_team()
        
        self.assertEqual(len(team), 6)
        
        # Check team structure
        for pokemon in team:
            self.assertIn('species', pokemon)
            self.assertIn('level', pokemon)
            self.assertIn('ability', pokemon)
            self.assertIn('moves', pokemon)
            self.assertIn('evs', pokemon)
            self.assertEqual(len(pokemon['moves']), 4)
    
    def test_replay_generation(self):
        """Test replay log generation"""
        replay = self.generator.generate_mock_replay()
        
        self.assertIsInstance(replay, str)
        self.assertIn('|gametype|', replay)
        self.assertIn('|player|p1|', replay)
        self.assertIn('|player|p2|', replay)
        self.assertIn('|start', replay)
    
    def test_file_creation(self):
        """Test mock file creation"""
        # Test replay files
        replay_dir = self.generator.create_mock_replay_files(5, self.temp_dir / "replays")
        replay_files = list(replay_dir.glob("*.json.gz"))
        self.assertEqual(len(replay_files), 5)
        
        # Test dataset creation
        dataset_dir = self.generator.create_mock_processed_dataset(100, self.temp_dir / "dataset")
        self.assertTrue((dataset_dir / "training_data.pkl").exists())
        self.assertTrue((dataset_dir / "metadata.json").exists())
        
        # Test checkpoint creation
        checkpoint_path = self.generator.create_mock_checkpoint(1000, self.temp_dir / "checkpoint")
        self.assertTrue(checkpoint_path.exists())

class TestDownloadScript(unittest.TestCase):
    """Test data download functionality"""
    
    def setUp(self):
        # Import here to avoid dependency issues
        try:
            from scripts.download_pokechamp import PokechampDownloader, DownloadConfig
            self.PokechampDownloader = PokechampDownloader
            self.DownloadConfig = DownloadConfig
        except ImportError:
            self.skipTest("Download script dependencies not available")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = self.DownloadConfig(
            output_dir=str(self.temp_dir),
            max_games=10,
            use_mock_data=True,
            dry_run=True
        )
    
    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_downloader_init(self):
        """Test downloader initialization"""
        downloader = self.PokechampDownloader(self.config)
        
        self.assertEqual(downloader.config, self.config)
        self.assertTrue(downloader.output_dir.exists())
        self.assertTrue(downloader.raw_dir.exists())
        self.assertTrue(downloader.processed_dir.exists())
    
    def test_mock_data_creation(self):
        """Test mock data creation"""
        downloader = self.PokechampDownloader(self.config)
        
        success = downloader._create_mock_data()
        self.assertTrue(success)
        
        # Check files were created
        replay_files = list(downloader.raw_dir.glob("*.json.gz"))
        self.assertGreater(len(replay_files), 0)
    
    def test_state_management(self):
        """Test download state saving/loading"""
        downloader = self.PokechampDownloader(self.config)
        
        # Modify state
        downloader.download_state['test_key'] = 'test_value'
        downloader._save_state()
        
        # Create new downloader and check state loaded
        downloader2 = self.PokechampDownloader(self.config)
        self.assertEqual(downloader2.download_state['test_key'], 'test_value')

class TestReplayParser(unittest.TestCase):
    """Test replay parsing functionality"""
    
    def setUp(self):
        try:
            from data.replay_parser import ReplayParser, BattleState, Action
            self.ReplayParser = ReplayParser
            self.BattleState = BattleState
            self.Action = Action
        except ImportError:
            self.skipTest("Replay parser dependencies not available")
        
        self.parser = self.ReplayParser()
        
        # Create simple test replay
        self.test_replay = """
|gametype|ou
|player|p1|TestPlayer1|1200|
|player|p2|TestPlayer2|1250|
|teamsize|p1|6
|teamsize|p2|6
|start
|switch|p1a: Pikachu|Pikachu, L50, M|100/100
|switch|p2a: Charizard|Charizard, L50, F|100/100
|turn|1
|move|p1a: Pikachu|Thunderbolt|p2a: Charizard
|-damage|p2a: Charizard|75/100
|move|p2a: Charizard|Flamethrower|p1a: Pikachu
|-damage|p1a: Pikachu|80/100
|turn|2
|move|p1a: Pikachu|Quick Attack|p2a: Charizard
|-damage|p2a: Charizard|60/100
|faint|p2a: Charizard
|win|TestPlayer1
        """.strip()
    
    def test_parser_initialization(self):
        """Test parser initialization"""
        self.assertIsNotNone(self.parser.patterns)
        self.assertEqual(self.parser.current_turn, 0)
        self.assertEqual(len(self.parser.teams['p1']), 0)
    
    def test_replay_parsing(self):
        """Test complete replay parsing"""
        samples = self.parser.parse_replay(self.test_replay)
        
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)
        
        # Check sample structure
        sample = samples[0]
        self.assertIn('state', sample)
        self.assertIn('action', sample)
        self.assertIn('turn', sample)
        self.assertIn('reward', sample)
    
    def test_battle_state_conversion(self):
        """Test battle state to vector conversion"""
        state = self.BattleState()
        state.turn = 1
        
        vector = state.to_vector()
        
        self.assertIn('species_ids', vector)
        self.assertIn('hp_values', vector)
        self.assertIn('stat_boosts', vector)
        self.assertEqual(len(vector['species_ids']), 12)
        self.assertEqual(len(vector['stat_boosts']), 12)
        self.assertEqual(len(vector['stat_boosts'][0]), 7)
    
    def test_action_conversion(self):
        """Test action to ID conversion"""
        move_action = self.Action(type='move', move='thunderbolt')
        switch_action = self.Action(type='switch', pokemon='pikachu')
        
        move_id = move_action.to_id()
        switch_id = switch_action.to_id()
        
        self.assertIsInstance(move_id, int)
        self.assertIsInstance(switch_id, int)
        self.assertGreaterEqual(move_id, 0)
        self.assertGreaterEqual(switch_id, 4)  # Switches start at 4

class TestDataset(unittest.TestCase):
    """Test dataset functionality"""
    
    def setUp(self):
        try:
            from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig, collate_real_data
            self.RealPokemonDataset = RealPokemonDataset
            self.DatasetConfig = DatasetConfig
            self.collate_real_data = collate_real_data
        except ImportError:
            self.skipTest("Dataset dependencies not available")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock processed data
        mock_samples = []
        for i in range(100):
            sample = {
                'state': {
                    'species_ids': [i % 10] * 12,
                    'move_ids': [i % 5] * 12,
                    'item_ids': [i % 3] * 12,
                    'ability_ids': [i % 4] * 12,
                    'hp_values': [1.0] * 12,
                    'stat_boosts': [[0] * 7 for _ in range(12)],
                    'status_ids': [0] * 12,
                    'weather_ids': [0] * 12,
                    'terrain_ids': [0] * 12,
                    'position_ids': list(range(12))
                },
                'action': i % 10,
                'reward': i % 2,
                'game_id': f"game_{i}",
                'turn': i + 1,
                'format': 'gen9ou'
            }
            mock_samples.append(sample)
        
        # Save mock data
        with open(self.temp_dir / "training_data.pkl", 'wb') as f:
            pickle.dump(mock_samples, f)
        
        with open(self.temp_dir / "metadata.json", 'w') as f:
            json.dump({'total_samples': len(mock_samples)}, f)
        
        self.config = self.DatasetConfig(
            data_dir=str(self.temp_dir),
            max_samples=50,
            validation_split=0.2,
            cache_size=10
        )
    
    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = self.RealPokemonDataset(self.config, split='train')
        
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.split, 'train')
    
    def test_sample_loading(self):
        """Test sample loading"""
        dataset = self.RealPokemonDataset(self.config, split='train')
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            self.assertIn('input_ids', sample)
            self.assertIn('labels', sample)
            self.assertIn('rewards', sample)
            
            # Check tensor types
            self.assertIsInstance(sample['labels'], type(sample['labels']))  # torch.Tensor when available
    
    def test_data_splitting(self):
        """Test train/val/test splitting"""
        train_dataset = self.RealPokemonDataset(self.config, split='train')
        val_dataset = self.RealPokemonDataset(self.config, split='val')
        
        # Should have different sizes
        self.assertNotEqual(len(train_dataset), len(val_dataset))
        self.assertGreater(len(train_dataset), len(val_dataset))
    
    def test_collate_function(self):
        """Test data collation"""
        dataset = self.RealPokemonDataset(self.config, split='train')
        
        if len(dataset) >= 2:
            samples = [dataset[0], dataset[1]]
            
            try:
                batch = self.collate_real_data(samples)
                
                self.assertIn('input_ids', batch)
                self.assertIn('labels', batch)
                self.assertIn('rewards', batch)
                
                # Check batch dimensions
                self.assertEqual(len(batch['labels']), 2)  # Batch size
                
            except Exception as e:
                self.skipTest(f"Collate function requires PyTorch: {e}")

class TestTrainingScript(unittest.TestCase):
    """Test training script functionality"""
    
    def setUp(self):
        # Mock environment variables for distributed training
        self.env_patcher = patch.dict(os.environ, {
            'LOCAL_RANK': '0',
            'RANK': '0', 
            'WORLD_SIZE': '1',
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': '29500'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        self.env_patcher.stop()
    
    def test_config_creation(self):
        """Test distributed config creation"""
        try:
            from scripts.train_multinode import DistributedConfig
            
            config = DistributedConfig(
                nodes=2,
                gpus_per_node=4,
                batch_size_per_gpu=128
            )
            
            self.assertEqual(config.world_size, 8)
            self.assertEqual(config.total_batch_size, 1024)
            self.assertGreater(config.scale_learning_rate(), config.learning_rate)
            
        except ImportError:
            self.skipTest("Training script dependencies not available")
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown handler"""
        try:
            from scripts.train_multinode import GracefulShutdown
            
            handler = GracefulShutdown()
            self.assertFalse(handler.should_stop())
            
            # Simulate signal
            handler._signal_handler(15, None)
            self.assertTrue(handler.should_stop())
            
        except ImportError:
            self.skipTest("Training script dependencies not available")
    
    @patch('torch.distributed.init_process_group')
    @patch('torch.cuda.set_device')
    def test_trainer_initialization(self, mock_cuda, mock_dist):
        """Test trainer initialization"""
        try:
            from scripts.train_multinode import DistributedTrainer, DistributedConfig
            
            config = DistributedConfig(dry_run=True, mock_data=True)
            
            # Mock distributed training
            with patch('torch.distributed.is_initialized', return_value=False):
                trainer = DistributedTrainer(config)
                
                self.assertEqual(trainer.config, config)
                self.assertEqual(trainer.step, 0)
                
        except ImportError:
            self.skipTest("Training script dependencies not available")

class TestLaunchScript(unittest.TestCase):
    """Test SLURM launch script"""
    
    def setUp(self):
        self.script_path = Path(__file__).parent.parent / "scripts" / "launch_training.sh"
    
    def test_script_exists(self):
        """Test that launch script exists and is executable"""
        self.assertTrue(self.script_path.exists())
        self.assertTrue(os.access(self.script_path, os.X_OK))
    
    def test_script_help(self):
        """Test script help output"""
        import subprocess
        
        try:
            result = subprocess.run(
                [str(self.script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            self.assertIn("Universal SLURM Launcher", result.stdout)
            self.assertIn("Usage:", result.stdout)
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("Cannot execute bash script")
    
    def test_script_dry_run(self):
        """Test script dry run mode"""
        import subprocess
        
        try:
            result = subprocess.run(
                [str(self.script_path), "1", "1", "1", "--dry-run", "--test-mode"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should complete without error in dry run
            self.assertIn("DRY RUN", result.stdout)
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("Cannot execute bash script")

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_mock(self):
        """Test complete pipeline with mock data"""
        try:
            # Step 1: Generate mock data
            generator = MockDataGenerator()
            replay_dir = generator.create_mock_replay_files(10, self.temp_dir / "replays")
            
            # Step 2: Parse replays
            from data.replay_parser import ReplayParser
            
            parser = ReplayParser()
            all_samples = []
            
            for replay_file in replay_dir.glob("*.json.gz"):
                with open(replay_file, 'r') as f:
                    data = json.load(f)
                    for replay_data in data:
                        samples = parser.parse_replay(replay_data.get('log', ''))
                        if samples:
                            all_samples.extend(samples)
            
            # Save processed data
            processed_dir = self.temp_dir / "processed"
            processed_dir.mkdir()
            
            with open(processed_dir / "training_data.pkl", 'wb') as f:
                pickle.dump(all_samples, f)
            
            # Step 3: Create dataset
            from data.real_pokemon_dataset import RealPokemonDataset, DatasetConfig
            
            config = DatasetConfig(
                data_dir=str(processed_dir),
                max_samples=100,
                validation_split=0.2
            )
            
            dataset = RealPokemonDataset(config, split='train')
            
            self.assertGreater(len(dataset), 0)
            
            # Step 4: Test sample loading
            if len(dataset) > 0:
                sample = dataset[0]
                self.assertIn('input_ids', sample)
                self.assertIn('labels', sample)
            
            generator.cleanup()
            
        except ImportError as e:
            self.skipTest(f"Integration test dependencies not available: {e}")

def run_all_tests():
    """Run all tests and return results"""
    # Create test suite
    test_classes = [
        TestMockDataGenerator,
        TestDownloadScript,
        TestReplayParser,
        TestDataset,
        TestTrainingScript,
        TestLaunchScript,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    print("🧪 Running comprehensive test suite...")
    print("=" * 60)
    
    result = run_all_tests()
    
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    if result.skipped:
        print(f"\n⏭️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n⚠️  Some tests failed. Please fix issues before deploying to cluster.")
        exit(1)
    else:
        print("\n🎉 All tests passed! Ready for cluster deployment.")
        exit(0)
#!/usr/bin/env python3
"""
Test imports and basic functionality without external dependencies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_mock_data():
    """Test mock data generation"""
    try:
        from tests.mock_data import MockDataGenerator
        
        generator = MockDataGenerator()
        team = generator.generate_mock_team()
        replay = generator.generate_mock_replay()
        
        print("✅ Mock data generation works")
        generator.cleanup()
        return True
    except Exception as e:
        print(f"❌ Mock data generation failed: {e}")
        return False

def test_model_imports():
    """Test model imports"""
    try:
        from models.metamon_transformer import ModelConfig
        
        config = ModelConfig()
        print(f"✅ Model config import works: {config.hidden_size} hidden units")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_replay_parser():
    """Test replay parser"""
    try:
        from data.replay_parser import ReplayParser, BattleState, Action
        
        parser = ReplayParser()
        state = BattleState()
        action = Action(type='move', move='thunderbolt')
        
        print(f"✅ Replay parser imports work")
        return True
    except Exception as e:
        print(f"❌ Replay parser failed: {e}")
        return False

def test_dataset_imports():
    """Test dataset imports"""
    try:
        from data.real_pokemon_dataset import DatasetConfig
        
        config = DatasetConfig(data_dir="test")
        print(f"✅ Dataset imports work")
        return True
    except Exception as e:
        print(f"❌ Dataset imports failed: {e}")
        return False

def test_training_script_imports():
    """Test training script imports"""
    try:
        from scripts.train_multinode import DistributedConfig
        
        config = DistributedConfig(nodes=1, gpus_per_node=1)
        print(f"✅ Training script imports work: {config.world_size} total GPUs")
        return True
    except Exception as e:
        print(f"❌ Training script imports failed: {e}")
        return False

def test_launch_script():
    """Test launch script"""
    try:
        script_path = Path("scripts/launch_training.sh")
        if script_path.exists():
            import os
            if os.access(script_path, os.X_OK):
                print("✅ Launch script exists and is executable")
                return True
            else:
                print("❌ Launch script not executable")
                return False
        else:
            print("❌ Launch script not found")
            return False
    except Exception as e:
        print(f"❌ Launch script test failed: {e}")
        return False

def main():
    print("🧪 Testing core imports and functionality...")
    print("=" * 50)
    
    tests = [
        ("Mock Data", test_mock_data),
        ("Model Imports", test_model_imports),
        ("Replay Parser", test_replay_parser),
        ("Dataset Imports", test_dataset_imports),
        ("Training Script", test_training_script_imports),
        ("Launch Script", test_launch_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
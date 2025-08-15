#!/usr/bin/env python3
"""
Debug script to isolate the model forward pass issue
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.metamon_transformer import MetamonTransformer, ModelConfig, StateEncoder

def debug_model():
    print("🔍 Debugging model forward pass...")
    
    # Create model
    config = ModelConfig()
    print(f"Model config: {config}")
    
    model = MetamonTransformer(config)
    print(f"✅ Model created: {model.get_parameter_count():,} parameters")
    
    # Create encoder
    encoder = StateEncoder()
    print("✅ Encoder created")
    
    # Generate dummy input
    dummy_input = encoder.encode_battle_state(None)
    print("✅ Dummy input generated")
    
    print("\n📏 Input shapes:")
    for key, value in dummy_input.items():
        print(f"  {key}: {value.shape} ({value.dtype})")
    
    try:
        print("\n🚀 Testing embeddings...")
        with torch.no_grad():
            # Test embeddings layer
            embeddings = model.embeddings(
                species_ids=dummy_input['species_ids'],
                move_ids=dummy_input['move_ids'],
                item_ids=dummy_input['item_ids'],
                ability_ids=dummy_input['ability_ids'],
                hp_values=dummy_input['hp_values'],
                stat_boosts=dummy_input['stat_boosts'],
                status_ids=dummy_input['status_ids'],
                weather_ids=dummy_input['weather_ids'],
                terrain_ids=dummy_input['terrain_ids'],
                position_ids=dummy_input['position_ids']
            )
            print(f"✅ Embeddings shape: {embeddings.shape}")
            
            # Test first transformer layer
            print("\n🔄 Testing first transformer layer...")
            layer_output = model.layers[0](embeddings)
            print(f"✅ First layer output shape: {layer_output.shape}")
            
            # Test full forward pass
            print("\n🎯 Testing full forward pass...")
            output = model(dummy_input)
            print("✅ Full forward pass successful!")
            
            print("\n📤 Output shapes:")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = debug_model()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
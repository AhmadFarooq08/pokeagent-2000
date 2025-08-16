# Complete Pokemon Agent 2000+ ELO Implementation Guide

## CRITICAL CONTEXT & CURRENT STATE

### Current Performance
- **Current ELO**: 1020 (10% win rate) - COMPLETELY UNACCEPTABLE
- **Target ELO**: 2000+ (bare minimum)
- **Gap**: Need ~1000 ELO improvement

### Major Issues Discovered

#### 1. CRITICAL DATA PROBLEM
Your downloaded data is **COMPLETELY EMPTY**:
```json
{"id": "game_0_0", "log": "", "winner": "", "players": [], "format": "gen9ou", "endType": "normal"}
```
- Every game has empty log, winner, and players
- Located in: `data/pokechamp/raw/*.json.gz`
- This is placeholder data, NOT real Pokemon battles

#### 2. Fundamental Approach Flaw
- You were using **pure Behavior Cloning** (just cross-entropy loss)
- **NEVER used rewards or value functions**
- This can NEVER achieve 2000+ ELO
- Metamon uses **Offline RL with advantage weighting**

#### 3. Data Format Mismatch
- Code expects: `sample['state']`, `sample['action']`, `sample['reward']`
- Actual data has: `sample['log']`, `sample['winner']`, `sample['players']`
- Need to parse battle logs into state-action-reward tuples

### Hardware Resources
- **4x NVIDIA A100-40GB GPUs** (160GB total VRAM)
- Single node: c2105
- Time remaining: ~2 hours on current allocation

### Repository Structure
```
~/pokeagent-2000/
├── metamon/                 # UT-Austin Metamon (just cloned)
├── data/
│   ├── pokechamp/          # Empty placeholder data
│   │   ├── raw/            # .json.gz files with empty games
│   │   └── processed/      # Empty directory
│   └── processed_replays/  # Check if this has real data
├── models/                 # Your model implementations
├── scripts/               # Training scripts
└── checkpoints_4gpu/      # Where checkpoints would save
```

## METAMON ANALYSIS (What Makes It Work)

### Key Components for 2000+ ELO
1. **200M Parameter Transformer** - Large enough for complex strategies
2. **Offline RL Training** - Not behavior cloning!
3. **Advantage Weighting** - Learns from good vs bad actions
4. **Synthetic Data Augmentation** - Prevents overfitting
5. **Proper State Representation** - Encodes full game state

### Metamon's Training Pipeline
```python
# Simplified Metamon approach
for batch in dataloader:
    states, actions, rewards = batch
    
    # 1. Value prediction
    values = model.get_values(states)
    value_loss = MSE(values, rewards)
    
    # 2. Advantage computation
    advantages = rewards - values.detach()
    weights = torch.where(advantages > 0, 1.0, 0.2)  # Upweight good actions
    
    # 3. Policy loss with advantage weighting
    policy_logits = model.get_policy(states)
    policy_loss = CrossEntropy(policy_logits, actions) * weights
    
    # 4. Combined loss
    total_loss = policy_loss + 0.5 * value_loss
```

## COMPLETE IMPLEMENTATION PLAN

### Phase 1: Understand Metamon Structure [30 minutes]
```bash
cd ~/pokeagent-2000/metamon

# 1. Check their README and requirements
cat README.md
cat requirements.txt

# 2. Find their main training script
find . -name "*.py" | grep -E "(train|main)" | head -10

# 3. Understand their data format
find . -name "*.json" -o -name "*.pkl" | head -5
# Check if they have example data
ls data/ 2>/dev/null || ls dataset/ 2>/dev/null || ls examples/ 2>/dev/null

# 4. Find their model architecture
find . -name "*.py" | xargs grep -l "class.*Model\|class.*Transformer" | head -5

# 5. Check their config files
find . -name "*.yaml" -o -name "*.yml" -o -name "*config*.py" | head -10
```

### Phase 2: Get REAL Pokemon Data [1 hour]

#### Option A: Find Metamon's Data Source
```bash
# Check if Metamon has data download scripts
find ~/pokeagent-2000/metamon -name "*.py" | xargs grep -l "download\|dataset\|huggingface"

# Check their data processing pipeline
find ~/pokeagent-2000/metamon -name "*.py" | xargs grep -l "process.*replay\|parse.*log"
```

#### Option B: Get Real Pokemon Showdown Data
```python
# Create: ~/pokeagent-2000/scripts/get_real_data.py
import requests
import json
import gzip
from pathlib import Path

# Sources for real Pokemon battle data:
# 1. Pokemon Showdown replay database
# 2. Poke-env recorded battles
# 3. Metamon's dataset (if publicly available)

def download_real_replays():
    """Download actual Pokemon battle replays with logs"""
    # This would fetch from:
    # - replay.pokemonshowdown.com
    # - Or HuggingFace datasets with real battles
    # - Or Metamon's training data
    pass
```

#### Option C: Use Pokemon Showdown Simulator
```bash
# Install Pokemon Showdown locally to generate battles
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
# Generate battles with actual game logs
```

### Phase 3: Process Battle Logs into Training Data [2 hours]

```python
# Create: ~/pokeagent-2000/scripts/process_replays.py
import json
import numpy as np
from typing import List, Dict, Tuple

class BattleProcessor:
    """Convert battle logs to state-action-reward tuples"""
    
    def parse_battle_log(self, log: str) -> List[Dict]:
        """
        Parse battle log into training samples
        
        Input: Raw battle log string
        Output: List of {state, action, reward, next_state}
        """
        samples = []
        lines = log.split('\n')
        
        current_state = self.initialize_state()
        
        for line in lines:
            if line.startswith('|move|'):
                # Extract move action
                action = self.parse_move(line)
                
            elif line.startswith('|switch|'):
                # Extract switch action
                action = self.parse_switch(line)
                
            elif line.startswith('|win|'):
                # Extract winner for rewards
                winner = self.parse_winner(line)
                
            # Build state representation
            state_vector = self.encode_state(current_state)
            
            samples.append({
                'state': state_vector,
                'action': action,
                'reward': reward,
                'done': is_battle_over
            })
            
        return samples
    
    def encode_state(self, game_state) -> np.ndarray:
        """
        Encode game state as vector
        
        Includes:
        - Active Pokemon (both sides)
        - HP values
        - Status conditions
        - Field conditions
        - Team composition
        """
        # This is what your model expects
        return state_vector
```

### Phase 4: Fix Data Loading Pipeline [1 hour]

```python
# Fix: ~/pokeagent-2000/data/real_pokemon_dataset.py

def _convert_sample_to_tensors(self, sample: Dict) -> Dict[str, torch.Tensor]:
    """Convert sample to PyTorch tensors"""
    
    # Handle different data formats
    if 'state' in sample:
        # Pre-processed format
        state = sample['state']
        action = sample['action']
        reward = sample.get('reward', 0)
        
    elif 'log' in sample:
        # Raw replay format - need to process
        if not sample['log']:  # Empty log
            return self._get_dummy_sample()
            
        # Process the battle log
        processor = BattleProcessor()
        processed = processor.parse_battle_log(sample['log'])
        
        if not processed:
            return self._get_dummy_sample()
            
        # Take first sample (or implement sequence handling)
        state = processed[0]['state']
        action = processed[0]['action']
        
        # Compute reward from winner
        reward = 1.0 if sample['winner'] == 'player1' else 0.0
        
    else:
        # Unknown format
        logger.error(f"Unknown sample format: {sample.keys()}")
        return self._get_dummy_sample()
    
    # Convert to tensors
    return {
        'input_ids': torch.tensor(state, dtype=torch.float32),
        'labels': torch.tensor(action, dtype=torch.long),
        'rewards': torch.tensor(reward, dtype=torch.float32)
    }
```

### Phase 5: Implement Metamon's Training Approach [2 hours]

```python
# Update: ~/pokeagent-2000/scripts/train_metamon.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from models.metamon_transformer import MetamonTransformer

class MetamonTrainer:
    def __init__(self, config):
        self.model = MetamonTransformer(config)
        self.optimizer = AdamW(self.model.parameters(), lr=3e-4)
        
    def compute_advantages(self, rewards, values, gamma=0.99):
        """Compute advantages for offline RL"""
        # For offline RL, advantages = rewards - values
        advantages = rewards - values.detach()
        
        # Metamon's "Binary" variant
        weights = torch.where(advantages > 0, 1.0, 0.2)
        return weights
    
    def train_step(self, batch):
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        
        # Forward pass
        policy_logits, values = self.model(states)
        
        # 1. Value loss (train critic)
        value_loss = nn.functional.mse_loss(values.squeeze(), rewards)
        
        # 2. Compute advantages
        weights = self.compute_advantages(rewards, values)
        
        # 3. Policy loss (train actor) with advantage weighting
        policy_loss = nn.functional.cross_entropy(
            policy_logits.view(-1, policy_logits.size(-1)),
            actions.view(-1),
            reduction='none'
        )
        policy_loss = (policy_loss * weights.view(-1)).mean()
        
        # 4. Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_value': values.mean().item(),
            'mean_advantage': weights.mean().item()
        }
```

### Phase 6: Training Execution [2-6 hours]

```bash
# 1. First test with small data to ensure pipeline works
cd ~/pokeagent-2000
python scripts/test_data_pipeline.py --num-samples 100

# 2. If test passes, launch full training
torchrun --master_port=29501 --nproc_per_node=4 scripts/train_metamon.py \
    --model-size 200M \
    --batch-size-per-gpu 64 \
    --learning-rate 3e-4 \
    --max-steps 100000 \
    --data-dir data/processed_battles \
    --output-dir checkpoints_metamon \
    --checkpoint-minutes 10 \
    --use-offline-rl \
    --advantage-weighting \
    --log-every 100

# 3. Monitor training
tail -f checkpoints_metamon/training.log

# 4. Check metrics
grep "elo\|win_rate" checkpoints_metamon/training.log | tail -20
```

## CRITICAL DECISION POINTS

### Decision 1: Data Source
**MUST RESOLVE IMMEDIATELY**

A. **Use Metamon's data pipeline** (RECOMMENDED)
   - They already solved this problem
   - Their data works for 2000+ ELO
   
B. **Download new dataset**
   - Risk: Might get empty data again
   - Need to verify data quality first
   
C. **Generate synthetic battles**
   - Use Pokemon Showdown simulator
   - Time-consuming but guaranteed quality

### Decision 2: Implementation Approach

A. **Adapt Metamon code directly** (FASTEST)
   ```bash
   cd ~/pokeagent-2000/metamon
   # Modify their config for your needs
   python train.py --your-config
   ```

B. **Port Metamon approach to your code** (RECOMMENDED)
   - Take their working components
   - Integrate into your pipeline
   - Maintain your code structure

C. **Start fresh with Metamon architecture**
   - Most time-consuming
   - Highest risk of errors

## TROUBLESHOOTING GUIDE

### Problem: "No data files found"
```bash
# Check what files exist
find ~/pokeagent-2000 -name "*.json" -o -name "*.pkl" | wc -l

# Check if files have content
zcat data/pokechamp/raw/train_chunk_0000.json.gz | python -c "import json,sys; d=json.load(sys.stdin); print(f'Has log: {bool(d[0][\"log\"])}' if d else 'Empty')"
```

### Problem: "Failed to convert sample to tensors: 'state'"
The data format doesn't match. Need to:
1. Check actual data structure
2. Update `_convert_sample_to_tensors` function
3. Or preprocess data into expected format

### Problem: Training loss not decreasing
Check if you're using:
- ✅ Offline RL loss (not just cross-entropy)
- ✅ Advantage weighting
- ✅ Value head training
- ✅ Proper learning rate (3e-4)
- ✅ Gradient clipping

### Problem: Low ELO after training
Verify:
1. Data quality (not empty logs)
2. Model size (200M parameters minimum)
3. Training steps (100k+ needed)
4. Reward signal (binary win/loss)

## IMMEDIATE NEXT STEPS (DO THIS NOW)

```bash
# 1. Check Metamon's structure
cd ~/pokeagent-2000/metamon
ls -la
cat README.md | head -100

# 2. Find their training command
grep -r "python.*train" . --include="*.md" --include="*.sh"

# 3. Check their data format
find . -name "*.json" | head -1 | xargs head -100

# 4. Locate their model file
find . -name "*.py" | xargs grep -l "class.*Model"

# 5. Run their training (if possible)
# Look for train.py or main.py
python train.py --help 2>/dev/null || python main.py --help 2>/dev/null
```

## SUCCESS CRITERIA

1. **Training runs without errors** ✓
2. **Loss decreases over time** ✓
3. **ELO increases during evaluation** ✓
4. **Reaches 1500 ELO** (intermediate goal)
5. **Reaches 2000+ ELO** (minimum target)
6. **Maintains 2000+ ELO consistently** (success)

## TIME ESTIMATES

- Understanding Metamon: 30 minutes
- Getting real data: 1-2 hours
- Fixing data pipeline: 1 hour
- Implementing training: 2 hours
- Training to 2000 ELO: 4-8 hours
- Total: 8-14 hours

With 2 hours remaining on current allocation:
1. Focus on understanding Metamon
2. Get data pipeline working
3. Start training
4. Resume in next allocation

## REASONING BEHIND EACH DECISION

### Why Metamon First?
- **Proven to work**: They achieve 2000+ ELO
- **Complete solution**: Data + Model + Training
- **Time efficient**: Faster than debugging from scratch
- **Learning opportunity**: See what actually works

### Why Offline RL Over Behavior Cloning?
- **Behavior cloning ceiling**: ~1200 ELO max
- **Offline RL potential**: 2000+ ELO proven
- **Advantage weighting**: Learns good vs bad actions
- **Value function**: Provides better credit assignment

### Why 200M Parameters?
- **Complexity**: Pokemon has huge state/action space
- **Metamon proof**: Their 200M model reaches 2000+ ELO
- **Transformer scaling**: Bigger models = better performance
- **GPU capacity**: 4x A100 can handle this size

### Why This Data Processing?
- **Current data is empty**: Must get real battles
- **Format mismatch**: Need state-action-reward tuples
- **Metamon compatibility**: Match their successful format
- **Training efficiency**: Preprocessed data trains faster

## FINAL CRITICAL NOTES

1. **YOUR DATA IS EMPTY** - This is the #1 problem to solve
2. **You need Offline RL** - Behavior cloning will never reach 2000 ELO
3. **Metamon is the blueprint** - Follow their approach exactly first
4. **Time is limited** - Focus on getting training started

Remember: 2000 ELO is the MINIMUM. Current 1020 ELO is completely unacceptable. Every decision should move toward that goal.
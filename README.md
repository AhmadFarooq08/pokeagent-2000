# PokeAgent-2000: Metamon Architecture Replication

A complete implementation of the Metamon architecture for achieving 2000+ ELO on the NeurIPS 2025 PokeAgent Challenge. This project replicates the state-of-the-art 200M parameter Transformer approach used by top-performing agents.

## 🎯 Target Performance
- **Goal**: 2000+ ELO on PokeAgent server
- **Current SOTA**: Metamon-Kadabra at 1917 ELO (Gen9OU)
- **Architecture**: 200M parameter Transformer with offline RL training

## 🏗️ Architecture Overview

### Core Components
1. **Transformer Model**: 200M parameter GPT-2 style architecture optimized for Pokemon battles
2. **Behavior Cloning**: Initial training on 3.5M human replay dataset
3. **Offline RL**: PPO fine-tuning with KL penalty to optimize for winning
4. **Self-Play**: Data augmentation through agent vs agent battles
5. **Fast Inference**: Optimized agent for <10 second move times

### Key Features
- **Partial Observability Handling**: Hidden information tracking and inference
- **Opponent Modeling**: Learning from revealed moves, items, and abilities
- **Strategic Planning**: Long-horizon reasoning through attention mechanisms
- **Meta Adaptation**: Robust to evolving strategies and rule changes

## 📁 Project Structure

```
pokeagent-2000/
├── configs/                    # Training configurations
│   └── training_config.py     # Complete hyperparameter settings
├── data/                      # Dataset storage
├── models/                    # Model architectures
│   └── metamon_transformer.py # 200M parameter Transformer
├── scripts/                   # Training and data processing
│   ├── download_dataset.py    # Download 3.5M replay dataset
│   ├── reconstruct_replays.py # Convert spectator → first-person view
│   ├── train_bc.py           # Behavior cloning training
│   ├── train_rl.py           # Offline RL fine-tuning
│   ├── self_play.py          # Self-play data generation
│   └── compete.py            # Competition deployment
├── inference/                 # Optimized inference
│   └── fast_agent.py         # <10s inference agent
├── teams/                     # Pokemon teams
├── checkpoints/              # Model checkpoints
└── logs/                     # Training and battle logs
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# On cluster computer
git clone <your-repo-url>
cd pokeagent-2000

# Setup environment
chmod +x setup_environment.sh
./setup_environment.sh

# Activate environment
conda activate pokeagent-2000
```

### 2. Dataset Download
```bash
# Download 3.5M battle replay dataset
python scripts/download_dataset.py \
    --output-dir data \
    --filter-gen9ou \
    --validate

# Reconstruct first-person view (CRITICAL STEP)
python scripts/reconstruct_replays.py \
    --input-dir data/pokechamp_raw \
    --output-dir data/processed_replays \
    --format-filter gen9ou
```

### 3. Training Pipeline

#### Stage 1: Behavior Cloning (Days 1-5)
```bash
python scripts/train_bc.py \
    --data-dir data/processed_replays \
    --output-dir checkpoints \
    --config-path configs/training_config.py

# Expected: ~60% move prediction accuracy
```

#### Stage 2: Offline RL (Days 6-15)
```bash
python scripts/train_rl.py \
    --data-dir data/processed_replays \
    --bc-checkpoint checkpoints/best_bc_model.pt \
    --output-dir checkpoints

# Expected: 1600-1800 ELO performance
```

#### Stage 3: Self-Play (Days 16-18)
```bash
python scripts/self_play.py \
    --model-path checkpoints/best_rl_model.pt \
    --output-dir data/self_play \
    --num-games 100000

# Generates additional training data
```

### 4. Competition Deployment
```bash
python scripts/compete.py \
    --checkpoint checkpoints/final_model.pt \
    --team teams/competitive_team.txt \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD \
    --num-battles 50 \
    --format gen9ou
```

## ⚙️ Configuration

### Model Configuration
- **Parameters**: 200M (24 layers, 1024 hidden, 16 heads)
- **Sequence Length**: 2048 tokens
- **Vocabulary**: 50K (species, moves, items, abilities)
- **Training Steps**: 1M total (200K BC + 800K RL)

### Hardware Requirements
- **GPU Memory**: 40GB+ (A100 recommended)
- **Training Time**: ~2-3 weeks on 8x A100
- **Inference**: <10 seconds per move (optimized)

### Critical Parameters
```python
# From configs/training_config.py
batch_size = 256
learning_rate = 3e-4
kl_weight = 0.1  # Prevents forgetting BC policy
temperature = 0.8  # Sampling diversity
```

## 📊 Expected Performance Progression

| Stage | ELO | Win Rate | Key Milestone |
|-------|-----|----------|---------------|
| Random | ~1000 | 10% | Baseline |
| BC Only | ~1400 | 40% | Human imitation |
| RL Tuned | ~1700 | 65% | Win optimization |
| Self-Play | ~1850 | 75% | Strategy discovery |
| Final | **2000+** | **80%+** | Competition ready |

## 🔧 Troubleshooting

### Common Issues

#### 1. Dataset Reconstruction Fails
```bash
# Check spectator log format
python -c "
import pickle
with open('data/processed_replays/sample.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[0].keys())
"
```

#### 2. Training Memory Issues
```bash
# Reduce batch size in configs/training_config.py
batch_size = 128  # Instead of 256
gradient_accumulation_steps = 8  # Instead of 4
```

#### 3. Inference Too Slow
```bash
# Enable optimizations in inference/fast_agent.py
use_half_precision = True
use_torchscript = True
cache_size = 10000
```

### Performance Debugging
```bash
# Test inference speed
python inference/fast_agent.py \
    --checkpoint checkpoints/final_model.pt \
    --num-tests 100

# Expected: <1.0s average inference time
```

## 📈 Monitoring

### Training Metrics
- **Behavior Cloning**: Move prediction accuracy (target: 60%+)
- **Offline RL**: Episode returns, policy/value loss, KL divergence
- **Self-Play**: Win rate diversity, strategy coverage

### Competition Metrics
- **ELO Rating**: Primary performance measure
- **Win Rate**: Secondary measure
- **Move Time**: Must stay <10 seconds
- **Cache Hit Rate**: Inference optimization

### Logging
```bash
# View training progress
wandb login
# Logs automatically sync to Weights & Biases

# Monitor competition
tail -f logs/competition_summary_*.json
```

## 🏆 Competition Strategy

### Team Selection
- Use proven high-ELO teams from Smogon
- Focus on bulky offense + hazard control
- Counter common PokeAgent meta strategies

### Opponent Adaptation
- Track revealed moves/items/abilities
- Build probabilistic models of hidden info
- Adapt strategy based on opponent patterns

### Time Management
- Average <5 seconds per move for safety margin
- Cache common positions
- Use confidence thresholds for quick decisions

## 🔬 Research Components

This implementation replicates findings from:
- **Metamon**: "Human-Level Competitive Pokémon via Scalable Offline RL"
- **PokéChamp**: "Expert-level Minimax Language Agent"
- **PokéLLMon**: "Human-Parity Agent with Large Language Models"

### Key Innovations
1. **Spectator → First-Person Reconstruction**: Critical for partial observability
2. **Massive Offline Dataset**: 3.5M human battles
3. **KL-Regularized RL**: Prevents catastrophic forgetting
4. **Temperature Diversity**: Multiple agents for robust self-play

## 📝 Citation

If you use this code for research, please cite:
```bibtex
@misc{pokeagent2000,
  title={PokeAgent-2000: Metamon Architecture Replication for Competitive Pokemon},
  author={[Your Name]},
  year={2025},
  howpublished={\\url{https://github.com/[your-username]/pokeagent-2000}}
}
```

## 🤝 Contributing

This project implements the complete Metamon architecture as described in the research literature. Key areas for improvement:

1. **Enhanced State Encoding**: More sophisticated battle state representation
2. **Multi-Format Support**: Beyond Gen9OU
3. **Real-Time Adaptation**: Online learning during battles
4. **Ensemble Methods**: Multiple model voting

## 📜 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **UT Austin RPL Lab**: Original Metamon research
- **NeurIPS 2025**: PokeAgent Challenge organizers
- **Pokémon Showdown**: Battle simulation platform
- **poke-env**: Python interface library

---

**Target Achievement**: 2000+ ELO on PokeAgent Competition 🚀
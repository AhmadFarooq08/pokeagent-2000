# Pokemon Agent 2000 - Real Data Training Pipeline

## 🚀 Complete Multi-Node Training Implementation

This is the **production-ready** implementation for training a 200M+ parameter Pokemon agent using real battle data and distributed training across multiple nodes/GPUs.

### ⚠️ **IMPORTANT: This replaces the dummy data approach**

The previous training used random dummy data and achieved only ~1020 ELO. This implementation uses:
- **Real Pokemon battle data** from 3.5M+ games
- **Metamon architecture replication** (200M parameters)
- **Multi-node distributed training** (scales to 32+ GPUs)
- **Robust checkpoint/resume** for long training runs
- **Comprehensive testing and validation**

## 📊 **Performance Targets**

| Configuration | GPUs | Batch Size | Steps/6hr | Expected ELO |
|---------------|------|------------|-----------|--------------|
| Single Node  | 4    | 1024       | 540K      | 1400-1500    |
| Dual Node    | 8    | 2048       | 1.08M     | 1500-1600    |
| Quad Node    | 16   | 4096       | 2.16M     | 1600-1700    |
| Large Scale  | 32   | 8192       | 4.32M     | 1700-1800    |

## 🎯 **Quick Start (Cluster)**

### 1. **Download Real Data** (First Time Only)
```bash
# Download and process 3.5M Pokemon battles
python scripts/download_pokechamp.py \
    --output-dir /scratch/pokechamp \
    --max-games 500000 \
    --min-elo 1200 \
    --num-workers 32

# Parse replays to training data
python data/replay_parser.py \
    --input-dir /scratch/pokechamp/raw \
    --output-dir /scratch/pokechamp/processed \
    --num-workers 64 \
    --batch-size 5000
```

### 2. **Launch Training**
```bash
# Single node (4 GPUs, 6 hours)
./scripts/launch_training.sh 1 4 6

# Dual node (8 GPUs, 6 hours)  
./scripts/launch_training.sh 2 4 6

# Large scale (16 GPUs, 8 hours)
./scripts/launch_training.sh 4 4 8 --partition gpu-large

# Resume from checkpoint
./scripts/launch_training.sh 2 4 6 --resume

# Test run with limited data
./scripts/launch_training.sh 1 1 1 --test-mode --mock-data
```

### 3. **Monitor Training**
```bash
# Check job status
squeue -u $USER

# Monitor logs
tail -f slurm-JOBID.out

# Check checkpoints
ls -la checkpoints_*gpu/
```

## 📁 **File Structure**

```
pokeagent-2000/
├── scripts/
│   ├── download_pokechamp.py      # Download real Pokemon data
│   ├── train_multinode.py         # Multi-node distributed training
│   └── launch_training.sh         # Universal SLURM launcher
├── data/
│   ├── replay_parser.py           # Parse Pokemon Showdown logs
│   └── real_pokemon_dataset.py    # Efficient dataset loader
├── models/
│   └── metamon_transformer.py     # 200M parameter model
├── tests/
│   ├── mock_data.py              # Mock data for testing
│   ├── test_components.py        # Comprehensive test suite
│   └── benchmark_data_pipeline.py # Performance benchmarks
├── scripts/
│   └── validate_setup.py         # Pre-flight validation
└── configs/
    └── training_config.py        # Training configurations
```

## 🔧 **Component Details**

### **Data Pipeline**
- **Download**: Retrieves 3.5M+ real Pokemon battles from Hugging Face
- **Parser**: Converts Pokemon Showdown logs to state-action pairs
- **Dataset**: Efficient loading with caching and multi-processing
- **Filtering**: ELO-based filtering for high-quality games

### **Model Architecture** 
- **Base**: 200M parameter Transformer (matches Metamon)
- **Layers**: 24 transformer layers, 1024 hidden, 16 attention heads
- **Inputs**: Pokemon species, moves, items, abilities, HP, stats, field conditions
- **Outputs**: Policy (action probabilities) + Value (win probability)

### **Training System**
- **Distributed**: PyTorch DDP across arbitrary nodes/GPUs
- **Scaling**: Automatic batch size and learning rate scaling
- **Checkpoints**: Saves every 15 minutes with full resume capability
- **Time Management**: Respects SLURM time limits with graceful shutdown
- **Mixed Precision**: FP16 training for memory efficiency

### **SLURM Integration**
- **Flexible**: Supports 1-32+ GPUs across multiple nodes
- **Auto-configuration**: Detects account, partition, QOS automatically
- **Resource Management**: Auto-calculates memory and CPU requirements
- **Robust**: Handles node failures and job interruptions

## 🧪 **Testing & Validation**

### **Pre-flight Validation**
```bash
# Comprehensive system check
python scripts/validate_setup.py --comprehensive

# Quick validation
python scripts/validate_setup.py

# Performance benchmarks
python tests/benchmark_data_pipeline.py --comprehensive
```

### **Component Tests**
```bash
# Test all components
python tests/test_components.py

# Test imports and basic functionality
python test_imports.py

# Test individual components
python tests/mock_data.py
python data/replay_parser.py --test-mode
python data/real_pokemon_dataset.py
```

### **Launch Script Tests**
```bash
# Test help and configuration
./scripts/launch_training.sh --help

# Dry run test
./scripts/launch_training.sh 1 1 1 --dry-run --test-mode

# Mock data test
./scripts/launch_training.sh 1 1 0.5 --mock-data --test-mode
```

## ⚙️ **Configuration Options**

### **Training Parameters**
```bash
--max-steps 200000              # Training steps (default: 200K)
--learning-rate 3e-4            # Base learning rate (auto-scaled)
--batch-size-per-gpu 256        # Per-GPU batch size
--checkpoint-minutes 15         # Checkpoint frequency
--time-limit-hours 6            # SLURM time limit
```

### **Data Parameters**
```bash
--data-dir /path/to/data         # Processed data directory
--max-samples 1000000           # Limit training samples
--min-elo 1200                  # Filter by minimum ELO
```

### **Resource Parameters**
```bash
--nodes 2                       # Number of nodes
--gpus-per-node 4               # GPUs per node
--memory 128                    # GB RAM per node
--partition gpu-large           # SLURM partition
--account myaccount             # SLURM account
```

### **Debugging Parameters**
```bash
--dry-run                       # Simulate without training
--mock-data                     # Use mock data for testing
--test-mode                     # Quick test (1000 steps)
--debug                         # Enable debug logging
```

## 📈 **Performance Optimization**

### **Recommended Configurations**

**Single Node (4 GPUs):**
```bash
./scripts/launch_training.sh 1 4 6 512 \
    --checkpoint-minutes 20 \
    --eval-every 10000
```

**Dual Node (8 GPUs):**
```bash
./scripts/launch_training.sh 2 4 6 256 \
    --checkpoint-minutes 15 \
    --eval-every 5000
```

**Large Scale (16+ GPUs):**
```bash
./scripts/launch_training.sh 4 4 8 256 \
    --checkpoint-minutes 10 \
    --eval-every 2500 \
    --partition gpu-large \
    --exclusive
```

### **Performance Tuning**
- **Batch Size**: Larger batches (512+) for better GPU utilization
- **Workers**: 2-4 data loading workers per GPU
- **Checkpoints**: Less frequent checkpoints for large scale training
- **Memory**: Reserve 32GB+ per GPU for model + data

## 🔧 **Troubleshooting**

### **Common Issues**

**"CUDA out of memory"**
```bash
# Reduce batch size
./scripts/launch_training.sh 2 4 6 128  # Instead of 256
```

**"Data loading slow"**
```bash
# Check data directory exists
ls -la /path/to/data/pokechamp/processed/

# Test data pipeline
python tests/benchmark_data_pipeline.py
```

**"Training not starting"**
```bash
# Check SLURM configuration
sinfo
sacctmgr show user $USER

# Validate system
python scripts/validate_setup.py
```

**"Job killed due to time limit"**
```bash
# Resume from checkpoint
./scripts/launch_training.sh 2 4 6 --resume

# Check checkpoint directory
ls -la checkpoints_8gpu/
```

### **Debugging Commands**
```bash
# Check job status
squeue -j JOBID

# Check job details
scontrol show job JOBID

# Check node resources
sinfo -N -l

# Monitor GPU usage
nvidia-smi

# Check logs
tail -f slurm-JOBID.out
```

## 📚 **Implementation Notes**

### **Key Improvements Over Dummy Data**
1. **Real Data**: 3.5M actual Pokemon battles vs random noise
2. **Proper State Representation**: Actual game states vs dummy tensors
3. **Action Mapping**: Real moves/switches vs random actions
4. **Win Conditions**: Actual game outcomes vs random labels
5. **Strategic Learning**: Learns actual Pokemon strategy

### **Architecture Decisions**
- **Metamon Replication**: Follows proven 200M parameter architecture
- **Distributed Training**: Scales to meet 2000+ ELO target
- **Robust Infrastructure**: Handles long training runs on HPC clusters
- **Comprehensive Testing**: Validates all components before deployment

### **Expected Training Time**
- **Data Download**: 2-4 hours (one-time setup)
- **Data Processing**: 1-2 hours (one-time setup)
- **Training to 1600 ELO**: 20-40 hours on 8+ GPUs
- **Training to 1800 ELO**: 60-100 hours on 16+ GPUs

### **Resource Requirements**
- **Minimum**: 1 GPU, 32GB RAM, 100GB storage
- **Recommended**: 8 GPUs, 256GB RAM, 500GB storage
- **Large Scale**: 16+ GPUs, 512GB+ RAM, 1TB+ storage

---

## 🎯 **Next Steps for Cluster Deployment**

1. **Pull latest changes**: `git pull origin main`
2. **Run validation**: `python scripts/validate_setup.py --comprehensive`
3. **Download data**: `python scripts/download_pokechamp.py --max-games 100000`
4. **Launch training**: `./scripts/launch_training.sh 2 4 6`
5. **Monitor progress**: `tail -f slurm-*.out`

**Target**: Achieve 1600+ ELO within 24-48 hours of training on 8+ GPUs with real data.

---

**🚨 Critical Success Factors:**
- ✅ Use real Pokemon battle data (not dummy data)
- ✅ Scale to 8+ GPUs for competitive training speed
- ✅ Run comprehensive validation before cluster deployment
- ✅ Monitor training progress and adjust configuration as needed
- ✅ Aim for 1600+ ELO as realistic initial target (2000+ requires advanced methods)
# CLAUDE.md - Instructions for Claude Code

## General Rules
- Always provide cluster commands in a single line, never multi-line format
- User is working on cluster node, not local machine - don't try to run commands for them

## Dataset Issues Fixed
- RealPokemonDataset NO LONGER uses mock data fallback
- Separate MockPokemonDataset class for testing only
- Production code will FAIL FAST if no real data found
- Supports both JSON and PKL file formats

## Cluster Instructions for User

### 1. Kill Running Processes and Clean Up
```bash
cd ~/pokeagent-2000 && pkill -f train_multinode && sleep 2 && rm -rf checkpoints_4gpu/ wandb/ logs/ runs/ && find . -name "*.log" -delete && find . -name "slurm-*.out" -delete
```

### 2. Pull Latest Changes
```bash
git pull origin main
```

### 3. Check Available Data Directories
```bash
ls -la data/pokechamp/ && ls -la data/processed_replays/
```

### 4. Training Commands (Try These Paths)

**Option 1: Use pokechamp data**
```bash
torchrun --nproc_per_node=4 scripts/train_multinode.py --nodes 1 --gpus-per-node 4 --batch-size-per-gpu 256 --max-steps 200000 --time-limit-hours 6 --checkpoint-minutes 15 --learning-rate 3e-4 --data-dir data/pokechamp --output-dir checkpoints_4gpu --log-every 100 --eval-every 5000
```

**Option 2: Use processed_replays data**
```bash
torchrun --nproc_per_node=4 scripts/train_multinode.py --nodes 1 --gpus-per-node 4 --batch-size-per-gpu 256 --max-steps 200000 --time-limit-hours 6 --checkpoint-minutes 15 --learning-rate 3e-4 --data-dir data/processed_replays --output-dir checkpoints_4gpu --log-every 100 --eval-every 5000
```

### 5. If Data Not Found Error
Check error message for exact path being searched and update --data-dir accordingly
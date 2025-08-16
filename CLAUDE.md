# CLAUDE.md - Instructions for Claude Code

## General Rules
- Always provide cluster commands in a single line, never multi-line format
- User is working on cluster node, not local machine - don't try to run commands for them

## Training Command (Single Line)
```bash
torchrun --nproc_per_node=4 scripts/train_multinode.py --nodes 1 --gpus-per-node 4 --batch-size-per-gpu 256 --max-steps 200000 --time-limit-hours 6 --checkpoint-minutes 15 --learning-rate 3e-4 --data-dir data/pokechamp/processed --output-dir checkpoints_4gpu --log-every 100 --eval-every 5000
```
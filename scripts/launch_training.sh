#!/bin/bash
# Universal SLURM Launcher for Multi-Node Pokemon Training
# Supports arbitrary node/GPU configurations with automatic resource optimization

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-0}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Universal SLURM Launcher for Pokemon Agent Training

Usage: $0 [OPTIONS] <nodes> <gpus_per_node> <time_hours> [batch_size_per_gpu]

Arguments:
  nodes              Number of nodes to use (default: 1)
  gpus_per_node      GPUs per node (default: 4)  
  time_hours         Time limit in hours (default: 6)
  batch_size_per_gpu Batch size per GPU (default: 256)

Options:
  --data-dir DIR          Data directory (default: data/pokechamp/processed)
  --output-dir DIR        Output directory (default: checkpoints_Ngpu)
  --max-steps N           Maximum training steps (default: 200000)
  --learning-rate LR      Learning rate (default: 3e-4)
  --checkpoint-minutes N  Checkpoint frequency in minutes (default: 15)
  --resume               Resume from latest checkpoint
  --resume-checkpoint PATH Resume from specific checkpoint
  --dry-run              Dry run mode (don't actually train)
  --mock-data            Use mock data for testing
  --debug                Enable debug mode
  --partition NAME       SLURM partition (default: gpu)
  --qos NAME             SLURM QOS (default: auto-detect)
  --account NAME         SLURM account (default: auto-detect)
  --email EMAIL          Email for notifications
  --job-name NAME        Job name (default: pokeagent_Ngpu)
  --memory GB            Memory per node in GB (default: auto-calculate)
  --exclusive            Request exclusive node access
  --no-wandb             Disable wandb logging
  --test-mode            Quick test with limited steps
  --help, -h             Show this help message

Examples:
  # Single node, 4 GPUs, 6 hours
  $0 1 4 6

  # 2 nodes, 8 total GPUs, 6 hours, larger batches
  $0 2 4 6 512

  # Large scale: 4 nodes, 16 GPUs, 8 hours
  $0 4 4 8 256 --partition gpu-large

  # Resume training from checkpoint
  $0 2 4 6 --resume

  # Test run with mock data
  $0 1 1 1 --test-mode --mock-data

Environment Variables:
  DEBUG=1               Enable debug output
  SLURM_ACCOUNT         Default SLURM account
  SLURM_PARTITION       Default SLURM partition
  SLURM_QOS             Default SLURM QOS
  WANDB_API_KEY         Weights & Biases API key

EOF
}

# Default values
NODES=1
GPUS_PER_NODE=4
TIME_HOURS=6
BATCH_SIZE_PER_GPU=256

# Configuration options
DATA_DIR="data/pokechamp/processed"
OUTPUT_DIR=""
MAX_STEPS=200000
LEARNING_RATE="3e-4"
CHECKPOINT_MINUTES=15
RESUME=false
RESUME_CHECKPOINT=""
DRY_RUN=false
MOCK_DATA=false
DEBUG=false
PARTITION=${SLURM_PARTITION:-"gpu"}
QOS=${SLURM_QOS:-""}
ACCOUNT=${SLURM_ACCOUNT:-""}
EMAIL=""
JOB_NAME=""
MEMORY=""
EXCLUSIVE=false
NO_WANDB=false
TEST_MODE=false

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --checkpoint-minutes)
            CHECKPOINT_MINUTES="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --resume-checkpoint)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --mock-data)
            MOCK_DATA=true
            shift
            ;;
        --debug)
            DEBUG=true
            export DEBUG=1
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --qos)
            QOS="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --exclusive)
            EXCLUSIVE=true
            shift
            ;;
        --no-wandb)
            NO_WANDB=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            # Parse positional arguments
            if [[ -z "${NODES_SET:-}" ]]; then
                NODES="$1"
                NODES_SET=1
            elif [[ -z "${GPUS_SET:-}" ]]; then
                GPUS_PER_NODE="$1"
                GPUS_SET=1
            elif [[ -z "${TIME_SET:-}" ]]; then
                TIME_HOURS="$1"
                TIME_SET=1
            elif [[ -z "${BATCH_SET:-}" ]]; then
                BATCH_SIZE_PER_GPU="$1"
                BATCH_SET=1
            else
                log_error "Too many positional arguments: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate inputs
if ! [[ "$NODES" =~ ^[1-9][0-9]*$ ]]; then
    log_error "Invalid number of nodes: $NODES"
    exit 1
fi

if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    log_error "Invalid GPUs per node: $GPUS_PER_NODE"
    exit 1
fi

if ! [[ "$TIME_HOURS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    log_error "Invalid time hours: $TIME_HOURS"
    exit 1
fi

# Calculate derived values
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS))

# Set defaults based on configuration
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="checkpoints_${TOTAL_GPUS}gpu"
fi

if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="pokeagent_${TOTAL_GPUS}gpu"
fi

if [[ -z "$MEMORY" ]]; then
    # Auto-calculate memory: ~32GB per GPU + overhead
    MEMORY_PER_GPU=32
    MEMORY=$((MEMORY_PER_GPU * GPUS_PER_NODE + 16))
fi

# Test mode adjustments
if [[ "$TEST_MODE" == "true" ]]; then
    MAX_STEPS=1000
    CHECKPOINT_MINUTES=2
    TIME_HOURS=0.5
    MOCK_DATA=true
    JOB_NAME="${JOB_NAME}_test"
fi

# Convert time to SLURM format
TIME_SLURM=$(printf "%02d:%02d:00" $((${TIME_HOURS%.*})) $(( (${TIME_HOURS#*.} * 60 / 100) )))

# Display configuration
log_info "=== Training Configuration ==="
log_info "Nodes: $NODES"
log_info "GPUs per node: $GPUS_PER_NODE"
log_info "Total GPUs: $TOTAL_GPUS"
log_info "Time limit: ${TIME_HOURS}h ($TIME_SLURM)"
log_info "Batch size per GPU: $BATCH_SIZE_PER_GPU"
log_info "Total batch size: $TOTAL_BATCH_SIZE"
log_info "Max steps: $MAX_STEPS"
log_info "Output directory: $OUTPUT_DIR"
log_info "Job name: $JOB_NAME"
log_info "Memory per node: ${MEMORY}GB"

if [[ "$TEST_MODE" == "true" ]]; then
    log_warn "TEST MODE: Limited steps and mock data"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN: No actual training will occur"
fi

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    log_error "sbatch not found. Are you on a SLURM cluster?"
    exit 1
fi

# Auto-detect account and QOS if not specified
if [[ -z "$ACCOUNT" ]]; then
    ACCOUNT=$(sacctmgr show user $USER format=account%50 -n -P | head -1 | cut -d'|' -f1)
    if [[ -n "$ACCOUNT" ]]; then
        log_info "Auto-detected account: $ACCOUNT"
    fi
fi

if [[ -z "$QOS" ]]; then
    QOS=$(sacctmgr show user $USER format=qos%50 -n -P | head -1 | cut -d'|' -f1)
    if [[ -n "$QOS" ]]; then
        log_info "Auto-detected QOS: $QOS"
    fi
fi

# Validate data directory exists (for real data)
if [[ "$MOCK_DATA" == "false" && ! -d "$DATA_DIR" ]]; then
    log_warn "Data directory not found: $DATA_DIR"
    log_warn "Make sure to download data first with: python scripts/download_pokechamp.py"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate SLURM script
SCRIPT_PATH="${OUTPUT_DIR}/launch_${JOB_NAME}.sh"

log_info "Generating SLURM script: $SCRIPT_PATH"

cat > "$SCRIPT_PATH" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=$(($(nproc) / GPUS_PER_NODE))
#SBATCH --time=$TIME_SLURM
#SBATCH --mem=${MEMORY}GB
#SBATCH --partition=$PARTITION
EOF

# Add optional SLURM parameters
if [[ -n "$QOS" ]]; then
    echo "#SBATCH --qos=$QOS" >> "$SCRIPT_PATH"
fi

if [[ -n "$ACCOUNT" ]]; then
    echo "#SBATCH --account=$ACCOUNT" >> "$SCRIPT_PATH"
fi

if [[ -n "$EMAIL" ]]; then
    echo "#SBATCH --mail-user=$EMAIL" >> "$SCRIPT_PATH"
    echo "#SBATCH --mail-type=BEGIN,END,FAIL" >> "$SCRIPT_PATH"
fi

if [[ "$EXCLUSIVE" == "true" ]]; then
    echo "#SBATCH --exclusive" >> "$SCRIPT_PATH"
fi

# Add script body
cat >> "$SCRIPT_PATH" << EOF

# Job information
echo "=== Job Information ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job name: \$SLURM_JOB_NAME"
echo "Nodes: \$SLURM_JOB_NUM_NODES"
echo "Tasks per node: \$SLURM_NTASKS_PER_NODE"
echo "CPUs per task: \$SLURM_CPUS_PER_TASK"
echo "GPUs per node: \$SLURM_GPUS_PER_NODE"
echo "Partition: \$SLURM_JOB_PARTITION"
echo "Start time: \$(date)"
echo "Working directory: \$(pwd)"
echo ""

# Load modules (adjust for your cluster)
module purge
module load python/3.10-anaconda || module load python/3.10 || echo "Warning: Could not load Python module"
module load cuda/12.1 || module load cuda || echo "Warning: Could not load CUDA module"

# Activate conda environment
source activate pokeagent-2000 || echo "Warning: Could not activate conda environment"

# Environment setup
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set master node
export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Distributed training environment
export WORLD_SIZE=\$SLURM_NTASKS
export NNODES=\$SLURM_JOB_NUM_NODES
export NPROC_PER_NODE=\$SLURM_GPUS_PER_NODE

echo "=== Distributed Setup ==="
echo "Master address: \$MASTER_ADDR"
echo "Master port: \$MASTER_PORT"
echo "World size: \$WORLD_SIZE"
echo "Nodes: \$NNODES"
echo "Processes per node: \$NPROC_PER_NODE"
echo ""

# Build training command
TRAIN_CMD="torchrun \\
    --nnodes=\$NNODES \\
    --nproc_per_node=\$NPROC_PER_NODE \\
    --master_addr=\$MASTER_ADDR \\
    --master_port=\$MASTER_PORT \\
    scripts/train_multinode.py \\
    --nodes $NODES \\
    --gpus-per-node $GPUS_PER_NODE \\
    --batch-size-per-gpu $BATCH_SIZE_PER_GPU \\
    --max-steps $MAX_STEPS \\
    --time-limit-hours $TIME_HOURS \\
    --checkpoint-minutes $CHECKPOINT_MINUTES \\
    --learning-rate $LEARNING_RATE \\
    --data-dir $DATA_DIR \\
    --output-dir $OUTPUT_DIR \\
    --log-every 100 \\
    --eval-every 5000"

# Add optional flags
EOF

# Add conditional flags
if [[ "$RESUME" == "true" ]]; then
    echo "TRAIN_CMD=\"\$TRAIN_CMD --resume\"" >> "$SCRIPT_PATH"
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "TRAIN_CMD=\"\$TRAIN_CMD --resume-checkpoint $RESUME_CHECKPOINT\"" >> "$SCRIPT_PATH"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "TRAIN_CMD=\"\$TRAIN_CMD --dry-run\"" >> "$SCRIPT_PATH"
fi

if [[ "$MOCK_DATA" == "true" ]]; then
    echo "TRAIN_CMD=\"\$TRAIN_CMD --mock-data\"" >> "$SCRIPT_PATH"
fi

if [[ "$DEBUG" == "true" ]]; then
    echo "TRAIN_CMD=\"\$TRAIN_CMD --debug-mode\"" >> "$SCRIPT_PATH"
fi

# Add execution and cleanup
cat >> "$SCRIPT_PATH" << EOF

echo "=== Training Command ==="
echo "\$TRAIN_CMD"
echo ""

# Run training
echo "Starting training at \$(date)"
srun \$TRAIN_CMD

EXIT_CODE=\$?

echo ""
echo "=== Job Completed ==="
echo "Exit code: \$EXIT_CODE"
echo "End time: \$(date)"

# Job summary
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code \$EXIT_CODE"
fi

# Save job info
cat > "${OUTPUT_DIR}/job_info_\${SLURM_JOB_ID}.json" << EOJSON
{
    "job_id": "\$SLURM_JOB_ID",
    "job_name": "\$SLURM_JOB_NAME",
    "nodes": \$SLURM_JOB_NUM_NODES,
    "gpus_per_node": \$SLURM_GPUS_PER_NODE,
    "total_gpus": $TOTAL_GPUS,
    "batch_size_per_gpu": $BATCH_SIZE_PER_GPU,
    "total_batch_size": $TOTAL_BATCH_SIZE,
    "max_steps": $MAX_STEPS,
    "time_limit_hours": $TIME_HOURS,
    "start_time": "\$(date -Iseconds)",
    "exit_code": \$EXIT_CODE,
    "partition": "\$SLURM_JOB_PARTITION",
    "account": "\$SLURM_JOB_ACCOUNT"
}
EOJSON

exit \$EXIT_CODE
EOF

# Make script executable
chmod +x "$SCRIPT_PATH"

# Submit job
log_info "Submitting job to SLURM..."

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "DRY RUN: Would execute: sbatch $SCRIPT_PATH"
    log_info "Generated script saved at: $SCRIPT_PATH"
    exit 0
fi

# Submit the job
JOB_OUTPUT=$(sbatch "$SCRIPT_PATH" 2>&1)
if [[ $? -eq 0 ]]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+')
    log_info "Job submitted successfully!"
    log_info "Job ID: $JOB_ID"
    log_info "Job name: $JOB_NAME"
    log_info "Script: $SCRIPT_PATH"
    
    # Show job status
    log_info ""
    log_info "=== Job Status ==="
    squeue -j "$JOB_ID" --format="%.18i %.9P %.20j %.8u %.8T %.10M %.6D %R" || true
    
    log_info ""
    log_info "Monitor with: squeue -u $USER"
    log_info "Cancel with: scancel $JOB_ID"
    log_info "Logs will be in: slurm-${JOB_ID}.out"
    
    if [[ -n "$EMAIL" ]]; then
        log_info "Email notifications will be sent to: $EMAIL"
    fi
    
else
    log_error "Failed to submit job:"
    echo "$JOB_OUTPUT"
    exit 1
fi
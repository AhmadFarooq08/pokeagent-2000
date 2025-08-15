#!/bin/bash
# Environment setup script for Metamon replication
# Run this on the cluster computer

set -e  # Exit on any error

echo "=== Setting up PokeAgent-2000 Environment ==="

# Create conda environment
echo "Creating conda environment..."
conda create -n pokeagent-2000 python=3.10 -y

# Activate environment
echo "Activating environment..."
source activate pokeagent-2000

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install transformers and ML libraries
echo "Installing core ML libraries..."
pip install transformers==4.36.0
pip install datasets==2.15.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.0

# Install RL libraries
echo "Installing RL libraries..."
pip install stable-baselines3==2.2.1
pip install gymnasium==0.29.1
pip install torch-ac==1.4.0

# Install Pokemon libraries
echo "Installing Pokemon-specific libraries..."
pip install poke-env==0.6.0

# Install utility libraries
echo "Installing utility libraries..."
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install pandas==2.1.4
pip install tqdm==4.66.1
pip install rich==13.7.0
pip install wandb==0.16.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Install development tools
echo "Installing development tools..."
pip install black==23.12.0
pip install isort==5.13.2
pip install flake8==6.1.0
pip install pytest==7.4.3

# Clone and install amago framework (critical for Metamon replication)
echo "Installing amago framework..."
if [ ! -d "amago" ]; then
    git clone https://github.com/UT-Austin-RPL/amago.git
fi
cd amago && pip install -e . && cd ..

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import poke_env; print(f'Poke-env version: {poke_env.__version__}')"

echo "=== Environment setup complete! ==="
echo "To activate: conda activate pokeagent-2000"
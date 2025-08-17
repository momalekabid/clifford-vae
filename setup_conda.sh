#!/bin/bash
set -e

ENV_NAME="hvae"
PYTHON_VERSION="3.11"

echo "Setting up environment: $ENV_NAME"

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing core packages..."
conda install -y \
    numpy>=1.24 \
    scipy>=1.10 \
    matplotlib>=3.7 \
    scikit-learn>=1.3 \
    pandas>=1.5

# Install PyTorch 
echo "Installing PyTorch..."
# GPU 
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# CPU
conda install pytorch torchvision torchaudio -c pytorch -y

# Install remaining via pip
echo "Installing additional packages..."
pip install wandb lpips

echo "Setup complete! Activate with: conda activate $ENV_NAME"
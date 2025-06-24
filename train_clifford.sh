#!/bin/bash

#SBATCH --account=???
#SBATCH --job-name=clifford-vae
#SBATCH --time=24:00:00
#SBATCH --output=clifford_%j.out
#SBATCH --error=clifford_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=m8abid@uwaterloo.ca
#SBATCH --mail-type=ALL

. /etc/profile.d/modules.sh
module add cuda/12.1

echo "Activating virtual environment..."
source ~/clifford-vae/.venv/bin/activate
if [ $? -ne 0 ]; then
  echo "Failed to activate virtual environment. Exiting."
  exit 1
fi
echo "Virtual environment activated."

echo "Running Clifford-VAE experiment..."
cd ~/clifford-vae/src/clifford
python3 mnist_clifford.py

deactivate

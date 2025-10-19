# Clifford and Hyperspherical VAEs

PyTorch implementation of VAEs with different latent priors: Gaussian, PowerSpherical, and Clifford-torus. 

## Setup

```bash
pip install -r requirements.txt
```
## conda (setup for SLURM)
```bash
bash setup_conda.sh
conda activate hvae
cd vmf && pip3 install .
mkdir -p logs  # for slurm job outputs
```

## defaults

- **mnist_clifpws.py**: d=[2,5,10,20,40, 80], epochs=500, n_runs=20
- **mnist_vmf.py**: d=[2,5,10,20,40], epochs=500, n_runs=20
- **fashion_train.py**: datasets=[fashionmnist, cifar10], latent_dims=[2,4,128,512,1024,2048,4096], n_trials=5, epochs=500
## Usage

**CNN VAE** (FashionMNIST, CIFAR10, higher dimensions: clifford, powerspherical, gaussian):
```bash
python3 cnn/fashion_train.py
```

**MLP VAE** (MNIST, replicating base experiments from the original paper):
```bash
python3 mnist/mnist_clifpws.py
python3 mnist/mnist_vmf.py
```

**note**: gaussian VAE has l2 normalization enabled by default. use `--no-l2_norm` to disable.

run `python3 cnn/fashion_train.py --help` or `python3 mnist/mnist_clifpws.py --help` for all available options.

## SLURM submission

create slurm job scripts (e.g., `slurm_mnist_clifpws.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=mnist-clifpws
#SBATCH --output=logs/mnist_clifpws_%j.out
#SBATCH --error=logs/mnist_clifpws_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/.bashrc
conda activate clifford-vae
cd /path/to/clifford-vae

python3 mnist/mnist_clifpws.py
```

submit jobs:
```bash
sbatch slurm_mnist_clifpws.sh
sbatch slurm_mnist_vmf.sh
sbatch slurm_fashion.sh
```

## Folder structure 
- `cnn/`: with deeper arch  
- `mnist/`: mlp experiments for MNIST  
- `dists/clifford.py`: custom distributions
- `utils` : vsa and wandb plotting utils
- 'plot_results.py' run this to generate final plots of latent dimension on x, accuracy/F1 on Y-axis

## class labels
### fashion-mnist classes

- 0: t-shirt/top
- 1: trouser
- 2: pullover
- 3: dress
- 4: coat
- 5: sandal
- 6: shirt
- 7: sneaker
- 8: bag
- 9: ankle boot

### cifar-10 classes

- 0: airplane
- 1: automobile
- 2: bird
- 3: cat
- 4: deer
- 5: dog
- 6: frog
- 7: horse
- 8: ship
- 9: truck

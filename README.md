# Clifford and Hyperspherical VAEs

PyTorch implementation of VAEs with different latent priors: Gaussian, PowerSpherical, and Clifford-torus. 

## Setup

```bash
pip install -r requirements.txt
```
## conda (better for SLURM)
conda:
```bash
bash setup_conda.sh
conda activate hvae
cd vmf && pip3 install .
```

## Usage

**CNN VAE** (FashionMNIST, CIFAR10, higher dimensions: clifford, powerspherical, gaussian):
```bash
python cnn/fashion_train.py # also supports cifar10
```

**MLP VAE** (MNIST, replicating base experiments from the original paper):
```bash
python mnist/mnist_clifpws.py --visualize
python mnist/mnist_vmf.py --visualize 

## Folder structure 
- `cnn/`: CNN VAE with advanced reconstruction losses
- `mnist/`: MLP VAE experiments comparing distributions  
- `dists/clifford.py`: custom hyperspherical distributions
- `utils/wandb_utils.py`: Fourier/HRR property testing

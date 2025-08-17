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

**CNN VAE** (FashionMNIST, higher dimensions: clifford, powerspherical, gaussian):
```bash
python cnn/train_vcae.py --epochs 100 --batch_size 128 --l1_freq_weight=0.1 
```

**MLP VAE** (MNIST, replicating base experiments from the original paper):
```bash
python mnist/mnist_most.py --d_dims 2 5 10 20 40 --visualize
python mnist/mnist_vmf.py --d_dims 2 5 10 20 40 --visualize 
```

*CNN*
- `--recon_loss l1_freq`: L1 + frequency domain loss
- `--use_perceptual`: add LPIPS perceptual loss (not tested)
- `--no_wandb`: disable logging
*MLP*
- `--visualize`: generate t-SNE, PCA, reconstructions for MLP MNIST experiments

## Folder structure 
- `cnn/`: CNN VAE with advanced reconstruction losses
- `mnist/`: MLP VAE experiments comparing distributions  
- `dists/clifford.py`: custom hyperspherical distributions
- `utils/wandb_utils.py`: Fourier/HRR property testing

Results saved to `results/` and `visualizations/` with optional w&b logging.

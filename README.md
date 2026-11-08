# Clifford-VAE

VAEs that learn vectors all unitary. This experimental suite also evals Gaussian (N), L2-normalised Gaussian (N_L2), power
spherical (S) and von Mises-Fisher (V, MNIST only).

```bash
# setup for uv/pip/conda 
uv sync
pip install -r requirements.txt
conda env create -f environment.yml

# Reproduce paper in one go: 20 runs per dim, timing, tables and figures (resumable)
python reproduce.py --datasets cifar10,fashionmnist,mnist --workers 8 --gpus 0,1
python reproduce.py --datasets celeba
python reproduce.py --datasets shapes3d,omniglot # run these
python reproduce.py --report_only            # generate only timing, tables, figures from finished runs
```

Control how many runs with these flags for `reproduce.py` and `run_experiments.py`):

```bash
--datasets cifar10,mnist      # which datasets (default: mnist, fashionmnist, cifar10, celeba)
--runs 5                      # runs per setting (default 20; celeba 5, shapes3d 3, omniglot 5)
--runs 6-20                   # only runs 6 to 20, e.g. to split the work over two machines
--dims 128,256                # which latent sizes (default: every size in the paper)
--priors C,N                  # which priors (default: all)
--workers 8 --gpus 0,1        # runs at once, spread over these gpus
```

| `--datasets` | runs with the defaults |
|---|---|
| `mnist` | 600 (6 sizes x 5 priors x 20 runs) |
| `fashionmnist`, `cifar10` | 480 each (6 sizes x 4 priors x 20 runs) |
| `celeba` | 60 (3 sizes x 4 priors x 5 runs) |
| `shapes3d` | 24 (4 priors x beta 1 and 4 x 3 runs) |
| `omniglot` | 30 (3 priors x beta 1 and 4 x 5 runs) |

```bash
# the steps on their own
python run_experiments.py --datasets cifar10 --workers 4    # train + evaluate
python run_experiments.py --list                            # what's done, running, failed
python timing.py --datasets cifar10,fashionmnist            # cost numbers, on an idle gpu
python tables.py                                            # -> runs/_aggregate/tables
python figures.py                                           # -> runs/_aggregate/figures

# Test one run
python train_cifar10.py --prior C --d 256 --seed 1
python train_cifar10.py --prior C --d 256 --seed 1 --kappa_shared   # the submission's one kappa per image

# Re-run eval on finished runs
python evaluate.py runs --only vsa,raw --skip_existing
```

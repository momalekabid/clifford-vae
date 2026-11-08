"""mnist, mlp vae, binarised, bernoulli likelihood (davidson et al. 2018 setup), constant beta.

    python train_mnist.py --prior C --d 10 --seed 1
"""
from clifford_vae.train import main

DIMS = [2, 5, 10, 20, 40, 128]
PRIORS = ["N", "N_L2", "S", "V", "C"]

if __name__ == "__main__":
    main("mnist", DIMS, PRIORS)

"""fashionmnist (32px grey), cnn vae, l1 reconstruction, cyclical beta.

    python train_fashionmnist.py --prior C --d 256 --seed 1
"""
from clifford_vae.train import main

DIMS = [128, 256, 512, 1024, 2048, 4096]
PRIORS = ["N", "N_L2", "S", "C"]

if __name__ == "__main__":
    main("fashionmnist", DIMS, PRIORS)

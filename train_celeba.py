"""celeba (64px rgb, 162k train images), 5-block cnn vae, l1 reconstruction, cyclical beta, the
same 500-epoch schedule as cifar-10 (an epoch costs ~13 cifar-10 epochs). k-NN predicts the 40
attributes; role-filler groups are identities.

    python train_celeba.py --prior C --d 256 --seed 1
"""
from clifford_vae.train import main

DIMS = [128, 256, 512]
PRIORS = ["N", "N_L2", "S", "C"]

if __name__ == "__main__":
    main("celeba", DIMS, PRIORS)

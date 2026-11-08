"""omniglot (lake et al. 2015): train on the background alphabets with 4 of every character's 20
drawers held out. N, N_L2 and C, every code 512 long as in karunaratne et al. 2021 (N, N_L2 at
d=512; C at d=256, code length 2d), the cnn's cyclical
ramp up to beta_max 1 or 4. evaluations (clifford_vae/omniglot.py): few-shot classification on
the evaluation alphabets with the codes as an hd key memory (karunaratne et al. 2021), and hrr vs
vector-arithmetic analogies that compose a held-out character x drawer pair from three seen ones.

    python train_omniglot.py --prior C --d 256 --seed 1 --beta_max 4
"""
from clifford_vae.train import main

DIMS = [256, 512]
PRIORS = ["N", "N_L2", "C"]

if __name__ == "__main__":
    main("omniglot", DIMS, PRIORS)

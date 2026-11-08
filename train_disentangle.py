"""reviewer 1's compositionality test: 3d shapes with 10 of the 40 object-hue x shape pairs held
out of training (every hue misses one shape). N, N_L2, S and C at d=128, beta_max 1 and 4 (beta-vae).
reports reconstruction and factor accuracy on seen vs unseen pairs, a linear probe from codes to
hue / shape (fit on seen, scored on unseen) and hrr vs vector-arithmetic analogies that build an
unseen pair from three seen ones (clifford_vae/disentangle.py).

    python train_disentangle.py --prior C --d 128 --seed 1 --beta_max 4
"""
from clifford_vae.train import main

DIMS = [128]
PRIORS = ["N", "N_L2", "S", "C"]
OVERRIDES = dict(epochs=100, warmup=20, cycle=20, patience=20)

if __name__ == "__main__":
    main("shapes3d", DIMS, PRIORS, OVERRIDES)

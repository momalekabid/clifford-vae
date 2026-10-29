# one interface over the cnn and mlp vaes for every prior.
#   N      gaussian                     N_L2   gaussian, mu and z l2-normalised
#   S      power spherical              V      von mises-fisher (mlp only)
#   C      clifford (power spherical per circle, one kappa per circle)
# a beta-vae is any of these trained with --beta_max > 1 (shapes2d uses beta_max = 4).
#
# N_L2: the kl is the gaussian kl of the pre-normalisation latent. normalisation is
# treated as the first layer of the decoder, so this is the exact kl of that model.
import math

import torch
import torch.nn.functional as F

from revision import common  # noqa: F401  (puts repo + patched vmf on sys.path)
from cnn.models import VAE
from mnist.mlp_vae import MLPVAE, vae_loss
from dists.clifford import CliffordTorusUniform, HypersphericalUniform

PRIORS = ["N", "N_L2", "S", "V", "C"]
EUCLIDEAN_PRIORS = {"N", "N_L2"}  # k-NN metric; cosine for the rest


def build(prior: str, d: int, data, dev):
    if data.arch == "cnn":
        dist = {"N": "gaussian", "N_L2": "gaussian", "S": "powerspherical", "C": "clifford"}[prior]
        m = VAE(d, data.channels, dist, str(dev), recon_loss_type="l1", l2_normalize=(prior == "N_L2"),
                img_size=data.img_size)
    else:
        dist = {"N": "normal", "N_L2": "normal", "S": "powerspherical", "V": "vmf", "C": "clifford"}[prior]
        # S and V live on S^d in R^{d+1} (davidson et al. 2018 convention)
        z = d + 1 if prior in ("S", "V") else d
        x_dim = data.channels * data.img_size * data.img_size
        m = MLPVAE(0, z, dist, l2_normalize=(prior == "N_L2"), x_dim=x_dim).to(dev)
    m.prior_name = prior
    m.arch = data.arch
    return m


def loss(model, x, beta: float):
    """returns total, recon, kl (per image, mean over batch) and the posterior."""
    if model.arch == "cnn":
        xr, q, p, _ = model(x)
        out = model.compute_loss(x, xr, q, p, beta)
        return out["total_loss"], out["recon_loss"], out["kld_loss"], q
    out = vae_loss(model, x, beta=beta, return_dict=True)
    return out["total"], out["recon"], out["kl"], None


def concentration_stats(model, x):
    """mean kappa and the fraction of kappas at the cap (S, V, C)."""
    if model.prior_name not in ("S", "V", "C"):
        return {}
    with torch.no_grad():
        if model.arch == "cnn":
            _, kappa = model.encoder(x)
        else:
            _, kappa = model.encode(x.view(len(x), -1))
    from cnn.models import kappa_cap
    if model.prior_name == "C":
        cap = kappa_cap(2, clifford=True)
    else:
        cap = kappa_cap(model.latent_dim if model.arch == "cnn" else model.z_dim)
    return {"kappa_mean": kappa.mean().item(), "kappa_at_cap": (kappa >= cap * 0.999).float().mean().item()}


@torch.no_grad()
def mean_codes(model, images, bs=1000):
    out = []
    for idx in images.batches(bs):
        out.append(model.mean_code(images.get(idx)).float())
    return torch.cat(out)


@torch.no_grad()
def decode(model, z):
    return model.decoder(z)


def prior_samples(model, n: int, gen: torch.Generator):
    """samples from each prior, for generation fid."""
    prior = model.prior_name
    dev = next(model.parameters()).device
    if prior == "C":
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        return CliffordTorusUniform(model.latent_dim, device=dev).rsample((n,))
    d = model.latent_dim
    z = torch.randn(n, d, generator=gen).to(dev)
    if prior in ("N_L2", "S"):
        z = F.normalize(z, dim=-1)  # uniform on the sphere = normalised gaussian
    return z


def n_params(model):
    return sum(p.numel() for p in model.parameters())

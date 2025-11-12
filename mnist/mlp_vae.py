import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dists.clifford import (
    PowerSpherical,
    HypersphericalUniform,
    CliffordPowerSphericalDistribution,
    CliffordTorusUniform,
)


class MLPVAE(nn.Module):
    def __init__(self, h_dim: int, z_dim: int, distribution: str = "normal", l2_normalize: bool = False):
        super().__init__()
        self.z_dim = z_dim
        self.distribution = distribution
        self.l2_normalize = l2_normalize

        # encoder: 784 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        if self.distribution == "normal":
            self.fc_mean = nn.Linear(128, z_dim)
            self.fc_var = nn.Linear(128, z_dim)
        else:
            self.fc_mean = nn.Linear(128, z_dim)
            self.fc_scale = nn.Linear(128, 1)

        # decoder: (z or 2z) -> 128 -> 256 -> 784
        decoder_in_dim = 2 * z_dim if self.distribution == "clifford" else z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

        # xavier/glorot initialization
        def init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        if self.distribution == "normal":
            z_mean = self.fc_mean(h)
            if self.l2_normalize:
                z_mean = F.normalize(z_mean, p=2, dim=-1)
            return z_mean, self.fc_var(h)
        elif self.distribution in ["powerspherical", "vmf"]:
            z_mean = F.normalize(self.fc_mean(h), p=2, dim=-1)
            z_scale = F.softplus(self.fc_scale(h)) + 1
            z_scale = torch.clamp(z_scale, max=1.0)
            return z_mean, z_scale
        else:  # clifford
            z_mean_angles = self.fc_mean(h)
            z_scale = F.softplus(self.fc_scale(h)) + 1
            z_scale = torch.clamp(z_scale, max=1.0)
            return z_mean_angles, z_scale

    def reparameterize(self, z_mean: torch.Tensor, z_param2: torch.Tensor):
        device = z_mean.device
        if self.distribution == "normal":
            std = torch.exp(0.5 * z_param2) + 1e-6
            q_z = torch.distributions.Normal(z_mean, std)
            p_z = torch.distributions.Normal(
                torch.zeros_like(z_mean), torch.ones_like(std)
            )
        elif self.distribution == "powerspherical":
            q_z = PowerSpherical(z_mean, z_param2.squeeze(-1))
            p_z = HypersphericalUniform(self.z_dim, device=device, validate_args=False)
        elif self.distribution == "vmf":
            from hyperspherical_vae.distributions import VonMisesFisher
            from hyperspherical_vae.distributions.hyperspherical_uniform import (
                HypersphericalUniform as VMFUniform,
            )

            q_z = VonMisesFisher(z_mean, z_param2)
            p_z = VMFUniform(self.z_dim - 1, device=device, validate_args=False)
        else:  # clifford
            q_z = CliffordPowerSphericalDistribution(z_mean, z_param2)
            p_z = CliffordTorusUniform(self.z_dim, device=device, validate_args=False)
        return q_z, p_z

    def forward(self, x: torch.Tensor):
        z_mean, z_param2 = self.encode(x.view(-1, 784))
        q_z, p_z = self.reparameterize(z_mean, z_param2)
        z = q_z.rsample()
        if self.distribution == "normal" and self.l2_normalize:
            z = F.normalize(z, p=2, dim=-1)
        x_recon = self.decoder(z)
        return (z_mean, z_param2), (q_z, p_z), z, x_recon


def vae_loss(model: MLPVAE, x: torch.Tensor, beta: float = 1.0, return_dict: bool = False):
    """compute vae loss with optional detailed metrics.

    args:
        model: the vae model
        x: input batch
        beta: kl weight for beta-vae
        return_dict: if true, return dict with all metrics; if false, return scalar loss

    returns:
        if return_dict: dict with keys 'total', 'recon', 'kl', 'entropy', 'elbo'
        else: scalar total loss
    """
    _, (q_z, p_z), _, x_recon = model(x)

    # reconstruction loss (negative log likelihood)
    recon = F.binary_cross_entropy_with_logits(
        x_recon, x.view(-1, 784), reduction="sum"
    ) / x.size(0)

    # kl divergence
    kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    # entropy of approximate posterior
    try:
        entropy = q_z.entropy().mean()
    except (NotImplementedError, AttributeError):
        # fallback for distributions without entropy
        entropy = torch.tensor(0.0, device=x.device)

    total = recon + beta * kl

    if return_dict:
        return {
            "total": total,
            "recon": recon,
            "kl": kl,
            "entropy": entropy,
            "elbo": -recon - kl,  # elbo = log p(x|z) - kl = -recon - kl
        }
    return total


def compute_log_likelihood(model: MLPVAE, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
    """compute importance-weighted log-likelihood estimate (iwae bound).

    args:
        model: the vae model
        x: input batch (B, 784) or (B, 1, 28, 28)
        n_samples: number of mc samples for importance weighting

    returns:
        mc estimate of log p(x) averaged over batch
    """
    x_flat = x.view(-1, 784)
    z_mean, z_param2 = model.encode(x_flat)
    q_z, p_z = model.reparameterize(z_mean, z_param2)

    # sample n times: shape (n_samples, batch, z_dim)
    z = q_z.rsample(torch.Size([n_samples]))

    # decode each sample
    if model.distribution == "clifford":
        # clifford samples are already 2*z_dim
        x_recon = model.decoder(z)
    else:
        x_recon = model.decoder(z)

    # log p(z) - prior log prob
    log_p_z = p_z.log_prob(z)
    if model.distribution == "normal":
        log_p_z = log_p_z.sum(-1)  # (n_samples, batch)
    # for other distributions, log_prob already returns scalar per sample

    # log p(x|z) - reconstruction likelihood
    x_expanded = x_flat.unsqueeze(0).expand(n_samples, -1, -1)  # (n_samples, batch, 784)
    log_p_x_z = -F.binary_cross_entropy_with_logits(
        x_recon, x_expanded, reduction='none'
    ).sum(-1)  # (n_samples, batch)

    # log q(z|x) - encoder log prob
    log_q_z_x = q_z.log_prob(z)
    if model.distribution == "normal":
        log_q_z_x = log_q_z_x.sum(-1)  # (n_samples, batch)

    # importance weighted estimate: log(1/n * sum_i w_i) where w_i = p(x,z)/q(z|x)
    # = logsumexp(log_p_x_z + log_p_z - log_q_z_x) - log(n)
    log_weights = log_p_x_z + log_p_z - log_q_z_x  # (n_samples, batch)
    ll = log_weights.logsumexp(dim=0) - torch.log(torch.tensor(float(n_samples), device=x.device))

    return ll.mean()


def compute_test_metrics(model: MLPVAE, loader, device, n_iwae_samples: int = 10) -> dict:
    """compute evaluation metrics on a dataset for results table.

    uses importance-weighted bound for LL estimate like davidson et al.

    returns dict with:
        - ll: log-likelihood estimate (iwae bound)
        - entropy: entropy of approximate posterior L[q]
        - recon: reconstruction error (negative, i.e. log p(x|z))
        - kl: kl divergence
    """
    model.eval()
    metrics = {"ll": 0.0, "entropy": 0.0, "recon": 0.0, "kl": 0.0}
    n_total = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            batch_size = x.size(0)

            # get elbo components
            result = vae_loss(model, x, beta=1.0, return_dict=True)

            # accumulate (recon is positive bce, so negate for log p(x|z))
            metrics["recon"] += -result["recon"].item() * batch_size
            metrics["kl"] += result["kl"].item() * batch_size
            metrics["entropy"] += result["entropy"].item() * batch_size

            # importance-weighted ll estimate
            ll = compute_log_likelihood(model, x, n_samples=n_iwae_samples)
            metrics["ll"] += ll.item() * batch_size

            n_total += batch_size

    # average over dataset
    for k in metrics:
        metrics[k] /= n_total

    return metrics

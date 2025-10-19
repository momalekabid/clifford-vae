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
            z_scale = torch.clamp(z_scale, max=100.0)
            return z_mean, z_scale
        else:  # clifford
            z_mean_angles = self.fc_mean(h)
            z_scale = F.softplus(self.fc_scale(h)) + 1
            z_scale = torch.clamp(z_scale, max=100.0)
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


def vae_loss(model: MLPVAE, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    _, (q_z, p_z), _, x_recon = model(x)
    recon = F.binary_cross_entropy_with_logits(
        x_recon, x.view(-1, 784), reduction="sum"
    ) / x.size(0)
    kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    return recon + beta * kl

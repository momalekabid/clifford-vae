# mlp vae for dsprites — FC(1200)->FC(1200) following L-VAE (Ozcan et al. 2025)
# binary sprites: sigmoid output + BCE loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dists.clifford import (
    PowerSpherical,
    HypersphericalUniform,
    CliffordPowerSphericalDistribution,
    CliffordTorusUniform,
)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 1200,
        distribution: str = "gaussian",
        l2_normalize: bool = False,
        concentration_floor: float = 0.05,
    ):
        super().__init__()
        self.distribution = distribution
        self.l2_normalize = l2_normalize
        self.concentration_floor = concentration_floor

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        if distribution == "gaussian":
            self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        else:
            self.fc_concentration = nn.Linear(hidden_dim, 1)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        if self.distribution == "gaussian":
            if self.l2_normalize:
                mu = F.normalize(mu, p=2, dim=-1)
            return mu, self.fc_log_var(h)
        elif self.distribution == "powerspherical":
            mu = F.normalize(mu, p=2, dim=-1)
            kappa = torch.clamp(F.softplus(self.fc_concentration(h)) + 0.8, max=10.0)
            return mu, kappa
        elif self.distribution == "clifford":
            kappa = torch.clamp(F.softplus(self.fc_concentration(h)) + self.concentration_floor, max=10.0)
            return mu, kappa


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 1200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_activation = "sigmoid"

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, z):
        # returns logits; sigmoid applied in loss or externally
        return self.net(z)


class DSpritesVAE(nn.Module):
    """mlp vae for dsprites with same interface as cnn.models.VAE."""

    def __init__(
        self,
        latent_dim: int,
        distribution: str,
        device: str,
        img_size: int = 64,
        in_channels: int = 1,
        hidden_dim: int = 1200,
        recon_loss_type: str = "bce",
        l1_weight: float = 1.0,
        l2_normalize: bool = False,
        concentration_floor: float = 0.05,
        use_learnable_beta: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.device = device
        self.l2_normalize = l2_normalize
        self.recon_loss_type = recon_loss_type
        self.l1_weight = l1_weight
        self.use_learnable_beta = use_learnable_beta
        self.img_size = img_size
        self.in_channels = in_channels

        input_dim = img_size * img_size * in_channels

        # scale floor with dim for clifford
        if distribution == "clifford":
            if latent_dim < 256:
                concentration_floor = 0.04
            elif latent_dim <= 512:
                concentration_floor = 0.07
            elif latent_dim <= 1024:
                concentration_floor = 0.10
            elif latent_dim <= 2048:
                concentration_floor = 0.13
            else:
                concentration_floor = 0.16
        self.concentration_floor = concentration_floor

        if use_learnable_beta:
            self.log_sigma_0 = nn.Parameter(torch.zeros(1))
            self.log_sigma_1 = nn.Parameter(torch.zeros(1))

        self.encoder = MLPEncoder(
            input_dim, latent_dim, hidden_dim,
            distribution=distribution,
            l2_normalize=l2_normalize,
            concentration_floor=concentration_floor,
        )

        dec_in_dim = 2 * latent_dim if distribution == "clifford" else latent_dim
        self.decoder = MLPDecoder(dec_in_dim, input_dim, hidden_dim)

        self.to(device)

    def reparameterize(self, mu, params):
        if self.distribution == "gaussian":
            q_z = torch.distributions.Normal(mu, torch.exp(0.5 * params) + 1e-6)
            p_z = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(params)
            )
            z = q_z.rsample()
            if self.l2_normalize:
                z = F.normalize(z, p=2, dim=-1)
            return z, q_z, p_z
        elif self.distribution == "powerspherical":
            q_z = PowerSpherical(mu, params.squeeze(-1))
            p_z = HypersphericalUniform(
                self.latent_dim, device=self.device, validate_args=False
            )
            return q_z.rsample(), q_z, p_z
        elif self.distribution == "clifford":
            q_z = CliffordPowerSphericalDistribution(mu, params.expand_as(mu))
            p_z = CliffordTorusUniform(
                self.latent_dim, device=self.device, validate_args=False
            )
            z = q_z.rsample()
            return z, q_z, p_z

    def get_flat_latent(self, x):
        """encode and return flat latent vector."""
        x_flat = x.view(x.size(0), -1)
        mu, params = self.encoder(x_flat)
        z, _, _ = self.reparameterize(mu, params)
        return z

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        mu, params = self.encoder(x_flat)
        z, q_z, p_z = self.reparameterize(mu, params)
        logits = self.decoder(z)
        # store logits for BCE in compute_loss
        self._last_logits = logits
        x_recon = torch.sigmoid(logits).view(x.shape)
        return x_recon, q_z, p_z, mu

    def compute_loss(self, x, x_recon, q_z, p_z, beta=1.0):
        B = x.size(0)
        x_flat = x.view(B, -1)

        if self.distribution == "gaussian":
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).sum(dim=1).mean()
        else:
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        if self.recon_loss_type == "bce":
            logits = self._last_logits.view(B, -1)
            recon_loss = F.binary_cross_entropy_with_logits(
                logits, x_flat, reduction="sum"
            ) / B
        elif self.recon_loss_type == "mse":
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / B
        elif self.recon_loss_type == "l1":
            recon_loss = self.l1_weight * F.l1_loss(x_recon, x, reduction="sum") / B
        else:
            raise ValueError(f"unknown recon loss: {self.recon_loss_type}")

        if self.use_learnable_beta:
            sigma_0 = torch.exp(self.log_sigma_0)
            sigma_1 = torch.exp(self.log_sigma_1)
            total_loss = (1.0 / (sigma_0 ** 2)) * recon_loss + (1.0 / (sigma_1 ** 2)) * kld + sigma_0 ** 2 + sigma_1 ** 2
            effective_beta = (sigma_0 / sigma_1) ** 2
        else:
            total_loss = recon_loss + beta * kld
            effective_beta = beta

        try:
            entropy = q_z.entropy().mean()
        except (NotImplementedError, AttributeError):
            entropy = torch.tensor(0.0, device=x.device)

        result = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kld_loss": kld,
            "entropy": entropy,
            "effective_beta": effective_beta,
        }

        if self.use_learnable_beta:
            result["sigma_0"] = sigma_0.item()
            result["sigma_1"] = sigma_1.item()

        return result

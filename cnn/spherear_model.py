# sphereAR-style S-VAE for CIFAR-10
# per-token latent architecture with constant-norm constraint,
# using clifford/powerspherical/gaussian distributions from dists/clifford.py

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


class ResDownBlock(nn.Module):
    """residual downsampling block with groupnorm + silu"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, max(1, in_ch // 4)), in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, max(1, out_ch // 4)), out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Conv2d(
            in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ResUpBlock(nn.Module):
    """residual upsampling block with groupnorm + silu"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = min(32, max(1, out_ch // 4))
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, max(1, in_ch // 4)), in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.shortcut(x) + self.block(x)
        x = x + self.block2(x)
        return x


class SphereAREncoder(nn.Module):
    """per-token encoder: outputs (mu, kappa) per spatial location.

    distribution controls what mu/kappa mean:
    - clifford: mu is angles (latent_dim), kappa is concentration
    - powerspherical: mu is unit direction (latent_dim), kappa is concentration
    - gaussian: mu is mean (latent_dim), log_var is variance
    """

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 3,
        distribution: str = "clifford",
        cnn_chs: list = None,
        concentration_floor: float = 0.03,
    ):
        super().__init__()
        self.distribution = distribution
        self.concentration_floor = concentration_floor

        if cnn_chs is None:
            cnn_chs = [64, 128, 256]

        self.input_conv = nn.Conv2d(in_channels, cnn_chs[0], kernel_size=3, stride=1, padding=1, bias=False)

        blocks = []
        for i in range(len(cnn_chs) - 1):
            blocks.append(ResDownBlock(cnn_chs[i], cnn_chs[i + 1]))
        self.down_blocks = nn.Sequential(*blocks)

        # output heads
        self.fc_mu = nn.Conv2d(cnn_chs[-1], latent_dim, kernel_size=1, bias=True)
        if distribution == "gaussian":
            self.fc_logvar = nn.Conv2d(cnn_chs[-1], latent_dim, kernel_size=1, bias=True)
        else:
            self.fc_kappa = nn.Conv2d(cnn_chs[-1], 1, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.down_blocks(x)  # (B, C, H', W')

        mu_map = self.fc_mu(x)  # (B, latent_dim, H', W')
        B, D, H, W = mu_map.shape
        # reshape to (B, num_tokens, latent_dim)
        mu = mu_map.permute(0, 2, 3, 1).reshape(B, H * W, D)

        if self.distribution == "gaussian":
            logvar_map = self.fc_logvar(x)
            logvar = logvar_map.permute(0, 2, 3, 1).reshape(B, H * W, D)
            return mu, logvar
        elif self.distribution == "powerspherical":
            mu = F.normalize(mu, p=2, dim=-1)
            kappa_map = self.fc_kappa(x)  # (B, 1, H', W')
            kappa = kappa_map.permute(0, 2, 3, 1).reshape(B, H * W)
            kappa = F.softplus(kappa) + 1.0
            kappa = torch.clamp(kappa, max=1.0)
            return mu, kappa
        elif self.distribution == "clifford":
            # mu is raw angles, kappa is concentration per token
            kappa_map = self.fc_kappa(x)
            kappa = kappa_map.permute(0, 2, 3, 1).reshape(B, H * W)
            kappa = F.softplus(kappa) + self.concentration_floor
            return mu, kappa


class SphereARDecoder(nn.Module):
    """per-token decoder that reconstructs from latent tokens."""

    def __init__(
        self,
        latent_dim: int = 16,
        out_channels: int = 3,
        cnn_chs: list = None,
        spatial_size: int = 4,
    ):
        super().__init__()
        if cnn_chs is None:
            cnn_chs = [256, 128, 64]

        self.spatial_size = spatial_size
        self.input_proj = nn.Linear(latent_dim, cnn_chs[0], bias=False)

        blocks = []
        for i in range(len(cnn_chs) - 1):
            blocks.append(ResUpBlock(cnn_chs[i], cnn_chs[i + 1]))
        self.up_blocks = nn.Sequential(*blocks)

        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(32, max(1, cnn_chs[-1] // 4)), cnn_chs[-1], eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(cnn_chs[-1], out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # z: (B, num_tokens, dec_dim)
        B, L, D = z.shape
        H = W = self.spatial_size

        x = self.input_proj(z)  # (B, L, C)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.up_blocks(x)
        x = self.output_conv(x)
        return x


class SphereARVAE(nn.Module):
    """per-token S-VAE using clifford/powerspherical/gaussian distributions.

    inspired by SphereAR (Ke & Xue, ICLR 2026): each spatial token gets its
    own latent vector with constant-norm constraint (for non-gaussian dists).

    for clifford: decoder input is 2*latent_dim per token (bivector via ifft).
    for powerspherical: decoder input is latent_dim per token.
    for gaussian: decoder input is latent_dim per token (optionally l2 normed).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 3,
        distribution: str = "clifford",
        device: str = "cpu",
        recon_loss_type: str = "l1",
        l1_weight: float = 1.0,
        encoder_chs: list = None,
        decoder_chs: list = None,
        use_learnable_beta: bool = False,
        l2_normalize: bool = False,
        concentration_floor: float = 0.03,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.device = device
        self.recon_loss_type = recon_loss_type
        self.l1_weight = l1_weight
        self.use_learnable_beta = use_learnable_beta
        self.l2_normalize = l2_normalize

        if encoder_chs is None:
            encoder_chs = [64, 128, 256]
        if decoder_chs is None:
            decoder_chs = encoder_chs[::-1]

        num_down = len(encoder_chs) - 1
        self.token_spatial_size = 32 // (2 ** num_down)

        self.encoder = SphereAREncoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
            distribution=distribution,
            cnn_chs=encoder_chs,
            concentration_floor=concentration_floor,
        )

        # decoder input dim depends on distribution
        dec_in_dim = 2 * latent_dim if distribution == "clifford" else latent_dim
        self.decoder = SphereARDecoder(
            latent_dim=dec_in_dim,
            out_channels=in_channels,
            cnn_chs=decoder_chs,
            spatial_size=self.token_spatial_size,
        )

        if use_learnable_beta:
            self.log_sigma_0 = nn.Parameter(torch.zeros(1))
            self.log_sigma_1 = nn.Parameter(torch.zeros(1))

        self.to(device)

    def reparameterize(self, mu, params):
        """reparameterize per-token: mu is (B, num_tokens, latent_dim)"""
        B, T, D = mu.shape

        if self.distribution == "gaussian":
            # params is log_var: (B, T, D)
            q_z = torch.distributions.Normal(mu, torch.exp(0.5 * params) + 1e-6)
            p_z = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(params)
            )
            z = q_z.rsample()
            if self.l2_normalize:
                z = F.normalize(z, p=2, dim=-1)
            return z, q_z, p_z

        elif self.distribution == "powerspherical":
            # mu: (B, T, D) unit vectors, params/kappa: (B, T)
            kappa = params
            # flatten tokens into batch for distribution
            mu_flat = mu.reshape(B * T, D)
            kappa_flat = kappa.reshape(B * T)
            q_z = PowerSpherical(mu_flat, kappa_flat)
            p_z = HypersphericalUniform(D, device=mu.device, validate_args=False)
            z = q_z.rsample().reshape(B, T, D)
            return z, q_z, p_z

        elif self.distribution == "clifford":
            # mu: (B, T, D) raw angles, params/kappa: (B, T)
            kappa = params
            mu_flat = mu.reshape(B * T, D)
            kappa_flat = kappa.reshape(B * T).unsqueeze(-1).expand_as(mu_flat)
            q_z = CliffordPowerSphericalDistribution(mu_flat, kappa_flat)
            p_z = CliffordTorusUniform(D, device=mu.device, validate_args=False)
            z = q_z.rsample().reshape(B, T, 2 * D)  # bivector
            return z, q_z, p_z

    def forward(self, x):
        mu, params = self.encoder(x)
        z, q_z, p_z = self.reparameterize(mu, params)
        x_recon = self.decoder(z)
        return x_recon, q_z, p_z, mu

    def compute_loss(self, x, x_recon, q_z, p_z, beta=1.0):
        B = x.size(0)

        if self.distribution == "gaussian":
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).sum(dim=-1).mean()
        else:
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        if self.recon_loss_type == "l1":
            recon_loss = self.l1_weight * (F.l1_loss(x_recon, x, reduction="sum") / B)
        elif self.recon_loss_type == "mse":
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / B
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

    def get_flat_latent(self, x):
        """get flattened latent for VSA tests"""
        mu, _ = self.encoder(x)
        # flatten tokens: (B, num_tokens, latent_dim) -> (B, num_tokens * latent_dim)
        return mu.reshape(mu.size(0), -1)

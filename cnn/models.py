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


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, in_channels: int, distribution: str):
        super().__init__()
        self.distribution = distribution
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = 512 * 2 * 2
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        if distribution == "gaussian":
            self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)
        else:
            self.fc_concentration = nn.Linear(self.flat_dim, 1)

    def forward(self, x):
        x = self.main(x).flatten(start_dim=1)
        mu = self.fc_mu(x)
        if self.distribution == "gaussian":
            return mu, self.fc_log_var(x)
        elif self.distribution == "powerspherical":
            mu = F.normalize(mu, p=2, dim=-1)
            kappa = F.softplus(self.fc_concentration(x)) + 1
            return mu, torch.clamp(kappa, max=100.0)
        elif self.distribution == "clifford":
            kappa = F.softplus(self.fc_concentration(x)) + 1
            return mu, torch.clamp(kappa, max=100.0)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 512, 2, 2)
        return self.main(x)


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        in_channels: int,
        distribution: str,
        device: str,
        recon_loss_type: str = "l1_freq",
        use_perceptual_loss: bool = False,
        l1_weight: float = 1.0,
        freq_weight: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.device = device

        self.recon_loss_type = recon_loss_type
        self.l1_weight = l1_weight
        self.freq_weight = freq_weight
        self.use_perceptual_loss = use_perceptual_loss

        self.encoder = Encoder(
            latent_dim, in_channels=in_channels, distribution=distribution
        )
        dec_in_dim = 2 * latent_dim if distribution == "clifford" else latent_dim
        self.decoder = Decoder(dec_in_dim, out_channels=in_channels)

        self.to(device)

    def _create_radial_freq_mask(self, height, width):
        """radial frequency mask that targets low frequencies."""
        center_y, center_x = height // 2, width // 2
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )
        dist_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_dist = torch.sqrt(
            torch.tensor(
                center_y**2 + center_x**2, device=self.device, dtype=torch.float32
            )
        )
        normalized_dist = dist_from_center / max_dist
        mask = 1.0 - normalized_dist
        return mask

    def reparameterize(self, mu, params):
        if self.distribution == "gaussian":
            q_z = torch.distributions.Normal(mu, torch.exp(0.5 * params) + 1e-6)
            p_z = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(params)
            )
            return q_z.rsample(), q_z, p_z
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
            z = q_z.rsample()  # shape (..., 2*latent_dim)
            return z, q_z, p_z

    def forward(self, x):
        mu, params = self.encoder(x)
        z, q_z, p_z = self.reparameterize(mu, params)
        x_recon = self.decoder(z)
        return x_recon, q_z, p_z, mu

    def compute_loss(self, x, x_recon, q_z, p_z, beta=1.0):
        B = x.size(0)

        if self.distribution == "gaussian":
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).sum(dim=1).mean()
        else:
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        recon_loss = 0.0

        if self.recon_loss_type == "mse":
            # sum over pixels, average over batch
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / B

        elif self.recon_loss_type == "l1_freq":
            # pixel L1: sum over pixels, average over batch
            if self.l1_weight > 0:
                recon_loss += self.l1_weight * (
                    F.l1_loss(x_recon, x, reduction="sum") / B
                )

            # frequency L1: sum over all freq bins, average over batch
            if self.freq_weight > 0:
                h, w = x.shape[-2:]
                freq_mask = self._create_radial_freq_mask(h, w)
                x_fft = torch.fft.fft2(x, dim=(-2, -1))
                x_recon_fft = torch.fft.fft2(x_recon, dim=(-2, -1))
                x_fft_mag = torch.abs(x_fft)
                x_recon_fft_mag = torch.abs(x_recon_fft)
                loss_freq = (
                    F.l1_loss(
                        x_recon_fft_mag * freq_mask,
                        x_fft_mag * freq_mask,
                        reduction="sum",
                    )
                    / B
                )
                recon_loss += self.freq_weight * loss_freq
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}"
            )

        total_loss = recon_loss + beta * kld

        return {"total_loss": total_loss, "recon_loss": recon_loss, "kld_loss": kld}

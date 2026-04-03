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


class ResBlock(nn.Module):
    """conv block with residual skip connection and stride-2 downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.act(self.conv(x)) + self.pool(self.skip(x))


class ResUpBlock(nn.Module):
    """transpose conv block with residual skip connection and stride-2 upsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.act(self.conv(x)) + self.up(self.skip(x))


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        in_channels: int,
        distribution: str,
        l2_normalize: bool = False,
        concentration_floor: float = 0.1,
        img_size: int = 32,
    ):
        super().__init__()
        self.distribution = distribution
        self.l2_normalize = l2_normalize
        self.concentration_floor = concentration_floor

        if img_size == 64:
            # 64->32->16->8->4->2
            chs = [in_channels, 64, 128, 256, 512, 512]
        else:
            # 32->16->8->4->2
            chs = [in_channels, 64, 128, 256, 512]

        self.blocks = nn.ModuleList([
            ResBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)
        ])
        self.flat_dim = 512 * 2 * 2
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        if distribution == "gaussian":
            self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)
        else:
            self.fc_concentration = nn.Linear(self.flat_dim, 1)

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        if self.distribution == "gaussian":
            if self.l2_normalize:
                mu = F.normalize(mu, p=2, dim=-1)
            return mu, self.fc_log_var(x)
        elif self.distribution == "powerspherical":
            mu = F.normalize(mu, p=2, dim=-1)
            kappa = torch.clamp(F.softplus(self.fc_concentration(x)) + 0.8, max=10.0)
            return mu, kappa
        elif self.distribution == "clifford":
            kappa = torch.clamp(F.softplus(self.fc_concentration(x)) + self.concentration_floor, max=10.0)
            return mu, kappa


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, img_size: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)

        if img_size == 64:
            chs = [512, 512, 256, 128, 64]
        else:
            chs = [512, 256, 128, 64]

        self.blocks = nn.ModuleList([
            ResUpBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)
        ])
        # final layer: no skip, just project to output channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(chs[-1], out_channels, 4, 2, 1),
            nn.Tanh(),
        )

        def init_weights(m):
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 512, 2, 2)
        for block in self.blocks:
            x = block(x)
        return self.final(x)


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        in_channels: int,
        distribution: str,
        device: str,
        recon_loss_type: str = "l1",
        use_perceptual_loss: bool = False,
        l1_weight: float = 1.0,
        freq_weight: float = 1.0,
        l2_normalize: bool = False,
        concentration_floor: float = 0.05, # empirically determined/safe: this is 1 for vmf/pws
        img_size: int = 32,
        use_learnable_beta: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.device = device
        self.l2_normalize = l2_normalize
        # scale floor with dim for clifford so kappa doesn't collapse at high d
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
        self.use_learnable_beta = use_learnable_beta

        self.recon_loss_type = recon_loss_type
        self.l1_weight = l1_weight
        self.freq_weight = freq_weight
        self.use_perceptual_loss = use_perceptual_loss

        # learnable beta uses log weights; σ₀ for reconstruction, σ₁ for kld
        if use_learnable_beta:
            self.log_sigma_0 = nn.Parameter(torch.zeros(1))
            self.log_sigma_1 = nn.Parameter(torch.zeros(1))

        self.encoder = Encoder(
            latent_dim,
            in_channels=in_channels,
            distribution=distribution,
            l2_normalize=l2_normalize,
            concentration_floor=concentration_floor,
            img_size=img_size,
        )
        dec_in_dim = 2 * latent_dim if distribution == "clifford" else latent_dim
        self.decoder = Decoder(dec_in_dim, out_channels=in_channels, img_size=img_size)

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
            z = q_z.rsample()  # shape (..., 2*latent_dim)
            return z, q_z, p_z

    def get_flat_latent(self, x):
        """encode and return flat latent vector (B, latent_dim or 2*latent_dim)."""
        mu, params = self.encoder(x)
        z, _, _ = self.reparameterize(mu, params)
        return z

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
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / B

        elif self.recon_loss_type == "l1":
            if self.l1_weight > 0:
                recon_loss += self.l1_weight * (
                    F.l1_loss(x_recon, x, reduction="sum") / B
                )

            # frequency l1 to penalize  
            # if self.freq_weight > 0:
            #     h, w = x.shape[-2:]
            #     freq_mask = self._create_radial_freq_mask(h, w)
            #     x_fft = torch.fft.fft2(x, dim=(-2, -1))
            #     x_recon_fft = torch.fft.fft2(x_recon, dim=(-2, -1))
            #     x_fft_mag = torch.abs(x_fft)
            #     x_recon_fft_mag = torch.abs(x_recon_fft)
            #     loss_freq = (
            #         F.l1_loss(
            #             x_recon_fft_mag * freq_mask,
            #             x_fft_mag * freq_mask,
            #             reduction="sum",
            #         )
            #         / B
            #     )
            #     recon_loss += self.freq_weight * loss_freq
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}"
            )

        # learnable beta: L = (1/σ₀²)*recon + (1/σ₁²)*kld + σ₀² + σ₁²
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


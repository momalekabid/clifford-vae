import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
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


class VAEWithLearnableRadius(nn.Module):
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
        # Radius learning controls
        learn_radius: bool = True,
        bayesian_radius: bool = True,
        radius_prior_mu: float = 0.0,  # prior over log(r)
        radius_prior_sigma: float = 0.25,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.device = device
        self.recon_loss_type = recon_loss_type
        self.l1_weight = l1_weight
        self.freq_weight = freq_weight
        self.use_perceptual_loss = use_perceptual_loss

        self.learn_radius = learn_radius and distribution in {"powerspherical", "clifford"}
        self.bayesian_radius = bayesian_radius
        self.radius_prior_mu = torch.tensor(radius_prior_mu, dtype=torch.float32)
        self.radius_prior_sigma = torch.tensor(radius_prior_sigma, dtype=torch.float32)

        self.encoder = Encoder(latent_dim, in_channels=in_channels, distribution=distribution)
        dec_in_dim = 2 * latent_dim if distribution == "clifford" else latent_dim
        self.decoder = Decoder(dec_in_dim, out_channels=in_channels)

        if self.use_perceptual_loss:
            try:
                import lpips

                self.lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
                for param in self.lpips_loss_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not available. Disabling perceptual loss.")
                self.use_perceptual_loss = False

        # Learnable radius parameters (global, batch-shared)
        if self.learn_radius:
            if self.bayesian_radius:
                # q(log r) = Normal(mu, sigma). Use softplus reparam for sigma.
                self.log_r_mu = nn.Parameter(torch.tensor(0.0))
                self.log_r_rho = nn.Parameter(torch.tensor(-1.3862944))  # softplus -> ~0.2
            else:
                # Deterministic radius with L2 prior penalty on log r
                self.log_r = nn.Parameter(torch.tensor(0.0))

        self.to(device)

    def _create_radial_freq_mask(self, height, width):
        center_y, center_x = height // 2, width // 2
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )
        dist_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_dist = torch.sqrt(
            torch.tensor(center_y**2 + center_x**2, device=self.device, dtype=torch.float32)
        )
        normalized_dist = dist_from_center / max_dist
        mask = 1.0 - normalized_dist
        return mask

    def _sample_radius(self, batch_size: int) -> tuple[torch.Tensor, float]:
        """Return r (B, 1) and KL_r scalar."""
        if not self.learn_radius:
            one = torch.ones(batch_size, 1, device=self.device, dtype=torch.float32)
            return one, 0.0

        if self.bayesian_radius:
            sigma = F.softplus(self.log_r_rho) + 1e-6
            eps = torch.randn(batch_size, 1, device=self.device, dtype=torch.float32)
            log_r = self.log_r_mu + sigma * eps
            r = torch.exp(log_r)

            # KL[q(log r) || p(log r)] where both are Normal
            mu_q, sigma_q = self.log_r_mu, sigma
            mu_p = self.radius_prior_mu.to(self.device)
            sigma_p = self.radius_prior_sigma.to(self.device)
            kl = torch.log(sigma_p / sigma_q) + (
                (sigma_q**2 + (mu_q - mu_p) ** 2) / (2 * sigma_p**2)
            ) - 0.5
            kl = kl.clamp(min=0.0)
            return r, float(kl)

        # Deterministic with L2 penalty on log r toward prior mean
        log_r = self.log_r
        r = torch.exp(log_r).expand(batch_size, 1)
        mu_p = self.radius_prior_mu.to(self.device)
        sigma_p = self.radius_prior_sigma.to(self.device)
        prior_penalty = ((log_r - mu_p) ** 2) / (2 * sigma_p**2)
        return r, float(prior_penalty)

    def reparameterize(self, mu, params, batch_size: int):
        if self.distribution == "gaussian":
            q_z = torch.distributions.Normal(mu, torch.exp(0.5 * params) + 1e-6)
            p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(params))
            z = q_z.rsample()
            return z, q_z, p_z, 0.0
        elif self.distribution == "powerspherical":
            q_z = PowerSpherical(mu, params.squeeze(-1))
            p_z = HypersphericalUniform(self.latent_dim, device=self.device, validate_args=False)
            z_dir = q_z.rsample()
            r, kl_r = self._sample_radius(batch_size)
            z = r * z_dir
            return z, q_z, p_z, kl_r
        elif self.distribution == "clifford":
            q_z = CliffordPowerSphericalDistribution(mu, params.expand_as(mu))
            p_z = CliffordTorusUniform(self.latent_dim, device=self.device, validate_args=False)
            z_dir = q_z.rsample()  # shape (..., 2*latent_dim)
            r, kl_r = self._sample_radius(batch_size)
            z = r * z_dir
            return z, q_z, p_z, kl_r

    def forward(self, x):
        mu, params = self.encoder(x)
        z, q_z, p_z, kl_r = self.reparameterize(mu, params, batch_size=x.size(0))
        x_recon = self.decoder(z)
        # stash KL_r for loss
        self._last_kl_r = torch.tensor(kl_r, device=self.device, dtype=torch.float32)
        return x_recon, q_z, p_z, mu

    def compute_loss(self, x, x_recon, q_z, p_z, beta=1.0):
        # --- KLD Loss for direction (as in original) ---
        if self.distribution == "gaussian":
            kld_dir = torch.distributions.kl.kl_divergence(q_z, p_z).sum(dim=1).mean()
        else:
            kld_dir = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        # --- Optional KL for radius ---
        kld_r = self._last_kl_r if hasattr(self, "_last_kl_r") else torch.tensor(0.0, device=self.device)

        # --- Reconstruction Loss ---
        recon_loss = 0.0
        if self.recon_loss_type == "mse":
            recon_loss = F.mse_loss(x_recon, x)
        elif self.recon_loss_type == "l1_freq":
            if self.l1_weight > 0:
                recon_loss += self.l1_weight * F.l1_loss(x_recon, x)
            if self.freq_weight > 0:
                h, w = x.shape[-2:]
                freq_mask = self._create_radial_freq_mask(h, w)
                x_fft = torch.fft.fft2(x, dim=(-2, -1))
                x_recon_fft = torch.fft.fft2(x_recon, dim=(-2, -1))
                x_fft_mag = torch.abs(x_fft)
                x_recon_fft_mag = torch.abs(x_recon_fft)
                loss_freq = F.l1_loss(x_recon_fft_mag * freq_mask, x_fft_mag * freq_mask)
                recon_loss += self.freq_weight * loss_freq
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.recon_loss_type}")

        if self.use_perceptual_loss:
            x_for_loss = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x
            x_recon_for_loss = x_recon.repeat(1, 3, 1, 1) if x_recon.size(1) == 1 else x_recon
            recon_loss += self.lpips_loss_fn(x_recon_for_loss, x_for_loss).mean()

        total_loss = recon_loss + beta * (kld_dir + kld_r)
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_dir + kld_r,
            "kld_dir": kld_dir,
            "kld_radius": kld_r,
        }



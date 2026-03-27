# hybrid CNN+ViT VAE matching CliffordAR (Ke & Xue, ICLR 2026) S-VAE architecture.
# supports clifford, powerspherical, and gaussian distributions.
# original: https://github.com/guolinke/CliffordAR

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

# ---- compatibility ----

try:
    RMSNorm = nn.RMSNorm
except AttributeError:
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---- 2d rotary position embeddings ----

def get_2d_pos(image_size, patch_size):
    grid_size = image_size // patch_size
    ys, xs = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing='ij'
    )
    return torch.stack([ys.flatten(), xs.flatten()], dim=-1).float()


def precompute_freqs_cis_2d(pos, head_dim, cls_token_num=0):
    half = head_dim // 4
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs_y = torch.outer(pos[:, 0], freqs)
    freqs_x = torch.outer(pos[:, 1], freqs)
    freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)
    freqs_cis = torch.polar(torch.ones_like(freqs_2d), freqs_2d)
    if cls_token_num > 0:
        cls_freqs = torch.ones(cls_token_num, freqs_cis.shape[1], dtype=freqs_cis.dtype)
        freqs_cis = torch.cat([cls_freqs, freqs_cis], dim=0)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    B, H, S, D = x.shape
    x_ = x.float().reshape(B, H, S, D // 2, 2)
    x_complex = torch.view_as_complex(x_)
    fc = freqs_cis[:S].unsqueeze(0).unsqueeze(0)
    x_rot = torch.view_as_real(x_complex * fc).reshape(B, H, S, D)
    return x_rot.type_as(x)


# ---- transformer components ----

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)
        d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis=None):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.wo(out.transpose(1, 2).reshape(B, S, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.attn = Attention(d_model, n_heads, causal=causal)
        self.norm2 = RMSNorm(d_model, eps=1e-6)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


# ---- CNN components (matching CliffordAR exactly) ----

class ResDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_ch // 4), in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch // 4), out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Conv2d(
            in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class PatchifyNet(nn.Module):
    def __init__(self, chs):
        super().__init__()
        layers = []
        for i in range(len(chs) - 1):
            layers.append(ResDownBlock(chs[i], chs[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NCHW_to_NLC(nn.Module):
    def forward(self, x):
        n, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(n, h * w, c)


class NLC_to_NCHW(nn.Module):
    def forward(self, x):
        n, l, c = x.shape
        h = w = int(l ** 0.5)
        return x.view(n, h, w, c).permute(0, 3, 1, 2)


class ResUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = min(32, out_ch // 4)
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_ch // 4), in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False
        )
        # extra residual block per stage (matching CliffordAR decoder)
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        h = self.block(x)
        x = self.shortcut(x)
        x = x + h
        x = x + self.block2(x)
        return x


class UnpatchifyNet(nn.Module):
    def __init__(self, chs):
        super().__init__()
        layers = []
        for i in range(len(chs) - 1):
            layers.append(ResUpBlock(chs[i], chs[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---- encoder / decoder ----

class ViTEncoder(nn.Module):
    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        d_model=512,
        cnn_chs=None,
        in_channels=3,
        image_size=256,
        patch_size=16,
        register_tokens=4,
    ):
        super().__init__()
        if cnn_chs is None:
            cnn_chs = [64, 64, 128, 256, 512]
        assert cnn_chs[-1] == d_model

        self.conv_in = nn.Conv2d(in_channels, cnn_chs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.patchify = nn.Sequential(PatchifyNet(cnn_chs), NCHW_to_NLC())

        self.register_num_tokens = register_tokens
        self.register_token = nn.Embedding(register_tokens, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, causal=False)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, eps=1e-6)
        self.output = nn.Linear(d_model, d_model, bias=False)

        raw_2d_pos = get_2d_pos(image_size, patch_size)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis_2d(
                raw_2d_pos, d_model // n_heads, cls_token_num=register_tokens
            ).clone(),
        )

    def forward(self, image):
        x = self.conv_in(image)
        x = self.patchify(x)
        x_null = self.register_token.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x_null, x], dim=1)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = x[:, self.register_num_tokens:, :]
        x = self.output(self.norm(x))
        return x


class ViTDecoder(nn.Module):
    def __init__(
        self,
        n_layers=12,
        n_heads=8,
        d_model=512,
        cnn_chs=None,
        out_channels=3,
        image_size=256,
        patch_size=16,
        register_tokens=4,
    ):
        super().__init__()
        if cnn_chs is None:
            cnn_chs = [512, 256, 128, 64, 64]
        assert d_model == cnn_chs[0]

        self.conv_in = nn.Sequential(
            NLC_to_NCHW(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            NCHW_to_NLC(),
        )

        self.register_num_tokens = register_tokens
        self.register_token = nn.Embedding(register_tokens, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, causal=False)
            for _ in range(n_layers)
        ])
        self.unpatchify = nn.Sequential(NLC_to_NCHW(), UnpatchifyNet(chs=cnn_chs))
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(16, cnn_chs[-1] // 4), cnn_chs[-1], eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(cnn_chs[-1], out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        raw_2d_pos = get_2d_pos(image_size, patch_size)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis_2d(
                raw_2d_pos, d_model // n_heads, cls_token_num=register_tokens
            ).clone(),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x_null = self.register_token.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x_null, x], dim=1)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = x[:, self.register_num_tokens:, :]
        x = self.unpatchify(x)
        x = self.conv_out(x)
        return x


# ---- default configs per image size ----

def default_config(image_size):
    """sensible defaults matching CliffordAR scale where possible."""
    if image_size == 256:
        # exact CliffordAR config: ~75M params
        return dict(
            cnn_chs=[64, 64, 128, 256, 512],
            z_channels=512,
            encoder_vit_layers=6,
            decoder_vit_layers=12,
            patch_size=16,
        )
    elif image_size == 64:
        # 3 CNN stages → 8×8 = 64 tokens
        return dict(
            cnn_chs=[64, 128, 256, 512],
            z_channels=512,
            encoder_vit_layers=4,
            decoder_vit_layers=8,
            patch_size=8,
        )
    elif image_size == 32:
        # 2 CNN stages → 8×8 = 64 tokens
        return dict(
            cnn_chs=[64, 256, 512],
            z_channels=512,
            encoder_vit_layers=4,
            decoder_vit_layers=8,
            patch_size=4,
        )
    else:
        # generic: compute number of stages to get ~8-16 spatial grid
        num_stages = max(1, int(math.log2(image_size)) - 3)
        chs = [64]
        c = 64
        for _ in range(num_stages):
            c = min(c * 2, 512)
            chs.append(c)
        return dict(
            cnn_chs=chs,
            z_channels=chs[-1],
            encoder_vit_layers=4,
            decoder_vit_layers=8,
            patch_size=image_size // (2 ** num_stages),
        )


# ---- main VAE class ----

class CliffordARVAE(nn.Module):
    """hybrid CNN+ViT VAE matching CliffordAR S-VAE architecture.

    supports clifford, powerspherical, and gaussian distributions.
    for clifford: decoder receives 2*latent_dim per token (bivector via IFFT).
    for powerspherical: decoder receives latent_dim per token, scaled by R=sqrt(d).
    for gaussian: decoder receives latent_dim per token.
    """

    def __init__(
        self,
        latent_dim=16,
        image_size=256,
        in_channels=3,
        distribution="clifford",
        device="cpu",
        recon_loss_type="l1",
        l1_weight=1.0,
        use_learnable_beta=False,
        l2_normalize=False,
        # architecture params (None = use defaults for image_size)
        cnn_chs=None,
        z_channels=None,
        encoder_vit_layers=None,
        decoder_vit_layers=None,
        patch_size=None,
        register_tokens=4,
        concentration_floor=0.03,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.distribution = distribution
        self.device = device
        self.recon_loss_type = recon_loss_type
        self.l1_weight = l1_weight
        self.use_learnable_beta = use_learnable_beta
        self.l2_normalize = l2_normalize
        self.concentration_floor = concentration_floor

        # fill in defaults
        cfg = default_config(image_size)
        if cnn_chs is None:
            cnn_chs = cfg["cnn_chs"]
        if z_channels is None:
            z_channels = cfg["z_channels"]
        if encoder_vit_layers is None:
            encoder_vit_layers = cfg["encoder_vit_layers"]
        if decoder_vit_layers is None:
            decoder_vit_layers = cfg["decoder_vit_layers"]
        if patch_size is None:
            patch_size = cfg["patch_size"]

        self.z_channels = z_channels
        self.patch_size = patch_size
        n_heads = z_channels // 64
        num_stages = len(cnn_chs) - 1
        self.grid_size = image_size // (2 ** num_stages)
        self.num_tokens = self.grid_size ** 2

        # encoder
        self.encoder_vit = ViTEncoder(
            n_layers=encoder_vit_layers,
            n_heads=n_heads,
            d_model=z_channels,
            cnn_chs=cnn_chs,
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            register_tokens=register_tokens,
        )

        # latent projection: z_channels → (latent_dim for mu) + (1 for kappa/logvar)
        if distribution == "gaussian":
            self.quant_proj = nn.Linear(z_channels, latent_dim * 2, bias=True)
        else:
            self.quant_proj = nn.Linear(z_channels, latent_dim + 1, bias=True)

        # decoder input projection: latent → z_channels
        dec_latent_dim = 2 * latent_dim if distribution == "clifford" else latent_dim
        self.post_quant_proj = nn.Linear(dec_latent_dim, z_channels, bias=False)

        # decoder
        self.decoder_vit = ViTDecoder(
            n_layers=decoder_vit_layers,
            n_heads=n_heads,
            d_model=z_channels,
            cnn_chs=cnn_chs[::-1],
            out_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            register_tokens=register_tokens,
        )

        if use_learnable_beta:
            self.log_sigma_0 = nn.Parameter(torch.zeros(1))
            self.log_sigma_1 = nn.Parameter(torch.zeros(1))

        self.to(device)

    def reparameterize(self, mu, params):
        """reparameterize per-token latents."""
        B, T, D = mu.shape

        if self.distribution == "gaussian":
            logvar = params
            q_z = torch.distributions.Normal(mu, torch.exp(0.5 * logvar) + 1e-6)
            p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))
            z = q_z.rsample()
            if self.l2_normalize:
                z = F.normalize(z, p=2, dim=-1)
            return z, q_z, p_z

        elif self.distribution == "powerspherical":
            kappa = params  # (B, T)
            mu_flat = mu.reshape(B * T, D)
            kappa_flat = kappa.reshape(B * T)
            q_z = PowerSpherical(mu_flat, kappa_flat)
            p_z = HypersphericalUniform(D, device=mu.device, validate_args=False)
            z = q_z.rsample().reshape(B, T, D)
            # scale by R = sqrt(d), matching CliffordAR
            z = z * (self.latent_dim ** 0.5)
            return z, q_z, p_z

        elif self.distribution == "clifford":
            kappa = params  # (B, T)
            mu_flat = mu.reshape(B * T, D)
            kappa_flat = kappa.reshape(B * T).unsqueeze(-1).expand_as(mu_flat)
            q_z = CliffordPowerSphericalDistribution(mu_flat, kappa_flat)
            p_z = CliffordTorusUniform(D, device=mu.device, validate_args=False)
            z = q_z.rsample().reshape(B, T, 2 * D)
            return z, q_z, p_z

    def encoder(self, x):
        """encoder → (mu, params) per token. compatible with training script interface."""
        h = self.encoder_vit(x)  # (B, T, z_channels)
        proj = self.quant_proj(h)  # (B, T, latent_dim+1 or latent_dim*2)

        if self.distribution == "gaussian":
            mu = proj[..., :self.latent_dim]
            logvar = proj[..., self.latent_dim:]
            return mu, logvar
        elif self.distribution == "powerspherical":
            mu = proj[..., :-1]
            kappa = proj[..., -1]
            mu = F.normalize(mu, p=2, dim=-1)
            kappa = torch.clamp(F.softplus(kappa) + 0.8, max=10.0)
            return mu, kappa
        elif self.distribution == "clifford":
            mu = proj[..., :-1]
            kappa = proj[..., -1]
            kappa = torch.clamp(F.softplus(kappa) + self.concentration_floor, max=10.0)
            return mu, kappa

    def decoder(self, z):
        """decode latent tokens to image.
        accepts z as (B, T, dec_dim) or flat (B, T*dec_dim) for backward compat.
        """
        dec_dim = 2 * self.latent_dim if self.distribution == "clifford" else self.latent_dim
        if z.dim() == 2:
            # flat input — reshape to (B, T, dec_dim)
            z = z.view(z.size(0), self.num_tokens, dec_dim)
        h = self.post_quant_proj(z)  # (B, T, z_channels)
        return self.decoder_vit(h)

    def forward(self, x):
        mu, params = self.encoder(x)
        z, q_z, p_z = self.reparameterize(mu, params)
        x_recon = self.decoder(z)
        return x_recon, q_z, p_z, mu

    def encode(self, x):
        """CliffordAR-compatible encode: returns (z, kl_loss)."""
        mu, params = self.encoder(x)
        z, q_z, p_z = self.reparameterize(mu, params)
        kl = torch.distributions.kl.kl_divergence(q_z, p_z)
        if self.distribution == "gaussian":
            kl_loss = kl.sum(dim=-1).mean()
        else:
            kl_loss = kl.mean()
        return z, kl_loss

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
        """flatten per-token z for VSA tests: (B, num_tokens * z_dim).
        for clifford, returns the bivector (2*latent_dim per token) so it's HRR-compatible.
        """
        mu, params = self.encoder(x)
        z, _, _ = self.reparameterize(mu, params)
        return z.reshape(z.size(0), -1)

    def normalize(self, x):
        """CliffordAR-compatible: L2 normalize and scale by R=sqrt(d)."""
        x = F.normalize(x, p=2, dim=-1)
        return x * (self.latent_dim ** 0.5)

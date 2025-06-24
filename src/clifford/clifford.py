import torch
import torch.nn as nn
from torch.distributions import VonMises, Distribution, constraints
from torch.distributions.utils import broadcast_all
import math
import numpy as np
from scipy.special import i0, i1
import torch.nn.functional as F
from torch.distributions.kl import register_kl
from power_spherical import PowerSpherical


class CliffordTorusDistribution(Distribution):
    """
    Distribution over points on a Clifford torus, from independent von Mises distributions.
    """

    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape[:-1]
        self.orig_dim = self.loc.shape[-1]
        event_shape = torch.Size([2 * self.orig_dim])
        super().__init__(batch_shape, event_shape, validate_args)
        self.device = loc.device
        self.dtype = loc.dtype

    def _von_mises_entropy(self, kappa):
        """
        Compute entropy of von Mises distribution
        H = log(2π I₀(κ)) - κ E[cos(x - μ)]
        where E[cos(x - μ)] = I₁(κ)/I₀(κ)
        """
        kappa_np = kappa.detach().cpu().numpy()

        # handle array-like input for Bessel functions
        if np.isscalar(kappa_np):
            i0_k = float(i0(kappa_np))
            i1_k = float(i1(kappa_np))
        else:
            i0_k = np.array([float(i0(k)) for k in kappa_np.flat]).reshape(
                kappa_np.shape
            )
            i1_k = np.array([float(i1(k)) for k in kappa_np.flat]).reshape(
                kappa_np.shape
            )

        # convert back to torch tensors with proper device placement
        i0_k = torch.tensor(i0_k, device=kappa.device, dtype=kappa.dtype)
        i1_k = torch.tensor(i1_k, device=kappa.device, dtype=kappa.dtype)

        # compute E[cos(x - μ)] = I₁(κ)/I₀(κ)
        expected_cos = i1_k / i0_k

        # compute entropy
        entropy = torch.log(2 * math.pi * i0_k) - kappa * expected_cos

        return entropy

    def rsample(self, sample_shape=torch.Size()):
        loc = self.loc
        concentration = self.concentration
        # sample from mu
        shape = self._extended_shape(sample_shape)
        theta_collection = torch.zeros(
            *shape[:-1],  # all but last dimension
            self.orig_dim,  # original dimension
            dtype=torch.float32,
            device=self.loc.device,
        )
        # theta_collection = torch.zeros(
        #     self.loc.shape,
        #     dtype=torch.float32,
        #     device=self.loc.device
        # )
        for i in range(self.orig_dim):
            von_mises = VonMises(loc=loc[..., i], concentration=concentration[..., i])
            theta = von_mises.sample(sample_shape)
            theta_collection[..., i] = theta

        n = self.orig_dim * 2
        theta_s = torch.zeros(
            (*theta_collection.shape[:-1], n), device=theta_collection.device
        )
        # ensure DC component (k=0), should be 0
        try:
            assert [theta_s[i, 0] == 0 for i in range(theta_s.shape[0])]
        except Exception as e:
            print(f"DC component is not 0: {e}")

        # print(theta_s)
        # fill positive frequencies (k > 0)
        theta_s[..., 1 : self.orig_dim] = theta_collection[..., 1:]
        # fill negative frequencies to ensure conjugate symmetry
        # For k from 1 to N/2-1: F(N-k) = F(k)*
        theta_s[..., -self.orig_dim + 1 :] = -torch.flip(
            theta_collection[..., 1:], dims=(-1,)
        )
        # print(theta_s[..., n//2])

        # convert to complex exponentials
        samples_complex = torch.exp(1j * theta_s)

        # orthonormal scales by 1/sqrt(N) instead of 1/n
        result = torch.fft.ifft(samples_complex, dim=-1, norm="ortho")
        # result = torch.fft.ifft(samples_complex, dim=-1)
        # print(f"result shape: {result.shape}")

        return result

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        freq_domain = torch.fft.fft(value, dim=-1, norm="ortho")
        first_half = freq_domain[..., : self.orig_dim]

        angles = torch.angle(first_half)
        log_prob = 0.0

        for i in range(self.orig_dim):
            von_mises = VonMises(
                loc=self.loc[..., i], concentration=self.concentration[..., i]
            )
            log_prob = log_prob + von_mises.log_prob(angles[..., i])

        return log_prob

    def entropy(self):
        """
        Compute entropy as sum of individual von Mises entropies.
        Since the distribution is a product of independent von Mises distributions,
        the total entropy is the sum of individual entropies.
        """
        total_entropy = 0.0
        for i in range(self.orig_dim):
            kappa = self.concentration[..., i]
            total_entropy = total_entropy + self._von_mises_entropy(kappa)
        return total_entropy


# --- helper: angle ~ PowerSpherical(S¹) --------------------------------------
def _powerspherical_angle(mu, scale, sample_shape):
    """
    mu, scale : [...]-shaped tensors (mu is angle in rad, scale>0)
    returns   : angle θ ~ PS(μ, scale) with shape sample_shape+mu.shape
    """
    # direction on S¹ that points at μ
    mean_dir = torch.stack((torch.cos(mu), torch.sin(mu)), -1)  # [...,2]
    ps = PowerSpherical(mean_dir, scale)  # sphere S¹
    v = ps.rsample(sample_shape)  # [...,2]
    return torch.atan2(v[..., 1], v[..., 0])  # angle


# --- variant that uses Power‑Spherical instead of von Mises ------------------
class CliffordPowerSphericalDistribution(CliffordTorusDistribution):
    """
    Faster sampling due to no VM bottleneck; each phase comes from a 1‑D Power‑Spherical.
    loc: [..., d]        (mean phase μ in rad)
    scale: [..., d]      (positive)
    """

    def rsample(self, sample_shape=torch.Size()):
        theta = _powerspherical_angle(
            self.loc, self.concentration, sample_shape
        )  # [...,d]
        # build conjugate‑symmetric θ_s exactly as the parent class did
        n = self.orig_dim * 2
        theta_s = torch.zeros((*theta.shape[:-1], n), device=theta.device)
        theta_s[..., 1 : self.orig_dim] = theta[..., 1:]
        theta_s[..., -self.orig_dim + 1 :] = -torch.flip(theta[..., 1:], (-1,))
        try:
            assert [theta_s[i, 0] == 0 for i in range(theta_s.shape[0])]
        except Exception as e:
            print(f"DC component is not 0: {e}")
        samples_c = torch.exp(1j * theta_s)
        return torch.fft.ifft(samples_c, dim=-1, norm="ortho")

    def log_prob(self, value):
        freq = torch.fft.fft(value, dim=-1, norm="ortho")[..., : self.orig_dim]
        angles = torch.atan2(freq.imag, freq.real)
        mean_dirs = torch.stack((torch.cos(self.loc), torch.sin(self.loc)), -1)
        vecs = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
        ps = PowerSpherical(mean_dirs, self.concentration)
        return ps.log_prob(vecs).sum(-1)  # sum over d

    def entropy(self):
        mean_dirs = torch.stack((torch.cos(self.loc), torch.sin(self.loc)), -1)
        ps = PowerSpherical(mean_dirs, self.concentration)
        return ps.entropy().sum(-1)


class CliffordTorusEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W_mu = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.W_mu(x)
        concentration = torch.exp(self.W_k(x))
        return CliffordTorusDistribution(mu, concentration)


class CliffordTorusUniform(torch.distributions.Distribution):
    """
    Uniform distribution on the Clifford torus.
    """

    arg_constraints = {
        "dim": torch.distributions.constraints.positive_integer,
    }
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=None):
        self.dim = (
            dim if isinstance(dim, torch.Tensor) else torch.tensor(dim, device=device)
        )
        super().__init__(validate_args=validate_args)
        self.device = device
        self.dtype = dtype

    def rsample(self, sample_shape=torch.Size()):
        # sample uniform angles
        shape = sample_shape + torch.Size([self.dim])
        angles = torch.rand(shape, device=self.device) * 2 * math.pi

        # convert to complex exponentials
        samples_complex = torch.zeros(
            sample_shape + torch.Size([2 * self.dim]),
            dtype=torch.complex64,
            device=self.device,
        )
        samples_complex[..., : self.dim] = torch.exp(1j * angles)
        # fill conjugate symmetric values
        samples_complex[..., self.dim :] = torch.conj(
            torch.flip(samples_complex[..., : self.dim], dims=(-1,))
        )

        # take inverse FFT
        result = torch.fft.ifft(samples_complex, dim=-1, norm="ortho")
        return result

    def log_prob(self, value):
        # uniform probability on the torus
        return -torch.ones_like(value[..., 0]) * self.log_normalizer()

    def log_normalizer(self):
        # log of surface area of torus = log((2π)^dim)
        return self.dim * math.log(2 * math.pi)

    def entropy(self):
        return self.log_normalizer()


@register_kl(CliffordPowerSphericalDistribution, CliffordTorusUniform)
def _kl_ps_uniform(p, q):
    return -p.entropy() + q.entropy()


@register_kl(CliffordTorusDistribution, CliffordTorusUniform)
def _kl_clifford_uniform(p, q):
    """
    Compute KL(p||q) between a Clifford torus distribution p and uniform distribution q.
    KL divergence is the sum of KL divergences for each von Mises component.
    """
    return -p.entropy() + q.entropy()


class ModelVAE(nn.Module):
    """Defined architecture as taken from experiments [] from
    (Davidson et al. ) https://arxiv.org/abs/1804.00891"""

    def __init__(self, h_dim: int, z_dim: int, device: str | torch.device = "cpu"):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(784, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_concentration = nn.Linear(h_dim, 1)

        # decoder network (note: Clifford sample is complex length 2*z_dim)
        # we convert to real (4*z_dim) then project back to z_dim real vector
        self.pre_fc = nn.Linear(4 * z_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, 784),
        )

        self.to(device)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x.view(-1, 784))
        mu = self.fc_mu(h)
        concentration = F.softplus(self.fc_concentration(h)) + 1
        return mu, concentration

    def reparameterize(self, mu: torch.Tensor, concentration: torch.Tensor):
        # q_z = CliffordTorusDistribution(mu, concentration.expand_as(mu))
        q_z = CliffordPowerSphericalDistribution(mu, concentration.expand_as(mu))
        p_z = CliffordTorusUniform(self.z_dim, device=mu.device)
        z_c = q_z.rsample()  # complex (B, 2*z_dim)
        z_real = torch.view_as_real(z_c).reshape(z_c.size(0), -1).float()
        z = self.pre_fc(z_real)
        return z, q_z, p_z

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        mu, kappa = self.encode(x)
        z, q_z, p_z = self.reparameterize(mu, kappa)
        x_recon = self.decoder(z)
        return (mu, kappa), (q_z, p_z), z, x_recon


def compute_loss(model: "ModelVAE", x: torch.Tensor, beta: float = 1.0):
    (mu, kappa), (q_z, p_z), _, x_recon = model(x)
    recon_loss = F.binary_cross_entropy_with_logits(
        x_recon, x.view(-1, 784), reduction="sum"
    ) / x.size(0)
    kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    return recon_loss + beta * kl

import math
import torch
from torch.distributions import (
    Distribution,
    constraints,
    TransformedDistribution,
    Beta,
    AffineTransform,
    Transform,
)
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl


def _get_eps(x: torch.Tensor) -> torch.Tensor:
    """Get small epsilon tensor matching device and dtype."""
    return torch.tensor(1e-7, device=x.device, dtype=x.dtype)


def _von_mises_entropy(kappa: torch.Tensor) -> torch.Tensor:
    """Compute entropy of von Mises distribution.
    H = log(2π I₀(κ)) - κ E[cos(x - μ)]
    where E[cos(x - μ)] = I₁(κ)/I₀(κ)
    """
    eps = _get_eps(kappa)
    log_i0 = torch.log(torch.special.i0e(kappa) + eps) + kappa
    log_i1 = torch.log(torch.special.i1e(kappa) + eps) + kappa
    ratio_i1_i0 = torch.exp(log_i1 - log_i0)
    return (
        torch.log(torch.tensor(2 * math.pi, device=kappa.device, dtype=kappa.dtype))
        + log_i0
        - kappa * ratio_i1_i0
    )


try:
    compiled_von_mises_entropy = torch.compile(_von_mises_entropy)
except Exception:
    compiled_von_mises_entropy = _von_mises_entropy


class _TTransform(Transform):
    domain = constraints.real
    codomain = constraints.real

    def _call(self, x):
        t = x[..., 0].unsqueeze(-1)
        v = x[..., 1:]
        eps = _get_eps(x)
        return torch.cat((t, v * torch.sqrt(torch.clamp(1 - t**2, min=eps))), -1)

    def _inverse(self, y):
        t = y[..., 0].unsqueeze(-1)
        v = y[..., 1:]
        eps = _get_eps(y)
        return torch.cat((t, v / torch.sqrt(torch.clamp(1 - t**2, min=eps))), -1)

    def log_abs_det_jacobian(self, x, y):
        t = x[..., 0]
        eps = _get_eps(x)
        return ((x.shape[-1] - 3) / 2) * torch.log(torch.clamp(1 - t**2, min=eps))


class _HouseholderRotationTransform(Transform):
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, loc):
        super().__init__()
        self.loc = loc
        self.e1 = torch.zeros_like(self.loc)
        self.e1[..., 0] = 1

    def _call(self, x):
        u = self.e1 - self.loc
        eps = _get_eps(u)
        u = u / (u.norm(dim=-1, keepdim=True) + eps)
        return x - 2 * (x * u).sum(-1, keepdim=True) * u

    def _inverse(self, y):
        return self._call(y)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros((), device=x.device, dtype=x.dtype)


class HypersphericalUniform(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=None):
        self.dim = dim
        self.device, self.dtype = device, dtype
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([dim]),
            validate_args=validate_args,
        )

    def rsample(self, sample_shape=torch.Size()):
        v = torch.randn(
            tuple(sample_shape) + tuple(self.event_shape),
            device=self.device,
            dtype=self.dtype,
        )
        eps = _get_eps(v)
        return v / (v.norm(dim=-1, keepdim=True) + eps)

    def log_prob(self, value):
        if self.dim <= 0:
            return torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)
        return torch.full_like(
            value[..., 0],
            math.lgamma(self.dim / 2)
            - (math.log(2) + (self.dim / 2) * math.log(math.pi)),
        )

    def entropy(self):
        if self.dim <= 0:
            return torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
        return -self.log_prob(torch.zeros(1, device=self.device, dtype=self.dtype))


class MarginalTDistribution(TransformedDistribution):
    has_rsample = True

    def __init__(self, dim, scale, validate_args=None):
        safe_scale = scale + _get_eps(scale)
        super().__init__(
            Beta(
                ((dim - 1) / 2) + safe_scale, (dim - 1) / 2, validate_args=validate_args
            ),
            AffineTransform(loc=-1, scale=2),
        )

    def entropy(self):
        return self.base_dist.entropy() + math.log(2)


class _JointTSDistribution(Distribution):
    has_rsample = True

    def __init__(self, marginal_t, marginal_s):
        super().__init__(
            batch_shape=marginal_t.batch_shape,
            event_shape=marginal_s.event_shape[:-1]
            + torch.Size([marginal_s.event_shape[-1] + 1]),
            validate_args=False,
        )
        self.marginal_t, self.marginal_s = marginal_t, marginal_s

    def rsample(self, sample_shape=torch.Size()):
        return torch.cat(
            (
                self.marginal_t.rsample(sample_shape).unsqueeze(-1),
                self.marginal_s.rsample(sample_shape + self.marginal_t.batch_shape),
            ),
            -1,
        )


class PowerSpherical(
    TransformedDistribution
):  # from Nicola De Cao https://github.com/nicola-decao/power_spherical
    # optimizations from https://evgeniia.tokarch.uk/blog/memory-optimization-for-kl-loss-calculation-in-pytorch/
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = loc, scale
        self.dim = loc.shape[-1]
        super().__init__(
            _JointTSDistribution(
                MarginalTDistribution(self.dim, scale, validate_args=validate_args),
                HypersphericalUniform(
                    self.dim - 1,
                    device=loc.device,
                    dtype=loc.dtype,
                    validate_args=validate_args,
                ),
            ),
            [_TTransform(), _HouseholderRotationTransform(loc)],
        )

    def log_normalizer(self):
        safe_scale = self.scale + _get_eps(self.scale)
        alpha = (self.dim - 1) / 2 + safe_scale
        beta = (self.dim - 1) / 2
        return -(
            (alpha + beta) * math.log(2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
            + beta * math.log(math.pi)
        )

    def log_prob(self, value):
        dot_product = torch.einsum("...d,...d->...", self.loc, value)
        eps = _get_eps(dot_product)
        safe_dot_product = torch.clamp(dot_product, min=-1.0 + eps, max=1.0 - eps)
        return self.log_normalizer() + self.scale * torch.log1p(safe_dot_product)

    def entropy(self):
        safe_scale = self.scale + _get_eps(self.scale)
        alpha = (self.dim - 1) / 2 + safe_scale
        beta = (self.dim - 1) / 2
        return -(
            self.log_normalizer()
            + safe_scale
            * (math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta))
        )


class CliffordTorusUniform(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=None):
        self.dim = dim
        self.device, self.dtype = device, dtype
        super().__init__(
            event_shape=torch.Size([2 * self.dim]), validate_args=validate_args
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + (self.dim,)
        angles = torch.rand(shape, device=self.device, dtype=self.dtype) * 2 * math.pi
        n = 2 * self.dim
        theta_s = torch.zeros(sample_shape + (n,), device=self.device, dtype=self.dtype)
        theta_s[..., 1 : self.dim] = angles[..., 1:]
        theta_s[..., -self.dim + 1 :] = -torch.flip(angles[..., 1:], dims=(-1,))
        samples_complex = torch.exp(1j * theta_s)
        return torch.fft.ifft(samples_complex, dim=-1).real

    def log_prob(self, value):
        return -torch.ones_like(value[..., 0]) * self.entropy()

    def entropy(self):
        return (self.dim - 1) * math.log(2 * math.pi)


class CliffordTorusDistribution(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        self.orig_dim = loc.shape[-1]
        super().__init__(
            batch_shape=loc.shape[:-1],
            event_shape=torch.Size([2 * self.orig_dim]),
            validate_args=validate_args,
        )
        self._von_mises = torch.distributions.VonMises(self.loc, self.concentration)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        theta_collection = self._von_mises.sample(sample_shape)
        n = 2 * self.orig_dim
        theta_s = torch.zeros(
            (*theta_collection.shape[:-1], n),
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        theta_s[..., 1 : self.orig_dim] = theta_collection[..., 1:]
        theta_s[..., -self.orig_dim + 1 :] = -torch.flip(
            theta_collection[..., 1:], dims=(-1,)
        )
        samples_complex = torch.exp(1j * theta_s)
        # check how close to real vectors
        assert torch.allclose(samples_complex, samples_complex.conj().flip(-1))
        return torch.fft.ifft(samples_complex, dim=-1).real

    def entropy(self):
        return compiled_von_mises_entropy(self.concentration)[..., 1:].sum(-1)


class CliffordPowerSphericalDistribution(CliffordTorusDistribution):
    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self, loc, concentration, validate_args=None, normalize_ifft: bool = True
    ):
        super().__init__(loc, concentration, validate_args=validate_args)
        self.normalize_ifft = normalize_ifft
        self.dtype = loc.dtype

    def rsample(self, sample_shape=torch.Size()):
        mean_dir = torch.stack((torch.cos(self.loc), torch.sin(self.loc)), -1)
        ps = PowerSpherical(mean_dir, self.concentration)
        v = ps.rsample(sample_shape)
        theta = torch.atan2(v[..., 1], v[..., 0])
        n = 2 * self.orig_dim
        theta_s = torch.zeros(
            (*theta.shape[:-1], n), device=theta.device, dtype=self.dtype
        )
        theta_s[..., 1 : self.orig_dim] = theta[..., 1:]
        theta_s[..., -self.orig_dim + 1 :] = -torch.flip(theta[..., 1:], (-1,))
        samples_c = torch.exp(1j * theta_s)
        if self.normalize_ifft:
            samples_c = samples_c / math.sqrt(n)
            return torch.fft.ifft(samples_c, dim=-1, norm="ortho").real
        return torch.fft.ifft(samples_c, dim=-1).real

    def log_prob(self, value):
        freq = torch.fft.fft(value, dim=-1, norm="ortho")[..., : self.orig_dim]
        angles = torch.angle(freq)
        mean_dirs = torch.stack((torch.cos(self.loc), torch.sin(self.loc)), -1)
        vecs = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
        ps = PowerSpherical(mean_dirs, self.concentration)
        return ps.log_prob(vecs).sum(-1)

    def entropy(self):
        mean_dirs = torch.stack((torch.cos(self.loc), torch.sin(self.loc)), -1)
        ps = PowerSpherical(mean_dirs, self.concentration)
        ent = ps.entropy()
        return ent[..., 1:].sum(-1)


@register_kl(CliffordPowerSphericalDistribution, CliffordTorusUniform)
def _kl_ps_uniform(p, q):
    return -p.entropy() + q.entropy()


@register_kl(CliffordTorusDistribution, CliffordTorusUniform)
def _kl_vm_uniform(p, q):
    return -p.entropy() + q.entropy()


@register_kl(PowerSpherical, HypersphericalUniform)
def _kl_powerspherical_uniform(p, q):
    return -p.entropy() + q.entropy()

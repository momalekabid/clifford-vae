# latent distributions: power spherical (de cao & aziz 2020) and the clifford torus built from it.
# vmf lives in clifford_vae/vmf (vendored, patched).
import math

import torch
from torch.distributions import AffineTransform, Beta, Distribution, Transform, TransformedDistribution, constraints
from torch.distributions.kl import register_kl


def _eps(x):
    return torch.tensor(1e-7, device=x.device, dtype=x.dtype)


def clifford_embed(angles: torch.Tensor) -> torch.Tensor:
    """d angles -> the real length-2d vector F^-1{h}, h hermitian with h_k = e^{i angle_k}.
    bin 0 and the nyquist bin get angle 0 (h = 1), so every |F{x}_k| = 1 and ||x|| = 1.
    angles[..., 0] is unused."""
    d = angles.shape[-1]
    th = torch.zeros((*angles.shape[:-1], 2 * d), device=angles.device, dtype=angles.dtype)
    th[..., 1:d] = angles[..., 1:]
    th[..., -d + 1:] = -torch.flip(angles[..., 1:], (-1,))
    return torch.fft.ifft(torch.exp(1j * th), dim=-1).real


class _TTransform(Transform):
    domain = constraints.real
    codomain = constraints.real

    def _call(self, x):
        t, v = x[..., :1], x[..., 1:]
        return torch.cat((t, v * torch.sqrt(torch.clamp(1 - t**2, min=_eps(x)))), -1)

    def _inverse(self, y):
        t, v = y[..., :1], y[..., 1:]
        return torch.cat((t, v / torch.sqrt(torch.clamp(1 - t**2, min=_eps(y)))), -1)

    def log_abs_det_jacobian(self, x, y):
        return ((x.shape[-1] - 3) / 2) * torch.log(torch.clamp(1 - x[..., 0] ** 2, min=_eps(x)))


class _Householder(Transform):
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, loc):
        super().__init__()
        self.loc = loc
        self.e1 = torch.zeros_like(loc)
        self.e1[..., 0] = 1

    def _call(self, x):
        u = self.e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + _eps(u))
        return x - 2 * (x * u).sum(-1, keepdim=True) * u

    def _inverse(self, y):
        return self._call(y)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros((), device=x.device, dtype=x.dtype)


class HypersphericalUniform(Distribution):
    """uniform on S^(dim-1) in R^dim."""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=False):
        self.dim, self.device, self.dtype = dim, device, dtype
        super().__init__(event_shape=torch.Size([dim]), validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        v = torch.randn(tuple(sample_shape) + (self.dim,), device=self.device, dtype=self.dtype)
        return v / (v.norm(dim=-1, keepdim=True) + _eps(v))

    def entropy(self):
        return torch.tensor(math.log(2) + (self.dim / 2) * math.log(math.pi) - math.lgamma(self.dim / 2))


class _MarginalT(TransformedDistribution):
    has_rsample = True

    def __init__(self, dim, scale):
        super().__init__(Beta((dim - 1) / 2 + scale + _eps(scale), (dim - 1) / 2), AffineTransform(loc=-1, scale=2))


class _JointTS(Distribution):
    has_rsample = True

    def __init__(self, t, s):
        super().__init__(batch_shape=t.batch_shape, event_shape=s.event_shape[:-1] + torch.Size([s.event_shape[-1] + 1]),
                         validate_args=False)
        self.t, self.s = t, s

    def rsample(self, sample_shape=torch.Size()):
        return torch.cat((self.t.rsample(sample_shape).unsqueeze(-1), self.s.rsample(sample_shape + self.t.batch_shape)), -1)


class PowerSpherical(TransformedDistribution):
    """power spherical on S^(d-1) (de cao & aziz 2020)."""

    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale, self.dim = loc, scale, loc.shape[-1]
        super().__init__(_JointTS(_MarginalT(self.dim, scale), HypersphericalUniform(self.dim - 1, loc.device, loc.dtype)),
                         [_TTransform(), _Householder(loc)])

    def log_normalizer(self):
        a = (self.dim - 1) / 2 + self.scale + _eps(self.scale)
        b = (self.dim - 1) / 2
        return -((a + b) * math.log(2) + torch.lgamma(a) - torch.lgamma(a + b) + b * math.log(math.pi))

    def log_prob(self, value):
        dot = torch.einsum("...d,...d->...", self.loc, value).clamp(-1 + 1e-7, 1 - 1e-7)
        return self.log_normalizer() + self.scale * torch.log1p(dot)

    def entropy(self):
        s = self.scale + _eps(self.scale)
        a, b = (self.dim - 1) / 2 + s, (self.dim - 1) / 2
        return -(self.log_normalizer() + s * (math.log(2) + torch.digamma(a) - torch.digamma(a + b)))


class CliffordTorusUniform(Distribution):
    """uniform on the clifford torus (S^1)^(d-1), as length-2d real vectors."""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=False):
        self.dim, self.device, self.dtype = dim, device, dtype
        super().__init__(event_shape=torch.Size([2 * dim]), validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        a = torch.rand(tuple(sample_shape) + (self.dim,), device=self.device, dtype=self.dtype) * 2 * math.pi
        return clifford_embed(a)

    def entropy(self):
        return (self.dim - 1) * math.log(2 * math.pi)


class CliffordPowerSpherical(Distribution):
    """product of power spherical distributions on S^1, one per angle, mapped through
    clifford_embed. concentration is (..., 1) (one kappa per image) or (..., d)."""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, loc, concentration):
        self.loc = loc
        self.concentration = concentration.expand_as(loc)
        super().__init__(batch_shape=loc.shape[:-1], event_shape=torch.Size([2 * loc.shape[-1]]), validate_args=False)

    def _circles(self):
        return PowerSpherical(torch.stack((torch.cos(self.loc), torch.sin(self.loc)), -1), self.concentration)

    def rsample(self, sample_shape=torch.Size()):
        e1 = torch.zeros((*self.loc.shape, 2), device=self.loc.device, dtype=self.loc.dtype)
        e1[..., 0] = 1
        v = PowerSpherical(e1, self.concentration).rsample(sample_shape)
        return clifford_embed(self.loc + torch.atan2(v[..., 1], v[..., 0]))

    @property
    def mean(self):
        return clifford_embed(self.loc)

    def entropy(self):
        # angle 0 is unused, so its circle does not count
        return self._circles().entropy()[..., 1:].sum(-1)


@register_kl(CliffordPowerSpherical, CliffordTorusUniform)
def _kl_cps_uniform(p, q):
    return -p.entropy() + q.entropy()


@register_kl(PowerSpherical, HypersphericalUniform)
def _kl_ps_uniform(p, q):
    return -p.entropy() + q.entropy().to(p.loc.device)

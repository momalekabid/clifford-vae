# checks for the clifford torus distribution after the revision changes.
# run: python -m pytest tests/ -q   (or python tests/test_clifford.py)
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dists.clifford import (
    CliffordPowerSphericalDistribution,
    CliffordTorusUniform,
    PowerSpherical,
    clifford_embed,
)
from cnn.models import VAE
from mnist.mlp_vae import MLPVAE, vae_loss


def test_embed_is_real_unitary_unit_norm():
    torch.manual_seed(0)
    for d in [3, 8, 64, 513]:
        x = clifford_embed(torch.rand(5, d) * 2 * math.pi)
        assert x.shape == (5, 2 * d)
        f = torch.fft.fft(x.double(), dim=-1)
        assert torch.allclose(f.abs(), torch.ones_like(f.abs()), atol=1e-5)
        assert torch.allclose(x.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_angle_zero_is_unused():
    a = torch.rand(2, 16) * 6
    b = a.clone()
    b[:, 0] += 1.234
    assert torch.allclose(clifford_embed(a), clifford_embed(b))


def test_per_circle_kappa_and_kl():
    torch.manual_seed(0)
    d = 12
    loc = torch.randn(4, d)
    kappa = torch.rand(4, d) * 20 + 0.1
    q = CliffordPowerSphericalDistribution(loc, kappa)
    p = CliffordTorusUniform(d)
    kl = torch.distributions.kl.kl_divergence(q, p)
    assert kl.shape == (4,)
    # kl = sum over circles 1..d-1 of the per-circle KL vs uniform on S^1
    e = torch.stack((torch.cos(loc), torch.sin(loc)), -1)
    per = -PowerSpherical(e, kappa).entropy() + math.log(2 * math.pi)
    assert torch.allclose(kl, per[:, 1:].sum(-1), atol=1e-4)
    assert (kl >= -1e-5).all()
    # changing kappa of circle 3 only changes the kl through that circle
    k2 = kappa.clone()
    k2[:, 3] *= 2
    kl2 = torch.distributions.kl.kl_divergence(CliffordPowerSphericalDistribution(loc, k2), p)
    per2 = -PowerSpherical(e, k2).entropy() + math.log(2 * math.pi)
    assert torch.allclose(kl2 - kl, per2[:, 3] - per[:, 3], atol=1e-4)


def test_samples_concentrate_on_mean_code():
    torch.manual_seed(0)
    d = 32
    loc = torch.randn(1, d).expand(2000, d)
    for k, lo in [(1.0, 0.0), (1000.0, 0.95)]:
        q = CliffordPowerSphericalDistribution(loc, torch.full((2000, d), k))
        z = q.rsample()
        f = torch.fft.fft(z, dim=-1)
        assert torch.allclose(f.abs(), torch.ones_like(f.abs()), atol=1e-4)
        cos = (z * q.mean).sum(-1).mean().item()
        assert cos > lo
    # low kappa: samples spread out, high kappa: close to the mean code
    q_lo = CliffordPowerSphericalDistribution(loc, torch.full((2000, d), 0.05))
    assert (q_lo.rsample() * q_lo.mean).sum(-1).mean().item() < 0.2


def test_uniform_prior_is_unitary():
    z = CliffordTorusUniform(20).rsample((100,))
    f = torch.fft.fft(z, dim=-1)
    assert torch.allclose(f.abs(), torch.ones_like(f.abs()), atol=1e-5)


def test_cnn_vae_clifford_per_circle_gradients():
    torch.manual_seed(0)
    m = VAE(16, 3, "clifford", "cpu")
    x = torch.randn(4, 3, 32, 32)
    xr, q, p, mu = m(x)
    assert q.concentration.shape == (4, 16)
    assert m.decoder.fc.in_features == 32
    loss = m.compute_loss(x, xr, q, p, 1.0)["total_loss"]
    loss.backward()
    g = m.encoder.fc_concentration.weight.grad
    assert g.shape == (16, 2048)
    # every circle except the unused index 0 gets a gradient
    assert (g[1:].abs().sum(-1) > 0).all()
    assert g[0].abs().sum() == 0
    c = m.mean_code(x)
    assert c.shape == (4, 32)
    assert torch.allclose(c.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_mlp_vae_kl_summed():
    torch.manual_seed(0)
    m = MLPVAE(0, 10, "normal")
    x = torch.rand(8, 784).round()
    z_mean, logv = m.encode(x)
    q = torch.distributions.Normal(z_mean, torch.exp(0.5 * logv) + 1e-6)
    p = torch.distributions.Normal(torch.zeros_like(z_mean), torch.ones_like(z_mean))
    want = torch.distributions.kl.kl_divergence(q, p).sum(-1).mean()
    got = vae_loss(m, x, return_dict=True)["kl"]
    assert torch.allclose(got, want, atol=1e-5)
    mc = MLPVAE(0, 10, "clifford")
    assert mc.fc_scale.out_features == 10
    assert mc.mean_code(x).shape == (8, 20)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

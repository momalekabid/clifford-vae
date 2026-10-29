import torch
import numpy as np
import scipy.special
from numbers import Number


class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):
        assert isinstance(v, Number), "v must be a scalar"

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        #         else:
        #             print(v, type(v), np.isclose(v, 0))
        #             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        # keep z's dtype (the old torch.Tensor(...) cast silently dropped float64 to float32)
        return torch.from_numpy(np.asarray(output)).to(device=z.device, dtype=z.dtype)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (
            None,
            grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z),
        )


class Ive(torch.nn.Module):
    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


ive = IveFunction.apply


def _log_ive_debye(v, z):
    # uniform asymptotic (debye) expansion of log(I_v(z) e^{-z}), leading term.
    # accurate when v is large, which is exactly where scipy's ive underflows to 0.
    t = z / v
    s = np.sqrt(1.0 + t * t)
    return v * (s + np.log(t / (1.0 + s))) - 0.5 * np.log(2 * np.pi * v) - 0.5 * np.log(s) - z


def _log_ive_np(v, z):
    z = np.asarray(z, dtype=np.float64)
    out = np.log(scipy.special.ive(v, z))
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = _log_ive_debye(v, z[bad])
    return out


class LogIveFunction(torch.autograd.Function):
    """log(I_v(z) e^{-z}) in float64, no epsilon. replaces log(ive(...) + 1e-20), which
    swamped I_v for large v and gave negative KLs (d=40) and divergence (d=128)."""

    @staticmethod
    def forward(ctx, v, z):
        assert isinstance(v, Number), "v must be a scalar"
        ctx.save_for_backward(z)
        ctx.v = v
        out = _log_ive_np(v, z.detach().cpu().numpy())
        return torch.from_numpy(out).to(device=z.device, dtype=z.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[-1]
        v = ctx.v
        zn = z.detach().cpu().numpy().astype(np.float64)
        # d/dz log(I_v e^{-z}) = I_{v-1}/I_v - v/z - 1
        ratio = np.exp(_log_ive_np(v - 1, zn) - _log_ive_np(v, zn))
        g = torch.from_numpy(ratio - v / zn - 1.0).to(device=z.device, dtype=z.dtype)
        return None, grad_output * g


log_ive = LogIveFunction.apply


##########
# The below provided approximations were provided in the
# respective source papers, to improve the stability of
# the Bessel fractions.
# I_(v/2)(k) / I_(v/2 - 1)(k)


# source: https://arxiv.org/pdf/1606.02008.pdf
def ive_fraction_approx(v, z):
    # I_(v/2)(k) / I_(v/2 - 1)(k) >= z / (v-1 + ((v+1)^2 + z^2)^0.5
    return z / (v - 1 + torch.pow(torch.pow(v + 1, 2) + torch.pow(z, 2), 0.5))


# source: https://arxiv.org/pdf/1902.02603.pdf
def ive_fraction_approx2(v, z, eps=1e-20):
    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(z, 2)).clamp(eps))
        )

    delta_0 = delta_a(0.0)
    delta_2 = delta_a(2.0)
    B_0 = z / (
        delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(z, 2))).clamp(eps)
    )
    B_2 = z / (
        delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(z, 2))).clamp(eps)
    )

    return (B_0 + B_2) / 2.0

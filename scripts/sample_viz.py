"""
interactive 3D visualization of samples from clifford torus, powerspherical,
and gaussian distributions on the unit sphere.
projects nD samples onto first 3 coordinates with wireframe sphere.
sliders for concentration (kappa) and latent dimension.

usage:
  python sample_viz.py              # interactive mode with sliders
  python sample_viz.py --save       # save static high-dpi figure to figures/
  python sample_viz.py --save --dim 8 --kappa 10
"""

import sys
import argparse
sys.path.insert(0, "..")

import torch
import numpy as np
import matplotlib.pyplot as plt
from dists.clifford import (
    CliffordPowerSphericalDistribution,
    PowerSpherical,
)

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="save static figure instead of showing interactive")
parser.add_argument("--dim", type=int, default=4, help="latent dimension for static figure")
parser.add_argument("--kappa", type=float, default=5.0, help="concentration for static figure")
parser.add_argument("--out", type=str, default="figures/distribution_samples_3d.png")
args = parser.parse_args()

N_SAMPLES = 3000

COLORS = {
    "gaussian (raw)": "#e74c3c",
    "gaussian (L2)": "#e67e22",
    "powerspherical": "#2ecc71",
    "clifford (vM)": "#3498db",
    "clifford (PS)": "#9b59b6",
}


def sample_gaussian(kappa, dim, n=N_SAMPLES):
    std = 1.0 / (1.0 + kappa)
    return torch.randn(n, dim) * std


def sample_powerspherical(kappa, dim, n=N_SAMPLES):
    loc = torch.zeros(dim)
    loc[0] = 1.0
    ps = PowerSpherical(loc, torch.tensor(kappa))
    return ps.rsample(sample_shape=torch.Size([n]))


def sample_clifford_vm(kappa, dim, n=N_SAMPLES):
    loc = torch.zeros(dim)
    concentration = torch.full((dim,), kappa)
    vm = torch.distributions.VonMises(loc, concentration)
    theta = vm.sample(torch.Size([n]))
    nd = 2 * dim
    theta_s = torch.zeros(n, nd)
    theta_s[:, 1:dim] = theta[:, 1:]
    theta_s[:, -dim + 1:] = -torch.flip(theta[:, 1:], dims=(-1,))
    return torch.fft.ifft(torch.exp(1j * theta_s), dim=-1).real


def sample_clifford_ps(kappa, dim, n=N_SAMPLES):
    loc = torch.zeros(dim)
    concentration = torch.full((dim,), kappa)
    dist = CliffordPowerSphericalDistribution(loc, concentration)
    return dist.rsample(sample_shape=torch.Size([n]))


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-7)


def to_3d(x):
    """take first 3 coords, pad if needed."""
    x = x.detach()
    d = x.shape[-1]
    if d < 3:
        x = torch.nn.functional.pad(x, (0, 3 - d))
    return x[:, :3].numpy()


def draw_wireframe(ax, alpha=0.08):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=alpha, linewidth=0.4)


def setup_ax(ax, lim=1.1):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("z", fontsize=8)


# columns: gaussian raw, gaussian L2, powerspherical, clifford vM, clifford PS
COLUMNS = [
    ("gaussian (raw)", sample_gaussian, False),
    ("gaussian (L2)", sample_gaussian, True),
    ("powerspherical", sample_powerspherical, False),
    ("clifford (vM)", sample_clifford_vm, False),
    ("clifford (PS)", sample_clifford_ps, False),
]

if args.save:
    import os
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    fig = plt.figure(figsize=(22, 5))
    fig.subplots_adjust(wspace=0.02, top=0.88, left=0.02, right=0.98, bottom=0.08)
    axes = [fig.add_subplot(1, 5, i + 1, projection="3d") for i in range(5)]

    kappa, dim = args.kappa, int(args.dim)
    for i, (name, func, do_norm) in enumerate(COLUMNS):
        samples = func(kappa, dim)
        pts = to_3d(normalize(samples) if do_norm else samples)
        ax = axes[i]
        if name != "gaussian (raw)":
            draw_wireframe(ax)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=0.5, alpha=0.3, c=COLORS[name], depthshade=True)
        ax.set_title(name, fontsize=11, pad=4)
        if name == "gaussian (raw)":
            lim = max(np.abs(pts).max() + 0.2, 1.2)
            setup_ax(ax, lim=lim)
        else:
            setup_ax(ax)

    fig.suptitle(f"samples projected to 3D  (dim={dim}, κ={kappa:.1f})", fontsize=14)
    plt.savefig(args.out, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"saved to {args.out}")

else:
    # interactive mode with sliders
    fig = plt.figure(figsize=(22, 5))
    fig.subplots_adjust(bottom=0.22, wspace=0.02, top=0.88, left=0.02, right=0.98)
    axes = [fig.add_subplot(1, 5, i + 1, projection="3d") for i in range(5)]

    def redraw(kappa, dim):
        dim = int(dim)
        for i, (name, func, do_norm) in enumerate(COLUMNS):
            samples = func(kappa, dim)
            pts = to_3d(normalize(samples) if do_norm else samples)
            ax = axes[i]
            ax.cla()
            if name != "gaussian (raw)":
                draw_wireframe(ax)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=0.5, alpha=0.3, c=COLORS[name], depthshade=True)
            ax.set_title(name, fontsize=10, pad=2)
            if name == "gaussian (raw)":
                lim = max(np.abs(pts).max() + 0.2, 1.2)
                setup_ax(ax, lim=lim)
            else:
                setup_ax(ax)
        fig.suptitle(f"samples projected to 3D  (dim={dim}, κ={kappa:.1f})", fontsize=13)
        fig.canvas.draw_idle()

    from matplotlib.widgets import Slider
    ax_kappa = fig.add_axes([0.15, 0.09, 0.7, 0.03])
    slider_kappa = Slider(ax_kappa, "κ", 0.1, 100.0, valinit=5.0, valstep=0.5)
    slider_kappa.on_changed(lambda _: redraw(slider_kappa.val, slider_dim.val))

    ax_dim = fig.add_axes([0.15, 0.03, 0.7, 0.03])
    slider_dim = Slider(ax_dim, "dim", 2, 32, valinit=4, valstep=1)
    slider_dim.on_changed(lambda _: redraw(slider_kappa.val, slider_dim.val))

    redraw(5.0, 4)
    plt.show()

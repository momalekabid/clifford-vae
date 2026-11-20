"""reviewer 2's overview figure: the clifford-vae pipeline drawn with its data. an image is encoded to
one angle per circle of the torus, the angles are embedded as a real unitary vector, and two hrr
operations (bind, unbind) are applied to the trained codes with the code of a second image before
decoding. the right panel shows what binding does on one circle. the two images and the reconstruction
come from one clifford run's raw/recon.npz (any dataset); the angles and vectors drawn are illustrative.

    python overview_figure.py runs/cifar10/C-d128/seed1 --out overview.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

INK, GREY, LINE = "#2B2B2B", "#6B6B6B", "#6C8EBF"  # text, captions, data-flow arrows
RED = "#C0505A"  # the hrr operations and the second code
ENC, ENC_E = "#DDEBD9", "#7FA67A"
DEC, DEC_E = "#E4DEF2", "#8E7CC3"
CA, CB, CAB = "#2196F3", "#8E7CC3", "#C0505A"  # angle colours: first code, second code, bound
CELLS = LinearSegmentedColormap.from_list("cells", ["#8E7CC3", "#F7F7FA", "#2196F3"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "cm", "font.size": 15})


def unitary(theta):
    """real vector whose fourier coefficients are exp(i theta), conjugate-symmetric."""
    m = len(theta)
    f = np.ones(2 * m + 2, dtype=complex)
    f[1:m + 1] = np.exp(1j * theta)
    f[m + 2:] = np.conj(f[1:m + 1][::-1])
    return np.fft.ifft(f).real


def image(ax, img, x, y, s=1.15):
    img = img.transpose(1, 2, 0).squeeze()
    ax.imshow(img, extent=(x - s / 2, x + s / 2, y - s / 2, y + s / 2), cmap="gray" if img.ndim == 2 else None, zorder=3,
              interpolation="nearest")
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fill=False, ec=INK, lw=1.0, zorder=4))


def arrow(ax, a, b, color=LINE, style="arc3,rad=0", lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, connectionstyle=style,
                                 shrinkA=0, shrinkB=0, zorder=2))


def network(ax, x0, x1, y, h0, h1, fc, ec, label):
    """a trapezoid: tall on the image side, short on the code side."""
    ax.add_patch(Polygon([(x0, y - h0 / 2), (x0, y + h0 / 2), (x1, y + h1 / 2), (x1, y - h1 / 2)], closed=True, fc=fc, ec=ec,
                         lw=1.6, joinstyle="round", zorder=3))
    ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=14.5, color=INK, zorder=4)


def angles(ax, x, y, theta, color, r=0.19, gap=0.47):
    """one small circle per latent angle, with the angle marked on it."""
    n = len(theta)
    xs = x + (np.arange(n) - (n - 1) / 2) * gap
    for xi, t in zip(xs, theta):
        ax.add_patch(Circle((xi, y), r, fc="white", ec=GREY, lw=1.1, zorder=3))
        ax.plot([xi, xi + r * np.cos(t)], [y, y + r * np.sin(t)], color=color, lw=1.4, zorder=4, solid_capstyle="round")
        ax.add_patch(Circle((xi + r * np.cos(t), y + r * np.sin(t)), 0.055, fc=color, ec="none", zorder=5))
    return xs[0] - r, xs[-1] + r


def strip(ax, x, y, v, w=0.3, h=1.5):
    """a vector drawn as a column of cells coloured by value."""
    n, s = len(v), np.abs(v).max()
    ch = h / n
    for i, vi in enumerate(v):
        ax.add_patch(Rectangle((x - w / 2, y + h / 2 - (i + 1) * ch), w, ch, fc=CELLS(0.5 + 0.5 * vi / s), ec="white", lw=0.5,
                               zorder=3))
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, ec=INK, lw=1.0, zorder=4))


def op(ax, x, y, label, r=0.27):
    ax.add_patch(Circle((x, y), r, fc="#FBEFEF", ec=RED, lw=1.6, zorder=4))
    ax.text(x, y - 0.012, r"$\circledast$", ha="center", va="center", fontsize=19.5, color=RED, zorder=5)
    ax.text(x, y + r + 0.12, label, ha="center", va="bottom", fontsize=15.0, color=RED)


def bracket(ax, x0, x1, y, label, color=GREY):
    ax.plot([x0, x0, x1, x1], [y - 0.1, y, y, y - 0.1], color=color, lw=1.2, solid_capstyle="round", zorder=2)
    ax.text((x0 + x1) / 2, y + 0.1, label, ha="center", va="bottom", fontsize=15.5, color=color, fontweight="bold")


def dial(ax, x, y, t, color, label, r=0.44):
    """one circle of the torus with an angle marked from the positive axis."""
    ax.add_patch(Circle((x, y), r, fc="white", ec=GREY, lw=1.2, zorder=3))
    ax.plot([x, x + r], [y, y], color="#C9C9C9", lw=1.0, zorder=3)
    arc = np.linspace(0, t, 40)
    ax.plot(x + 0.45 * r * np.cos(arc), y + 0.45 * r * np.sin(arc), color=color, lw=1.2, zorder=4)
    ax.plot([x, x + r * np.cos(t)], [y, y + r * np.sin(t)], color=color, lw=1.8, zorder=4, solid_capstyle="round")
    ax.add_patch(Circle((x + r * np.cos(t), y + r * np.sin(t)), 0.075, fc=color, ec="none", zorder=5))
    ax.text(x, y - r - 0.16, label, ha="center", va="top", fontsize=16.0, color=INK)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--out", default="overview.pdf")
    p.add_argument("--index", type=int, nargs=2, default=[3, 6], help="which two of the 10 saved test images")
    a = p.parse_args()
    z = np.load(os.path.join(a.run_dir, "raw", "recon.npz"))
    xa, xb, xr = z["x"][a.index[0]], z["x"][a.index[1]], z["mean"][a.index[0]]

    # illustrative angles: five circles per code; binding adds them
    rng = np.random.default_rng(4)
    ta, tb = rng.uniform(-np.pi, np.pi, 5), rng.uniform(-np.pi, np.pi, 5)
    za, zb, zab = unitary(ta), unitary(tb), unitary(ta + tb)

    y, yb = 3.35, 1.05
    fig, ax = plt.subplots(figsize=(13.9, 4.75))
    ax.set(xlim=(0.1, 15.5), ylim=(-0.1, 5.15), aspect="equal")
    ax.axis("off")

    # first image: encoder -> angles -> unitary vector
    image(ax, xa, 0.8, y)
    ax.text(0.8, y + 0.68, r"$\mathbf{x}_a$", ha="center", va="bottom", fontsize=17.5)
    arrow(ax, (1.45, y), (1.95, y))
    network(ax, 2.0, 3.15, y, 1.5, 0.8, ENC, ENC_E, "encoder")
    arrow(ax, (3.2, y), (3.62, y))
    l, r = angles(ax, 4.75, y, ta, CA)
    ax.text(4.75, y - 0.36, r"phases $\theta^a \in (\mathcal{S}^1)^{d-1}$", ha="center", va="top", fontsize=15.0, color=INK)
    arrow(ax, (r + 0.07, y), (6.95, y))
    ax.text((r + 0.07 + 6.95) / 2 - 0.08, y + 0.2, r"$\mathcal{F}^{-1}\{e^{i\theta}\}$", ha="center", va="bottom", fontsize=14.5,
            color=INK)
    strip(ax, 7.15, y, za)
    ax.text(7.15, y - 0.87, r"$\mathbf{z}_a$", ha="center", va="top", fontsize=16.5)

    # hrr operations on the trained codes
    arrow(ax, (7.35, y), (8.0, y))
    op(ax, 8.3, y, "bind")
    arrow(ax, (8.6, y), (9.2, y))
    strip(ax, 9.4, y, zab)
    ax.text(9.4, y - 0.87, r"$\mathbf{z}_a \circledast \mathbf{z}_b$", ha="center", va="top", fontsize=16.5)
    arrow(ax, (9.6, y), (10.2, y))
    op(ax, 10.5, y, "unbind")
    arrow(ax, (10.8, y), (11.4, y))
    strip(ax, 11.6, y, za)
    ax.text(11.6, y - 0.87, r"$\mathbf{z}_a$", ha="center", va="top", fontsize=16.5)

    # decoder
    arrow(ax, (11.8, y), (12.3, y))
    network(ax, 12.35, 13.5, y, 0.8, 1.5, DEC, DEC_E, "decoder")
    arrow(ax, (13.55, y), (14.05, y))
    image(ax, xr, 14.7, y)
    ax.text(14.7, y + 0.68, r"$\hat{\mathbf{x}}_a$", ha="center", va="bottom", fontsize=17.5)

    # second image through the same encoder; its code feeds both operations
    image(ax, xb, 0.8, yb)
    ax.text(0.8, yb + 0.68, r"$\mathbf{x}_b$", ha="center", va="bottom", fontsize=17.5)
    arrow(ax, (1.45, yb), (1.95, yb))
    network(ax, 2.0, 3.15, yb, 1.5, 0.8, ENC, ENC_E, "encoder")
    ax.plot([2.575, 2.575], [yb + 0.6, y - 0.6], color=ENC_E, lw=1.2, ls=(0, (3, 3)), zorder=1)
    ax.text(2.66, (y + yb) / 2, "shared\nweights", ha="left", va="center", fontsize=13.5, color=GREY, linespacing=1.1)
    arrow(ax, (3.2, yb), (3.62, yb))
    l, r = angles(ax, 4.75, yb, tb, CB)
    ax.text(4.75, yb - 0.36, r"phases $\theta^b$", ha="center", va="top", fontsize=15.0, color=INK)
    arrow(ax, (r + 0.07, yb), (6.95, yb))
    strip(ax, 7.15, yb, zb)
    ax.text(7.15, yb - 0.87, r"$\mathbf{z}_b$", ha="center", va="top", fontsize=16.5)
    arrow(ax, (7.35, yb + 0.15), (8.3, y - 0.3), color=RED, style="angle3,angleA=0,angleB=90")
    arrow(ax, (7.35, yb - 0.15), (10.5, y - 0.3), color=RED, style="angle3,angleA=0,angleB=90")
    ax.text(8.42, 2.0, r"$\mathbf{z}_b$", ha="left", va="center", fontsize=16.0, color=RED)
    ax.text(10.62, 2.0, r"$\mathbf{z}_b^{*}$", ha="left", va="center", fontsize=16.0, color=RED)

    # groups
    bracket(ax, 2.0, 7.3, 4.55, "Encoder (trained)")
    bracket(ax, 7.95, 10.85, 4.55, "HRR operations (after training)", RED)
    bracket(ax, 12.35, 13.5, 4.55, "Decoder (trained)")

    # inset, bottom right: what binding does on one circle
    ax.add_patch(Rectangle((11.95, -0.02), 3.45, 2.1, fill=False, ec=GREY, lw=1.1, ls=(0, (4, 3)), zorder=1))
    cx, cy, t1, t2 = 12.55, 1.02, np.deg2rad(55), np.deg2rad(80)
    ax.text(13.675, 1.86, "binding adds phases", ha="center", va="center", fontsize=14, color=GREY)
    dial(ax, cx, cy, t1, CA, r"$\theta^a_i$", r=0.4)
    ax.text(cx + 0.5625, cy, "+", ha="center", va="center", fontsize=18, color=INK)
    dial(ax, cx + 1.125, cy, t2, CB, r"$\theta^b_i$", r=0.4)
    ax.text(cx + 1.6875, cy, "=", ha="center", va="center", fontsize=18, color=INK)
    dial(ax, cx + 2.25, cy, t1 + t2, CAB, r"$\theta^a_i{+}\theta^b_i$", r=0.4)

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()

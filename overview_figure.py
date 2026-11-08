"""reviewer 2's overview figure of the clifford-vae pipeline. uses one clifford run's raw/recon.npz
for the example image and its reconstruction (any dataset).

    python overview_figure.py runs/cifar10/C-d128/seed1 --out overview.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402

BLUE, GREY, DARK = "#2196F3", "#9E9E9E", "#333333"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "font.size": 11})


def box(ax, x, y, w, h, text, fc="white", ec=DARK, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else None)


def arrow(ax, a, b, text=None, color=DARK, off=0.12, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + off, text, ha="center", va="bottom", fontsize=9.5, color=color)


def image(ax, img, x, y, s, label):
    im = img.transpose(1, 2, 0).squeeze()
    ax.imshow(im, extent=(x, x + s, y, y + s), cmap="gray" if im.ndim == 2 else None, zorder=3)
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="square,pad=0", fc="none", ec=DARK, lw=1, zorder=4))
    ax.text(x + s / 2, y - 0.12, label, ha="center", va="top", fontsize=10)


def circle_with_posterior(ax, cx, cy, r, theta, kappa, label):
    ax.add_patch(Circle((cx, cy), r, fc="none", ec=GREY, lw=1.2))
    width = 150 / (1 + kappa)  # wedge narrows as the concentration grows
    t = np.degrees(theta)
    ax.add_patch(Wedge((cx, cy), r, t - width, t + width, width=0.07, fc=BLUE, ec="none", alpha=0.55))
    ax.plot([cx, cx + r * np.cos(theta)], [cy, cy + r * np.sin(theta)], color=BLUE, lw=1.5)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), "o", color=BLUE, ms=4)
    ax.text(cx, cy - r - 0.1, label, ha="center", va="top", fontsize=9.5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--out", default="overview.pdf")
    p.add_argument("--index", type=int, default=3, help="which of the 10 saved test images")
    a = p.parse_args()
    z = np.load(os.path.join(a.run_dir, "raw", "recon.npz"))
    x, xr = z["x"][a.index], z["mean"][a.index]

    fig, ax = plt.subplots(figsize=(14.4, 6.0))
    ax.set(xlim=(0, 14.4), ylim=(0, 6.0), aspect="equal")
    ax.axis("off")

    # ---- top row: the vae
    y0 = 3.3
    ax.text(7.5, y0 + 2.35, r"training: $\mathcal{L}=\|\mathbf{x}-\hat{\mathbf{x}}\|_1+\beta\sum_{k=1}^{d-1}"
            r"\mathrm{KL}\left(q(\theta_k\mid\mathbf{x})\,\|\,\mathcal{U}(\mathcal{S}^1)\right)$", ha="center", fontsize=11.5)
    image(ax, x, 0.15, y0, 1.3, r"input $\mathbf{x}$")
    box(ax, 1.8, y0 + 0.25, 1.35, 0.8, "encoder\n(CNN / MLP)")
    arrow(ax, (1.5, y0 + 0.65), (1.78, y0 + 0.65))
    box(ax, 3.5, y0 + 0.2, 1.85, 0.9, r"angles $\boldsymbol{\mu}\in[-\pi,\pi]^d$" + "\n" + r"concentration $\kappa$", fs=10)
    arrow(ax, (3.17, y0 + 0.65), (3.48, y0 + 0.65))

    # posterior on the torus: a few circles
    ax.text(7.05, y0 + 1.45, r"posterior on the Clifford torus $(\mathcal{S}^1)^{d-1}$", ha="center", fontsize=10.5)
    for cx, th, lab in [(6.05, 0.6, r"$\theta_1$"), (6.95, 2.3, r"$\theta_2$"), (8.05, -1.2, r"$\theta_{d-1}$")]:
        circle_with_posterior(ax, cx, y0 + 0.65, 0.36, th, 6, lab)
    ax.text(7.5, y0 + 0.62, r"$\cdots$", fontsize=14, ha="center", va="center")
    ax.text(7.05, y0 - 0.25, r"power spherical on each $\mathcal{S}^1$, one $\kappa$ per image", ha="center",
            va="top", fontsize=9.5, color=BLUE)
    arrow(ax, (5.37, y0 + 0.65), (5.63, y0 + 0.65))

    box(ax, 8.95, y0 + 0.05, 2.05, 1.2, "unitary code\n" + r"$\mathbf{z}=\mathcal{F}^{-1}\{e^{i\theta_k}\}\in\mathbb{R}^{2d}$"
        + "\n" + r"$|\mathcal{F}\{\mathbf{z}\}_k|=1$ for all $k$", fc="#E3F2FD", ec=BLUE, fs=10)
    arrow(ax, (8.47, y0 + 0.65), (8.93, y0 + 0.65))
    box(ax, 11.35, y0 + 0.25, 1.2, 0.8, "decoder")
    arrow(ax, (11.02, y0 + 0.65), (11.33, y0 + 0.65))
    image(ax, xr, 12.9, y0, 1.3, r"reconstruction $\hat{\mathbf{x}}$")
    arrow(ax, (12.57, y0 + 0.65), (12.88, y0 + 0.65))

    # ---- bottom row: hrr algebra on the codes
    y1 = 0.35
    ax.plot([0.2, 14.2], [y1 + 2.45, y1 + 2.45], color=GREY, lw=0.8, ls=":")
    ax.text(0.2, y1 + 2.2, "HRR operations on the codes (every code is unitary, so binding is exactly invertible)",
            fontsize=10.5, weight="bold", va="top")
    cards = [
        ("bind", r"$\mathbf{a}\circledast\mathbf{b}$", r"phases add: $\theta^a_k+\theta^b_k$" + "\nstays on the torus (unitary)"),
        ("unbind", r"$(\mathbf{a}\circledast\mathbf{b})\circledast\mathbf{b}^{*}=\mathbf{a}$",
         r"involution $\mathbf{b}^{*}$: phases negate" + "\nexact inverse, no clean-up"),
        ("bundle", r"$\mathbf{R}=\frac{1}{\sqrt{k}}\sum_i \mathbf{r}_i\circledast\mathbf{f}_i$",
         "sum leaves the torus (not unitary)\n"
         + r"$\mathbf{R}\circledast\mathbf{r}_i^{*}=\frac{1}{\sqrt{k}}(\mathbf{f}_i+$crosstalk$)$"),
        ("clean-up", r"$\arg\max_j\ \cos(\mathbf{R}\circledast\mathbf{r}_i^{*},\ \mathbf{m}_j)$",
         "nearest neighbour in the\nitem memory of encoded images"),
    ]
    w = 3.3
    for i, (title, eq, note) in enumerate(cards):
        x0 = 0.25 + i * (w + 0.25)
        ax.add_patch(FancyBboxPatch((x0, y1), w, 1.65, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white",
                                    ec=BLUE if i < 2 else DARK, lw=1.2))
        ax.text(x0 + 0.15, y1 + 1.45, title, fontsize=11, weight="bold", va="top")
        ax.text(x0 + w / 2, y1 + 0.98, eq, fontsize=11.5, ha="center", va="center")
        ax.text(x0 + w / 2, y1 + (0.33 if note.count("\n") > 1 else 0.38), note, fontsize=9.5, ha="center", va="center", color=DARK)
        if i < 3:
            arrow(ax, (x0 + w + 0.02, y1 + 0.82), (x0 + w + 0.23, y1 + 0.82))
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()

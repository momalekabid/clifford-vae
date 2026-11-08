"""every figure of the paper from the runs' metrics.json and raw/ dumps (no training, no gpu).
style follows the submission: Gaussian purple, Gaussian (L2) green, S-PWS orange, Clifford blue.

    python figures.py                      # runs/ -> runs/_aggregate/figures/
    python figures.py --root results_rev --seed 1

figures:
  vsa_{ds}_d{d}.pdf          bundle capacity, unbinding depth, role-filler (image roles, random roles)
  rolefiller_{ds}_d{d}.pdf   role-filler variants: image roles (no shared class), random fillers,
                             cross-dataset fillers (fashionmnist roles, cifar-10 fillers)
  recon_{ds}_d{d}.png        first 10 test images and their reconstructions, one panel per prior
  recovery_{ds}_d{d}_{prior}_{inv|dagger}.png   decoded target after m bind/unbind cycles
  traversal_mnist_d{d}.png   clifford angle grid
  manifold_mnist_d2.pdf      every prior's own 2-d latent space: decoded grid (top), test codes by digit (bottom)
  torus_mnist_d{d}.pdf       clifford: an anchored angle grid tiled 2x2 (it wraps both ways), test
                             codes on angles 1, 2 (flat and on a donut)
  anchored_mnist_d{d}.png    clifford: angles 1, 2 swept around real test images
the random unitary baseline (length 2d, clifford's vector length) is cached in _aggregate/references.json.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from clifford_vae import common, vsa as V  # noqa: E402
from clifford_vae.stats import COLOR, DATASET, LABEL, MARKER, cell, hdi, load_runs  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "axes.labelsize": 13, "axes.linewidth": 0.8, "legend.fontsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

CNN_DIMS = [128, 256, 512, 1024, 2048, 4096]
DIMS = {"cifar10": CNN_DIMS, "fashionmnist": CNN_DIMS, "celeba": [128, 256, 512], "mnist": [2, 5, 10, 20, 40, 128]}
PRIORS = {"mnist": ["N", "N_L2", "S", "V", "C"]}
CNN_PRIORS = ["N", "N_L2", "S", "C"]
PANELS = {
    "bundle_capacity": ("k", "accuracy", "Bundle Capacity", "Number of Bundled Vectors ($k$)", "Retrieval Accuracy"),
    "unbinding_depth_inv": ("m", "similarity", "Unbinding Depth", "Number of Bind-Unbind Cycles ($m$)",
                            "Similarity to Original"),
    "role_filler": ("k", "accuracy", "Role-Filler, Image Roles", "Number of Bundled Pairs ($k$)", "Unbinding Accuracy"),
    "role_filler_random_roles": ("k", "accuracy", "Role-Filler, Random Roles", "Number of Bundled Pairs ($k$)",
                                 "Unbinding Accuracy"),
    "role_filler_random_fillers": ("k", "accuracy", "Role-Filler, Random Fillers", "Number of Bundled Pairs ($k$)",
                                   "Unbinding Accuracy"),
    "role_filler_cross": ("k", "accuracy", "Role-Filler, Cross-Dataset Fillers", "Number of Bundled Pairs ($k$)",
                          "Unbinding Accuracy"),
}


def _style(p):
    ref = p == "ref"
    return dict(color=COLOR[p], ls="--" if ref else "-", lw=1.2 if ref else (3.0 if p == "C" else 2.0), marker=MARKER[p],
                markersize=6 if ref else 3, markevery=5, label=LABEL[p], zorder=5 if p == "C" else 3)


def references(runs, path):
    """random unitary item memories at every clifford item dimension that occurs."""
    refs = common.read_json(path, {})
    dims = {common.get(m, "vsa.dim") for (ds, c), ms in runs.items() if c.startswith("C-") for m in ms}
    for D in sorted(x for x in dims if x):
        if str(D) not in refs:
            refs[str(D)] = V.reference(D)
            common.write_json(path, refs)
    return refs


def _curve(ax, results, name, p, label=None):
    xk, yk = PANELS[name][:2]
    results = [r for r in results if r and r.get(yk)]
    if not results:
        return
    x = np.array(results[0][xk])
    Y = np.array([r[yk][: len(x)] for r in results])
    ax.plot(x, Y.mean(0), **(_style(p) | ({"label": label} if label else {})))
    if len(Y) > 1 and p != "ref":
        b = np.array([hdi(Y[:, j]) for j in range(Y.shape[1])])
        ax.fill_between(x, b[:, 0], b[:, 1], color=COLOR[p], alpha=0.15, lw=0)


def vsa_figure(runs, ds, d, priors, refs, names, out, extra=None):
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    for ax, name in zip(axes, names):
        for p in priors:
            ms = runs.get((ds, cell(p, d)), [])
            res = (extra or {}).get(p) if name == "role_filler_cross" else [common.get(m, f"vsa.{name}") for m in ms]
            _curve(ax, res or [], name, p)
        cl = runs.get((ds, cell("C", d)), [])
        ref = refs.get(str(common.get(cl[0], "vsa.dim"))) if cl else None
        if ref and name in ref:
            _curve(ax, [ref[name]], name, "ref", f"Unitary HRR (length {ref['dim']})")
        _, _, title, xl, yl = PANELS[name]
        ax.set(title=f"{title} ($d={d}$)", xlabel=xl, ylabel=yl)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, lw=0.6)
    h, l = axes[0].get_legend_handles_labels()
    if not l:  # no run has these curves (vsa skipped)
        plt.close(fig)
        return
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{DATASET.get(ds, ds)} --- VSA Operation Benchmarks ($d={d}$)", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def cross_dataset(runs, d, priors, cache_path, roles_ds="fashionmnist", fillers_ds="cifar10"):
    """role_filler_cross from the saved codes of two runs with the same prior, d and seed."""
    cache = common.read_json(cache_path, {})
    out = {}
    for p in priors:
        res = []
        for m in runs.get((roles_ds, cell(p, d)), []):
            key = f"{p}-d{d}-s{m['_seed']}"
            if key not in cache:
                other = [o for o in runs.get((fillers_ds, cell(p, d)), []) if o["_seed"] == m["_seed"]]
                fr, ff = os.path.join(m["_dir"], "raw", "codes.npz"), other and os.path.join(other[0]["_dir"], "raw", "codes.npz")
                if not other or not os.path.exists(fr) or not os.path.exists(ff):
                    continue
                cr, cf = (torch.from_numpy(np.load(f)["codes"]).float() for f in (fr, ff))
                cache[key] = V.role_filler_cross(cr, cf, seed=m["_seed"])
                common.write_json(cache_path, cache)
            res.append(cache[key])
        out[p] = res
    return out


def _grid(imgs):
    """(rows, cols, C, H, W) uint8 -> one (H*rows, W*cols, C) image with 2px white padding."""
    r, c, C, H, W = imgs.shape
    out = np.full((r * (H + 2) + 2, c * (W + 2) + 2, C), 255, np.uint8)
    for i in range(r):
        for j in range(c):
            out[2 + i * (H + 2): 2 + i * (H + 2) + H, 2 + j * (W + 2): 2 + j * (W + 2) + W] = imgs[i, j].transpose(1, 2, 0)
    return out.squeeze(-1) if C == 1 else out


def _imsave(path, img, scale=3):
    plt.imsave(path, np.kron(img, np.ones((scale, scale) + ((1,) if img.ndim == 3 else ()), np.uint8)),
               cmap="gray" if img.ndim == 2 else None)


def _run(runs, ds, p, d, seed):
    return next((m for m in runs.get((ds, cell(p, d)), []) if m["_seed"] == seed), None)


def recon_figure(runs, ds, d, priors, seed, out, code="mean"):
    """the submission's layout: per prior, a 2-row strip (inputs, reconstructions)."""
    for p in priors:
        m = _run(runs, ds, p, d, seed)
        f = m and os.path.join(m["_dir"], "raw", "recon.npz")
        if not f or not os.path.exists(f):
            continue
        z = np.load(f)
        _imsave(out.replace(".png", f"_{p}.png"), _grid(np.stack([z["x"], z[code]])))


def recovery_figure(runs, ds, d, priors, seed, outdir):
    for p in priors:
        m = _run(runs, ds, p, d, seed)
        f = m and os.path.join(m["_dir"], "raw", "recovery.npz")
        if not f or not os.path.exists(f):
            continue
        z = np.load(f)
        for method in ("inv", "dagger"):
            fig, ax = plt.subplots(figsize=(1.5 * (len(z["depths"]) + 1), 2 * len(z["targets"])))
            g = _grid(z[method])
            ax.imshow(g, cmap="gray" if g.ndim == 2 else None)
            H = z[method].shape[-1] + 2
            ax.set_xticks([2 + H * i + H // 2 for i in range(len(z["depths"]) + 1)],
                          ["original"] + [f"$m={k}$" for k in z["depths"]])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{LABEL[p]}, unbind with {'involution' if method == 'inv' else 'exact inverse'}")
            fig.savefig(os.path.join(outdir, f"recovery_{ds}_d{d}_{p}_{method}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)


def traversal_figure(runs, d, seed, out):
    m = _run(runs, "mnist", "C", d, seed)
    f = m and os.path.join(m["_dir"], "raw", "traversal.npz")
    if f and os.path.exists(f):
        z = np.load(f)
        n = int(z["n_grid"])
        rows = int(z["rows"]) if "rows" in z.files else n
        _imsave(out, _grid(z["x"].reshape(rows, n, *z["x"].shape[1:])), scale=2)


DIGITS = plt.get_cmap("tab10")
TORUS_R, TORUS_r = 2.0, 0.8


def _manifold(runs, p, d, seed):
    m = _run(runs, "mnist", p, d, seed)
    f = m and os.path.join(m["_dir"], "raw", "manifold.npz")
    return np.load(f) if f and os.path.exists(f) else None


def _wrap(t):
    return np.mod(t + np.pi, 2 * np.pi) - np.pi


def _scatter_kw(y):
    return dict(c=DIGITS(y % 10), s=4, alpha=0.6, lw=0, rasterized=True)


def _digit_legend(fig):
    h = [plt.Line2D([], [], ls="", marker="o", color=DIGITS(i), label=str(i)) for i in range(10)]
    fig.legend(handles=h, loc="upper center", ncol=10, frameon=False, title="digit", bbox_to_anchor=(0.5, 0.0))


def manifold_d2_figure(runs, seed, out):
    """one column per prior at d=2: its decoded latent space and where the test digits land in it."""
    cols = [(p, z) for p in PRIORS["mnist"] if (z := _manifold(runs, p, 2, seed)) is not None and "grid" in z.files]
    if not cols:
        return
    fig = plt.figure(figsize=(4.2 * len(cols), 8.6))
    for j, (p, z) in enumerate(cols):
        ax = fig.add_subplot(2, len(cols), j + 1)
        r, c = z["grid_shape"]
        ax.imshow(_grid(z["grid"].reshape(r, c, *z["grid"].shape[1:])), cmap="gray")
        ax.set_title(LABEL[p], color=COLOR[p])
        ax.axis("off")
        lat, y = z["latent"], z["y"]
        if p in ("S", "V"):
            ax = fig.add_subplot(2, len(cols), len(cols) + j + 1, projection="3d")
            ax.scatter(*lat.T, **_scatter_kw(y))
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            continue
        ax = fig.add_subplot(2, len(cols), len(cols) + j + 1)
        if p == "N":
            ax.scatter(lat[:, 0], lat[:, 1], **_scatter_kw(y))
            ax.set(xlabel="$z_1$", ylabel="$z_2$")
        else:  # one circle: N_L2 direction, or clifford's single free angle; small radial jitter to show density
            t = np.arctan2(lat[:, 1], lat[:, 0]) if p == "N_L2" else lat[:, 1]
            rad = 1 + 0.08 * np.random.default_rng(0).standard_normal(len(t))
            ax.scatter(rad * np.cos(t), rad * np.sin(t), **_scatter_kw(y))
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle("MNIST, $d=2$: each prior's latent space (decoded) and the test digits in it", fontsize=15)
    _digit_legend(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def torus_figure(runs, d, seed, out):
    """clifford at d >= 3: the decoded grid over angles 1, 2 around the first anchor image, tiled 2x2
    (seamless, both axes are circles), and the test codes on angles 1 and 2, flat and on a donut."""
    z = _manifold(runs, "C", d, seed)
    if z is None or "anchors" not in z.files:
        return
    n = int(z["anchor_grid"])
    g = z["anchors"][0].reshape(n, n, *z["anchors"].shape[2:])
    tiled = np.concatenate([np.concatenate([g, g], 1)] * 2, 0)
    th1, th2, y = _wrap(z["latent"][:, 1]), _wrap(z["latent"][:, 2]), z["y"]
    fig = plt.figure(figsize=(19, 6.2))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(_grid(tiled), cmap="gray")
    ax.axhline(_grid(g).shape[0] - 1, color=COLOR["C"], lw=1.5, ls="--")
    ax.axvline(_grid(g).shape[1] - 1, color=COLOR["C"], lw=1.5, ls="--")
    ax.set_title(r"decoded over $\theta_1 \times \theta_2$, tiled $2\times 2$")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(th1, th2, **_scatter_kw(y))
    ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", aspect="equal",
           title="test codes (edges wrap)")
    ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    u, v = np.meshgrid(np.linspace(-np.pi, np.pi, 60), np.linspace(-np.pi, np.pi, 30))
    ax.plot_surface((TORUS_R + TORUS_r * np.cos(v)) * np.cos(u), (TORUS_R + TORUS_r * np.cos(v)) * np.sin(u),
                    TORUS_r * np.sin(v), color="#BBDEFB", alpha=0.12, lw=0)
    rr = TORUS_R + (TORUS_r + 0.02) * np.cos(th2)
    ax.scatter(rr * np.cos(th1), rr * np.sin(th1), (TORUS_r + 0.02) * np.sin(th2), **(_scatter_kw(y) | {"s": 7, "alpha": 0.8}))
    ax.view_init(elev=48, azim=-60)
    ax.set_box_aspect((1, 1, 0.4))
    ax.set_axis_off()
    ax.set_title("on the torus")
    fig.suptitle(f"MNIST, Clifford $d={d}$: two of its {d - 1} circles", fontsize=15)
    _digit_legend(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def anchored_figure(runs, d, seed, out):
    """clifford: angles 1 and 2 swept around real test images (other angles held at the image's)."""
    z = _manifold(runs, "C", d, seed)
    if z is None or "anchors" not in z.files:
        return
    n = int(z["anchor_grid"])
    panels = [_grid(a.reshape(n, n, *a.shape[1:])) for a in z["anchors"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.4))
    for ax, p, x in zip(np.atleast_1d(axes), panels, z["anchor_x"]):
        ax.imshow(p, cmap="gray")
        ax.axis("off")
        ins = ax.inset_axes([0.0, 1.01, 0.14, 0.14])
        ins.imshow(x.transpose(1, 2, 0).squeeze(), cmap="gray")
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("anchor", fontsize=10)
    fig.suptitle(rf"MNIST, Clifford $d={d}$: $\theta_1, \theta_2$ swept around real images", fontsize=15, y=1.06)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=common.RESULTS)
    p.add_argument("--seed", type=int, default=1, help="run shown in the image figures")
    a = p.parse_args()
    runs = load_runs(a.root)
    agg = os.path.join(a.root, "_aggregate")
    out = os.path.join(agg, "figures")
    os.makedirs(out, exist_ok=True)
    refs = references(runs, os.path.join(agg, "references.json"))
    present = {ds for ds, _ in runs}
    for ds in [x for x in DIMS if x in present]:
        priors = PRIORS.get(ds, CNN_PRIORS)
        for d in DIMS[ds]:
            if not any((ds, cell(q, d)) in runs for q in priors):
                continue
            vsa_figure(runs, ds, d, priors, refs, ["bundle_capacity", "unbinding_depth_inv", "role_filler",
                                                   "role_filler_random_roles"], os.path.join(out, f"vsa_{ds}_d{d}.pdf"))
            names = ["role_filler", "role_filler_random_fillers"]
            extra = None
            if ds == "fashionmnist" and "cifar10" in present:
                extra = cross_dataset(runs, d, priors, os.path.join(agg, "role_filler_cross.json"))
                names.append("role_filler_cross")
            vsa_figure(runs, ds, d, priors, refs, names, os.path.join(out, f"rolefiller_{ds}_d{d}.pdf"), extra)
            recon_figure(runs, ds, d, priors, a.seed, os.path.join(out, f"recon_{ds}_d{d}.png"))
            recovery_figure(runs, ds, d, priors, a.seed, out)
            if ds == "mnist":
                traversal_figure(runs, d, a.seed, os.path.join(out, f"traversal_mnist_d{d}.png"))
                torus_figure(runs, d, a.seed, os.path.join(out, f"torus_mnist_d{d}.pdf"))
                anchored_figure(runs, d, a.seed, os.path.join(out, f"anchored_mnist_d{d}.png"))
        if ds == "mnist":
            manifold_d2_figure(runs, a.seed, os.path.join(out, "manifold_mnist_d2.pdf"))
    print(f"figures in {out}")


if __name__ == "__main__":
    main()

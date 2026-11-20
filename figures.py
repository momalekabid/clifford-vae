"""every figure of the paper from the runs' metrics.json and raw/ dumps (no training, no gpu).
style follows the submission: Gaussian purple, Gaussian (L2) green, S-PWS orange, Clifford blue.

    python figures.py                      # runs/ -> runs/_aggregate/figures/
    python figures.py --root results_rev --seed 1

figures:
  vsa_{ds}_d{d}.pdf          2 x 2: bundle capacity, unbinding depth (top); role-filler with image roles and
                             with random unitary roles (bottom)
  rolefiller_{ds}_d{d}.pdf   role-filler with image roles (no shared class) next to image roles bound to
                             random unitary fillers
  recon_{ds}_d{d}.png        first 10 test images and their reconstructions, one panel per prior
  samples_{ds}_d{d}.png      20 decoded prior samples per prior
  shapes3d_unseen_grid.png   held-out pairs as images: sources, target, and each prior's three decodings
  shapes3d_composition.pdf   held-out hue x shape pairs: hrr analogy, vector analogy, decoding the pair's
                             own image, for every prior and the constant-beta gaussians (<root>_betavae)
  wason.pdf                  wason card task: trials solved against d, per dataset
  recovery_{ds}_d{d}_{prior}_{inv|dagger}.png   decoded target after m bind/unbind cycles
  traversal_mnist_d{d}.png   clifford angle grid
  manifold_mnist_2d.pdf      every prior at the d where its latent space is a surface (plane, sphere, torus):
                             decoded grid (top), test codes by digit (bottom). needs mnist N_L2 and C at d=3
                             (python train_mnist.py --prior C --d 3 --seed 1)
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
}


# canonical hrr item memories by length, filled in main. the hrr reference has the unitary length 2d; with HRR_HALF
# it has d components (half that length, the equal-component comparison of schlegel et al. 2022)
HRR_REFS = {}
HRR_HALF = False


def _style(p):
    ref = p in ("ref", "hrr")
    return dict(color=COLOR[p], ls="--" if ref else "-", lw=1.2 if ref else (3.0 if p == "C" else 2.0), marker=MARKER[p],
                markersize=6 if ref else 3, markevery=5, label=LABEL[p], zorder=5 if p == "C" else 3)


def references(runs, path, make=V.reference, half=False):
    """random item memories (unitary by default) at every clifford item dimension that occurs (half=True: at half
    of each, the number of phases)."""
    refs = common.read_json(path, {})
    dims = {common.get(m, "vsa.dim") for (ds, c), ms in runs.items() if c.startswith("C-") for m in ms}
    dims = {x // 2 if half else x for x in dims if x}
    for D in sorted(x for x in dims if x):
        stale = make is V.reference_hrr and refs.get(str(D), {}).get("random_atoms") != "hrr"  # older caches: unitary atoms
        if str(D) not in refs or refs[str(D)].get("n_trials", 0) < V.REF_TRIALS or stale:  # older caches used 10 trials
            refs[str(D)] = make(D)
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
    if len(Y) > 1 and p not in ("ref", "hrr"):
        b = np.array([hdi(Y[:, j]) for j in range(Y.shape[1])])
        ax.fill_between(x, b[:, 0], b[:, 1], color=COLOR[p], alpha=0.15, lw=0)


def vsa_figure(runs, ds, d, priors, refs, names, out):
    """four panels go on a 2 x 2 grid (readable at page width); fewer sit side by side."""
    grid = len(names) == 4
    if grid:
        fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.4))
        axes = axs.ravel()
    else:
        fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.8))
    n_max = max((len(runs.get((ds, cell(p, d)), [])) for p in priors), default=0)
    for ax, name in zip(axes, names):
        for p in priors:
            ms = runs.get((ds, cell(p, d)), [])
            res = [common.get(m, f"vsa.{name}") for m in ms]
            # a prior with fewer runs than the rest (vmf on mnist) says so in the legend
            _curve(ax, res, name, p, f"{LABEL[p]} ({len(ms)} runs)" if 0 < len(ms) < n_max else None)
        cl = runs.get((ds, cell("C", d)), [])
        ref = refs.get(str(common.get(cl[0], "vsa.dim"))) if cl else None
        if ref and name in ref:
            _curve(ax, [ref[name]], name, "ref", f"Unitary HRR (length {ref['dim']})")
        # canonical hrr at half the unitary length: d real components against the d phases of a clifford code or of
        # the unitary reference, the equal-component comparison of schlegel et al. (2022)
        hrr = HRR_REFS.get(str(ref["dim"] // 2 if HRR_HALF else ref["dim"])) if ref else None
        if hrr and name in hrr:
            _curve(ax, [hrr[name]], name, "hrr", f"HRR (length {hrr['dim']})")
        _, _, title, xl, yl = PANELS[name]
        ax.set(title=f"{title} ($d={d}$)", xlabel=xl, ylabel=yl)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, lw=0.6)
    h, l = axes[0].get_legend_handles_labels()
    if not l:  # no run has these curves (vsa skipped)
        plt.close(fig)
        return
    title = f"{DATASET.get(ds, ds)} --- VSA Operation Benchmarks ($d={d}$)"
    if grid:
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.04, 1, 0.98], h_pad=2.0, w_pad=2.0)
    else:
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


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


def samples_figure(runs, ds, d, priors, seed, out, n=20, per_row=10):
    """decoded prior samples, one block of n per prior (the images behind the fid numbers)."""
    blocks = []
    for p in priors:
        m = _run(runs, ds, p, d, seed)
        f = m and os.path.join(m["_dir"], "raw", "prior_samples.npz")
        if f and os.path.exists(f):
            x = np.load(f)["x"][:n]
            fid = common.get(m, "fid.fid")
            blocks.append((LABEL[p] + (f" (FID {fid:.0f})" if fid else ""), _grid(x.reshape(n // per_row, per_row, *x.shape[1:]))))
    if not blocks:
        return
    fig, axes = plt.subplots(len(blocks), 1, figsize=(7.2, 1.62 * len(blocks) + 0.2), squeeze=False)
    for ax, (title, g) in zip(axes[:, 0], blocks):
        ax.imshow(g, cmap="gray" if g.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="normal", loc="left", pad=3)
        ax.axis("off")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def composition_figure(runs, root, out):
    """3d shapes, held-out hue x shape pairs: three scores per model, mean and 95% hdi over runs."""
    beta = load_runs(os.path.normpath(root) + "_betavae")
    models = [(LABEL[p], COLOR[p], runs.get(("shapes3d", cell(p, 128)), [])) for p in ("C", "N", "N_L2", "S")]
    models += [(rf"{name}, $\beta={b}$", color, beta.get(("shapes3d", f"{p}-d128-b{b}"), []))
               for p, name, color, bs in (("N", "Gaussian", COLOR["N"], (4, 8, 16)), ("C", "Clifford", COLOR["C"], (0.5, 4, 8, 16)))
               for b in bs]
    models = [m for m in models if m[2]]
    if not models:
        return
    tests = [("composition.analogy.hrr", "Binding analogy\n" + r"$\mathbf{a}\circledast\mathbf{c}\circledast\mathbf{b}^{*}$"),
             ("composition.analogy.vector", "Vector analogy\n" + r"$\mathbf{a}-\mathbf{b}+\mathbf{c}$"),
             ("composition.unseen.attr_acc", "Decoding a held-out pair\nfrom its own image")]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.84 / len(models)
    for i, (label, color, ms) in enumerate(models):
        vals = [[common.get(m, k) for m in ms if common.get(m, k) is not None] for k, _ in tests]
        mean = np.array([np.mean(v) if v else np.nan for v in vals])
        ci = np.array([hdi(v) if v else (np.nan, np.nan) for v in vals])
        x = np.arange(len(tests)) + (i - (len(models) - 1) / 2) * w
        b = next((k for k, v in enumerate((0.5, 4, 8, 16)) if rf"\beta={v}$" in label), None)  # hatched; paler as beta grows
        ax.bar(x, mean, w * 0.92, color=color, alpha=1.0 if b is None else (0.9, 0.75, 0.5, 0.3)[b], label=label,
               hatch=None if b is None else ("..", "//", "//", "//")[b], edgecolor="white", linewidth=0.6,
               yerr=np.abs(ci.T - mean), error_kw=dict(lw=1, capsize=2, ecolor="0.2"))
    ax.set_xticks(range(len(tests)), [t for _, t in tests])
    ax.set(ylabel="Held-out pairs correct", ylim=(0, 1.05), title="3D Shapes --- Unseen Hue $\\times$ Shape Pairs ($d=128$)")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(loc="upper center", ncol=(len(models) + 1) // 2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def unseen_grid_figure(runs, seed, out, priors=("C", "N", "N_L2", "S"), cols=(0, 1, 2, 3), extra=()):
    """3d shapes: held-out hue x shape pairs as images, four blocks on a 2 x 2 grid. top left the three seen
    sources and the never-seen target; then what each prior decodes for the hrr analogy, the vector analogy
    and the target's own code. extra: (label, colour, npz path) rows appended to the three result blocks."""
    zs = {}
    for p in priors:
        m = _run(runs, "shapes3d", p, 128, seed)
        f = m and os.path.join(m["_dir"], "raw", "unseen_grid.npz")
        if f and os.path.exists(f):
            zs[p] = (LABEL[p], COLOR[p], np.load(f))
    for label, color, f in extra:
        if os.path.exists(f):
            zs[label] = (label, color, np.load(f))
    if not zs:
        return
    z0 = next(iter(zs.values()))[2]
    cols = [c for c in cols if c < len(z0["target"])]
    inputs = [(r"source $\mathbf{a}$ (seen)", "0.15", z0["a"]), (r"source $\mathbf{b}$ (seen)", "0.15", z0["b"]),
              (r"source $\mathbf{c}$ (seen)", "0.15", z0["c"]), ("target (held out)", "0.15", z0["target"])]
    blocks = [("Inputs", inputs)]
    for key, title in [("hrr", r"Binding analogy $\mathbf{a}\circledast\mathbf{c}\circledast\mathbf{b}^{*}$"),
                       ("vector", r"Vector analogy $\mathbf{a}-\mathbf{b}+\mathbf{c}$"), ("own", "Control: target reconstruction")]:
        blocks.append((title, [(label, color, z[key]) for label, color, z in zs.values()]))
    k, lab, head, gx, gy = len(cols), 2.55, 0.6, 0.55, 0.5  # image columns, label width, title height, gaps
    rows = max(len(b[1]) for b in blocks)
    bw, bh = lab + k, head + rows
    W, H = 2 * bw + gx, 2 * bh + gy
    fig, ax = plt.subplots(figsize=(0.72 * W, 0.72 * H))
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    for n, (title, rws) in enumerate(blocks):
        x0, y0 = (n % 2) * (bw + gx), (n // 2) * (bh + gy)
        # titles sit centred over the image columns, so no title runs past its block
        ax.text(x0 + lab + k / 2, y0 + head / 2, title, ha="center", va="center", fontsize=13.5, fontweight="bold")
        for r, (label, color, imgs) in enumerate(rws):
            y = y0 + head + r
            ax.text(x0 + lab - 0.14, y + 0.5, label, ha="right", va="center", fontsize=13, color=color)
            for c, j in enumerate(cols):
                ax.imshow(imgs[j].transpose(1, 2, 0), extent=(x0 + lab + c + 0.04, x0 + lab + c + 0.96, y + 0.96, y + 0.04),
                          interpolation="nearest")
    ax.set_aspect("equal")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def wason_figure(runs, agg, out, datasets=("mnist", "fashionmnist", "cifar10")):
    """wason card task: fraction of trials solved by the learned transform against d."""
    datasets = [ds for ds in datasets if any("wason" in m for (x, _), ms in runs.items() if x == ds for m in ms)]
    if not datasets:
        return
    ref = common.read_json(os.path.join(agg, "wason_reference.json"), {})
    uref = common.read_json(os.path.join(agg, "wason_reference_unitary.json"), {})
    grid = len(datasets) == 3  # three panels and the legend on a 2 x 2 grid, as in the vsa figures
    if grid:
        fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
        panels, leg = list(axs.ravel()[:3]), axs[1, 1]
        leg.axis("off")
    else:
        fig, axs = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8), squeeze=False)
        panels = list(axs[0])
    for ax, ds in zip(panels, datasets):
        dims = DIMS[ds]
        for p in PRIORS.get(ds, CNN_PRIORS):
            xs, mean, lo, hi = [], [], [], []
            for i, d in enumerate(dims):
                v = [m["wason"]["transform"] for m in runs.get((ds, cell(p, d)), []) if "wason" in m]
                if v:
                    a, b = hdi(v)
                    xs.append(i), mean.append(np.mean(v)), lo.append(a), hi.append(b)
            if xs:
                ax.plot(xs, mean, **(_style(p) | {"markevery": 1, "markersize": 5}))
                ax.fill_between(xs, lo, hi, color=COLOR[p], alpha=0.15, lw=0)
        mlp = [(i, [m["wason"]["mlp"] for q in PRIORS.get(ds, CNN_PRIORS) for m in runs.get((ds, cell(q, d)), [])
                    if "mlp" in m.get("wason", {})]) for i, d in enumerate(dims)]
        mlp = [(i, float(np.mean(v))) for i, v in mlp if v]
        if mlp:  # the notebook's foil: an mlp fit to the same two rules, averaged over priors
            ax.plot(*zip(*mlp), color="0.45", ls=":", lw=1.6, marker="x", markersize=6, label="MLP on the same two rules")
        r = [(i, common.get(ref, f"{2 * d}.transform") or common.get(ref, f"{ds}.{2 * d}.transform")) for i, d in enumerate(dims)]
        r = [(i, v) for i, v in r if v is not None]
        # random unitary concept vectors of the clifford code's length; the structural symbols are those of every curve
        u = [(i, common.get(uref, f"gaussian_symbols.{2 * d}.transform")) for i, d in enumerate(dims)]
        u = [(i, v) for i, v in u if v is not None]
        if u:
            ax.plot(*zip(*u), color=COLOR["ref"], ls="--", lw=1.2, marker="+", markersize=7, label="Random unitary vectors (length $2d$)")
        if r:
            ax.plot(*zip(*r), color=COLOR["hrr"], ls="--", lw=1.2, marker="x", markersize=6, label="Random Gaussian vectors (length $2d$)")
        ax.set_xticks(range(len(dims)), [str(d) for d in dims])
        ax.set(title=DATASET.get(ds, ds), xlabel="Latent Dimension ($d$)", ylabel="Trials Solved", ylim=(-0.03, 1.05))
        ax.grid(alpha=0.25, lw=0.6)
    entries = {}
    for ax in panels:  # every label once, in the order of first appearance
        for h, l in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(l, h)
    if grid:
        leg.legend(entries.values(), entries.keys(), loc="center", frameon=False, fontsize=13.5, handlelength=3, labelspacing=0.9)
        fig.suptitle("Wason Card Task --- Rule Transform Applied to Unseen Concepts", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0, w_pad=2.0)
    else:
        fig.legend(entries.values(), entries.keys(), loc="lower center", ncol=len(entries), frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("Wason Card Task --- Rule Transform Applied to Unseen Concepts", fontsize=15)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


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


MANIFOLD_DIMS = {"N": 2, "N_L2": 3, "S": 2, "V": 2, "C": 3}  # the d at which each prior's latent space is a surface


def manifold_2d_figure(runs, seed, out, dims=MANIFOLD_DIMS):
    """one column per prior, each at the d where its latent space is two-dimensional (plane, sphere or
    torus): the decoded latent space and where the test digits land in it."""
    cols = [(p, z) for p in PRIORS["mnist"] if (z := _manifold(runs, p, dims[p], seed)) is not None and "grid" in z.files]
    if not cols:
        return
    fig = plt.figure(figsize=(4.2 * len(cols), 8.6))
    for j, (p, z) in enumerate(cols):
        ax = fig.add_subplot(2, len(cols), j + 1)
        r, c = z["grid_shape"]
        ax.imshow(_grid(z["grid"].reshape(r, c, *z["grid"].shape[1:])), cmap="gray")
        ax.set_title(f"{LABEL[p]} ($d={dims[p]}$)", color=COLOR[p])
        ax.axis("off")
        lat, y = z["latent"], z["y"]
        if lat.shape[1] == 3 and p != "C":  # a sphere: S, V, and N_L2 at d=3
            ax = fig.add_subplot(2, len(cols), len(cols) + j + 1, projection="3d")
            ax.scatter(*(lat / np.linalg.norm(lat, axis=1, keepdims=True)).T, **_scatter_kw(y))
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            continue
        ax = fig.add_subplot(2, len(cols), len(cols) + j + 1)
        if p == "N":
            ax.scatter(lat[:, 0], lat[:, 1], **_scatter_kw(y))
            ax.set(xlabel="$z_1$", ylabel="$z_2$")
        elif p == "C" and lat.shape[1] == 3:  # the flat torus: both axes wrap
            ax.scatter(_wrap(lat[:, 1]), _wrap(lat[:, 2]), **_scatter_kw(y))
            ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), aspect="equal")
            ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
            ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        else:  # one circle: N_L2 direction, or clifford's single free angle; small radial jitter to show density
            t = np.arctan2(lat[:, 1], lat[:, 0]) if p == "N_L2" else lat[:, 1]
            rad = 1 + 0.08 * np.random.default_rng(0).standard_normal(len(t))
            ax.scatter(rad * np.cos(t), rad * np.sin(t), **_scatter_kw(y))
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle("MNIST: each prior at the $d$ where its latent space is a two-dimensional manifold", fontsize=15)
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
    p.add_argument("--vsa-only", action="store_true", help="only the vsa and role-filler curve figures")
    p.add_argument("--hrr-length", choices=["d", "2d"], default="2d",
                   help="length of the canonical hrr reference: the unitary length 2d, or d components (schlegel et al. 2022)")
    a = p.parse_args()
    runs = load_runs(a.root)
    agg = os.path.join(a.root, "_aggregate")
    out = os.path.join(agg, "figures")
    os.makedirs(out, exist_ok=True)
    refs = references(runs, os.path.join(agg, "references.json"))
    global HRR_HALF
    HRR_HALF = a.hrr_length == "d"
    HRR_REFS.update(references(runs, os.path.join(agg, "references_hrr_half.json" if HRR_HALF else "references_hrr.json"),
                               V.reference_hrr, half=HRR_HALF))
    present = {ds for ds, _ in runs}
    for ds in [x for x in DIMS if x in present]:
        priors = PRIORS.get(ds, CNN_PRIORS)
        for d in DIMS[ds]:
            if not any((ds, cell(q, d)) in runs for q in priors):
                continue
            vsa_figure(runs, ds, d, priors, refs, ["bundle_capacity", "unbinding_depth_inv", "role_filler",
                                                   "role_filler_random_roles"], os.path.join(out, f"vsa_{ds}_d{d}.pdf"))
            vsa_figure(runs, ds, d, priors, refs, ["role_filler", "role_filler_random_fillers"],
                       os.path.join(out, f"rolefiller_{ds}_d{d}.pdf"))
            if a.vsa_only:
                continue
            recon_figure(runs, ds, d, priors, a.seed, os.path.join(out, f"recon_{ds}_d{d}.png"))
            recovery_figure(runs, ds, d, priors, a.seed, out)
            samples_figure(runs, ds, d, priors, a.seed, os.path.join(out, f"samples_{ds}_d{d}.png"))
            if ds == "mnist":
                traversal_figure(runs, d, a.seed, os.path.join(out, f"traversal_mnist_d{d}.png"))
                torus_figure(runs, d, a.seed, os.path.join(out, f"torus_mnist_d{d}.pdf"))
                anchored_figure(runs, d, a.seed, os.path.join(out, f"anchored_mnist_d{d}.png"))
        if ds == "mnist":
            manifold_2d_figure(runs, a.seed, os.path.join(out, "manifold_mnist_2d.pdf"))
    if a.vsa_only:
        return print(f"vsa figures in {out}")
    composition_figure(runs, a.root, os.path.join(out, "shapes3d_composition.pdf"))
    unseen_grid_figure(runs, a.seed, os.path.join(out, "shapes3d_unseen_grid.png"))
    wason_figure(runs, agg, os.path.join(out, "wason.pdf"))
    print(f"figures in {out}")


if __name__ == "__main__":
    main()

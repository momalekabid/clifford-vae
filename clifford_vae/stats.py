# reporting, shared by tables.py and figures.py: loading runs, mean + 95% hdi of the mean
# (bootstrap, 20k resamples), the paper's tie rule, and its labels / colours.
import glob
import os
from collections import defaultdict

import numpy as np

from . import common

CI, N_BOOT = 0.95, 20000

LABEL = {"N": "Gaussian", "N_L2": "Gaussian (L2)", "S": r"$\mathcal{S}$-PWS", "V": r"$\mathcal{S}$-vMF",
         "C": "Clifford (Ours)", "ref": "Unitary HRR"}
TEX = {"N": r"$\mathcal{N}$", "N_L2": r"$\mathcal{N}_{L2}$", "S": r"$\mathcal{S}$", "V": r"$\mathcal{V}$",
       "C": r"$\mathcal{C}$ (Ours)"}
COLOR = {"N": "#9C27B0", "N_L2": "#4CAF50", "S": "#FF9800", "V": "#E91E63", "C": "#2196F3", "ref": "#555555"}
MARKER = {"N": "D", "N_L2": "^", "S": "s", "V": "v", "C": "o", "ref": "+"}
DATASET = {"cifar10": "CIFAR-10", "fashionmnist": "FashionMNIST", "mnist": "MNIST", "celeba": "CelebA",
           "shapes3d": "3D Shapes", "omniglot": "Omniglot"}


def hdi(x, ci=CI, n_boot=N_BOOT, seed=0):
    """narrowest interval holding ci of the bootstrap distribution of the mean."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean())) if len(x) else (np.nan, np.nan)
    boot = np.sort(np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(1))
    w = int(np.floor(ci * n_boot))
    i = int(np.argmin(boot[w:] - boot[: n_boot - w]))
    return float(boot[i]), float(boot[i + w])


def load_runs(root):
    """{(dataset, cell): [metrics of finished runs, with "_dir" and "_seed"]}"""
    runs = defaultdict(list)
    for m in sorted(glob.glob(os.path.join(root, "*", "*", "seed*", "metrics.json"))):
        r = common.read_json(m, {})
        if r.get("status") != "done":
            continue
        rd = os.path.dirname(m)
        r["_dir"], r["_seed"] = rd, int(os.path.basename(rd)[4:])
        ds, cell = rd.split(os.sep)[-3], rd.split(os.sep)[-2]
        runs[(ds, cell)].append(r)
    return runs


def knn_value(m, prior, n):
    """the paper's k-NN number: euclidean for N, cosine for the rest. runs from before the
    metric was logged both ways stored cosine as acc_n and euclidean as euc_acc_n."""
    k = m.get("knn", {})
    key = f"euc_acc_{n}" if prior == "N" else f"cos_acc_{n}"
    if key in k:
        return k[key]
    return k.get(f"euc_acc_{n}") if prior == "N" else k.get(f"acc_{n}")


def values(runs, ds, cell, path):
    out = []
    for m in runs.get((ds, cell), []):
        if path.startswith("knn.acc_"):
            v = knn_value(m, cell.split("-d")[0], int(path.split("_")[-1]))
        else:
            v = common.get(m, path)
        if v is not None and np.isfinite(v):
            out.append(float(v))
    return np.array(out)


def cell(prior, d, beta_max=1.0):
    return f"{prior}-d{d}" + ("" if beta_max == 1.0 else f"-b{beta_max:g}")


def mark(samples, higher_better, scale=1.0, nd=1):
    """{prior: latex}. bold + underline: best mean with an hdi disjoint from every other entry.
    underline: the best and every entry whose hdi overlaps it (a tie). decided on raw intervals."""
    st = {}
    for p, x in samples.items():
        if len(x):
            lo, hi = hdi(x)
            st[p] = (x.mean() * scale, lo * scale, hi * scale)
    if not st:
        return {p: "--" for p in samples}
    best = (max if higher_better else min)(st, key=lambda p: st[p][0])
    tied = {p for p in st if p != best and not (st[p][2] < st[best][1] or st[best][2] < st[p][1])}
    out = {}
    for p in samples:
        if p not in st:
            out[p] = r"\textit{n.c.}"
            continue
        m, lo, hi = st[p]
        s = f"{m:.{nd}f}"
        if tied and (p == best or p in tied):
            s = rf"\underline{{{s}}}"
        elif p == best:
            s = rf"\textbf{{\underline{{{s}}}}}"
        out[p] = rf"\shortstack{{{s}\\{{\tiny[{lo:.{nd}f},\,{hi:.{nd}f}]}}}}"
    return out

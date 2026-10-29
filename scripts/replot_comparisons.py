"""replot vsa comparison + individual plots + knn accuracy from saved trial data.

usage:
    python3 scripts/replot_comparisons.py fashionmnist
    python3 scripts/replot_comparisons.py cifar10
    python3 scripts/replot_comparisons.py fashionmnist results   # custom results root

generates per dataset:
    results/comparisons/<ds>/vsa_comparison_d<dim>.png         (3-panel, shaded)
    results/comparisons/<ds>/bundle_capacity_d<dim>.png        (single panel, shaded)
    results/comparisons/<ds>/role_filler_d<dim>.png            (single panel, shaded)
    results/comparisons/<ds>/self_binding_d<dim>.png           (single panel, shaded)
    results/comparisons/<ds>/knn_accuracy_vs_dim.png           (knn @100/600/1000 vs dim, error bars)
    results/comparisons/<ds>/fid_vs_dim.png                    (generation fid vs dim, error bars)
"""
import os
import sys
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    bind as vsa_bind,
    unbind as vsa_unbind,
    unitary_init as vsa_unitary_init,
)


plt.rcParams.update({
    # clean sans-serif (Helvetica/Arial), matching the Schlegel-style plots;
    # falls back to DejaVu Sans if Helvetica/Arial aren't installed.
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",   # so $\mathcal{S}$-PWS etc. stay sans-serif
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",         # bold panel titles like the reference
    "axes.labelsize": 13,
    "axes.linewidth": 0.8,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

# pretty dataset names for titles
DATASET_TITLE = {
    "cifar10": "CIFAR-10",
    "fashionmnist": "FashionMNIST",
    "mnist": "MNIST",
}

COLORS = {
    "clifford": "#2196F3",
    "powerspherical": "#FF9800",
    "gaussian": "#4CAF50",
    "gaussian_nol2": "#9C27B0",
    "vmf": "#E91E63",
    "random_hrr": "#999999",
    "unitary": "#555555",
}
LABELS = {
    "clifford": "Clifford (Ours)",
    "powerspherical": r"$\mathcal{S}$-PWS",
    "gaussian": "Gaussian (L2)",
    "gaussian_nol2": "Gaussian",
    "vmf": r"$\mathcal{S}$-vMF",
    "random_hrr": "HRR",
    "unitary": "Unitary HRR",
}
MARKERS = {
    "clifford": "o",
    "powerspherical": "s",
    "gaussian": "^",
    "gaussian_nol2": "D",
    "vmf": "v",
    "random_hrr": "x",
    "unitary": "+",
}
ORDER = ["random_hrr", "unitary", "gaussian_nol2", "gaussian", "vmf", "powerspherical", "clifford"]


def is_ref(d): return d in ("random_hrr", "unitary")


def style(d):
    # emphasize our method: thicker line, drawn on top
    lw = 1.2 if is_ref(d) else (3.0 if d == "clifford" else 2.0)
    return {
        "color": COLORS.get(d, "black"),
        "linestyle": "--" if is_ref(d) else "-",
        "linewidth": lw,
        "marker": MARKERS.get(d, "o"),
        "label": LABELS.get(d, d),
        "zorder": 5 if d == "clifford" else 3,
    }


def gather_trials(dataset_name, results_root):
    """returns {dim: {dist: [vsa_raw, ...]}}, {dim: {dist: [metrics, ...]}}."""
    pat = re.compile(rf"{re.escape(dataset_name)}-(.+?)-d(\d+)-l1-trial\d+$")
    vsa_by, met_by = {}, {}
    for d in sorted(glob.glob(os.path.join(results_root, f"{dataset_name}-*-l1-trial*"))):
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        dist, dim = m.group(1), int(m.group(2))
        vp, mp = os.path.join(d, "vsa_raw.json"), os.path.join(d, "metrics.json")
        if os.path.exists(vp):
            with open(vp) as f:
                vsa_by.setdefault(dim, {}).setdefault(dist, []).append(json.load(f))
        if os.path.exists(mp):
            with open(mp) as f:
                met_by.setdefault(dim, {}).setdefault(dist, []).append(json.load(f))
    return vsa_by, met_by


def hdi_band(mat, ci=0.95, n_boot=20000, seed=0):
    """mat: (n_runs, n_x). returns (lo, hi) arrays = 95% HDI of the MEAN at each x,
    by bootstrap (20k resamples, seed 0) -- same definition as the result tables."""
    mat = np.asarray(mat, float)
    n = mat.shape[0]
    if n < 2:
        m = mat.mean(0)
        return m.copy(), m.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))          # shared resample across columns
    boot = mat[idx].mean(axis=1)                         # (n_boot, n_x) bootstrap means
    boot.sort(axis=0)
    k = max(1, int(np.floor(ci * n_boot)))
    w = boot[k:] - boot[:n_boot - k]
    i = np.argmin(w, axis=0)                             # narrowest 95% window per column
    cols = np.arange(boot.shape[1])
    return boot[i, cols], boot[i + k, cols]


def agg_curve(trials, key):
    """trials: list of vsa_raw dicts. key in {'bundle_cap','role_filler'}.
    returns (ks, mean_acc, lo, hi) with lo/hi the 95% HDI of the mean."""
    ks = trials[0][key]["k"]
    accs = np.array([t[key]["accuracy"] for t in trials])
    lo, hi = hdi_band(accs)
    return np.asarray(ks), accs.mean(0), lo, hi


def agg_self_binding(trials, sb_key):
    k_vals = trials[0][sb_key]["k_values"]
    sims = np.array([t[sb_key]["k_sims"] for t in trials])
    lo, hi = hdi_band(sims)
    return np.asarray(k_vals), sims.mean(0), lo, hi


def build_refs(latent_dim, bc_k, rf_k, k_max=40, device="cpu"):
    refs = {}
    # random hrr
    ref = F.normalize(torch.randn(1000, latent_dim, device=device), p=2, dim=-1)
    rbc = vsa_bundle_capacity(d=latent_dim, n_items=1000, k_range=bc_k,
                               n_trials=20, normalize=True, device=device, item_memory=ref)
    rrf = vsa_binding_unbinding(d=latent_dim, n_items=1000, k_range=rf_k,
                                 n_trials=20, normalize=True, device=device,
                                 unbind_method="*", item_memory=ref, bind_with_random=True)
    zr = F.normalize(torch.randn(1, latent_dim, device=device), p=2, dim=-1)
    rsims = []
    for m in range(1, k_max + 1):
        cur = zr.clone()
        for _ in range(m): cur = vsa_bind(cur, zr)
        for _ in range(m): cur = vsa_unbind(cur, zr, method="*")
        rsims.append(F.cosine_similarity(cur, zr, dim=-1).mean().item())
    refs["random_hrr"] = {"bundle_cap": rbc, "role_filler": rrf,
                          "self_binding_k_values": list(range(1, k_max + 1)),
                          "self_binding_k_sims": rsims}
    # unitary
    uni = F.normalize(vsa_unitary_init(1000, latent_dim, device=device), p=2, dim=-1)
    ubc = vsa_bundle_capacity(d=latent_dim, n_items=1000, k_range=bc_k,
                               n_trials=20, normalize=True, device=device, item_memory=uni)
    urf = vsa_binding_unbinding(d=latent_dim, n_items=1000, k_range=rf_k,
                                 n_trials=20, normalize=True, device=device,
                                 unbind_method="*", item_memory=uni, bind_with_random=True)
    zu = F.normalize(vsa_unitary_init(1, latent_dim, device=device), p=2, dim=-1)
    usims = []
    for m in range(1, k_max + 1):
        cur = zu.clone()
        for _ in range(m): cur = vsa_bind(cur, zu)
        for _ in range(m): cur = vsa_unbind(cur, zu, method="*")
        usims.append(F.cosine_similarity(cur, zu, dim=-1).mean().item())
    refs["unitary"] = {"bundle_cap": ubc, "role_filler": urf,
                       "self_binding_k_values": list(range(1, k_max + 1)),
                       "self_binding_k_sims": usims}
    return refs


def plot_panel(ax, x, y, lo, hi, dist, **st):
    """plot one curve with a 95% HDI-of-the-mean band. style via st. no y-offsets: data is drawn as is."""
    dy = 0.0
    # filled shapes (circle/square/triangle/diamond) smaller; cross/x kept large
    msize = 6 if st.get("marker") in ("x", "+") else 3
    ax.plot(x, y + dy, markersize=msize, markeredgewidth=1.2, **st)
    if lo is not None and hi is not None and not is_ref(dist):
        ax.fill_between(x, lo + dy, hi + dy, color=st["color"], alpha=0.15, linewidth=0)


def plot_dim(dim, dim_aggs, dataset_name, out_dir):
    """dim_aggs: {dist: {'bc': (k, mu, std), 'rf': (k, mu, std), 'sb': (k, mu, std)}}"""
    # 3-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for dist in ORDER:
        a = dim_aggs.get(dist)
        if a is None:
            continue
        st = style(dist)
        if a.get("bc"): plot_panel(axes[0], *a["bc"], dist, **st)
        if a.get("sb"): plot_panel(axes[1], *a["sb"], dist, **st)
        if a.get("rf"): plot_panel(axes[2], *a["rf"], dist, **st)
    for ax, (xl, yl, ttl, ylim) in zip(axes, [
        ("Number of Bundled Vectors ($k$)", "Retrieval Accuracy", f"Bundle Capacity ($d={dim}$)", (0, 1.05)),
        ("Number of Recursive Bind-Unbind Cycles ($m$)", "Cosine Similarity to Original", f"Invertible Self-Binding ($d={dim}$)", (-0.1, 1.05)),
        ("Number of Bundled Role-Filler Pairs ($k$)", "Unbinding Accuracy", f"Role-Filler Capacity ($d={dim}$)", (0, 1.05)),
    ]):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(alpha=0.25, linewidth=0.6); ax.set_ylim(*ylim)
    # single shared legend below the panels (cleaner than three per-axis legends)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    title = DATASET_TITLE.get(dataset_name, dataset_name)
    fig.suptitle(f"{title} --- VSA Operation Benchmarks ($d={dim}$)", fontsize=15)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    p = os.path.join(out_dir, f"vsa_comparison_d{dim}.png")
    plt.savefig(p.replace('.png', '.pdf'), bbox_inches="tight"); plt.close()
    print(f"  -> {p.replace('.png', '.pdf')}")

    # individual panels
    for tag, ax_setup, key in [
        ("bundle_capacity", ("Number of Bundled Vectors ($k$)", "Retrieval Accuracy", f"Bundle Capacity ($d={dim}$)", (0, 1.05)), "bc"),
        ("role_filler", ("Number of Bundled Role-Filler Pairs ($k$)", "Unbinding Accuracy", f"Role-Filler Capacity ($d={dim}$)", (0, 1.05)), "rf"),
        ("self_binding", ("Number of Recursive Bind-Unbind Cycles ($m$)", "Cosine Similarity to Original", f"Invertible Self-Binding ($d={dim}$)", (-0.1, 1.05)), "sb"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for dist in ORDER:
            a = dim_aggs.get(dist)
            if a is None or not a.get(key):
                continue
            plot_panel(ax, *a[key], dist, **style(dist))
        xl, yl, ttl, ylim = ax_setup
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(*ylim)
        plt.tight_layout()
        p = os.path.join(out_dir, f"{tag}_d{dim}.png")
        plt.savefig(p.replace('.png', '.pdf')); plt.close()
        print(f"  -> {p.replace('.png', '.pdf')}")


def hdi_mean(vals, ci=0.95, n_boot=20000, seed=0):
    """95% HDI of the mean of a 1d list of per-run values (bootstrap, seed 0)."""
    x = np.asarray(vals, float)
    if len(x) < 2:
        m = float(x[0]) if len(x) else np.nan
        return m, m
    lo, hi = hdi_band(x[:, None], ci, n_boot, seed)
    return float(lo[0]), float(hi[0])


def _yerr(mus, los, his):
    """asymmetric yerr (2,N) for errorbar from mean/lo/hi arrays."""
    mus, los, his = map(np.asarray, (mus, los, his))
    return np.vstack([np.clip(mus - los, 0, None), np.clip(his - mus, 0, None)])


def plot_knn(met_by, dataset_name, out_dir):
    """one figure with 3 subplots (knn@100, knn@600, knn@1000), x=latent dim, y=mean ± std."""
    keys = ["knn_acc_100", "knn_acc_600", "knn_acc_1000"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # build {dist: {dim: [vals]}}
    all_dists = set()
    by_dist = {}
    for dim, by_d in met_by.items():
        for dist, trials in by_d.items():
            all_dists.add(dist)
            for k in keys:
                vals = [t.get(k) for t in trials if t.get(k) is not None]
                if vals:
                    by_dist.setdefault(dist, {}).setdefault(k, {}).setdefault(dim, []).extend(vals)

    for ax, k in zip(axes, keys):
        for dist in [d for d in ORDER if d in all_dists]:
            if dist not in by_dist or k not in by_dist[dist]:
                continue
            dims = sorted(by_dist[dist][k].keys())
            mus = [np.mean(by_dist[dist][k][d]) for d in dims]
            hd = [hdi_mean(by_dist[dist][k][d]) for d in dims]
            los, his = [h[0] for h in hd], [h[1] for h in hd]
            st = style(dist)
            ax.errorbar(dims, mus, yerr=_yerr(mus, los, his),
                        markersize=3, markeredgewidth=1.2, capsize=3, **st)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("latent dim $d$")
        ax.set_ylabel("kNN accuracy")
        ax.set_title(f"{k.replace('knn_acc_', 'k=')} train samples")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    fig.suptitle(f"{dataset_name} — kNN classification on latent embeddings", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "knn_accuracy_vs_dim.png")
    plt.savefig(p.replace('.png', '.pdf')); plt.close()
    print(f"  -> {p.replace('.png', '.pdf')}")


def plot_fid(met_by, dataset_name, out_dir):
    """one figure: x=latent dim (log2), y=generation FID mean ± std. lower is better."""
    all_dists = set()
    by_dist = {}
    for dim, by_d in met_by.items():
        for dist, trials in by_d.items():
            vals = [t.get("generation_fid") for t in trials
                    if t.get("generation_fid") is not None
                    and not (isinstance(t.get("generation_fid"), float) and np.isnan(t["generation_fid"]))]
            if vals:
                all_dists.add(dist)
                by_dist.setdefault(dist, {}).setdefault(dim, []).extend(vals)

    if not by_dist:
        print("  (no fid values found)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for dist in [d for d in ORDER if d in all_dists]:
        if dist not in by_dist:
            continue
        dims = sorted(by_dist[dist].keys())
        mus = [np.mean(by_dist[dist][d]) for d in dims]
        hd = [hdi_mean(by_dist[dist][d]) for d in dims]
        los, his = [h[0] for h in hd], [h[1] for h in hd]
        st = style(dist)
        ax.errorbar(dims, mus, yerr=_yerr(mus, los, his),
                    markersize=3, markeredgewidth=1.2, capsize=3, **st)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dim $d$")
    ax.set_ylabel("generation FID (↓ lower is better)")
    ax.set_title(f"{dataset_name} — Generation FID vs latent dim")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "fid_vs_dim.png")
    plt.savefig(p.replace('.png', '.pdf')); plt.close()
    print(f"  -> {p.replace('.png', '.pdf')}")


def main(dataset_name, results_root="results"):
    BC_K = list(range(5, 51, 5))
    RF_K = list(range(2, 21, 2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # write straight into figures/<dataset> so the paper's \includegraphics paths
    # pick up the regenerated pdfs (set CLIFFORD_PLOT_DIR to override).
    out_dir = os.environ.get("CLIFFORD_PLOT_DIR",
                             os.path.join("figures", dataset_name))
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing plots to: {out_dir}/")

    vsa_by, met_by = gather_trials(dataset_name, results_root)
    if not vsa_by:
        print(f"no trial data found in {results_root}/{dataset_name}-*-l1-trial*/")
        return

    for dim, by_dist in sorted(vsa_by.items()):
        print(f"\nd={dim}:")
        dim_aggs = {}
        for dist, trials in by_dist.items():
            print(f"  {dist}: {len(trials)} trials")
            dim_aggs[dist] = {
                "bc": agg_curve(trials, "bundle_cap"),
                "rf": agg_curve(trials, "role_filler"),
                "sb": agg_self_binding(trials, "self_binding_star"),
            }
        refs = build_refs(dim, BC_K, RF_K, device=device)
        for r, rd in refs.items():
            # reference baselines are drawn without a band (lo=hi=None)
            dim_aggs[r] = {
                "bc": (np.asarray(rd["bundle_cap"]["k"]),
                       np.asarray(rd["bundle_cap"]["accuracy"]), None, None),
                "rf": (np.asarray(rd["role_filler"]["k"]),
                       np.asarray(rd["role_filler"]["accuracy"]), None, None),
                "sb": (np.asarray(rd["self_binding_k_values"]),
                       np.asarray(rd["self_binding_k_sims"]), None, None),
            }
        plot_dim(dim, dim_aggs, dataset_name, out_dir)

    if met_by:
        print("\nknn:")
        plot_knn(met_by, dataset_name, out_dir)
        print("fid:")
        plot_fid(met_by, dataset_name, out_dir)
        print("csv tables:")
        dump_csv_tables(met_by, dataset_name, out_dir)


def dump_csv_tables(met_by, dataset_name, out_dir):
    """write knn + fid mean±std tables to csv, one row per (dim, dist)."""
    import csv
    knn_keys = ["knn_acc_100", "knn_acc_600", "knn_acc_1000",
                "knn_f1_100", "knn_f1_600", "knn_f1_1000"]
    knn_rows, fid_rows = [], []
    for dim in sorted(met_by.keys()):
        for dist in [d for d in ORDER if d in met_by[dim]]:
            trials = met_by[dim][dist]
            n = len(trials)
            row = {"dim": dim, "dist": dist, "n_trials": n}
            for k in knn_keys:
                vals = [t[k] for t in trials if t.get(k) is not None]
                if vals:
                    row[f"{k}_mean"] = round(float(np.mean(vals)) * 100, 2)
                    row[f"{k}_std"]  = round(float(np.std(vals))  * 100, 2)
                else:
                    row[f"{k}_mean"] = ""
                    row[f"{k}_std"]  = ""
            knn_rows.append(row)
            fid_vals = [t["generation_fid"] for t in trials
                        if t.get("generation_fid") is not None
                        and not (isinstance(t["generation_fid"], float) and np.isnan(t["generation_fid"]))]
            if fid_vals:
                fid_rows.append({
                    "dim": dim, "dist": dist, "n_trials": len(fid_vals),
                    "fid_mean": round(float(np.mean(fid_vals)), 2),
                    "fid_std":  round(float(np.std(fid_vals)), 2),
                })

    knn_path = os.path.join(out_dir, f"{dataset_name}_knn_table.csv")
    if knn_rows:
        cols = ["dim", "dist", "n_trials"] + [f"{k}_{s}" for k in knn_keys for s in ("mean","std")]
        with open(knn_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(knn_rows)
        print(f"  -> {knn_path}")

    fid_path = os.path.join(out_dir, f"{dataset_name}_fid_table.csv")
    if fid_rows:
        with open(fid_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dim","dist","n_trials","fid_mean","fid_std"])
            w.writeheader(); w.writerows(fid_rows)
        print(f"  -> {fid_path}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "fashionmnist"
    root = sys.argv[2] if len(sys.argv) > 2 else "results"
    main(ds, root)

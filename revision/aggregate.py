"""build every revision table and figure from results_rev/. re-run after adding seeds.

    python revision/aggregate.py            # tables + figures + run report
    python revision/aggregate.py --report   # run counts / failures only

reads <results>/<dataset>/<cell>/seed*/metrics.json. a cell's statistics use every finished
run in it. failed runs are counted and listed, never dropped silently.
outputs: <results>/_aggregate/{report.txt, summary.csv, tables/*.tex, figures/*.pdf}
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revision import common  # noqa: E402

CI = 0.95
N_BOOT = 20000
DISP = {"N": r"$\mathcal{N}$", "N_L2": r"$\mathcal{N}_{L2}$", "S": r"$\mathcal{S}$", "V": r"$\mathcal{V}$",
        "C": r"$\mathcal{C}$ (Ours)"}
PRIOR_ORDER = ["N", "N_L2", "S", "V", "C"]


# ---------------------------------------------------------------- statistics


def hdi(x, ci=CI, n_boot=N_BOOT, seed=0):
    """95% hdi of the mean: bootstrap the mean, take the narrowest interval holding 95%."""
    x = np.asarray(x, float)
    if len(x) < 2:
        m = float(x[0]) if len(x) else np.nan
        return m, m
    boot = np.sort(np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(1))
    k = int(np.floor(ci * n_boot))
    i = int(np.argmin(boot[k:] - boot[: n_boot - k]))
    return float(boot[i]), float(boot[i + k])


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ---------------------------------------------------------------- loading


def load_runs(root):
    """{(dataset, cell): {"done": [metrics...], "failed": [(seed, status)...]}}"""
    runs = defaultdict(lambda: {"done": [], "failed": [], "seeds": []})
    for mp in sorted(glob.glob(os.path.join(root, "*", "*", "seed*", "metrics.json"))):
        seed_dir = os.path.dirname(mp)
        cell_dir = os.path.dirname(seed_dir)
        key = (os.path.basename(os.path.dirname(cell_dir)), os.path.basename(cell_dir))
        try:
            m = json.load(open(mp))
        except Exception:
            m = {"status": "unreadable"}
        seed = int(os.path.basename(seed_dir)[4:])
        if m.get("status") == "done":
            m["_seed"] = seed
            runs[key]["done"].append(m)
            runs[key]["seeds"].append(seed)
        else:
            runs[key]["failed"].append((seed, m.get("status")))
    return runs


def cell_name(prior, d, beta_max=1.0):
    return f"{prior}-d{d}" + ("" if beta_max == 1.0 else f"-b{beta_max:g}")


def knn_path(m, prior, path):
    """N is scored with euclidean k-NN. runs from before that switch stored cosine as primary
    and euclidean under euc_acc_n."""
    if prior == "N" and path.startswith("knn.acc_") and get(m, "knn.metric") == "cosine":
        return path.replace("knn.acc_", "knn.euc_acc_")
    return path


def values(runs, dataset, cell, path):
    out = []
    for m in runs.get((dataset, cell), {"done": []})["done"]:
        v = get(m, knn_path(m, cell.split("-d")[0], path))
        if v is not None and np.isfinite(v):
            out.append(float(v))
    return np.array(out)


# ---------------------------------------------------------------- tables


def mark_cells(samples, higher_better, scale=1.0, nd=1):
    """bold+underline: best mean with an hdi disjoint from every other entry.
    underline: the best and every entry whose hdi overlaps it (a tie). decided on raw
    (unrounded) intervals."""
    stats = {}
    for p, x in samples.items():
        if len(x):
            lo, hi = hdi(x)
            stats[p] = (x.mean() * scale, lo * scale, hi * scale, len(x))
    if not stats:
        return {p: r"--" for p in samples}
    best = (max if higher_better else min)(stats, key=lambda p: stats[p][0])
    tied = {p for p in stats if p != best and not (stats[p][2] < stats[best][1] or stats[best][2] < stats[p][1])}
    out = {}
    for p in samples:
        if p not in stats:
            out[p] = r"\textit{n.c.}"
            continue
        m, lo, hi, n = stats[p]
        s = f"{m:.{nd}f}"
        if tied and (p == best or p in tied):
            s = rf"\underline{{{s}}}"
        elif not tied and p == best:
            s = rf"\textbf{{\underline{{{s}}}}}"
        out[p] = rf"\shortstack{{{s}\\{{\tiny[{lo:.{nd}f},\,{hi:.{nd}f}]}}}}"
    return out


def knn_table(runs, dataset, dims, priors, caption, label):
    budgets = [100, 600, 1000]
    nd = len(priors)
    L = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l|" + "|".join(["c" * nd] * 3) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{{nd}}}{{c{'|' if i < 2 else ''}}}{{$n_\ell={b}$}}"
                            for i, b in enumerate(budgets)) + r" \\",
         " & " + " & ".join(" & ".join(DISP[p] for p in priors) for _ in budgets) + r" \\", r"\midrule"]
    for d in dims:
        row = []
        for b in budgets:
            c = mark_cells({p: values(runs, dataset, cell_name(p, d), f"knn.acc_{b}") for p in priors}, True, 100)
            row += [c[p] for p in priors]
        L.append(rf"$d={d}$ & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(L)


def simple_table(runs, dataset, dims, priors, path, higher_better, caption, label, scale=1.0, nd=1):
    L = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{l" + "c" * len(priors) + "}", r"\toprule",
         " & " + " & ".join(DISP[p] for p in priors) + r" \\", r"\midrule"]
    for d in dims:
        c = mark_cells({p: values(runs, dataset, cell_name(p, d), path) for p in priors}, higher_better, scale, nd)
        L.append(rf"$d={d}$ & " + " & ".join(c[p] for p in priors) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def elbo_table(runs, dims, priors):
    L = [r"\begin{table}[H]", r"\centering", r"\caption{ELBO on MNIST (test set).}", r"\label{tab:mnist_elbo}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l|" + "|".join(["ccc"] * len(priors)) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DISP[p]}}}" for p in priors) + r" \\",
         " & " + " & ".join("Recon & KL & $-$ELBO" for _ in priors) + r" \\", r"\midrule"]
    for d in dims:
        neg = mark_cells({p: values(runs, "mnist", cell_name(p, d), "test_elbo.neg_elbo") for p in priors}, False)
        row = []
        for p in priors:
            for part in ("recon", "kl"):
                x = values(runs, "mnist", cell_name(p, d), f"test_elbo.{part}")
                if len(x):
                    lo, hi = hdi(x)
                    row.append(rf"\shortstack{{{x.mean():.1f}\\{{\tiny[{lo:.1f},\,{hi:.1f}]}}}}")
                else:
                    row.append(r"\textit{n.c.}")
            row.append(neg[p])
        L.append(rf"$d={d}$ & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(L)


def composition_table(runs, priors):
    cols = [("seen.recon_mae", "MAE seen", False, 1000), ("unseen.recon_mae", "MAE unseen", False, 1000),
            ("recon_gap", "gap", False, 1), ("unseen.both_acc", "attr. acc. unseen", True, 100)]
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{Shapes2D held-out shape--colour pairs. MAE $\times 10^3$ per pixel on $[0,1]$; gap = unseen/seen MAE; "
         r"attribute accuracy = shape and colour of the reconstruction both correct (\%).}",
         r"\label{tab:shapes2d}", r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
         r"$\beta_{\max}$ & & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for b in (1.0, 4.0):
        marks = {path: mark_cells({p: values(runs, "shapes2d", cell_name(p, 128, b), f"composition.{path}")
                                   for p in priors}, hb, sc, 2 if sc == 1 else 1) for path, _, hb, sc in cols}
        for p in priors:
            L.append(f"{b:g} & {DISP[p]} & " + " & ".join(marks[path][p] for path, *_ in cols) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def cost_table(runs, dataset, dims, priors):
    L = [r"\begin{table}[H]", r"\centering",
         rf"\caption{{Training and inference cost on {dataset} (mean over runs).}}", rf"\label{{tab:cost_{dataset}}}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"$d$ & & params (M) & s / epoch & epochs & peak mem (MB) & encode ms / 1k \\", r"\midrule"]
    for d in dims:
        for p in priors:
            c = cell_name(p, d)
            f = lambda k: values(runs, dataset, c, f"cost.{k}")
            vals = [f("n_params") / 1e6, f("mean_epoch_time_s"), f("epochs_run"), f("peak_mem_mb"), f("encode_ms_per_1k")]
            if not len(vals[0]):
                continue
            L.append(rf"{d} & {DISP[p]} & " + " & ".join(f"{v.mean():.2f}" if len(v) else "--" for v in vals) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ---------------------------------------------------------------- vsa figures


REF_SEEDS = list(range(10))


def references(dims, cache_path):
    """random hrr / unitary references at every item-memory dimension that occurs, 10 seeds."""
    from utils.vsa_schlegel import reference_results

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for D in sorted(dims):
        for kind in ("hrr", "unitary"):
            key = f"{kind}-{D}"
            if key not in cache:
                cache[key] = [reference_results(kind, D, seed=s) for s in REF_SEEDS]
                common.write_json(cache_path, cache)
    return cache


def vsa_figure(runs, dataset, d, priors, refs, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("bundle_capacity", "k", "accuracy", "bundle capacity", "k bundled"),
              ("unbinding_depth_inv", "m", "similarity", "unbinding depth (*)", "m bindings"),
              ("role_filler", "k", "accuracy", "role-filler, image roles", "k pairs"),
              ("symbolic_role_filler", "k", "accuracy", "role-filler, random roles", "k pairs")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    colors = {"N": "C0", "N_L2": "C1", "S": "C2", "V": "C4", "C": "C3"}
    # every prior at the same latent d, as in the paper. clifford with d angles is a length-2d
    # vector, so the random baseline is unitary at length 2d (same vector length as clifford).
    for ax, (name, xk, yk, title, xl) in zip(axes, panels):
        D_ref = None
        for p in priors:
            ms = runs.get((dataset, cell_name(p, d)), {"done": []})["done"]
            curves = [m["vsa"][name] for m in ms if "vsa" in m and name in m["vsa"]]
            if not curves:
                continue
            if p == "C":
                D_ref = ms[0]["vsa"]["dim"]
            x = np.array(curves[0][xk])
            Y = np.array([c[yk] for c in curves])
            mean = Y.mean(0)
            band = np.array([hdi(Y[:, j]) for j in range(Y.shape[1])]) if len(Y) > 1 else None
            ax.plot(x, mean, color=colors[p], label=f"{DISP[p]} d={d}")
            if band is not None:
                ax.fill_between(x, band[:, 0], band[:, 1], color=colors[p], alpha=0.15, lw=0)
        rs = refs.get(f"unitary-{D_ref}") if D_ref else None
        if rs:
            x = np.array(rs[0][name][xk])
            Y = np.array([r[name][yk] for r in rs])
            ax.plot(x, Y.mean(0), color="k", ls="--", lw=1, label=f"random unitary (length {D_ref})")
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("accuracy / similarity")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report


def report(runs, out_path):
    lines = [f"{'dataset':14s} {'cell':18s} done failed seeds"]
    for (ds, cell), r in sorted(runs.items()):
        fails = ",".join(f"s{s}:{st}" for s, st in r["failed"])
        lines.append(f"{ds:14s} {cell:18s} {len(r['done']):4d} {len(r['failed']):6d} "
                     f"{','.join(map(str, sorted(r['seeds'])))} {fails}")
    txt = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=common.RESULTS_ROOT)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    runs = load_runs(a.root)
    out = os.path.join(a.root, "_aggregate")
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    print(report(runs, os.path.join(out, "report.txt")))
    if a.report:
        return

    cnn_dims = [128, 256, 512, 1024, 2048, 4096]
    mn_dims = [2, 5, 10, 20, 40, 128]
    present = {ds for ds, _ in runs}
    tables = {}
    for ds, dims in [("cifar10", cnn_dims), ("fashionmnist", cnn_dims), ("stl10", [128, 256])]:
        if ds not in present:
            continue
        pri = ["N", "N_L2", "S", "C"]
        tables[f"{ds}_knn"] = knn_table(runs, ds, dims, pri, f"$k$-NN accuracy (\\%) on {ds}.", f"tab:{ds}_knn")
        tables[f"{ds}_fid"] = simple_table(runs, ds, dims, pri, "fid.fid", False, f"Generation FID on {ds}.", f"tab:{ds}_fid")
        tables[f"{ds}_cost"] = cost_table(runs, ds, dims, pri)
    if "mnist" in present:
        pri = ["N", "N_L2", "S", "V", "C"]
        tables["mnist_knn"] = knn_table(runs, "mnist", mn_dims, pri, "$k$-NN accuracy (\\%) on MNIST.", "tab:mnist_knn")
        tables["mnist_elbo"] = elbo_table(runs, mn_dims, pri)
    if "shapes2d" in present:
        tables["shapes2d"] = composition_table(runs, ["N", "N_L2", "S", "C"])
    for k, t in tables.items():
        with open(os.path.join(out, "tables", f"{k}.tex"), "w") as f:
            f.write(t + "\n")

    dims_needed = set()
    for (ds, cell), r in runs.items():
        for m in r["done"]:
            if "vsa" in m:
                dims_needed.add(m["vsa"]["dim"])
    refs = references(dims_needed, os.path.join(out, "vsa_references.json")) if dims_needed else {}
    for ds, dims in [("cifar10", cnn_dims), ("fashionmnist", cnn_dims), ("stl10", [128, 256]), ("mnist", mn_dims)]:
        if ds not in present:
            continue
        pri = ["N", "N_L2", "S", "V", "C"] if ds == "mnist" else ["N", "N_L2", "S", "C"]
        for d in dims:
            vsa_figure(runs, ds, d, pri, refs, os.path.join(out, "figures", f"vsa_{ds}_d{d}.pdf"))
    print(f"wrote {len(tables)} tables to {out}/tables, figures to {out}/figures")


if __name__ == "__main__":
    main()

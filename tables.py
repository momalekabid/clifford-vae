"""every latex table of the paper from the runs' metrics.json (no models, no gpu). mean with the
95% hdi of the mean over seeds; bold + underline = clear winner, underline = tied with the best.

    python tables.py                       # runs/ -> runs/_aggregate/tables/*.tex
    python tables.py --root results_rev
"""
import argparse
import os

import numpy as np

from clifford_vae import common
from clifford_vae.stats import TEX, cell, hdi, load_runs, mark, values
from clifford_vae.vsa import K_RANGE

CNN_DIMS = [128, 256, 512, 1024, 2048, 4096]
MNIST_DIMS = [2, 5, 10, 20, 40, 128]
CELEBA_DIMS = [128, 256, 512]
CNN_PRIORS = ["N", "N_L2", "S", "C"]
MNIST_PRIORS = ["N", "N_L2", "S", "V", "C"]


def _wrap(caption, label, colspec, header, rows, resize=True):
    L = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    L += [r"\resizebox{\textwidth}{!}{%"] if resize else []
    L += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    L += ["%", "}"] if resize else []
    return "\n".join(L + [r"\end{table}"])


def knn(runs, ds, dims, priors, n_runs):
    budgets = [100, 600, 1000]
    nd = len(priors)
    what = "mean accuracy over the 40 attributes" if ds == "celeba" else "accuracy"
    header = [" & " + " & ".join(rf"\multicolumn{{{nd}}}{{c{'|' if i < 2 else ''}}}{{$n_{{\ell}}={b}$}}"
                                 for i, b in enumerate(budgets)) + r" \\",
              " & " + " & ".join(" & ".join(TEX[p] for p in priors) for _ in budgets) + r" \\"]
    rows = []
    for d in dims:
        row = []
        for b in budgets:
            c = mark({p: values(runs, ds, cell(p, d), f"knn.acc_{b}") for p in priors}, True, 100)
            row += [c[p] for p in priors]
        rows.append(rf"$d={d}$ & " + " & ".join(row) + r" \\")
    cap = (rf"Semi-supervised $k$-NN {what} (\%) on {ds} ({n_runs} runs), mean with the 95\% HDI of the mean "
           r"(bootstrap, 20k resamples). $k=5$ neighbours fit on $n_\ell$ labelled training codes, scored on the full "
           r"test set; euclidean distance for $\mathcal{N}$, cosine for the other priors.")
    return _wrap(cap, f"tab:{ds}_knn", "l|" + "|".join(["c" * nd] * 3), header, rows)


def scalar(runs, ds, dims, priors, path, higher, what, label, n_runs, scale=1.0, nd=1):
    header = [" & " + " & ".join(TEX[p] for p in priors) + r" \\"]
    rows = []
    for d in dims:
        c = mark({p: values(runs, ds, cell(p, d), path) for p in priors}, higher, scale, nd)
        rows.append(rf"$d={d}$ & " + " & ".join(c[p] for p in priors) + r" \\")
    cap = rf"{what} on {ds} ({n_runs} runs), mean with the 95\% HDI of the mean (bootstrap, 20k resamples)."
    return _wrap(cap, label, "l" + "c" * len(priors), header, rows, resize=False)


def by_cell(runs, ds, cells, cols, what, label):
    """rows = cells (prior, d, beta_max), columns = (header, metric path, higher is better, scale, digits).
    marks compare the rows within each column."""
    header = [" & " + " & ".join(c[0] for c in cols) + r" \\"]
    names = [cell(p, d, b) for p, d, b in cells]
    marks = [mark({n: values(runs, ds, n, path) for n in names}, hi, sc, nd) for _, path, hi, sc, nd in cols]
    rows = []
    for (p, d, b), n in zip(cells, names):
        lab = rf"{TEX[p]} $d{{=}}{d}$" + (rf", $\beta{{=}}{b:g}$" if b != 1.0 else "")
        rows.append(lab + " & " + " & ".join(m[n] for m in marks) + r" \\")
    n_runs = max((len(runs.get((ds, n), [])) for n in names), default=0)
    cap = rf"{what} ({n_runs} runs), mean with the 95\% HDI of the mean (bootstrap, 20k resamples)."
    return _wrap(cap, label, "l" + "c" * len(cols), header, rows)


def elbo(runs, ds, dims, priors, n_runs):
    header = [" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{TEX[p]}}}" for p in priors) + r" \\",
              " & " + " & ".join("Recon & KL & $-$ELBO" for _ in priors) + r" \\"]
    rows = []
    for d in dims:
        neg = mark({p: values(runs, ds, cell(p, d), "test_elbo.neg_elbo") for p in priors}, False)
        row = []
        for p in priors:
            for part in ("recon", "kl"):
                x = values(runs, ds, cell(p, d), f"test_elbo.{part}")
                lo, hi = hdi(x) if len(x) else (0, 0)
                row.append(rf"\shortstack{{{x.mean():.1f}\\{{\tiny[{lo:.1f},\,{hi:.1f}]}}}}" if len(x) else r"\textit{n.c.}")
            row.append(neg[p])
        rows.append(rf"$d={d}$ & " + " & ".join(row) + r" \\")
    cap = (rf"ELBO on {ds} ({n_runs} runs), mean with the 95\% HDI of the mean. Recon $=-\log p(\mathbf{{x}}|\mathbf{{z}})$, "
           r"KL against the prior, $-$ELBO = Recon + KL ($\downarrow$).")
    return _wrap(cap, f"tab:{ds}_elbo", "l|" + "|".join(["ccc"] * len(priors)), header, rows)


def vsa_points(runs, ds, dims, priors, refs, tests, n_runs, label, what):
    """one column per (test, x), rows per d and prior, plus the random unitary baseline at clifford's length."""
    header = ["$d$ & & " + " & ".join(t[2] for t in tests) + r" \\"]
    rows = []
    for d in dims:
        marks = []
        for name, x, _ in tests:
            xk = "m" if name.startswith("unbinding") else "k"
            yk = "similarity" if name.startswith("unbinding") else "accuracy"
            samp = {}
            for p in priors:
                v = [common.get(m, f"vsa.{name}") for m in runs.get((ds, cell(p, d)), [])]
                samp[p] = np.array([r[yk][r[xk].index(x)] for r in v if r and x in r[xk]])
            marks.append(mark(samp, True, 1, 2))
        for i, p in enumerate(priors):
            rows.append((f"{d}" if i == 0 else "") + f" & {TEX[p]} & " + " & ".join(c[p] for c in marks) + r" \\")
        cl = runs.get((ds, cell("C", d)), [])
        D = common.get(cl[0], "vsa.dim") if cl else 2 * d
        ref = refs.get(str(D))
        if ref:
            cells = []
            for name, x, _ in tests:
                xk = "m" if name.startswith("unbinding") else "k"
                yk = "similarity" if xk == "m" else "accuracy"
                r = ref[name]
                cells.append(f"{r[yk][r[xk].index(x)]:.2f}")
            rows.append(rf" & {{\footnotesize random unitary, length {D}}} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows.pop()
    cap = (rf"{what} on {ds} ({n_runs} runs), every prior at latent dimension $d$, mean with the 95\% HDI of the mean. "
           r"Clifford with $d$ angles is a length-$2d$ unitary vector, so the random baseline is unitary at length $2d$.")
    return _wrap(cap, label, "ll" + "c" * len(tests), header, rows)


def cost(runs, ds, dims, priors):
    cols = [("cost.n_params", 1e-6, "params (M)", 2), ("cost.mean_epoch_time_s", 1, "s / epoch", 1),
            ("cost.epochs_run", 1, "epochs", 0), ("cost.peak_mem_mb", 1, "peak mem (MB)", 0),
            ("latency.encode_ms_per_1k", 1, "encode ms / 1k", 2)]
    header = ["$d$ & & " + " & ".join(c[2] for c in cols) + r" \\"]
    rows = []
    for d in dims:
        for i, p in enumerate(priors):
            cells = []
            for path, sc, _, nd in cols:
                x = values(runs, ds, cell(p, d), path)
                if not len(x) and path.startswith("latency."):
                    x = values(runs, ds, cell(p, d), path.replace("latency.", "cost."))  # older runs
                cells.append(f"{x.mean() * sc:.{nd}f}" if len(x) else "--")
            rows.append((f"{d}" if i == 0 else "") + f" & {TEX[p]} & " + " & ".join(cells) + r" \\")
    cap = (rf"Training and inference cost on {ds} (mean over runs). Times from the sweep, where runs shared a gpu; "
           r"see timing.py for the clean per-cell numbers.")
    return _wrap(cap, f"tab:{ds}_cost", "ll" + "r" * len(cols), header, rows, resize=False)


def timing(path, ds, dims, priors):
    t = common.read_json(path)
    if not t:
        return None
    header = ["$d$ & & s / epoch & peak mem (GB) & encode ms / 1k & decode ms / 1k \\\\"]
    rows = []
    for d in dims:
        for i, p in enumerate(priors):
            r = t.get(f"{ds}/{cell(p, d)}")
            if r:
                rows.append((f"{d}" if i == 0 else "") + f" & {TEX[p]} & {r['s_per_epoch']:.1f} & "
                            f"{r['peak_mem_mb'] / 1024:.2f} & {r['encode_ms_per_1k']:.2f} & {r['decode_ms_per_1k']:.2f} \\\\")
    cap = rf"Clean training and inference cost on {ds}, one run at a time on an idle {t.get('_gpu', 'gpu')}."
    return _wrap(cap, f"tab:{ds}_timing", "llrrrr", header, rows, resize=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=common.RESULTS)
    a = p.parse_args()
    runs = load_runs(a.root)
    agg = os.path.join(a.root, "_aggregate")
    os.makedirs(os.path.join(agg, "tables"), exist_ok=True)
    refs = common.read_json(os.path.join(agg, "references.json"), {})  # written by figures.py
    present = {ds for ds, _ in runs}
    n = lambda ds: max((len(v) for (d, _), v in runs.items() if d == ds), default=0)  # noqa: E731
    main_tests = [("bundle_capacity", 10, "bundle $k{=}10$"), ("unbinding_depth_inv", 10, "depth $m{=}10$"),
                  ("role_filler", 10, "RF image $k{=}10$"), ("role_filler_random_roles", 20, "RF rand.\\ roles $k{=}20$"),
                  ("role_filler_random_fillers", 20, "RF rand.\\ fillers $k{=}20$")]
    two = [("two_bundle", k, f"$k{{=}}{k}$") for k in (10, 20, 30, 40, 50) if k in K_RANGE]
    out = {}
    for ds, dims, priors in [("cifar10", CNN_DIMS, CNN_PRIORS), ("fashionmnist", CNN_DIMS, CNN_PRIORS),
                             ("celeba", CELEBA_DIMS, CNN_PRIORS), ("mnist", MNIST_DIMS, MNIST_PRIORS)]:
        if ds not in present:
            continue
        out[f"{ds}_knn"] = knn(runs, ds, dims, priors, n(ds))
        if ds == "mnist":
            out[f"{ds}_elbo"] = elbo(runs, ds, dims, priors, n(ds))
        else:
            out[f"{ds}_fid"] = scalar(runs, ds, dims, priors, "fid.fid", False, r"Generation FID ($\downarrow$)",
                                      f"tab:{ds}_fid", n(ds))
            out[f"{ds}_cost"] = cost(runs, ds, dims, priors)
            t = timing(os.path.join(agg, "timing.json"), ds, dims, priors)
            if t:
                out[f"{ds}_timing"] = t
        out[f"{ds}_vsa"] = vsa_points(runs, ds, dims, priors, refs, main_tests, n(ds), f"tab:{ds}_vsa", "VSA benchmarks")
        out[f"{ds}_two_bundle"] = vsa_points(runs, ds, dims, priors, refs, two, n(ds), f"tab:{ds}_two_bundle",
                                             "Appendix: the submission's two-bundle test (chance 0.5)")
    betas = (1.0, 4.0)
    if "shapes3d" in present:
        cells = [(p, 128, b) for b in betas for p in ("N", "N_L2", "S", "C")]
        c = "composition."
        out["shapes3d_composition"] = by_cell(runs, "shapes3d", cells, [
            ("MAE seen", c + "seen.recon_mae", False, 100, 2), ("MAE unseen", c + "unseen.recon_mae", False, 100, 2),
            ("attr.\\ seen", c + "seen.attr_acc", True, 100, 1), ("attr.\\ unseen", c + "unseen.attr_acc", True, 100, 1),
            ("probe hue", c + "probe_acc_hue_unseen", True, 100, 1), ("probe shape", c + "probe_acc_shape_unseen", True, 100, 1),
            ("HRR analogy", c + "analogy.hrr", True, 100, 1), ("vector analogy", c + "analogy.vector", True, 100, 1)],
            "3D Shapes, held-out object-hue $\\times$ shape pairs: reconstruction MAE ($\\times 100$), hue and shape of "
            "the reconstruction both right (\\%), linear probe fit on seen pairs and scored on unseen ones (\\%), and "
            "unseen pairs built by HRR or vector-arithmetic analogies (\\%)", "tab:shapes3d_composition")
    if "omniglot" in present:
        lengths = {"N": 512, "N_L2": 512, "C": 256}
        cells = [(p, lengths[p], b) for b in betas for p in ("N", "N_L2", "C")]
        fs = []
        for m, n in ((5, 1), (5, 5), (20, 1), (20, 5), (100, 5)):
            fs.append((f"{m}-way {n}-shot", f"fewshot.{m}way{n}shot", True, 100, 1))
        out["omniglot_fewshot"] = by_cell(runs, "omniglot", cells, fs,
            "Omniglot few-shot classification on the evaluation alphabets with an HRR memory: every support code bound "
            "to a random unitary label for its class, all pairs bundled into one vector, each query unbound with its "
            "own code and cleaned up against the labels (every code 512 long), accuracy (\\%)",
            "tab:omniglot_fewshot")
        a = "composition.analogy."
        out["omniglot_composition"] = by_cell(runs, "omniglot", cells, [
            (f"{k} {s}", a + f"{k}_{s}", s != "mae", 100, 1 if s == "char_acc" else 2)
            for k in ("hrr", "vector", "copy") for s in ("cos", "mae", "char_acc")],
            "Omniglot held-out character $\\times$ drawer pairs built from three seen drawings: HRR, vector "
            "arithmetic, and copying the same character by another drawer. cos to the target's code, image MAE "
            "($\\times 100$), character accuracy of the decoded image (\\%)", "tab:omniglot_composition")
    for k, t in out.items():
        with open(os.path.join(agg, "tables", f"{k}.tex"), "w") as f:
            f.write(t + "\n")
    print(f"wrote {len(out)} tables to {agg}/tables")


if __name__ == "__main__":
    main()

# figures and tables from factor_suite.py output: cosine matrices, factor-pair matrices, the per-value transform
# table (binding and vector, mean with hdi over runs), and the factor-move images with correct / wrong marks.
#   python analysis/shapes3d_factor_figures.py <suite prefix> <figure dir> <table dir> [images.npz]   (from the repo root)
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import figures  # paper fonts and style
from clifford_vae.stats import COLOR, hdi, mark

PRE, FIG, TAB = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(PRE + ".json")); runs = r["runs"]; NAMES = r["names"]
PRI = [("C", "Clifford (Ours)"), ("N", "Gaussian"), ("N_L2", "Gaussian (L2)"), ("S", r"$\mathcal{S}$-PWS")]
of = lambda p: [v for k, v in runs.items() if k.split("|")[0] == p]  # noqa: E731
nruns = {p: len(of(p)) for p, _ in PRI}
print("runs per prior:", nruns)
def stat(vals, scale=1.0):
    v = np.array(vals) * scale
    lo, hi = hdi(v) if len(v) > 1 else (v[0], v[0])
    return v.mean(), lo, hi

# ---- cosine matrices (mean over runs)
KEYS = ["a", "b", "c", "target", "binding", "vector"]
B, Vv = r"$\mathbf{a}\circledast\mathbf{c}\circledast\mathbf{b}^{*}$", r"$\mathbf{a}-\mathbf{b}+\mathbf{c}$"
YL = [r"$\mathbf{a}$", r"$\mathbf{b}$", r"$\mathbf{c}$", "target", "binding\n" + B, "vector\n" + Vv]
XL = [r"$\mathbf{a}$", r"$\mathbf{b}$", r"$\mathbf{c}$", "target", B, Vv]
fig, axs = plt.subplots(2, 2, figsize=(12.4, 10.9))
for ax, (p, title) in zip(axs.ravel(), PRI):
    m = np.eye(6)
    for i, a in enumerate(KEYS):
        for j, b in enumerate(KEYS):
            if i < j: m[i, j] = m[j, i] = np.mean([x["cosine"][f"{a}|{b}"] for x in of(p)])
    im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=12.5, color="white" if m[i, j] > 0.6 else "0.15")
    ax.set_title(title, color=COLOR[p], fontsize=16, pad=8)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(XL, fontsize=11.5, rotation=30, ha="right", rotation_mode="anchor"); ax.set_yticklabels(YL, fontsize=11.5)
    ax.tick_params(length=0); ax.axhline(2.5, color="k", lw=1.0); ax.axvline(2.5, color="k", lw=1.0)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.8)
fig.tight_layout(w_pad=2.0, h_pad=2.0)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Mean cosine similarity", fontsize=13)
fig.savefig(os.path.join(FIG, "shapes3d_cosine.pdf"), bbox_inches="tight"); fig.savefig(PRE + "_cosine.png", dpi=95, bbox_inches="tight"); plt.close(fig)

# ---- factor-pair matrices (mean over runs)
fig, axs = plt.subplots(1, 4, figsize=(17.5, 5.1))
for ax, (p, title) in zip(axs, PRI):
    m = np.full((6, 6), np.nan)
    for i in range(6):
        for j in range(i + 1, 6):
            k = f"{NAMES[i]} x {NAMES[j]}"
            m[i, j], m[j, i] = np.mean([x["pairs"][k]["bind"] for x in of(p)]), np.mean([x["pairs"][k]["vector"] for x in of(p)])
    im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
    for i in range(6):
        for j in range(6):
            if i != j: ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=12, color="white" if m[i, j] > 0.6 else "0.15")
    ax.plot([-0.5, 5.5], [-0.5, 5.5], color="0.4", lw=1.0)
    ax.set_title(title, color=COLOR[p], fontsize=15, pad=8)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(NAMES, fontsize=11, rotation=35, ha="right", rotation_mode="anchor"); ax.set_yticklabels(NAMES if ax is axs[0] else [], fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.8)
fig.suptitle(r"Above the diagonal: binding analogy $\mathbf{a}\circledast\mathbf{c}\circledast\mathbf{b}^{*}$      Below the diagonal: vector analogy $\mathbf{a}-\mathbf{b}+\mathbf{c}$      (decoded images with all six factors correct, %)", fontsize=12.5, y=1.02)
fig.tight_layout(w_pad=1.2)
cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.012); cb.set_label("Decoded images correct", fontsize=12)
fig.savefig(os.path.join(FIG, "shapes3d_pairs.pdf"), bbox_inches="tight"); fig.savefig(PRE + "_pairs.png", dpi=80, bbox_inches="tight"); plt.close(fig)

# ---- table: one transform per factor value, binding and vector, mean with hdi
TEXP = [("N", r"$\mathcal{N}$"), ("N_L2", r"$\mathcal{N}_{L2}$"), ("S", r"$\mathcal{S}$"), ("C", r"$\mathcal{C}$ (Ours)")]
facs = ["orientation", "scale", "floor hue", "wall hue"]
cell = lambda v: (lambda m, lo, hi: rf"\shortstack{{{m:.1f}\\{{\tiny[{lo:.1f},\,{hi:.1f}]}}}}")(*stat(v, 100))  # noqa: E731
n = max(nruns.values())
L = [r"\begin{table}[H]", r"\centering",
     rf"\caption{{\rev{{3D Shapes: one fixed transform per factor value. For each value of a factor, a transform is estimated on $300$ scenes: by binding ($\circledast$), the mean of $\mathbf{{z}}_v\circledast\mathbf{{z}}_0^{{*}}$ (for the Clifford-VAE its phases), and by vector arithmetic ($+$), the mean of $\mathbf{{z}}_v-\mathbf{{z}}_0$. An image is moved from one value to another by applying the two transforms and decoding; the table gives the decoded images with all six factors correct (\%, $\uparrow$) on $150$ other scenes, {n} runs, mean with the 95\% HDI. In each row the best mean is in bold and underlined if its HDI is disjoint from all others; entries whose HDIs overlap the best are underlined. Object hue and shape are omitted because no scene has all their values in the training set.}}}}",
     r"\label{tab:shapes3d_offsets}", r"\renewcommand{\arraystretch}{1.25}", r"\begin{tabular}{l" + "cc" * 4 + "}", r"\toprule",
     " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for _, t in TEXP) + r" \\", "Factor & " + " & ".join(r"$\circledast$ & $+$" for _ in TEXP) + r" \\", r"\midrule"]
for f in facs:
    # marks as in the other tables: best of the eight cells in a row, ties underlined
    mk = mark({(p, q): np.array([x["offsets"][f][q] for x in of(p)]) for p, _ in TEXP for q in ("bind", "vector")}, True, 100, 1)
    L.append(f.capitalize() + " & " + " & ".join(mk[(p, "bind")] + " & " + mk[(p, "vector")] for p, _ in TEXP) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TAB, "shapes3d_offsets.tex"), "w").write("\n".join(L) + "\n")

# ---- numbers for the text
S = {}
for p, _ in PRI:
    c = {k: stat([x["cosine"][k] for x in of(p)]) for k in ("target|binding", "target|vector", "binding|vector")}
    pb = {k: np.mean([x["pairs"][k]["bind"] for x in of(p)]) for k in of(p)[0]["pairs"]}
    pv = {k: np.mean([x["pairs"][k]["vector"] for x in of(p)]) for k in of(p)[0]["pairs"]}
    S[p] = {"cos": {k: [round(float(a), 3) for a in v] for k, v in c.items()}, "pairs_bind_min": round(float(min(pb.values())), 3), "pairs_bind_max": round(float(max(pb.values())), 3),
            "pairs_bind_ge99": int(sum(v >= 0.99 for v in pb.values())), "pairs_bind_mean": round(float(np.mean(list(pb.values()))), 3), "pairs_vector_mean": round(float(np.mean(list(pv.values()))), 3),
            "pairs_bind": {k: round(float(v), 3) for k, v in pb.items()}, "offsets": {f: {q: [round(float(a), 3) for a in stat([x["offsets"][f][q] for x in of(p)])] for q in ("bind", "vector")} for f in facs}}
json.dump(S, open(PRE + "_summary.json", "w"), indent=1)
for p, _ in PRI:
    print(p, "cos", S[p]["cos"], "| pairs bind mean", S[p]["pairs_bind_mean"], ">=99%:", S[p]["pairs_bind_ge99"], "min", S[p]["pairs_bind_min"], "max", S[p]["pairs_bind_max"], "| vector mean", S[p]["pairs_vector_mean"])
    print("   offsets", {f: (S[p]["offsets"][f]["bind"][0], S[p]["offsets"][f]["vector"][0]) for f in facs})

# ---- factor-move images, one per factor, wrong decodings marked with a small red cross.
# the images come from <prefix>_images.npz, or from a fourth argument (e.g. the best checkpoint of each prior,
# merged by shapes3d_best_images.py)
IMG = sys.argv[4] if len(sys.argv) > 4 else PRE + "_images.npz"
if os.path.exists(IMG):
    c = 0.42
    z = np.load(IMG)
    ROWS = [("C", "Clifford (Ours)"), ("N", "Gaussian"), ("N_L2", "Gaussian (L2)"), ("S", r"$\mathcal{S}$-PWS")]
    for f in ("orientation", "scale", "floor hue", "wall hue"):
        true = z[f"true|{f}"]; src = int(z[f"src|{f}"]); nv = len(true)
        rows = [(true, None, "true images", "k")]
        for p, name in ROWS:
            for kind, lab in (("bind", "binding"), ("vector", "vector")):
                rows.append((z[f"{p}|{f}|{kind}"], z[f"ok|{p}|{f}|{kind}"], f"{name}, {lab}", COLOR[p]))
        fig, axs = plt.subplots(9, nv, figsize=(nv * c, 9 * c))
        for q, (imgs, ok, lab, col) in enumerate(rows):
            for v in range(nv):
                ax = axs[q, v]; ax.axis("off"); ax.imshow(imgs[v].transpose(1, 2, 0), interpolation="nearest")
                if ok is not None and v != src and not ok[v]:
                    ax.text(0.95, 0.03, "×", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#D32F2F", fontweight="bold", fontfamily="DejaVu Sans",
                            path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
                if v == src: ax.add_patch(plt.Rectangle((-0.5, -0.5), 32, 32, fill=False, ec="0.1", lw=1.1, ls=(0, (2, 1.2)), clip_on=False))
            axs[q, 0].text(-0.2, 0.5, lab, transform=axs[q, 0].transAxes, ha="right", va="center", fontsize=8, color=col, fontweight="bold" if q == 0 else "normal")
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        name = os.path.join(FIG, "shapes3d_factor_moves_" + f.replace(" ", "_"))
        fig.savefig(name + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(name + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
        from PIL import Image; print(f, Image.open(name + ".png").size)

print("figures done")

# the main-text figures as journal figure files: no title inside the image, panels lettered.
#   python analysis/journal_figures.py <out dir>   (from the repo root)
import os, sys
sys.path.insert(0, os.getcwd())
import figures
from clifford_vae import common, vsa as V
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
figures.JOURNAL = True
runs = figures.load_runs(common.RESULTS); agg = os.path.join(common.RESULTS, "_aggregate")
refs = figures.references(runs, os.path.join(agg, "references.json"))
figures.HRR_REFS.update(figures.references(runs, os.path.join(agg, "references_hrr.json"), V.reference_hrr))
names = ["bundle_capacity", "unbinding_depth_inv", "role_filler", "role_filler_random_roles"]
figures.vsa_figure(runs, "fashionmnist", 128, figures.CNN_PRIORS, refs, names, os.path.join(out, "vsa_fashionmnist_d128.pdf"))
figures.vsa_figure(runs, "cifar10", 512, figures.CNN_PRIORS, refs, names, os.path.join(out, "vsa_cifar10_d512.pdf"))
present = {ds for ds, _ in runs}
for ds in [x for x in figures.DIMS if x in present]:
    priors = figures.PRIORS.get(ds, figures.CNN_PRIORS)
    for d in figures.DIMS[ds]:
        if any((ds, figures.cell(q, d)) in runs for q in priors):
            figures.vsa_figure(runs, ds, d, priors, refs, names, os.path.join(out, f"vsa_{ds}_d{d}.pdf"))
            figures.vsa_figure(runs, ds, d, priors, refs, ["role_filler", "role_filler_random_fillers"], os.path.join(out, f"rolefiller_{ds}_d{d}.pdf"))
figures.wason_figure(runs, agg, os.path.join(out, "wason.pdf"))
figures.composition_figure(runs, common.RESULTS, os.path.join(out, "shapes3d_composition.pdf"))
figures.unseen_grid_figure(runs, 1, os.path.join(out, "shapes3d_unseen_grid.png"))
print("journal figures in", out)

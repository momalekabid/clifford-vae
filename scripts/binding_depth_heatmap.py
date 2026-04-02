"""
binding depth heatmap: similarity as f(# dimensions, binding depth)
reproduces schlegel et al. fig 6 style for HRR, unitary, and clifford vectors
"""
import sys, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vsa import hrr_init, unitary_init, normalize_vectors, bind, unbind
from scripts.bundle_heatmap import clifford_init


def run_depth_sweep(init_fn, name, dims, max_depth=40, n_trials=10, device="cpu"):
    depths = list(range(1, max_depth + 1))
    sim_matrix = np.full((len(dims), len(depths)), np.nan)

    for i, d in enumerate(dims):
        print(f"  {name} d={d}...")
        for j, m in enumerate(depths):
            trial_sims = []
            for _ in range(n_trials):
                vecs = init_fn(m + 1, d, device=device)
                vecs = normalize_vectors(vecs)
                target = vecs[0:1]
                partners = vecs[1:]
                bound = target.clone()
                for k in range(m):
                    bound = bind(bound, partners[k:k+1])
                recovered = bound.clone()
                for k in range(m - 1, -1, -1):
                    recovered = unbind(recovered, partners[k:k+1])
                sim = torch.nn.functional.cosine_similarity(recovered, target, dim=-1).mean().item()
                trial_sims.append(sim)
            sim_matrix[i, j] = np.mean(trial_sims)

    return sim_matrix, depths


def plot_heatmaps(results, dims, depths, save_path=None):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for ax, (name, sim_matrix) in zip(axes, results.items()):
        im = ax.pcolormesh(np.arange(len(depths) + 1), np.arange(len(dims) + 1),
                           sim_matrix, cmap=plt.cm.jet, norm=norm, shading="flat")
        ax.set_xticks(np.arange(0, len(depths), 5) + 0.5)
        ax.set_xticklabels([depths[i] for i in range(0, len(depths), 5)], fontsize=7)
        ax.set_yticks(np.arange(len(dims)) + 0.5)
        ax.set_yticklabels(dims, fontsize=7)
        ax.set_xlabel("binding depth $m$", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("# dimensions", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), ax=axes, shrink=0.8, label="cosine similarity")
    fig.suptitle("Approximate Inverse Binding Depth", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"saved to {save_path}")
    plt.show()


# also generate 1D curves (schlegel fig 6 style) at a fixed d
def plot_curves(results_at_d, depths, d, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"HRR": "tab:gray", "Unitary (FHRR)": "tab:green", "Clifford": "tab:purple"}
    for name, sims in results_at_d.items():
        ax.plot(depths, sims, "o-", markersize=3, label=name, color=colors.get(name, None))
    ax.set_xlabel("binding depth $m$")
    ax.set_ylabel("cosine similarity to original")
    ax.set_title(f"Approximate Inverse Binding Depth (d={d})")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--max_depth", type=int, default=40)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vsa-heatmaps")
    args = p.parse_args()

    dims = [4, 16, 64, 144, 256, 484, 512, 1024]
    init_fns = {
        "HRR": lambda n, d, device=args.device: hrr_init(n, d, device=device),
        "Unitary (FHRR)": lambda n, d, device=args.device: unitary_init(n, d, device=device),
    }

    results = {}
    for name, fn in init_fns.items():
        print(f"running {name}...")
        mat, depths = run_depth_sweep(fn, name, dims, max_depth=args.max_depth,
                                       n_trials=args.n_trials, device=args.device)
        results[name] = mat

    heatmap_path = "figures/binding_depth_heatmap.png"
    curves_path = "figures/binding_depth_curves_d1024.png"

    plot_heatmaps(results, dims, depths, save_path=heatmap_path)

    # 1D curves at d=1024
    d_idx = dims.index(1024)
    curves = {name: mat[d_idx].tolist() for name, mat in results.items()}
    plot_curves(curves, depths, d=1024, save_path=curves_path)

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name="binding-depth-heatmap",
                         config={"n_trials": args.n_trials, "max_depth": args.max_depth,
                                 "dims": dims})
        wandb.log({"binding_depth_heatmap": wandb.Image(heatmap_path),
                    "binding_depth_curves_d1024": wandb.Image(curves_path)})
        for name, mat in results.items():
            for i, d in enumerate(dims):
                for j, m in enumerate(depths):
                    if not np.isnan(mat[i, j]):
                        wandb.log({f"{name}/d{d}_depth{m}": mat[i, j]})
        wandb.finish()

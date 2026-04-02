"""
role-filler unbinding heatmap: accuracy as f(# dimensions, # bundled pairs)
reproduces schlegel et al. fig 7/8 style for HRR, unitary, and clifford vectors
"""
import sys, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vsa import (hrr_init, unitary_init, normalize_vectors, bind, unbind,
                        bundle, similarity)
from scripts.bundle_heatmap import clifford_init


def run_rolefiller_sweep(init_fn, name, dims, k_range, n_items=1000, n_trials=10, device="cpu"):
    acc_matrix = np.full((len(dims), len(k_range)), np.nan)

    for i, d in enumerate(dims):
        print(f"  {name} d={d}...")
        items = init_fn(n_items, d, device=device)
        items = normalize_vectors(items)

        for j, k in enumerate(k_range):
            if 2 * k > n_items:
                continue
            trial_accs = []
            for _ in range(n_trials):
                idx = torch.randperm(n_items)[:2 * k]
                roles = items[idx[:k]]
                fillers = items[idx[k:2 * k]]
                pairs = bind(roles, fillers)
                bundled = bundle(pairs, normalize=True)
                correct = 0
                for ii in range(k):
                    recovered = unbind(bundled.unsqueeze(0), roles[ii].unsqueeze(0)).squeeze()
                    sims = similarity(recovered, items)
                    if torch.argmax(sims).item() == idx[k + ii].item():
                        correct += 1
                trial_accs.append(correct / k)
            acc_matrix[i, j] = np.mean(trial_accs)

    return acc_matrix


def plot_heatmaps(results, dims, k_range, save_path=None):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for ax, (name, acc_matrix) in zip(axes, results.items()):
        masked = np.ma.masked_invalid(acc_matrix)
        ax.pcolormesh(np.arange(len(k_range) + 1), np.arange(len(dims) + 1),
                      masked, cmap=plt.cm.jet, norm=norm, shading="flat")
        ax.set_xticks(np.arange(len(k_range)) + 0.5)
        ax.set_xticklabels(k_range, rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(dims)) + 0.5)
        ax.set_yticklabels(dims, fontsize=7)
        ax.set_xlabel("# pairs", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("# dimensions", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), ax=axes, shrink=0.8, label="accuracy")
    fig.suptitle("Role-Filler Unbinding Capacity", fontsize=13, fontweight="bold")
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
    p.add_argument("--n_items", type=int, default=1000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vsa-heatmaps")
    args = p.parse_args()

    dims = [4, 16, 64, 144, 256, 484, 512, 1024]
    k_range = list(range(2, 52, 4))

    init_fns = {
        "HRR": lambda n, d, device=args.device: hrr_init(n, d, device=device),
        "Unitary (FHRR)": lambda n, d, device=args.device: unitary_init(n, d, device=device),
    }

    results = {}
    for name, fn in init_fns.items():
        print(f"running {name}...")
        results[name] = run_rolefiller_sweep(fn, name, dims, k_range,
                                              n_items=args.n_items, n_trials=args.n_trials,
                                              device=args.device)

    save_path = "figures/rolefiller_heatmap.png"
    plot_heatmaps(results, dims, k_range, save_path=save_path)

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name="rolefiller-heatmap",
                         config={"n_trials": args.n_trials, "n_items": args.n_items,
                                 "dims": dims, "k_range": k_range})
        wandb.log({"rolefiller_heatmap": wandb.Image(save_path)})
        for name, mat in results.items():
            for i, d in enumerate(dims):
                for j, k in enumerate(k_range):
                    if not np.isnan(mat[i, j]):
                        wandb.log({f"{name}/d{d}_k{k}": mat[i, j]})
        wandb.finish()

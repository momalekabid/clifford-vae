"""
bundle capacity heatmap: accuracy as f(# dimensions, # bundled vectors)
reproduces schlegel et al. fig 3 style heatmaps for HRR, unitary, and clifford vectors
"""
import sys, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vsa import hrr_init, unitary_init, normalize_vectors, bundle, test_bundle_capacity
import torch.nn.functional as F


def clifford_init(n: int, d: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """sample from clifford torus: product of S^1 circles, then ifft to real domain"""
    # d_circles = d // 2 (each circle contributes 2 real dims)
    angles = torch.rand(n, d, device=device, dtype=dtype) * (2 * math.pi)
    freq_dim = 2 * d
    theta_s = torch.zeros(n, freq_dim, device=device, dtype=dtype)
    theta_s[:, 0] = 1.0
    theta_s[:, 1:d] = angles[:, 1:]
    theta_s[:, -d + 1:] = -torch.flip(angles[:, 1:], dims=(-1,))
    if freq_dim % 2 == 0:
        theta_s[:, freq_dim // 2] = 1.0
    fv = torch.cos(theta_s) + 1j * torch.sin(theta_s)
    vectors = torch.fft.ifft(fv).real.float()
    return vectors


def run_heatmap(init_fn, name, dims, k_range, n_items=1000, n_trials=20, device="cpu"):
    """run bundle capacity test across dims and k values, return accuracy matrix"""
    acc_matrix = np.full((len(dims), len(k_range)), np.nan)

    for i, d in enumerate(dims):
        print(f"  {name} d={d}...")
        # for clifford, init_fn produces vectors of size 2*d in real domain
        vectors = init_fn(n_items, d, device=device)
        actual_d = vectors.shape[1]
        vectors = normalize_vectors(vectors)

        for j, k in enumerate(k_range):
            if 2 * k > n_items:
                continue
            trial_accs = []
            for _ in range(n_trials):
                indices = torch.randperm(n_items, device=device)[:2 * k]
                X = vectors[indices[:k]]
                Xp = vectors[indices[k:2 * k]]
                C1 = bundle(X, normalize=True)
                C2 = bundle(Xp, normalize=True)
                sim1 = F.cosine_similarity(X, C1.unsqueeze(0), dim=-1)
                sim2 = F.cosine_similarity(X, C2.unsqueeze(0), dim=-1)
                acc = (sim1 > sim2).float().mean().item()
                trial_accs.append(acc)
            acc_matrix[i, j] = np.mean(trial_accs)

    return acc_matrix


def plot_heatmaps(results, dims, k_range, save_path=None):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for ax, (name, acc_matrix) in zip(axes, results.items()):
        # mask nans for display
        masked = np.ma.masked_invalid(acc_matrix)
        im = ax.pcolormesh(
            np.arange(len(k_range) + 1),
            np.arange(len(dims) + 1),
            masked,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )
        ax.set_xticks(np.arange(len(k_range)) + 0.5)
        ax.set_xticklabels(k_range, rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(dims)) + 0.5)
        ax.set_yticklabels(dims, fontsize=7)
        ax.set_xlabel("# bundled vectors", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("# dimensions", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, label="accuracy")
    fig.suptitle("Bundle Capacity Heatmaps", fontsize=13, fontweight="bold")
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
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_items", type=int, default=1000)
    args = p.parse_args()

    device = args.device

    # dims and k values similar to schlegel et al.
    dims = [4, 16, 64, 144, 256, 484, 512, 1024]
    k_range = list(range(3, 52, 4))  # 3, 7, 11, ..., 51

    init_fns = {
        "HRR": lambda n, d, dev=device: hrr_init(n, d, device=dev),
        "Unitary (FHRR)": lambda n, d, dev=device: unitary_init(n, d, device=dev),
    }

    results = {}
    for name, fn in init_fns.items():
        print(f"running {name}...")
        results[name] = run_heatmap(fn, name, dims, k_range,
                                     n_items=args.n_items, n_trials=args.n_trials,
                                     device=device)

    plot_heatmaps(results, dims, k_range, save_path="figures/bundle_heatmap.png")

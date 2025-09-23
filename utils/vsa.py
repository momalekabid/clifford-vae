import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import math


def hrr_init(n: int, d: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.randn(n, d, device=device, dtype=dtype) / math.sqrt(
        d
    )  # N(0, 1/sqrt(d))


def normalize_vectors(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    fa = torch.fft.fft(a, dim=-1)
    fb = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(fa * fb, dim=-1).real


def unbind(ab: torch.Tensor, b: torch.Tensor, method: str = "inv") -> torch.Tensor:
    if method == "inv":
        b_inv = torch.cat([b[..., :1], torch.flip(b[..., 1:], dims=[-1])], dim=-1)
        return bind(ab, b_inv)
    elif method == "deconv":
        fab = torch.fft.fft(ab, dim=-1)
        fb = torch.fft.fft(b, dim=-1)
        eps = 1e-8
        fa = fab / (fb + eps)
        return torch.fft.ifft(fa, dim=-1).real
    else:
        raise ValueError(f"please specify unbind method: {method}")


def bundle(vectors: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    s = torch.sum(vectors, dim=0)
    if normalize:
        s = s / math.sqrt(vectors.shape[0])
    return s


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.device != b.device:
        b = b.to(a.device)
    return F.cosine_similarity(a, b, dim=-1)


def test_bundle_capacity(
    d: int = 1024,
    n_items: int = 1000,
    k_range: List[int] = None,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    decoder=None,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
) -> Dict:
    if k_range is None:
        k_range = list(range(2, min(51, n_items // 2), 2))

    if item_memory is None:
        item_memory = hrr_init(n_items, d, device=device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
    else:
        item_memory = item_memory[:n_items].to(device)
        if normalize:
            item_memory = normalize_vectors(item_memory)

    results = {"k": [], "accuracy": [], "std": []}

    for k in k_range:
        trial_accs = []

        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[:k]
            selected = item_memory[indices]

            bundled = bundle(selected, normalize=True)

            sims = similarity(bundled, item_memory)

            top_k_sims, top_k_indices = torch.topk(sims, k)
            # print(top_k_sims)
            correct = torch.isin(top_k_indices, indices).sum().item()
            trial_accs.append(correct / k)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results["k"].append(k)
        results["accuracy"].append(mean_acc)
        results["std"].append(std_acc)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            results["k"],
            results["accuracy"],
            yerr=results["std"],
            marker="o",
            capsize=3,
        )
        plt.xlabel("Number of bundled vectors (k)")
        plt.ylabel("Retrieval accuracy")
        plt.title(f"Bundle Capacity (d={d}, N={n_items})")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "bundle_capacity.png"), dpi=200)
        plt.close()

    return results


def test_binding_unbinding_pairs(
    d: int = 1024,
    n_items: int = 1000,
    k_range: List[int] = None,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    unbind_method: str = "inv",
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
) -> Dict:
    if k_range is None:
        k_range = list(range(2, min(31, n_items // 4), 2))

    if item_memory is None:
        item_memory = hrr_init(n_items, d, device=device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
    else:
        item_memory = item_memory[:n_items].to(device)
        if normalize:
            item_memory = normalize_vectors(item_memory)

    results = {"k": [], "accuracy": [], "std": []}

    for k in k_range:
        trial_accs = []

        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[: 2 * k]
            roles = item_memory[indices[:k]]
            fillers = item_memory[indices[k : 2 * k]]

            pairs = bind(roles, fillers)

            bundled = bundle(pairs, normalize=True)

            correct = 0
            for i in range(k):
                recovered = unbind(
                    bundled.unsqueeze(0), roles[i].unsqueeze(0), method=unbind_method
                ).squeeze()

                sims = similarity(recovered, item_memory)
                best_idx = torch.argmax(sims).item()

                target_filler_idx = indices[k + i].item()
                if best_idx == target_filler_idx:
                    correct += 1

            trial_accs.append(correct / k)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results["k"].append(k)
        results["accuracy"].append(mean_acc)
        results["std"].append(std_acc)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            results["k"],
            results["accuracy"],
            yerr=results["std"],
            marker="s",
            capsize=3,
            color="red",
        )
        plt.xlabel("Number of bundled role-filler pairs (k)")
        plt.ylabel("Unbinding accuracy")
        plt.title(f"Role-Filler Query Capacity (d={d}, N={n_items}, {unbind_method})")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"unbind_bundled_pairs_{unbind_method}.png"),
                dpi=200,
            )
        plt.close()

    return results


def run_all_tests(
    d: int = 1024,
    n_items: int = 1000,
    normalize: bool = True,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
) -> Dict:
    import os

    print(f"Running VSA tests with d={d}, N={n_items}, normalize={normalize}")
    if item_memory is not None:
        print("Using provided latent representations for VSA tests")

    results = {}

    print("Testing bundle capacity...")
    results["bundle_capacity"] = test_bundle_capacity(
        d=d,
        n_items=n_items,
        normalize=normalize,
        device=device,
        plot=True,
        item_memory=item_memory,
        save_dir=save_dir,
    )

    print("Testing binding/unbinding of bundled pairs...")
    results["binding_unbinding"] = test_binding_unbinding_pairs(
        d=d,
        n_items=n_items,
        normalize=normalize,
        device=device,
        plot=True,
        item_memory=item_memory,
        save_dir=save_dir,
    )

    print("\n=== Summary ===")
    bc = results["bundle_capacity"]
    max_k_99 = max(
        [k for k, acc in zip(bc["k"], bc["accuracy"]) if acc >= 0.99], default=0
    )
    print(f"Bundle capacity: max k with >99% accuracy = {max_k_99}")

    bu = results["binding_unbinding"]
    max_k_90 = max(
        [k for k, acc in zip(bu["k"], bu["accuracy"]) if acc >= 0.90], default=0
    )
    print(f"Binding/unbinding: max k with >90% accuracy = {max_k_90}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_all_tests(
        d=1024, n_items=1000, normalize=True, device=device, save_dir="vsa_results"
    )
    plt.show()

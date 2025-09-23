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


def test_bundle_capacity_by_class(
    d: int = 1024,
    n_items: int = 1000,
    k_range: List[int] = None,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    test_type: str = "diverse_classes",  # "diverse_classes" or "similar_classes"
) -> Dict:
    """
    test bundle capacity with class-aware sampling
    test_type:
    - "diverse_classes": bundle one vector from each of k different classes
    - "similar_classes": bundle k vectors from the same class
    """
    if k_range is None:
        if test_type == "diverse_classes":
            k_range = list(range(2, 11))  # max 10 classes for mnist
        else:
            k_range = list(range(2, min(51, n_items // 2), 2))

    if item_memory is None:
        item_memory = hrr_init(n_items, d, device=device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        # create fake labels for random vectors
        labels = torch.randint(0, 10, (n_items,), device=device)
    else:
        item_memory = item_memory[:n_items].to(device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        if labels is None:
            labels = torch.randint(0, 10, (n_items,), device=device)
        else:
            labels = labels[:n_items].to(device)

    results = {"k": [], "accuracy": [], "std": [], "similarity_matrices": []}
    n_classes = len(torch.unique(labels))

    for k in k_range:
        trial_accs = []
        trial_sim_matrices = []

        for _ in range(n_trials):
            if test_type == "diverse_classes":
                # select one vector from each of k different classes
                if k > n_classes:
                    continue  # skip if k > number of available classes

                unique_classes = torch.unique(labels)[:k]
                selected_indices = []
                for class_id in unique_classes:
                    class_indices = torch.where(labels == class_id)[0]
                    if len(class_indices) > 0:
                        selected_idx = class_indices[torch.randint(0, len(class_indices), (1,))]
                        selected_indices.append(selected_idx.item())

                if len(selected_indices) < k:
                    continue

                indices = torch.tensor(selected_indices, device=device)
            else:  # similar_classes
                # select k vectors from the same random class
                unique_classes = torch.unique(labels)
                class_id = unique_classes[torch.randint(0, len(unique_classes), (1,))]
                class_indices = torch.where(labels == class_id)[0]
                if len(class_indices) < k:
                    continue
                indices = class_indices[torch.randperm(len(class_indices))[:k]]

            selected = item_memory[indices]
            bundled = bundle(selected, normalize=True)

            # compute similarity with all vectors
            sims = similarity(bundled, item_memory)

            # store similarity matrix for analysis
            selected_sims = sims[indices].cpu().numpy()
            trial_sim_matrices.append(selected_sims)

            # compute retrieval accuracy
            top_k_sims, top_k_indices = torch.topk(sims, k)
            correct = torch.isin(top_k_indices, indices).sum().item()
            trial_accs.append(correct / k)

        if trial_accs:  # only add if we have valid trials
            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)
            mean_sim_matrix = np.mean(trial_sim_matrices, axis=0) if trial_sim_matrices else None

            results["k"].append(k)
            results["accuracy"].append(mean_acc)
            results["std"].append(std_acc)
            results["similarity_matrices"].append(mean_sim_matrix)

    if plot and results["k"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # plot accuracy
        ax1.errorbar(
            results["k"],
            results["accuracy"],
            yerr=results["std"],
            marker="o",
            capsize=3,
            color="blue" if test_type == "diverse_classes" else "red",
        )
        ax1.set_xlabel("Number of bundled vectors (k)")
        ax1.set_ylabel("Retrieval accuracy")
        title = f"Bundle Capacity - {test_type.replace('_', ' ').title()} (d={d}, N={n_items})"
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)

        # plot similarity matrix heatmap for largest k
        if results["similarity_matrices"] and results["similarity_matrices"][-1] is not None:
            sim_matrix = results["similarity_matrices"][-1]
            im = ax2.imshow(sim_matrix.reshape(1, -1), cmap="viridis", aspect="auto")
            ax2.set_title(f"Similarity to bundled vectors (k={results['k'][-1]})")
            ax2.set_xlabel("Vector index in bundle")
            ax2.set_ylabel("")
            ax2.set_yticks([])
            plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            filename = f"bundle_capacity_{test_type}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=200)
        plt.close()

    return results


def test_bundle_capacity_class_analysis(
    d: int = 1024,
    n_items: int = 1000,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Dict:
    """
    comprehensive analysis comparing diverse vs similar class bundling
    and examining class distribution in bundles
    """
    results = {}

    # test diverse classes (one from each class)
    print("testing bundle capacity with diverse classes...")
    results["diverse"] = test_bundle_capacity_by_class(
        d=d, n_items=n_items, k_range=list(range(2, 11)), n_trials=n_trials,
        normalize=normalize, device=device, plot=plot, save_dir=save_dir,
        item_memory=item_memory, labels=labels, test_type="diverse_classes"
    )

    # test similar classes (all from same class)
    print("testing bundle capacity with similar classes...")
    results["similar"] = test_bundle_capacity_by_class(
        d=d, n_items=n_items, k_range=list(range(2, 21, 2)), n_trials=n_trials,
        normalize=normalize, device=device, plot=plot, save_dir=save_dir,
        item_memory=item_memory, labels=labels, test_type="similar_classes"
    )

    # test regular bundle capacity for comparison
    print("testing regular bundle capacity (random sampling)...")
    results["random"] = test_bundle_capacity(
        d=d, n_items=n_items, k_range=list(range(2, 21, 2)), n_trials=n_trials,
        normalize=normalize, device=device, plot=plot, save_dir=save_dir,
        item_memory=item_memory
    )

    # create comprehensive comparison plot
    if plot:
        plt.figure(figsize=(12, 8))

        # plot all three curves
        if results["diverse"]["k"]:
            plt.errorbar(
                results["diverse"]["k"], results["diverse"]["accuracy"],
                yerr=results["diverse"]["std"], marker="o", capsize=3,
                label="diverse classes (one per class)", color="blue"
            )

        if results["similar"]["k"]:
            plt.errorbar(
                results["similar"]["k"], results["similar"]["accuracy"],
                yerr=results["similar"]["std"], marker="s", capsize=3,
                label="similar classes (same class)", color="red"
            )

        if results["random"]["k"]:
            plt.errorbar(
                results["random"]["k"], results["random"]["accuracy"],
                yerr=results["random"]["std"], marker="^", capsize=3,
                label="random sampling", color="green"
            )

        plt.xlabel("number of bundled vectors (k)")
        plt.ylabel("retrieval accuracy")
        plt.title(f"bundle capacity analysis by class diversity (d={d}, n={n_items})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "bundle_capacity_class_analysis.png"), dpi=200)
        plt.close()

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_all_tests(
        d=1024, n_items=1000, normalize=True, device=device, save_dir="vsa_results"
    )
    plt.show()

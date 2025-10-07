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


def invert(a: torch.Tensor) -> torch.Tensor:
    """invert operation: [a0, a1, a2, ..., a_{n-1}] -> [a0, a_{n-1}, ..., a2, a1]"""
    head = a[..., :1]
    tail = a[..., 1:]
    return torch.cat([head, torch.flip(tail, dims=[-1])], dim=-1)


def unbind(ab: torch.Tensor, b: torch.Tensor, method: str = "inv") -> torch.Tensor:
    """
    unbind operation for vsa
    - inv/*: x̂ = (ab) ⊛ a^{-1}, where a^{-1} = (a_n, ..., a_2, a_1)
    - †/deconv: x̂ = IFFT( FFT(ab) / FFT(a) )
    """
    if method == "inv" or method == "*":
        b_inv = invert(b)
        return bind(ab, b_inv)
    elif method == "†" or method == "deconv":
        fab = torch.fft.fft(ab, dim=-1)
        fb = torch.fft.fft(b, dim=-1)
        eps = 1e-12
        fa = fab / (fb + eps)
        return torch.fft.ifft(fa, dim=-1).real
    else:
        raise ValueError(f"unsupported unbind method: {method}")


def bundle(vectors: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    s = torch.sum(vectors, dim=0)
    if normalize:
        s = s / math.sqrt(vectors.shape[0])  # divide by sqrt(k)
    return s


def permute_vector(v: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """apply permutation to vector (braiding/shuffling)"""
    return v[..., perm]


def unpermute_vector(v: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """reverse permutation (unbraiding/unshuffling)"""
    inv_perm = torch.argsort(perm)
    return v[..., inv_perm]


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.device != b.device:
        b = b.to(a.device)
    return F.cosine_similarity(a, b, dim=-1)


def test_bundle_capacity(
    d: int = 1024,
    n_items: int = 1000,
    k_range=None,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    decoder=None,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    use_braiding: bool = False,
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

    # generate different random permutation for each vector (paper sec. "Shuffling to Store...")
    # "if two images are shuffled differently, the shuffled versions will have only incidental similarity"
    perm_dict = {}
    if use_braiding:
        braided_memory = torch.zeros_like(item_memory)
        for i in range(n_items):
            perm = torch.randperm(d, device=device)
            perm_dict[i] = perm
            braided_memory[i] = permute_vector(item_memory[i], perm)
        item_memory = braided_memory

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
    k_range=None,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    unbind_method: str = "inv",
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    use_braiding: bool = False,
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

            if use_braiding:
                # braid each pair with unique permutation before bundling
                braided_pairs = []
                perms = []
                for i in range(k):
                    perm = torch.randperm(d, device=device)
                    perms.append(perm)
                    braided = permute_vector(pairs[i], perm)
                    braided_pairs.append(braided)
                bundled = bundle(torch.stack(braided_pairs), normalize=True)
            else:
                bundled = bundle(pairs, normalize=True)

            correct = 0
            for i in range(k):
                if use_braiding:
                    # unbraid before unbinding
                    unbraided = unpermute_vector(bundled, perms[i])
                    recovered = unbind(
                        unbraided.unsqueeze(0),
                        roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()
                else:
                    recovered = unbind(
                        bundled.unsqueeze(0),
                        roles[i].unsqueeze(0),
                        method=unbind_method,
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
        braid_label = " [BRAIDED]" if use_braiding else ""
        plt.errorbar(
            results["k"],
            results["accuracy"],
            yerr=results["std"],
            marker="s",
            capsize=3,
            color="red",
        )
        plt.xlabel("number of bundled role-filler pairs (k)")
        plt.ylabel("unbinding accuracy")
        plt.title(
            f"role-filler query capacity{braid_label} (d={d}, N={n_items}, {unbind_method})"
        )
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            braid_suffix = "_braided" if use_braiding else ""
            plt.savefig(
                os.path.join(
                    save_dir, f"unbind_bundled_pairs_{unbind_method}{braid_suffix}.png"
                ),
                dpi=200,
            )
        plt.close()

    return results


def test_binding_unbinding_with_self_binding(
    d: int = 1024,
    n_items: int = 1000,
    k_range=None,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    unbind_method: str = "inv",
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    use_braiding: bool = False,
) -> Dict:
    """
    test binding/unbinding with nested self-binding structures:
    - role ⊛ (filler ⊛ filler)  - self-bound filler
    - (role ⊛ role) ⊛ filler     - self-bound role
    - role ⊛ (filler ⊛ role)     - circular binding

    bundle k such pairs and test recovery of the filler
    """
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

    results = {
        "k": [],
        "accuracy_self_filler": [],
        "accuracy_self_role": [],
        "accuracy_circular": [],
        "std_self_filler": [],
        "std_self_role": [],
        "std_circular": [],
    }

    for k in k_range:
        trial_accs_self_filler = []
        trial_accs_self_role = []
        trial_accs_circular = []

        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[: 2 * k]
            roles = item_memory[indices[:k]]
            fillers = item_memory[indices[k : 2 * k]]

            # pattern 1: role ⊛ (filler ⊛ filler)
            self_bound_fillers = bind(fillers, fillers)
            pairs_self_filler = bind(roles, self_bound_fillers)

            # pattern 2: (role ⊛ role) ⊛ filler
            self_bound_roles = bind(roles, roles)
            pairs_self_role = bind(self_bound_roles, fillers)

            # pattern 3: role ⊛ (filler ⊛ role)
            filler_role_bound = bind(fillers, roles)
            pairs_circular = bind(roles, filler_role_bound)

            # bundle with optional braiding
            if use_braiding:
                # braid each pattern's pairs
                braided_pairs_1, perms_1 = [], []
                braided_pairs_2, perms_2 = [], []
                braided_pairs_3, perms_3 = [], []

                for i in range(k):
                    perm = torch.randperm(d, device=device)
                    perms_1.append(perm)
                    braided_pairs_1.append(permute_vector(pairs_self_filler[i], perm))

                    perm = torch.randperm(d, device=device)
                    perms_2.append(perm)
                    braided_pairs_2.append(permute_vector(pairs_self_role[i], perm))

                    perm = torch.randperm(d, device=device)
                    perms_3.append(perm)
                    braided_pairs_3.append(permute_vector(pairs_circular[i], perm))

                bundled_1 = bundle(torch.stack(braided_pairs_1), normalize=True)
                bundled_2 = bundle(torch.stack(braided_pairs_2), normalize=True)
                bundled_3 = bundle(torch.stack(braided_pairs_3), normalize=True)
            else:
                bundled_1 = bundle(pairs_self_filler, normalize=True)
                bundled_2 = bundle(pairs_self_role, normalize=True)
                bundled_3 = bundle(pairs_circular, normalize=True)

            # test recovery for pattern 1: role ⊛ (filler ⊛ filler)
            # to recover filler: unbind role, then unbind filler from result
            correct_1 = 0
            for i in range(k):
                if use_braiding:
                    unbraided = unpermute_vector(bundled_1, perms_1[i])
                    filler_filler = unbind(
                        unbraided.unsqueeze(0),
                        roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()
                else:
                    filler_filler = unbind(
                        bundled_1.unsqueeze(0),
                        roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()

                # now unbind filler from (filler ⊛ filler)
                recovered = unbind(
                    filler_filler.unsqueeze(0),
                    fillers[i].unsqueeze(0),
                    method=unbind_method,
                ).squeeze()

                sims = similarity(recovered, item_memory)
                best_idx = torch.argmax(sims).item()
                if best_idx == indices[k + i].item():
                    correct_1 += 1

            # test recovery for pattern 2: (role ⊛ role) ⊛ filler
            # to recover filler: first unbind to get filler, then unbind role from (role ⊛ role)
            correct_2 = 0
            for i in range(k):
                if use_braiding:
                    unbraided = unpermute_vector(bundled_2, perms_2[i])
                    # unbind (role ⊛ role) to get filler
                    recovered = unbind(
                        unbraided.unsqueeze(0),
                        self_bound_roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()
                else:
                    recovered = unbind(
                        bundled_2.unsqueeze(0),
                        self_bound_roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()

                sims = similarity(recovered, item_memory)
                best_idx = torch.argmax(sims).item()
                if best_idx == indices[k + i].item():
                    correct_2 += 1

            # test recovery for pattern 3: role ⊛ (filler ⊛ role)
            # to recover filler: unbind role, then unbind role again
            correct_3 = 0
            for i in range(k):
                if use_braiding:
                    unbraided = unpermute_vector(bundled_3, perms_3[i])
                    filler_role = unbind(
                        unbraided.unsqueeze(0),
                        roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()
                else:
                    filler_role = unbind(
                        bundled_3.unsqueeze(0),
                        roles[i].unsqueeze(0),
                        method=unbind_method,
                    ).squeeze()

                # unbind role from (filler ⊛ role)
                recovered = unbind(
                    filler_role.unsqueeze(0),
                    roles[i].unsqueeze(0),
                    method=unbind_method,
                ).squeeze()

                sims = similarity(recovered, item_memory)
                best_idx = torch.argmax(sims).item()
                if best_idx == indices[k + i].item():
                    correct_3 += 1

            trial_accs_self_filler.append(correct_1 / k)
            trial_accs_self_role.append(correct_2 / k)
            trial_accs_circular.append(correct_3 / k)

        results["k"].append(k)
        results["accuracy_self_filler"].append(np.mean(trial_accs_self_filler))
        results["accuracy_self_role"].append(np.mean(trial_accs_self_role))
        results["accuracy_circular"].append(np.mean(trial_accs_circular))
        results["std_self_filler"].append(np.std(trial_accs_self_filler))
        results["std_self_role"].append(np.std(trial_accs_self_role))
        results["std_circular"].append(np.std(trial_accs_circular))

    if plot:
        plt.figure(figsize=(10, 6))
        braid_label = " [BRAIDED]" if use_braiding else ""

        plt.errorbar(
            results["k"],
            results["accuracy_self_filler"],
            yerr=results["std_self_filler"],
            marker="o",
            capsize=3,
            label="role ⊛ (filler ⊛ filler)",
        )
        plt.errorbar(
            results["k"],
            results["accuracy_self_role"],
            yerr=results["std_self_role"],
            marker="s",
            capsize=3,
            label="(role ⊛ role) ⊛ filler",
        )
        plt.errorbar(
            results["k"],
            results["accuracy_circular"],
            yerr=results["std_circular"],
            marker="^",
            capsize=3,
            label="role ⊛ (filler ⊛ role)",
        )

        plt.xlabel("number of bundled pairs (k)")
        plt.ylabel("unbinding accuracy")
        plt.title(
            f"self-binding patterns{braid_label} (d={d}, N={n_items}, {unbind_method})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()

        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            braid_suffix = "_braided" if use_braiding else ""
            plt.savefig(
                os.path.join(
                    save_dir, f"self_binding_patterns_{unbind_method}{braid_suffix}.png"
                ),
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


def test_bundle_capacity_confusion_matrix(
    d: int = 1024,
    n_items: int = 2500,  # 50 classes x 50 items each
    n_classes: int = 50,
    n_trials: int = 10,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Dict:
    """
    test bundle capacity with 50x50 confusion matrix to analyze
    how duplicate class member similarities are distributed differently
    in clifford vs gaussian latent spaces
    """
    if item_memory is None:
        item_memory = hrr_init(n_items, d, device=device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        labels = torch.randint(0, n_classes, (n_items,), device=device)
    else:
        item_memory = item_memory[:n_items].to(device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        if labels is None:
            labels = torch.randint(0, n_classes, (n_items,), device=device)
        else:
            labels = labels[:n_items].to(device)

    unique_classes = torch.unique(labels).cpu().numpy()
    if len(unique_classes) < n_classes:
        print(f"warning: only {len(unique_classes)} classes found, need {n_classes}")
        n_classes = len(unique_classes)

    # build class-to-items mapping
    class_to_items = {}
    for class_id in unique_classes[:n_classes]:
        class_indices = torch.where(labels == class_id)[0]
        if len(class_indices) > 0:
            class_to_items[class_id] = class_indices

    confusion_matrices = []
    similarity_distributions = []

    for trial in range(n_trials):
        # select one item from each class for bundling
        selected_indices = []
        selected_classes = []

        for class_id in unique_classes[:n_classes]:
            if class_id in class_to_items and len(class_to_items[class_id]) > 1:
                # randomly select one item for bundling
                selected_idx = class_to_items[class_id][
                    torch.randint(0, len(class_to_items[class_id]), (1,))
                ]
                selected_indices.append(selected_idx.item())
                selected_classes.append(class_id)

        if len(selected_indices) < 10:  # need minimum classes for meaningful analysis
            continue

        # bundle the selected items
        selected_vectors = item_memory[selected_indices]
        bundled = bundle(selected_vectors, normalize=True)

        # compute confusion matrix: similarity of bundle to all items per class
        confusion_matrix = np.zeros((len(selected_classes), len(selected_classes)))
        all_similarities = []

        for i, class_id in enumerate(selected_classes):
            class_indices = class_to_items[class_id]
            class_vectors = item_memory[class_indices]
            sims = similarity(bundled.unsqueeze(0), class_vectors).cpu().numpy()
            all_similarities.extend(sims)

            # average similarity to this class
            mean_sim = np.mean(sims)
            confusion_matrix[i, i] = mean_sim

            # cross-class similarities
            for j, other_class_id in enumerate(selected_classes):
                if i != j:
                    other_indices = class_to_items[other_class_id]
                    other_vectors = item_memory[other_indices]
                    cross_sims = (
                        similarity(bundled.unsqueeze(0), other_vectors).cpu().numpy()
                    )
                    confusion_matrix[i, j] = np.mean(cross_sims)

        confusion_matrices.append(confusion_matrix)
        similarity_distributions.append(all_similarities)

    if confusion_matrices:
        avg_confusion = np.mean(confusion_matrices, axis=0)
        std_confusion = np.std(confusion_matrices, axis=0)

        results = {
            "avg_confusion_matrix": avg_confusion,
            "std_confusion_matrix": std_confusion,
            "similarity_distributions": similarity_distributions,
            "diagonal_similarities": [np.diag(cm) for cm in confusion_matrices],
            "off_diagonal_similarities": [
                cm[~np.eye(cm.shape[0], dtype=bool)] for cm in confusion_matrices
            ],
        }

        if plot and save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # plot average confusion matrix
            im1 = axes[0, 0].imshow(avg_confusion, cmap="viridis", aspect="auto")
            axes[0, 0].set_title("average bundle-to-class similarity matrix")
            axes[0, 0].set_xlabel("class index")
            axes[0, 0].set_ylabel("class index")
            plt.colorbar(im1, ax=axes[0, 0])

            # plot diagonal vs off-diagonal distributions
            all_diag = np.concatenate([np.diag(cm) for cm in confusion_matrices])
            all_off_diag = np.concatenate(
                [cm[~np.eye(cm.shape[0], dtype=bool)] for cm in confusion_matrices]
            )

            axes[0, 1].hist(
                all_diag, bins=30, alpha=0.7, label="same class", density=True
            )
            axes[0, 1].hist(
                all_off_diag, bins=30, alpha=0.7, label="different class", density=True
            )
            axes[0, 1].set_xlabel("cosine similarity")
            axes[0, 1].set_ylabel("density")
            axes[0, 1].set_title("similarity distributions")
            axes[0, 1].legend()

            # plot similarity distribution over all trials
            all_sims = np.concatenate(similarity_distributions)
            axes[1, 0].hist(all_sims, bins=50, alpha=0.7, edgecolor="black")
            axes[1, 0].set_xlabel("cosine similarity")
            axes[1, 0].set_ylabel("count")
            axes[1, 0].set_title(
                f"overall similarity distribution (μ={np.mean(all_sims):.3f})"
            )

            # plot separation metric
            separation_scores = []
            for cm in confusion_matrices:
                diag_mean = np.mean(np.diag(cm))
                off_diag_mean = np.mean(cm[~np.eye(cm.shape[0], dtype=bool)])
                separation_scores.append(diag_mean - off_diag_mean)

            axes[1, 1].boxplot(separation_scores)
            axes[1, 1].set_ylabel("separation score")
            axes[1, 1].set_title(
                f"class separation (μ={np.mean(separation_scores):.3f})"
            )

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, "bundle_capacity_confusion_50x50.png"), dpi=200
            )
            plt.close()

            results["confusion_plot_path"] = os.path.join(
                save_dir, "bundle_capacity_confusion_50x50.png"
            )

        return results

    return {"avg_confusion_matrix": None}


def test_per_class_bundle_capacity_two_items(
    d: int = 1024,
    n_items: int = 1000,
    n_classes: int = 10,
    items_per_class: int = 2,
    n_trials: int = 20,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_braiding: bool = False,
    per_class_braid: bool = False,
) -> Dict:
    """
    bundle 2 items from each of the n_classes classes
    compute similarity matrix between all individual bundles (a1, a2, b1, b2, ...)
    resulting matrix is 2*n_classes x 2*n_classes showing similarities between all pairs

    per_class_braid: if true, all items from same class use same permutation (class-level braiding)
    """
    if item_memory is None:
        item_memory = hrr_init(n_items, d, device=device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        labels = torch.randint(0, n_classes, (n_items,), device=device)
    else:
        item_memory = item_memory[:n_items].to(device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        if labels is None:
            labels = torch.randint(0, n_classes, (n_items,), device=device)
        else:
            labels = labels[:n_items].to(device)

    # apply permutation based on braiding mode
    perm_dict = {}
    if use_braiding:
        braided_memory = torch.zeros_like(item_memory)
        if per_class_braid:
            # per-class braiding: same permutation for all items in same class
            unique_classes = torch.unique(labels).cpu().numpy()
            class_to_perm = {}
            for class_id in unique_classes:
                class_to_perm[int(class_id)] = torch.randperm(d, device=device)

            for i in range(n_items):
                class_id = int(labels[i].item())
                perm = class_to_perm[class_id]
                perm_dict[i] = perm
                braided_memory[i] = permute_vector(item_memory[i], perm)
        else:
            # random braiding: unique permutation for each item
            for i in range(n_items):
                perm = torch.randperm(d, device=device)
                perm_dict[i] = perm
                braided_memory[i] = permute_vector(item_memory[i], perm)
        item_memory = braided_memory

    unique_classes = torch.unique(labels).cpu().numpy()
    if len(unique_classes) < n_classes:
        print(f"warning: only {len(unique_classes)} classes found, need {n_classes}")
        n_classes = len(unique_classes)

    # build class-to-items mapping
    class_to_items = {}
    for class_id in unique_classes[:n_classes]:
        class_indices = torch.where(labels == class_id)[0]
        if len(class_indices) >= items_per_class:
            class_to_items[class_id] = class_indices

    # filter to only use classes with enough items
    valid_classes = [c for c in unique_classes[:n_classes] if c in class_to_items]
    if len(valid_classes) < n_classes:
        print(f"warning: only {len(valid_classes)} classes have enough items")
        n_classes = len(valid_classes)

    similarity_matrices = []

    for trial in range(n_trials):
        # select items_per_class items from each class
        selected_bundles = []
        bundle_labels = []

        for class_id in valid_classes:
            class_indices = class_to_items[class_id]
            # randomly select items_per_class items
            perm = torch.randperm(len(class_indices))[:items_per_class]

            # create a bundle for each selected item
            for idx in perm:
                item_vector = item_memory[class_indices[idx]]
                selected_bundles.append(item_vector)
                bundle_labels.append(class_id)

        if len(selected_bundles) < n_classes * items_per_class:
            continue

        # compute pairwise similarities between all bundles
        n_bundles = len(selected_bundles)
        similarity_matrix = np.zeros((n_bundles, n_bundles))

        bundles_tensor = torch.stack(selected_bundles)

        for i in range(n_bundles):
            sims = similarity(bundles_tensor[i].unsqueeze(0), bundles_tensor)
            similarity_matrix[i, :] = sims.cpu().numpy()

        similarity_matrices.append(similarity_matrix)

    if similarity_matrices:
        avg_similarity = np.mean(similarity_matrices, axis=0)
        std_similarity = np.std(similarity_matrices, axis=0)

        results = {
            "avg_similarity_matrix": avg_similarity,
            "std_similarity_matrix": std_similarity,
            "n_bundles": n_classes * items_per_class,
            "n_classes": n_classes,
            "items_per_class": items_per_class,
        }

        if plot and save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # plot similarity matrix
            im = ax.imshow(avg_similarity, cmap="viridis", aspect="auto")
            if per_class_braid:
                braid_label = " [PER-CLASS BRAID]"
            elif use_braiding:
                braid_label = " [RANDOM BRAID]"
            else:
                braid_label = ""
            ax.set_title(
                f"bundle similarity matrix{braid_label}\n({items_per_class} items per class, {n_classes} classes)"
            )

            # create labels: a1, a2, b1, b2, c1, c2, ...
            tick_labels = []
            for i in range(n_classes):
                for j in range(items_per_class):
                    tick_labels.append(f"{chr(97+i)}{j+1}")

            ax.set_xticks(range(len(tick_labels)))
            ax.set_yticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels, rotation=90)
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("bundle")
            ax.set_ylabel("bundle")
            plt.colorbar(im, ax=ax, label="cosine similarity")

            plt.tight_layout()
            if per_class_braid:
                filename = "bundle_two_per_class_similarity_per_class_braid.png"
            elif use_braiding:
                filename = "bundle_two_per_class_similarity_braid.png"
            else:
                filename = "bundle_two_per_class_similarity.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=200)
            plt.close()

        return results

    return {"avg_similarity_matrix": None}


def test_per_class_bundle_capacity(
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
    bundle 1 item from each of the 10 classes (10 total items)
    then query that bundle against each class to see retrieval matrix
    """
    if item_memory is None:
        item_memory = hrr_init(n_items, d, device=device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        labels = torch.randint(0, 10, (n_items,), device=device)
    else:
        item_memory = item_memory[:n_items].to(device)
        if normalize:
            item_memory = normalize_vectors(item_memory)
        if labels is None:
            labels = torch.randint(0, 10, (n_items,), device=device)
        else:
            labels = labels[:n_items].to(device)

    unique_classes = torch.unique(labels).cpu().numpy()
    if len(unique_classes) < 10:
        print(f"warning: only {len(unique_classes)} classes found, need 10")
        return {"retrieval_matrix": None, "class_names": []}

    results = {
        "retrieval_matrices": [],
        "class_names": [f"class_{int(c)}" for c in unique_classes[:10]],
    }

    for _ in range(n_trials):
        # select one item from each of the first 10 classes
        selected_indices = []
        for class_id in unique_classes[:10]:
            class_indices = torch.where(labels == class_id)[0]
            if len(class_indices) == 0:
                continue
            # randomly select one item from this class
            selected_idx = class_indices[torch.randint(0, len(class_indices), (1,))]
            selected_indices.append(selected_idx.item())

        if len(selected_indices) < 10:
            continue

        # bundle the 10 items (1 from each class)
        selected_vectors = item_memory[selected_indices]
        bundled = bundle(selected_vectors, normalize=True)

        # compute similarity matrix: for each class, get top similarities
        retrieval_matrix = np.zeros((10, 10))  # 10 classes x top 10 retrieved

        for class_idx, class_id in enumerate(unique_classes[:10]):
            class_indices = torch.where(labels == class_id)[0]
            if len(class_indices) == 0:
                continue

            # get similarities to all items in this class
            class_vectors = item_memory[class_indices]
            sims = similarity(bundled.unsqueeze(0), class_vectors)

            # get top 10 similarities (or all if fewer than 10)
            top_k = min(10, len(class_indices))
            top_sims, _ = torch.topk(sims, top_k)

            retrieval_matrix[class_idx, :top_k] = top_sims.cpu().numpy()

        results["retrieval_matrices"].append(retrieval_matrix)

    if results["retrieval_matrices"]:
        avg_retrieval_matrix = np.mean(results["retrieval_matrices"], axis=0)
        results["avg_retrieval_matrix"] = avg_retrieval_matrix

        if plot and save_dir:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # plot average retrieval matrix
            im1 = ax1.imshow(avg_retrieval_matrix, cmap="viridis", aspect="auto")
            ax1.set_title(
                "average retrieval similarities\n(bundle of 1 from each class)"
            )
            ax1.set_xlabel("top-k retrieved items")
            ax1.set_ylabel("query class")
            ax1.set_yticks(range(10))
            ax1.set_yticklabels([f"class {i}" for i in range(10)])
            plt.colorbar(im1, ax=ax1)

            diagonal_sims = avg_retrieval_matrix[
                :, 0
            ]  # top-1 similarity for each class
            ax2.bar(range(10), diagonal_sims)
            ax2.set_title("top-1 retrieval similarity per class")
            ax2.set_xlabel("class")
            ax2.set_ylabel("similarity")
            ax2.set_xticks(range(10))
            ax2.set_xticklabels([f"{i}" for i in range(10)])
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            import os

            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "per_class_bundle_analysis.png"), dpi=200
            )
            plt.close()

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_all_tests(
        d=1024, n_items=1000, normalize=True, device=str(device), save_dir="vsa_results"
    )
    plt.show()

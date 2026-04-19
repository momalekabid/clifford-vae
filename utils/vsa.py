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


def unitary_init(
    n: int, d: int, device="cpu", dtype=torch.float32, eps=1e-3
) -> torch.Tensor:
    """init n vectors of dimension d with unit fourier magnitude (|F[k]|=1 for all k).
    based on make_good_unitary from sspspace
    """
    vectors = torch.zeros(n, d, device=device, dtype=dtype)
    n_phases = (d - 1) // 2
    for i in range(n):
        a = torch.rand(n_phases, device=device)
        sign = torch.sign(torch.rand(n_phases, device=device) - 0.5)
        phi = sign * math.pi * (eps + a * (1 - 2 * eps))

        fv = torch.zeros(d, device=device, dtype=torch.complex64)
        fv[0] = 1.0
        fv[1 : (d + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
        fv[d // 2 + 1 :] = torch.flip(torch.conj(fv[1 : (d + 1) // 2]), dims=(0,))
        if d % 2 == 0:
            fv[d // 2] = 1.0

        vectors[i] = torch.fft.ifft(fv).real
    return vectors


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
    unbind operation for HRR vsa
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
    bind_with_random: bool = False,
    baseline_d: Optional[int] = None,
) -> Dict:
    """
    for each k:
      - draw 2k items from item_memory
      - X = first k, X' = second k
      - C1 = bundle(X), C2 = bundle(X')
      - accuracy = fraction of x ∈ X where cos(x, C1) > cos(x, C2)
    """
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
            # need 2k items: first k → X, next k → X'
            n_needed = min(2 * k, n_items)
            if n_needed < 2:
                trial_accs.append(0.0)
                continue
            actual_k = n_needed // 2
            indices = torch.randperm(n_items, device=device)[:n_needed]
            X = item_memory[indices[:actual_k]]
            Xp = item_memory[indices[actual_k : actual_k * 2]]

            C1 = bundle(X, normalize=True)
            C2 = bundle(Xp, normalize=True)

            # sim of each x ∈ X to both bundles
            # C1 shape: (d,), broadcast against X: (k, d)
            sim_to_C1 = F.cosine_similarity(
                X, C1.unsqueeze(0).expand(actual_k, -1), dim=-1
            )
            sim_to_C2 = F.cosine_similarity(
                X, C2.unsqueeze(0).expand(actual_k, -1), dim=-1
            )
            acc = (sim_to_C1 > sim_to_C2).float().mean().item()
            trial_accs.append(acc)

        mean_acc = float(np.mean(trial_accs))
        std_acc = float(np.std(trial_accs))
        results["k"].append(k)
        results["accuracy"].append(mean_acc)
        results["std"].append(std_acc)

    if plot:
        import os

        # compute baselines with HRR and unitary vectors
        # use baseline_d (encoder dim) so clifford baselines match the conceptual dim
        bd = baseline_d if baseline_d is not None else d
        baselines = {}
        for bname, init_fn in [("HRR", hrr_init), ("unitary", unitary_init)]:
            bvecs = init_fn(n_items, bd, device="cpu")
            if normalize:
                bvecs = normalize_vectors(bvecs)
            b_res = {"k": [], "accuracy": [], "std": []}
            for k in k_range:
                trial_accs = []
                for _ in range(min(n_trials, 10)):
                    n_needed = min(2 * k, n_items)
                    actual_k = n_needed // 2
                    idx = torch.randperm(n_items)[:n_needed]
                    X = bvecs[idx[:actual_k]]
                    Xp = bvecs[idx[actual_k:actual_k * 2]]
                    C1 = bundle(X, normalize=True)
                    C2 = bundle(Xp, normalize=True)
                    s1 = F.cosine_similarity(X, C1.unsqueeze(0).expand(actual_k, -1), dim=-1)
                    s2 = F.cosine_similarity(X, C2.unsqueeze(0).expand(actual_k, -1), dim=-1)
                    trial_accs.append((s1 > s2).float().mean().item())
                b_res["k"].append(k)
                b_res["accuracy"].append(float(np.mean(trial_accs)))
                b_res["std"].append(float(np.std(trial_accs)))
            baselines[bname] = b_res

        display_d = baseline_d if baseline_d is not None else d
        plt.figure(figsize=(8, 5))
        plt.errorbar(results["k"], results["accuracy"], yerr=results["std"],
                     marker="o", capsize=3, label="Learned Latents", color="tab:blue", linewidth=2)
        plt.errorbar(baselines["HRR"]["k"], baselines["HRR"]["accuracy"],
                     yerr=baselines["HRR"]["std"], marker="^", capsize=3,
                     label="HRR (Random)", color="tab:gray", linestyle="--", alpha=0.8)
        plt.errorbar(baselines["unitary"]["k"], baselines["unitary"]["accuracy"],
                     yerr=baselines["unitary"]["std"], marker="v", capsize=3,
                     label="Random Unitary", color="tab:green", linestyle="--", alpha=0.8)
        plt.xlabel("Number of Bundled Vectors ($k$)")
        plt.ylabel("Retrieval Accuracy")
        plt.title(f"Bundle Capacity ($d={display_d}$, $N={n_items}$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "bundle_capacity.png"), dpi=500)
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
    bind_with_random: bool = True,
    baseline_d: Optional[int] = None,
) -> Dict:
    """
    test binding/unbinding with bundled pairs.

    if bind_with_random=True (default):
        - select k random vectors from item_memory
        - bind each with a random HRR-initialized vector
        - bundle the bound pairs
        - test recovery by unbinding and checking similarity

    if bind_with_random=False:
        - classic role-filler binding test
        - select k roles and k fillers from item_memory
        - bind each role with its filler
        - bundle and test recovery
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

    # move to cpu for fft — cuFFT chokes on large flattened latent dims
    item_memory = item_memory.cpu()

    results = {"k": [], "accuracy": [], "std": []}

    for k in k_range:
        trial_accs = []

        for _ in range(n_trials):
            if bind_with_random:
                indices = torch.randperm(n_items)[:k]
                fillers = item_memory[indices]
                roles = unitary_init(k, d, device="cpu")
                if normalize:
                    roles = normalize_vectors(roles)
            else:
                indices = torch.randperm(n_items)[: 2 * k]
                roles = item_memory[indices[:k]]
                fillers = item_memory[indices[k : 2 * k]]

            pairs = bind(roles, fillers)

            if use_braiding:
                braided_pairs = []
                perms = []
                for i in range(k):
                    perm = torch.randperm(d)
                    perms.append(perm)
                    braided = permute_vector(pairs[i], perm)
                    braided_pairs.append(braided)
                bundled = bundle(torch.stack(braided_pairs), normalize=True)
            else:
                bundled = bundle(pairs, normalize=True)

            correct = 0
            for i in range(k):
                if use_braiding:
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

                if bind_with_random:
                    target_filler_idx = indices[i].item()
                else:
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
        import os

        # compute baselines with HRR and unitary vectors
        bd = baseline_d if baseline_d is not None else d
        baselines = {}
        for bname, init_fn in [("HRR", hrr_init), ("unitary", unitary_init)]:
            bvecs = init_fn(n_items, bd, device="cpu")
            if normalize:
                bvecs = normalize_vectors(bvecs)
            b_res = {"k": [], "accuracy": [], "std": []}
            for k in k_range:
                trial_accs = []
                for _ in range(min(n_trials, 10)):
                    if bind_with_random:
                        idx = torch.randperm(n_items)[:k]
                        fillers = bvecs[idx]
                        roles = unitary_init(k, bd, device="cpu")
                        if normalize:
                            roles = normalize_vectors(roles)
                    else:
                        idx = torch.randperm(n_items)[:2 * k]
                        roles = bvecs[idx[:k]]
                        fillers = bvecs[idx[k:2 * k]]
                    pairs = bind(roles, fillers)
                    bundled = bundle(pairs, normalize=True)
                    correct = 0
                    for i in range(k):
                        recovered = unbind(bundled.unsqueeze(0), roles[i].unsqueeze(0),
                                           method=unbind_method).squeeze()
                        sims = similarity(recovered, bvecs)
                        best_idx = torch.argmax(sims).item()
                        target_idx = idx[i].item() if bind_with_random else idx[k + i].item()
                        if best_idx == target_idx:
                            correct += 1
                    trial_accs.append(correct / k)
                b_res["k"].append(k)
                b_res["accuracy"].append(float(np.mean(trial_accs)))
                b_res["std"].append(float(np.std(trial_accs)))
            baselines[bname] = b_res

        display_d = baseline_d if baseline_d is not None else d
        bind_label = " (Random Keys)" if bind_with_random else ""
        plt.figure(figsize=(8, 5))
        plt.errorbar(results["k"], results["accuracy"], yerr=results["std"],
                     marker="s", capsize=3, label="Learned Latents", color="tab:blue", linewidth=2)
        plt.errorbar(baselines["HRR"]["k"], baselines["HRR"]["accuracy"],
                     yerr=baselines["HRR"]["std"], marker="^", capsize=3,
                     label="HRR (Random)", color="tab:gray", linestyle="--", alpha=0.8)
        plt.errorbar(baselines["unitary"]["k"], baselines["unitary"]["accuracy"],
                     yerr=baselines["unitary"]["std"], marker="v", capsize=3,
                     label="Random Unitary", color="tab:green", linestyle="--", alpha=0.8)
        plt.xlabel("Number of Bundled Role-Filler Pairs ($k$)")
        plt.ylabel("Unbinding Accuracy")
        plt.title(f"Role-Filler Query Capacity{bind_label} ($d={display_d}$, $N={n_items}$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "role_filler_capacity.png"), dpi=500)
        plt.close()

    return results



def test_per_class_bundle_capacity_k_items(
    d: int = 1024,
    n_items: int = 1000,
    n_classes: int = 10,
    items_per_class: int = 2,
    n_trials: int = 1,
    normalize: bool = True,
    device: str = "cpu",
    plot: bool = False,
    save_dir: Optional[str] = None,
    item_memory: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    item_images: Optional[torch.Tensor] = None,
    use_braiding: bool = False,
    per_class_braid: bool = False,
    class_names: Optional[list] = None,
) -> Dict:
    """
    bundle k items from each of the first n_classes from available classes...
    then, computes similarity matrix showing similarities between all pairs

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

    perm_dict = {}
    if use_braiding:
        print(f"  applying braiding to item memory...")
        braided_memory = torch.zeros_like(item_memory)
        if per_class_braid:
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
            for i in range(n_items):
                perm = torch.randperm(d, device=device)
                perm_dict[i] = perm
                braided_memory[i] = permute_vector(item_memory[i], perm)
        item_memory = braided_memory

    unique_classes = torch.unique(labels).cpu().numpy()
    if len(unique_classes) < n_classes:
        print(f"warning: only {len(unique_classes)} classes found, need {n_classes}")
        n_classes = len(unique_classes)

    class_to_items = {}
    for class_id in unique_classes[:n_classes]:
        class_indices = torch.where(labels == class_id)[0]
        if len(class_indices) >= items_per_class:
            class_to_items[class_id] = class_indices

    valid_classes = [c for c in unique_classes[:n_classes] if c in class_to_items]
    if len(valid_classes) < n_classes:
        print(f"warning: only {len(valid_classes)} classes have enough items")
        n_classes = len(valid_classes)

    similarity_matrices = []
    last_bundle_img_indices = []

    print(
        f"Computing similarity matrix for {items_per_class} across {n_classes} classes..."
    )
    for trial in range(n_trials):
        if n_trials > 1:
            print(f"    trial {trial + 1}/{n_trials}...", end=" ", flush=True)
        selected_bundles = []
        bundle_labels = []
        bundle_img_indices = []

        for class_id in valid_classes:
            class_indices = class_to_items[class_id]
            # use first items_per_class items per class for deterministic 1:1 comparability
            selected_indices = class_indices[:items_per_class]

            for idx in selected_indices:
                item_vector = item_memory[idx]
                selected_bundles.append(item_vector)
                bundle_labels.append(class_id)
                bundle_img_indices.append(idx.item())

        if len(selected_bundles) < n_classes * items_per_class:
            continue

        n_bundles = len(selected_bundles)
        similarity_matrix = np.zeros((n_bundles, n_bundles))

        bundles_tensor = torch.stack(selected_bundles)

        for i in range(n_bundles):
            sims = similarity(bundles_tensor[i].unsqueeze(0), bundles_tensor)
            similarity_matrix[i, :] = sims.cpu().numpy()

        similarity_matrices.append(similarity_matrix)
        if n_trials > 1:
            print(f"***")
        last_bundle_img_indices = bundle_img_indices

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
            from matplotlib.gridspec import GridSpec

            os.makedirs(save_dir, exist_ok=True)

            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(1, 2, width_ratios=[1, 0.5], wspace=0.3)

            ax_sim = fig.add_subplot(gs[0])
            im = ax_sim.imshow(avg_similarity, cmap="viridis", aspect="auto")
            if per_class_braid:
                braid_label = " (Per-Class Braiding)"
            elif use_braiding:
                braid_label = " (Random Braiding)"
            else:
                braid_label = ""
            ax_sim.set_title(
                f"Bundle Similarity Matrix{braid_label}\n({items_per_class} Item per Class, {n_classes} Classes)",
                fontsize=14,
                fontweight="bold",
            )

            tick_labels = []
            for i, cls_id in enumerate(valid_classes):
                name = class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else str(int(cls_id))
                if items_per_class == 1:
                    tick_labels.append(name)
                else:
                    for j in range(items_per_class):
                        tick_labels.append(f"{name}.{j + 1}")

            ax_sim.set_xticks(range(len(tick_labels)))
            ax_sim.set_yticks(range(len(tick_labels)))
            ax_sim.set_xticklabels(tick_labels, rotation=90)
            ax_sim.set_yticklabels(tick_labels)
            ax_sim.set_xlabel("Bundle Index", fontsize=12)
            ax_sim.set_ylabel("Bundle Index", fontsize=12)
            plt.colorbar(im, ax=ax_sim, label="cosine similarity")

            ax_images = fig.add_subplot(gs[1])
            ax_images.axis("off")

            if item_images is not None and len(last_bundle_img_indices) > 0:
                is_grayscale = item_images[0].shape[0] == 1
                img_h, img_w = item_images[0].shape[-2], item_images[0].shape[-1]

                # layout: items_per_class columns, n_classes rows
                canvas_h = n_classes * img_h
                canvas_w = items_per_class * img_w

                if is_grayscale:
                    canvas = np.ones((canvas_h, canvas_w)) * 0.5
                else:
                    canvas = np.ones((canvas_h, canvas_w, 3)) * 0.5

                for idx, img_idx in enumerate(last_bundle_img_indices):
                    row = idx // items_per_class
                    col = idx % items_per_class
                    img = item_images[img_idx]
                    img = (img * 0.5 + 0.5).clamp(0, 1)
                    if img.shape[0] == 1:
                        img_np = img.squeeze(0).cpu().numpy()
                    else:
                        img_np = img.permute(1, 2, 0).cpu().numpy()
                    y_start = row * img_h
                    x_start = col * img_w

                    if is_grayscale:
                        canvas[y_start : y_start + img_h, x_start : x_start + img_w] = (
                            img_np
                        )
                    else:
                        canvas[
                            y_start : y_start + img_h, x_start : x_start + img_w, :
                        ] = img_np

                if is_grayscale:
                    ax_images.imshow(canvas, cmap="gray")
                else:
                    ax_images.imshow(canvas)

                ax_images.set_title(
                    f"Images ({n_classes} Classes $\\times$ {items_per_class} Items)",
                    fontsize=12,
                    fontweight="bold",
                )

            if per_class_braid:
                filename = "bundle_similarity_matrix_per_class_braid.png"
            elif use_braiding:
                filename = "bundle_similarity_matrix_braid.png"
            else:
                filename = "bundle_similarity_matrix.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=500)
            plt.close()
            print(f" Saved plots for bundle conf matrix to {save_dir}/{filename}")

        return results

    return {"avg_similarity_matrix": None}

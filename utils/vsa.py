import os
import math
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt


def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft(
        torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1
    ).real


def vsa_invert(a: torch.Tensor) -> torch.Tensor:
    head = a[..., :1]
    tail = a[..., 1:]
    return torch.cat([head, torch.flip(tail, dims=[-1])], dim=-1)


def _unbind(ab: torch.Tensor, b: torch.Tensor, method: str = "pseudo") -> torch.Tensor:
    if method == "pseudo":
        return _bind(ab, vsa_invert(b))
    elif method == "deconv":
        Fab = torch.fft.fft(ab, dim=-1)
        Fb = torch.fft.fft(b, dim=-1)
        rec = torch.fft.ifft(Fab / (Fb + 1e-12), dim=-1).real
        return rec


def _fft_make_unitary(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Fx = torch.fft.fft(x, dim=-1)
    mags = torch.abs(Fx)
    Fx_unit = Fx / torch.clamp(mags, min=eps)
    return torch.fft.ifft(Fx_unit, dim=-1).real


def vsa_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _bind(a, b)


def vsa_unbind(ab: torch.Tensor, b: torch.Tensor, method: str = "pseudo") -> torch.Tensor:
    return _unbind(ab, b, method=method)


def _extract_latent_mu(model, x: torch.Tensor):
    out = model(x)
    if isinstance(out, (tuple, list)):
        if len(out) == 4 and isinstance(out[0], tuple):
            (z_mean, _), _, _, _ = out
            return z_mean
        elif len(out) == 4:
            _, _, _, mu = out
            return mu
        else:
            return out[-1]
    return out


_CLASS_NAMES = {
    "fashionmnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}


def _resolve_class_names(dataset_name: str, count: int) -> list:
    key = (dataset_name or "").lower()
    names = _CLASS_NAMES.get(key, [str(i) for i in range(count)])
    return names[:count]


@torch.no_grad()
def test_self_binding(
    model,
    loader,
    device,
    output_dir,
    k_self_bind: int = 50,
    unbind_method: str = "pseudo",
):
    try:
        model.eval()
        with torch.no_grad():
            all_z = []
            all_labels = []
            for x, y in loader:
                x = x.to(device)
                out = model(x)
                if isinstance(out, (tuple, list)):
                    if len(out) == 4 and isinstance(out[0], tuple):
                        (z_mean, _), _, z, _ = out
                    elif len(out) == 4:
                        _, q_z, _, mu = out
                        z = q_z.rsample() if getattr(model, "distribution", None) == "clifford" else mu
                    else:
                        z = out[-1]
                else:
                    z = out
                all_z.append(z.detach())
                all_labels.append(y)
                if len(torch.cat(all_z, 0)) >= 100:
                    break

            if not all_z:
                return {"binding_k_self_similarity": 0.0, "similarity_after_k_binds_plot_path": None}

            all_z = torch.cat(all_z, 0)
            all_labels = torch.cat(all_labels, 0)

            unique_labels = torch.unique(all_labels)[:3]
            selected_z = []
            selected_labels = []

            for label in unique_labels:
                label_mask = all_labels == label
                if label_mask.sum() > 0:
                    indices = torch.where(label_mask)[0]
                    random_idx = indices[torch.randint(0, len(indices), (1,))]
                    selected_z.append(all_z[random_idx])
                    selected_labels.append(label.item())

            if not selected_z:
                selected_z = [all_z[0]]
                selected_labels = [all_labels[0].item()]

    except Exception:
        return {"binding_k_self_similarity": 0.0, "similarity_after_k_binds_plot_path": None}

    if getattr(model, "distribution", None) == "gaussian":
        selected_z = [torch.nn.functional.normalize(z.unsqueeze(0), p=2, dim=-1) for z in selected_z]
    else:
        selected_z = [z.unsqueeze(0) for z in selected_z]

    a = selected_z[0]
    ab = a.clone()
    for _ in range(k_self_bind):
        ab = _bind(ab, a)
    for _ in range(k_self_bind):
        ab = _unbind(ab, a, method=unbind_method)
    cos_sim = torch.nn.functional.cosine_similarity(ab, a, dim=-1).mean().item()

    sims = []
    recon_every = 10
    all_recon_vectors = []
    recon_steps = []

    for m in range(1, k_self_bind + 1):
        cur = a.clone()
        for _ in range(m):
            cur = _bind(cur, a)
        for _ in range(m):
            cur = _unbind(cur, a, method=unbind_method)
        sim_m = torch.nn.functional.cosine_similarity(cur, a, dim=-1).mean().item()
        sims.append(sim_m)

    for i, start_vec in enumerate(selected_z):
        recon_vectors_for_this_start = [start_vec.squeeze(0)]
        for m in range(1, k_self_bind + 1):
            cur = start_vec.clone()
            for _ in range(m):
                cur = _bind(cur, start_vec)
            for _ in range(m):
                cur = _unbind(cur, start_vec, method=unbind_method)
            if (m % recon_every == 0) or (m == k_self_bind):
                recon_vectors_for_this_start.append(cur.squeeze(0))
                if i == 0:
                    recon_steps.append(m)
        all_recon_vectors.append(recon_vectors_for_this_start)

    path_bind_curve = os.path.join(output_dir, f"similarity_after_k_binds_{unbind_method}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    xs = np.arange(1, k_self_bind + 1)
    plt.plot(xs, sims, marker="o")
    plt.ylim(0.0, 1.05)
    plt.xlabel("m (bind m times then unbind m times)")
    plt.ylabel("Cosine similarity to original")
    plt.title("Similarity After K Binds")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_bind_curve, dpi=200, bbox_inches="tight")
    plt.close()

    recon_paths = None
    try:
        if all_recon_vectors:
            recon_paths = os.path.join(output_dir, f"recon_after_k_binds_{unbind_method}.png")
            all_vectors = []
            for recon_vectors_for_start in all_recon_vectors:
                for vec in recon_vectors_for_start:
                    all_vectors.append(vec)
            with torch.no_grad():
                imgs = model.decoder(torch.stack(all_vectors, 0))
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'output_activation'):
                    if model.decoder.output_activation == 'sigmoid':
                        imgs = torch.sigmoid(imgs) # this detects if it's MNIST or not
                    else:
                        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                else:
                    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                imgs = imgs.cpu()

            C, h, w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
            n_rows = len(all_recon_vectors)
            n_cols = len(all_recon_vectors[0]) if all_recon_vectors else 1
            canvas = torch.zeros(C, n_rows * h, n_cols * w)
            img_idx = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if img_idx < len(imgs):
                        canvas[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = imgs[img_idx]
                        img_idx += 1
            plt.figure(figsize=(max(12, n_cols * 1.5), max(6, n_rows * 2)))
            if C == 1:
                plt.imshow(canvas.squeeze(0), cmap="gray")
            else:
                plt.imshow(canvas.permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])
            plt.title("Reconstructions after bind+unbind m times")
            plt.tight_layout()
            plt.savefig(recon_paths, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception:
        recon_paths = None

    return {
        "binding_k_self_similarity": cos_sim,
        "similarity_after_k_binds_plot_path": path_bind_curve,
        "recon_after_k_binds_plot_path": recon_paths,
    }


@torch.no_grad()
def test_vsa_operations(
    model,
    loader,
    device,
    output_dir,
    n_test_pairs: int = 50,
    unbind_method: str = "pseudo",
):
    model.eval()

    latents = []
    for x, _ in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)
        latents.append(mu.detach())
        if len(torch.cat(latents, 0)) >= n_test_pairs * 2:
            break

    if not latents:
        return {"vsa_bind_unbind_similarity": 0.0, "vsa_bind_unbind_plot": None}

    z_all = torch.cat(latents, 0)[: n_test_pairs * 2]
    if z_all.shape[0] < 2:
        return {"vsa_bind_unbind_similarity": 0.0, "vsa_bind_unbind_plot": None}

    z_all = torch.nn.functional.normalize(z_all, p=2, dim=-1)

    single_bind_sims = []
    for i in range(min(n_test_pairs, z_all.shape[0] // 2)):
        key_idx = np.random.randint(z_all.shape[0])
        key = z_all[key_idx]
        value = z_all[i]
        a = key.unsqueeze(0)
        b = value.unsqueeze(0)
        bound = _bind(a, b)
        recovered = _unbind(bound, a, method=unbind_method)
        sim = torch.nn.functional.cosine_similarity(
            recovered, value.unsqueeze(0), dim=-1
        ).item()
        single_bind_sims.append(sim)

    avg_single_sim = float(np.mean(single_bind_sims)) if single_bind_sims else 0.0

    path_vsa_test = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        if single_bind_sims:
            path_vsa_test = os.path.join(output_dir, f"vsa_bind_unbind_{unbind_method}.png")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(single_bind_sims, bins=20, alpha=0.7, edgecolor="black")
            plt.axvline(np.mean(single_bind_sims), color="red", linestyle="--", label=f"Mean: {np.mean(single_bind_sims):.3f}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Count")
            plt.title("Bind-Unbind Performance")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(single_bind_sims, "o-", alpha=0.7, markersize=4)
            plt.axhline(np.mean(single_bind_sims), color="red", linestyle="--", alpha=0.8)
            plt.xlabel("Test Index")
            plt.ylabel("Cosine Similarity")
            plt.title("Per-Test Similarity")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path_vsa_test, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: VSA bind/unbind plotting failed: {e}")

    return {"vsa_bind_unbind_similarity": avg_single_sim, "vsa_bind_unbind_plot": path_vsa_test}


@torch.no_grad()
def test_hrr_sentence(
    model,
    loader,
    device,
    output_dir,
    unbind_method: str = "pseudo",
    dataset_name: str = None,
):
""" basically, we normalize the vectors, bind them with a unitary (randomly generated) vector, 
and then unbind them with the same unitary vector. if similarity is high, then the vector is well-bound. 
"""
    model.eval()

    by_label = defaultdict(list)
    for x, y in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)
        for vec, lbl in zip(mu.detach(), y.tolist()):
            by_label[lbl].append(vec)
        if sum(len(v) for v in by_label.values()) >= 1000:
            break

    if not by_label or any(len(v) == 0 for v in by_label.values() if v is not None):
        return {"hrr_fashion_object_acc": 0.0, "hrr_fashion_plot": None}

    class_vecs = []
    valid_labels = []
    for k in range(10):
        if len(by_label[k]) > 0:
            m = torch.stack(by_label[k], 0).mean(0)
            class_vecs.append(m)
            valid_labels.append(k)
    if len(class_vecs) < 2:
        return {"hrr_fashion_object_acc": 0.0, "hrr_fashion_plot": None}
    class_vecs = torch.stack(class_vecs, 0)
    class_vecs = torch.nn.functional.normalize(class_vecs, p=2, dim=-1)

    D = class_vecs.shape[-1]
    role_item = torch.randn(D, device=device, dtype=class_vecs.dtype)
    role_container = torch.randn(D, device=device, dtype=class_vecs.dtype)
    role_item = torch.nn.functional.normalize(role_item, p=2, dim=-1)
    role_container = torch.nn.functional.normalize(role_container, p=2, dim=-1)
    role_item = _fft_make_unitary(role_item)
    role_container = _fft_make_unitary(role_container)

    class_names = _resolve_class_names(dataset_name or "", 10)

    all_correct = 0
    total_tests = 0
    all_plots = []

    for test_item_idx in range(len(class_vecs)):
        test_container_idx = (test_item_idx + 1) % len(class_vecs)

        item_vec = class_vecs[test_item_idx].unsqueeze(0)
        container_vec = class_vecs[test_container_idx].unsqueeze(0)

        a = role_item.unsqueeze(0)
        c = role_container.unsqueeze(0)
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        c = torch.nn.functional.normalize(c, p=2, dim=-1)
        item_vec = torch.nn.functional.normalize(item_vec, p=2, dim=-1)
        container_vec = torch.nn.functional.normalize(container_vec, p=2, dim=-1)

        memory = _bind(a, item_vec) + _bind(c, container_vec)
        recovered_item = _unbind(memory, a, method=unbind_method).squeeze(0)
        recovered_item = torch.nn.functional.normalize(recovered_item, p=2, dim=-1)

        sims = torch.nn.functional.cosine_similarity(
            recovered_item.unsqueeze(0), class_vecs, dim=-1
        )
        best = int(torch.argmax(sims).item())
        is_correct = 1.0 if best == test_item_idx else 0.0
        all_correct += is_correct
        total_tests += 1

        if test_item_idx < 3:
            all_plots.append({
                'sims': sims.detach().cpu().numpy(),
                'expected_idx': test_item_idx,
                'predicted_idx': best,
                'expected_label': class_names[valid_labels[test_item_idx]],
                'predicted_label': class_names[valid_labels[best]],
                'correct': is_correct
            })

    avg_accuracy = all_correct / max(1, total_tests)

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        tag = (dataset_name or "dataset").lower()
        path = os.path.join(output_dir, f"hrr_{tag}_all_classes_{unbind_method}.png")

        n_plots = len(all_plots)
        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
            if n_plots == 1:
                axes = [axes]

            for i, plot_info in enumerate(all_plots):
                ax = axes[i]
                heights = plot_info['sims']
                colors = ["C0"] * len(heights)
                colors[plot_info['expected_idx']] = "green"

                bars = ax.bar(np.arange(len(valid_labels)), heights, alpha=0.8, color=colors)
                try:
                    bars[plot_info['predicted_idx']].set_edgecolor("red")
                    bars[plot_info['predicted_idx']].set_linewidth(2.0)
                except Exception:
                    pass

                labels_subset = [class_names[valid_labels[j]] for j in range(len(valid_labels))]
                ax.set_xticks(np.arange(len(labels_subset)))
                ax.set_xticklabels(labels_subset, rotation=30, ha="right")
                ax.set_ylabel("cosine sim")

                status = "✓" if plot_info['correct'] else "✗"
                ax.set_title(f"{status} Expected: {plot_info['expected_label']}\nPredicted: {plot_info['predicted_label']}")

            plt.suptitle(f"HRR Decode Test - All Classes (Avg Acc: {avg_accuracy:.2f})")
            plt.tight_layout()
            plt.savefig(path, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception:
        path = None

    return {"hrr_fashion_object_acc": float(avg_accuracy), "hrr_fashion_plot": path}


@torch.no_grad()
def test_hrr_fashionmnist_sentence(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("dataset_name", "fashionmnist")
    return test_hrr_sentence(*args, **kwargs)


@torch.no_grad()
def test_bundle_capacity(
    model,
    loader,
    device,
    output_dir,
    n_items: int = 1000,
    k_range: list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    n_trials: int = 10,
):
    model.eval()

    latents = []
    while len(torch.cat(latents, 0) if latents else []) < n_items:
        for x, _ in loader:
            x = x.to(device)
            mu = _extract_latent_mu(model, x)
            latents.append(mu.detach())
            if len(torch.cat(latents, 0)) >= n_items:
                break
        if not loader:
            break

    if not latents or len(torch.cat(latents, 0)) < (max(k_range) if k_range else 0):
        return {"bundle_capacity_plot": None, "bundle_capacity_accuracies": {}}

    item_memory = torch.cat(latents, 0)[:n_items]
    item_memory = torch.nn.functional.normalize(item_memory, p=2, dim=-1)

    accuracies = {}
    for k in k_range:
        if k > n_items:
            continue

        trial_accuracies = []
        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[:k]
            chosen_vectors = item_memory[indices]

            bundle = torch.sum(chosen_vectors, dim=0) / math.sqrt(k)

            bundle_norm = torch.nn.functional.normalize(bundle, p=2, dim=-1)
            sims = torch.matmul(item_memory, bundle_norm)

            top_k_indices = torch.topk(sims, k).indices

            retrieved_indices = set(top_k_indices.cpu().numpy())
            original_indices = set(indices.cpu().numpy())

            correctly_retrieved = len(retrieved_indices.intersection(original_indices))
            accuracy = correctly_retrieved / k
            trial_accuracies.append(accuracy)

        accuracies[k] = np.mean(trial_accuracies)

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "bundle_capacity.png")
        plt.figure(figsize=(7, 5))
        ks = sorted(accuracies.keys())
        accs = [accuracies[k_] for k_ in ks]
        plt.plot(ks, accs, marker='o')
        plt.xlabel("Number of Bundled Vectors (k)")
        plt.ylabel("Retrieval Accuracy")
        plt.title(f"Bundling Capacity (D={item_memory.shape[1]}, N={n_items})")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot bundle capacity: {e}")
        path = None

    return {"bundle_capacity_plot": path, "bundle_capacity_accuracies": accuracies}


@torch.no_grad()
def test_unbinding_of_bundled_pairs(
    model,
    loader,
    device,
    output_dir,
    unbind_method: str = "pseudo",
    n_items: int = 1000,
    k_range: list = [2, 5, 10, 15, 20],
    n_trials: int = 10,
):
    model.eval()

    latents = []
    while len(torch.cat(latents, 0) if latents else []) < n_items:
        for x, _ in loader:
            x = x.to(device)
            mu = _extract_latent_mu(model, x)
            latents.append(mu.detach())
            if len(torch.cat(latents, 0)) >= n_items:
                break
        if not latents:
            break

    required_vecs = (max(k_range) * 2) if k_range else 0
    if not latents or len(torch.cat(latents, 0)) < required_vecs:
        return {"unbind_bundled_plot": None, "unbind_bundled_accuracies": {}}

    item_memory = torch.cat(latents, 0)[:n_items]
    item_memory = torch.nn.functional.normalize(item_memory, p=2, dim=-1)

    accuracies = {}
    for k in k_range:
        if k * 2 > n_items:
            continue

        trial_accuracies = []
        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[:k*2]

            D = item_memory.shape[-1]
            roles = torch.randn(k, D, device=device, dtype=item_memory.dtype)
            roles = torch.nn.functional.normalize(roles, p=2, dim=-1)
            roles = _fft_make_unitary(roles)

            fillers = item_memory[indices[k:2*k]]

            bound_pairs = _bind(roles, fillers)
            bundle = torch.sum(bound_pairs, dim=0) / math.sqrt(k)

            recovered_fillers = _unbind(bundle.unsqueeze(0), roles, method=unbind_method)
            best_matches = torch.argmax(torch.nn.functional.cosine_similarity(recovered_fillers.unsqueeze(1), item_memory.unsqueeze(0), dim=-1), dim=1)

            correctly_recovered = 0
            for i in range(k):
                original_filler_idx = indices[k+i]
                if best_matches[i] == original_filler_idx:
                    correctly_recovered += 1

            trial_accuracies.append(correctly_recovered / k)

        accuracies[k] = np.mean(trial_accuracies)

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"unbind_bundled_pairs_{unbind_method}.png")
        plt.figure(figsize=(7, 5))
        ks = sorted(accuracies.keys())
        accs = [accuracies[k_] for k_ in ks]
        plt.plot(ks, accs, marker='o')
        plt.xlabel("Number of Bundled Pairs (k)")
        plt.ylabel("Filler Retrieval Accuracy")
        plt.title(f"Unbinding of Bundled Pairs (D={item_memory.shape[1]}, N={n_items})")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot unbinding of bundled pairs: {e}")
        path = None

    return {"unbind_bundled_plot": path, "unbind_bundled_accuracies": accuracies}



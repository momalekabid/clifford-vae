import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    import wandb
except Exception:
    wandb = None

from .vsa import bind, unbind, invert


def test_self_binding(
    model,
    loader,
    device,
    output_dir,
    k_self_bind: int = 50,
    unbind_method: str = "*",
    img_shape=(1, 28, 28),
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
                        z = (
                            q_z.rsample()
                            if getattr(model, "distribution", None) == "clifford"
                            else mu
                        )
                    else:
                        z = out[-1]
                else:
                    z = out
                all_z.append(z.detach())
                all_labels.append(y)
                if len(torch.cat(all_z, 0)) >= 100:
                    break

            if not all_z:
                return {
                    "binding_k_self_similarity": 0.0,
                    "similarity_after_k_binds_plot_path": None,
                }

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
        return {
            "binding_k_self_similarity": 0.0,
            "similarity_after_k_binds_plot_path": None,
        }

    if getattr(model, "distribution", None) == "gaussian":
        selected_z = [
            torch.nn.functional.normalize(z.unsqueeze(0), p=2, dim=-1)
            for z in selected_z
        ]
    else:
        selected_z = [z.unsqueeze(0) for z in selected_z]

    a = selected_z[0]
    ab = a.clone()
    for _ in range(k_self_bind):
        ab = bind(ab, a)
    for _ in range(k_self_bind):
        ab = unbind(ab, a, method=unbind_method)
    cos_sim = torch.nn.functional.cosine_similarity(ab, a, dim=-1).mean().item()

    sims = []
    recon_every = 10
    all_recon_vectors = []
    recon_steps = []

    for m in range(1, k_self_bind + 1):
        cur = a.clone()
        for _ in range(m):
            cur = bind(cur, a)
        for _ in range(m):
            cur = unbind(cur, a, method=unbind_method)
        sim_m = torch.nn.functional.cosine_similarity(cur, a, dim=-1).mean().item()
        sims.append(sim_m)

    for i, start_vec in enumerate(selected_z):
        recon_vectors_for_this_start = [start_vec.squeeze(0)]
        for m in range(1, k_self_bind + 1):
            cur = start_vec.clone()
            for _ in range(m):
                cur = bind(cur, start_vec)
            for _ in range(m):
                cur = unbind(cur, start_vec, method=unbind_method)
            if (m % recon_every == 0) or (m == k_self_bind):
                recon_vectors_for_this_start.append(cur.squeeze(0))
                if i == 0:
                    recon_steps.append(m)
        all_recon_vectors.append(recon_vectors_for_this_start)

    path_bind_curve = os.path.join(
        output_dir, f"similarity_after_k_binds_{unbind_method}.png"
    )
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

    try:
        if all_recon_vectors:
            recon_paths = os.path.join(
                output_dir, f"recon_after_k_binds_{unbind_method}.png"
            )

            all_vectors = []
            row_labels = []

            for i, recon_vectors_for_start in enumerate(all_recon_vectors):
                for vec in recon_vectors_for_start:
                    all_vectors.append(vec)
                class_label = selected_labels[i] if i < len(selected_labels) else i
                row_labels.append(f"Class {class_label}")

            with torch.no_grad():
                imgs = model.decoder(torch.stack(all_vectors, 0))
                if hasattr(model, "decoder") and hasattr(
                    model.decoder, "output_activation"
                ):
                    if model.decoder.output_activation == "sigmoid":
                        imgs = torch.sigmoid(imgs)
                    else:
                        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                else:
                    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                # reshape flat output to image dimensions
                imgs = imgs.view(-1, *img_shape)
                imgs = imgs.cpu()

            C, h, w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
            n_rows = len(all_recon_vectors)
            n_cols = len(all_recon_vectors[0]) if all_recon_vectors else 1

            canvas = torch.zeros(C, n_rows * h, n_cols * w)

            img_idx = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if img_idx < len(imgs):
                        canvas[
                            :, row * h : (row + 1) * h, col * w : (col + 1) * w
                        ] = imgs[img_idx]
                        img_idx += 1

            plt.figure(figsize=(max(12, n_cols * 1.5), max(6, n_rows * 2)))
            if C == 1:
                plt.imshow(canvas.squeeze(0), cmap="gray")
            else:
                plt.imshow(canvas.permute(1, 2, 0))

            # is this even used
            col_labels = [f"m=0"] + [f"m={m}" for m in recon_steps]
            plt.xticks([])
            plt.yticks([])

            class_info = ", ".join(
                [
                    f"Class {label}"
                    for label in selected_labels[: len(all_recon_vectors)]
                ]
            )
            plt.title(f"Reconstructions after bind+unbind m times\nRows: {class_info}")
            plt.tight_layout()
            plt.savefig(recon_paths, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(e)
        recon_paths = None

    return {
        "binding_k_self_similarity": cos_sim,
        "similarity_after_k_binds_plot_path": path_bind_curve,
        "recon_after_k_binds_plot_path": recon_paths,
    }


def test_cross_class_bind_unbind(
    model,
    loader,
    device,
    output_dir,
    unbind_method: str = "*",
    img_shape=(1, 28, 28),
):
    """
    Test binding/unbinding different classes, with reconstructions.
    """
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
                        z = (
                            q_z.rsample()
                            if getattr(model, "distribution", None) == "clifford"
                            else mu
                        )
                    else:
                        z = out[-1]
                else:
                    z = out
                all_z.append(z.detach())
                all_labels.append(y)
                if len(torch.cat(all_z, 0)) >= 200:
                    break

            if not all_z:
                return {
                    "cross_class_bind_unbind_similarity": 0.0,
                    "cross_class_bind_unbind_plot_path": None,
                }

            all_z = torch.cat(all_z, 0)
            all_labels = torch.cat(all_labels, 0)

            unique_labels = torch.unique(all_labels)
            if len(unique_labels) < 2:
                return {
                    "cross_class_bind_unbind_similarity": 0.0,
                    "cross_class_bind_unbind_plot_path": None,
                }

            class_a_label = unique_labels[0]
            class_b_label = unique_labels[1]

            class_a_mask = all_labels == class_a_label
            class_b_mask = all_labels == class_b_label

            if class_a_mask.sum() == 0 or class_b_mask.sum() == 0:
                return {
                    "cross_class_bind_unbind_similarity": 0.0,
                    "cross_class_bind_unbind_plot_path": None,
                }

            class_a_indices = torch.where(class_a_mask)[0]
            class_b_indices = torch.where(class_b_mask)[0]

            a_idx = class_a_indices[torch.randint(0, len(class_a_indices), (1,))]
            b_idx = class_b_indices[torch.randint(0, len(class_b_indices), (1,))]

            a = all_z[a_idx]
            b = all_z[b_idx]

    except Exception:
        return {
            "cross_class_bind_unbind_similarity": 0.0,
            "cross_class_bind_unbind_plot_path": None,
        }

    if getattr(model, "distribution", None) == "gaussian":
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)

    # bind a and b
    ab = bind(a, b)

    # to recover b (ab ⊛ a^-1 = b)
    recovered_b = unbind(ab, a, method=unbind_method)

    #  to recover a (ab ⊛ b^-1 = a)
    recovered_a = unbind(ab, b, method=unbind_method)

    # calculate similarities
    sim_b = torch.nn.functional.cosine_similarity(recovered_b, b, dim=-1).mean().item()
    sim_a = torch.nn.functional.cosine_similarity(recovered_a, a, dim=-1).mean().item()

    avg_sim = (sim_a + sim_b) / 2.0

    # create reconstruction visualizations if we have a decoder
    plot_path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"cross_class_bind_unbind_{unbind_method}.png"
        )

        with torch.no_grad():
            vectors_to_decode = torch.stack(
                [
                    a.squeeze(0),
                    b.squeeze(0),
                    recovered_a.squeeze(0),
                    recovered_b.squeeze(0),
                ],
                0,
            )
            imgs = model.decoder(vectors_to_decode)

            if hasattr(model, "decoder") and hasattr(
                model.decoder, "output_activation"
            ):
                if model.decoder.output_activation == "sigmoid":
                    imgs = torch.sigmoid(imgs)
                else:
                    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            else:
                imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            # reshape flat output to image dimensions
            imgs = imgs.view(-1, *img_shape)
            imgs = imgs.cpu()

        C, h, w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
        canvas = torch.zeros(C, h, 4 * w)

        labels = [
            f"A (class {class_a_label.item()})",
            f"B (class {class_b_label.item()})",
            f"Recovered A (sim: {sim_a:.3f})",
            f"Recovered B (sim: {sim_b:.3f})",
        ]

        for i in range(4):
            canvas[:, :, i * w : (i + 1) * w] = imgs[i]

        plt.figure(figsize=(12, 3))
        if C == 1:
            plt.imshow(canvas.squeeze(0), cmap="gray")
        else:
            plt.imshow(canvas.permute(1, 2, 0))

        plt.xticks([w // 2 + i * w for i in range(4)], labels, rotation=15, ha="right")
        plt.yticks([])
        plt.title(f"Cross-Class Bind/Unbind Test (Avg Sim: {avg_sim:.3f})")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(e)
        plot_path = None

    return {
        "cross_class_bind_unbind_similarity": avg_sim,
        "cross_class_bind_unbind_plot_path": plot_path,
    }


class WandbLogger:
    def __init__(self, args):
        self.use = (wandb is not None) and (not args.no_wandb)
        if self.use:
            self.project = args.wandb_project
            self.run = None

    def start_run(self, name, args):
        if self.use:
            self.run = wandb.init(project=self.project, name=name, config=vars(args))

    def watch_model(self, model):
        if self.use:
            wandb.watch(model, log="gradients", log_freq=100)

    def log_metrics(self, d):
        if self.use and self.run is not None:
            try:
                self.run.log(d)
            except Exception:
                pass

    def log_summary(self, d):
        if self.use and self.run is not None:
            try:
                self.run.summary.update(d)
            except Exception:
                pass

    def log_images(self, images):
        if self.use and self.run is not None:
            try:
                to_log = {}
                for k, v in images.items():
                    if isinstance(v, str) and os.path.exists(v):
                        to_log[k] = wandb.Image(v)
                    else:
                        to_log[k] = v
                self.run.log(to_log)
            except Exception:
                pass

    def finish_run(self):
        if self.use and self.run is not None:
            self.run.finish()


def _extract_latent_mu(model, x: torch.Tensor):
    out = model(x)
    if isinstance(out, (tuple, list)):
        if len(out) == 4 and isinstance(out[0], tuple):
            # ((z_mean, z_param2), (q_z,p_z), z, x_recon)
            (z_mean, _), _, _, _ = out
            return z_mean
        elif len(out) == 4:
            # (x_recon, q_z, p_z, mu)
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


@torch.no_grad()
def compute_class_means(model, loader, device, max_per_class: int = 1000):
    model.eval()
    sums = {}
    counts = {}
    dist_type = getattr(model, "distribution", "normal")

    for x, y in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)
        mu = mu.detach()
        for i, label in enumerate(y.tolist()):
            if label not in counts:
                counts[label] = 0
                sums[label] = torch.zeros_like(mu[i])
            if counts[label] < max_per_class:
                sums[label] = sums[label] + mu[i]
                counts[label] += 1

    means = {}
    for label, total in sums.items():
        c = max(1, min(counts[label], 10))
        vec = total / c
        if dist_type == "powerspherical":
            vec = torch.nn.functional.normalize(
                vec, p=2, dim=-1
            )  # should already be normalized, just in case (unit-sphere)
        means[label] = vec
    return means


@torch.no_grad()
def evaluate_mean_vector_cosine(model, loader, device, class_means: dict):
    model.eval()
    labels_sorted = sorted(class_means.keys())
    mean_vector = torch.stack([class_means[k] for k in labels_sorted], dim=0).to(device)
    dist_type = getattr(model, "distribution", "normal")

    correct = 0
    total = 0
    per_class_correct = {k: 0 for k in labels_sorted}
    per_class_total = {k: 0 for k in labels_sorted}

    for x, y in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)

        # use cosine similarity or dot product for all distributions
        sims = torch.nn.functional.cosine_similarity(
            mu.unsqueeze(1), mean_vector.unsqueeze(0), dim=-1
        )
        preds = sims.argmax(dim=1).cpu()

        y_cpu = y.cpu()
        correct += (preds == y_cpu).sum().item()
        total += y_cpu.numel()
        for yi, pi in zip(y_cpu.tolist(), preds.tolist()):
            per_class_total[yi] += 1
            if yi == labels_sorted[pi]:
                per_class_correct[yi] += 1

    acc = correct / max(1, total)
    per_class_acc = {
        k: (per_class_correct[k] / max(1, per_class_total[k])) for k in labels_sorted
    }
    return acc, per_class_acc


@torch.no_grad()
def plot_clifford_torus_latent_scatter(
    model, loader, device, output_dir, dims=(0, 1), dataset_name: str = None
):
    if getattr(model, "distribution", None) != "clifford" or model.latent_dim < 2:
        return None

    model.eval()
    angles = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            _, _, _, mu = out
        else:
            mu = out
        a = ((mu + math.pi) % (2 * math.pi)) - math.pi
        angles.append(a.detach().cpu())
        labels.append(y)
        if len(torch.cat(labels)) >= 4000:
            break

    A = torch.cat(angles, 0)
    Y = torch.cat(labels, 0).numpy()
    ax0, ax1 = dims
    xs = A[:, ax0].numpy()
    ys = A[:, ax1].numpy()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir, f"clifford_torus_latent_scatter_{dataset_name or 'dataset'}.png"
    )
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(xs, ys, c=Y, cmap="tab10", s=6, alpha=0.8)
    plt.colorbar(sc)
    plt.xlim(-math.pi, math.pi)
    plt.ylim(-math.pi, math.pi)
    plt.xlabel(f"angle[{ax0}]")
    plt.ylabel(f"angle[{ax1}]")
    plt.title("Clifford Torus Latent Angles")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _angles_to_clifford_vector(
    angles: torch.Tensor, normalize_ifft: bool = True
) -> torch.Tensor:
    # angles shape (..., d), produce (..., 2d) real vector following CliffordPowerSphericalDistribution mapping
    d = angles.shape[-1]
    n = 2 * d
    device = angles.device
    dtype = angles.dtype
    theta_s = torch.zeros((*angles.shape[:-1], n), device=device, dtype=dtype)
    if d > 1:
        theta_s[..., 1:d] = angles[..., 1:]
        theta_s[..., -d + 1 :] = -torch.flip(angles[..., 1:], (-1,))
    samples_c = torch.exp(1j * theta_s)
    if normalize_ifft:
        samples_c = samples_c / math.sqrt(n)
        return torch.fft.ifft(samples_c, dim=-1, norm="ortho").real
    return torch.fft.ifft(samples_c, dim=-1).real


@torch.no_grad()
def plot_clifford_torus_recon_grid(  # TODO fix /verify.. might only work for latend dims 2/4 otherwise pca to find best 2 dims to visualize
    model, device, output_dir, dims=(0, 1), n_grid: int = 16
):
    if getattr(model, "distribution", None) != "clifford" or model.latent_dim < 2:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clifford_torus_recon_grid.png")

    angles0 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    angles1 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    mesh0, mesh1 = torch.meshgrid(angles0, angles1, indexing="ij")
    A = torch.zeros(n_grid * n_grid, model.latent_dim, device=device)
    A[:, dims[0]] = mesh0.reshape(-1)
    A[:, dims[1]] = mesh1.reshape(-1)

    Z = _angles_to_clifford_vector(A, normalize_ifft=True)

    model.eval()
    imgs = model.decoder(Z).detach().cpu()
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    # make grid
    H = n_grid
    W = n_grid
    C = imgs.shape[1]
    h, w = imgs.shape[-2:]
    canvas = torch.zeros(C, H * h, W * w)
    for i in range(H):
        for j in range(W):
            canvas[:, i * h : (i + 1) * h, j * w : (j + 1) * w] = imgs[i * W + j]
    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.title("Decoder Reconstructions over Torus Grid")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


@torch.no_grad()
def test_vsa_operations(
    model,
    loader,
    device,
    output_dir,
    n_test_pairs: int = 50,
    unbind_method: str = "*",
    unitary_keys: bool = False,
    normalize_vectors: bool = True,
    project_vectors: bool = False,
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

    dist_type = getattr(model, "distribution", "normal")
    if dist_type == "powerspherical" or normalize_vectors:
        z_all = torch.nn.functional.normalize(z_all, p=2, dim=-1)

    single_bind_sims = []
    for i in range(min(n_test_pairs, z_all.shape[0] // 2)):
        key_idx = np.random.randint(z_all.shape[0])
        key = z_all[key_idx]
        value = z_all[i]
        a = key.unsqueeze(0)
        b = value.unsqueeze(0)

        bound = bind(a, b)
        recovered = unbind(bound, a, method=unbind_method)
        sim = torch.nn.functional.cosine_similarity(
            recovered, value.unsqueeze(0), dim=-1
        ).item()
        single_bind_sims.append(sim)

    avg_single_sim = float(np.mean(single_bind_sims)) if single_bind_sims else 0.0

    path_vsa_test = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        if single_bind_sims:
            path_vsa_test = os.path.join(
                output_dir, f"vsa_bind_unbind_{unbind_method}.png"
            )
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(single_bind_sims, bins=20, alpha=0.7, edgecolor="black")
            plt.axvline(
                np.mean(single_bind_sims),  # weird type error but still works
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(single_bind_sims):.3f}",
            )
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Count")
            plt.title("Bind-Unbind Performance")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(single_bind_sims, "o-", alpha=0.7, markersize=4)
            plt.axhline(
                np.mean(single_bind_sims), color="red", linestyle="--", alpha=0.8
            )
            plt.xlabel("Test Index")
            plt.ylabel("Cosine Similarity")
            plt.title("Per-Test Similarity")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path_vsa_test, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: VSA bind/unbind plotting failed: {e}")

    return {
        "vsa_bind_unbind_similarity": avg_single_sim,
        "vsa_bind_unbind_plot": path_vsa_test,
    }


def plot_clifford_manifold_visualization(
    model, device, output_dir, n_grid=12, dims=(0, 1), img_shape=(1, 28, 28)
):
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    if (
        getattr(model, "distribution", None) != "clifford"
        or latent_dim is None
        or latent_dim < 2
    ):
        return None

    os.makedirs(output_dir, exist_ok=True)
    return _plot_clifford_manifold_original(model, device, output_dir, n_grid, dims, img_shape)


def _plot_clifford_manifold_original(model, device, output_dir, n_grid=12, dims=(0, 1), img_shape=(1, 28, 28)):
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    path = os.path.join(output_dir, "clifford_manifold_visualization.png")

    model.eval()
    with torch.no_grad():
        angles0 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
        angles1 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
        mesh0, mesh1 = torch.meshgrid(angles0, angles1, indexing="ij")

        A = torch.zeros(n_grid * n_grid, latent_dim, device=device)
        A[:, dims[0]] = mesh0.reshape(-1)
        A[:, dims[1]] = mesh1.reshape(-1)

        Z = _angles_to_clifford_vector(A, normalize_ifft=True)

        x_recon = model.decoder(Z)

        if hasattr(model, "decoder") and hasattr(model.decoder, "output_activation"):
            if model.decoder.output_activation == "sigmoid":
                x_recon = torch.sigmoid(x_recon)
            elif model.decoder.output_activation == "tanh":
                x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
            else:
                x_recon = x_recon.clamp(0, 1)
        else:
            x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)

        # reshape flat output to image dimensions
        x_recon = x_recon.view(-1, *img_shape)

    C = x_recon.shape[1]
    h, w = x_recon.shape[-2:]
    canvas = torch.zeros(C, n_grid * h, n_grid * w)

    for i in range(n_grid):
        for j in range(n_grid):
            idx = i * n_grid + j
            if idx < len(x_recon):
                canvas[:, i * h : (i + 1) * h, j * w : (j + 1) * w] = x_recon[idx]

    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())

    plt.xticks([])
    plt.yticks([])
    plt.title(
        f"Clifford Torus Manifold Systematic Traversal (dims {dims[0]},{dims[1]})"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return path


def plot_powerspherical_manifold_visualization(
    model, device, output_dir, n_samples=256, dims=(0, 1), img_shape=(1, 28, 28)
):
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    if (
        getattr(model, "distribution", None) != "powerspherical"
        or latent_dim is None
        or latent_dim < 2
    ):
        return None

    os.makedirs(output_dir, exist_ok=True)
    return _plot_powerspherical_manifold_original(model, device, output_dir, n_samples, img_shape)


def _plot_powerspherical_manifold_original(model, device, output_dir, n_samples=256, img_shape=(1, 28, 28)):
    path = os.path.join(output_dir, "powerspherical_manifold_visualization.png")
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))

    # reduce to 12x12 grid to save memory (instead of 16x16)
    grid_size = 12
    n_samples = grid_size * grid_size  # 144 samples

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        x_recon = model.decoder(z)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
        # reshape flat output to image dimensions
        x_recon = x_recon.view(-1, *img_shape).cpu()
    x_recon_grid = x_recon[: grid_size * grid_size]

    C = x_recon_grid.shape[1]
    h, w = x_recon_grid.shape[-2:]
    canvas = torch.zeros(C, grid_size * h, grid_size * w)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(x_recon_grid):
                canvas[:, i * h : (i + 1) * h, j * w : (j + 1) * w] = x_recon_grid[idx]

    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())

    plt.xticks([])
    plt.yticks([])
    plt.title("PowerSpherical Manifold Reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return path


def plot_gaussian_manifold_visualization(
    model, device, output_dir, n_samples=144, dims=(0, 1), img_shape=(1, 28, 28)
):
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    dist = getattr(model, "distribution", None)
    if (
        dist not in ["gaussian", "normal"]
        or latent_dim is None
        or latent_dim < 2
    ):
        return None

    os.makedirs(output_dir, exist_ok=True)
    return _plot_gaussian_manifold_original(model, device, output_dir, n_samples, img_shape)


def _plot_gaussian_manifold_original(model, device, output_dir, n_samples=144, img_shape=(1, 28, 28)):
    path = os.path.join(output_dir, "gaussian_manifold_visualization.png")
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))

    # reduce to 12x12 grid to save memory (instead of 16x16)
    grid_size = 12
    n_samples = grid_size * grid_size  # 144 samples

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        x_recon = model.decoder(z)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
        # reshape flat output to image dimensions
        x_recon = x_recon.view(-1, *img_shape)

    x_recon_grid = x_recon[: grid_size * grid_size]

    C = x_recon_grid.shape[1]
    h, w = x_recon_grid.shape[-2:]
    canvas = torch.zeros(C, grid_size * h, grid_size * w)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(x_recon_grid):
                canvas[:, i * h : (i + 1) * h, j * w : (j + 1) * w] = x_recon_grid[idx]

    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())

    plt.xticks([])
    plt.yticks([])
    plt.title("Gaussian Manifold Reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return path


def generate_confusion_matrix_with_sample_visualisation(
    model,
    test_loader,
    device,
    save_path,
    n_samples_per_class=10,
    class_names=None,
):
    """
    generate confusion matrix with misclassified sample images showing why confusion happens

    args:
        model: trained vae model
        test_loader: dataloader for test set
        device: torch device
        save_path: path to save visualisation
        n_samples_per_class: number of misclassified samples to show per confused pair
        class_names: list of class names (optional)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_images = []

    # detect actual classes present in test_loader
    actual_classes_present = set()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            # encode to latent space
            if hasattr(model, 'encode'):
                # mlp vae
                z_mean, _ = model.encode(x.view(x.size(0), -1))
            else:
                # cnn vae
                _, _, _, z_mean = model(x)

            for i in range(len(y)):
                label = y[i].item()
                actual_classes_present.add(label)
                all_images.append(x[i].cpu())

            all_labels.extend(y.cpu().numpy())

    # only work with classes that are present
    actual_classes = sorted(actual_classes_present)

    # compute class centroids for classification
    class_centroids = defaultdict(list)
    idx = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            if hasattr(model, 'encode'):
                z_mean, _ = model.encode(x.view(x.size(0), -1))
            else:
                _, _, _, z_mean = model(x)

            for i in range(len(y)):
                label = y[i].item()
                class_centroids[label].append(z_mean[i].cpu().numpy())

    # average to get centroids
    for label in actual_classes:
        if class_centroids[label]:
            class_centroids[label] = np.mean(class_centroids[label], axis=0)

    # now compute predictions and collect misclassified examples
    all_preds = []
    misclassified = defaultdict(list)  # (true_class, pred_class) -> list of image indices

    idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            if hasattr(model, 'encode'):
                z_mean, _ = model.encode(x.view(x.size(0), -1))
            else:
                _, _, _, z_mean = model(x)

            z_mean_np = z_mean.cpu().numpy()

            # compute distances to all centroids (only for present classes)
            for i, z in enumerate(z_mean_np):
                distances = [np.linalg.norm(z - class_centroids[label]) for label in actual_classes]
                pred = actual_classes[np.argmin(distances)]
                all_preds.append(pred)

                true_label = y[i].item()
                if pred != true_label:
                    misclassified[(true_label, pred)].append(idx)

                idx += 1

    # compute confusion matrix
    from matplotlib.gridspec import GridSpec
    cm = confusion_matrix(all_labels, all_preds, labels=actual_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # get actual class names for classes present
    if class_names:
        actual_class_names = [class_names[i] for i in actual_classes]
    else:
        actual_class_names = [str(i) for i in actual_classes]

    # find top confused pairs
    confused_pairs = []
    for true_idx, true_class in enumerate(actual_classes):
        for pred_idx, pred_class in enumerate(actual_classes):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confused_pairs.append((
                    cm[true_idx, pred_idx],
                    true_class,
                    pred_class,
                    actual_class_names[true_idx],
                    actual_class_names[pred_idx]
                ))

    confused_pairs.sort(reverse=True)
    top_confused = confused_pairs[:6]  # show top 6 confused pairs

    # create figure
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # top: confusion matrix
    ax_cm = fig.add_subplot(gs[0])

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'normalized frequency'},
        ax=ax_cm,
        xticklabels=actual_class_names,
        yticklabels=actual_class_names,
    )

    ax_cm.set_xlabel('predicted class', fontsize=12, fontweight='bold')
    ax_cm.set_ylabel('true class', fontsize=12, fontweight='bold')
    ax_cm.set_title('confusion matrix', fontsize=14, fontweight='bold')

    # bottom: misclassified examples
    ax_img = fig.add_subplot(gs[1])
    ax_img.axis('off')

    if not top_confused:
        ax_img.text(0.5, 0.5, 'no misclassifications!', ha='center', va='center', fontsize=20)
    else:
        # create grid showing misclassified examples
        n_pairs = len(top_confused)
        n_samples = min(5, n_samples_per_class)  # show up to 5 examples per pair

        is_grayscale = all_images[0].shape[0] == 1
        img_h, img_w = 32, 32

        # create canvas
        canvas_h = n_pairs * img_h
        canvas_w = n_samples * img_w

        if is_grayscale:
            canvas = np.ones((canvas_h, canvas_w)) * 0.5
        else:
            canvas = np.ones((canvas_h, canvas_w, 3)) * 0.5

        for pair_idx, (count, true_class, pred_class, true_name, pred_name) in enumerate(top_confused):
            # get misclassified images for this pair
            img_indices = misclassified[(true_class, pred_class)][:n_samples]

            for img_idx, global_idx in enumerate(img_indices):
                img = all_images[global_idx]

                # denormalize
                img = (img * 0.5 + 0.5).clamp(0, 1)

                # convert to numpy
                if img.shape[0] == 1:
                    img_np = img.squeeze(0).numpy()
                else:
                    img_np = img.permute(1, 2, 0).numpy()

                # place in canvas
                y_start = pair_idx * img_h
                x_start = img_idx * img_w

                if is_grayscale:
                    canvas[y_start:y_start+img_h, x_start:x_start+img_w] = img_np
                else:
                    canvas[y_start:y_start+img_h, x_start:x_start+img_w, :] = img_np

        if is_grayscale:
            ax_img.imshow(canvas, cmap='gray')
        else:
            ax_img.imshow(canvas)

        # add labels
        for pair_idx, (count, true_class, pred_class, true_name, pred_name) in enumerate(top_confused):
            y_pos = (pair_idx + 0.5) * img_h
            label = f'{true_name} → {pred_name} ({int(count)})'
            ax_img.text(-10, y_pos, label, fontsize=10, ha='right', va='center', fontweight='bold')

        ax_img.set_title(f'top misclassified examples (true → predicted)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"saved confusion matrix with misclassified samples to {save_path}")
    plt.close()

    return cm, cm_normalized


def generate_confusion_matrix_simple_visualisation(
    predictions,
    labels,
    save_path,
    class_names=None,
):
    """
    generate simple confusion matrix from predictions and labels

    args:
        predictions: array of predicted labels
        labels: array of true labels
        save_path: path to save visualisation
        class_names: list of class names (optional)
    """
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'normalized frequency'},
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
    )

    plt.xlabel('predicted class', fontsize=12, fontweight='bold')
    plt.ylabel('true class', fontsize=12, fontweight='bold')
    plt.title('confusion matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"saved confusion matrix to {save_path}")
    plt.close()

    return cm, cm_normalized

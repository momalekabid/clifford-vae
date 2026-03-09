import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    plt.xlabel("Number of Recursive Bind-Unbind Cycles ($m$)")
    plt.ylabel("Cosine Similarity to Original")
    plt.title("Invertible Self-Binding")
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

            plt.xticks([])
            plt.yticks([])

            class_info = ", ".join(
                [
                    f"Class {label}"
                    for label in selected_labels[: len(all_recon_vectors)]
                ]
            )
            plt.title(f"Reconstructions After $m$ Recursive Bind-Unbind Cycles\nRows: {class_info}")
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
        "k_sims": sims,
        "k_values": list(range(1, k_self_bind + 1)),
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

    ab = bind(a, b)
    recovered_b = unbind(ab, a, method=unbind_method)
    recovered_a = unbind(ab, b, method=unbind_method)

    sim_b = torch.nn.functional.cosine_similarity(recovered_b, b, dim=-1).mean().item()
    sim_a = torch.nn.functional.cosine_similarity(recovered_a, a, dim=-1).mean().item()

    avg_sim = (sim_a + sim_b) / 2.0

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
        plt.title(f"Cross-Class Binding and Unbinding (Average Similarity: {avg_sim:.3f})")
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
            )
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
    plt.xlabel(f"Phase Angle $\\theta_{{{ax0}}}$")
    plt.ylabel(f"Phase Angle $\\theta_{{{ax1}}}$")
    plt.title("Clifford Torus Latent Phase Angles")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _angles_to_clifford_vector(
    angles: torch.Tensor, normalize_ifft: bool = True
) -> torch.Tensor:
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
def plot_clifford_torus_recon_grid(
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
    plt.title("Decoder Reconstructions over Clifford Torus Grid")
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
                np.mean(single_bind_sims),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(single_bind_sims):.3f}",
            )
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Count")
            plt.title("Binding and Unbinding Performance")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(single_bind_sims, "o-", alpha=0.7, markersize=4)
            plt.axhline(
                np.mean(single_bind_sims), color="red", linestyle="--", alpha=0.8
            )
            plt.xlabel("Test Index")
            plt.ylabel("Cosine Similarity")
            plt.title("Per-Test Cosine Similarity")
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
        f"Clifford Torus Manifold Traversal (Dimensions {dims[0]}, {dims[1]})"
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

    grid_size = 12
    n_samples = grid_size * grid_size

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        x_recon = model.decoder(z)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
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
    plt.title("Power Spherical Manifold Reconstructions")
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

    grid_size = 12
    n_samples = grid_size * grid_size

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        x_recon = model.decoder(z)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
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
    plt.title("Gaussian Manifold Random Sample Reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return path


def plot_cross_dist_comparison_dim(dim_results, latent_dim, dataset_name, output_dir):
    """
    3-panel comparison graph for all distributions at a single latent_dim.
    panels: bundle capacity | self-binding | role-filler capacity.
    dim_results: dict {dist_name: {"bundle_cap", "role_filler", "self_binding_k_sims", "self_binding_k_values"}}
    random_hrr entry shown as dashed reference line.
    """
    COLORS = {
        "clifford": "#2196F3",
        "powerspherical": "#FF9800",
        "gaussian": "#4CAF50",
        "gaussian_nol2": "#9C27B0",
        "random_hrr": "#999999",
    }
    LABELS = {
        "clifford": "Clifford",
        "powerspherical": "PowerSpherical",
        "gaussian": "Gaussian (L2)",
        "gaussian_nol2": "Gaussian",
        "random_hrr": "random HRR (ref.)",
    }
    ORDER = ["clifford", "powerspherical", "gaussian", "gaussian_nol2", "random_hrr"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for dist_name in ORDER:
        metrics = dim_results.get(dist_name)
        if metrics is None:
            continue
        ls = "--" if dist_name == "random_hrr" else "-"
        color = COLORS.get(dist_name, "black")
        label = LABELS.get(dist_name, dist_name)

        bc = metrics.get("bundle_cap")
        if bc and bc.get("k") and bc.get("accuracy"):
            axes[0].plot(bc["k"], bc["accuracy"], marker="o", markersize=3,
                         color=color, linestyle=ls, label=label)

        k_sims = metrics.get("self_binding_k_sims", [])
        k_vals = metrics.get("self_binding_k_values", [])
        if k_sims and k_vals:
            axes[1].plot(k_vals, k_sims, marker="o", markersize=3,
                         color=color, linestyle=ls, label=label)

        rf = metrics.get("role_filler")
        if rf and rf.get("k") and rf.get("accuracy"):
            axes[2].plot(rf["k"], rf["accuracy"], marker="s", markersize=3,
                         color=color, linestyle=ls, label=label)

    axes[0].set_xlabel("Number of Bundled Vectors ($k$)")
    axes[0].set_ylabel("Retrieval Accuracy")
    axes[0].set_title(f"Bundle Capacity ($d={latent_dim}$)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Number of Recursive Bind-Unbind Cycles ($m$)")
    axes[1].set_ylabel("Cosine Similarity to Original")
    axes[1].set_title(f"Invertible Self-Binding ($d={latent_dim}$)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-0.1, 1.05)

    axes[2].set_xlabel("Number of Bundled Role-Filler Pairs ($k$)")
    axes[2].set_ylabel("Unbinding Accuracy")
    axes[2].set_title(f"Role-Filler Capacity ($d={latent_dim}$)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1.05)

    fig.suptitle(f"{dataset_name} — VSA comparison (d={latent_dim})", fontsize=13)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"vsa_comparison_d{latent_dim}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_across_dims_comparison(across_dim_results, latent_dims_used, dataset_name, output_dir):
    """
    outputs knn accuracy, f1, and fid results as:
      1. latex table (booktabs) ready to paste into paper
      2. csv for easy parsing
      3. wandb.Table if wandb is active
    returns path to the .tex file.
    """
    LABELS_TEX = {
        "clifford": "$\\mathcal{C}$-VAE",
        "powerspherical": "$\\mathcal{S}$-VAE",
        "gaussian": "$\\mathcal{N}$-VAE (L2)",
        "gaussian_nol2": "$\\mathcal{N}$-VAE",
    }
    LABELS_PLAIN = {
        "clifford": "Clifford",
        "powerspherical": "PowerSpherical",
        "gaussian": "Gaussian (L2)",
        "gaussian_nol2": "Gaussian",
    }

    dist_order = [d for d in ["gaussian_nol2", "gaussian", "powerspherical", "clifford"]
                  if d in across_dim_results and across_dim_results[d].get("dims")]

    if not dist_order:
        return None

    dims = across_dim_results[dist_order[0]]["dims"]
    train_sizes = [100, 600, 1000]
    metrics = ["knn", "f1"]
    metric_keys = {
        "knn": ["knn_100", "knn_600", "knn_1000"],
        "f1": ["f1_100", "f1_600", "f1_1000"],
    }
    os.makedirs(output_dir, exist_ok=True)

    def fmt_pct(v):
        if v <= 1.0:
            return f"{v * 100:.1f}"
        return f"{v:.1f}"

    # --- collect raw data ---
    # rows: list of (dist_name, metric_name, n_train, [values per dim])
    rows = []
    for dist_name in dist_order:
        data = across_dim_results[dist_name]
        for m in metrics:
            for n_train, key in zip(train_sizes, metric_keys[m]):
                vals = data.get(key, [])
                # pad if needed
                vals = vals + [float("nan")] * (len(dims) - len(vals))
                rows.append((dist_name, m, n_train, vals[:len(dims)]))

    # find best per (metric, n_train, dim_idx) — highest is best
    from collections import defaultdict
    best_vals = defaultdict(lambda: (float("-inf"), None))
    for dist_name, m, n_train, vals in rows:
        for di, v in enumerate(vals):
            if np.isnan(v):
                continue
            col_key = (m, n_train, di)
            if v > best_vals[col_key][0]:
                best_vals[col_key] = (v, dist_name)
    best_dist = {k: dist for k, (_, dist) in best_vals.items()}

    # --- 1. latex table ---
    # format: like paper table — grouped columns by train size
    # columns per group: one per distribution
    n_dists = len(dist_order)
    dist_syms = [LABELS_TEX[d] for d in dist_order]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Semi-supervised $k$-NN results on {dataset_name.replace('_', ' ').title()} (CNN, across latent dimensions).}}")
    lines.append(f"\\label{{tab:{dataset_name}_cnn_knn}}")

    # column spec: l | (n_dists cols) per train size
    col_spec = "l" + ("|" + "c" * n_dists) * len(train_sizes)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # header row 1: train sizes
    header1 = " "
    for n_train in train_sizes:
        header1 += f" & \\multicolumn{{{n_dists}}}{{c|}}{{{n_train}}}"
    header1 = header1.rstrip("|") + " \\\\"
    lines.append(header1)

    # header row 2: distribution names
    header2 = "Method"
    for _ in train_sizes:
        for sym in dist_syms:
            header2 += f" & {sym}"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    # one row per dim, one section per metric
    for m, m_label in [("knn", "Accuracy"), ("f1", "Macro F1")]:
        lines.append(f"\\multicolumn{{{1 + n_dists * len(train_sizes)}}}{{l}}{{\\textit{{{m_label}}}}} \\\\")
        for di, d in enumerate(dims):
            row_str = f"$d = {d}$"
            for n_train in train_sizes:
                for dist_name in dist_order:
                    # find this row's value
                    val = float("nan")
                    for dn, rm, rn, vals in rows:
                        if dn == dist_name and rm == m and rn == n_train:
                            val = vals[di]
                            break
                    if np.isnan(val):
                        row_str += " & —"
                    else:
                        s = fmt_pct(val)
                        if best_dist.get((m, n_train, di)) == dist_name:
                            row_str += f" & \\textbf{{{s}}}"
                        else:
                            row_str += f" & {s}"
            row_str += " \\\\"
            lines.append(row_str)
        lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_str = "\n".join(lines)
    tex_path = os.path.join(output_dir, f"{dataset_name}_results.tex")
    with open(tex_path, "w") as f:
        f.write(tex_str)
    print(f"latex table saved to {tex_path}")

    # --- 2. csv ---
    csv_lines = ["method,metric,n_train," + ",".join(f"d={d}" for d in dims)]
    for dist_name, m, n_train, vals in rows:
        label = LABELS_PLAIN[dist_name]
        n_str = str(n_train) if n_train else "—"
        val_strs = [f"{v:.4f}" if not np.isnan(v) else "" for v in vals]
        csv_lines.append(f"{label},{m},{n_str}," + ",".join(val_strs))
    csv_path = os.path.join(output_dir, f"{dataset_name}_results.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"csv saved to {csv_path}")

    # --- 3. wandb table ---
    try:
        import wandb
        if wandb.run is not None:
            columns = ["Method", "Metric", "N_train"] + [f"d={d}" for d in dims]
            wb_rows = []
            for dist_name, m, n_train, vals in rows:
                label = LABELS_PLAIN[dist_name]
                wb_rows.append([label, m, n_train or "—"] + [round(v, 4) if not np.isnan(v) else None for v in vals])
            wb_table = wandb.Table(columns=columns, data=wb_rows)
            wandb.log({f"{dataset_name}_results": wb_table})
            print(f"logged wandb table: {dataset_name}_results")
    except Exception as e:
        print(f"wandb table logging skipped: {e}")

    return tex_path


@torch.no_grad()
def plot_latent_dimension_exploration(
    model, loader, device, output_dir, n_dims_to_explore=6, n_steps=9, img_shape=(1, 28, 28)
):
    """
    explore individual latent dimensions by encoding a sample and varying each dimension.
    inspired by stitch fix's style space exploration.

    for clifford: varies angles in [-pi, pi]
    for gaussian: varies in [-3, 3] (3 std devs)
    """
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    dist = getattr(model, "distribution", None)

    if latent_dim is None or latent_dim < 4:
        return None

    model.eval()
    for x, y in loader:
        x_sample = x[0:1].to(device)
        break

    out = model(x_sample)
    if isinstance(out, (tuple, list)):
        _, _, _, mu = out
    else:
        mu = out

    base_latent = mu.detach().clone()

    dims_to_explore = min(n_dims_to_explore, latent_dim)
    if latent_dim > 10:
        dim_indices = [int(i * latent_dim / dims_to_explore) for i in range(dims_to_explore)]
    else:
        dim_indices = list(range(dims_to_explore))

    if dist == "clifford":
        variation_range = torch.linspace(-math.pi, math.pi, n_steps, device=device)
    else:
        variation_range = torch.linspace(-3.0, 3.0, n_steps, device=device)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dist}_style_exploration.png")

    all_reconstructions = []

    for dim_idx in dim_indices:
        row_reconstructions = []
        for val in variation_range:
            modified_latent = base_latent.clone()
            modified_latent[:, dim_idx] = val
            if dist == "clifford":
                z = _angles_to_clifford_vector(modified_latent, normalize_ifft=True)
            else:
                z = modified_latent
            x_recon = model.decoder(z)

            if hasattr(model, "decoder") and hasattr(model.decoder, "output_activation"):
                if model.decoder.output_activation == "sigmoid":
                    x_recon = torch.sigmoid(x_recon)
                else:
                    x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
            else:
                x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)

            x_recon = x_recon.view(-1, *img_shape)
            row_reconstructions.append(x_recon[0])

        all_reconstructions.append(row_reconstructions)

    C = img_shape[0]
    h, w = img_shape[1], img_shape[2]
    n_rows = len(dim_indices)
    n_cols = n_steps
    canvas = torch.zeros(C, n_rows * h, n_cols * w)

    for row_idx, row_recons in enumerate(all_reconstructions):
        for col_idx, img in enumerate(row_recons):
            canvas[:, row_idx * h : (row_idx + 1) * h, col_idx * w : (col_idx + 1) * w] = img.cpu()

    fig_height = max(8, n_rows * 1.5)
    fig_width = max(12, n_cols * 1.5)
    plt.figure(figsize=(fig_width, fig_height))

    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())

    plt.yticks(
        [h * i + h // 2 for i in range(n_rows)],
        [f"Dim {dim_indices[i]}" for i in range(n_rows)]
    )

    if dist == "clifford":
        range_str = "[-π, π]"
    else:
        range_str = "[-3σ, 3σ]"

    plt.xticks(
        [w * i + w // 2 for i in range(n_cols)],
        [f"{variation_range[i]:.2f}" for i in range(n_cols)],
        rotation=45
    )

    plt.title(
        f"{dist.capitalize()} Latent Space Traversal ($d={latent_dim}$)\n"
        f"Each Row Shows Variations Along One Latent Dimension {range_str}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return path



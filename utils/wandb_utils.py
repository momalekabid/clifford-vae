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

from .vsa import bind, unbind, invert, hrr_init, unitary_init, normalize_vectors


def _get_flat_z(model, x):
    """extract decoder-ready z from model, flattened to (B, flat_dim)."""
    if hasattr(model, 'encode') and not hasattr(model, 'get_flat_latent'):
        # mlp vae: encode() expects flat input
        x_in = x if x.dim() == 2 else x.view(x.size(0), -1)
        z, _ = model.encode(x_in)
    elif hasattr(model, 'reparameterize') and hasattr(model, 'encoder'):
        # vit/rescnn models: encoder -> reparameterize gives (B, T, D) or (B, T, 2*D)
        mu, params = model.encoder(x)
        z, _, _ = model.reparameterize(mu, params)
    else:
        out = model(x)
        z = out[-1] if isinstance(out, (tuple, list)) else out

    # flatten per-token to per-sample
    if z.dim() == 3:
        z = z.reshape(z.size(0), -1)

    return z


def _decode(model, z):
    """wrapper around model.decoder() that reshapes flat vectors for per-token models."""
    if z.dim() == 2 and hasattr(model, "num_tokens"):
        dec_dim = 2 * model.latent_dim if model.distribution == "clifford" else model.latent_dim
        z = z.view(z.size(0), model.num_tokens, dec_dim)
    return model.decoder(z)


def test_self_binding(
    model,
    loader,
    device,
    output_dir,
    k_self_bind: int = 40,
    unbind_method: str = "*",
    img_shape=(1, 28, 28),
    n_trials: int = 10,
):
    """schlegel et al. sec 3.2 — sequential bind with random latent partners, then unbind in reverse.
    measures cosine similarity between recovered vector and original at each depth m.
    uses real encoded latents as partners (not synthetic random vectors).
    """
    try:
        model.eval()
        with torch.no_grad():
            all_z = []
            all_labels = []
            for x, y in loader:
                x = x.to(device)
                z = _get_flat_z(model, x)
                all_z.append(z.detach())
                all_labels.append(y)
                if len(torch.cat(all_z, 0)) >= 200:
                    break

            if not all_z:
                return {
                    "binding_k_self_similarity": 0.0,
                    "similarity_after_k_binds_plot_path": None,
                }

            all_z = torch.cat(all_z, 0)
            all_labels = torch.cat(all_labels, 0)

    except Exception:
        return {
            "binding_k_self_similarity": 0.0,
            "similarity_after_k_binds_plot_path": None,
        }

    if getattr(model, "distribution", None) == "gaussian":
        all_z = torch.nn.functional.normalize(all_z, p=2, dim=-1)

    # cap depth to available partners
    max_depth = min(k_self_bind, len(all_z) - 1)

    # --- curve 1: self-binding (bind with self m times, unbind m times) ---
    self_depth_sims = {m: [] for m in range(1, max_depth + 1)}
    for trial in range(n_trials):
        idx = torch.randint(0, len(all_z), (1,)).item()
        target = all_z[idx:idx+1]
        for m in range(1, max_depth + 1):
            bound = target.clone()
            for _ in range(m):
                bound = bind(bound, target)
            recovered = bound.clone()
            for _ in range(m):
                recovered = unbind(recovered, target, method=unbind_method)
            sim = torch.nn.functional.cosine_similarity(recovered, target, dim=-1).mean().item()
            self_depth_sims[m].append(sim)

    # --- curve 2: random-partner binding (schlegel sec 3.2) ---
    rand_depth_sims = {m: [] for m in range(1, max_depth + 1)}
    for trial in range(n_trials):
        idx = torch.randint(0, len(all_z), (1,)).item()
        target = all_z[idx:idx+1]

        other_idx = [i for i in range(len(all_z)) if i != idx]
        perm = torch.randperm(len(other_idx))[:max_depth]
        partners = all_z[[other_idx[p] for p in perm]]

        for m in range(1, max_depth + 1):
            bound = target.clone()
            for i in range(m):
                bound = bind(bound, partners[i:i+1])
            recovered = bound.clone()
            for i in range(m - 1, -1, -1):
                recovered = unbind(recovered, partners[i:i+1], method=unbind_method)
            sim = torch.nn.functional.cosine_similarity(recovered, target, dim=-1).mean().item()
            rand_depth_sims[m].append(sim)

    depths = list(range(1, max_depth + 1))
    self_means = [np.mean(self_depth_sims[m]) for m in depths]
    self_stds = [np.std(self_depth_sims[m]) for m in depths]
    rand_means = [np.mean(rand_depth_sims[m]) for m in depths]
    rand_stds = [np.std(rand_depth_sims[m]) for m in depths]

    # use random-partner similarity as the primary metric (harder test)
    cos_sim = rand_means[-1] if rand_means else 0.0
    # expose both for logging
    mean_sims = rand_means

    path_bind_curve = os.path.join(
        output_dir, f"similarity_after_k_binds_{unbind_method}.png"
    )
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(depths, self_means, "o-", markersize=5, label="Self-Binding", color="tab:blue", linewidth=2)
    ax.fill_between(depths,
                    [m - s for m, s in zip(self_means, self_stds)],
                    [m + s for m, s in zip(self_means, self_stds)],
                    alpha=0.15, color="tab:blue")

    ax.plot(depths, rand_means, "s-", markersize=5, label="Random Latent Partners", color="tab:orange", linewidth=2)
    ax.fill_between(depths,
                    [m - s for m, s in zip(rand_means, rand_stds)],
                    [m + s for m, s in zip(rand_means, rand_stds)],
                    alpha=0.15, color="tab:orange")

    # baselines: use encoder dim (not bivector dim) so clifford compares fairly
    d = getattr(model, "latent_dim", all_z.shape[-1])
    for bname, init_fn, color, marker in [
        ("HRR (Random)", hrr_init, "tab:gray", "^"),
        ("Random Unitary", unitary_init, "tab:green", "v"),
    ]:
        b_depth_sims = {m: [] for m in range(1, max_depth + 1)}
        for trial in range(n_trials):
            bvecs = init_fn(max_depth + 1, d, device="cpu")
            bvecs = normalize_vectors(bvecs)
            target = bvecs[0:1]
            partners = bvecs[1:]
            for m in range(1, max_depth + 1):
                bound = target.clone()
                for i in range(m):
                    bound = bind(bound, partners[i:i+1])
                recovered = bound.clone()
                for i in range(m - 1, -1, -1):
                    recovered = unbind(recovered, partners[i:i+1], method=unbind_method)
                sim = torch.nn.functional.cosine_similarity(recovered, target, dim=-1).mean().item()
                b_depth_sims[m].append(sim)
        b_means = [np.mean(b_depth_sims[m]) for m in depths]
        b_stds = [np.std(b_depth_sims[m]) for m in depths]
        ax.plot(depths, b_means, marker=marker, markersize=5, label=bname, color=color,
                linestyle="--", alpha=0.8)
        ax.fill_between(depths,
                        [m - s for m, s in zip(b_means, b_stds)],
                        [m + s for m, s in zip(b_means, b_stds)],
                        alpha=0.08, color=color)

    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Binding Depth $m$")
    ax.set_ylabel("Cosine Similarity to Original")
    # use model's latent_dim if available (avoids showing 2*d for clifford bivectors)
    display_d = getattr(model, "latent_dim", all_z.shape[-1])
    ax.set_title(f"Approximate Inverse Binding Depth ($d={display_d}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_bind_curve, dpi=500, bbox_inches="tight")
    plt.close()

    # decode reconstructions at selected depths for a few example vectors
    recon_paths = None
    try:
        recon_every = max(1, max_depth // 5)
        recon_depths = [m for m in depths if m % recon_every == 0 or m == max_depth]

        # pick 3 different class examples
        unique_labels = torch.unique(all_labels)[:3]
        example_indices = []
        example_labels = []
        for label in unique_labels:
            mask = all_labels == label
            if mask.sum() > 0:
                idx = torch.where(mask)[0][0].item()
                example_indices.append(idx)
                example_labels.append(label.item())

        if example_indices:
            all_recon_vectors = []
            for ex_idx in example_indices:
                target = all_z[ex_idx:ex_idx+1]
                row_vecs = [target.squeeze(0)]

                other_idx = [i for i in range(len(all_z)) if i != ex_idx]
                perm = torch.randperm(len(other_idx))[:max_depth]
                partners = all_z[[other_idx[p] for p in perm]]

                for m in recon_depths:
                    bound = target.clone()
                    for i in range(m):
                        bound = bind(bound, partners[i:i+1])
                    recovered = bound.clone()
                    for i in range(m - 1, -1, -1):
                        recovered = unbind(recovered, partners[i:i+1], method=unbind_method)
                    row_vecs.append(recovered.squeeze(0))
                all_recon_vectors.append(row_vecs)

            recon_paths = os.path.join(
                output_dir, f"recon_after_k_binds_{unbind_method}.png"
            )

            flat_vecs = []
            for row in all_recon_vectors:
                flat_vecs.extend(row)

            with torch.no_grad():
                imgs = _decode(model, torch.stack(flat_vecs, 0))
                if hasattr(model, "decoder") and hasattr(model.decoder, "output_activation"):
                    if model.decoder.output_activation == "sigmoid":
                        imgs = torch.sigmoid(imgs)
                    else:
                        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                else:
                    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                imgs = imgs.view(-1, *img_shape).cpu()

            C, h, w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
            n_rows = len(all_recon_vectors)
            n_cols = len(all_recon_vectors[0])
            canvas = torch.zeros(C, n_rows * h, n_cols * w)

            img_idx = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if img_idx < len(imgs):
                        canvas[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = imgs[img_idx]
                        img_idx += 1

            fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.5), max(4, n_rows * 2)))
            if C == 1:
                ax.imshow(canvas.squeeze(0).numpy(), cmap="gray")
            else:
                ax.imshow(canvas.permute(1, 2, 0).numpy())

            col_labels = ["original"] + [f"m={m}" for m in recon_depths]
            ax.set_xticks([w * i + w // 2 for i in range(n_cols)])
            ax.set_xticklabels(col_labels, fontsize=8)
            ax.set_yticks([h * i + h // 2 for i in range(n_rows)])
            ax.set_yticklabels([f"class {l}" for l in example_labels], fontsize=9)
            ax.set_title("Decoded Recovery After $m$ Sequential Bind-Unbind Cycles")
            plt.tight_layout()
            plt.savefig(recon_paths, dpi=500, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(e)
        recon_paths = None

    return {
        "binding_k_self_similarity": cos_sim,
        "similarity_after_k_binds_plot_path": path_bind_curve,
        "recon_after_k_binds_plot_path": recon_paths,
        "k_sims": mean_sims,
        "k_values": depths,
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
            mu = z_mean
        elif len(out) == 4:
            # (x_recon, q_z, p_z, mu)
            _, _, _, mu = out
        else:
            mu = out[-1]
    else:
        mu = out
    # flatten per-token to per-sample for vit/rescnn
    if mu.dim() == 3:
        mu = mu.reshape(mu.size(0), -1)
    return mu


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
    plt.savefig(path, dpi=500, bbox_inches="tight")
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
    imgs = _decode(model, Z).detach().cpu()
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
    plt.savefig(path, dpi=500, bbox_inches="tight")
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
            plt.hist(single_bind_sims, bins=20, alpha=0.8, edgecolor="black")
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
            plt.plot(single_bind_sims, "o-", alpha=0.8, markersize=5)
            plt.axhline(
                np.mean(single_bind_sims), color="red", linestyle="--", alpha=0.8
            )
            plt.xlabel("Test Index")
            plt.ylabel("Cosine Similarity")
            plt.title("Per-Test Cosine Similarity")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path_vsa_test, dpi=500, bbox_inches="tight")
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

        x_recon = _decode(model, Z)

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
    plt.savefig(path, dpi=500, bbox_inches="tight")
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
        x_recon = _decode(model, z)
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
    plt.savefig(path, dpi=500, bbox_inches="tight")
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
        x_recon = _decode(model, z)
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
    plt.savefig(path, dpi=500, bbox_inches="tight")
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
        lw = 1 if dist_name == "random_hrr" else 2
        color = COLORS.get(dist_name, "black")
        label = LABELS.get(dist_name, dist_name)

        bc = metrics.get("bundle_cap")
        if bc and bc.get("k") and bc.get("accuracy"):
            axes[0].plot(bc["k"], bc["accuracy"], marker="o", markersize=5,
                         color=color, linestyle=ls, label=label, linewidth=lw)

        k_sims = metrics.get("self_binding_k_sims", [])
        k_vals = metrics.get("self_binding_k_values", [])
        if k_sims and k_vals:
            axes[1].plot(k_vals, k_sims, marker="o", markersize=5,
                         color=color, linestyle=ls, label=label, linewidth=lw)

        rf = metrics.get("role_filler")
        if rf and rf.get("k") and rf.get("accuracy"):
            axes[2].plot(rf["k"], rf["accuracy"], marker="s", markersize=5,
                         color=color, linestyle=ls, label=label, linewidth=lw)

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

    fig.suptitle(f"{dataset_name} — VSA Comparison ($d={latent_dim}$)", fontsize=13)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"vsa_comparison_d{latent_dim}.png")
    plt.savefig(save_path, dpi=500)
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
    # check if mean_cosine data is available
    has_mean_cosine = any(
        len(across_dim_results[d].get("mean_cosine", [])) > 0
        for d in dist_order
    )
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
        # mean_cosine has no train_size breakdown
        if has_mean_cosine:
            vals = data.get("mean_cosine", [])
            vals = vals + [float("nan")] * (len(dims) - len(vals))
            rows.append((dist_name, "mean_cosine", None, vals[:len(dims)]))

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

    # mean cosine accuracy section (no train size breakdown, spans all columns)
    if has_mean_cosine:
        lines.append(f"\\multicolumn{{{1 + n_dists * len(train_sizes)}}}{{l}}{{\\textit{{Mean Cosine Acc.}}}} \\\\")
        for di, d in enumerate(dims):
            row_str = f"$d = {d}$"
            # find best across dists for this dim
            best_mc_val, best_mc_dist = float("-inf"), None
            for dist_name in dist_order:
                for dn, rm, rn, vals in rows:
                    if dn == dist_name and rm == "mean_cosine":
                        v = vals[di]
                        if not np.isnan(v) and v > best_mc_val:
                            best_mc_val, best_mc_dist = v, dist_name
                        break
            for n_train in train_sizes:
                for dist_name in dist_order:
                    val = float("nan")
                    for dn, rm, rn, vals in rows:
                        if dn == dist_name and rm == "mean_cosine":
                            val = vals[di]
                            break
                    if np.isnan(val):
                        row_str += " & —"
                    else:
                        s = fmt_pct(val)
                        if dist_name == best_mc_dist:
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
            x_recon = _decode(model, z)

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
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close()

    return path


def _decode_vectors(model, vectors, img_shape):
    """decode latent vectors to images, handling different model types."""
    with torch.no_grad():
        imgs = _decode(model, vectors)
        if hasattr(model, "decoder") and hasattr(model.decoder, "output_activation"):
            if model.decoder.output_activation == "sigmoid":
                imgs = torch.sigmoid(imgs)
            else:
                imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        else:
            # mlp decoder outputs logits (sigmoid), cnn/spherear use tanh
            if imgs.shape[-1] == 784 or (len(imgs.shape) == 2 and imgs.shape[-1] == 784):
                imgs = torch.sigmoid(imgs)
            else:
                imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        imgs = imgs.view(-1, *img_shape)
    return imgs.cpu()


def test_pairwise_bind_bundle_decode(
    model, loader, device, output_dir,
    class_names=None, img_shape=None,
    n_classes=10,
):
    """
    for each pair of classes, decode bind(a,b) and bundle(a,b).
    grid: rows = class pairs, cols = [orig_a, orig_b, decode(bind), decode(bundle)]
    """
    from itertools import combinations

    model.eval()

    # collect one decoder-ready z and original image per class
    class_z = {}
    class_img = {}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # get decoder-ready z depending on model type
            if hasattr(model, 'get_flat_latent'):
                z = model.get_flat_latent(x)
            elif hasattr(model, 'encode'):
                x_in = x if x.dim() == 2 else x.view(x.size(0), -1)
                out = model(x_in)
                if isinstance(out, (tuple, list)) and len(out) == 4 and isinstance(out[0], tuple):
                    (_, _), _, z, _ = out
                else:
                    z_mean, _ = model.encode(x_in)
                    z = z_mean
            else:
                out = model(x)
                z = out[-1] if isinstance(out, (tuple, list)) else out

            for i in range(len(y)):
                label = y[i].item()
                if label not in class_z and len(class_z) < n_classes:
                    class_z[label] = z[i:i+1]
                    class_img[label] = x[i:i+1]

            if len(class_z) >= n_classes:
                break

    if len(class_z) < 2:
        return {"pairwise_bind_bundle_path": None}

    os.makedirs(output_dir, exist_ok=True)
    labels = sorted(class_z.keys())
    pairs = list(combinations(labels, 2))

    if img_shape is None:
        sample_x = class_img[labels[0]]
        img_shape = tuple(sample_x.shape[1:])

    rows = []
    pair_labels = []

    sims_a = []
    sims_b = []

    for la, lb in pairs:
        za = class_z[la]
        zb = class_z[lb]

        # bind and bundle operate on last dim, works for any shape
        z_bind = bind(za, zb)
        z_bundle = (za + zb) / math.sqrt(2)

        # unbind to recover individual items
        recovered_a = unbind(z_bind, zb)
        recovered_b = unbind(z_bind, za)

        sim_a = torch.nn.functional.cosine_similarity(recovered_a, za, dim=-1).mean().item()
        sim_b = torch.nn.functional.cosine_similarity(recovered_b, zb, dim=-1).mean().item()
        sims_a.append(sim_a)
        sims_b.append(sim_b)

        # decode all 6: orig_a, orig_b, bind, bundle, recovered_a, recovered_b
        all_z = torch.cat([za, zb, z_bind, z_bundle, recovered_a, recovered_b], dim=0)
        imgs = _decode_vectors(model, all_z, img_shape)
        rows.append(imgs)

        na = class_names[la] if class_names and la < len(class_names) else str(la)
        nb = class_names[lb] if class_names and lb < len(class_names) else str(lb)
        pair_labels.append(f"{na}+{nb}")

    # build grid
    C, H, W = img_shape
    n_rows = len(rows)
    n_cols = 6
    canvas = torch.zeros(C, n_rows * H, n_cols * W)

    for r, imgs in enumerate(rows):
        for c in range(min(n_cols, len(imgs))):
            canvas[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = imgs[c]

    fig_h = max(4, n_rows * 0.8)
    fig_w = max(6, n_cols * 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if C == 1:
        ax.imshow(canvas.squeeze(0).numpy(), cmap="gray")
    else:
        ax.imshow(canvas.permute(1, 2, 0).numpy())

    ax.set_yticks([H * i + H // 2 for i in range(n_rows)])
    ax.set_yticklabels(pair_labels, fontsize=max(4, 8 - n_rows // 10))
    ax.set_xticks([W * i + W // 2 for i in range(n_cols)])
    ax.set_xticklabels(["A", "B", "bind(A,B)", "bundle(A,B)", "unbind→A", "unbind→B"], fontsize=8)
    avg_sim = (sum(sims_a) + sum(sims_b)) / (2 * len(sims_a)) if sims_a else 0
    ax.set_title(f"Pairwise Bind, Bundle & Unbind Recovery (Avg Cosine Sim: {avg_sim:.3f})")
    plt.tight_layout()

    path = os.path.join(output_dir, "pairwise_bind_bundle_decode.png")
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close()

    avg_unbind_sim = (sum(sims_a) + sum(sims_b)) / (2 * len(sims_a)) if sims_a else 0
    return {
        "pairwise_bind_bundle_path": path,
        "n_pairs": len(pairs),
        "avg_unbind_similarity": avg_unbind_sim,
    }


def test_cross_class_bind_unbind(
    model,
    loader,
    device,
    output_dir,
    img_shape=(1, 28, 28),
    class_a: int = None,
    class_b: int = None,
):
    """cross-class bind/unbind with decoded reconstructions.
    2x4 grid:
      row 1: A | B | decode(bind(A,B)) | decode(bundle(A,B))
      row 2: recovered A (*) | recovered B (*) | recovered A (†) | recovered B (†)
    if class_a/class_b not specified, uses first two classes found.
    """
    empty = {"cross_class_bind_unbind_similarity": 0.0, "cross_class_bind_unbind_plot_path": None}
    try:
        model.eval()
        with torch.no_grad():
            all_z = []
            all_labels = []
            for x, y in loader:
                x = x.to(device)
                z = _get_flat_z(model, x)
                all_z.append(z.detach())
                all_labels.append(y)
                if len(torch.cat(all_z, 0)) >= 200:
                    break

            if not all_z:
                return empty

            all_z = torch.cat(all_z, 0)
            all_labels = torch.cat(all_labels, 0)

            unique_labels = torch.unique(all_labels)
            if len(unique_labels) < 2:
                return empty

            if class_a is not None and class_b is not None:
                class_a_label = torch.tensor(class_a)
                class_b_label = torch.tensor(class_b)
            else:
                class_a_label = unique_labels[0]
                class_b_label = unique_labels[1]

            a_mask = all_labels == class_a_label
            b_mask = all_labels == class_b_label
            if a_mask.sum() == 0 or b_mask.sum() == 0:
                return empty

            a = all_z[torch.where(a_mask)[0][0:1]]
            b = all_z[torch.where(b_mask)[0][0:1]]

    except Exception:
        return empty

    if getattr(model, "distribution", None) == "gaussian":
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)

    ab = bind(a, b)
    ab_bundle = (a + b) / math.sqrt(2)

    # both unbind methods
    rec_a_star = unbind(ab, b, method="*")
    rec_b_star = unbind(ab, a, method="*")
    rec_a_dag = unbind(ab, b, method="†")
    rec_b_dag = unbind(ab, a, method="†")

    sim_star = (
        torch.nn.functional.cosine_similarity(rec_a_star, a, dim=-1).mean().item()
        + torch.nn.functional.cosine_similarity(rec_b_star, b, dim=-1).mean().item()
    ) / 2.0
    sim_dag = (
        torch.nn.functional.cosine_similarity(rec_a_dag, a, dim=-1).mean().item()
        + torch.nn.functional.cosine_similarity(rec_b_dag, b, dim=-1).mean().item()
    ) / 2.0

    plot_path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "cross_class_bind_unbind.png")

        # row 1: A, B, bind(A,B), bundle(A,B)
        # row 2: rec_a_star, rec_b_star, rec_a_dag, rec_b_dag
        vectors = torch.cat([a, b, ab, ab_bundle, rec_a_star, rec_b_star, rec_a_dag, rec_b_dag], dim=0)
        imgs = _decode_vectors(model, vectors, img_shape)

        C, h, w = img_shape
        n_cols = 4
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))

        row1_labels = [
            f"A (cls {class_a_label.item()})",
            f"B (cls {class_b_label.item()})",
            "decode bind(A,B)",
            "decode bundle(A,B)",
        ]
        row2_labels = [
            f"rec A (* {sim_star:.3f})",
            f"rec B (* {sim_star:.3f})",
            f"rec A (\u2020 {sim_dag:.3f})",
            f"rec B (\u2020 {sim_dag:.3f})",
        ]

        for col in range(n_cols):
            for row, labels in enumerate([row1_labels, row2_labels]):
                idx = row * n_cols + col
                img = imgs[idx]
                ax = axes[row][col]
                if C == 1:
                    ax.imshow(img.squeeze(0), cmap="gray")
                else:
                    ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
                ax.set_title(labels[col], fontsize=9)
                ax.axis("off")

        dist_name = getattr(model, "distribution", "unknown")
        fig.suptitle(f"Cross-Class Bind/Unbind ({dist_name})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=500, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  cross-class plot error: {e}")
        plot_path = None

    return {
        "cross_class_bind_unbind_similarity": (sim_star + sim_dag) / 2.0,
        "cross_class_bind_unbind_similarity_star": sim_star,
        "cross_class_bind_unbind_similarity_dag": sim_dag,
        "cross_class_bind_unbind_plot_path": plot_path,
    }


def sample_prior_z(dist_name, latent_dim, n, device, l2_normalize=False):
    """sample n latent vectors from the prior."""
    if dist_name == "clifford":
        angles = torch.rand(n, latent_dim, device=device) * (2 * math.pi)
        freq_dim = 2 * latent_dim
        theta_s = torch.zeros(n, freq_dim, device=device)
        theta_s[:, 1:latent_dim] = angles[:, 1:]
        theta_s[:, -latent_dim + 1:] = -torch.flip(angles[:, 1:], dims=(-1,))
        return torch.fft.ifft(torch.exp(1j * theta_s), dim=-1).real.float()
    elif dist_name == "powerspherical":
        z = torch.randn(n, latent_dim, device=device)
        return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        z = torch.randn(n, latent_dim, device=device)
        if l2_normalize:
            z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return z


def compute_fid(model, test_loader, device, dist_name, latent_dim,
                in_channels=3, n_samples=2048, batch_size=256):
    """frechet inception distance between prior samples (decoded) and test set.
    requires torchmetrics; returns nan if not installed.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  torchmetrics not available, skipping FID")
        return {"fid": float("nan")}

    model.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # real images
    n_real = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x_01 = (x.to(device) * 0.5 + 0.5).clamp(0, 1)
            if in_channels == 1:
                x_01 = x_01.repeat(1, 3, 1, 1)
            fid_metric.update(x_01, real=True)
            n_real += len(x)
            if n_real >= n_samples:
                break

    # fake images from prior
    l2_norm = getattr(model, "l2_normalize", False)
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            bs = min(batch_size, n_samples - n_done)
            z = sample_prior_z(dist_name, latent_dim, bs, device, l2_normalize=l2_norm)
            imgs_01 = (_decode(model, z) * 0.5 + 0.5).clamp(0, 1)
            if in_channels == 1:
                imgs_01 = imgs_01.repeat(1, 3, 1, 1)
            fid_metric.update(imgs_01, real=False)
            n_done += bs

    score = fid_metric.compute().item()
    fid_metric.reset()
    return {"fid": score}

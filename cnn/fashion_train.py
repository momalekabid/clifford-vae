import argparse
import math
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as tu
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import time
import json

import torch.nn.functional as F
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models import VAE
from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_cross_class_bind_unbind,
    compute_class_means,
    evaluate_mean_vector_cosine,
    plot_clifford_manifold_visualization,
    plot_powerspherical_manifold_visualization,
    plot_gaussian_manifold_visualization,
    plot_latent_dimension_exploration,
    plot_cross_dist_comparison_dim,
    plot_across_dims_comparison,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    test_per_class_bundle_capacity_k_items,
    # test_binding_unbinding_with_self_binding,
    bind as vsa_bind,
    unbind as vsa_unbind,
    normalize_vectors as vsa_normalize,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0, "entropy": 0.0}
    concentration_stats = []
    sigma_0_vals = []
    sigma_1_vals = []
    effective_beta_vals = []

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, q_z, p_z, _ = model(x)
        losses = model.compute_loss(x, x_recon, q_z, p_z, beta)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k in ["total", "recon", "kld"]:
            sums[k] += losses[f"{k}_loss"].item() * x.size(0)
        sums["entropy"] += losses["entropy"].item() * x.size(0)

        if model.use_learnable_beta:
            sigma_0_vals.append(losses["sigma_0"])
            sigma_1_vals.append(losses["sigma_1"])
        effective_beta_vals.append(losses["effective_beta"] if isinstance(losses["effective_beta"], float) else losses["effective_beta"].item())

        if hasattr(model, "distribution") and model.distribution in [
            "powerspherical",
            "clifford",
        ]:
            if hasattr(q_z, "concentration"):
                concentration_stats.append(q_z.concentration.detach())

    n = len(loader.dataset)
    result = {f"train/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["train/entropy"] = sums["entropy"] / n

    if model.use_learnable_beta and sigma_0_vals:
        result["train/sigma_0"] = np.mean(sigma_0_vals)
        result["train/sigma_1"] = np.mean(sigma_1_vals)
    if effective_beta_vals:
        result["train/effective_beta"] = np.mean(effective_beta_vals)

    if concentration_stats and hasattr(model, "distribution"):
        all_concentrations = torch.cat(concentration_stats, dim=0)
        result[f"train/{model.distribution}_concentration_mean"] = (
            all_concentrations.mean().item()
        )
        result[f"train/{model.distribution}_concentration_std"] = (
            all_concentrations.std().item()
        )
        result[f"train/{model.distribution}_concentration_max"] = (
            all_concentrations.max().item()
        )
        result[f"train/{model.distribution}_concentration_min"] = (
            all_concentrations.min().item()
        )

    return result


def test_epoch(model, loader, device):
    model.eval()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0, "entropy": 0.0}
    concentration_stats = []
    sigma_0_vals = []
    sigma_1_vals = []
    effective_beta_vals = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, q_z, p_z, _ = model(x)
            losses = model.compute_loss(x, x_recon, q_z, p_z, beta=1.0)
            for k in ["total", "recon", "kld"]:
                sums[k] += losses[f"{k}_loss"].item() * x.size(0)
            sums["entropy"] += losses["entropy"].item() * x.size(0)

            if model.use_learnable_beta:
                sigma_0_vals.append(losses["sigma_0"])
                sigma_1_vals.append(losses["sigma_1"])
            effective_beta_vals.append(losses["effective_beta"] if isinstance(losses["effective_beta"], float) else losses["effective_beta"].item())

            if hasattr(model, "distribution") and model.distribution in [
                "powerspherical",
                "clifford",
            ]:
                if hasattr(q_z, "concentration"):
                    concentration_stats.append(q_z.concentration.detach())

    n = len(loader.dataset)
    result = {f"test/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["test/entropy"] = sums["entropy"] / n

    if model.use_learnable_beta and sigma_0_vals:
        result["test/sigma_0"] = np.mean(sigma_0_vals)
        result["test/sigma_1"] = np.mean(sigma_1_vals)
    if effective_beta_vals:
        result["test/effective_beta"] = np.mean(effective_beta_vals)

    if concentration_stats and hasattr(model, "distribution"):
        all_concentrations = torch.cat(concentration_stats, dim=0)
        result[f"test/{model.distribution}_concentration_mean"] = (
            all_concentrations.mean().item()
        )
        result[f"test/{model.distribution}_concentration_std"] = (
            all_concentrations.std().item()
        )
        result[f"test/{model.distribution}_concentration_max"] = (
            all_concentrations.max().item()
        )
        result[f"test/{model.distribution}_concentration_min"] = (
            all_concentrations.min().item()
        )

    return result


def save_reconstructions(model, loader, device, path, n_images=10):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n_images].to(device)
    with torch.no_grad():
        x_recon, _, _, _ = model(x)

    grid = torch.cat([x.cpu(), x_recon.cpu()], dim=0)
    grid = (grid * 0.5 + 0.5).clamp(0, 1)
    tu.save_image(grid, path, nrow=n_images)
    return path


def slerp(z1, z2, t):
    """spherical linear interpolation between two vectors"""
    z1_norm = z1 / z1.norm(dim=-1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=-1, keepdim=True)
    dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega.abs().min() < 1e-6:
        return (1 - t) * z1_norm + t * z2_norm
    s1 = torch.sin((1 - t) * omega) / sin_omega
    s2 = torch.sin(t * omega) / sin_omega
    return s1 * z1_norm + s2 * z2_norm


def lerp(z1, z2, t):
    """linear interpolation between two vectors"""
    return (1 - t) * z1 + t * z2


def clifford_manifold_interp(z1, z2, t, latent_dim):
    """
    Converts to angle space, interpolates angles (with wraparound), converts back.
    this keeps the interpolation on the torus rather than cutting through ambient space.
    """
    freq1 = torch.fft.fft(z1, dim=-1)
    freq2 = torch.fft.fft(z2, dim=-1)
    angles1 = torch.angle(freq1[..., :latent_dim])
    angles2 = torch.angle(freq2[..., :latent_dim])
    diff = angles2 - angles1
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    angles_interp = angles1 + t * diff

    n = 2 * latent_dim
    theta_s = torch.zeros((*angles_interp.shape[:-1], n), device=z1.device, dtype=z1.dtype)
    theta_s[..., 1:latent_dim] = angles_interp[..., 1:]
    theta_s[..., -latent_dim+1:] = -torch.flip(angles_interp[..., 1:], dims=(-1,))
    samples_c = torch.exp(1j * theta_s)
    return torch.fft.ifft(samples_c, dim=-1).real


def get_fixed_interp_pairs(loader, n_pairs=5, seed=42):
    """
    Uses a fixed seed so the same images are used across all distribution runs;
    Returns a list of (img1, img2, class1, class2) tuples (raw cpu tensors).
    """
    rng = np.random.RandomState(seed)

    class_images = {}
    for x, y in loader:
        for i in range(len(y)):
            label = y[i].item()
            if label not in class_images:
                class_images[label] = x[i].cpu()
        if len(class_images) >= 10:
            break

    classes = sorted(class_images.keys())
    pairs = []
    used = set()
    for _ in range(n_pairs * 10):
        c1, c2 = rng.choice(classes, 2, replace=False)
        key = (min(c1, c2), max(c1, c2))
        if key not in used:
            used.add(key)
            pairs.append((class_images[c1], class_images[c2], c1, c2))
        if len(pairs) >= n_pairs:
            break

    return pairs


def plot_latent_interpolations(model, fixed_pairs, device, save_dir, n_steps=10):
    """
    Clifford: generates both slerp and manifold interpolations.
    Powerspherical: uses slerp. for gaussian: uses lerp.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")
    latent_dim = getattr(model, "latent_dim", None)

    n_pairs = len(fixed_pairs)

    if dist == "clifford":
        interp_configs = [
            ("slerp", slerp),
            ("manifold", lambda z1, z2, t: clifford_manifold_interp(z1, z2, t, latent_dim)),
        ]
    elif dist == "powerspherical":
        interp_configs = [("slerp", slerp)]
    else:
        interp_configs = [("lerp", lerp)]

    saved_paths = []
    ts = torch.linspace(0, 1, n_steps).to(device)

    for interp_name, interp_fn in interp_configs:
        fig, axes = plt.subplots(n_pairs, n_steps + 2, figsize=(2 * (n_steps + 2), 2 * n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for row, (img1, img2, c1, c2) in enumerate(fixed_pairs):
                x1 = img1.unsqueeze(0).to(device)
                x2 = img2.unsqueeze(0).to(device)
                mu1, params1 = model.encoder(x1)
                mu2, params2 = model.encoder(x2)
                z1, _, _ = model.reparameterize(mu1, params1)
                z2, _, _ = model.reparameterize(mu2, params2)

                img1_show = (img1 * 0.5 + 0.5).clamp(0, 1)
                if img1_show.shape[0] == 1:
                    axes[row, 0].imshow(img1_show.squeeze(0), cmap="gray")
                else:
                    axes[row, 0].imshow(img1_show.permute(1, 2, 0))
                axes[row, 0].set_title(f"Class {c1}")
                axes[row, 0].axis("off")

                for i, t in enumerate(ts):
                    z_interp = interp_fn(z1, z2, t.item())
                    x_recon = model.decoder(z_interp)
                    img = (x_recon[0].cpu() * 0.5 + 0.5).clamp(0, 1)
                    if img.shape[0] == 1:
                        axes[row, i + 1].imshow(img.squeeze(0), cmap="gray")
                    else:
                        axes[row, i + 1].imshow(img.permute(1, 2, 0))
                    axes[row, i + 1].set_title(f"t={t.item():.1f}")
                    axes[row, i + 1].axis("off")

                img2_show = (img2 * 0.5 + 0.5).clamp(0, 1)
                if img2_show.shape[0] == 1:
                    axes[row, -1].imshow(img2_show.squeeze(0), cmap="gray")
                else:
                    axes[row, -1].imshow(img2_show.permute(1, 2, 0))
                axes[row, -1].set_title(f"Class {c2}")
                axes[row, -1].axis("off")

        plt.suptitle(f"latent interpolation ({interp_name})", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"interpolation_{interp_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        saved_paths.append(save_path)

    return saved_paths[0] if saved_paths else None


def plot_latent_distributions(model, loader, device, save_path, n_dims=50, n_samples=2000):
    """
    plot histograms of individual latent dimensions (mu from encoder).
    contrasts the learned distribution shape across dimensions -- useful for
    seeing how gaussian vs clifford vs powerspherical latents differ structurally.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")

    all_mu = []
    n_collected = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            all_mu.append(mu.cpu())
            n_collected += len(x)
            if n_collected >= n_samples:
                break

    mu = torch.cat(all_mu, dim=0)[:n_samples].numpy()
    latent_dim = mu.shape[1]
    n_show = min(n_dims, latent_dim)

    n_cols = 10
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_show):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        vals = mu[:, i]
        ax.hist(vals, bins=40, density=True, alpha=0.8, color="steelblue", edgecolor="none")
        ax.set_title(f"Latent Dimension {i+1}", fontsize=7)
        ax.tick_params(labelsize=5)

    for i in range(n_show, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    mean_std = np.std(mu, axis=0).mean()
    mean_mean = np.mean(np.abs(np.mean(mu, axis=0)))
    fig.suptitle(
        f"{dist} latent space distribution (d={latent_dim}, n={len(mu)})\n"
        f"avg |mean|={mean_mean:.3f}, avg std={mean_std:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_latent_tsne(model, loader, device, save_path, n_samples=2000):
    """
    t-sne visualization of latent space colored by class label.
    runs multiple perplexities (5, 30, 50) since each reveals different
    structure scales — low perplexity shows local clusters, high shows
    global geometry. uses 5000 iterations for proper convergence.
    see: https://distill.pub/2016/misread-tsne/
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")

    all_mu = []
    all_labels = []
    n_collected = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, _, _, mu = model(x)
            all_mu.append(mu.cpu())
            all_labels.append(y)
            n_collected += len(x)
            if n_collected >= n_samples:
                break

    mu = torch.cat(all_mu, dim=0)[:n_samples].numpy()
    labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()
    n_classes = len(np.unique(labels))

    perplexities = [5, 30, 50]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(6 * len(perplexities), 5))

    for i, perp in enumerate(perplexities):
        print(f"  running t-sne (perplexity={perp}) on {len(mu)} samples (d={mu.shape[1]})...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perp,
            max_iter=5000,
            learning_rate="auto",
        )
        z_2d = tsne.fit_transform(mu)

        ax = axes[i]
        scatter = ax.scatter(
            z_2d[:, 0], z_2d[:, 1],
            c=labels, cmap=plt.get_cmap("tab10", n_classes),
            s=8, alpha=0.7,
        )
        ax.set_title(f"Perplexity = {perp}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(scatter, ax=axes[-1], label="class")
    fig.suptitle(f"{dist}: t-sne of latent space (d={mu.shape[1]})", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def filter_dataset_by_class(dataset, exclude_class):
    """filter dataset to exclude a specific class"""
    if exclude_class < 0:
        return dataset

    indices = [i for i, (_, label) in enumerate(dataset) if label != exclude_class]
    return torch.utils.data.Subset(dataset, indices)


def get_excluded_class_subset(dataset, exclude_class):
    """get only the excluded class samples"""
    if exclude_class < 0:
        return None

    indices = [i for i, (_, label) in enumerate(dataset) if label == exclude_class]
    return torch.utils.data.Subset(dataset, indices) if indices else None



def sample_prior_z(dist_name, latent_dim, n, device, l2_normalize=False):
    """sample n latent vectors from the prior.
    clifford: uniform on (S^1)^{d-1} via d-1 free angles (DC+Nyquist fixed at 0).
    powerspherical: uniform on S^{d-1} (κ→0 limit).
    gaussian: isotropic N(0,I), or unit sphere if l2_normalize.
    """
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


def compute_generation_fid(model, dist_name, latent_dim, test_loader, device,
                            in_channels, n_samples=2048, batch_size=256):
    """self-FID: frechet inception distance between prior samples (decoded) and test set.
    lower = generated images closer to the real data distribution.
    requires torchmetrics; silently returns nan if not installed.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  torchmetrics not available, skipping generation FID")
        return float("nan")

    model.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

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

    l2_norm = getattr(model, "l2_normalize", False)
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            bs = min(batch_size, n_samples - n_done)
            z = sample_prior_z(dist_name, latent_dim, bs, device, l2_normalize=l2_norm)
            imgs_01 = (model.decoder(z) * 0.5 + 0.5).clamp(0, 1)
            if in_channels == 1:
                imgs_01 = imgs_01.repeat(1, 3, 1, 1)
            fid_metric.update(imgs_01, real=False)
            n_done += bs

    score = fid_metric.compute().item()
    fid_metric.reset()
    return score


def perform_knn_evaluation(
    model, train_loader, test_loader, device, n_samples_list=[100, 600, 1000]
):
    """k-nn classification on latent embeddings with multiple training sample sizes."""
    print("knn eval in progress")
    model.eval()

    def encode_dataset(loader):
        latents, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                _, _, _, mu = model(x.to(device))
                latents.append(mu.cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(latents), np.concatenate(labels)

    X_train_full, y_train_full = encode_dataset(train_loader)
    X_test, y_test = encode_dataset(test_loader)

    metric = (
        "cosine"
        if getattr(model, "distribution", None) in ["powerspherical", "clifford"]
        else "euclidean"
    )

    results = {}
    for n_samples in n_samples_list:
        if n_samples > len(X_train_full):
            print(
                f"Warning: k-NN sample size {n_samples} > training data size {len(X_train_full)}. Skipping."
            )
            continue

        indices = np.random.choice(len(X_train_full), n_samples, replace=False)
        X_train_sample, y_train_sample = X_train_full[indices], y_train_full[indices]

        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train_sample, y_train_sample)

        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results[f"knn_acc_{n_samples}"] = float(accuracy)
        results[f"knn_f1_{n_samples}"] = float(f1)
        print(f"knn accuracy with {n_samples} training samples: {accuracy:.4f}")

    return results


def main(args):
    script_start_time = time.time()
    timing_results = {}

    print(f"Device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = args.latent_dims if args.latent_dims else [2048, 4096]
    distributions = ["clifford", "powerspherical", "gaussian", "gaussian_nol2"]
    datasets_to_test = ["fashionmnist", "cifar10"]

    # per-distribution lr overrides
    dist_lr = {
        "clifford": args.lr,
        "powerspherical": 1e-4,
        "gaussian": args.lr,
        "gaussian_nol2": args.lr,
    }

    dataset_map = {
        "fashionmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }

    class_names_map = {
        "fashionmnist": ["tshirt", "trouser", "pullover", "dress", "coat",
                         "sandal", "shirt", "sneaker", "bag", "boot"],
        "cifar10": ["plane", "auto", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    }

    for dataset_name in datasets_to_test:
        is_color = dataset_name == "cifar10"
        in_channels = 3 if is_color else 1
        IMG_SHAPE = (3, 32, 32) if is_color else (1, 32, 32)
        dataset_class = dataset_map[dataset_name]
        class_names = class_names_map[dataset_name]
        norm_mean, norm_std = (
            ((0.5,), (0.5,)) if not is_color else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        train_set_full = dataset_class(
            "data", train=True, download=True, transform=transform
        )
        test_set_full = dataset_class(
            "data", train=False, download=True, transform=transform
        )

        train_set = filter_dataset_by_class(train_set_full, args.exclude_class)
        test_set = filter_dataset_by_class(test_set_full, args.exclude_class)
        excluded_test_set = get_excluded_class_subset(test_set_full, args.exclude_class)

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        excluded_test_loader = None
        if excluded_test_set is not None:
            excluded_test_loader = DataLoader(
                excluded_test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
            )
            print(
                f"excluding class {args.exclude_class} from training. excluded test set size: {len(excluded_test_set)}"
            )

        # extract once per dataset so all distributions interpolate the same images
        fixed_interp_pairs = get_fixed_interp_pairs(test_loader, n_pairs=5, seed=42)

        BC_K_RANGE = list(range(5, 51, 5))   # bundle capacity
        RF_K_RANGE = list(range(2, 21, 2))    # role-filler

        across_dim_results = {d: {"knn_100": [], "knn_600": [], "knn_1000": [], "f1_100": [], "f1_600": [], "f1_1000": [], "dims": []} for d in distributions}

        for latent_dim in latent_dims:
            dim_results = {}  # dist_name -> metrics dict

            for dist_name in distributions:
                for trial in range(args.n_trials):
                    trial_suffix = f"-trial{trial+1}" if args.n_trials > 1 else ""
                    exp_name = f"{dataset_name}-{dist_name}-d{latent_dim}-{args.recon_loss}{trial_suffix}"
                    output_dir = f"results/{exp_name}"
                    os.makedirs(output_dir, exist_ok=True)

                    print(f"\n== {exp_name} ==")
                    exp_start_time = time.time()
                    logger.start_run(exp_name, args)

                    # gaussian_nol2 is gaussian without l2 normalization
                    if dist_name == "gaussian_nol2":
                        actual_dist = "gaussian"
                        l2_norm = False
                    elif dist_name == "gaussian":
                        actual_dist = "gaussian"
                        l2_norm = args.l2_norm
                    else:
                        actual_dist = dist_name
                        l2_norm = False
                    model = VAE(
                        latent_dim=latent_dim,
                        in_channels=in_channels,
                        distribution=actual_dist,
                        device=DEVICE,
                        recon_loss_type=args.recon_loss,
                        l1_weight=args.l1_weight,
                        freq_weight=0.0,
                        l2_normalize=l2_norm,
                        use_learnable_beta=args.use_learnable_beta,
                    )
                    logger.watch_model(model)
                    cur_lr = dist_lr.get(dist_name, args.lr)
                    if args.use_learnable_beta:
                        sigma_ids = {id(model.log_sigma_0), id(model.log_sigma_1)}
                        optimizer = optim.AdamW([
                            {"params": [p for p in model.parameters() if id(p) not in sigma_ids], "lr": cur_lr},
                            {"params": [model.log_sigma_0, model.log_sigma_1], "lr": cur_lr * 0.1},
                        ])
                    else:
                        optimizer = optim.AdamW(model.parameters(), lr=cur_lr)
                    best = float("inf")
                    patience_counter = 0
                    train_start_time = time.time()

                    def kl_beta_for_epoch(e: int) -> float:
                        # initial warmup
                        if e < args.warmup_epochs:
                            return (
                                min(1.0, (e + 1) / max(1, args.warmup_epochs))
                                * args.max_beta
                            )
                        if args.cycle_epochs <= 0:
                            return args.max_beta
                        # cyclical annealing option, tri-schedule in [min_beta, max_beta]
                        cycle_pos = (e - args.warmup_epochs) % args.cycle_epochs
                        half = max(1, args.cycle_epochs // 2)
                        if cycle_pos <= half:
                            t = cycle_pos / half
                        else:
                            t = (args.cycle_epochs - cycle_pos) / max(
                                1, args.cycle_epochs - half
                            )
                        return args.min_beta + (args.max_beta - args.min_beta) * t

                    for epoch in range(args.epochs):
                        if args.use_learnable_beta:
                            beta = 1.0
                        else:
                            beta = kl_beta_for_epoch(epoch)

                        train_losses = train_epoch(
                            model, train_loader, optimizer, DEVICE, beta
                        )
                        test_losses = test_epoch(model, test_loader, DEVICE)
                        val = test_losses["test/recon_loss"] + test_losses["test/kld_loss"]
                        if np.isfinite(val) and val < best:
                            best = val
                            torch.save(
                                model.state_dict(), f"{output_dir}/best_model.pt"
                            )
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        metrics_to_log = {
                            "epoch": epoch,
                            **train_losses,
                            **test_losses,
                            "best_test_total_loss": best,
                        }
                        if not args.use_learnable_beta:
                            metrics_to_log["beta"] = beta

                        logger.log_metrics(metrics_to_log)

                        if args.patience > 0 and patience_counter >= args.patience:
                            print(
                                f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)"
                            )
                            break

                    train_time = time.time() - train_start_time
                    print(
                        f"best total loss (recon+kld): {best:.4f}, training time: {train_time:.2f}s"
                    )

                    if os.path.exists(f"{output_dir}/best_model.pt"):
                        ckpt = torch.load(f"{output_dir}/best_model.pt", map_location=DEVICE)
                        if not args.use_learnable_beta:
                            ckpt = {k: v for k, v in ckpt.items()
                                    if k not in ("log_sigma_0", "log_sigma_1")}
                        model.load_state_dict(ckpt)

                        eval_start_time = time.time()
                        normalize_vectors = True
                        images = {}

                        latents = []
                        labels_list = []
                        images_list = []
                        for x, y in test_loader:
                            x = x.to(DEVICE)
                            _, _, _, mu = model(x)
                            latents.append(mu.detach())
                            labels_list.append(y)
                            images_list.append(x.cpu())
                            if len(torch.cat(latents, 0)) >= 1000:
                                break
                        item_memory = torch.cat(latents, 0)[:1000]
                        item_labels = torch.cat(labels_list, 0)[:1000].to(DEVICE)
                        item_images = torch.cat(images_list, 0)[:1000]

                        # === test 1: 1-item-per-class similarity matrix (no braiding) ===
                        t0 = time.time()
                        print(
                            f"running 1-item-per-class test ({dist_name}, no braiding)..."
                        )
                        two_per_class_res = test_per_class_bundle_capacity_k_items(
                            d=item_memory.shape[-1],
                            n_items=1000,
                            n_classes=10,
                            items_per_class=1,
                            n_trials=1,
                            normalize=normalize_vectors,
                            device=DEVICE,
                            plot=True,
                            save_dir=output_dir,
                            item_memory=item_memory,
                            labels=item_labels,
                            item_images=item_images,
                            use_braiding=False,
                            class_names=class_names,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        # === test 2: bundle capacity (schlegel et al. sec 3.1) ===
                        t0 = time.time()
                        print(f"running bundle capacity test ({dist_name})...")
                        bundle_cap_raw = vsa_bundle_capacity(
                            d=item_memory.shape[-1],
                            n_items=1000,
                            k_range=BC_K_RANGE,
                            n_trials=20,
                            normalize=normalize_vectors,
                            device=DEVICE,
                            plot=True,
                            save_dir=output_dir,
                            item_memory=item_memory,
                            use_braiding=False,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        # === test 3: role-filler unbinding (schlegel et al. sec 3.3) ===
                        t0 = time.time()
                        print(f"running role-filler unbinding test ({dist_name})...")
                        # role-filler variants: (random_keys, braiding) combos
                        rf_variants = [
                            (True, False, "role_filler_capacity"),
                            (False, False, "role_filler_no_random_keys"),
                            (True, True, "role_filler_braided"),
                            (False, True, "role_filler_no_random_keys_braided"),
                        ]
                        rf_results = {}
                        for bind_rand, braid, rf_name in rf_variants:
                            label = f"bind_with_random={bind_rand}, braiding={braid}"
                            print(f"  running role-filler ({label})...")
                            rf_res = vsa_binding_unbinding(
                                d=item_memory.shape[-1],
                                n_items=1000,
                                k_range=RF_K_RANGE,
                                n_trials=20,
                                normalize=normalize_vectors,
                                device=DEVICE,
                                plot=True,
                                unbind_method="*",
                                save_dir=output_dir,
                                item_memory=item_memory,
                                bind_with_random=bind_rand,
                                use_braiding=braid,
                            )
                            rf_results[rf_name] = rf_res
                            # rename saved plot to variant name
                            default_plot = os.path.join(output_dir, "role_filler_capacity.png")
                            variant_plot = os.path.join(output_dir, f"{rf_name}.png")
                            if os.path.exists(default_plot) and rf_name != "role_filler_capacity":
                                os.rename(default_plot, variant_plot)

                        role_filler_raw = rf_results.get("role_filler_capacity", {})
                        print(f"  all role-filler variants completed in {time.time() - t0:.2f}s")

                        t0 = time.time()
                        print(f"running self-binding test ({dist_name})...")
                        fourier_star = test_self_binding(
                            model,
                            test_loader,
                            DEVICE,
                            output_dir,
                            unbind_method="*",
                            img_shape=IMG_SHAPE,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")
                        fourier_perp = {}

                        t0 = time.time()
                        print(f"running cross-class bind/unbind test ({dist_name})...")
                        cross_class_star = test_cross_class_bind_unbind(
                            model,
                            test_loader,
                            DEVICE,
                            output_dir,
                            unbind_method="*",  # O(d), only shifting
                            img_shape=IMG_SHAPE,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        t0 = time.time()
                        print(f"generating reconstructions...")
                        recon_path = save_reconstructions(
                            model,
                            test_loader,
                            DEVICE,
                            f"{output_dir}/reconstructions.png",
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        t0 = time.time()
                        print(f"generating latent distribution visualization...")
                        latent_dist_path = plot_latent_distributions(
                            model,
                            test_loader,
                            DEVICE,
                            f"{output_dir}/latent_distributions.png",
                            n_dims=50,
                            n_samples=2000,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        # t-sne visualization of latent space
                        t0 = time.time()
                        print(f"generating t-sne visualization...")
                        tsne_path = plot_latent_tsne(
                            model,
                            test_loader,
                            DEVICE,
                            f"{output_dir}/tsne.png",
                            n_samples=2000,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        t0 = time.time()
                        print(f"generating latent interpolations...")
                        interp_path = plot_latent_interpolations(
                            model,
                            fixed_interp_pairs,
                            DEVICE,
                            output_dir,
                            n_steps=10,
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        # knn eval
                        t0 = time.time()
                        print(f"running knn evaluation...")
                        knn_metrics = perform_knn_evaluation(
                            model, train_loader, test_loader, DEVICE, [100, 600, 1000]
                        )
                        print(f"  completed in {time.time() - t0:.2f}s")

                        train_subset = torch.utils.data.Subset(
                            train_set, list(range(min(5000, len(train_set))))
                        )
                        train_subset_loader = DataLoader(
                            train_subset, batch_size=args.batch_size, shuffle=False
                        )
                        class_means = compute_class_means(
                            model, train_subset_loader, DEVICE, max_per_class=1000
                        )
                        mean_vector_acc, _ = evaluate_mean_vector_cosine(
                            model, test_loader, DEVICE, class_means
                        )
                        mean_metric_key = "mean_vector_cosine_acc"
                        print(f"{mean_metric_key}: ", mean_vector_acc)

                        t0 = time.time()
                        print(f"computing generation fid...")
                        gen_fid = compute_generation_fid(
                            model, dist_name, latent_dim, test_loader, DEVICE,
                            in_channels, n_samples=2048,
                        )
                        print(f"  generation_fid={gen_fid:.2f}  ({time.time() - t0:.2f}s)")

                        fourier_metrics = {}
                        fourier_metrics.update(
                            {
                                f"*/{k}": v
                                for k, v in fourier_star.items()
                                if isinstance(v, (int, float, bool))
                            }
                        )
                        fourier_metrics.update(
                            {
                                f"†/{k}": v
                                for k, v in fourier_perp.items()
                                if isinstance(v, (int, float, bool))
                            }
                        )

                        logger.log_metrics(
                            {
                                **knn_metrics,
                                **fourier_metrics,
                                mean_metric_key: float(mean_vector_acc),
                                "final_best_total_loss": best,
                                "generation_fid": gen_fid,
                                "cross_class_bind_unbind_similarity_star": cross_class_star.get(
                                    "cross_class_bind_unbind_similarity", 0.0
                                ),
                            }
                        )

                        images.update(
                            {
                                "reconstructions": recon_path,
                                "latent_distributions": latent_dist_path,
                                "tsne": tsne_path,
                                "latent_interpolation": interp_path,
                            }
                        )

                        two_per_class_plot = os.path.join(
                            output_dir, "bundle_similarity_matrix.png"
                        )
                        if os.path.exists(two_per_class_plot):
                            images["bundle_similarity_matrix"] = two_per_class_plot
                        bc_plot = os.path.join(output_dir, "bundle_capacity.png")
                        if os.path.exists(bc_plot):
                            images["bundle_capacity"] = bc_plot
                        for rf_name in ["role_filler_capacity", "role_filler_no_random_keys",
                                        "role_filler_braided", "role_filler_no_random_keys_braided"]:
                            rf_plot = os.path.join(output_dir, f"{rf_name}.png")
                            if os.path.exists(rf_plot):
                                images[rf_name] = rf_plot

                        sp = fourier_star.get("similarity_after_k_binds_plot_path")
                        sd = fourier_perp.get("similarity_after_k_binds_plot_path")
                        if sp:
                            images["similarity_after_k_binds_*"] = sp
                        if sd:
                            images["similarity_after_k_binds_†"] = sd

                        rp = fourier_star.get("recon_after_k_binds_plot_path")
                        rd = fourier_perp.get("recon_after_k_binds_plot_path")
                        if rp:
                            images["recon_after_k_binds_*"] = rp
                        if rd:
                            images["recon_after_k_binds_†"] = rd

                        if cross_class_star.get("cross_class_bind_unbind_plot_path"):
                            images["cross_class_binding_star"] = cross_class_star[
                                "cross_class_bind_unbind_plot_path"
                            ]

                        if dist_name == "clifford" and 2 <= model.latent_dim <= 8:
                            cliff_viz = plot_clifford_manifold_visualization(
                                model,
                                DEVICE,
                                output_dir,
                                n_grid=16,
                                dims=(0, 1),
                                img_shape=IMG_SHAPE,
                            )
                            if cliff_viz:
                                images["clifford_manifold_visualization"] = cliff_viz

                        elif (
                            dist_name == "powerspherical" and 2 <= model.latent_dim <= 8
                        ):
                            pow_viz = plot_powerspherical_manifold_visualization(
                                model,
                                DEVICE,
                                output_dir,
                                n_samples=1000,
                                dims=(0, 1),
                                img_shape=IMG_SHAPE,
                            )
                            if pow_viz:
                                images["powerspherical_manifold_visualization"] = (
                                    pow_viz
                                )

                        elif dist_name == "gaussian" and 2 <= model.latent_dim <= 8:
                            gauss_viz = plot_gaussian_manifold_visualization(
                                model,
                                DEVICE,
                                output_dir,
                                n_samples=1000,
                                dims=(0, 1),
                                img_shape=IMG_SHAPE,
                            )
                            if gauss_viz:
                                images["gaussian_manifold_visualization"] = gauss_viz

                        excluded_metrics = {}
                        if excluded_test_loader is not None:
                            print(
                                f"\nevaluating on excluded class {args.exclude_class}..."
                            )
                            excluded_losses = test_epoch(
                                model, excluded_test_loader, DEVICE
                            )
                            excluded_metrics = {
                                f"excluded_class_{args.exclude_class}/test_total_loss": excluded_losses[
                                    "test/total_loss"
                                ],
                                f"excluded_class_{args.exclude_class}/test_recon_loss": excluded_losses[
                                    "test/recon_loss"
                                ],
                                f"excluded_class_{args.exclude_class}/test_kld_loss": excluded_losses[
                                    "test/kld_loss"
                                ],
                            }

                            excluded_recon_path = save_reconstructions(
                                model,
                                excluded_test_loader,
                                DEVICE,
                                f"{output_dir}/reconstructions_excluded_class_{args.exclude_class}.png",
                            )
                            images[
                                f"excluded_class_{args.exclude_class}_reconstructions"
                            ] = excluded_recon_path

                        summary = {
                            "final_best_total_loss": best,
                            **fourier_metrics,
                            **knn_metrics,
                            **excluded_metrics,
                            mean_metric_key: float(mean_vector_acc),
                            "generation_fid": gen_fid,
                        }
                        logger.log_summary(summary)
                        logger.log_images(images)

                        metrics_save_path = f"{output_dir}/metrics.json"
                        with open(metrics_save_path, "w") as f:
                            json.dump(summary, f, indent=2)
                        print(f"saved metrics to {metrics_save_path}")

                        eval_time = time.time() - eval_start_time
                        exp_time = time.time() - exp_start_time

                        timing_key = f"{exp_name}"
                        timing_results[timing_key] = {
                            "train_time_s": train_time,
                            "eval_time_s": eval_time,
                            "total_exp_time_s": exp_time,
                        }
                        print(
                            f"eval time: {eval_time:.2f}s, total exp time: {exp_time:.2f}s"
                        )

                        dim_results[dist_name] = {
                            "bundle_cap": bundle_cap_raw,
                            "role_filler": role_filler_raw,
                            "role_filler_variants": rf_results,
                            "self_binding_k_sims": fourier_star.get("k_sims", []),
                            "self_binding_k_values": fourier_star.get("k_values", []),
                            "knn_acc": knn_metrics.get("knn_acc_1000", 0.0),
                            "gen_fid": gen_fid,
                        }
                        across_dim_results[dist_name]["dims"].append(latent_dim)
                        across_dim_results[dist_name]["knn_100"].append(
                            knn_metrics.get("knn_acc_100", 0.0)
                        )
                        across_dim_results[dist_name]["knn_600"].append(
                            knn_metrics.get("knn_acc_600", 0.0)
                        )
                        across_dim_results[dist_name]["knn_1000"].append(
                            knn_metrics.get("knn_acc_1000", 0.0)
                        )
                        across_dim_results[dist_name]["f1_100"].append(
                            knn_metrics.get("knn_f1_100", 0.0)
                        )
                        across_dim_results[dist_name]["f1_600"].append(
                            knn_metrics.get("knn_f1_600", 0.0)
                        )
                        across_dim_results[dist_name]["f1_1000"].append(
                            knn_metrics.get("knn_f1_1000", 0.0)
                        )

                    logger.finish_run()

            try:
                ref_items = torch.randn(1000, latent_dim, device=DEVICE)
                ref_items = F.normalize(ref_items, p=2, dim=-1)
                ref_bc = vsa_bundle_capacity(
                    d=latent_dim, n_items=1000, k_range=BC_K_RANGE,
                    n_trials=20, normalize=True, device=DEVICE,
                    item_memory=ref_items,
                )
                ref_rf = vsa_binding_unbinding(
                    d=latent_dim, n_items=1000, k_range=RF_K_RANGE,
                    n_trials=20, normalize=True, device=DEVICE,
                    unbind_method="*", item_memory=ref_items, bind_with_random=False,
                )
                z_ref = F.normalize(torch.randn(1, latent_dim, device=DEVICE), p=2, dim=-1)
                k_max = 50
                ref_sims = []
                for m in range(1, k_max + 1):
                    cur = z_ref.clone()
                    for _ in range(m):
                        cur = vsa_bind(cur, z_ref)
                    for _ in range(m):
                        cur = vsa_unbind(cur, z_ref, method="*")
                    ref_sims.append(F.cosine_similarity(cur, z_ref, dim=-1).mean().item())
                dim_results["random_hrr"] = {
                    "bundle_cap": ref_bc,
                    "role_filler": ref_rf,
                    "self_binding_k_sims": ref_sims,
                    "self_binding_k_values": list(range(1, k_max + 1)),
                }
                comp_dir = f"results/comparisons/{dataset_name}"
                comp_path = plot_cross_dist_comparison_dim(
                    dim_results, latent_dim, dataset_name, comp_dir
                )
                print(f"saved cross-dist comparison to {comp_path}")
            except Exception as e:
                print(f"warning: cross-dist comparison failed for d={latent_dim}: {e}")

        try:
            comp_dir = f"results/comparisons/{dataset_name}"
            across_path = plot_across_dims_comparison(
                across_dim_results, latent_dims, dataset_name, comp_dir
            )
            print(f"saved across-dims comparison to {across_path}")
        except Exception as e:
            print(f"warning: across-dims comparison failed for {dataset_name}: {e}")

    script_total_time = time.time() - script_start_time
    timing_results["total_script_time_s"] = script_total_time
    with open("fashion_train_timing.json", "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\ntotal script execution time: {script_total_time:.2f}s")
    print(f"timing results saved to fashion_train_timing.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="clifford vae experiments on fashionmnist/cifar10"
    )
    p.add_argument("--epochs", type=int, default=500, help="training epochs")
    p.add_argument("--warmup_epochs", type=int, default=100, help="kl warmup epochs (ignored if --use_learnable_beta)")
    p.add_argument("--batch_size", type=int, default=256, help="batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument(
        "--no-l2_norm",
        dest="l2_norm",
        action="store_false",
        help="disable L2 normalization for Gaussian VAE (default: enabled for Gaussian only)",
    )
    p.set_defaults(l2_norm=True)
    p.add_argument(
        "--recon_loss",
        type=str,
        default="l1",
        choices=["mse", "l1"],
        help="reconstruction loss type",
    )
    p.add_argument("--l1_weight", type=float, default=1.0, help="l1 pixel loss weight")
    p.add_argument("--max_beta", type=float, default=1.0, help="max kl beta (ignored if --use_learnable_beta)")
    p.add_argument(
        "--min_beta", type=float, default=0.1, help="min kl beta during cycles (ignored if --use_learnable_beta)"
    )
    p.add_argument(
        "--use_learnable_beta",
        action="store_true",
        help="use learnable beta (L-VAE) instead of fixed/scheduled beta - eliminates need for warmup and beta scheduling",
    )
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    p.add_argument(
        "--wandb_project",
        type=str,
        default="clifford-experiments-CNN",
        help="wandb project name",
    )
    p.add_argument("--patience", type=int, default=75, help="early stopping patience")
    p.add_argument(
        "--cycle_epochs",
        type=int,
        default=100,
        help="cycle length for cyclical kl beta (0=off)",
    )
    p.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="trials per config for statistical averaging",
    )
    p.add_argument(
        "--exclude_class",
        type=int,
        default=-1,
        help="exclude class (-1=none) | fmnist: 0:tshirt 1:trouser 2:pullover 3:dress 4:coat 5:sandal 6:shirt 7:sneaker 8:bag 9:boot | cifar: 0:plane 1:auto 2:bird 3:cat 4:deer 5:dog 6:frog 7:horse 8:ship 9:truck",
    )
    p.add_argument(
        "--latent_dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="latent dims to test",
    )
    p.add_argument(
        "--braid",
        action="store_true",
        help="run braiding tests",
    )
    args = p.parse_args()
    main(args)

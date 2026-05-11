import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as tu
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import time
import json
import math

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models import VAE as CNNVAE
from utils.wandb_utils import (
    WandbLogger,
    plot_cross_dist_comparison_dim,
    plot_across_dims_comparison,
    test_self_binding,
    test_cross_class_bind_unbind,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    test_per_class_bundle_capacity_k_items,
    unitary_init as vsa_unitary_init,
    bind as vsa_bind,
    unbind as vsa_unbind,
    normalize_vectors as vsa_normalize,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

CLASS_NAMES = [
    "plane",
    "auto",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0, "entropy": 0.0}
    concentration_stats = []
    effective_beta_vals = []
    sigma_0_vals = []
    sigma_1_vals = []

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
        eb = losses["effective_beta"]
        effective_beta_vals.append(eb if isinstance(eb, float) else eb.item())

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
    effective_beta_vals = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, q_z, p_z, _ = model(x)
            losses = model.compute_loss(x, x_recon, q_z, p_z, beta=1.0)
            for k in ["total", "recon", "kld"]:
                sums[k] += losses[f"{k}_loss"].item() * x.size(0)
            sums["entropy"] += losses["entropy"].item() * x.size(0)
            eb = losses["effective_beta"]
            effective_beta_vals.append(eb if isinstance(eb, float) else eb.item())

            if hasattr(model, "distribution") and model.distribution in [
                "powerspherical",
                "clifford",
            ]:
                if hasattr(q_z, "concentration"):
                    concentration_stats.append(q_z.concentration.detach())

    n = len(loader.dataset)
    result = {f"test/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["test/entropy"] = sums["entropy"] / n
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
    return (1 - t) * z1 + t * z2


def get_latents(model, loader, device, n_samples=1000):
    """extract flattened latent vectors from model"""
    model.eval()
    latents = []
    labels_list = []
    images_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu = model.get_flat_latent(x)
            latents.append(mu.detach())
            labels_list.append(y)
            images_list.append(x.cpu())
            if len(torch.cat(latents, 0)) >= n_samples:
                break

    item_memory = torch.cat(latents, 0)[:n_samples]
    item_labels = torch.cat(labels_list, 0)[:n_samples].to(device)
    item_images = torch.cat(images_list, 0)[:n_samples]
    return item_memory, item_labels, item_images


def perform_knn_evaluation(model, train_loader, test_loader, device, sample_sizes):
    model.eval()

    all_train_z, all_train_y = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            z = model.get_flat_latent(x)
            all_train_z.append(z.cpu())
            all_train_y.append(y)
    all_train_z = torch.cat(all_train_z, 0).numpy()
    all_train_y = torch.cat(all_train_y, 0).numpy()

    all_test_z, all_test_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            z = model.get_flat_latent(x)
            all_test_z.append(z.cpu())
            all_test_y.append(y)
    all_test_z = torch.cat(all_test_z, 0).numpy()
    all_test_y = torch.cat(all_test_y, 0).numpy()

    metrics = {}
    for n in sample_sizes:
        n_train = min(n, len(all_train_z))
        n_test = min(n, len(all_test_z))
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(all_train_z[:n_train], all_train_y[:n_train])
        preds = knn.predict(all_test_z[:n_test])
        acc = accuracy_score(all_test_y[:n_test], preds)
        f1 = f1_score(all_test_y[:n_test], preds, average="weighted")
        metrics[f"knn_acc_{n}"] = acc
        metrics[f"knn_f1_{n}"] = f1

    return metrics


def plot_latent_tsne(model, loader, device, save_path, n_samples=2000):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.get_flat_latent(x)
            zs.append(z.cpu())
            ys.append(y)
            if len(torch.cat(zs, 0)) >= n_samples:
                break

    z_all = torch.cat(zs, 0)[:n_samples].numpy()
    y_all = torch.cat(ys, 0)[:n_samples].numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_all)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for c in range(10):
        mask = y_all == c
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=5, alpha=0.5, label=CLASS_NAMES[c])
    ax.legend(markerscale=3)
    ax.set_title("t-SNE of latent space")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def filter_dataset_by_class(dataset, exclude_class):
    if exclude_class < 0:
        return dataset
    indices = [i for i, (_, y) in enumerate(dataset) if y != exclude_class]
    return torch.utils.data.Subset(dataset, indices)


def get_excluded_class_subset(dataset, exclude_class):
    if exclude_class < 0:
        return None
    indices = [i for i, (_, y) in enumerate(dataset) if y == exclude_class]
    return torch.utils.data.Subset(dataset, indices)


def sample_prior_z(dist_name, latent_dim, n, device, l2_normalize=False):
    """sample from the prior for each distribution type."""
    if dist_name == "clifford":
        from dists.clifford import CliffordTorusUniform
        prior = CliffordTorusUniform(latent_dim, device=device)
        return prior.rsample((n,))  # (n, 2*latent_dim)
    elif dist_name == "powerspherical":
        z = torch.randn(n, latent_dim, device=device)
        return F.normalize(z, dim=-1)
    else:
        z = torch.randn(n, latent_dim, device=device)
        if l2_normalize:
            z = F.normalize(z, dim=-1)
        return z


def compute_generation_fid(model, dist_name, latent_dim, test_loader, device,
                            in_channels, n_samples=2048, batch_size=256):
    """frechet inception distance between prior samples (decoded) and test set."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  torchmetrics not available, skipping fid")
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
    z_dim = getattr(model, "latent_dim", latent_dim)
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            bs = min(batch_size, n_samples - n_done)
            z = sample_prior_z(dist_name, z_dim, bs, device, l2_normalize=l2_norm)
            imgs_01 = (model.decoder(z) * 0.5 + 0.5).clamp(0, 1)
            if in_channels == 1:
                imgs_01 = imgs_01.repeat(1, 3, 1, 1)
            fid_metric.update(imgs_01, real=False)
            n_done += bs

    score = fid_metric.compute().item()
    fid_metric.reset()
    return score


def main(args):
    script_start_time = time.time()
    timing_results = {}

    print(f"device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = args.latent_dims if args.latent_dims else [256, 512, 1024, 2048]
    distributions = (
        args.distributions
        if args.distributions
        else ["clifford", "powerspherical", "gaussian"]
    )

    dist_lr = {
        "clifford": args.lr,
        "powerspherical": 1e-4,
        "gaussian": args.lr,
    }

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set_full = datasets.CIFAR10(
        "data", train=True, download=True, transform=transform
    )
    test_set_full = datasets.CIFAR10(
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
            excluded_test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        print(
            f"excluding class {args.exclude_class}. excluded test set size: {len(excluded_test_set)}"
        )

    IMG_SHAPE = (3, 32, 32)
    BC_K_RANGE = list(range(5, 51, 5))
    RF_K_RANGE = list(range(2, 21, 2))

    across_dim_results = {
        d: {
            "knn_100": [],
            "knn_600": [],
            "knn_1000": [],
            "f1_100": [],
            "f1_600": [],
            "f1_1000": [],
            "mean_cosine": [],
            "dims": [],
        }
        for d in distributions
    }
    trial_metrics = {}

    for latent_dim in latent_dims:
        dim_results = {}

        for dist_name in distributions:
            for trial in range(args.n_trials):
                trial_suffix = f"-trial{trial+1}" if args.n_trials > 1 else ""
                exp_name = (
                    f"cifar10-{dist_name}-d{latent_dim}-{args.recon_loss}{trial_suffix}"
                )
                output_dir = f"results/{exp_name}"
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n== {exp_name} ==")
                exp_start_time = time.time()
                logger.start_run(exp_name, args)

                print(f"  {dist_name}: flat z, d={latent_dim} (CNN w/ residual)")
                model = CNNVAE(
                    latent_dim=latent_dim,
                    in_channels=3,
                    distribution=dist_name,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    use_learnable_beta=args.use_learnable_beta,
                    l2_normalize=(dist_name == "gaussian" and args.l2_norm),
                    img_size=32,
                )

                logger.watch_model(model)
                cur_lr = dist_lr.get(dist_name, args.lr)

                if args.use_learnable_beta:
                    sigma_ids = {id(model.log_sigma_0), id(model.log_sigma_1)}
                    optimizer = optim.AdamW(
                        [
                            {
                                "params": [
                                    p
                                    for p in model.parameters()
                                    if id(p) not in sigma_ids
                                ],
                                "lr": cur_lr,
                            },
                            {
                                "params": [model.log_sigma_0, model.log_sigma_1],
                                "lr": cur_lr * 0.1,
                            },
                        ]
                    )
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=cur_lr)

                best = float("inf")
                patience_counter = 0
                train_start_time = time.time()

                def kl_beta_for_epoch(e: int) -> float:
                    if e < args.warmup_epochs:
                        return (
                            min(1.0, (e + 1) / max(1, args.warmup_epochs))
                            * args.max_beta
                        )
                    if args.cycle_epochs <= 0:
                        return args.max_beta
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
                    beta = 1.0 if args.use_learnable_beta else kl_beta_for_epoch(epoch)

                    train_losses = train_epoch(
                        model, train_loader, optimizer, DEVICE, beta
                    )
                    test_losses = test_epoch(model, test_loader, DEVICE)

                    val = test_losses["test/recon_loss"] + test_losses["test/kld_loss"]
                    if np.isfinite(val) and val < best:
                        best = val
                        torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
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
                            f"early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)"
                        )
                        break

                train_time = time.time() - train_start_time
                print(
                    f"best total loss (recon+kld): {best:.4f}, training time: {train_time:.2f}s"
                )

                if os.path.exists(f"{output_dir}/best_model.pt"):
                    ckpt = torch.load(
                        f"{output_dir}/best_model.pt", map_location=DEVICE
                    )
                    if not args.use_learnable_beta:
                        ckpt = {
                            k: v
                            for k, v in ckpt.items()
                            if k not in ("log_sigma_0", "log_sigma_1")
                        }
                    model.load_state_dict(ckpt)

                    eval_start_time = time.time()
                    images = {}

                    # get latents for VSA tests
                    item_memory, item_labels, item_images = get_latents(
                        model, test_loader, DEVICE, n_samples=1000
                    )

                    # reconstructions
                    recon_path = save_reconstructions(
                        model,
                        test_loader,
                        DEVICE,
                        f"{output_dir}/reconstructions.png",
                    )
                    images["reconstructions"] = recon_path

                    # t-sne: only on first trial per (dim, dist) to save time
                    if trial == 0:
                        tsne_path = plot_latent_tsne(
                            model,
                            test_loader,
                            DEVICE,
                            f"{output_dir}/tsne.png",
                            n_samples=2000,
                        )
                        images["tsne"] = tsne_path

                    # knn evaluation
                    knn_metrics = perform_knn_evaluation(
                        model,
                        train_loader,
                        test_loader,
                        DEVICE,
                        [100, 600, 1000],
                    )

                    # bundle capacity test
                    print(f"running bundle capacity test ({dist_name})...")
                    bundle_cap_raw = vsa_bundle_capacity(
                        d=item_memory.shape[-1],
                        n_items=1000,
                        k_range=BC_K_RANGE,
                        n_trials=20,
                        normalize=True,
                        device=DEVICE,
                        plot=False,
                        save_dir=output_dir,
                        item_memory=item_memory,
                        use_braiding=False,
                        baseline_d=latent_dim,
                    )

                    # 1-item-per-class similarity
                    print(f"running 1-item-per-class test ({dist_name})...")
                    test_per_class_bundle_capacity_k_items(
                        d=latent_dim,
                        n_items=1000,
                        n_classes=10,
                        items_per_class=1,
                        n_trials=1,
                        normalize=True,
                        device=DEVICE,
                        plot=False,
                        save_dir=output_dir,
                        item_memory=item_memory,
                        labels=item_labels,
                        item_images=item_images,
                        use_braiding=False,
                        class_names=CLASS_NAMES,
                    )

                    # role-filler unbinding test (random keys)
                    print(f"running role-filler unbinding test ({dist_name})...")
                    rf_results = {}
                    rf_res = vsa_binding_unbinding(
                        d=item_memory.shape[-1],
                        n_items=1000,
                        k_range=RF_K_RANGE,
                        n_trials=20,
                        normalize=True,
                        device=DEVICE,
                        plot=False,
                        unbind_method="*",
                        save_dir=output_dir,
                        item_memory=item_memory,
                        bind_with_random=True,
                        baseline_d=latent_dim,
                    )
                    rf_results["role_filler_capacity"] = rf_res
                    role_filler_raw = rf_res

                    # cross-class bind/unbind test (two random pairs) — only first trial to save time
                    cross_class_1 = {}
                    cross_class_2 = {}
                    if trial == 0:
                        pair_classes = random.sample(range(10), 4)
                        print(f"running cross-class bind/unbind test ({dist_name}): pairs {pair_classes[:2]}, {pair_classes[2:]}...")
                        cross_class_1 = test_cross_class_bind_unbind(
                            model, test_loader, DEVICE, output_dir,
                            img_shape=(3, 32, 32),
                            class_a=pair_classes[0], class_b=pair_classes[1],
                        )
                        cross_class_2 = test_cross_class_bind_unbind(
                            model, test_loader, DEVICE, os.path.join(output_dir, "pair2"),
                            img_shape=(3, 32, 32),
                            class_a=pair_classes[2], class_b=pair_classes[3],
                        )

                    # self-binding: bind z with itself k times, unbind k times, measure similarity
                    deconv_dir = f"{output_dir}/deconv"
                    os.makedirs(deconv_dir, exist_ok=True)
                    print(f"running self-binding test * ({dist_name})...")
                    fourier_star = test_self_binding(
                        model,
                        test_loader,
                        DEVICE,
                        output_dir,
                        unbind_method="*",
                        img_shape=IMG_SHAPE,
                    )
                    print(f"running self-binding test † ({dist_name})...")
                    fourier_deconv = test_self_binding(
                        model,
                        test_loader,
                        DEVICE,
                        deconv_dir,
                        unbind_method="†",
                        img_shape=IMG_SHAPE,
                    )

                    mean_vector_acc = 0.0

                    # log everything
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
                            for k, v in fourier_deconv.items()
                            if isinstance(v, (int, float, bool))
                        }
                    )

                    # compute fid
                    gen_fid = compute_generation_fid(
                        model, dist_name, latent_dim,
                        test_loader, DEVICE, in_channels=3,
                        n_samples=2048, batch_size=256
                    )
                    print(f"generation FID: {gen_fid:.2f}")

                    logger.log_metrics(
                        {
                            **knn_metrics,
                            **fourier_metrics,
                            "mean_vector_cosine_acc": float(mean_vector_acc),
                            "final_best_total_loss": best,
                            **({"generation_fid": gen_fid} if gen_fid is not None and not math.isnan(gen_fid) else {}),
                        }
                    )

                    # per-trial vsa plots no longer logged to wandb — comparison plot is the canonical view

                    summary = {
                        "final_best_total_loss": best,
                        **fourier_metrics,
                        **knn_metrics,
                        "mean_vector_cosine_acc": float(mean_vector_acc),
                    }
                    logger.log_summary(summary)
                    logger.log_images(images)

                    # save raw vsa curve data to disk so we can replot locally without wandb
                    try:
                        raw_vsa = {
                            "bundle_cap": bundle_cap_raw,
                            "role_filler": role_filler_raw,
                            "self_binding_star": {
                                "k_values": fourier_star.get("k_values", []),
                                "k_sims": fourier_star.get("k_sims", []),
                            },
                            "self_binding_dagger": {
                                "k_values": fourier_deconv.get("k_values", []),
                                "k_sims": fourier_deconv.get("k_sims", []),
                            },
                        }
                        def _jsonable(o):
                            if isinstance(o, dict):
                                return {k: _jsonable(v) for k, v in o.items()}
                            if isinstance(o, (list, tuple)):
                                return [_jsonable(v) for v in o]
                            if hasattr(o, "tolist"):
                                return o.tolist()
                            return o
                        with open(f"{output_dir}/vsa_raw.json", "w") as f:
                            json.dump(_jsonable(raw_vsa), f)
                    except Exception as e:
                        print(f"warning: failed to save raw vsa data: {e}")

                    metrics_save_path = f"{output_dir}/metrics.json"
                    with open(metrics_save_path, "w") as f:
                        json.dump(summary, f, indent=2)

                    eval_time = time.time() - eval_start_time
                    exp_time = time.time() - exp_start_time
                    timing_results[exp_name] = {
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
                        "mean_cosine": float(mean_vector_acc),
                    }
                    key = (latent_dim, dist_name)
                    if key not in trial_metrics:
                        trial_metrics[key] = []
                    trial_metrics[key].append({
                        "knn_acc_100": knn_metrics.get("knn_acc_100", 0.0),
                        "knn_acc_600": knn_metrics.get("knn_acc_600", 0.0),
                        "knn_acc_1000": knn_metrics.get("knn_acc_1000", 0.0),
                        "knn_f1_100": knn_metrics.get("knn_f1_100", 0.0),
                        "knn_f1_600": knn_metrics.get("knn_f1_600", 0.0),
                        "knn_f1_1000": knn_metrics.get("knn_f1_1000", 0.0),
                        "mvc": float(mean_vector_acc),
                        "fid": gen_fid if gen_fid is not None and not math.isnan(gen_fid) else float("nan"),
                        "best_loss": best,
                    })
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
                    across_dim_results[dist_name]["mean_cosine"].append(
                        float(mean_vector_acc)
                    )

                logger.finish_run()

        # random HRR baseline + cross-dist comparison
        try:
            ref_items = torch.randn(1000, latent_dim, device=DEVICE)
            ref_items = F.normalize(ref_items, p=2, dim=-1)
            ref_bc = vsa_bundle_capacity(
                d=latent_dim,
                n_items=1000,
                k_range=BC_K_RANGE,
                n_trials=20,
                normalize=True,
                device=DEVICE,
                item_memory=ref_items,
            )
            ref_rf = vsa_binding_unbinding(
                d=latent_dim,
                n_items=1000,
                k_range=RF_K_RANGE,
                n_trials=20,
                normalize=True,
                device=DEVICE,
                unbind_method="*",
                item_memory=ref_items,
                bind_with_random=True,
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

            # unitary HRR reference: vectors with unit fourier magnitude
            uni_items = vsa_unitary_init(1000, latent_dim, device=DEVICE)
            uni_items = F.normalize(uni_items, p=2, dim=-1)
            uni_bc = vsa_bundle_capacity(
                d=latent_dim, n_items=1000, k_range=BC_K_RANGE,
                n_trials=20, normalize=True, device=DEVICE,
                item_memory=uni_items,
            )
            uni_rf = vsa_binding_unbinding(
                d=latent_dim, n_items=1000, k_range=RF_K_RANGE,
                n_trials=20, normalize=True, device=DEVICE,
                unbind_method="*", item_memory=uni_items, bind_with_random=True,
            )
            z_uni = vsa_unitary_init(1, latent_dim, device=DEVICE)
            z_uni = F.normalize(z_uni, p=2, dim=-1)
            uni_sims = []
            for m in range(1, k_max + 1):
                cur = z_uni.clone()
                for _ in range(m):
                    cur = vsa_bind(cur, z_uni)
                for _ in range(m):
                    cur = vsa_unbind(cur, z_uni, method="*")
                uni_sims.append(F.cosine_similarity(cur, z_uni, dim=-1).mean().item())
            dim_results["unitary"] = {
                "bundle_cap": uni_bc,
                "role_filler": uni_rf,
                "self_binding_k_sims": uni_sims,
                "self_binding_k_values": list(range(1, k_max + 1)),
            }
            comp_dir = "results/comparisons/cifar10"
            comp_path = plot_cross_dist_comparison_dim(
                dim_results, latent_dim, "cifar10", comp_dir
            )
            print(f"saved cross-dist comparison to {comp_path}")
            if not args.no_wandb and comp_path and os.path.exists(comp_path):
                try:
                    import wandb as _wandb
                    summary_run = _wandb.init(
                        project=args.wandb_project,
                        name=f"comparison-cifar10-d{latent_dim}",
                        job_type="comparison",
                        reinit=True,
                        config={"dataset": "cifar10", "latent_dim": latent_dim},
                    )
                    _wandb.log({
                        f"vsa_comparison_d{latent_dim}": _wandb.Image(comp_path),
                    })
                    summary_run.finish()
                except Exception as e:
                    print(f"warning: wandb upload of cross-dist plot failed: {e}")
        except Exception as e:
            print(f"warning: cross-dist comparison failed for d={latent_dim}: {e}")

    try:
        comp_dir = "results/comparisons/cifar10"
        across_path = plot_across_dims_comparison(
            across_dim_results, latent_dims, "cifar10", comp_dir
        )
        print(f"saved across-dims comparison to {across_path}")
    except Exception as e:
        print(f"warning: across-dims comparison failed for cifar10: {e}")

    # build unified csv results table
    if trial_metrics:
        try:
            import pandas as pd
            rows = []
            for (ldim, dist), trials in sorted(trial_metrics.items()):
                row = {"d": ldim, "dist": dist}
                for metric in ["knn_acc_100", "knn_acc_600", "knn_acc_1000",
                               "knn_f1_100", "knn_f1_600", "knn_f1_1000",
                               "mvc"]:
                    vals = [t[metric] * 100 for t in trials]
                    row[metric] = f"{np.mean(vals):.1f}±{np.std(vals):.1f}" if len(vals) > 1 else f"{vals[0]:.1f}"
                fid_vals = [t["fid"] for t in trials if not math.isnan(t["fid"])]
                row["fid"] = f"{np.mean(fid_vals):.1f}±{np.std(fid_vals):.1f}" if len(fid_vals) > 1 else (f"{fid_vals[0]:.1f}" if fid_vals else "N/A")
                loss_vals = [t["best_loss"] for t in trials]
                row["best_loss"] = f"{np.mean(loss_vals):.4f}±{np.std(loss_vals):.4f}" if len(loss_vals) > 1 else f"{loss_vals[0]:.4f}"
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv("cifar10_results.csv", index=False)
            print(f"\n{'='*25} cifar10 results {'='*25}")
            print(df.to_string(index=False))
            print("saved to cifar10_results.csv")
        except ImportError:
            with open("cifar10_results.json", "w") as f:
                json.dump([{"dim": k[0], "dist": k[1], "trials": v} for k, v in trial_metrics.items()], f, indent=2)

    script_total_time = time.time() - script_start_time
    timing_results["total_script_time_s"] = script_total_time
    with open("cifar10_train_timing.json", "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\ntotal script execution time: {script_total_time:.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="cifar-10 VAE experiments with cliffordAR S-VAE + baselines"
    )
    p.add_argument("--epochs", type=int, default=1000, help="training epochs")
    p.add_argument("--warmup_epochs", type=int, default=25, help="kl warmup epochs")
    p.add_argument("--batch_size", type=int, default=256, help="batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument(
        "--no-l2_norm",
        dest="l2_norm",
        action="store_false",
        help="disable L2 normalization for Gaussian VAE",
    )
    p.set_defaults(l2_norm=True)
    p.add_argument("--recon_loss", type=str, default="l1", choices=["mse", "l1"])
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--max_beta", type=float, default=1.0)
    p.add_argument("--min_beta", type=float, default=0.1)
    p.add_argument("--use_learnable_beta", action="store_true")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    p.add_argument(
        "--wandb_project", type=str, default="clifford-experiments-CNN-cifar10"
    )
    p.add_argument("--patience", type=int, default=25, help="early stopping patience")
    p.add_argument("--cycle_epochs", type=int, default=100)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument(
        "--exclude_class",
        type=int,
        default=-1,
        help="exclude class (-1=none) | 0:plane 1:auto 2:bird 3:cat 4:deer 5:dog 6:frog 7:horse 8:ship 9:truck",
    )
    p.add_argument(
        "--latent_dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
    )
    p.add_argument(
        "--distributions",
        type=str,
        nargs="+",
        default=None,
        help="distributions to test (default: spherear clifford powerspherical gaussian)",
    )
    args = p.parse_args()
    main(args)

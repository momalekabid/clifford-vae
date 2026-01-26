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
import time
import json
from scipy import stats

import torch.nn.functional as F
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models import VAE
from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_cross_class_bind_unbind,
    plot_clifford_manifold_visualization,
    plot_powerspherical_manifold_visualization,
    plot_gaussian_manifold_visualization,
    plot_cross_dist_comparison_dim,
    plot_across_dims_comparison,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    bind as vsa_bind,
    unbind as vsa_unbind,
)



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

IMG_SIZE = 64
IN_CHANNELS = 3
IMG_SHAPE = (3, 64, 64)


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


def plot_latent_distributions(model, loader, device, save_path, n_dims=50, n_samples=2000):
    """
    plot histograms of individual latent dimensions (mu from encoder).
    contrasts the learned distribution shape across dimensions.
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
        ax.set_title(f"latent var {i+1}", fontsize=7)
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


def plot_latent_interpolations(model, loader, device, save_dir, n_steps=10, n_pairs=5):
    """
    plot interpolations between pairs of samples.
    celeba, we just pick random pairs.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")

    all_z = []
    all_imgs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, params = model.encoder(x)
            z, _, _ = model.reparameterize(mu, params)
            all_z.append(z)
            all_imgs.append(x.cpu())
            if len(torch.cat(all_z, 0)) >= 100:
                break
    all_z = torch.cat(all_z, 0)[:100]
    all_imgs = torch.cat(all_imgs, 0)[:100]

    pairs = []
    for _ in range(n_pairs):
        idx1, idx2 = np.random.choice(len(all_z), 2, replace=False)
        pairs.append((idx1, idx2))

    if dist == "clifford":
        interp_fn = slerp
        interp_name = "slerp"
    elif dist == "powerspherical":
        interp_fn = slerp
        interp_name = "slerp"
    else:
        interp_fn = lerp
        interp_name = "lerp"

    ts = torch.linspace(0, 1, n_steps).to(device)
    fig, axes = plt.subplots(n_pairs, n_steps + 2, figsize=(2 * (n_steps + 2), 2 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, (idx1, idx2) in enumerate(pairs):
            z1 = all_z[idx1].unsqueeze(0)
            z2 = all_z[idx2].unsqueeze(0)
            img1 = all_imgs[idx1]
            img2 = all_imgs[idx2]

            img1_show = (img1 * 0.5 + 0.5).clamp(0, 1)
            axes[row, 0].imshow(img1_show.permute(1, 2, 0))
            axes[row, 0].set_title("src")
            axes[row, 0].axis("off")

            for i, t in enumerate(ts):
                z_interp = interp_fn(z1, z2, t.item())
                x_recon = model.decoder(z_interp)
                img = (x_recon[0].cpu() * 0.5 + 0.5).clamp(0, 1)
                axes[row, i + 1].imshow(img.permute(1, 2, 0))
                axes[row, i + 1].set_title(f"t={t.item():.1f}")
                axes[row, i + 1].axis("off")

            img2_show = (img2 * 0.5 + 0.5).clamp(0, 1)
            axes[row, -1].imshow(img2_show.permute(1, 2, 0))
            axes[row, -1].set_title("tgt")
            axes[row, -1].axis("off")

    plt.suptitle(f"latent interpolation ({interp_name})", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"interpolation_{interp_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_phase_attribute_correlation(model, loader, device, save_dir, attr_names,
                                     n_samples=5000, top_k=20):
    """
    compute point-biserial correlation between each latent dimension (or phase for
    clifford) and each of the 40 celeba binary attributes. shows which latent dims
    encode which semantic attributes.

    for clifford: extracts phase angles from mu (the encoder output before sampling).
    for gaussian/powerspherical: uses raw mu dimensions.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")
    latent_dim = model.latent_dim

    all_mu = []
    all_attrs = []
    n_collected = 0
    with torch.no_grad():
        for x, attrs in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            all_mu.append(mu.cpu())
            all_attrs.append(attrs)
            n_collected += len(x)
            if n_collected >= n_samples:
                break

    mu = torch.cat(all_mu, dim=0)[:n_samples].numpy()
    attrs = torch.cat(all_attrs, dim=0)[:n_samples].numpy()  # (n_samples, 40)

    if dist == "clifford":
        phases = np.arctan2(np.sin(mu), np.cos(mu))
        latent_repr = phases
        repr_label = "phase"
    else:
        latent_repr = mu
        repr_label = "dim"

    n_attrs = attrs.shape[1]
    n_latent = latent_repr.shape[1]

    corr_matrix = np.zeros((n_latent, n_attrs))
    pval_matrix = np.ones((n_latent, n_attrs))
    for i in range(n_latent):
        for j in range(n_attrs):
            if attrs[:, j].std() < 1e-6:
                continue
            r, p = stats.pointbiserialr(attrs[:, j], latent_repr[:, i])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    flat_idx = np.argsort(np.abs(corr_matrix).flatten())[::-1][:top_k]
    top_pairs = []
    for idx in flat_idx:
        li, ai = divmod(idx, n_attrs)
        top_pairs.append((li, ai, corr_matrix[li, ai], pval_matrix[li, ai]))

    fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.35)))
    labels = [f"{repr_label}{p[0]}<->{attr_names[p[1]]}" for p in top_pairs]
    values = [p[2] for p in top_pairs]
    colors = ['steelblue' if v > 0 else 'coral' for v in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("correlation (r)")
    ax.set_title(f"{dist}: top {top_k} latent-attribute correlations")
    ax.invert_yaxis()
    plt.tight_layout()

    bar_path = os.path.join(save_dir, "top_attribute_correlations.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()

    corr_data = {
        "top_pairs": [
            {"latent_dim": int(p[0]), "attribute": attr_names[p[1]],
             "correlation": float(p[2]), "pvalue": float(p[3])}
            for p in top_pairs
        ],
        "max_abs_correlation": float(np.abs(corr_matrix).max()),
        "mean_abs_correlation": float(np.abs(corr_matrix).mean()),
    }
    with open(os.path.join(save_dir, "phase_attribute_correlations.json"), "w") as f:
        json.dump(corr_data, f, indent=2)

    print(f"  top 5 correlations:")
    for p in top_pairs[:5]:
        print(f"    {repr_label}{p[0]} <-> {attr_names[p[1]]}: r={p[2]:.3f} (p={p[3]:.2e})")

    return {
        "bar_path": bar_path,
        "corr_data": corr_data,
    }


def plot_attribute_traversal(model, loader, device, save_dir, attr_names,
                              corr_data, n_steps=9, n_samples_per_attr=3):
    """
    for the top correlated latent-attribute pairs, vary that latent dimension
    while keeping others fixed, and decode to visualize what changes.
    this shows whether individual latent dims control specific semantic attributes.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")

    with torch.no_grad():
        x_batch, _ = next(iter(loader))
        x_batch = x_batch[:20].to(device)
        mu, params = model.encoder(x_batch)
        z, _, _ = model.reparameterize(mu, params)

    top_pairs = corr_data["top_pairs"]
    seen_attrs = set()
    selected = []
    for p in top_pairs:
        if p["attribute"] not in seen_attrs and len(selected) < 5:
            selected.append(p)
            seen_attrs.add(p["attribute"])

    if not selected:
        return None

    n_attrs = len(selected)
    fig, axes = plt.subplots(
        n_attrs * n_samples_per_attr, n_steps,
        figsize=(2 * n_steps, 2 * n_attrs * n_samples_per_attr)
    )
    if n_attrs * n_samples_per_attr == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for attr_idx, pair_info in enumerate(selected):
            dim_idx = pair_info["latent_dim"]
            attr_name = pair_info["attribute"]
            corr_val = pair_info["correlation"]

            for sample_idx in range(n_samples_per_attr):
                row = attr_idx * n_samples_per_attr + sample_idx
                z_base = z[sample_idx].clone()

                # determine traversal range from the data
                dim_vals = mu[:, dim_idx] if dim_idx < mu.shape[1] else mu[:, 0]
                lo = dim_vals.min().item()
                hi = dim_vals.max().item()
                margin = (hi - lo) * 0.2
                traverse_range = torch.linspace(lo - margin, hi + margin, n_steps).to(device)

                for step_idx, val in enumerate(traverse_range):
                    z_mod = z_base.clone().unsqueeze(0)

                    if dist == "clifford":
                        mu_mod = mu[sample_idx].clone().unsqueeze(0)
                        mu_mod[0, dim_idx] = val
                        z_mod, _, _ = model.reparameterize(mu_mod, params[sample_idx].unsqueeze(0))
                    else:
                        if dim_idx < z_mod.shape[1]:
                            z_mod[0, dim_idx] = val

                    x_recon = model.decoder(z_mod)
                    img = (x_recon[0].cpu() * 0.5 + 0.5).clamp(0, 1)
                    axes[row, step_idx].imshow(img.permute(1, 2, 0))
                    axes[row, step_idx].axis("off")

                    if sample_idx == 0 and step_idx == 0:
                        axes[row, 0].set_ylabel(
                            f"{attr_name}\n(r={corr_val:.2f})",
                            fontsize=7, rotation=0, labelpad=60, va='center'
                        )

    fig.suptitle(f"{dist}: attribute traversal via latent dimension manipulation", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "attribute_traversal.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def sample_prior_z(dist_name, latent_dim, n, device, l2_normalize=False):
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
                            n_samples=2048, batch_size=256):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("  torchmetrics not available, skipping generation fid")
        return float("nan")
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    n_real = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x_01 = (x.to(device) * 0.5 + 0.5).clamp(0, 1)
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
            fid_metric.update(imgs_01, real=False)
            n_done += bs
    score = fid_metric.compute().item()
    fid_metric.reset()
    return score


def main(args):
    script_start_time = time.time()
    timing_results = {}

    print(f"Device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = args.latent_dims if args.latent_dims else [2048, 4096]
    distributions = ["clifford", "powerspherical", "gaussian", "gaussian_nol2"]
    BC_K_RANGE = list(range(5, 51, 5))
    RF_K_RANGE = list(range(2, 21, 2))
    across_dim_results = {d: {"knn": [], "fid": [], "dims": []} for d in distributions}

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasets.CelebA(
        "data", split="train", download=True, transform=transform, target_type="attr"
    )
    test_set = datasets.CelebA(
        "data", split="test", download=True, transform=transform, target_type="attr"
    )
    attr_names = train_set.attr_names

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"celeba train: {len(train_set)}, test: {len(test_set)}")

    for latent_dim in latent_dims:
        dim_results = {}

        for dist_name in distributions:
            for trial in range(args.n_trials):
                trial_suffix = f"-trial{trial+1}" if args.n_trials > 1 else ""
                exp_name = f"celeba-{dist_name}-d{latent_dim}-{args.recon_loss}{trial_suffix}"
                output_dir = f"results/{exp_name}"
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n== {exp_name} ==")
                exp_start_time = time.time()
                logger.start_run(exp_name, args)

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
                    in_channels=IN_CHANNELS,
                    distribution=actual_dist,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    freq_weight=0.0,
                    l2_normalize=l2_norm,
                    img_size=IMG_SIZE,
                    use_learnable_beta=args.use_learnable_beta,
                )
                logger.watch_model(model)
                if args.use_learnable_beta:
                    sigma_ids = {id(model.log_sigma_0), id(model.log_sigma_1)}
                    optimizer = optim.AdamW([
                        {"params": [p for p in model.parameters() if id(p) not in sigma_ids], "lr": args.lr},
                        {"params": [model.log_sigma_0, model.log_sigma_1], "lr": args.lr * 0.1},
                    ])
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
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
                    # log beta only if not using learnable (since effective_beta is already logged)
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
                    images = {}

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

                    t0 = time.time()
                    print(f"generating latent interpolations...")
                    interp_path = plot_latent_interpolations(
                        model,
                        test_loader,
                        DEVICE,
                        output_dir,
                        n_steps=10,
                        n_pairs=5,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    t0 = time.time()
                    print(f"collecting item memory for vsa tests...")
                    with torch.no_grad():
                        latents = []
                        for x, _ in test_loader:
                            x = x.to(DEVICE)
                            _, _, _, mu = model(x)
                            latents.append(mu.detach())
                            if sum(l.shape[0] for l in latents) >= 1000:
                                break
                        item_memory = torch.cat(latents, 0)[:1000]
                    print(f"  collected {item_memory.shape[0]} latents ({time.time() - t0:.2f}s)")

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

                    t0 = time.time()
                    print(f"running cross-class bind/unbind test ({dist_name})...")
                    cross_class_star = test_cross_class_bind_unbind(
                        model,
                        test_loader,
                        DEVICE,
                        output_dir,
                        unbind_method="*",
                        img_shape=IMG_SHAPE,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    # bundle capacity (schlegel et al. sec 5.4)
                    t0 = time.time()
                    print(f"running bundle capacity test ({dist_name})...")
                    bundle_cap_raw = vsa_bundle_capacity(
                        d=item_memory.shape[-1],
                        n_items=min(1000, item_memory.shape[0]),
                        k_range=BC_K_RANGE,
                        n_trials=20,
                        normalize=True,
                        device=DEVICE,
                        plot=True,
                        save_dir=output_dir,
                        item_memory=item_memory,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    # role-filler unbinding (schlegel et al. sec 3.3)
                    t0 = time.time()
                    print(f"running role-filler unbinding test ({dist_name})...")
                    role_filler_raw = vsa_binding_unbinding(
                        d=item_memory.shape[-1],
                        n_items=min(1000, item_memory.shape[0]),
                        k_range=RF_K_RANGE,
                        n_trials=20,
                        normalize=True,
                        device=DEVICE,
                        plot=True,
                        unbind_method="*",
                        save_dir=output_dir,
                        item_memory=item_memory,
                        bind_with_random=True,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    t0 = time.time()
                    print(f"running phase-attribute correlation analysis...")
                    corr_results = plot_phase_attribute_correlation(
                        model, test_loader, DEVICE, output_dir, attr_names,
                        n_samples=5000, top_k=50,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    t0 = time.time()
                    print(f"generating attribute traversal visualization...")
                    traversal_path = plot_attribute_traversal(
                        model, test_loader, DEVICE, output_dir, attr_names,
                        corr_results["corr_data"], n_steps=9, n_samples_per_attr=3,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    t0 = time.time()
                    print(f"computing generation fid...")
                    gen_fid = compute_generation_fid(
                        model, dist_name, latent_dim, test_loader, DEVICE,
                        n_samples=2048,
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

                    images.update(
                        {
                            "reconstructions": recon_path,
                            "latent_distributions": latent_dist_path,
                            "latent_interpolation": interp_path,
                        }
                    )

                    if corr_results.get("bar_path"):
                        images["top_attribute_correlations"] = corr_results["bar_path"]
                    if traversal_path:
                        images["attribute_traversal"] = traversal_path

                    bc_plot = os.path.join(output_dir, "bundle_capacity.png")
                    if os.path.exists(bc_plot):
                        images["bundle_capacity"] = bc_plot
                    rf_plot = os.path.join(output_dir, "role_filler_capacity.png")
                    if os.path.exists(rf_plot):
                        images["role_filler_capacity"] = rf_plot

                    sp = fourier_star.get("similarity_after_k_binds_plot_path")
                    if sp:
                        images["similarity_after_k_binds_*"] = sp
                    rp = fourier_star.get("recon_after_k_binds_plot_path")
                    if rp:
                        images["recon_after_k_binds_*"] = rp
                    if cross_class_star.get("cross_class_bind_unbind_plot_path"):
                        images["cross_class_binding_star"] = cross_class_star[
                            "cross_class_bind_unbind_plot_path"
                        ]

                    if dist_name == "clifford" and 2 <= model.latent_dim <= 8:
                        cliff_viz = plot_clifford_manifold_visualization(
                            model, DEVICE, output_dir,
                            n_grid=16, dims=(0, 1), img_shape=IMG_SHAPE,
                        )
                        if cliff_viz:
                            images["clifford_manifold_visualization"] = cliff_viz
                    elif dist_name == "powerspherical" and 2 <= model.latent_dim <= 8:
                        pow_viz = plot_powerspherical_manifold_visualization(
                            model, DEVICE, output_dir,
                            n_samples=1000, dims=(0, 1), img_shape=IMG_SHAPE,
                        )
                        if pow_viz:
                            images["powerspherical_manifold_visualization"] = pow_viz
                    elif dist_name == "gaussian" and 2 <= model.latent_dim <= 8:
                        gauss_viz = plot_gaussian_manifold_visualization(
                            model, DEVICE, output_dir,
                            n_samples=1000, dims=(0, 1), img_shape=IMG_SHAPE,
                        )
                        if gauss_viz:
                            images["gaussian_manifold_visualization"] = gauss_viz

                    logger.log_metrics(
                        {
                            **fourier_metrics,
                            "final_best_total_loss": best,
                            "generation_fid": gen_fid,
                            "cross_class_bind_unbind_similarity_star": cross_class_star.get(
                                "cross_class_bind_unbind_similarity", 0.0
                            ),
                        }
                    )

                    summary = {
                        "final_best_total_loss": best,
                        **fourier_metrics,
                        "generation_fid": gen_fid,
                        "max_attr_correlation": corr_results["corr_data"]["max_abs_correlation"],
                        "mean_attr_correlation": corr_results["corr_data"]["mean_abs_correlation"],
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
                        "self_binding_k_sims": fourier_star.get("k_sims", []),
                        "self_binding_k_values": fourier_star.get("k_values", []),
                        "gen_fid": gen_fid,
                    }
                    across_dim_results[dist_name]["dims"].append(latent_dim)
                    across_dim_results[dist_name]["fid"].append(gen_fid)

                logger.finish_run()

            try:
                ref_items = F.normalize(
                    torch.randn(1000, latent_dim, device=DEVICE), p=2, dim=-1
                )
                ref_bc = vsa_bundle_capacity(
                    d=latent_dim, n_items=1000, k_range=BC_K_RANGE,
                    n_trials=20, normalize=True, device=DEVICE, item_memory=ref_items,
                )
                ref_rf = vsa_binding_unbinding(
                    d=latent_dim, n_items=1000, k_range=RF_K_RANGE,
                    n_trials=20, normalize=True, device=DEVICE,
                    unbind_method="*", item_memory=ref_items, bind_with_random=True,
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
                dim_results["random_unitary"] = {
                    "bundle_cap": ref_bc,
                    "role_filler": ref_rf,
                    "self_binding_k_sims": ref_sims,
                    "self_binding_k_values": list(range(1, k_max + 1)),
                }
                comp_dir = "results/comparisons/celeba"
                comp_path = plot_cross_dist_comparison_dim(
                    dim_results, latent_dim, "celeba", comp_dir
                )
                print(f"saved cross-dist comparison to {comp_path}")
            except Exception as e:
                print(f"warning: cross-dist comparison failed for d={latent_dim}: {e}")

    # across-dim comparison after all latent_dims
    try:
        across_path = plot_across_dims_comparison(
            across_dim_results, latent_dims, "celeba", "results/comparisons/celeba"
        )
        print(f"saved across-dims comparison to {across_path}")
    except Exception as e:
        print(f"warning: across-dims comparison failed: {e}")

    script_total_time = time.time() - script_start_time
    timing_results["total_script_time_s"] = script_total_time
    with open("celeba_train_timing.json", "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\ntotal script execution time: {script_total_time:.2f}s")
    print(f"timing results saved to celeba_train_timing.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="clifford vae experiments on celeba"
    )
    p.add_argument("--epochs", type=int, default=500, help="training epochs")
    p.add_argument("--warmup_epochs", type=int, default=100, help="kl warmup epochs (ignored if --use_learnable_beta)")
    p.add_argument("--batch_size", type=int, default=256, help="batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument(
        "--no-l2_norm",
        dest="l2_norm",
        action="store_false",
        help="disable L2 normalization for Gaussian VAE",
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
        default="clifford-experiments-CelebA",
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
        "--latent_dims",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="latent dims to test",
    )
    args = p.parse_args()
    main(args)

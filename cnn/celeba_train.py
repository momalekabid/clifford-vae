import argparse
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

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models import VAE, compute_test_metrics
from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_cross_class_bind_unbind,
    plot_clifford_manifold_visualization,
    plot_powerspherical_manifold_visualization,
    plot_gaussian_manifold_visualization,
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

        if hasattr(model, "distribution") and model.distribution in [
            "powerspherical",
            "clifford",
        ]:
            if hasattr(q_z, "concentration"):
                concentration_stats.append(q_z.concentration.detach())

    n = len(loader.dataset)
    result = {f"train/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["train/entropy"] = sums["entropy"] / n

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
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, q_z, p_z, _ = model(x)
            losses = model.compute_loss(x, x_recon, q_z, p_z, beta=1.0)
            for k in ["total", "recon", "kld"]:
                sums[k] += losses[f"{k}_loss"].item() * x.size(0)
            sums["entropy"] += losses["entropy"].item() * x.size(0)

            if hasattr(model, "distribution") and model.distribution in [
                "powerspherical",
                "clifford",
            ]:
                if hasattr(q_z, "concentration"):
                    concentration_stats.append(q_z.concentration.detach())

    n = len(loader.dataset)
    result = {f"test/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["test/entropy"] = sums["entropy"] / n

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
    celeba has no class labels, so we just pick random pairs.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")

    # collect samples
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

    # pick random pairs
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


def plot_fourier_coefficients(model, loader, device, save_path, n_samples=10):
    """
    plot fourier coefficient magnitudes of the actual decoder input z.
    for clifford: FFT(z) should have unit magnitude by construction.
    """
    model.eval()
    dist = getattr(model, "distribution", "gaussian")

    all_latents = []
    n_collected = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, params = model.encoder(x)
            z, _, _ = model.reparameterize(mu, params)
            all_latents.append(z.cpu())
            n_collected += len(x)
            if n_collected >= 200:
                break

    latents = torch.cat(all_latents, dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, min(n_samples, len(latents))))
    for i in range(min(n_samples, len(latents))):
        z = latents[i]
        fft_coeffs = np.fft.fft(z)
        magnitudes = np.abs(fft_coeffs)
        n_coeffs = len(magnitudes) // 2
        axes[0].plot(range(n_coeffs), magnitudes[:n_coeffs], alpha=0.6, linewidth=1, color=colors[i])

    axes[0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='unit magnitude')
    axes[0].set_xlabel("frequency bin k")
    axes[0].set_ylabel("magnitude")
    axes[0].set_title(f"{dist}: FFT magnitudes of decoder input z ({n_samples} samples)")
    axes[0].legend()

    all_mag_per_bin = []
    for z in latents:
        fft_coeffs = np.fft.fft(z)
        magnitudes = np.abs(fft_coeffs)
        all_mag_per_bin.append(magnitudes[:len(magnitudes)//2])

    all_mag_per_bin = np.array(all_mag_per_bin)
    mean_per_bin = np.mean(all_mag_per_bin, axis=0)
    std_per_bin = np.std(all_mag_per_bin, axis=0)
    freq_bins = np.arange(len(mean_per_bin))

    axes[1].plot(freq_bins, mean_per_bin, color='blue', linewidth=2, label='mean')
    axes[1].fill_between(freq_bins, mean_per_bin - std_per_bin, mean_per_bin + std_per_bin,
                         alpha=0.3, color='blue', label='+-1 std')
    axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='unit magnitude')
    axes[1].set_xlabel("frequency bin k")
    axes[1].set_ylabel("magnitude")
    axes[1].set_title(f"{dist}: mean FFT magnitude across {len(latents)} samples")
    axes[1].legend()

    all_magnitudes = all_mag_per_bin[:, 1:].flatten()
    mean_mag = np.mean(all_magnitudes)
    std_mag = np.std(all_magnitudes)
    near_unit = np.mean(np.abs(all_magnitudes - 1.0) < 0.1) * 100

    stats_text = f"mean: {mean_mag:.3f}, std: {std_mag:.3f}, % near unit (+-0.1): {near_unit:.1f}%"
    fig.suptitle(f"{dist} latents: FFT structure of decoder input z\n{stats_text}", fontsize=11)

    plt.tight_layout()
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

    # for clifford, convert mu to phase angles for more interpretable correlation
    if dist == "clifford":
        # mu is the raw encoder output (angles before torus projection)
        # wrap to [-pi, pi] for circular analysis
        phases = np.arctan2(np.sin(mu), np.cos(mu))
        latent_repr = phases
        repr_label = "phase"
    else:
        latent_repr = mu
        repr_label = "dim"

    n_attrs = attrs.shape[1]
    n_latent = latent_repr.shape[1]

    # compute point-biserial correlation for each (latent_dim, attribute) pair
    corr_matrix = np.zeros((n_latent, n_attrs))
    pval_matrix = np.ones((n_latent, n_attrs))
    for i in range(n_latent):
        for j in range(n_attrs):
            # skip if attribute is constant
            if attrs[:, j].std() < 1e-6:
                continue
            r, p = stats.pointbiserialr(attrs[:, j], latent_repr[:, i])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    # find top-k strongest correlations
    flat_idx = np.argsort(np.abs(corr_matrix).flatten())[::-1][:top_k]
    top_pairs = []
    for idx in flat_idx:
        li, ai = divmod(idx, n_attrs)
        top_pairs.append((li, ai, corr_matrix[li, ai], pval_matrix[li, ai]))

    # plot 1: heatmap of top correlated dimensions
    # pick the latent dims and attributes that appear in top pairs
    top_latent_dims = sorted(set(p[0] for p in top_pairs))[:20]
    top_attr_dims = sorted(set(p[1] for p in top_pairs))[:20]

    sub_corr = corr_matrix[np.ix_(top_latent_dims, top_attr_dims)]

    fig, ax = plt.subplots(figsize=(max(10, len(top_attr_dims) * 0.6),
                                     max(6, len(top_latent_dims) * 0.4)))
    im = ax.imshow(sub_corr, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(top_attr_dims)))
    ax.set_xticklabels([attr_names[i] for i in top_attr_dims], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top_latent_dims)))
    ax.set_yticklabels([f"{repr_label} {i}" for i in top_latent_dims], fontsize=7)
    ax.set_title(f"{dist}: latent {repr_label}-attribute correlation (top {top_k} pairs)", fontsize=10)
    plt.colorbar(im, ax=ax, label="point-biserial r")
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, "phase_attribute_correlation.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    # plot 2: bar chart of top-k correlations
    fig, ax = plt.subplots(figsize=(12, 5))
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

    # save raw correlation data
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
        "heatmap_path": heatmap_path,
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

    # get a few reference images
    with torch.no_grad():
        x_batch, _ = next(iter(loader))
        x_batch = x_batch[:20].to(device)
        mu, params = model.encoder(x_batch)
        z, _, _ = model.reparameterize(mu, params)

    top_pairs = corr_data["top_pairs"]
    # pick top 5 unique attributes
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
                        # for clifford, we need to modify the phase and re-project
                        # modify mu, then re-sample through reparameterize
                        mu_mod = mu[sample_idx].clone().unsqueeze(0)
                        mu_mod[0, dim_idx] = val
                        z_mod, _, _ = model.reparameterize(mu_mod, params[sample_idx].unsqueeze(0))
                    else:
                        # for gaussian/powerspherical, directly modify the z vector
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


def main(args):
    script_start_time = time.time()
    timing_results = {}

    print(f"Device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = args.latent_dims if args.latent_dims else [2048, 4096]
    distributions = ["clifford", "powerspherical", "gaussian"]

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # celeba uses 'split' kwarg instead of 'train'
    # target_type='attr' gives us 40 binary attribute labels per image
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
        for dist_name in distributions:
            for trial in range(args.n_trials):
                trial_suffix = f"-trial{trial+1}" if args.n_trials > 1 else ""
                exp_name = f"celeba-{dist_name}-d{latent_dim}-{args.recon_loss}{trial_suffix}"
                output_dir = f"results/{exp_name}"
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n== {exp_name} ==")
                exp_start_time = time.time()
                logger.start_run(exp_name, args)

                l2_norm = args.l2_norm if dist_name == "gaussian" else False
                model = VAE(
                    latent_dim=latent_dim,
                    in_channels=IN_CHANNELS,
                    distribution=dist_name,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    freq_weight=0.0,
                    l2_normalize=l2_norm,
                    img_size=IMG_SIZE,
                )
                logger.watch_model(model)
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

                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            **train_losses,
                            **test_losses,
                            "best_test_total_loss": best,
                            "beta": beta,
                        }
                    )

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
                    model.load_state_dict(
                        torch.load(
                            f"{output_dir}/best_model.pt", map_location=DEVICE
                        )
                    )

                    # compute elbo metrics
                    test_metrics = compute_test_metrics(
                        model, test_loader, DEVICE, n_iwae_samples=10
                    )
                    print(
                        f"test metrics - LL: {test_metrics['ll']:.2f}, "
                        f"L[q]: {test_metrics['entropy']:.2f}, "
                        f"RE: {test_metrics['recon']:.2f}, "
                        f"KL: {test_metrics['kl']:.2f}"
                    )

                    eval_start_time = time.time()
                    images = {}

                    # reconstructions
                    t0 = time.time()
                    print(f"generating reconstructions...")
                    recon_path = save_reconstructions(
                        model,
                        test_loader,
                        DEVICE,
                        f"{output_dir}/reconstructions.png",
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    # latent distribution histograms
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

                    # fourier coefficient visualization
                    t0 = time.time()
                    print(f"generating fourier coefficient visualization...")
                    fourier_viz_path = plot_fourier_coefficients(
                        model,
                        test_loader,
                        DEVICE,
                        f"{output_dir}/fourier_coefficients.png",
                        n_samples=5,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    # latent interpolation visualization
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

                    # self-binding test
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

                    # cross-class bind/unbind test
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

                    # phase-attribute correlation analysis
                    t0 = time.time()
                    print(f"running phase-attribute correlation analysis...")
                    corr_results = plot_phase_attribute_correlation(
                        model, test_loader, DEVICE, output_dir, attr_names,
                        n_samples=5000, top_k=20,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

                    # attribute traversal visualization
                    t0 = time.time()
                    print(f"generating attribute traversal visualization...")
                    traversal_path = plot_attribute_traversal(
                        model, test_loader, DEVICE, output_dir, attr_names,
                        corr_results["corr_data"], n_steps=9, n_samples_per_attr=3,
                    )
                    print(f"  completed in {time.time() - t0:.2f}s")

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
                            "fourier_coefficients": fourier_viz_path,
                            "latent_interpolation": interp_path,
                        }
                    )

                    if corr_results.get("heatmap_path"):
                        images["phase_attribute_correlation"] = corr_results["heatmap_path"]
                    if corr_results.get("bar_path"):
                        images["top_attribute_correlations"] = corr_results["bar_path"]
                    if traversal_path:
                        images["attribute_traversal"] = traversal_path

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
                            "cross_class_bind_unbind_similarity_star": cross_class_star.get(
                                "cross_class_bind_unbind_similarity", 0.0
                            ),
                        }
                    )

                    summary = {
                        "final_best_total_loss": best,
                        "test/ll": test_metrics["ll"],
                        "test/entropy": test_metrics["entropy"],
                        "test/recon": test_metrics["recon"],
                        "test/kl": test_metrics["kl"],
                        **fourier_metrics,
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

                logger.finish_run()

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
    p.add_argument("--epochs", type=int, default=50, help="training epochs")
    p.add_argument("--warmup_epochs", type=int, default=10, help="kl warmup epochs")
    p.add_argument("--batch_size", type=int, default=128, help="batch size")
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
    p.add_argument("--max_beta", type=float, default=1.0, help="max kl beta")
    p.add_argument(
        "--min_beta", type=float, default=0.1, help="min kl beta during cycles"
    )
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    p.add_argument(
        "--wandb_project",
        type=str,
        default="clifford-experiments-CelebA",
        help="wandb project name",
    )
    p.add_argument("--patience", type=int, default=20, help="early stopping patience")
    p.add_argument(
        "--cycle_epochs",
        type=int,
        default=0,
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

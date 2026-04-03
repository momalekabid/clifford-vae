# dSprites training — learn HRR-compatible latent codes for compositional factor manipulation
# dataset: 64x64 binary sprites with 6 ground truth factors
# (color, shape, scale, orientation, posX, posY)

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
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

from cnn.cliffordar_model import CliffordARVAE, HybridVAE
from cnn.models import VAE as CNNVAE
from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_pairwise_bind_bundle_decode,
    compute_class_means,
    evaluate_mean_vector_cosine,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    test_per_class_bundle_capacity_k_items,
    bind as vsa_bind,
    unbind as vsa_unbind,
    normalize_vectors as vsa_normalize,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# dSprites factor names and metadata
FACTOR_NAMES = ["color", "shape", "scale", "orientation", "posX", "posY"]
SHAPE_NAMES = ["square", "ellipse", "heart"]


class DSpritesDataset(Dataset):
    """dSprites dataset loader from npz file."""

    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path, allow_pickle=True, encoding="latin1")
        self.imgs = data["imgs"]  # (737280, 64, 64) uint8
        self.latents_values = data["latents_values"]  # (737280, 6) float64
        self.latents_classes = data["latents_classes"]  # (737280, 6) int64
        self.transform = transform

        # factor sizes: color=1, shape=3, scale=6, orientation=40, posX=32, posY=32
        self.metadata = data["metadata"][()]
        self.n_samples = len(self.imgs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)  # (64, 64) in {0, 1}
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)
        if self.transform:
            img = self.transform(img)

        # use shape as the "class" label for knn (3 classes)
        shape_label = int(self.latents_classes[idx, 1])
        return img, shape_label


def get_factor_subsets(dataset, factor_idx, n_per_value=100):
    """get subsets of images grouped by a specific factor value.
    useful for factor-swapping experiments.
    """
    classes = dataset.latents_classes[:, factor_idx]
    unique_vals = np.unique(classes)
    subsets = {}
    for v in unique_vals:
        indices = np.where(classes == v)[0]
        if len(indices) > n_per_value:
            indices = np.random.choice(indices, n_per_value, replace=False)
        subsets[int(v)] = indices
    return subsets


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0, "entropy": 0.0}
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
        eb = losses["effective_beta"]
        effective_beta_vals.append(eb if isinstance(eb, float) else eb.item())

    n = len(loader.dataset)
    result = {f"train/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["train/entropy"] = sums["entropy"] / n
    if effective_beta_vals:
        result["train/effective_beta"] = np.mean(effective_beta_vals)
    return result


def test_epoch(model, loader, device):
    model.eval()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0, "entropy": 0.0}
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

    n = len(loader.dataset)
    result = {f"test/{k}_loss": v / n for k, v in sums.items() if k != "entropy"}
    result["test/entropy"] = sums["entropy"] / n
    if effective_beta_vals:
        result["test/effective_beta"] = np.mean(effective_beta_vals)
    return result


def save_reconstructions(model, loader, device, path, n_images=10):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n_images].to(device)
    with torch.no_grad():
        x_recon, _, _, _ = model(x)
    grid = torch.cat([x.cpu(), x_recon.cpu()], dim=0).clamp(0, 1)
    tu.save_image(grid, path, nrow=n_images)
    return path


def get_latents(model, loader, device, n_samples=1000):
    """extract flattened latent vectors from model."""
    model.eval()
    latents, labels_list, images_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.get_flat_latent(x)
            latents.append(z.detach())
            labels_list.append(y)
            images_list.append(x.cpu())
            if len(torch.cat(latents, 0)) >= n_samples:
                break

    item_memory = torch.cat(latents, 0)[:n_samples]
    item_labels = torch.cat(labels_list, 0)[:n_samples].to(device)
    item_images = torch.cat(images_list, 0)[:n_samples]
    return item_memory, item_labels, item_images


def get_latents_with_factors(model, dataset, device, indices):
    """encode specific indices and return (z, factor_values, factor_classes, images)."""
    model.eval()
    imgs = []
    for i in indices:
        img, _ = dataset[i]
        imgs.append(img)
    x = torch.stack(imgs).to(device)

    with torch.no_grad():
        z = model.get_flat_latent(x)

    factor_values = dataset.latents_values[indices]
    factor_classes = dataset.latents_classes[indices]
    return z, factor_values, factor_classes, x.cpu()


def beta_vae_disentanglement_metric(model, dataset, device, n_trials=5000, batch_size=64):
    """beta-VAE disentanglement metric (Higgins et al. 2017).
    for each trial:
      1. pick a fixed factor k
      2. sample two datapoints that share the same value for factor k
         but differ in all other factors
      3. encode both, compute |z1 - z2|
      4. record (argmax |z1 - z2|, k)
    then train a majority-vote classifier on the argmax -> factor mapping.
    returns accuracy (higher = more disentangled).
    """
    from collections import Counter

    model.eval()
    all_classes = dataset.latents_classes  # (N, 6)
    n_factors = all_classes.shape[1]

    # precompute indices per factor value
    factor_indices = {}
    for f in range(1, n_factors):  # skip color (always 1)
        vals = np.unique(all_classes[:, f])
        factor_indices[f] = {int(v): np.where(all_classes[:, f] == v)[0] for v in vals}

    votes = []  # list of (argmax_dim, true_factor)

    with torch.no_grad():
        for _ in range(n_trials):
            # pick a random factor to hold fixed
            k = np.random.randint(1, n_factors)
            vals = list(factor_indices[k].keys())
            v = vals[np.random.randint(len(vals))]
            pool = factor_indices[k][v]

            if len(pool) < 2:
                continue

            # sample two indices with same factor k value
            pair = np.random.choice(pool, 2, replace=False)
            img1, _ = dataset[pair[0]]
            img2, _ = dataset[pair[1]]
            x = torch.stack([img1, img2]).to(device)
            z = model.get_flat_latent(x)
            diff = torch.abs(z[0] - z[1])
            top_dim = diff.argmax().item()
            votes.append((top_dim, k))

    if not votes:
        return {"disentanglement_accuracy": 0.0}

    # majority vote classifier: for each latent dim, what factor does it vote for?
    dim_to_factor_counts = {}
    for dim, factor in votes:
        if dim not in dim_to_factor_counts:
            dim_to_factor_counts[dim] = Counter()
        dim_to_factor_counts[dim][factor] += 1

    # assign each dim to its majority factor
    dim_to_factor = {}
    for dim, counts in dim_to_factor_counts.items():
        dim_to_factor[dim] = counts.most_common(1)[0][0]

    # compute accuracy
    correct = sum(1 for dim, factor in votes if dim_to_factor.get(dim) == factor)
    acc = correct / len(votes)

    print(f"  disentanglement metric: {acc:.4f} ({len(votes)} trials)")
    return {"disentanglement_accuracy": acc}


def plot_latent_traversals(model, dataset, device, save_path, n_latents=10, n_steps=8):
    """traverse individual latent dimensions and decode, beta-VAE style.
    picks a reference image, then sweeps each latent dim across [-3, 3] std.
    """
    model.eval()
    # pick a reference image
    ref_idx = np.random.randint(len(dataset))
    ref_img, _ = dataset[ref_idx]
    ref_x = ref_img.unsqueeze(0).to(device)

    with torch.no_grad():
        z_ref = model.get_flat_latent(ref_x)  # (1, d)

    d = z_ref.shape[-1]
    n_show = min(n_latents, d)

    # find the latent dims with highest variance across a sample
    with torch.no_grad():
        sample_zs = []
        for i in range(0, min(1000, len(dataset)), 1):
            img, _ = dataset[i]
            z = model.get_flat_latent(img.unsqueeze(0).to(device))
            sample_zs.append(z.cpu())
        sample_zs = torch.cat(sample_zs, 0)
        var = sample_zs.var(dim=0)
        top_dims = var.argsort(descending=True)[:n_show].numpy()

    # sweep values
    sweep = torch.linspace(-3, 3, n_steps)

    fig, axes = plt.subplots(n_show, n_steps + 1, figsize=(2 * (n_steps + 1), 2 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, dim in enumerate(top_dims):
        # show reference
        axes[row, 0].imshow(ref_img.squeeze(), cmap="gray")
        axes[row, 0].set_title(f"ref", fontsize=8)
        axes[row, 0].axis("off")

        for col, val in enumerate(sweep):
            z_mod = z_ref.clone()
            z_mod[0, dim] = val.item()
            with torch.no_grad():
                recon = model.decoder(z_mod.to(device)).cpu()
            axes[row, col + 1].imshow(recon.squeeze().clamp(0, 1), cmap="gray")
            axes[row, col + 1].set_title(f"z[{dim}]={val:.1f}", fontsize=7)
            axes[row, col + 1].axis("off")

        axes[row, 0].set_ylabel(f"dim {dim}\nvar={var[dim]:.3f}", fontsize=8)

    plt.suptitle("Latent Traversals (top variance dims)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved latent traversals to {save_path}")
    return save_path


def factor_swap_experiment(model, dataset, device, save_dir, n_pairs=5):
    """the big one: encode two sprites, unbind a factor, rebind with the other's factor, decode.

    for each pair of sprites that differ in exactly one factor:
    1. encode both -> z_a, z_b
    2. bind z_a with a role vector for the differing factor
    3. unbind the factor, rebind with z_b's factor value
    4. decode and compare

    this tests whether clifford binding can manipulate individual factors in pixel space.
    """
    os.makedirs(save_dir, exist_ok=True)

    # for each non-trivial factor (shape, scale, orientation, posX, posY)
    for factor_idx in range(1, 6):
        factor_name = FACTOR_NAMES[factor_idx]
        factor_classes = dataset.latents_classes[:, factor_idx]
        unique_vals = np.unique(factor_classes)

        if len(unique_vals) < 2:
            continue

        print(f"  factor swap: {factor_name} ({len(unique_vals)} values)")

        fig, axes = plt.subplots(n_pairs, 5, figsize=(12, 2.5 * n_pairs))
        if n_pairs == 1:
            axes = axes[np.newaxis, :]

        for pair_idx in range(n_pairs):
            # pick two different factor values
            v1, v2 = np.random.choice(unique_vals, 2, replace=False)

            # find sprites that match on all OTHER factors but differ on this one
            mask_v1 = factor_classes == v1
            mask_v2 = factor_classes == v2
            idx_v1 = np.where(mask_v1)[0]
            idx_v2 = np.where(mask_v2)[0]

            # pick a random sprite from v1
            src_idx = np.random.choice(idx_v1)
            src_factors = dataset.latents_classes[src_idx].copy()

            # find a sprite in v2 that matches on all other factors
            target_mask = mask_v2.copy()
            for f in range(6):
                if f == factor_idx:
                    continue
                target_mask &= (dataset.latents_classes[:, f] == src_factors[f])
            target_indices = np.where(target_mask)[0]

            if len(target_indices) == 0:
                # fallback: just use any sprite with v2
                target_indices = idx_v2

            tgt_idx = np.random.choice(target_indices)

            # encode both
            with torch.no_grad():
                src_img, _ = dataset[src_idx]
                tgt_img, _ = dataset[tgt_idx]
                src_x = src_img.unsqueeze(0).to(device)
                tgt_x = tgt_img.unsqueeze(0).to(device)

                z_src = model.get_flat_latent(src_x)  # (1, flat_dim)
                z_tgt = model.get_flat_latent(tgt_x)

                # bind src with tgt, then unbind src -> should recover tgt-like
                bound = vsa_bind(z_src.cpu(), z_tgt.cpu())
                recovered_src = vsa_unbind(bound, z_tgt.cpu(), method="*")
                recovered_tgt = vsa_unbind(bound, z_src.cpu(), method="*")

                # decode
                src_recon = model.decoder(z_src).cpu()
                tgt_recon = model.decoder(z_tgt).cpu()
                # decode the recovered vectors
                recovered_src_recon = model.decoder(recovered_src.to(device)).cpu()

            # compute similarities
            sim_src = F.cosine_similarity(z_src.cpu(), recovered_src, dim=-1).item()

            # plot: [source, target, bound_decoded, recovered_src, recovered_tgt]
            for ax in axes[pair_idx]:
                ax.axis("off")

            axes[pair_idx, 0].imshow(src_img.squeeze(), cmap="gray")
            axes[pair_idx, 0].set_title(f"src ({factor_name}={v1})")
            axes[pair_idx, 1].imshow(tgt_img.squeeze(), cmap="gray")
            axes[pair_idx, 1].set_title(f"tgt ({factor_name}={v2})")
            axes[pair_idx, 2].imshow(src_recon.squeeze().clamp(0, 1), cmap="gray")
            axes[pair_idx, 2].set_title("src recon")
            axes[pair_idx, 3].imshow(tgt_recon.squeeze().clamp(0, 1), cmap="gray")
            axes[pair_idx, 3].set_title("tgt recon")
            axes[pair_idx, 4].imshow(recovered_src_recon.squeeze().clamp(0, 1), cmap="gray")
            axes[pair_idx, 4].set_title(f"unbind(bind(s,t),t)\nsim={sim_src:.3f}")

        plt.suptitle(f"factor swap: {factor_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"factor_swap_{factor_name}.png"), dpi=200)
        plt.close()
        print(f"    saved factor_swap_{factor_name}.png")


def bundle_by_factor(model, dataset, device, save_dir, factor_idx=1, n_per_class=10):
    """bundle sprites by a factor (e.g. shape) and decode the prototype.
    shows whether bundled representations capture the factor's essence.
    """
    os.makedirs(save_dir, exist_ok=True)
    factor_name = FACTOR_NAMES[factor_idx]
    classes = dataset.latents_classes[:, factor_idx]
    unique_vals = np.unique(classes)

    fig, axes = plt.subplots(len(unique_vals), n_per_class + 1, figsize=(2 * (n_per_class + 1), 2 * len(unique_vals)))
    if len(unique_vals) == 1:
        axes = axes[np.newaxis, :]

    for row, val in enumerate(unique_vals):
        indices = np.where(classes == val)[0]
        chosen = np.random.choice(indices, min(n_per_class, len(indices)), replace=False)

        # encode and bundle
        with torch.no_grad():
            imgs = []
            for i in chosen:
                img, _ = dataset[i]
                imgs.append(img)
            x = torch.stack(imgs).to(device)
            z = model.get_flat_latent(x)  # (n, flat_dim)

            # bundle via addition
            bundled = z.mean(dim=0, keepdim=True)  # (1, flat_dim)
            bundled_recon = model.decoder(bundled).cpu()

        # plot individual sprites
        for col, i in enumerate(chosen):
            img, _ = dataset[i]
            axes[row, col].imshow(img.squeeze(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                label = SHAPE_NAMES[val] if factor_idx == 1 and val < len(SHAPE_NAMES) else f"{factor_name}={val}"
                axes[row, col].set_ylabel(label, fontsize=10)

        # plot bundled prototype
        axes[row, -1].imshow(bundled_recon.squeeze().clamp(0, 1), cmap="gray")
        axes[row, -1].set_title("bundle")
        axes[row, -1].axis("off")

    plt.suptitle(f"bundled prototypes by {factor_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bundle_by_{factor_name}.png"), dpi=200)
    plt.close()
    print(f"  saved bundle_by_{factor_name}.png")


def perform_knn_evaluation(model, train_loader, test_loader, device, sample_sizes):
    model.eval()
    all_train_z, all_train_y = [], []
    all_test_z, all_test_y = [], []

    with torch.no_grad():
        for x, y in train_loader:
            z = model.get_flat_latent(x.to(device))
            all_train_z.append(z.cpu())
            all_train_y.append(y)
        for x, y in test_loader:
            z = model.get_flat_latent(x.to(device))
            all_test_z.append(z.cpu())
            all_test_y.append(y)

    X_train = torch.cat(all_train_z, 0).numpy()
    y_train = torch.cat(all_train_y, 0).numpy()
    X_test = torch.cat(all_test_z, 0).numpy()
    y_test = torch.cat(all_test_y, 0).numpy()

    results = {}
    for n in sample_sizes:
        if n > len(X_train):
            n = len(X_train)
        idx = np.random.choice(len(X_train), n, replace=False)
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train[idx], y_train[idx])
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[f"knn_acc_{n}"] = acc
        results[f"knn_f1_{n}"] = f1
        print(f"  knn@{n}: acc={acc:.4f}, f1={f1:.4f}")
    return results


def plot_tsne(model, loader, device, save_path, n_samples=2000):
    """run t-SNE on latent codes and scatter plot colored by shape label."""
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model.get_flat_latent(x.to(device))
            zs.append(z.cpu())
            ys.append(y)
            if sum(len(v) for v in zs) >= n_samples:
                break
    Z = torch.cat(zs, 0)[:n_samples].numpy()
    Y = torch.cat(ys, 0)[:n_samples].numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(8, 8))
    for cls_idx, name in enumerate(SHAPE_NAMES):
        mask = Y == cls_idx
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.5, label=name)
    ax.legend()
    ax.set_title("t-SNE of latent codes (colored by shape)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  saved t-SNE plot to {save_path}")
    return save_path


def main(args):
    print(f"Device: {DEVICE}")

    # load dataset
    npz_path = args.data_path
    if not os.path.exists(npz_path):
        print(f"downloading dSprites dataset...")
        os.makedirs(os.path.dirname(npz_path) or "data", exist_ok=True)
        import urllib.request
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        urllib.request.urlretrieve(url, npz_path)
        print(f"saved to {npz_path}")

    full_dataset = DSpritesDataset(npz_path)
    print(f"loaded {len(full_dataset)} sprites")

    # train/test split (90/10)
    n_total = len(full_dataset)
    n_test = int(0.1 * n_total)
    n_train = n_total - n_test
    train_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    distributions = args.distributions if args.distributions else ["clifford", "gaussian"]
    latent_dims = args.latent_dims

    logger = WandbLogger(args)
    all_run_metrics = []

    for latent_dim in latent_dims:
        for dist_name in distributions:
            exp_name = f"dsprites-{dist_name}-d{latent_dim}-{args.recon_loss}"
            output_dir = f"results/{exp_name}"
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n== {exp_name} ==")
            logger.start_run(exp_name, args)

            if args.arch == "vit":
                model_latent_dim = max(4, latent_dim // 64)
                print(f"  {dist_name}: 64 tokens x {model_latent_dim}d = {64 * model_latent_dim}d total (CNN+ViT)")
                model = CliffordARVAE(
                    latent_dim=model_latent_dim,
                    image_size=64,
                    in_channels=1,
                    distribution=dist_name,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    use_learnable_beta=args.use_learnable_beta,
                    l2_normalize=(dist_name == "gaussian" and args.l2_norm),
                )
            elif args.arch == "hybrid":
                model_latent_dim = max(4, latent_dim // 16)
                print(f"  {dist_name}: per-token CNN, d={model_latent_dim} per token (hybrid)")
                model = HybridVAE(
                    latent_dim=model_latent_dim,
                    in_channels=1,
                    distribution=dist_name,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    use_learnable_beta=args.use_learnable_beta,
                    l2_normalize=(dist_name == "gaussian" and args.l2_norm),
                    img_size=64,
                )
            else:
                print(f"  {dist_name}: flat z, d={latent_dim} (CNN w/ residual)")
                model = CNNVAE(
                    latent_dim=latent_dim,
                    in_channels=1,
                    distribution=dist_name,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    use_learnable_beta=args.use_learnable_beta,
                    l2_normalize=(dist_name == "gaussian" and args.l2_norm),
                    img_size=64,
                )

            logger.watch_model(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)

            best = float("inf")
            patience_counter = 0
            train_start_time = time.time()

            def kl_beta_for_epoch(e):
                if e < args.warmup_epochs:
                    return min(1.0, (e + 1) / max(1, args.warmup_epochs)) * args.max_beta
                if args.cycle_epochs <= 0:
                    return args.max_beta
                cycle_pos = (e - args.warmup_epochs) % args.cycle_epochs
                half = max(1, args.cycle_epochs // 2)
                if cycle_pos <= half:
                    t = cycle_pos / half
                else:
                    t = (args.cycle_epochs - cycle_pos) / max(1, args.cycle_epochs - half)
                return args.min_beta + (args.max_beta - args.min_beta) * t

            for epoch in range(args.epochs):
                beta = 1.0 if args.use_learnable_beta else kl_beta_for_epoch(epoch)
                train_losses = train_epoch(model, train_loader, optimizer, DEVICE, beta)
                test_losses = test_epoch(model, test_loader, DEVICE)

                val = test_losses["test/recon_loss"] + test_losses["test/kld_loss"]
                if np.isfinite(val) and val < best:
                    best = val
                    torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.log_metrics({
                    "epoch": epoch,
                    "beta": beta,
                    **train_losses,
                    **test_losses,
                })

                if epoch % 20 == 0 or epoch == args.epochs - 1:
                    print(f"  epoch {epoch}: train_loss={train_losses['train/total_loss']:.2f} "
                          f"test_loss={test_losses['test/total_loss']:.2f} beta={beta:.3f}")

                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

            train_time = time.time() - train_start_time
            print(f"best total loss (recon+kld): {best:.4f}, training time: {train_time:.2f}s")

            # load best model
            ckpt_path = f"{output_dir}/best_model.pt"
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

            model.eval()
            eval_start_time = time.time()

            # save reconstructions
            recon_path = save_reconstructions(model, test_loader, DEVICE, f"{output_dir}/reconstructions.png")

            # knn on shape labels (3 classes)
            print("running knn evaluation (shape classification)...")
            knn_results = perform_knn_evaluation(
                model, train_loader, test_loader, DEVICE,
                sample_sizes=[100, 600, 1000],
            )

            # factor swap experiment — the main event
            print("running factor swap experiments...")
            factor_swap_experiment(model, full_dataset, DEVICE, f"{output_dir}/factor_swaps", n_pairs=5)

            # bundle by shape
            print("running bundle-by-shape experiment...")
            bundle_by_factor(model, full_dataset, DEVICE, output_dir, factor_idx=1, n_per_class=8)

            # VSA tests
            print("running VSA tests...")
            item_memory, item_labels, item_images = get_latents(model, test_loader, DEVICE, n_samples=1000)

            # bundle capacity
            vsa_bundle_capacity(
                d=item_memory.shape[-1],
                n_items=1000,
                k_range=list(range(5, 51, 5)),
                n_trials=10,
                normalize=True,
                device=DEVICE,
                plot=True,
                save_dir=output_dir,
                item_memory=item_memory,
            )

            # per-class bundle similarity (by shape)
            test_per_class_bundle_capacity_k_items(
                d=item_memory.shape[-1],
                n_items=1000,
                n_classes=3,
                items_per_class=1,
                n_trials=1,
                normalize=True,
                device=DEVICE,
                plot=True,
                save_dir=output_dir,
                item_memory=item_memory,
                labels=item_labels,
                item_images=item_images,
                class_names=SHAPE_NAMES,
            )

            # role-filler
            print("running role-filler test...")
            vsa_binding_unbinding(
                d=item_memory.shape[-1],
                n_items=1000,
                k_range=list(range(2, 21, 2)),
                n_trials=10,
                normalize=True,
                device=DEVICE,
                plot=True,
                unbind_method="*",
                save_dir=output_dir,
                item_memory=item_memory,
                bind_with_random=True,
            )

            # pairwise bind-bundle-decode test
            print("running pairwise bind-bundle-decode test...")
            pairwise_result = test_pairwise_bind_bundle_decode(
                model,
                test_loader,
                DEVICE,
                output_dir,
                class_names=SHAPE_NAMES,
                img_shape=(1, 64, 64),
                n_classes=3,
            )
            pairwise_bind_bundle_path = pairwise_result.get("pairwise_bind_bundle_path")

            # self-binding: bind z with itself k times, unbind k times, measure similarity
            deconv_dir = f"{output_dir}/deconv"
            os.makedirs(deconv_dir, exist_ok=True)
            print("running self-binding test * ...")
            fourier_star = test_self_binding(
                model,
                test_loader,
                DEVICE,
                output_dir,
                unbind_method="*",
                img_shape=(1, 64, 64),
            )
            print("running self-binding test † ...")
            fourier_deconv = test_self_binding(
                model,
                test_loader,
                DEVICE,
                deconv_dir,
                unbind_method="†",
                img_shape=(1, 64, 64),
            )

            # mean vector cosine classification
            print("running mean vector cosine eval...")
            from torch.utils.data import Subset
            # use a subset of train for computing class means
            train_subset_size = min(10000, len(train_dataset))
            train_subset = Subset(train_dataset, list(range(train_subset_size)))
            train_subset_loader = DataLoader(
                train_subset, batch_size=args.batch_size, shuffle=False
            )
            class_means = compute_class_means(
                model, train_subset_loader, DEVICE, max_per_class=1000
            )
            mean_vector_acc, _ = evaluate_mean_vector_cosine(
                model, test_loader, DEVICE, class_means
            )
            print(f"  mean_vector_cosine_acc: {mean_vector_acc}")

            # beta-VAE disentanglement metric
            print("running beta-VAE disentanglement metric...")
            disent_results = beta_vae_disentanglement_metric(
                model, full_dataset, DEVICE, n_trials=5000,
            )

            # latent traversals
            print("running latent traversals...")
            traversal_path = plot_latent_traversals(
                model, full_dataset, DEVICE,
                f"{output_dir}/latent_traversals.png",
                n_latents=10, n_steps=8,
            )

            all_run_metrics.append({
                "d": latent_dim, "dist": dist_name,
                "knn_acc_100": knn_results.get("knn_acc_100", 0.0) * 100,
                "knn_acc_600": knn_results.get("knn_acc_600", 0.0) * 100,
                "knn_acc_1000": knn_results.get("knn_acc_1000", 0.0) * 100,
                "knn_f1_100": knn_results.get("knn_f1_100", 0.0) * 100,
                "knn_f1_600": knn_results.get("knn_f1_600", 0.0) * 100,
                "knn_f1_1000": knn_results.get("knn_f1_1000", 0.0) * 100,
                "mvc": float(mean_vector_acc) * 100,
                "disentanglement": disent_results.get("disentanglement_accuracy", 0.0) * 100,
                "best_loss": best,
            })

            # t-SNE visualization
            print("running t-SNE visualization...")
            tsne_path = plot_tsne(model, test_loader, DEVICE, f"{output_dir}/tsne.png")

            eval_time = time.time() - eval_start_time

            # collect fourier/self-binding metrics
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

            # log everything
            logger.log_metrics({
                **knn_results,
                **fourier_metrics,
                **disent_results,
                "best_test_total_loss": best,
                "mean_vector_cosine_acc": float(mean_vector_acc),
                "approx_inv_depth_star": next((d for d, s in zip(fourier_star.get("k_values", []), fourier_star.get("k_sims", [])) if s < 0.5), fourier_star.get("k_values", [0])[-1] if fourier_star.get("k_values") else 0),
                "approx_inv_depth_deconv": next((d for d, s in zip(fourier_deconv.get("k_values", []), fourier_deconv.get("k_sims", [])) if s < 0.5), fourier_deconv.get("k_values", [0])[-1] if fourier_deconv.get("k_values") else 0),
            })
            logger.log_summary({
                **knn_results,
                **fourier_metrics,
                **disent_results,
                "best_test_total_loss": best,
                "final_best_total_loss": best,
                "mean_vector_cosine_acc": float(mean_vector_acc),
            })

            # log images
            images_to_log = {}
            if os.path.exists(recon_path):
                images_to_log["reconstructions"] = recon_path
            for fname in os.listdir(f"{output_dir}/factor_swaps"):
                if fname.endswith(".png"):
                    images_to_log[fname.replace(".png", "")] = os.path.join(output_dir, "factor_swaps", fname)
            bundle_path = os.path.join(output_dir, f"bundle_by_shape.png")
            if os.path.exists(bundle_path):
                images_to_log["bundle_by_shape"] = bundle_path
            if pairwise_bind_bundle_path and os.path.exists(pairwise_bind_bundle_path):
                images_to_log["pairwise_bind_bundle"] = pairwise_bind_bundle_path
            if os.path.exists(tsne_path):
                images_to_log["tsne"] = tsne_path
            if os.path.exists(traversal_path):
                images_to_log["latent_traversals"] = traversal_path
            if images_to_log:
                logger.log_images(images_to_log)

            logger.finish_run()

            print(f"eval time: {eval_time:.2f}s, total: {train_time + eval_time:.2f}s")

    if all_run_metrics:
        try:
            import pandas as pd
            df = pd.DataFrame(all_run_metrics)
            df.to_csv("dsprites_results.csv", index=False)
            print(f"\n{'='*25} dsprites results {'='*25}")
            print(df.to_string(index=False))
            print("saved to dsprites_results.csv")
        except ImportError:
            with open("dsprites_results.json", "w") as f:
                json.dump(all_run_metrics, f, indent=2)

    print("\ndone!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="dSprites VAE training with HRR compositional evaluation")

    p.add_argument("--data_path", type=str, default="data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--warmup_epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--recon_loss", type=str, default="l1", choices=["mse", "l1", "bce"])
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--max_beta", type=float, default=2.0)
    p.add_argument("--min_beta", type=float, default=0.1)
    p.add_argument("--use_learnable_beta", action="store_true")
    p.add_argument("--l2_norm", action="store_true", default=True)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="dsprites-clifford")
    p.add_argument("--patience", type=int, default=75)
    p.add_argument("--cycle_epochs", type=int, default=200)
    p.add_argument("--latent_dims", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--distributions", type=str, nargs="+", default=None)
    p.add_argument("--arch", type=str, default="cnn", choices=["cnn", "vit", "hybrid"],
                   help="backbone: cnn (flat latent w/ residual) or vit (hybrid cnn+vit, per-token)")

    args = p.parse_args()
    main(args)

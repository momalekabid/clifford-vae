"""generate compact bind/bundle/unbind comparison figure for the paper.
loads two checkpoints (clifford + gaussian_l2) and shows selected class pairs side by side.
usage:
  python -m scripts.paper_bind_bundle_figure \
    --clifford_ckpt path/to/clifford.pt \
    --gaussian_ckpt path/to/gaussian_l2.pt \
    --dataset fashionmnist --latent_dim 4096 --arch hybrid \
    --output figures/bind_bundle_comparison.png
"""

import argparse
import math
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vsa import bind, unbind


FASHIONMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# hand-picked pairs: (class_a, class_b, reason)
SELECTED_PAIRS = [
    (0, 6, "similar"),     # t-shirt + shirt
    (5, 7, "similar"),     # sandal + sneaker
    (3, 4, "similar"),     # dress + coat
    (1, 8, "different"),   # trouser + bag
    (3, 7, "different"),   # dress + sneaker
    (2, 9, "different"),   # pullover + ankle boot
]


def load_model(ckpt_path, distribution, latent_dim, arch, dataset, device):
    """load a model from checkpoint."""
    if arch == "hybrid":
        from cnn.cliffordar_model import HybridVAE
        img_size = 32 if dataset == "cifar10" else 32
        in_ch = 3 if dataset == "cifar10" else 1
        if dataset == "fashionmnist":
            img_size = 32
            in_ch = 1
        model = HybridVAE(
            latent_dim=latent_dim,
            in_channels=in_ch,
            distribution=distribution,
            img_size=img_size,
        )
    elif arch == "cnn":
        from cnn.models import VAE as CNNVAE
        in_ch = 3 if dataset == "cifar10" else 1
        model = CNNVAE(
            latent_dim=latent_dim,
            in_channels=in_ch,
            distribution=distribution,
            device=str(device),
        )
    else:
        raise ValueError(f"unsupported arch: {arch}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_flat_z(model, x):
    """get flat latent from model."""
    if hasattr(model, 'get_flat_latent'):
        return model.get_flat_latent(x)
    elif hasattr(model, 'reparameterize') and hasattr(model, 'encoder'):
        mu, params = model.encoder(x)
        z, _, _ = model.reparameterize(mu, params)
        if z.dim() == 3:
            z = z.reshape(z.size(0), -1)
        return z
    elif hasattr(model, 'encode'):
        x_in = x if x.dim() == 2 else x.view(x.size(0), -1)
        z, _ = model.encode(x_in)
        return z
    raise RuntimeError("can't extract latent from model")


def decode_z(model, z, img_shape):
    """decode latent back to image."""
    with torch.no_grad():
        if hasattr(model, 'get_flat_latent'):
            # per-token model: need to reshape z back to tokens
            if hasattr(model, 'num_tokens'):
                nt = model.num_tokens
                per_tok = z.shape[-1] // nt
                z_tok = z.view(z.size(0), nt, per_tok)
                imgs = model.decoder(z_tok)
            else:
                imgs = model.decoder(z)
        else:
            imgs = model.decoder(z)

        # handle activation
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'output_activation'):
            if model.decoder.output_activation == "sigmoid":
                imgs = torch.sigmoid(imgs)
            elif model.decoder.output_activation == "tanh":
                imgs = imgs * 0.5 + 0.5
        else:
            imgs = torch.sigmoid(imgs)

        if imgs.dim() == 2:
            imgs = imgs.view(-1, *img_shape)
        return imgs.clamp(0, 1).cpu()


def collect_class_samples(loader, model, device, n_classes=10):
    """collect one z and image per class."""
    class_z = {}
    class_img = {}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = get_flat_z(model, x)
            for i in range(len(y)):
                label = y[i].item()
                if label not in class_z and len(class_z) < n_classes:
                    class_z[label] = z[i:i+1]
                    class_img[label] = x[i:i+1]
            if len(class_z) >= n_classes:
                break
    return class_z, class_img


def make_panel(model, class_z, class_img, pairs, img_shape, class_names):
    """generate rows of [A, B, bind(A,B), bundle(A,B), unbind->A, unbind->B]."""
    rows = []
    pair_labels = []
    sims = []

    for la, lb, _ in pairs:
        za, zb = class_z[la], class_z[lb]

        z_bind = bind(za, zb)
        z_bundle = (za + zb) / math.sqrt(2)
        recovered_a = unbind(z_bind, zb)
        recovered_b = unbind(z_bind, za)

        sim_a = F.cosine_similarity(recovered_a, za, dim=-1).mean().item()
        sim_b = F.cosine_similarity(recovered_b, zb, dim=-1).mean().item()
        sims.append((sim_a + sim_b) / 2)

        all_z = torch.cat([za, zb, z_bind, z_bundle, recovered_a, recovered_b], dim=0)
        imgs = decode_z(model, all_z, img_shape)
        # also prepend original images
        orig_a = class_img[la].cpu()
        orig_b = class_img[lb].cpu()
        # use decoded originals for consistency (from z, not raw pixels)
        rows.append(imgs)
        pair_labels.append(f"{class_names[la]}+{class_names[lb]}")

    return rows, pair_labels, sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clifford_ckpt", required=True)
    parser.add_argument("--gaussian_ckpt", required=True)
    parser.add_argument("--dataset", default="fashionmnist", choices=["fashionmnist", "cifar10"])
    parser.add_argument("--latent_dim", type=int, default=4096)
    parser.add_argument("--arch", default="cnn", choices=["cnn", "hybrid"])
    parser.add_argument("--output", default="figures/bind_bundle_comparison.png")
    parser.add_argument("--pairs", type=str, default=None,
                        help="comma-separated pairs like '0-6,5-7,1-8' to override defaults")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # load data
    if args.dataset == "fashionmnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
        class_names = FASHIONMNIST_CLASSES
        img_shape = (1, 32, 32)
        in_ch = 1
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_ds = datasets.CIFAR10("data", train=False, download=True, transform=transform)
        class_names = ["plane", "auto", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
        img_shape = (3, 32, 32)
        in_ch = 3

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # parse pairs
    if args.pairs:
        pairs = []
        for p in args.pairs.split(","):
            a, b = p.split("-")
            pairs.append((int(a), int(b), "custom"))
    else:
        pairs = SELECTED_PAIRS

    # load models
    print("loading clifford model...")
    cliff_model = load_model(args.clifford_ckpt, "clifford", args.latent_dim, args.arch, args.dataset, device)
    print("loading gaussian (l2) model...")
    gauss_model = load_model(args.gaussian_ckpt, "gaussian", args.latent_dim, args.arch, args.dataset, device)
    # enable l2 norm on gaussian
    if hasattr(gauss_model, 'l2_normalize'):
        gauss_model.l2_normalize = True

    # collect samples (use same loader for both — different z but same input images)
    cliff_z, cliff_img = collect_class_samples(loader, cliff_model, device)
    gauss_z, gauss_img = collect_class_samples(loader, gauss_model, device)

    # generate panels
    cliff_rows, pair_labels, cliff_sims = make_panel(
        cliff_model, cliff_z, cliff_img, pairs, img_shape, class_names)
    gauss_rows, _, gauss_sims = make_panel(
        gauss_model, gauss_z, gauss_img, pairs, img_shape, class_names)

    # build figure: two columns (gaussian_l2 | clifford), each with 6 sub-columns
    n_pairs = len(pairs)
    n_cols = 6  # A, B, bind, bundle, unbind->A, unbind->B
    col_labels = ["A", "B", "bind(A,B)", "bundle(A,B)", "unbind→A", "unbind→B"]

    C, H, W = img_shape
    gap = 4  # pixel gap between gaussian and clifford panels

    total_w = 2 * n_cols * W + gap
    total_h = n_pairs * H

    canvas = torch.ones(C, total_h, total_w) * 0.3  # dark gray separator

    for r in range(n_pairs):
        # gaussian panel (left)
        for c in range(n_cols):
            canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = gauss_rows[r][c]
        # clifford panel (right)
        for c in range(n_cols):
            x_off = n_cols * W + gap
            canvas[:, r*H:(r+1)*H, x_off + c*W:x_off + (c+1)*W] = cliff_rows[r][c]

    fig, ax = plt.subplots(figsize=(14, n_pairs * 1.8))

    if C == 1:
        ax.imshow(canvas.squeeze(0).numpy(), cmap="gray")
    else:
        ax.imshow(canvas.permute(1, 2, 0).numpy())

    # y-axis: pair labels
    ax.set_yticks([H * i + H // 2 for i in range(n_pairs)])
    ax.set_yticklabels(pair_labels, fontsize=9)

    # x-axis: column labels for both panels
    gauss_ticks = [W * i + W // 2 for i in range(n_cols)]
    cliff_ticks = [n_cols * W + gap + W * i + W // 2 for i in range(n_cols)]
    all_ticks = gauss_ticks + cliff_ticks
    all_labels = col_labels + col_labels
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=7, rotation=30, ha="right")

    # panel headers
    gauss_center = (n_cols * W) / 2
    cliff_center = n_cols * W + gap + (n_cols * W) / 2
    avg_gauss_sim = np.mean(gauss_sims)
    avg_cliff_sim = np.mean(cliff_sims)

    ax.text(gauss_center, -H * 0.3,
            f"Gaussian (L2) — avg sim: {avg_gauss_sim:.3f}",
            ha="center", fontsize=11, fontweight="bold", color="#1f77b4")
    ax.text(cliff_center, -H * 0.3,
            f"Clifford — avg sim: {avg_cliff_sim:.3f}",
            ha="center", fontsize=11, fontweight="bold", color="#2ca02c")

    ax.set_title("Pairwise Bind, Bundle & Unbind Recovery", fontsize=13, fontweight="bold", pad=30)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()

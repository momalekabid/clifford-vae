"""generate compact bind/bundle/unbind comparison figure for the paper.
loads two checkpoints and runs test_pairwise_bind_bundle_decode on each,
then stitches selected rows side by side.

usage:
  python -m scripts.paper_bind_bundle_figure \
    --clifford_ckpt path/to/clifford.pt \
    --gaussian_ckpt path/to/gaussian_l2.pt \
    --dataset fashionmnist --latent_dim 256 --arch cnn \
    --output figures/bind_bundle_comparison.png
"""

import argparse
import os
import sys
import tempfile
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import test_pairwise_bind_bundle_decode


FASHIONMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

CIFAR_CLASSES = [
    "plane", "auto", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_model(ckpt_path, distribution, latent_dim, arch, dataset, device):
    if arch == "hybrid":
        from cnn.cliffordar_model import HybridVAE
        img_size = 64 if dataset == "dsprites" else 32
        in_ch = 3 if dataset == "cifar10" else 1
        model = HybridVAE(
            latent_dim=latent_dim, in_channels=in_ch,
            distribution=distribution, img_size=img_size,
        )
    elif arch == "cnn":
        from cnn.models import VAE
        in_ch = 3 if dataset == "cifar10" else 1
        model = VAE(
            latent_dim=latent_dim, in_channels=in_ch,
            distribution=distribution, device=str(device),
        )
    else:
        raise ValueError(f"unsupported arch: {arch}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_pairwise(model, loader, device, class_names, img_shape, n_classes):
    """run the standard pairwise test and return the output image path + avg sim."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = test_pairwise_bind_bundle_decode(
            model, loader, device, tmpdir,
            class_names=class_names,
            img_shape=img_shape,
            n_classes=n_classes,
        )
        path = result.get("pairwise_bind_bundle_path")
        avg_sim = result.get("avg_unbind_similarity", 0.0)
        if path and os.path.exists(path):
            img = mpimg.imread(path)
            return img, avg_sim
    return None, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clifford_ckpt", required=True)
    parser.add_argument("--gaussian_ckpt", required=True)
    parser.add_argument("--dataset", default="fashionmnist", choices=["fashionmnist", "cifar10"])
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--arch", default="cnn", choices=["cnn", "hybrid"])
    parser.add_argument("--output", default="bind_bundle_comparison.png")
    args = parser.parse_args()

    # force cpu for mps complex op compatibility
    device = torch.device("cpu")

    if args.dataset == "fashionmnist":
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
        class_names = FASHIONMNIST_CLASSES
        img_shape = (1, 32, 32)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        test_ds = datasets.CIFAR10("data", train=False, download=True, transform=transform)
        class_names = CIFAR_CLASSES
        img_shape = (3, 32, 32)

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("loading clifford model...")
    cliff_model = load_model(args.clifford_ckpt, "clifford", args.latent_dim, args.arch, args.dataset, device)

    print("loading gaussian (l2) model...")
    gauss_model = load_model(args.gaussian_ckpt, "gaussian", args.latent_dim, args.arch, args.dataset, device)
    if hasattr(gauss_model, 'l2_normalize'):
        gauss_model.l2_normalize = True

    print("running pairwise test on gaussian...")
    gauss_img, gauss_sim = run_pairwise(gauss_model, loader, device, class_names, img_shape, 10)

    print("running pairwise test on clifford...")
    cliff_img, cliff_sim = run_pairwise(cliff_model, loader, device, class_names, img_shape, 10)

    if gauss_img is None or cliff_img is None:
        print("error: one of the models failed to produce output")
        return

    # stitch side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(gauss_img.shape[0] / 80, 8)))

    ax1.imshow(gauss_img)
    ax1.set_title(f"Gaussian (L2) — avg sim: {gauss_sim:.3f}", fontsize=13, fontweight="bold", color="#1f77b4")
    ax1.axis("off")

    ax2.imshow(cliff_img)
    ax2.set_title(f"Clifford — avg sim: {cliff_sim:.3f}", fontsize=13, fontweight="bold", color="#2ca02c")
    ax2.axis("off")

    fig.suptitle("Pairwise Bind, Bundle & Unbind Recovery", fontsize=15, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()

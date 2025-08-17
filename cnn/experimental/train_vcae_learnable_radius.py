import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models_learnable_radius import VAEWithLearnableRadius
from utils.wandb_utils import WandbLogger, test_fourier_properties


DEVICE = (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0}
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
    n = len(loader.dataset)
    return {f"train/{k}_loss": v / n for k, v in sums.items()}


def test_epoch(model, loader, device):
    model.eval()
    sums = {"total": 0.0, "recon": 0.0, "kld": 0.0}
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, q_z, p_z, _ = model(x)
            losses = model.compute_loss(x, x_recon, q_z, p_z, beta=1.0)
            for k in ["total", "recon", "kld"]:
                sums[k] += losses[f"{k}_loss"].item() * x.size(0)
    n = len(loader.dataset)
    return {f"test/{k}_loss": v / n for k, v in sums.items()}


def save_reconstructions(model, loader, device, path, n_images=10):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n_images].to(device)
    with torch.no_grad():
        x_recon, _, _, _ = model(x)
    import torchvision
    grid = torch.cat([x.cpu(), x_recon.cpu()], dim=0)
    grid = (grid * 0.5 + 0.5).clamp(0, 1)
    torchvision.utils.save_image(grid, path, nrow=n_images)
    return path


def main(args):
    print(f"Device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = [args.latent_dim]
    distributions = args.distributions
    dataset_name = args.dataset
    is_color = dataset_name == "cifar10"
    in_channels = 3 if is_color else 1
    dataset_map = {"fashionmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}
    norm_mean, norm_std = (
        ((0.5,), (0.5,)) if not is_color else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    dataset_class = dataset_map[dataset_name]
    train_set = dataset_class("data", train=True, download=True, transform=transform)
    test_set = dataset_class("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    for latent_dim in latent_dims:
        for dist_name in distributions:
            exp_name = f"{dataset_name}-{dist_name}-d{latent_dim}-{args.recon_loss}-learnR"
            output_dir = f"results/{exp_name}"
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n== {exp_name} ==")
            logger.start_run(exp_name, args)

            model = VAEWithLearnableRadius(
                latent_dim=latent_dim,
                in_channels=in_channels,
                distribution=dist_name,
                device=DEVICE,
                recon_loss_type=args.recon_loss,
                use_perceptual_loss=args.use_perceptual,
                l1_weight=args.l1_weight,
                freq_weight=args.freq_weight,
                learn_radius=args.learn_radius,
                bayesian_radius=args.bayesian_radius,
                radius_prior_mu=args.radius_prior_mu,
                radius_prior_sigma=args.radius_prior_sigma,
            )

            logger.watch_model(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            best = float("inf")

            for epoch in range(args.epochs):
                beta = min(1.0, (epoch + 1) / args.warmup_epochs) * args.max_beta
                train_losses = train_epoch(model, train_loader, optimizer, DEVICE, beta)
                test_losses = test_epoch(model, test_loader, DEVICE)
                val = test_losses["test/total_loss"]
                if np.isfinite(val) and val < best:
                    best = val
                    torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                logger.log_metrics({
                    "epoch": epoch,
                    **train_losses,
                    **test_losses,
                    "best_test_loss": best,
                    "beta": beta,
                })

            print(f"best loss: {best:.4f}")

            if os.path.exists(f"{output_dir}/best_model.pt"):
                model.load_state_dict(torch.load(f"{output_dir}/best_model.pt", map_location=DEVICE))

                # Fourier/HRR diagnostics
                fourier = test_fourier_properties(model, test_loader, DEVICE, output_dir)

                # Reconstructions
                recon_path = save_reconstructions(model, test_loader, DEVICE, f"{output_dir}/reconstructions.png")

                images = {
                    "fft_spectrum": fourier.pop("fft_spectrum_plot_path"),
                    "reconstructions": recon_path,
                }
                summary = {"final_best_loss": best, **fourier}
                logger.log_summary(summary)
                logger.log_images(images)

            logger.finish_run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fashionmnist", choices=["fashionmnist", "cifar10"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--distributions", nargs="+", default=["powerspherical", "clifford", "gaussian"]) 
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_loss", type=str, default="l1_freq", choices=["mse", "l1_freq"])
    p.add_argument("--use_perceptual", action="store_true", help="Use perceptual LPIPS loss")
    p.add_argument("--l1_weight", type=float, default=1.0, help="Weight for L1 pixel loss")
    p.add_argument("--freq_weight", type=float, default=0.15, help="Weight for frequency domain loss")
    p.add_argument("--max_beta", type=float, default=1.0)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="refactored-cnn-vae-learnR")

    # Radius learning controls
    p.add_argument("--learn_radius", action="store_true", help="Enable learnable radius for hyperspherical priors")
    p.add_argument("--bayesian_radius", action="store_true", help="Use Bayesian (Normal over log r) radius; otherwise deterministic with L2 prior")
    p.add_argument("--radius_prior_mu", type=float, default=0.0)
    p.add_argument("--radius_prior_sigma", type=float, default=0.25)

    args = p.parse_args()
    main(args)



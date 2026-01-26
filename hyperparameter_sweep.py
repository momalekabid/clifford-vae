"""
hyperparameter sweep for cnn vae experiments.
runs short training (20 epochs) to find best lr/beta for each latent dimension.
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn.models import VAE

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    return {k: v / n for k, v in sums.items()}


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
    return {k: v / n for k, v in sums.items()}


def run_single_config(
    dataset_name,
    dist_name,
    latent_dim,
    lr,
    max_beta,
    concentration_floor,
    epochs,
    batch_size,
    warmup_epochs,
    train_loader,
    test_loader,
    in_channels,
    use_wandb,
    wandb_project,
):
    """run single hyperparameter configuration and return results."""
    config_name = f"sweep-{dataset_name}-{dist_name}-d{latent_dim}-lr{lr}-beta{max_beta}-cf{concentration_floor}"

    # initialize wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=config_name,
            config={
                "dataset": dataset_name,
                "distribution": dist_name,
                "latent_dim": latent_dim,
                "lr": lr,
                "max_beta": max_beta,
                "concentration_floor": concentration_floor,
                "epochs": epochs,
                "batch_size": batch_size,
                "warmup_epochs": warmup_epochs,
            },
            reinit=True,
        )

    model = VAE(
        latent_dim=latent_dim,
        in_channels=in_channels,
        distribution=dist_name,
        device=DEVICE,
        recon_loss_type="l1",
        l1_weight=1.0,
        freq_weight=0.0,
        l2_normalize=False,
        concentration_floor=concentration_floor,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # track metrics
    kl_history = []
    recon_history = []
    best_recon = float("inf")

    for epoch in range(epochs):
        # linear warmup to max_beta
        beta = min(1.0, (epoch + 1) / max(1, warmup_epochs)) * max_beta

        train_losses = train_epoch(model, train_loader, optimizer, DEVICE, beta)
        test_losses = test_epoch(model, test_loader, DEVICE)

        kl_history.append(test_losses["kld"])
        recon_history.append(test_losses["recon"])

        if test_losses["recon"] < best_recon:
            best_recon = test_losses["recon"]

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/total_loss": train_losses["total"],
                    "train/recon_loss": train_losses["recon"],
                    "train/kld_loss": train_losses["kld"],
                    "test/total_loss": test_losses["total"],
                    "test/recon_loss": test_losses["recon"],
                    "test/kld_loss": test_losses["kld"],
                    "beta": beta,
                }
            )

        # print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(
                f"  epoch {epoch+1}: kl={test_losses['kld']:.4f}, recon={test_losses['recon']:.4f}"
            )

    # check for posterior collapse (kl < 0.1 at end of training)
    final_kl = kl_history[-1] if kl_history else 0
    collapsed = final_kl < 0.1

    results = {
        "config_name": config_name,
        "dataset": dataset_name,
        "distribution": dist_name,
        "latent_dim": latent_dim,
        "lr": lr,
        "max_beta": max_beta,
        "concentration_floor": concentration_floor,
        "final_kl": final_kl,
        "final_recon": recon_history[-1] if recon_history else float("inf"),
        "best_recon": best_recon,
        "collapsed": collapsed,
        "kl_history": kl_history,
        "recon_history": recon_history,
    }

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                "final_kl": final_kl,
                "final_recon": results["final_recon"],
                "best_recon": best_recon,
                "posterior_collapsed": int(collapsed),
            }
        )
        wandb.finish()

    return results


def run_sweep(args):
    """run hyperparameter sweep."""
    print(f"device: {DEVICE}")
    print(f"running sweep with {len(args.latent_dims)} dims x {len(args.lrs)} lrs x {len(args.betas)} betas x {len(args.concentration_floors)} concentration_floors")

    # dataset setup
    all_results = []
    timing_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"dataset: {dataset_name}")
        print(f"{'='*50}")

        # configure dataset
        if dataset_name == "fashionmnist":
            dataset_class = datasets.FashionMNIST
            in_channels = 1
            norm_mean, norm_std = (0.5,), (0.5,)
        elif dataset_name == "cifar10":
            dataset_class = datasets.CIFAR10
            in_channels = 3
            norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            raise ValueError(f"unknown dataset: {dataset_name}")

        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        train_set = dataset_class("data", train=True, download=True, transform=transform)
        test_set = dataset_class("data", train=False, download=True, transform=transform)

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        for dist_name in args.distributions:
            print(f"\n--- distribution: {dist_name} ---")

            for latent_dim in args.latent_dims:
                print(f"\n  latent_dim: {latent_dim}")

                for lr in args.lrs:
                    for max_beta in args.betas:
                        for concentration_floor in args.concentration_floors:
                            config_start = time.time()
                            print(f"    lr={lr}, beta={max_beta}, cf={concentration_floor} ...", end=" ", flush=True)

                            try:
                                result = run_single_config(
                                    dataset_name=dataset_name,
                                    dist_name=dist_name,
                                    latent_dim=latent_dim,
                                    lr=lr,
                                    max_beta=max_beta,
                                    concentration_floor=concentration_floor,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    warmup_epochs=args.warmup_epochs,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    in_channels=in_channels,
                                    use_wandb=not args.no_wandb,
                                    wandb_project=args.wandb_project,
                                )
                                all_results.append(result)

                                status = "COLLAPSED" if result["collapsed"] else "OK"
                                config_time = time.time() - config_start
                                print(
                                    f"{status}, kl={result['final_kl']:.3f}, recon={result['best_recon']:.3f} ({config_time:.1f}s)"
                                )

                                timing_results[result["config_name"]] = config_time

                            except Exception as e:
                                print(f"ERROR: {e}")
                                all_results.append(
                                    {
                                        "config_name": f"sweep-{dataset_name}-{dist_name}-d{latent_dim}-lr{lr}-beta{max_beta}-cf{concentration_floor}",
                                        "error": str(e),
                                    }
                                )

    # save all results
    output_dir = "sweep_results"
    os.makedirs(output_dir, exist_ok=True)

    results_path = f"{output_dir}/sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults saved to {results_path}")

    # save timing
    timing_path = f"{output_dir}/sweep_timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing_results, f, indent=2)

    # find best configs per dataset/distribution/dim
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS (non-collapsed, lowest recon)")
    print("=" * 60)

    for dataset_name in args.datasets:
        for dist_name in args.distributions:
            for latent_dim in args.latent_dims:
                matching = [
                    r
                    for r in all_results
                    if r.get("dataset") == dataset_name
                    and r.get("distribution") == dist_name
                    and r.get("latent_dim") == latent_dim
                    and not r.get("collapsed", True)
                    and "error" not in r
                ]

                if matching:
                    best = min(matching, key=lambda x: x["best_recon"])
                    print(
                        f"{dataset_name}/{dist_name}/d{latent_dim}: lr={best['lr']}, beta={best['max_beta']}, "
                        f"cf={best.get('concentration_floor', 0.07)}, kl={best['final_kl']:.3f}, recon={best['best_recon']:.3f}"
                    )
                else:
                    print(
                        f"{dataset_name}/{dist_name}/d{latent_dim}: ALL COLLAPSED or errored"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hyperparameter sweep for cnn vae")

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["fashionmnist"],
        choices=["fashionmnist", "cifar10"],
        help="datasets to sweep",
    )
    parser.add_argument(
        "--distributions",
        type=str,
        nargs="+",
        default=["clifford"],
        choices=["gaussian", "powerspherical", "clifford"],
        help="distributions to sweep",
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="latent dimensions to sweep",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        help="learning rates to sweep",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0],
        help="max beta values to sweep",
    )
    parser.add_argument(
        "--concentration_floors",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.07, 0.1, 0.2],
        help="concentration floor values to sweep (for clifford distribution)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="epochs per config (keep short for sweep)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="warmup epochs for kl annealing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="disable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="clifford-vae-sweep",
        help="wandb project name",
    )

    args = parser.parse_args()
    run_sweep(args)

#!/usr/bin/env python3
"""
unified script to run training experiments and generate publication-quality plots.
runs both fashion_train.py and plot_results.py with configurable parameters.

usage:
    # run training and plotting with default settings:
    python run_experiments.py

    # skip training and only generate plots:
    python run_experiments.py --skip-training

    # custom config:
    python run_experiments.py --epochs 100 --latent-dims 4 128 512 --datasets fashionmnist
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_training(args):
    """run the training script with specified parameters"""
    print("\n" + "=" * 80)
    print("STARTING TRAINING PHASE")
    print("=" * 80 + "\n")

    cmd = [
        sys.executable,
        "cnn/fashion_train.py",
        "--epochs",
        str(args.epochs),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--max_beta",
        str(args.max_beta),
        "--patience",
        str(args.patience),
        "--cycle_epochs",
        str(args.cycle_epochs),
        "--recon_loss",
        args.recon_loss,
        "--l1_weight",
        str(args.l1_weight),
    ]

    if args.latent_dims:
        cmd.extend(["--latent_dims"] + [str(d) for d in args.latent_dims])

    if args.braid:
        cmd.append("--braid")

    if args.no_wandb:
        cmd.append("--no_wandb")

    if args.wandb_project:
        cmd.extend(["--wandb_project", args.wandb_project])

    print(f"running command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nerror: training script failed with code {result.returncode}")
        if not args.continue_on_error:
            sys.exit(result.returncode)
    else:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    return result.returncode


def run_plotting(args):
    """run the plotting script to generate figures"""
    print("\n" + "=" * 80)
    print("STARTING PLOTTING PHASE")
    print("=" * 80 + "\n")

    # import plotting utilities
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.plotting import generate_all_plots

    try:
        generate_all_plots(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            source=args.plot_source,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            recon_loss=args.recon_loss
        )

        print("\n" + "=" * 80)
        print("PLOTTING COMPLETED SUCCESSFULLY")
        print(f"plots saved to: {args.output_dir}/")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\nerror: plotting failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="unified script to run clifford-vae experiments and generate plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # training parameters
    train_group = parser.add_argument_group("training parameters")
    train_group.add_argument("--epochs", type=int, default=500, help="training epochs")
    train_group.add_argument(
        "--warmup_epochs", type=int, default=100, help="kl warmup epochs"
    )
    train_group.add_argument("--batch_size", type=int, default=128, help="batch size")
    train_group.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    train_group.add_argument("--max_beta", type=float, default=1.0, help="max kl beta")
    train_group.add_argument(
        "--patience", type=int, default=75, help="early stopping patience"
    )
    train_group.add_argument(
        "--cycle_epochs",
        type=int,
        default=100,
        help="cycle length for cyclical kl beta (0=off)",
    )
    train_group.add_argument(
        "--recon_loss",
        type=str,
        default="l1",
        choices=["mse", "l1"],
        help="reconstruction loss type",
    )
    train_group.add_argument(
        "--l1_weight", type=float, default=1.0, help="l1 pixel loss weight"
    )
    train_group.add_argument(
        "--latent_dims",
        type=int,
        nargs="+",
        default=None,
        help="latent dimensions to test (default: [4, 1024, 4096])",
    )
    train_group.add_argument("--braid", action="store_true", help="run braiding tests")

    # plotting parameters
    plot_group = parser.add_argument_group("plotting parameters")
    plot_group.add_argument(
        "--plot_source",
        type=str,
        default="json",
        choices=["json", "wandb", "auto"],
        help="source for plotting data",
    )
    plot_group.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="directory containing experiment results",
    )
    plot_group.add_argument(
        "--output_dir",
        type=str,
        default="aggregate_plots",
        help="directory to save plots",
    )

    # wandb parameters
    wandb_group = parser.add_argument_group("wandb parameters")
    wandb_group.add_argument(
        "--no_wandb", action="store_true", help="disable wandb logging"
    )
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default="clifford-experiments-CNN",
        help="wandb project name",
    )
    wandb_group.add_argument(
        "--wandb_entity", type=str, default=None, help="wandb entity (username/org)"
    )

    # execution control
    exec_group = parser.add_argument_group("execution control")
    exec_group.add_argument(
        "--skip-training",
        action="store_true",
        help="skip training phase, only generate plots",
    )
    exec_group.add_argument(
        "--skip-plotting",
        action="store_true",
        help="skip plotting phase, only run training",
    )
    exec_group.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue to plotting even if training fails",
    )

    args = parser.parse_args()

    # validate arguments
    if args.skip_training and args.skip_plotting:
        print("error: cannot skip both training and plotting")
        sys.exit(1)

    # run training
    train_result = 0
    if not args.skip_training:
        train_result = run_training(args)
        if train_result != 0 and not args.continue_on_error:
            sys.exit(train_result)
    else:
        print("skipping training phase (--skip-training specified)")

    # run plotting
    plot_result = 0
    if not args.skip_plotting:
        plot_result = run_plotting(args)
    else:
        print("skipping plotting phase (--skip-plotting specified)")

    # final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    if not args.skip_training:
        print(f"training: {'✓ success' if train_result == 0 else '✗ failed'}")
    if not args.skip_plotting:
        print(f"plotting: {'✓ success' if plot_result == 0 else '✗ failed'}")
    print("=" * 80 + "\n")

    return max(train_result, plot_result)


if __name__ == "__main__":
    sys.exit(main())

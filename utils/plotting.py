#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from pathlib import Path

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, will only use local json files")


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# color scheme
COLORS = {
    "clifford": "#2ecc71",  # green
    "gaussian": "#e74c3c",  # red
    "powerspherical": "#3498db",  # blue
    "vmf": "#9b59b6",  # purple
}

MARKERS = {"clifford": "o", "gaussian": "s", "powerspherical": "^", "vmf": "D"}

LINESTYLES = {"clifford": "-", "gaussian": "--", "powerspherical": "-.", "vmf": ":"}


def extract_config_from_exp_name(exp_name):
    """extract dataset, distribution, latent_dim, recon_loss, trial from exp name"""
    parts = exp_name.split("-")
    dataset = parts[0]
    dist = parts[1]
    latent_dim = int(parts[2].replace("d", ""))
    recon_loss = parts[3] if len(parts) > 3 else "l1"

    trial = 1
    if len(parts) > 4 and parts[4].startswith("trial"):
        trial = int(parts[4].replace("trial", ""))

    return dataset, dist, latent_dim, recon_loss, trial


def load_results_from_json(results_dir="results"):
    """scan results directory for metrics.json files"""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"results directory {results_dir} not found")
        return results

    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            exp_name = exp_dir.name
            dataset, dist, latent_dim, recon_loss, trial = extract_config_from_exp_name(
                exp_name
            )

            results.append(
                {
                    "dataset": dataset,
                    "distribution": dist,
                    "latent_dim": latent_dim,
                    "recon_loss": recon_loss,
                    "trial": trial,
                    "metrics": metrics,
                }
            )
        except Exception as e:
            print(f"error loading {metrics_file}: {e}")

    return results


def load_results_from_wandb(project_name="clifford-experiments-CNN", entity=None):
    """load results from wandb"""
    if not WANDB_AVAILABLE:
        print("wandb not available")
        return []

    api = wandb.Api()
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)

    results = []
    for run in runs:
        summary = run.summary._json_dict
        exp_name = run.name

        try:
            dataset, dist, latent_dim, recon_loss, trial = extract_config_from_exp_name(
                exp_name
            )
        except:
            print(f"skipping run {exp_name} - couldn't parse name")
            continue

        results.append(
            {
                "dataset": dataset,
                "distribution": dist,
                "latent_dim": latent_dim,
                "recon_loss": recon_loss,
                "trial": trial,
                "metrics": summary,
            }
        )

    return results


def aggregate_metrics(results, dataset_filter=None, recon_loss_filter="l1"):
    filtered = [
        r
        for r in results
        if (dataset_filter is None or r["dataset"] == dataset_filter)
        and r["recon_loss"] == recon_loss_filter
    ]

    grouped = defaultdict(lambda: defaultdict(list))

    for r in filtered:
        dist = r["distribution"]
        latent_dim = r["latent_dim"]
        metrics = r["metrics"]

        # extract relevant metrics
        metric_names = [
            "knn_acc_100",
            "knn_acc_600",
            "knn_acc_1000",
            "knn_f1_100",
            "knn_f1_600",
            "knn_f1_1000",
            "mean_vector_cosine_acc",
            "final_best_loss",
            "*/similarity_after_k_binds_mean",
            "†/similarity_after_k_binds_mean",
        ]

        for metric_name in metric_names:
            if metric_name in metrics:
                grouped[(dist, latent_dim)][metric_name].append(metrics[metric_name])

    # compute mean and std
    aggregated = defaultdict(lambda: defaultdict(dict))

    for (dist, latent_dim), metrics_dict in grouped.items():
        for metric_name, values in metrics_dict.items():
            aggregated[dist][latent_dim][f"{metric_name}_mean"] = np.mean(values)
            aggregated[dist][latent_dim][f"{metric_name}_std"] = np.std(values)
            aggregated[dist][latent_dim][f"{metric_name}_count"] = len(values)

    return aggregated


def plot_single_metric(
    aggregated,
    metric_name,
    ylabel,
    title,
    save_path=None,
    log_scale=False,
    ylim=None,
    show_grid=True,
):
    """plot a single metric with improved styling"""
    fig, ax = plt.subplots(figsize=(8, 5))

    distributions = sorted(aggregated.keys())

    for dist in distributions:
        latent_dims = sorted(aggregated[dist].keys())
        means = []
        stds = []

        for ld in latent_dims:
            if f"{metric_name}_mean" in aggregated[dist][ld]:
                means.append(aggregated[dist][ld][f"{metric_name}_mean"])
                stds.append(aggregated[dist][ld][f"{metric_name}_std"])
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        # filter out nans
        valid_idx = ~np.isnan(means)
        if not valid_idx.any():
            continue

        latent_dims_valid = np.array(latent_dims)[valid_idx]
        means_valid = means[valid_idx]
        stds_valid = stds[valid_idx]

        color = COLORS.get(dist, "#95a5a6")
        marker = MARKERS.get(dist, "o")
        linestyle = LINESTYLES.get(dist, "-")

        label = (
            dist.replace("clifford", "Clifford")
            .replace("gaussian", "Gaussian")
            .replace("powerspherical", "Power Spherical")
        )

        ax.errorbar(
            latent_dims_valid,
            means_valid,
            yerr=stds_valid,
            label=label,
            marker=marker,
            markersize=8,
            color=color,
            linewidth=2.5,
            linestyle=linestyle,
            capsize=4,
            capthick=1.5,
            alpha=0.9,
            markeredgewidth=1.5,
            markeredgecolor="white",
        )

    ax.set_xlabel("Latent Dimension", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, pad=15)

    if log_scale:
        ax.set_xscale("log", base=2)

    if ylim:
        ax.set_ylim(ylim)

    if show_grid:
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    ax.legend(frameon=True, fancybox=True, shadow=True, loc="best", ncol=1)

    all_dims = sorted(
        set([ld for dist_data in aggregated.values() for ld in dist_data.keys()])
    )
    ax.set_xticks(all_dims)
    ax.set_xticklabels([str(x) for x in all_dims], rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"saved to {save_path} and {pdf_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_grid(aggregated, dataset_name, save_path=None):
    """create a 2x2 grid comparing key metrics"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    metrics_config = [
        ("knn_acc_1000", "k-NN Accuracy\n(1000 samples)", (0, 0), (0.5, 1.0)),
        ("mean_vector_cosine_acc", "Mean Vector\nCosine Accuracy", (0, 1), (0.5, 1.0)),
        ("final_best_loss", "Reconstruction Loss", (1, 0), None),
        ("knn_f1_1000", "k-NN F1 Score\n(1000 samples)", (1, 1), (0.5, 1.0)),
    ]

    for metric_name, ylabel, (row, col), ylim in metrics_config:
        ax = fig.add_subplot(gs[row, col])

        distributions = sorted(aggregated.keys())

        for dist in distributions:
            latent_dims = sorted(aggregated[dist].keys())
            means = []
            stds = []

            for ld in latent_dims:
                if f"{metric_name}_mean" in aggregated[dist][ld]:
                    means.append(aggregated[dist][ld][f"{metric_name}_mean"])
                    stds.append(aggregated[dist][ld][f"{metric_name}_std"])
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)

            valid_idx = ~np.isnan(means)
            if not valid_idx.any():
                continue

            latent_dims_valid = np.array(latent_dims)[valid_idx]
            means_valid = means[valid_idx]
            stds_valid = stds[valid_idx]

            color = COLORS.get(dist, "#95a5a6")
            marker = MARKERS.get(dist, "o")
            linestyle = LINESTYLES.get(dist, "-")

            label = (
                dist.replace("clifford", "Clifford")
                .replace("gaussian", "Gaussian")
                .replace("powerspherical", "Power Spherical")
            )

            ax.errorbar(
                latent_dims_valid,
                means_valid,
                yerr=stds_valid,
                label=label,
                marker=marker,
                markersize=7,
                color=color,
                linewidth=2.2,
                linestyle=linestyle,
                capsize=3,
                capthick=1.3,
                alpha=0.9,
                markeredgewidth=1.2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Latent Dimension", fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=11)

        if ylim:
            ax.set_ylim(ylim)

        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

        all_dims = sorted(
            set([ld for dist_data in aggregated.values() for ld in dist_data.keys()])
        )
        ax.set_xticks(all_dims)
        ax.set_xticklabels([str(x) for x in all_dims], rotation=0)

        if row == 0 and col == 1:
            ax.legend(frameon=True, fancybox=True, shadow=True, loc="best", fontsize=9)

    dataset_title = dataset_name.replace("fashionmnist", "Fashion-MNIST").replace(
        "cifar10", "CIFAR-10"
    )
    fig.suptitle(
        f"Performance Comparison on {dataset_title}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"saved comparison grid to {save_path} and {pdf_path}")
    else:
        plt.show()

    plt.close()


def plot_binding_quality(aggregated, dataset_name, save_path=None):
    """plot binding quality metrics (* vs † methods)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    metrics = [
        ("*/similarity_after_k_binds_mean", "Binding Quality (Involution)", ax1),
        ("†/similarity_after_k_binds_mean", "Binding Quality (Reciprocal)", ax2),
    ]

    for metric_name, title, ax in metrics:
        distributions = sorted(aggregated.keys())

        for dist in distributions:
            latent_dims = sorted(aggregated[dist].keys())
            means = []
            stds = []

            for ld in latent_dims:
                if f"{metric_name}_mean" in aggregated[dist][ld]:
                    means.append(aggregated[dist][ld][f"{metric_name}_mean"])
                    stds.append(aggregated[dist][ld][f"{metric_name}_std"])
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)

            valid_idx = ~np.isnan(means)
            if not valid_idx.any():
                continue

            latent_dims_valid = np.array(latent_dims)[valid_idx]
            means_valid = means[valid_idx]
            stds_valid = stds[valid_idx]

            color = COLORS.get(dist, "#95a5a6")
            marker = MARKERS.get(dist, "o")
            linestyle = LINESTYLES.get(dist, "-")

            label = (
                dist.replace("clifford", "Clifford")
                .replace("gaussian", "Gaussian")
                .replace("powerspherical", "Power Spherical")
            )

            ax.errorbar(
                latent_dims_valid,
                means_valid,
                yerr=stds_valid,
                label=label,
                marker=marker,
                markersize=7,
                color=color,
                linewidth=2.2,
                linestyle=linestyle,
                capsize=3,
                capthick=1.3,
                alpha=0.9,
                markeredgewidth=1.2,
                markeredgecolor="white",
            )

        ax.set_xlabel("Latent Dimension", fontweight="bold")
        ax.set_ylabel("Similarity (Cosine)", fontweight="bold")
        ax.set_title(title, pad=10)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

        all_dims = sorted(
            set([ld for dist_data in aggregated.values() for ld in dist_data.keys()])
        )
        ax.set_xticks(all_dims)
        ax.set_xticklabels([str(x) for x in all_dims], rotation=0)
        ax.set_ylim([0, 1.05])

        if ax == ax1:
            ax.legend(frameon=True, fancybox=True, shadow=True, loc="best")

    dataset_title = dataset_name.replace("fashionmnist", "Fashion-MNIST").replace(
        "cifar10", "CIFAR-10"
    )
    fig.suptitle(
        f"HRR Binding Quality on {dataset_title}", fontsize=14, fontweight="bold", y=1.0
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"saved binding quality plot to {save_path} and {pdf_path}")
    else:
        plt.show()

    plt.close()


def extract_bundle_data(metrics_file):
    """extract bundle capacity accuracy data from metrics json"""
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        bundle_accs = {}
        for key, val in metrics.items():
            if "bundle_acc_k" in key and "_no_braid" in key:
                # extract k value
                k_str = key.split("_k")[1].split("_")[0]
                k = int(k_str)
                bundle_accs[k] = val

        return sorted(bundle_accs.items())
    except Exception as e:
        print(f"error reading {metrics_file}: {e}")
        return []


def plot_bundle_capacity_comparison(
    results_dir, output_path, dataset_filter="fashionmnist"
):
    """create comparison plot of bundle capacity across distributions and dimensions"""
    results_path = Path(results_dir)

    data = {}

    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        if dataset_filter not in exp_name:
            continue

        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        # parse exp name
        parts = exp_name.split("-")
        if len(parts) < 3:
            continue

        dist = parts[1]
        latent_dim = int(parts[2].replace("d", ""))

        bundle_data = extract_bundle_data(metrics_file)
        if bundle_data:
            data[(dist, latent_dim)] = bundle_data

    if not data:
        print(f"no bundle capacity data found for {dataset_filter}")
        return

    # create figure with subplots for each distribution
    distributions = sorted(set(k[0] for k in data.keys()))
    latent_dims = sorted(set(k[1] for k in data.keys()))

    fig, axes = plt.subplots(
        1, len(distributions), figsize=(5 * len(distributions), 4.5)
    )
    if len(distributions) == 1:
        axes = [axes]

    for idx, dist in enumerate(distributions):
        ax = axes[idx]

        for latent_dim in latent_dims:
            if (dist, latent_dim) not in data:
                continue

            bundle_data = data[(dist, latent_dim)]
            k_vals = [item[0] for item in bundle_data]
            accs = [item[1] for item in bundle_data]

            color = COLORS.get(dist, "#95a5a6")
            ax.plot(
                k_vals,
                accs,
                marker="o",
                markersize=6,
                linewidth=2,
                alpha=0.8,
                label=f"d={latent_dim}",
            )

        ax.set_xlabel("Bundle Size (k)", fontweight="bold")
        ax.set_ylabel("Retrieval Accuracy", fontweight="bold")

        dist_title = (
            dist.replace("clifford", "Clifford")
            .replace("gaussian", "Gaussian")
            .replace("powerspherical", "Power Spherical")
        )
        ax.set_title(dist_title, fontweight="bold")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=True, fancybox=True)
        ax.set_ylim([0, 1.05])

    dataset_title = dataset_filter.replace("fashionmnist", "Fashion-MNIST").replace(
        "cifar10", "CIFAR-10"
    )
    fig.suptitle(
        f"Bundle Capacity Comparison on {dataset_title}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"saved bundle capacity comparison to {output_path}")
    plt.close()


def generate_all_plots(
    results_dir="results",
    output_dir="aggregate_plots",
    source="json",
    wandb_project="clifford-experiments-CNN",
    wandb_entity=None,
    recon_loss="l1",
):
    """main function to generate all plots from experiment results"""

    # load results
    results = []

    if source == "json":
        print(f"loading results from {results_dir}...")
        results = load_results_from_json(results_dir)
    elif source == "wandb":
        print(f"loading results from wandb project {wandb_project}...")
        results = load_results_from_wandb(wandb_project, wandb_entity)
    else:
        print("trying both json and wandb...")
        results = load_results_from_json(results_dir)
        if not results and WANDB_AVAILABLE:
            results = load_results_from_wandb(wandb_project, wandb_entity)

    if not results:
        print("no results found! make sure experiments have saved metrics.json files.")
        return

    print(f"loaded {len(results)} experiment results")

    datasets = sorted(set(r["dataset"] for r in results))
    print(f"found datasets: {datasets}")

    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        print(f"\ngenerating plots for {dataset}...")

        aggregated = aggregate_metrics(
            results, dataset_filter=dataset, recon_loss_filter=recon_loss
        )

        if not aggregated:
            print(f"no data for {dataset}")
            continue

        dataset_title = dataset.replace("fashionmnist", "Fashion-MNIST").replace(
            "cifar10", "CIFAR-10"
        )

        # comparison grid
        plot_comparison_grid(
            aggregated, dataset, save_path=f"{output_dir}/{dataset}_comparison_grid.png"
        )

        # individual high-quality plots
        plot_single_metric(
            aggregated,
            metric_name="knn_acc_1000",
            ylabel="Classification Accuracy",
            title=f"k-NN Classification Performance on {dataset_title}\n(1000 training samples)",
            save_path=f"{output_dir}/{dataset}_knn_accuracy.png",
            ylim=(0.5, 1.0),
        )

        plot_single_metric(
            aggregated,
            metric_name="mean_vector_cosine_acc",
            ylabel="Classification Accuracy",
            title=f"Mean Vector Cosine Classification on {dataset_title}",
            save_path=f"{output_dir}/{dataset}_mean_vector_accuracy.png",
            ylim=(0.5, 1.0),
        )

        plot_single_metric(
            aggregated,
            metric_name="final_best_loss",
            ylabel="Reconstruction Loss",
            title=f"Final Reconstruction Loss on {dataset_title}",
            save_path=f"{output_dir}/{dataset}_reconstruction_loss.png",
        )

        # binding quality plots
        plot_binding_quality(
            aggregated, dataset, save_path=f"{output_dir}/{dataset}_binding_quality.png"
        )

        # bundle capacity comparison
        plot_bundle_capacity_comparison(
            results_dir,
            f"{output_dir}/{dataset}_bundle_capacity_comparison.png",
            dataset_filter=dataset,
        )

    print(f"\nall plots saved to {output_dir}/")
    print(f"both .png (300 dpi) and .pdf versions generated for publication")

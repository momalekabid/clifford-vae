#!/usr/bin/env python3
"""
aggregate and plot experiment results across latent dimensions for different distributions.
reads from wandb or local json files saved by fashion_train.py

usage:
    # from wandb:
    python plot_results.py --source wandb --wandb_project clifford-experiments-CNN

    # from local json:
    python plot_results.py --source json --results_dir results/

    # auto-detect:
    python plot_results.py
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, will only use local json files")


def extract_config_from_exp_name(exp_name):
    """extract dataset, distribution, latent_dim, recon_loss, trial from exp name
    format: {dataset}-{dist}-d{latent_dim}-{recon_loss}-trial{n} or without trial suffix
    """
    parts = exp_name.split('-')
    dataset = parts[0]
    dist = parts[1]
    latent_dim = int(parts[2].replace('d', ''))
    recon_loss = parts[3] if len(parts) > 3 else 'l1'

    trial = 1
    if len(parts) > 4 and parts[4].startswith('trial'):
        trial = int(parts[4].replace('trial', ''))

    return dataset, dist, latent_dim, recon_loss, trial


def load_timing_data(timing_file='cnn/fashion_train_timing.json'):
    """load timing data from json file"""
    timing_results = []

    if not os.path.exists(timing_file):
        print(f"timing file {timing_file} not found")
        return timing_results

    try:
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)

        for exp_name, times in timing_data.items():
            if exp_name == 'total_script_time_s':
                continue

            try:
                dataset, dist, latent_dim, recon_loss, trial = extract_config_from_exp_name(exp_name)
                timing_results.append({
                    'dataset': dataset,
                    'distribution': dist,
                    'latent_dim': latent_dim,
                    'recon_loss': recon_loss,
                    'trial': trial,
                    'train_time_s': times.get('train_time_s', 0),
                    'eval_time_s': times.get('eval_time_s', 0),
                    'total_exp_time_s': times.get('total_exp_time_s', 0)
                })
            except Exception as e:
                print(f"skipping {exp_name}: {e}")
    except Exception as e:
        print(f"error loading timing file: {e}")

    return timing_results


def load_results_from_json(results_dir='results'):
    """scan results directory for metrics.json files"""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"results directory {results_dir} not found")
        return results

    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / 'metrics.json'
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            exp_name = exp_dir.name
            dataset, dist, latent_dim, recon_loss, trial = extract_config_from_exp_name(exp_name)

            results.append({
                'dataset': dataset,
                'distribution': dist,
                'latent_dim': latent_dim,
                'recon_loss': recon_loss,
                'trial': trial,
                'metrics': metrics
            })
        except Exception as e:
            print(f"error loading {metrics_file}: {e}")

    return results


def load_results_from_wandb(project_name='clifford-experiments-CNN', entity=None):
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

        # extract relevant info
        exp_name = run.name
        try:
            dataset, dist, latent_dim, recon_loss, trial = extract_config_from_exp_name(exp_name)
        except:
            print(f"skipping run {exp_name} - couldn't parse name")
            continue

        results.append({
            'dataset': dataset,
            'distribution': dist,
            'latent_dim': latent_dim,
            'recon_loss': recon_loss,
            'trial': trial,
            'metrics': summary
        })

    return results


def aggregate_metrics(results, dataset_filter=None, recon_loss_filter='l1'):
    """aggregate results by distribution and latent dimension, compute mean and std across trials"""
    # filter
    filtered = [r for r in results
                if (dataset_filter is None or r['dataset'] == dataset_filter)
                and r['recon_loss'] == recon_loss_filter]

    # group by distribution and latent_dim
    grouped = defaultdict(lambda: defaultdict(list))

    for r in filtered:
        dist = r['distribution']
        latent_dim = r['latent_dim']
        metrics = r['metrics']

        # extract metrics we care about
        for metric_name in ['knn_acc_100', 'knn_acc_600', 'knn_acc_1000',
                           'knn_f1_100', 'knn_f1_600', 'knn_f1_1000',
                           'mean_vector_cosine_acc', 'final_best_loss']:
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


def aggregate_timing(timing_results, dataset_filter=None, recon_loss_filter='l1'):
    """aggregate timing data by distribution and latent dimension, compute mean and std across trials"""
    # filter
    filtered = [r for r in timing_results
                if (dataset_filter is None or r['dataset'] == dataset_filter)
                and r['recon_loss'] == recon_loss_filter]

    # group by distribution and latent_dim
    grouped = defaultdict(lambda: defaultdict(list))

    for r in filtered:
        dist = r['distribution']
        latent_dim = r['latent_dim']

        for time_metric in ['train_time_s', 'eval_time_s', 'total_exp_time_s']:
            if time_metric in r:
                grouped[(dist, latent_dim)][time_metric].append(r[time_metric])

    # compute mean and std
    aggregated = defaultdict(lambda: defaultdict(dict))

    for (dist, latent_dim), metrics_dict in grouped.items():
        for metric_name, values in metrics_dict.items():
            aggregated[dist][latent_dim][f"{metric_name}_mean"] = np.mean(values)
            aggregated[dist][latent_dim][f"{metric_name}_std"] = np.std(values)
            aggregated[dist][latent_dim][f"{metric_name}_count"] = len(values)

    return aggregated


def plot_metric_vs_latent_dim(aggregated, metric_name='knn_acc_1000',
                               ylabel='Accuracy', title_suffix='',
                               save_path=None):
    """plot metric vs latent dimension with separate lines for each distribution"""
    plt.figure(figsize=(10, 6))

    # sort distributions for consistent colors
    distributions = sorted(aggregated.keys())
    colors = {'gaussian': '#f39c12', 'powerspherical': '#3498db', 'clifford': '#27ae60'}
    markers = {'gaussian': 'o', 'powerspherical': 's', 'clifford': '^'}

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

        color = colors.get(dist, '#95a5a6')
        marker = markers.get(dist, 'o')

        plt.errorbar(latent_dims, means, yerr=stds,
                    label=f'{dist.capitalize()} (std across runs)',
                    marker=marker, markersize=8,
                    color=color, linewidth=2, capsize=5, capthick=2)

    plt.xlabel('Latent Dimension', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{ylabel} vs Latent Dimension{title_suffix}', fontsize=14)
    plt.legend(fontsize=10, frameon=True)
    plt.grid(alpha=0.3, linestyle='--')

    # set x-axis to show actual latent dims
    all_dims = sorted(set([ld for dist_data in aggregated.values() for ld in dist_data.keys()]))
    plt.xticks(all_dims, [str(x) for x in all_dims], rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_timing_vs_latent_dim(aggregated_timing, time_metric='total_exp_time_s',
                               ylabel='Time (seconds)', title_suffix='',
                               save_path=None):
    """plot timing metric vs latent dimension with separate lines for each distribution"""
    plt.figure(figsize=(10, 6))

    # sort distributions for consistent colors
    distributions = sorted(aggregated_timing.keys())
    colors = {'gaussian': '#f39c12', 'powerspherical': '#3498db', 'clifford': '#27ae60'}
    markers = {'gaussian': 'o', 'powerspherical': 's', 'clifford': '^'}

    for dist in distributions:
        latent_dims = sorted(aggregated_timing[dist].keys())
        means = []
        stds = []

        for ld in latent_dims:
            if f"{time_metric}_mean" in aggregated_timing[dist][ld]:
                means.append(aggregated_timing[dist][ld][f"{time_metric}_mean"])
                stds.append(aggregated_timing[dist][ld][f"{time_metric}_std"])
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        color = colors.get(dist, '#95a5a6')
        marker = markers.get(dist, 'o')

        plt.errorbar(latent_dims, means, yerr=stds,
                    label=f'{dist.capitalize()} (std across runs)',
                    marker=marker, markersize=8,
                    color=color, linewidth=2, capsize=5, capthick=2)

    plt.xlabel('Latent Dimension', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{ylabel} vs Latent Dimension{title_suffix}', fontsize=14)
    plt.legend(fontsize=10, frameon=True)
    plt.grid(alpha=0.3, linestyle='--')

    # set x-axis to show actual latent dims
    all_dims = sorted(set([ld for dist_data in aggregated_timing.values() for ld in dist_data.keys()]))
    plt.xticks(all_dims, [str(x) for x in all_dims], rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main(args):
    # load results
    results = []

    if args.source == 'json':
        print(f"loading results from {args.results_dir}...")
        results = load_results_from_json(args.results_dir)
    elif args.source == 'wandb':
        print(f"loading results from wandb project {args.wandb_project}...")
        results = load_results_from_wandb(args.wandb_project, args.wandb_entity)
    else:
        print("trying both json and wandb...")
        results = load_results_from_json(args.results_dir)
        if not results and WANDB_AVAILABLE:
            results = load_results_from_wandb(args.wandb_project, args.wandb_entity)

    if not results:
        print("no results found! make sure experiments have saved metrics.json files.")
        print("you may need to run fashion_train.py first or check wandb.")
        return

    print(f"loaded {len(results)} experiment results")

    # load timing data
    timing_results = load_timing_data(args.timing_file)
    print(f"loaded {len(timing_results)} timing results")

    # get unique datasets
    datasets = sorted(set(r['dataset'] for r in results))
    print(f"found datasets: {datasets}")

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # generate plots for each dataset
    for dataset in datasets:
        print(f"\ngenerating plots for {dataset}...")

        aggregated = aggregate_metrics(results,
                                      dataset_filter=dataset,
                                      recon_loss_filter=args.recon_loss)

        if not aggregated:
            print(f"no data for {dataset}")
            continue

        # plot knn accuracy for different sample sizes
        for n_samples in [100, 600, 1000]:
            metric_name = f'knn_acc_{n_samples}'
            plot_metric_vs_latent_dim(
                aggregated,
                metric_name=metric_name,
                ylabel='Accuracy',
                title_suffix=f'\n({dataset}, {n_samples} training samples)',
                save_path=f"{args.output_dir}/{dataset}_accuracy_vs_latent_dim_{n_samples}samples.png"
            )

        # plot knn f1 score
        for n_samples in [100, 600, 1000]:
            metric_name = f'knn_f1_{n_samples}'
            plot_metric_vs_latent_dim(
                aggregated,
                metric_name=metric_name,
                ylabel='F1 Score',
                title_suffix=f'\n({dataset}, {n_samples} training samples)',
                save_path=f"{args.output_dir}/{dataset}_f1_vs_latent_dim_{n_samples}samples.png"
            )

        # plot mean vector cosine accuracy
        plot_metric_vs_latent_dim(
            aggregated,
            metric_name='mean_vector_cosine_acc',
            ylabel='Mean Vector Cosine Accuracy',
            title_suffix=f'\n({dataset})',
            save_path=f"{args.output_dir}/{dataset}_mean_vector_acc_vs_latent_dim.png"
        )

        # plot final best loss
        plot_metric_vs_latent_dim(
            aggregated,
            metric_name='final_best_loss',
            ylabel='Final Best Loss',
            title_suffix=f'\n({dataset})',
            save_path=f"{args.output_dir}/{dataset}_loss_vs_latent_dim.png"
        )

        # plot timing metrics if available
        if timing_results:
            print(f"generating timing plots for {dataset}...")
            aggregated_timing = aggregate_timing(timing_results,
                                                dataset_filter=dataset,
                                                recon_loss_filter=args.recon_loss)

            if aggregated_timing:
                # plot training time
                plot_timing_vs_latent_dim(
                    aggregated_timing,
                    time_metric='train_time_s',
                    ylabel='Training Time (seconds)',
                    title_suffix=f'\n({dataset})',
                    save_path=f"{args.output_dir}/{dataset}_train_time_vs_latent_dim.png"
                )

                # plot evaluation time
                plot_timing_vs_latent_dim(
                    aggregated_timing,
                    time_metric='eval_time_s',
                    ylabel='Evaluation Time (seconds)',
                    title_suffix=f'\n({dataset})',
                    save_path=f"{args.output_dir}/{dataset}_eval_time_vs_latent_dim.png"
                )

                # plot total experiment time
                plot_timing_vs_latent_dim(
                    aggregated_timing,
                    time_metric='total_exp_time_s',
                    ylabel='Total Experiment Time (seconds)',
                    title_suffix=f'\n({dataset})',
                    save_path=f"{args.output_dir}/{dataset}_total_time_vs_latent_dim.png"
                )
            else:
                print(f"no timing data for {dataset}")

    print(f"\nall plots saved to {args.output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate and plot experiment results')
    parser.add_argument('--source', type=str, default='auto',
                       choices=['json', 'wandb', 'auto'],
                       help='source of results (json files, wandb, or auto-detect)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='directory containing result subdirectories with metrics.json')
    parser.add_argument('--wandb_project', type=str, default='clifford-experiments-CNN',
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='wandb entity (username/org)')
    parser.add_argument('--output_dir', type=str, default='aggregate_plots',
                       help='directory to save plots')
    parser.add_argument('--recon_loss', type=str, default='l1',
                       help='filter by reconstruction loss type')
    parser.add_argument('--timing_file', type=str, default='cnn/fashion_train_timing.json',
                       help='path to timing json file')

    args = parser.parse_args()
    main(args)

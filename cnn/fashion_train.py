import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as tu
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models import VAE
from utils.wandb_utils import (
    WandbLogger,
    test_fourier_properties,
    compute_class_means,
    evaluate_mean_vector_cosine,
    test_hrr_sentence,
    test_bundle_capacity,
    test_unbinding_of_bundled_pairs,
    plot_clifford_manifold_visualization,
    plot_powerspherical_manifold_visualization,
    plot_gaussian_manifold_visualization,
)


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


# evaluation and visualization helpers
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


def generate_tsne_plot(model, loader, device, path, n_samples=2000):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, _, _, mu = model(x.to(device))
            latents.append(mu.cpu().numpy())
            labels.append(y.numpy())
            if len(np.concatenate(labels)) >= n_samples:
                break
    Z = np.concatenate(latents)[:n_samples]
    Y = np.concatenate(labels)[:n_samples]
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    pts = tsne.fit_transform(Z)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=Y, cmap="tab10", s=8, alpha=0.8)
    plt.colorbar(sc, ticks=np.unique(Y))
    plt.title("t-SNE of Latent Means (μ)")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_pca_plot(model, loader, device, path, n_samples=2000):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, _, _, mu = model(x.to(device))
            latents.append(mu.cpu().numpy())
            labels.append(y.numpy())
            if len(np.concatenate(labels)) >= n_samples:
                break
    Z = np.concatenate(latents)[:n_samples]
    Y = np.concatenate(labels)[:n_samples]

    pca = PCA(n_components=min(50, Z.shape[1]))
    Z_pca = pca.fit_transform(Z)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=Y, cmap="tab10", s=8, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Latent Means (μ)")
    plt.colorbar(sc, ticks=np.unique(Y))
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def perform_knn_evaluation(
    model, train_loader, test_loader, device, n_samples_list=[100, 600, 1000, 2048]
):
    """k-nn classification on latent embeddings with multiple training sample sizes."""
    print("knn eval in progress")
    model.eval()

    def encode_dataset(loader):
        latents, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                _, _, _, mu = model(x.to(device))
                latents.append(mu.cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(latents), np.concatenate(labels)

    X_train_full, y_train_full = encode_dataset(train_loader)
    X_test, y_test = encode_dataset(test_loader)

    # cosine metric for hyperspherical distributions
    metric = (
        "cosine"
        if getattr(model, "distribution", None) in ["powerspherical", "clifford"]
        else "euclidean"
    )

    results = {}
    for n_samples in n_samples_list:
        if n_samples > len(X_train_full):
            print(
                f"Warning: k-NN sample size {n_samples} > training data size {len(X_train_full)}. Skipping."
            )
            continue

        indices = np.random.choice(len(X_train_full), n_samples, replace=False)
        X_train_sample, y_train_sample = X_train_full[indices], y_train_full[indices]

        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train_sample, y_train_sample)

        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results[f"knn_acc_{n_samples}"] = float(accuracy)
        results[f"knn_f1_{n_samples}"] = float(f1)
        print(f"knn accuracy with {n_samples} training samples: {accuracy:.4f}")

    return results


def main(args):
    print(f"Device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = [128, 256, 512, 1024] #, 2048, 4096, 5000]
    distributions = ["powerspherical", "clifford", "gaussian"]
    datasets_to_test = ["fashionmnist"] #, "cifar10"]
    dataset_map = {"fashionmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}

    for dataset_name in datasets_to_test:
        is_color = dataset_name == "cifar10"
        in_channels = 3 if is_color else 1
        dataset_class = dataset_map[dataset_name]
        norm_mean, norm_std = (
            ((0.5,), (0.5,)) if not is_color else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        train_set = dataset_class(
            "data", train=True, download=True, transform=transform
        )
        test_set = dataset_class(
            "data", train=False, download=True, transform=transform
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        all_bundle_capacity_results = defaultdict(lambda: {"dims": [], "max_k_at_99_acc": []})
        all_unbind_bundled_results = defaultdict(lambda: {"dims": [], "max_k_at_99_acc": []})

        for latent_dim in latent_dims:
            for dist_name in distributions:
                exp_name = f"{dataset_name}-{dist_name}-d{latent_dim}-{args.recon_loss}"
                output_dir = f"results/{exp_name}"
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n== {exp_name} ==")
                logger.start_run(exp_name, args)

                model = VAE(
                    latent_dim=latent_dim,
                    in_channels=in_channels,
                    distribution=dist_name,
                    device=DEVICE,
                    recon_loss_type=args.recon_loss,
                    l1_weight=args.l1_weight,
                    freq_weight=args.freq_weight,
                )
                logger.watch_model(model)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr)
                best = float("inf")
                patience_counter = 0

                def kl_beta_for_epoch(e: int) -> float:
                    # initial warmup
                    if e < args.warmup_epochs:
                        return (
                            min(1.0, (e + 1) / max(1, args.warmup_epochs))
                            * args.max_beta
                        )
                    if args.cycle_epochs <= 0:
                        return args.max_beta
                    # cyclical annealing option, tri-schedule in [min_beta, max_beta]
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
                    val = test_losses["test/total_loss"]
                    if np.isfinite(val) and val < best:
                        best = val
                        torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            **train_losses,
                            **test_losses,
                            "best_test_loss": best,
                            "beta": beta,
                        }
                    )

                    if args.patience > 0 and patience_counter >= args.patience:
                        print(
                            f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)"
                        )
                        break

                print(f"best loss: {best:.4f}")

                if os.path.exists(f"{output_dir}/best_model.pt"):
                    model.load_state_dict(
                        torch.load(f"{output_dir}/best_model.pt", map_location=DEVICE)
                    )

                    use_unitary_keys = True
                    normalize_vectors = getattr(args, "vsa_normalize", True)

                    bundle_cap_res = test_bundle_capacity(
                        model,
                        test_loader,
                        DEVICE,
                        output_dir,
                        n_items=1000,
                        k_range=list(range(5, 51, 5)),
                        n_trials=20,
                        normalize_vectors=normalize_vectors,
                    )
                    
                    unbind_bundled_res_pseudo = test_unbinding_of_bundled_pairs(
                        model,
                        test_loader,
                        DEVICE,
                        output_dir,
                        unbind_method="pseudo",
                        n_items=1000,
                        k_range=list(range(5, 31, 5)),
                        n_trials=20,
                        normalize_vectors=normalize_vectors,
                        unitary_keys=use_unitary_keys,
                    )
                    
                    bundle_accs = bundle_cap_res.get("bundle_capacity_accuracies", {})
                    if bundle_accs:
                        ks = sorted(bundle_accs.keys())
                        k_at_99 = 0
                        for k_val in ks:
                            if bundle_accs[k_val] >= 0.99:
                                k_at_99 = k_val
                            else:
                                break
                        if k_at_99 > 0:
                            all_bundle_capacity_results[dist_name]["dims"].append(latent_dim)
                            all_bundle_capacity_results[dist_name]["max_k_at_99_acc"].append(k_at_99)

                    unbind_accs = unbind_bundled_res_pseudo.get("unbind_bundled_accuracies", {})
                    if unbind_accs:
                        ks = sorted(unbind_accs.keys())
                        k_at_99 = 0
                        for k_val in ks:
                            if unbind_accs[k_val] >= 0.99:
                                k_at_99 = k_val
                            else:
                                break
                        if k_at_99 > 0:
                            all_unbind_bundled_results[dist_name]["dims"].append(latent_dim)
                            all_unbind_bundled_results[dist_name]["max_k_at_99_acc"].append(k_at_99)


                    fourier_pseudo = test_fourier_properties(
                        model, test_loader, DEVICE, output_dir, unbind_method="pseudo"
                    )
                    fourier_deconv = test_fourier_properties(
                        model, test_loader, DEVICE, output_dir, unbind_method="deconv"
                    )

                    # reconstructions
                    recon_path = save_reconstructions(
                        model, test_loader, DEVICE, f"{output_dir}/reconstructions.png"
                    )

                    # t-SNE
                    tsne_path = generate_tsne_plot(
                        model, test_loader, DEVICE, f"{output_dir}/tsne.png"
                    )

                    # PCA
                    pca_path = generate_pca_plot(
                        model, test_loader, DEVICE, f"{output_dir}/pca.png"
                    )

                    # knn eval
                    knn_metrics = perform_knn_evaluation(
                        model, train_loader, test_loader, DEVICE, [100, 600, 1000]
                    )

                    train_subset = torch.utils.data.Subset(
                        train_set, list(range(min(5000, len(train_set))))
                    )
                    train_subset_loader = DataLoader(
                        train_subset, batch_size=args.batch_size, shuffle=False
                    )
                    class_means = compute_class_means(
                        model, train_subset_loader, DEVICE, max_per_class=1000
                    )
                    mean_vector_acc, _ = evaluate_mean_vector_cosine(
                        model, test_loader, DEVICE, class_means
                    )
                    mean_metric_key = "mean_vector_cosine_acc"
                    print(f"{mean_metric_key}: ", mean_vector_acc)

                    # vsa_bind_sim_pseudo = 0.0
                    # vsa_bind_sim_deconv = 0.0

                    fourier_metrics = {}
                    fourier_metrics.update({
                        f"pseudo/{k}": v
                        for k, v in fourier_pseudo.items()
                        if isinstance(v, (int, float, bool))
                    })
                    fourier_metrics.update({
                        f"deconv/{k}": v
                        for k, v in fourier_deconv.items()
                        if isinstance(v, (int, float, bool))
                    })

                    logger.log_metrics(
                        {
                            **knn_metrics,
                            **fourier_metrics,
                            mean_metric_key: float(mean_vector_acc),
                            # "vsa_bind_unbind_similarity_pseudo": float(vsa_bind_sim_pseudo),
                            # "vsa_bind_unbind_similarity_deconv": float(vsa_bind_sim_deconv),
                            "final_best_loss": best,
                        }
                    )

                    images = {
                        "reconstructions": recon_path,
                        "tsne": tsne_path,
                        "pca": pca_path,
                    }
                    if bundle_cap_res.get("bundle_capacity_plot"):
                        images["bundle_capacity"] = bundle_cap_res["bundle_capacity_plot"]
                    if unbind_bundled_res_pseudo.get("unbind_bundled_plot"):
                        images["unbind_bundled_pseudo"] = unbind_bundled_res_pseudo["unbind_bundled_plot"]
                        
                    sp = fourier_pseudo.get("similarity_after_k_binds_plot_path")
                    sd = fourier_deconv.get("similarity_after_k_binds_plot_path")
                    if sp:
                        images["similarity_after_k_binds_pseudo"] = sp
                    if sd:
                        images["similarity_after_k_binds_deconv"] = sd

                    if dist_name == "clifford" and model.latent_dim >= 2:
                        # manifold visualization
                        cliff_viz = plot_clifford_manifold_visualization(
                            model, DEVICE, output_dir, n_samples=1000, dims=(0, 1)
                        )
                        if cliff_viz:
                            images["clifford_manifold_visualization"] = cliff_viz

                    elif dist_name == "powerspherical" and model.latent_dim >= 2:
                        pow_viz = plot_powerspherical_manifold_visualization(
                            model, DEVICE, output_dir, n_samples=1000, dims=(0, 1)
                        )
                        if pow_viz:
                            images["powerspherical_manifold_visualization"] = pow_viz

                    elif dist_name == "gaussian" and model.latent_dim >= 2:
                        gauss_viz = plot_gaussian_manifold_visualization(
                            model, DEVICE, output_dir, n_samples=1000, dims=(0, 1)
                        )
                        if gauss_viz:
                            images["gaussian_manifold_visualization"] = gauss_viz

                    hrr_fashion_pseudo = test_hrr_sentence(
                        model, test_loader, DEVICE, output_dir,
                        unbind_method="pseudo", unitary_keys=True,
                        normalize_vectors=normalize_vectors,
                        dataset_name=dataset_name,
                    )
                    hrr_fashion_deconv = test_hrr_sentence(
                        model, test_loader, DEVICE, output_dir,
                        unbind_method="deconv", unitary_keys=True,
                        normalize_vectors=normalize_vectors,
                        dataset_name=dataset_name,
                    )
                    # hrr_fashion_pseudo_proj = test_hrr_sentence(
                    #     model, test_loader, DEVICE, output_dir, unbind_method="pseudo", unitary_keys=True, normalize_vectors=normalize_vectors, project_fillers=True, dataset_name=dataset_name
                    # )
                    # hrr_fashion_deconv_proj = test_hrr_sentence(
                    #     model, test_loader, DEVICE, output_dir, unbind_method="deconv", unitary_keys=True, normalize_vectors=normalize_vectors, project_fillers=True, dataset_name=dataset_name
                    # )

                    if hrr_fashion_pseudo.get("hrr_fashion_plot"):
                        images["hrr_fashion_pseudo"] = hrr_fashion_pseudo["hrr_fashion_plot"]
                    if hrr_fashion_deconv.get("hrr_fashion_plot"):
                        images["hrr_fashion_deconv"] = hrr_fashion_deconv["hrr_fashion_plot"]
                    # if hrr_fashion_pseudo_proj.get("hrr_fashion_plot"):
                    #     images["hrr_fashion_pseudo_proj"] = hrr_fashion_pseudo_proj["hrr_fashion_plot"]
                    # if hrr_fashion_deconv_proj.get("hrr_fashion_plot"):
                    #     images["hrr_fashion_deconv_proj"] = hrr_fashion_deconv_proj["hrr_fashion_plot"]
                    summary = {
                        "final_best_loss": best,
                        **fourier_metrics,
                        **knn_metrics,
                        mean_metric_key: float(mean_vector_acc),
                        # "vsa_bind_unbind_similarity_pseudo": float(vsa_bind_sim_pseudo),
                        # "vsa_bind_unbind_similarity_deconv": float(vsa_bind_sim_deconv),
                    }
                    logger.log_summary(summary)
                    logger.log_images(images)

                logger.finish_run()

        for results, name in [
            (all_bundle_capacity_results, "bundle_capacity"),
            (all_unbind_bundled_results, "unbind_bundled_pairs"),
        ]:
            plt.figure(figsize=(8, 6))
            for dist_name, data in results.items():
                if data["dims"]:
                    plt.plot(data["dims"], data["max_k_at_99_acc"], marker='o', label=dist_name)
            plt.xlabel("Latent Dimension")
            plt.ylabel("Max k for >= 99% Accuracy")
            plt.title(f"{name.replace('_', ' ').title()} on {dataset_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_path = f"results/{dataset_name}_{name}_comparison.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved summary plot to {plot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--warmup_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--recon_loss", type=str, default="mse", choices=["mse", "l1_freq"]
    )
    p.add_argument(
        "--l1_weight", type=float, default=1.0, help="Weight for L1 pixel loss"
    )
    p.add_argument(
        "--freq_weight",
        type=float,
        default=0.25,
        help="Weight for frequency domain loss",
    )
    p.add_argument("--max_beta", type=float, default=1.0)
    p.add_argument(
        "--min_beta", type=float, default=0.01, help="Minimum KL beta during cycles"
    )
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="conv-experiments-sep-2025")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument(
        "--cycle_epochs",
        type=int,
        default=100,
        help="Cycle length for cyclical KL beta after warmup (0=disabled)",
    )
    p.add_argument(
        "--vsa_normalize",
        action="store_true",
        default=True,
        help="Normalize vectors for vsa tests)",
    )
    args = p.parse_args()
    main(args)

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.utils as tu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_cross_class_bind_unbind,
    compute_class_means,
    evaluate_mean_vector_cosine,
    plot_powerspherical_manifold_visualization,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    test_per_class_bundle_capacity_two_items,
)
from mnist.mlp_vae import MLPVAE, vae_loss


class BinarizeWithRandomThreshold:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x > torch.rand_like(x)).float()


def encode_dataset(model, loader, device):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            z_mean, _ = model.encode(x.to(device).view(-1, 784))
            zs.append(z_mean.cpu())
            ys.append(y)
    return torch.cat(zs, 0).numpy(), torch.cat(ys, 0).numpy()


def perform_knn_evaluation(model, train_loader, test_loader, device, n_samples_list):
    """k-NN classification on latent embeddings with multiple training sample sizes."""
    X_train_full, y_train_full = encode_dataset(model, train_loader, device)
    X_test, y_test = encode_dataset(model, test_loader, device)

    results = {}
    metric = "cosine"

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
        print(f"  knn acc w/ {n_samples} for train, test: {accuracy:.4f}, f1: {f1:.4f}")

    return results


def plot_reconstructions(model, loader, device, filepath):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x[:8].to(device)
        _, _, _, x_recon = model(x)

        originals = x.cpu()
        recons = torch.sigmoid(x_recon.cpu()).view_as(originals)
        comparison = torch.cat([originals, recons])

        grid = tu.make_grid(comparison, nrow=8, pad_value=0.5)

        plt.figure(figsize=(10, 3))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("top: original images | bottom: reconstructed images")
        plt.axis("off")
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()
    return filepath


def plot_interpolations(model, loader, device, filepath, steps=10):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        idx1 = (y == y[0]).nonzero(as_tuple=True)[0][0].item()
        idx2 = (y != y[0]).nonzero(as_tuple=True)[0][0].item()
        x1, x2 = x[idx1].unsqueeze(0).to(device), x[idx2].unsqueeze(0).to(device)

        z_mean1, _ = model.encode(x1.view(1, -1))
        z_mean2, _ = model.encode(x2.view(1, -1))

        alphas = torch.linspace(0, 1, steps, device=device)
        interp_z = []

        # vmf interp
        for alpha in alphas:
            z = (1 - alpha) * z_mean1 + alpha * z_mean2
            interp_z.append(torch.nn.functional.normalize(z, p=2, dim=-1))
        interp_z = torch.cat(interp_z, dim=0)

        x_recon_interp = torch.sigmoid(model.decoder(interp_z)).view(-1, 1, 28, 28)

        grid = tu.make_grid(x_recon_interp, nrow=steps, pad_value=0.5)
        plt.figure(figsize=(12, 2))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.title("latent space interpolation (vmf)")
        plt.axis("off")
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()
    return filepath


def plot_latent_space(model, loader, device, filepath, n_plot=1000):
    """generate t-sne visualization."""
    print(f"generating t-sne plot for {n_plot} points...")
    X_z, y = encode_dataset(model, loader, device)

    # reduce memory usage
    X_z, y = X_z[:n_plot], y[:n_plot]

    # use single perplexity to avoid memory issues
    perplexity = 30
    print(f"running t-sne with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    z_tsne = tsne.fit_transform(X_z)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        z_tsne[:, 0],
        z_tsne[:, 1],
        c=y,
        cmap=plt.get_cmap("tab10", 10),
        s=10,
        alpha=0.8,
    )
    plt.title("t-sne of latent space (μ) for vmf-vae")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    return filepath


def plot_pca_analysis(model, loader, device, filepath, n_plot=1000):
    """generate pca analysis: scatter + explained variance."""
    print(f"generating pca analysis for {n_plot} points...")
    X_z, y = encode_dataset(model, loader, device)
    X_z, y = X_z[:n_plot], y[:n_plot]

    pca = PCA(n_components=min(50, X_z.shape[1]))
    Z_pca = pca.fit_transform(X_z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sc = ax1.scatter(
        Z_pca[:, 0], Z_pca[:, 1], c=y, cmap=plt.get_cmap("tab10", 10), s=10, alpha=0.8
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax1.set_title("pca of latent space (μ) - vmf")
    plt.colorbar(sc, ax=ax1, ticks=np.unique(y))

    # explained variance plot
    n_components = min(20, len(pca.explained_variance_ratio_))
    ax2.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    ax2.set_xlabel("principal component")
    ax2.set_ylabel("explained variance ratio")
    ax2.set_title(
        f"pca explained variance\n(total: {pca.explained_variance_ratio_.sum():.1%})"
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    return filepath


def run(args):
    script_start_time = time.time()
    timing_results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data loading w/ dynamic binarization
    transform = transforms.Compose(
        [transforms.ToTensor(), BinarizeWithRandomThreshold()]
    )
    full_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    train_eval_loader = DataLoader(train_dataset, batch_size=1024)
    test_eval_loader = DataLoader(test_dataset, batch_size=1024)

    final_results = []
    knn_samples = [100, 600, 1000]
    logger = WandbLogger(args)

    for d_manifold in args.d_dims:
        print(f"\n{'='*30}\n== vMF MANIFOLD DIMENSION d = {d_manifold} ==\n{'='*30}")

        agg_results = {s: [] for s in knn_samples}

        # vMF: dist on sphere S^d is embedded in R^(d+1)
        model_z_dim = d_manifold + 1

        print(
            f"\n--- Testing vMF-VAE with d={d_manifold} (model z_dim={model_z_dim}) ---"
        )

        for run in range(args.n_runs):
            print(f"\n--- Run {run+1}/{args.n_runs} ---")
            run_start_time = time.time()

            # wandb setup
            if logger.use:
                logger.start_run(f"vmf-d{d_manifold}-run{run+1}", args)

            # model & optimizer
            model = MLPVAE(h_dim=args.h_dim, z_dim=model_z_dim, distribution="vmf").to(
                device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # training
            best_val_loss = float("inf")
            patience_counter = 0
            model_path = f"best_model_vmf_d{d_manifold}_run{run}.pt"
            train_start_time = time.time()

            for epoch in range(args.epochs):
                model.train()
                beta = min(
                    1.0, (epoch + 1) / max(1, args.warmup_epochs)
                )  # kl annealing
                total_train_loss = 0
                for x_mb, _ in train_loader:
                    optimizer.zero_grad()
                    loss = vae_loss(model, x_mb.to(device), beta=beta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += loss.item()

                # val
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for x_mb, _ in val_loader:
                        total_val_loss += vae_loss(
                            model, x_mb.to(device), beta=1.0
                        ).item()

                avg_val_loss = total_val_loss / len(val_loader)

                if logger.use:
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            "train_loss": total_train_loss / len(train_loader),
                            "val_loss": avg_val_loss,
                            "kl_beta": beta,
                        }
                    )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1

                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            train_time = time.time() - train_start_time
            print(f"training time for vmf-d{d_manifold}-run{run+1}: {train_time:.2f}s")

            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))

                eval_start_time = time.time()
                # knn eval
                knn_results = perform_knn_evaluation(
                    model, train_eval_loader, test_eval_loader, device, knn_samples
                )
                for n_samples in knn_samples:
                    if f"knn_acc_{n_samples}" in knn_results:
                        agg_results[n_samples].append(
                            knn_results[f"knn_acc_{n_samples}"]
                        )

                # fourier property testing of latent vectors
                fourier_pseudo = {}
                fourier_deconv = {}

                # test loader for fourier tests
                test_subset = torch.utils.data.Subset(
                    test_dataset, list(range(min(1000, len(test_dataset))))
                )
                test_subset_loader = DataLoader(test_subset, batch_size=64)
                fourier_pseudo = test_self_binding(
                    model,
                    test_subset_loader,
                    device,
                    f"visualizations/d_{d_manifold}/vmf",
                    unbind_method="*",
                )
                fourier_deconv = test_self_binding(
                    model,
                    test_subset_loader,
                    device,
                    f"visualizations/d_{d_manifold}/vmf",
                    unbind_method="†",
                )

                # visualizations
                vis_dir = f"visualizations/d_{d_manifold}/vmf"
                os.makedirs(vis_dir, exist_ok=True)
                print(f"Generating visualizations in {vis_dir}...")

                recon_path = plot_reconstructions(
                    model,
                    test_eval_loader,
                    device,
                    os.path.join(vis_dir, "reconstructions.png"),
                )
                tsne_path = plot_latent_space(
                    model,
                    test_eval_loader,
                    device,
                    os.path.join(vis_dir, "tsne.png"),
                )
                pca_path = plot_pca_analysis(
                    model,
                    test_eval_loader,
                    device,
                    os.path.join(vis_dir, "pca.png"),
                )
                interp_path = plot_interpolations(
                    model,
                    test_eval_loader,
                    device,
                    os.path.join(vis_dir, "interpolations.png"),
                )

                # manifold visualization for vmf
                if d_manifold >= 2:
                    vmf_viz = plot_powerspherical_manifold_visualization(
                        model, device, vis_dir, n_samples=1000, dims=(0, 1)
                    )

                if logger.use:
                    images_to_log = {
                        "Reconstructions": recon_path,
                        "Latent t-SNE": tsne_path,
                        "Latent PCA": pca_path,
                        "Interpolations": interp_path,
                    }
                    for tag, fr in {
                        "*": fourier_pseudo,
                        "†": fourier_deconv,
                    }.items():
                        if fr.get("similarity_after_k_binds_plot_path"):
                            images_to_log[f"Similarity_After_K_Binds_{tag}"] = fr[
                                "similarity_after_k_binds_plot_path"
                            ]

                    # add manifold visualization to wandb
                    if d_manifold >= 2 and vmf_viz:
                        images_to_log["vMF_Manifold"] = vmf_viz

                    logger.log_images(images_to_log)

                if logger.use:
                    # wandb logging k-NN as metrics
                    knn_metrics = {
                        k: v for k, v in knn_results.items() if k.startswith("knn_")
                    }
                    fourier_metrics = {}
                    fourier_metrics.update(
                        {
                            f"pseudo/{k}": v
                            for k, v in fourier_pseudo.items()
                            if isinstance(v, (int, float, bool))
                        }
                    )
                    fourier_metrics.update(
                        {
                            f"deconv/{k}": v
                            for k, v in fourier_deconv.items()
                            if isinstance(v, (int, float, bool))
                        }
                    )

                    # compute class means and mean vector cosine accuracy
                    train_subset = torch.utils.data.Subset(
                        train_dataset, list(range(min(5000, len(train_dataset))))
                    )
                    train_subset_loader = DataLoader(train_subset, batch_size=256)
                    class_means = compute_class_means(
                        model, train_subset_loader, device, max_per_class=1000
                    )
                    mean_vector_acc, per_class_acc = evaluate_mean_vector_cosine(
                        model, test_eval_loader, device, class_means
                    )

                    logger.log_metrics(
                        {
                            **knn_metrics,
                            **fourier_metrics,
                            "mean_vector_cosine_acc": float(mean_vector_acc),
                            "final_val_loss": best_val_loss,
                        }
                    )

                    summary_metrics = {
                        **knn_metrics,
                        "final_val_loss": best_val_loss,
                        **fourier_metrics,
                        "mean_vector_cosine_acc": float(mean_vector_acc),
                    }
                    logger.log_summary(summary_metrics)
                    logger.finish_run()

                eval_time = time.time() - eval_start_time
                run_time = time.time() - run_start_time

                # store timing info
                timing_key = f"vmf_d{d_manifold}_run{run+1}"
                timing_results[timing_key] = {
                    "train_time_s": train_time,
                    "eval_time_s": eval_time,
                    "total_run_time_s": run_time,
                }
                print(f"eval time: {eval_time:.2f}s, total run time: {run_time:.2f}s")

                os.remove(model_path)

        row_data = {"d": d_manifold}
        for n_samples in knn_samples:
            accuracies = agg_results[n_samples]
            if accuracies:
                mean_acc, std_acc = np.mean(accuracies) * 100, np.std(accuracies) * 100
                row_data[f"vMF_{n_samples}"] = f"{mean_acc:.1f}±{std_acc:.1f}"
            else:
                row_data[f"vMF_{n_samples}"] = "N/A"
        final_results.append(row_data)

    if final_results:
        import pandas as pd

        df = pd.DataFrame(final_results).set_index("d")
        print("\n" + "=" * 25 + " FINAL vMF RESULTS (k-NN Accuracy %) " + "=" * 25)
        print(df.to_string())
        df.to_csv("mnist_vmf_knn_results.csv")
    else:
        print("No results were generated.")

    # save timing results
    script_total_time = time.time() - script_start_time
    timing_results["total_script_time_s"] = script_total_time
    with open("mnist_vmf_timing.json", "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\ntotal script execution time: {script_total_time:.2f}s")
    print(f"timing results saved to mnist_vmf_timing.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vMF-VAE experiments on MNIST.")

    parser.add_argument(
        "--d_dims",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 40],
        help="Latent manifold dimensions to test",
    )
    parser.add_argument("--h_dim", type=int, default=128, help="Hidden layer size")

    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=100, help="KL annealing warmup epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )  # modified for greater stability at d=40
    parser.add_argument(
        "--l2_norm",
        action="store_true",
        help="L2 normalize latents (not typically used for vMF)",
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        default=20,
        help="Number of runs for statistical averaging",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--wandb_project", type=str, default="aug-19-mnistvmf", help="W&B project name"
    )

    args = parser.parse_args()
    run(args)

import math
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import WandbLogger, test_fourier_properties
from mnist.mlp_vae import MLPVAE, vae_loss


class BinarizeWithRandomThreshold:
    """Binarizes a tensor by comparing each element to a random value."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x > torch.rand_like(x)).float()


def encode_dataset(model, loader, device):
    """Encodes a full dataset into latent mean representations."""
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
    print("Performing k-NN evaluation...")
    X_train_full, y_train_full = encode_dataset(model, train_loader, device)
    X_test, y_test = encode_dataset(model, test_loader, device)
    
    results = {}
    metric = "cosine"  # vMF uses cosine metric
    
    for n_samples in n_samples_list:
        if n_samples > len(X_train_full):
            print(f"Warning: k-NN sample size {n_samples} > training data size {len(X_train_full)}. Skipping.")
            continue

        indices = np.random.choice(len(X_train_full), n_samples, replace=False)
        X_train_sample, y_train_sample = X_train_full[indices], y_train_full[indices]

        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train_sample, y_train_sample)
        
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[n_samples] = accuracy
        print(f"  k-NN Accuracy with {n_samples} training samples: {accuracy:.4f}")
        
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
        
        import torchvision
        grid = torchvision.utils.make_grid(comparison, nrow=8, pad_value=0.5)
        
        plt.figure(figsize=(10, 3))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Top: Original Images | Bottom: Reconstructed Images")
        plt.axis('off')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
        
        # vMF interp
        for alpha in alphas:
            z = (1 - alpha) * z_mean1 + alpha * z_mean2
            interp_z.append(torch.nn.functional.normalize(z, p=2, dim=-1))
        interp_z = torch.cat(interp_z, dim=0)

        x_recon_interp = torch.sigmoid(model.decoder(interp_z)).view(-1, 1, 28, 28)
        
        import torchvision
        grid = torchvision.utils.make_grid(x_recon_interp, nrow=steps, pad_value=0.5)
        plt.figure(figsize=(12, 2))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.title(f"Latent Space Interpolation (vMF)")
        plt.axis('off')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    return filepath


def plot_latent_space(model, loader, device, filepath, n_plot=2000):
    """Generate t-SNE visualization grid with multiple perplexity values."""
    print(f"Generating t-SNE plot grid for {n_plot} points...")
    X_z, y = encode_dataset(model, loader, device)
    
    X_z, y = X_z[:n_plot], y[:n_plot]

    perplexities = [5, 15, 30, 50, 100]
    n_perplexities = len(perplexities)
    
    fig, axes = plt.subplots(1, n_perplexities, figsize=(n_perplexities * 5, 5))
    if n_perplexities == 1: # this makes sure axes is a list
        axes = [axes]

    for i, p in enumerate(perplexities):
        ax = axes[i]
        print(f"  Running t-SNE with perplexity={p}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=p, max_iter=1000)
        z_tsne = tsne.fit_transform(X_z)
        
        ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap=plt.get_cmap("tab10", 10), s=10, alpha=0.8)
        ax.set_title(f"Perplexity: {p}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"t-SNE of Latent Space (μ) for vMF-VAE", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_pca_analysis(model, loader, device, filepath, n_plot=2000):
    """Generate PCA analysis: scatter + explained variance."""
    from sklearn.decomposition import PCA
    print(f"Generating PCA analysis for {n_plot} points...")
    X_z, y = encode_dataset(model, loader, device)
    X_z, y = X_z[:n_plot], y[:n_plot]
    
    pca = PCA(n_components=min(50, X_z.shape[1]))
    Z_pca = pca.fit_transform(X_z)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sc = ax1.scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, cmap=plt.get_cmap("tab10", 10), s=10, alpha=0.8)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title("PCA of Latent Space (μ) - vMF")
    plt.colorbar(sc, ax=ax1, ticks=np.unique(y))
    
    # explained variance plot
    n_components = min(20, len(pca.explained_variance_ratio_))
    ax2.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title(f'PCA Explained Variance\n(Total: {pca.explained_variance_ratio_.sum():.1%})')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data loading w/ dynamic binarization 
    transform = transforms.Compose([
        transforms.ToTensor(),
        BinarizeWithRandomThreshold()
    ])
    full_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    train_eval_loader = DataLoader(train_dataset, batch_size=1024)
    test_eval_loader = DataLoader(test_dataset, batch_size=1024)
    
    final_results = []
    knn_samples = [100, 600, 1000, 2048]
    logger = WandbLogger(args)

    for d_manifold in args.d_dims:
        print(f"\n{'='*30}\n== vMF MANIFOLD DIMENSION d = {d_manifold} ==\n{'='*30}")
        
        agg_results = {s: [] for s in knn_samples}

        # vMF: dist on sphere S^d is embedded in R^(d+1)
        model_z_dim = d_manifold + 1

        print(f"\n--- Testing vMF-VAE with d={d_manifold} (model z_dim={model_z_dim}) ---")

        for run in range(args.n_runs):
            print(f"\n--- Run {run+1}/{args.n_runs} ---")
            
            # wandb setup
            if logger.use:
                logger.start_run(f"vmf-d{d_manifold}-run{run+1}", args)

            # model & optimizer
            model = MLPVAE(h_dim=args.h_dim, z_dim=model_z_dim, distribution="vmf").to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # training
            best_val_loss = float("inf")
            patience_counter = 0
            model_path = f"best_model_vmf_d{d_manifold}_run{run}.pt"

            for epoch in range(args.epochs):
                model.train()
                beta = min(1.0, (epoch + 1) / max(1, args.warmup_epochs))  # kl annealing
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
                        total_val_loss += vae_loss(model, x_mb.to(device), beta=1.0).item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                
                if logger.use:
                    logger.log_metrics({
                        "epoch": epoch,
                        "train_loss": total_train_loss / len(train_loader),
                        "val_loss": avg_val_loss,
                        "kl_beta": beta
                    })

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                
                # knn eval
                knn_accuracies = perform_knn_evaluation(
                    model, train_eval_loader, test_eval_loader, device, knn_samples
                )
                for n_samples, acc in knn_accuracies.items():
                    agg_results[n_samples].append(acc)

                # fourier property testing of latent vectors
                fourier_results = {}

                # test loader for fourier tests
                test_subset = torch.utils.data.Subset(test_dataset, list(range(min(1000, len(test_dataset)))))
                test_subset_loader = DataLoader(test_subset, batch_size=64)
                fourier_results = test_fourier_properties(model, test_subset_loader, device, f"visualizations/d_{d_manifold}/vmf")

                # visualizations
                if args.visualize or logger.use:
                    vis_dir = f"visualizations/d_{d_manifold}/vmf"
                    os.makedirs(vis_dir, exist_ok=True)
                    print(f"Generating visualizations in {vis_dir}...")
                    
                    recon_path = plot_reconstructions(model, test_eval_loader, device, os.path.join(vis_dir, "reconstructions.png"))
                    tsne_path = plot_latent_space(model, test_eval_loader, device, os.path.join(vis_dir, "tsne.png"))
                    pca_path = plot_pca_analysis(model, test_eval_loader, device, os.path.join(vis_dir, "pca.png"))
                    interp_path = plot_interpolations(model, test_eval_loader, device, os.path.join(vis_dir, "interpolations.png"))
                    
                    if logger.use:
                        images_to_log = {
                            "Reconstructions": recon_path,
                            "Latent t-SNE": tsne_path,
                            "Latent PCA": pca_path,
                            "Interpolations": interp_path
                        }
                        if fourier_results and "fft_spectrum_plot_path" in fourier_results:
                            images_to_log["Fourier_Analysis"] = fourier_results["fft_spectrum_plot_path"]
                        logger.log_images(images_to_log)

                if logger.use:
                    # wandb logging k-NN as metrics (for charts/visualization)
                    knn_metrics = {f"knn_acc_{k}": v for k, v in knn_accuracies.items()}
                    fourier_metrics = {k: v for k, v in fourier_results.items() if isinstance(v, (int, float, bool))}
                    
                    logger.log_metrics({
                        **knn_metrics,
                        **fourier_metrics,
                        "final_val_loss": best_val_loss,
                    })
                    
                    summary_metrics = {
                        **knn_metrics,
                        "final_val_loss": best_val_loss,
                        **fourier_results
                    }
                    logger.log_summary(summary_metrics)
                    logger.finish_run()
                
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
        print("\n" + "="*25 + " FINAL vMF RESULTS (k-NN Accuracy %) " + "="*25)
        print(df.to_string())
        df.to_csv("mnist_vmf_knn_results.csv")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vMF-VAE experiments on MNIST.")
    
    parser.add_argument('--d_dims', type=int, nargs='+', default=[2, 5, 10, 20, 40], help='Latent manifold dimensions to test')
    parser.add_argument('--h_dim', type=int, default=128, help='Hidden layer size')
    
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (0 to disable)')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='KL annealing warmup epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate') # modified for greater stability at d=40

    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs for statistical averaging')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='aug-17-vmfvae', help='W&B project name')
    
    args = parser.parse_args()
    run(args)



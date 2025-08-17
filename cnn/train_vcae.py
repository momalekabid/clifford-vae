import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.models import VAE
from utils.wandb_utils import WandbLogger, test_fourier_properties


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
    import torchvision

    grid = torch.cat([x.cpu(), x_recon.cpu()], dim=0)
    grid = (grid * 0.5 + 0.5).clamp(0, 1)
    torchvision.utils.save_image(grid, path, nrow=n_images)
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
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    pts = tsne.fit_transform(Z)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=Y, cmap="tab10", s=8, alpha=0.8)
    plt.colorbar(sc, ticks=np.unique(Y))
    plt.title("t-SNE of Latent Means (μ)")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_pca_plot(model, loader, device, path, n_samples=2000):
    from sklearn.decomposition import PCA
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA scatter plot (first 2 components)
    sc = ax1.scatter(Z_pca[:, 0], Z_pca[:, 1], c=Y, cmap="tab10", s=8, alpha=0.8)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title("PCA of Latent Means (μ)")
    plt.colorbar(sc, ax=ax1, ticks=np.unique(Y))
    
    # explained variance plot
    n_components = min(20, len(pca.explained_variance_ratio_))
    ax2.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title(f'PCA Explained Variance\n(Total: {pca.explained_variance_ratio_.sum():.1%})')
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def perform_knn_evaluation(model, train_loader, test_loader, device, n_samples_list=[100, 600, 1000, 2048]):
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
    metric = "cosine" if getattr(model, 'distribution', None) in ["powerspherical", "clifford"] else "euclidean"
    
    results = {}
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
        f1 = f1_score(y_test, y_pred, average="macro")
        
        results[f"knn_acc_{n_samples}"] = float(accuracy)
        results[f"knn_f1_{n_samples}"] = float(f1)
        print(f"knn accuracy with {n_samples} training samples: {accuracy:.4f}")
        
    return results


def main(args):
    print(f"Device: {DEVICE}")
    logger = WandbLogger(args)

    latent_dims = [128, 1024, 4096]
    distributions = ["powerspherical", "clifford", "gaussian"]
    datasets_to_test = ["fashionmnist"]
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
                    use_perceptual_loss=args.use_perceptual,
                    l1_weight=args.l1_weight,
                    freq_weight=args.freq_weight,
                )
                logger.watch_model(model)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr)
                best = float("inf")

                for epoch in range(args.epochs):
                    beta = min(1.0, (epoch + 1) / args.warmup_epochs) * args.max_beta
                    train_losses = train_epoch(
                        model, train_loader, optimizer, DEVICE, beta
                    )
                    test_losses = test_epoch(model, test_loader, DEVICE)
                    val = test_losses["test/total_loss"]
                    if np.isfinite(val) and val < best:
                        best = val
                        torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            **train_losses,
                            **test_losses,
                            "best_test_loss": best,
                            "beta": beta,
                        }
                    )

                print(f"best loss: {best:.4f}")

                if os.path.exists(f"{output_dir}/best_model.pt"):
                    model.load_state_dict(
                        torch.load(f"{output_dir}/best_model.pt", map_location=DEVICE)
                    )

                    # fourier property testing of latent vectors
                    fourier = test_fourier_properties(
                        model, test_loader, DEVICE, output_dir
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
                        model, train_loader, test_loader, DEVICE, [100, 600, 1000, 2048]
                    )

                    images = {
                        "fft_spectrum": fourier.pop("fft_spectrum_plot_path"),
                        "reconstructions": recon_path,
                        "tsne": tsne_path,
                        "pca": pca_path,
                    }
                    summary = {
                        "final_best_loss": best,
                        **fourier,
                        **knn_metrics,
                    }
                    logger.log_summary(summary)
                    logger.log_images(images)

                logger.finish_run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_loss", type=str, default="l1_freq", choices=["mse", "l1_freq"])
    p.add_argument("--use_perceptual", action="store_true", help="Use perceptual LPIPS loss")
    p.add_argument("--l1_weight", type=float, default=1.0, help="Weight for L1 pixel loss")
    p.add_argument("--freq_weight", type=float, default=0.25, help="Weight for frequency domain loss")
    p.add_argument("--max_beta", type=float, default=1.0)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="aug-17-vcae")
    args = p.parse_args()
    main(args)




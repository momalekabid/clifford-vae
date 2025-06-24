import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from clifford import ModelVAE, compute_loss

# params from https://arxiv.org/abs/1804.00891
# patience/delta hand-picked
H_DIM = 128
Z_DIM = 10
BATCH_SIZE = 64
EPOCHS = 100
KNN_EVAL_SAMPLES = [100, 600, 1000]
N_RUNS = 3
Z_DIMS = [2, 5, 10, 20, 40, 64, 128]  # , 256]
PATIENCE = 0.5
DELTA = 5


### dataset & cuda/mps init
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: (x > torch.rand_like(x)).float()
        ),  # dynamic binarization
    ]
)

dataset = datasets.MNIST("../datasets", train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = datasets.MNIST(
    "../datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Using device: {device}")


def get_fig_path(z_dim, name):
    path = f"../results/figures/clifford_vae/{z_dim}"
    os.makedirs(path, exist_ok=True)
    return f"{path}/{name}.png"


def encode_dataset(model, data_loader, device):
    """Get latent representations for entire dataset"""
    model.eval()
    all_z = []
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            all_z.append(mu.cpu())
            all_labels.append(labels)

    return torch.cat(all_z, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def vis_random_samples(model, data_loader, device, n=8, save=None):
    """
    Plot n random test images and their reconstructions side by side.
    """
    model.eval()
    with torch.no_grad():
        # grab first batch
        data, _ = next(iter(data_loader))
        data = data.to(device)
        # unpack model outputs :: (mu,kappa),(q_z,p_z),z,recon
        _, _, _, recon = model(data)
        # detach and move to CPU
        orig = data[:n].cpu().detach()
        recon = recon[:n].cpu().detach()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(orig[i].view(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(torch.sigmoid(recon[i]).view(28, 28), cmap="gray")
        axes[1, i].axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    plt.close()


def vis_tsne(
    model, data_loader, device, n=1000, perplexity=30, max_iter=1000, save=None
):
    """
    t-SNE of the first n latent means
    """
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            mu = model.encode(x.view(x.size(0), -1))[0]
            latents.append(mu.cpu().detach().numpy())
            labels.append(y.numpy())
            if sum(len(l) for l in labels) >= n:
                break
    latents = np.concatenate(latents, axis=0)[:n]
    labels = np.concatenate(labels, axis=0)[:n]

    tsne = TSNE(
        n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42
    )
    zs = tsne.fit_transform(latents)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap="tab10", s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE of Latent Space")
    if save:
        plt.savefig(save, dpi=200)
    plt.close()


def vis_interp(model, data_loader, device, steps=10, save=None):
    """
    Linear interpolation between two test samples and plot.
    """
    model.eval()
    with torch.no_grad():
        # pick two (different?) classes
        # definitely a better way to do this e.g so we know which classes in question we sample
        data, labels = next(iter(data_loader))
        data = data.to(device)
        # find two indices
        idx1 = (labels == labels[0]).nonzero()[0].item()
        idx2 = (labels != labels[0]).nonzero()[0].item()
        x1, x2 = data[idx1], data[idx2]
        # encode
        mu1, _ = model.encode(x1.view(1, -1))
        mu2, _ = model.encode(x2.view(1, -1))
        # interpolate in latent
        interps = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decoder(z).cpu().detach()
            interps.append(torch.sigmoid(recon).view(28, 28).numpy())

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
    for i in range(steps):
        axes[i].imshow(interps[i], cmap="gray")
        axes[i].axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
    plt.close()


def perform_knn_evaluation(model, train_loader, test_loader, device):
    X_train, y_train = encode_dataset(model, train_loader, device)
    results = {}

    for n_samples in KNN_EVAL_SAMPLES:
        test_subset = torch.utils.data.Subset(
            test_loader.dataset,
            indices=torch.randperm(len(test_loader.dataset))[:n_samples],
        )
        test_loader_subset = DataLoader(test_subset, batch_size=BATCH_SIZE)

        X_test, y_test = encode_dataset(model, test_loader_subset, device)

        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[n_samples] = accuracy

    return results


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    device,
    epochs=EPOCHS,
    z_dim=None,
):
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / 50, 1.0))
    # the original paper claims a linear warmup for 100 epochs...
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            (_, _), (q_z, p_z), _, x_recon = model(data)
            loss = compute_loss(model, data)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                (_, _), (q_z, p_z), _, x_recon = model(data)
                loss = compute_loss(model, data)
                total_loss += loss.item()

        val_loss = total_loss / len(val_loader)
        scheduler.step()

        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss - DELTA:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return perform_knn_evaluation(model, train_loader, test_loader, device)


def run_experiment(z_dim, device, n_runs=N_RUNS):
    results = {samples: [] for samples in KNN_EVAL_SAMPLES}

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs} for z_dim = {z_dim}")
        model = ModelVAE(H_DIM, z_dim, device=device).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)

        run_results = train_and_evaluate(
            model, train_loader, val_loader, test_loader, optimizer, device, z_dim=z_dim
        )
        vis_random_samples(
            model, test_loader, device, save=get_fig_path(z_dim, "random")
        )

        vis_tsne(model, test_loader, device, n=1500, save=get_fig_path(z_dim, "tsne"))

        vis_interp(model, test_loader, device, save=get_fig_path(z_dim, "interpolated"))
        for n_samples, accuracy in run_results.items():
            results[n_samples].append(accuracy)

    return results


def calculate_statistics(results):
    return {
        n_samples: (np.mean(accuracies) * 100, np.std(accuracies) * 100)
        for n_samples, accuracies in results.items()
    }


if __name__ == "__main__":
    results_table = []

    for z_dim in Z_DIMS:
        print(f"\nRunning experiments for z_dim = {z_dim}")
        results = run_experiment(z_dim, device)
        stats = calculate_statistics(results)

        for n_samples in KNN_EVAL_SAMPLES:
            mean, std = stats[n_samples]
            results_table.append(
                {
                    "d": z_dim,
                    "n_samples": n_samples,
                    "Clifford-VAE": f"{mean:.1f}±{std:.1f}",
                }
            )
            print(f"d={z_dim}, n_samples={n_samples}: {mean:.1f}±{std:.1f}")

    df = pd.DataFrame(results_table)
    df = df.pivot(index="d", columns="n_samples", values="Clifford-VAE")

    print("\nFinal Results:")
    print(df.to_string())
    df.to_csv("../../results/clifford_vae_results.csv")

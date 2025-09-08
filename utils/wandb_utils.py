import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import wandb
except Exception:
    wandb = None




class WandbLogger:
    def __init__(self, args):
        self.use = (wandb is not None) and (not args.no_wandb)
        if self.use:
            self.project = args.wandb_project
            self.run = None

    def start_run(self, name, args):
        if self.use:
            self.run = wandb.init(project=self.project, name=name, config=vars(args))

    def watch_model(self, model):
        if self.use:
            wandb.watch(model, log="gradients", log_freq=100)

    def log_metrics(self, d):
        if self.use and self.run is not None:
            try:
                self.run.log(d)
            except Exception:
                pass

    def log_summary(self, d):
        if self.use and self.run is not None:
            try:
                self.run.summary.update(d)
            except Exception:
                pass

    def log_images(self, images):
        if self.use and self.run is not None:
            try:
                to_log = {}
                for k, v in images.items():
                    if isinstance(v, str) and os.path.exists(v):
                        to_log[k] = wandb.Image(v)
                    else:
                        to_log[k] = v
                self.run.log(to_log)
            except Exception:
                pass

    def finish_run(self):
        if self.use and self.run is not None:
            self.run.finish()


def _extract_latent_mu(model, x: torch.Tensor):
    out = model(x)
    if isinstance(out, (tuple, list)):
        if len(out) == 4 and isinstance(out[0], tuple):
            (z_mean, _), _, _, _ = out
            return z_mean
        elif len(out) == 4:
            _, _, _, mu = out
            return mu
        else:
            return out[-1]
    return out



@torch.no_grad()
def compute_class_means(model, loader, device, max_per_class: int = 1000):
    model.eval()
    sums = {}
    counts = {}
    dist_type = getattr(model, "distribution", "normal")

    for x, y in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)
        mu = mu.detach()
        for i, label in enumerate(y.tolist()):
            if label not in counts:
                counts[label] = 0
                sums[label] = torch.zeros_like(mu[i])
            if counts[label] < max_per_class:
                sums[label] = sums[label] + mu[i]
                counts[label] += 1

    means = {}
    for label, total in sums.items():
        c = max(1, min(counts[label], 10))
        vec = total / c
        if dist_type == "powerspherical":
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1) # should already be normalized, just in case (unit-sphere)
        means[label] = vec
    return means


@torch.no_grad()
def evaluate_mean_vector_cosine(model, loader, device, class_means: dict):
    model.eval()
    labels_sorted = sorted(class_means.keys())
    mean_vector = torch.stack([class_means[k] for k in labels_sorted], dim=0).to(device)
    dist_type = getattr(model, "distribution", "normal")

    correct = 0
    total = 0
    per_class_correct = {k: 0 for k in labels_sorted}
    per_class_total = {k: 0 for k in labels_sorted}

    for x, y in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)

        if dist_type == "powerspherical":
            mu_norm = torch.nn.functional.normalize(mu, p=2, dim=-1)
            mean_norm = torch.nn.functional.normalize(mean_vector, p=2, dim=-1)
            sims = torch.nn.functional.cosine_similarity(
                mu_norm.unsqueeze(1), mean_norm.unsqueeze(0), dim=-1
            )
            preds = sims.argmax(dim=1).cpu()
        elif dist_type == "clifford":
            sims = torch.nn.functional.cosine_similarity(
                mu.unsqueeze(1), mean_vector.unsqueeze(0), dim=-1
            )
            preds = sims.argmax(dim=1).cpu()
        else:
            dists = torch.cdist(mu, mean_vector, p=2)
            preds = dists.argmin(dim=1).cpu()

        y_cpu = y.cpu()
        correct += (preds == y_cpu).sum().item()
        total += y_cpu.numel()
        for yi, pi in zip(y_cpu.tolist(), preds.tolist()):
            per_class_total[yi] += 1
            if yi == labels_sorted[pi]:
                per_class_correct[yi] += 1

    acc = correct / max(1, total)
    per_class_acc = {
        k: (per_class_correct[k] / max(1, per_class_total[k])) for k in labels_sorted
    }
    return acc, per_class_acc





@torch.no_grad()
def plot_clifford_torus_latent_scatter(model, loader, device, output_dir, dims=(0, 1), dataset_name: str = None):
    if getattr(model, "distribution", None) != "clifford" or model.latent_dim < 2:
        return None

    model.eval()
    angles = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            _, _, _, mu = out
        else:
            mu = out
        a = ((mu + math.pi) % (2 * math.pi)) - math.pi
        angles.append(a.detach().cpu())
        labels.append(y)
        if len(torch.cat(labels)) >= 4000:
            break

    A = torch.cat(angles, 0)
    Y = torch.cat(labels, 0).numpy()
    ax0, ax1 = dims
    xs = A[:, ax0].numpy()
    ys = A[:, ax1].numpy()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"clifford_torus_latent_scatter_{dataset_name or 'dataset'}.png")
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(xs, ys, c=Y, cmap="tab10", s=6, alpha=0.8)
    plt.colorbar(sc)
    plt.xlim(-math.pi, math.pi)
    plt.ylim(-math.pi, math.pi)
    plt.xlabel(f"angle[{ax0}]")
    plt.ylabel(f"angle[{ax1}]")
    plt.title("Clifford Torus Latent Angles")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _angles_to_clifford_vector(angles: torch.Tensor, normalize_ifft: bool = True) -> torch.Tensor:
    # angles shape (..., d), produce (..., 2d) real vector following CliffordPowerSphericalDistribution mapping
    d = angles.shape[-1]
    n = 2 * d
    device = angles.device
    dtype = angles.dtype
    theta_s = torch.zeros((*angles.shape[:-1], n), device=device, dtype=dtype)
    if d > 1:
        theta_s[..., 1:d] = angles[..., 1:]
        theta_s[..., -d + 1 :] = -torch.flip(angles[..., 1:], (-1,))
    samples_c = torch.exp(1j * theta_s)
    if normalize_ifft:
        samples_c = samples_c / math.sqrt(n)
        return torch.fft.ifft(samples_c, dim=-1, norm="ortho").real
    return torch.fft.ifft(samples_c, dim=-1).real


@torch.no_grad()
def plot_clifford_torus_recon_grid(model, device, output_dir, dims=(0, 1), n_grid: int = 16):
    if getattr(model, "distribution", None) != "clifford" or model.latent_dim < 2:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clifford_torus_recon_grid.png")

    angles0 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    angles1 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    mesh0, mesh1 = torch.meshgrid(angles0, angles1, indexing="ij")
    A = torch.zeros(n_grid * n_grid, model.latent_dim, device=device)
    A[:, dims[0]] = mesh0.reshape(-1)
    A[:, dims[1]] = mesh1.reshape(-1)

    Z = _angles_to_clifford_vector(A, normalize_ifft=True)

    model.eval()
    imgs = model.decoder(Z)
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'output_activation'):
        if model.decoder.output_activation == 'sigmoid':
            imgs = torch.sigmoid(imgs)
        elif model.decoder.output_activation == 'tanh':
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        else:
            imgs = imgs.clamp(0, 1)
    else:
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu()
    # make grid
    H = n_grid
    W = n_grid
    C = imgs.shape[1]
    h, w = imgs.shape[-2:]
    canvas = torch.zeros(C, H * h, W * w)
    for i in range(H):
        for j in range(W):
            canvas[:, i * h : (i + 1) * h, j * w : (j + 1) * w] = imgs[i * W + j]
    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.title("Decoder Reconstructions over Torus Grid")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path




def plot_clifford_manifold_visualization(model, device, output_dir, n_samples=1000, dims=(0, 1)):
    """Clifford manifold visualization. Samples angles from [-pi, pi] and converts to Clifford vectors."""
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    if getattr(model, "distribution", None) != "clifford" or latent_dim is None or latent_dim < 2:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clifford_manifold_visualization.png")
    
    model.eval()
    with torch.no_grad():
        angles = torch.rand(n_samples, latent_dim, device=device) * 2 * math.pi - math.pi
        
        # convert angles to Clifford vectors using the same method as interpolation
        n = 2 * latent_dim
        theta_s = torch.zeros(n_samples, n, device=device, dtype=angles.dtype)
        if latent_dim > 1:
            theta_s[:, 1:latent_dim] = angles[:, 1:]
            theta_s[:, -latent_dim + 1:] = -torch.flip(angles[:, 1:], dims=(-1,))
        
        samples_complex = torch.exp(1j * theta_s)
        Z = torch.fft.ifft(samples_complex, dim=-1, norm="ortho").real.to(torch.float32)
        
        x_recon = model.decoder(Z)
        
        # sigmoid for our MNIST model
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'output_activation'):
            if model.decoder.output_activation == 'sigmoid':
                x_recon = torch.sigmoid(x_recon)
            elif model.decoder.output_activation == 'tanh':
                x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
            else:
                x_recon = x_recon.clamp(0, 1)
        else:
            x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
    
    n_grid = int(math.sqrt(n_samples))
    if n_grid * n_grid > n_samples:
        n_grid = int(math.sqrt(n_samples))
    
    grid_size = min(n_grid, 16)  # limit grid size for visibility
    x_recon_grid = x_recon[:grid_size * grid_size]
    
    # Create grid layout
    C = x_recon_grid.shape[1]
    h, w = x_recon_grid.shape[-2:]
    canvas = torch.zeros(C, grid_size * h, grid_size * w)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(x_recon_grid):
                canvas[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = x_recon_grid[idx]
    
    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.title("Clifford Manifold Reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return path


def plot_powerspherical_manifold_visualization(model, device, output_dir, n_samples=1000, dims=(0, 1)):
    """Visualize PowerSpherical manifold latent space."""
    # Check for both latent_dim and z_dim attributes
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    if getattr(model, "distribution", None) != "powerspherical" or latent_dim is None or latent_dim < 2:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "powerspherical_manifold_visualization.png")
    
    # sample points from the hypersphere
    model.eval()
    with torch.no_grad():
        # random samples on the hypersphere
        z = torch.randn(n_samples, latent_dim, device=device)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        
        # decode to see reconstructions
        x_recon = model.decoder(z)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
    
    # visualization grid
    n_grid = int(math.sqrt(n_samples))
    if n_grid * n_grid > n_samples:
        n_grid = int(math.sqrt(n_samples))
    
    # grid
    grid_size = min(n_grid, 16)  # limit grid size for visibility
    x_recon_grid = x_recon[:grid_size * grid_size]
    
    # grid
    C = x_recon_grid.shape[1]
    h, w = x_recon_grid.shape[-2:]
    canvas = torch.zeros(C, grid_size * h, grid_size * w)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(x_recon_grid):
                canvas[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = x_recon_grid[idx]
    
    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.title("PowerSpherical Manifold Reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return path


def plot_gaussian_manifold_visualization(model, device, output_dir, n_samples=1000, dims=(0, 1)):
    """Visualize Gaussian manifold latent space."""
    latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim", None))
    if getattr(model, "distribution", None) != "gaussian" or latent_dim is None or latent_dim < 2:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gaussian_manifold_visualization.png")
    
    # sample points from the Gaussian distribution
    model.eval()
    with torch.no_grad():
        # random samples from standard normal
        z = torch.randn(n_samples, latent_dim, device=device)
        
        x_recon = model.decoder(z)
        x_recon = (x_recon * 0.5 + 0.5).clamp(0, 1)
    
    # visualization grid
    n_grid = int(math.sqrt(n_samples))
    if n_grid * n_grid > n_samples:
        n_grid = int(math.sqrt(n_samples))
    
    grid_size = min(n_grid, 16)  # limit grid size for visibility
    x_recon_grid = x_recon[:grid_size * grid_size]
    
    C = x_recon_grid.shape[1]
    h, w = x_recon_grid.shape[-2:]
    canvas = torch.zeros(C, grid_size * h, grid_size * w)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(x_recon_grid):
                canvas[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = x_recon_grid[idx]
    
    plt.figure(figsize=(8, 8))
    if C == 1:
        plt.imshow(canvas.squeeze(0).cpu().numpy(), cmap="gray")
    else:
        plt.imshow(canvas.permute(1, 2, 0).cpu().numpy())
    
    plt.xticks([])
    plt.yticks([])
    plt.title("Gaussian Manifold Reconstructions")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return path

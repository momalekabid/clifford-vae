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


def _unit_magnitude_fraction(F: torch.Tensor, tol: float = 0.05) -> float:
    mags = torch.abs(F)
    return float((torch.abs(mags - 1.0) < tol).float().mean().item())


def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft(
        torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1
    ).real


def _unbind(ab: torch.Tensor, b: torch.Tensor, method: str = "pseudo") -> torch.Tensor:
    """
      - pseudo: x̂ = (ab) ⊛ a^{-1}, where a^{-1} = (a_n, ..., a_2, a_1)  
      - deconv: x̂ = IFFT( FFT(ab) / FFT(a) ) 
    """
    if method == "pseudo":
        return _bind(ab, vsa_invert(b))
    elif method == "deconv":
        Fab = torch.fft.fft(ab, dim=-1)
        Fb = torch.fft.fft(b, dim=-1)
        rec = torch.fft.ifft(Fab / (Fb + 1e-12), dim=-1).real
        return rec

def _fft_make_unitary(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Fx = torch.fft.fft(x, dim=-1)
    mags = torch.abs(Fx)
    Fx_unit = Fx / torch.clamp(mags, min=eps)
    return torch.fft.ifft(Fx_unit, dim=-1).real



def test_fourier_properties(
    model,
    loader,
    device,
    output_dir,
    k_self_bind: int = 50,
    unbind_method: str = "pseudo",
):
    try:
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(loader))
            x = x.to(device)
            out = model(x)
            if isinstance(out, (tuple, list)):
                if len(out) == 4 and isinstance(out[0], tuple):
                    (z_mean, _), _, z, _ = out
                elif len(out) == 4:
                    _, q_z, _, mu = out
                    z = q_z.rsample() if getattr(model, "distribution", None) == "clifford" else mu
                else:
                    z = out[-1]
            else:
                z = out
    except Exception:
        return {"binding_k_self_similarity": 0.0, "similarity_after_k_binds_plot_path": None}

    if getattr(model, "distribution", None) == "gaussian":
        z = torch.nn.functional.normalize(z, p=2, dim=-1)

    a = z[:1]
    ab = a.clone()
    for _ in range(k_self_bind):
        ab = _bind(ab, a)
    for _ in range(k_self_bind):
        ab = _unbind(ab, a, method=unbind_method)
    cos_sim = torch.nn.functional.cosine_similarity(ab, a, dim=-1).mean().item()

    sims = []
    for m in range(1, k_self_bind + 1):
        cur = a.clone()
        for _ in range(m):
            cur = _bind(cur, a)
        for _ in range(m):
            cur = _unbind(cur, a, method=unbind_method)
        sim_m = torch.nn.functional.cosine_similarity(cur, a, dim=-1).mean().item()
        sims.append(sim_m)

    path_bind_curve = os.path.join(output_dir, f"similarity_after_k_binds_{unbind_method}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    xs = np.arange(1, k_self_bind + 1)
    plt.plot(xs, sims, marker="o")
    plt.ylim(0.0, 1.05)
    plt.xlabel("m (bind m times then unbind m times)")
    plt.ylabel("Cosine similarity to original")
    plt.title("Similarity After K Binds")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_bind_curve, dpi=200, bbox_inches="tight")
    plt.close()

    return {"binding_k_self_similarity": cos_sim, "similarity_after_k_binds_plot_path": path_bind_curve}


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
        if self.use:
            self.run.log(d)

    def log_summary(self, d):
        if self.use:
            self.run.summary.update(d)

    def log_images(self, images):
        if self.use:
            to_log = {}
            for k, v in images.items():
                if isinstance(v, str) and os.path.exists(v):
                    to_log[k] = wandb.Image(v)
                else:
                    to_log[k] = v
            self.run.log(to_log)

    def finish_run(self):
        if self.use and self.run is not None:
            self.run.finish()


def _extract_latent_mu(model, x: torch.Tensor):
    out = model(x)
    if isinstance(out, (tuple, list)):
        if len(out) == 4 and isinstance(out[0], tuple):
            # ((z_mean, z_param2), (q_z,p_z), z, x_recon)
            (z_mean, _), _, _, _ = out
            return z_mean
        elif len(out) == 4:
            # (x_recon, q_z, p_z, mu)
            _, _, _, mu = out
            return mu
        else:
            return out[-1]
    return out

_CLASS_NAMES = {
    "fashionmnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "mnist": [str(i) for i in range(10)],
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}

def _resolve_class_names(dataset_name: str, count: int) -> list:
    key = (dataset_name or "").lower()
    names = _CLASS_NAMES.get(key, [str(i) for i in range(count)])
    return names[:count]



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
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
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



def vsa_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _bind(a, b)


def vsa_unbind(ab: torch.Tensor, b: torch.Tensor, method: str = "pseudo") -> torch.Tensor:
    return _unbind(ab, b, method=method)


def vsa_invert(a: torch.Tensor) -> torch.Tensor:
    # [a0, a1, a2, ..., a_{n-1}] -> [a0, a_{n-1}, ..., a2, a1]
    head = a[..., :1]
    tail = a[..., 1:]
    return torch.cat([head, torch.flip(tail, dims=[-1])], dim=-1)


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
    imgs = model.decoder(Z).detach().cpu()
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
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


@torch.no_grad()
def test_vsa_operations(
    model,
    loader,
    device,
    output_dir,
    n_test_pairs: int = 50,
    unbind_method: str = "pseudo",
    unitary_keys: bool = False,
    normalize_vectors: bool = True,
    project_vectors: bool = False,
):
    model.eval()

    latents = []
    for x, _ in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)
        latents.append(mu.detach())
        if len(torch.cat(latents, 0)) >= n_test_pairs * 2:
            break

    if not latents:
        return {"vsa_bind_unbind_similarity": 0.0, "vsa_bind_unbind_plot": None}

    z_all = torch.cat(latents, 0)[: n_test_pairs * 2]
    if z_all.shape[0] < 2:
        return {"vsa_bind_unbind_similarity": 0.0, "vsa_bind_unbind_plot": None}

    dist_type = getattr(model, "distribution", "normal")
    if dist_type == "powerspherical" or normalize_vectors:
        z_all = torch.nn.functional.normalize(z_all, p=2, dim=-1)

    single_bind_sims = []
    for i in range(min(n_test_pairs, z_all.shape[0] // 2)):
        key_idx = np.random.randint(z_all.shape[0])
        key = z_all[key_idx]
        if unitary_keys:
            key = _fft_make_unitary(key)
        value = z_all[i]
        a = key.unsqueeze(0)
        b = value.unsqueeze(0)
        if project_vectors:
            a = _fft_make_unitary(a.squeeze(0)).unsqueeze(0)
            b = _fft_make_unitary(b.squeeze(0)).unsqueeze(0)
        bound = _bind(a, b)
        recovered = _unbind(bound, a, method=unbind_method)
        sim = torch.nn.functional.cosine_similarity(
            recovered, value.unsqueeze(0), dim=-1
        ).item()
        single_bind_sims.append(sim)

    avg_single_sim = float(np.mean(single_bind_sims)) if single_bind_sims else 0.0

    path_vsa_test = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        if single_bind_sims:
            path_vsa_test = os.path.join(output_dir, f"vsa_bind_unbind_{unbind_method}.png")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(single_bind_sims, bins=20, alpha=0.7, edgecolor="black")
            plt.axvline(np.mean(single_bind_sims), color="red", linestyle="--", label=f"Mean: {np.mean(single_bind_sims):.3f}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Count")
            plt.title("Bind-Unbind Performance")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(single_bind_sims, "o-", alpha=0.7, markersize=4)
            plt.axhline(np.mean(single_bind_sims), color="red", linestyle="--", alpha=0.8)
            plt.xlabel("Test Index")
            plt.ylabel("Cosine Similarity")
            plt.title("Per-Test Similarity")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path_vsa_test, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Warning: VSA bind/unbind plotting failed: {e}")

    return {"vsa_bind_unbind_similarity": avg_single_sim, "vsa_bind_unbind_plot": path_vsa_test}



@torch.no_grad()
def test_hrr_sentence(
    model,
    loader,
    device,
    output_dir,
    unbind_method: str = "pseudo",
    unitary_keys: bool = False,
    normalize_vectors: bool = True,
    project_fillers: bool = False,
    dataset_name: str = None,
):
    model.eval()

    # collect latent means grouped by label
    by_label = defaultdict(list)
    for x, y in loader:
        x = x.to(device)
        mu = _extract_latent_mu(model, x)
        for vec, lbl in zip(mu.detach(), y.tolist()):
            by_label[lbl].append(vec)
        if sum(len(v) for v in by_label.values()) >= 1000:
            break

    if not by_label or any(len(v) == 0 for v in by_label.values() if v is not None):
        return {"hrr_fashion_object_acc": 0.0, "hrr_fashion_plot": None}

    class_vecs = []
    for k in range(10):
        if len(by_label[k]) == 0:
            continue
        m = torch.stack(by_label[k], 0).mean(0)
        class_vecs.append(m)
    if len(class_vecs) < 2:
        return {"hrr_fashion_object_acc": 0.0, "hrr_fashion_plot": None}
    class_vecs = torch.stack(class_vecs, 0)

    dist_type = getattr(model, "distribution", "normal")
    if dist_type == "powerspherical" or normalize_vectors:
        class_vecs = torch.nn.functional.normalize(class_vecs, p=2, dim=-1)

    rng = np.random.default_rng(0)
    role_item = class_vecs[rng.integers(0, class_vecs.shape[0])]
    role_container = class_vecs[rng.integers(0, class_vecs.shape[0])]
    if unitary_keys:
        role_item = _fft_make_unitary(role_item)
        role_container = _fft_make_unitary(role_container)

    class_names = _resolve_class_names(dataset_name or "", class_vecs.shape[0])
    candidates_idx = [i for i in range(class_vecs.shape[0])]
    # choose two distinct classes deterministically for visual stability
    item_idx = (7 if 7 < class_vecs.shape[0] else candidates_idx[0])
    container_idx = (8 if 8 < class_vecs.shape[0] else candidates_idx[-1])
    if item_idx == container_idx and len(candidates_idx) > 1:
        container_idx = (item_idx + 1) % len(candidates_idx)
    item_vec = class_vecs[item_idx].unsqueeze(0)
    container_vec = class_vecs[container_idx].unsqueeze(0)
    if project_fillers:
        item_vec = _fft_make_unitary(item_vec)
        container_vec = _fft_make_unitary(container_vec)

    a = role_item.unsqueeze(0)
    c = role_container.unsqueeze(0)
    if normalize_vectors:
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        c = torch.nn.functional.normalize(c, p=2, dim=-1)
        item_vec = torch.nn.functional.normalize(item_vec, p=2, dim=-1)
        container_vec = torch.nn.functional.normalize(container_vec, p=2, dim=-1)

    # memory and decod
    memory = _bind(a, item_vec) + _bind(c, container_vec)
    recovered_item = _unbind(memory, a, method=unbind_method).squeeze(0)
    if normalize_vectors:
        recovered_item = torch.nn.functional.normalize(recovered_item, p=2, dim=-1)

    sims = torch.nn.functional.cosine_similarity(
        recovered_item.unsqueeze(0), class_vecs, dim=-1
    )
    best = int(torch.argmax(sims).item())
    is_correct = 1.0 if best == item_idx else 0.0

    # plot
    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        tag = (dataset_name or "dataset").lower()
        path = os.path.join(output_dir, f"hrr_{tag}_{unbind_method}.png")
        labels = class_names
        plt.figure(figsize=(8, 3))
        heights = sims.detach().cpu().numpy()
        colors = ["C0"] * len(labels)
        colors[item_idx] = "green"
        bars = plt.bar(np.arange(len(labels)), heights, alpha=0.8, color=colors)
        try:
            bars[best].set_edgecolor("red")
            bars[best].set_linewidth(2.0)
        except Exception:
            pass
        plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        plt.ylabel("cosine sim")
        title_expected = labels[item_idx]
        title_pred = labels[best]
        plt.title(f"Decode item from HRR (expected={title_expected}, predicted={title_pred})")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        path = None

    return {"hrr_fashion_object_acc": float(is_correct), "hrr_fashion_plot": path}


# Backward-compat wrapper
@torch.no_grad()
def test_hrr_fashionmnist_sentence(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("dataset_name", "fashionmnist")
    return test_hrr_sentence(*args, **kwargs)


@torch.no_grad()
def test_bundle_capacity(
    model,
    loader,
    device,
    output_dir,
    n_items: int = 1000,
    k_range: list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    n_trials: int = 10,
    normalize_vectors: bool = True,
):
    model.eval()

    latents = []
    while len(torch.cat(latents, 0) if latents else []) < n_items:
        for x, _ in loader:
            x = x.to(device)
            mu = _extract_latent_mu(model, x)
            latents.append(mu.detach())
            if len(torch.cat(latents, 0)) >= n_items:
                break
        if not loader:
            break 
    
    if not latents or len(torch.cat(latents, 0)) < (max(k_range) if k_range else 0):
        return {"bundle_capacity_plot": None, "bundle_capacity_accuracies": {}}
    
    item_memory = torch.cat(latents, 0)[:n_items]
    
    dist_type = getattr(model, "distribution", "normal")
    if dist_type == "powerspherical" or normalize_vectors:
        item_memory = torch.nn.functional.normalize(item_memory, p=2, dim=-1)

    accuracies = {}
    for k in k_range:
        if k > n_items:
            continue
        
        trial_accuracies = []
        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[:k]
            chosen_vectors = item_memory[indices]
            
            bundle = torch.sum(chosen_vectors, dim=0)
            if normalize_vectors and torch.norm(bundle) > 1e-6:
                 bundle = torch.nn.functional.normalize(bundle, p=2, dim=-1)

            sims = torch.nn.functional.cosine_similarity(bundle.unsqueeze(0), item_memory)
            
            top_k_indices = torch.topk(sims, k).indices
            
            retrieved_indices = set(top_k_indices.cpu().numpy())
            original_indices = set(indices.cpu().numpy())
            
            correctly_retrieved = len(retrieved_indices.intersection(original_indices))
            accuracy = correctly_retrieved / k
            trial_accuracies.append(accuracy)
            
        accuracies[k] = np.mean(trial_accuracies)

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "bundle_capacity.png")
        plt.figure(figsize=(7, 5))
        ks = sorted(accuracies.keys())
        accs = [accuracies[k_] for k_ in ks]
        plt.plot(ks, accs, marker='o')
        plt.xlabel("Number of Bundled Vectors (k)")
        plt.ylabel("Retrieval Accuracy")
        plt.title(f"Bundling Capacity (D={item_memory.shape[1]}, N={n_items})")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot bundle capacity: {e}")
        path = None
        
    return {"bundle_capacity_plot": path, "bundle_capacity_accuracies": accuracies}


@torch.no_grad()
def test_unbinding_of_bundled_pairs(
    model,
    loader,
    device,
    output_dir,
    unbind_method: str = "pseudo",
    n_items: int = 1000,
    k_range: list = [2, 5, 10, 15, 20],
    n_trials: int = 10,
    normalize_vectors: bool = True,
    unitary_keys: bool = False,
):
    model.eval()

    latents = []
    while len(torch.cat(latents, 0) if latents else []) < n_items:
        for x, _ in loader:
            x = x.to(device)
            mu = _extract_latent_mu(model, x)
            latents.append(mu.detach())
            if len(torch.cat(latents, 0)) >= n_items:
                break
        if not latents: break
    
    required_vecs = (max(k_range) * 2) if k_range else 0
    if not latents or len(torch.cat(latents, 0)) < required_vecs:
        return {"unbind_bundled_plot": None, "unbind_bundled_accuracies": {}}

    item_memory = torch.cat(latents, 0)[:n_items]
    dist_type = getattr(model, "distribution", "normal")
    if dist_type == "powerspherical" or normalize_vectors:
        item_memory = torch.nn.functional.normalize(item_memory, p=2, dim=-1)

    accuracies = {}
    for k in k_range:
        if k * 2 > n_items:
            continue
            
        trial_accuracies = []
        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[:k*2]
            roles = item_memory[indices[:k]]
            fillers = item_memory[indices[k:]]

            if unitary_keys:
                D = roles.shape[-1]
                roles = torch.randn(k, D, device=device, dtype=item_memory.dtype) / math.sqrt(D)

            bound_pairs = _bind(roles, fillers)
            bundle = torch.sum(bound_pairs, dim=0)

            correctly_recovered = 0
            
            recovered_fillers = _unbind(bundle.unsqueeze(0), roles, method=unbind_method)
            sims = torch.bmm(recovered_fillers.unsqueeze(1), fillers.unsqueeze(2)).squeeze()
            if sims.dim() == 0: # handle k=1 case
                sims = sims.unsqueeze(0)
            best_matches = torch.argmax(torch.nn.functional.cosine_similarity(recovered_fillers.unsqueeze(1), item_memory.unsqueeze(0), dim=-1), dim=1)
            
            for i in range(k):
                original_filler_idx = indices[k+i]
                if best_matches[i] == original_filler_idx:
                    correctly_recovered += 1

            trial_accuracies.append(correctly_recovered / k) # accuracy over fillers for this trial
            
        accuracies[k] = np.mean(trial_accuracies)

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"unbind_bundled_pairs_{unbind_method}.png")
        plt.figure(figsize=(7, 5))
        ks = sorted(accuracies.keys())
        accs = [accuracies[k_] for k_ in ks]
        plt.plot(ks, accs, marker='o')
        plt.xlabel("Number of Bundled Pairs (k)")
        plt.ylabel("Filler Retrieval Accuracy")
        plt.title(f"Unbinding of Bundled Pairs (D={item_memory.shape[1]}, N={n_items})")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot unbinding of bundled pairs: {e}")
        path = None
        
    return {"unbind_bundled_plot": path, "unbind_bundled_accuracies": accuracies}


def plot_radial_freq_mask(mask: np.ndarray, filename: str) -> str:
    """Plot and save the radial frequency weighting mask."""
    plt.figure(figsize=(4, 4))
    plt.imshow(mask, cmap="magma", origin="lower")
    plt.axis("off")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()
    return filename


def plot_clifford_torus_3d_visualization(model, device, output_dir, dims=(0, 1, 2), n_grid=16):
    """Create 3D visualization of Clifford torus latent space."""
    if getattr(model, "distribution", None) != "clifford" or model.latent_dim < 3:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clifford_torus_3d_visualization.png")
    
    # 3D grid of angles
    angles0 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    angles1 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    angles2 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    
    mesh0, mesh1, mesh2 = torch.meshgrid(angles0, angles1, angles2, indexing="ij")
    A = torch.zeros(n_grid * n_grid * n_grid, model.latent_dim, device=device)
    A[:, dims[0]] = mesh0.reshape(-1)
    A[:, dims[1]] = mesh1.reshape(-1)
    A[:, dims[2]] = mesh2.reshape(-1)
    
    # convert to Clifford vector representation
    Z = _angles_to_clifford_vector(A, normalize_ifft=True)
    
    # sample points for visualization (not all points to avoid overcrowding)
    sample_indices = torch.randperm(len(Z))[:min(1000, len(Z))]
    Z_sample = Z[sample_indices]
    A_sample = A[sample_indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D scatter of latent points
    ax.scatter(
        A_sample[:, dims[0]].cpu().numpy(),
        A_sample[:, dims[1]].cpu().numpy(),
        A_sample[:, dims[2]].cpu().numpy(),
        c=torch.norm(Z_sample, dim=1).cpu().numpy(),
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    
    ax.set_xlabel(f'Angle[{dims[0]}]')
    ax.set_ylabel(f'Angle[{dims[1]}]')
    ax.set_zlabel(f'Angle[{dims[2]}]')
    ax.set_title('Clifford Torus 3D Latent Space Visualization')
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return path


def plot_clifford_torus_fourier_analysis(model, device, output_dir, dims=(0, 1), n_grid=16):
    """Analyze Fourier properties of Clifford torus latent space."""
    if getattr(model, "distribution", None) != "clifford" or model.latent_dim < 2:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clifford_torus_fourier_analysis.png")
    
    # 2D grid of angles
    angles0 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    angles1 = torch.linspace(-math.pi, math.pi, n_grid, device=device)
    mesh0, mesh1 = torch.meshgrid(angles0, angles1, indexing="ij")
    A = torch.zeros(n_grid * n_grid, model.latent_dim, device=device)
    A[:, dims[0]] = mesh0.reshape(-1)
    A[:, dims[1]] = mesh1.reshape(-1)
    
    # convert to Clifford vector representation
    Z = _angles_to_clifford_vector(A, normalize_ifft=True)
    
    # analyze FFT properties
    Z_reshaped = Z.reshape(n_grid, n_grid, -1)
    Z_fft = torch.fft.fft2(Z_reshaped, dim=(0, 1))
    Z_fft_mag = torch.abs(Z_fft)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # original angles
    im1 = axes[0, 0].imshow(mesh0.cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title(f'Angle[{dims[0]}] Grid')
    axes[0, 0].set_xlabel('Grid X')
    axes[0, 0].set_ylabel('Grid Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(mesh1.cpu().numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title(f'Angle[{dims[1]}] Grid')
    axes[0, 1].set_xlabel('Grid X')
    axes[0, 1].set_ylabel('Grid Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # FFT magnitude
    im3 = axes[1, 0].imshow(Z_fft_mag[..., 0].cpu().numpy(), cmap='magma', origin='lower')
    axes[1, 0].set_title('FFT Magnitude (First Component)')
    axes[1, 0].set_xlabel('Frequency X')
    axes[1, 0].set_ylabel('Frequency Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # vector magnitude
    Z_mag = torch.norm(Z_reshaped, dim=-1)
    im4 = axes[1, 1].imshow(Z_mag.cpu().numpy(), cmap='plasma', origin='lower')
    axes[1, 1].set_title('Vector Magnitude')
    axes[1, 1].set_xlabel('Grid X')
    axes[1, 1].set_ylabel('Grid Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return path


def plot_powerspherical_manifold_visualization(model, device, output_dir, n_samples=1000, dims=(0, 1)):
    """Visualize PowerSpherical manifold latent space."""
    if getattr(model, "distribution", None) != "powerspherical" or model.latent_dim < 2:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "powerspherical_manifold_visualization.png")
    
    # sample points from the hypersphere
    model.eval()
    with torch.no_grad():
        # random samples on the hypersphere
        z = torch.randn(n_samples, model.latent_dim, device=device)
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
    if getattr(model, "distribution", None) != "gaussian" or model.latent_dim < 2:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gaussian_manifold_visualization.png")
    
    # sample points from the Gaussian distribution
    model.eval()
    with torch.no_grad():
        # random samples from standard normal
        z = torch.randn(n_samples, model.latent_dim, device=device)
        
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

# synth test set for periodic images
def test_lowfreq_torus_visualization(model, device, output_dir, N=4, im_size=64, batch_size=256):
    """Generate and visualize low-frequency periodic images on the Clifford torus."""
    try:
        from utils.low_freq_generator import LowFreqGenerator
        from torch.utils.data import TensorDataset, DataLoader
        
        # generates one batch of synthetic low-frequency images
        gen = LowFreqGenerator(batch_size=batch_size, N=N, im_size=im_size).generate()
        x_np = next(gen)[0]  # shape (B,H,W,C)
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).to(device) # (B,C,H,W)
        dummy_labels = torch.zeros(batch_size, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, dummy_labels), batch_size=batch_size)

        # Latent scatter
        scatter = plot_clifford_torus_latent_scatter(
            model, loader, device, output_dir, dims=(0,1), dataset_name='lowfreq'
        )
        # uniform grid of angles)
        grid = plot_clifford_torus_recon_grid(
            model, device, output_dir, dims=(0,1), n_grid=16
        )
        return {'lowfreq_scatter': scatter, 'lowfreq_recon_grid': grid}
    except ImportError:
        print("Warning: LowFreqGenerator not available, skipping lowfreq test")
        return {'lowfreq_scatter': None, 'lowfreq_recon_grid': None}
    except Exception as e:
        print(f"Warning: Lowfreq test failed: {e}")
        return {'lowfreq_scatter': None, 'lowfreq_recon_grid': None}


def test_random_fourier_torus_visualization(model, device, output_dir, N=10, im_size=64, batch_size=256, gamma=0.5):
    """Generate random periodic images via random Fourier components and visualize on Clifford torus."""
    try:
        from utils.low_freq_generator import LowFreqGenerator
        from torch.utils.data import TensorDataset, DataLoader
        
        # generates one batch of random-Fourier images (translations of random periodic picture) taken from DiffusionVAE github
        gen = LowFreqGenerator(batch_size=batch_size, N=N, im_size=im_size, discounting=gamma).generate()
        x_np = next(gen)[0]  # shape (B,H,W,C)
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).to(device)
        dummy = torch.zeros(batch_size, dtype=torch.long, device=device)
        loader = DataLoader(TensorDataset(x, dummy), batch_size=batch_size)

        scatter = plot_clifford_torus_latent_scatter(model, loader, device, output_dir, dims=(0,1), dataset_name='randfourier')
        grid = plot_clifford_torus_recon_grid(model, device, output_dir, dims=(0,1), n_grid=16)
        return {'randfourier_scatter': scatter, 'randfourier_recon_grid': grid}
    except ImportError:
        print("Warning: LowFreqGenerator not available, skipping random fourier test")
        return {'randfourier_scatter': None, 'randfourier_recon_grid': None}
    except Exception as e:
        print(f"Warning: Random fourier test failed: {e}")
        return {'randfourier_scatter': None, 'randfourier_recon_grid': None}


def test_simple_cosine_translation(model, device, output_dir, im_size=64, batch_size=1024):
    """Generate translations of pic(θ,φ)=cosθ+cosφ on a torus, then visualize latent scatter & recon grid."""
    try:
        from torch.utils.data import TensorDataset, DataLoader
        
        # sample random shifts Δθ,Δφ
        shifts = torch.rand(batch_size, 2, device=device) * 2 * math.pi - math.pi
        # build images
        xs = shifts[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        ys = shifts[:, 1].unsqueeze(-1).unsqueeze(-1)
        grid = torch.linspace(-math.pi, math.pi, im_size, device=device)
        th, ph = torch.meshgrid(grid, grid, indexing='ij')
        pic = (th + xs).cos() + (ph + ys).cos()  # (B,H,W)
        x = pic.unsqueeze(1)  # (B,1,H,W)
        dummy = torch.zeros(batch_size, dtype=torch.long, device=device)
        loader = DataLoader(TensorDataset(x, dummy), batch_size=batch_size)

        scatter = plot_clifford_torus_latent_scatter(model, loader, device, output_dir, dims=(0,1), dataset_name='simplecos')
        recon = plot_clifford_torus_recon_grid(model, device, output_dir, dims=(0,1), n_grid=16)
        return {'simplecos_scatter': scatter, 'simplecos_recon': recon}
    except Exception as e:
        print(f"Warning: Simple cosine test failed: {e}")
        return {'simplecos_scatter': None, 'simplecos_recon': None}

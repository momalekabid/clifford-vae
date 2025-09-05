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


def _unit_magnitude_fraction(f: torch.Tensor, tol: float = 0.05) -> float:
    mags = torch.abs(f)
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
            all_z = []
            all_labels = []
            for x, y in loader:
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
                all_z.append(z.detach())
                all_labels.append(y)
                if len(torch.cat(all_z, 0)) >= 100:
                    break
            
            if not all_z:
                return {"binding_k_self_similarity": 0.0, "similarity_after_k_binds_plot_path": None}
                
            all_z = torch.cat(all_z, 0)
            all_labels = torch.cat(all_labels, 0)
            
            unique_labels = torch.unique(all_labels)[:3]
            selected_z = []
            selected_labels = []
            
            for label in unique_labels:
                label_mask = all_labels == label
                if label_mask.sum() > 0:
                    indices = torch.where(label_mask)[0]
                    random_idx = indices[torch.randint(0, len(indices), (1,))]
                    selected_z.append(all_z[random_idx])
                    selected_labels.append(label.item())
            
            if not selected_z:
                selected_z = [all_z[0]]
                selected_labels = [all_labels[0].item()]
                
    except Exception:
        return {"binding_k_self_similarity": 0.0, "similarity_after_k_binds_plot_path": None}

    if getattr(model, "distribution", None) == "gaussian":
        selected_z = [torch.nn.functional.normalize(z.unsqueeze(0), p=2, dim=-1) for z in selected_z]
    else:
        selected_z = [z.unsqueeze(0) for z in selected_z]

    a = selected_z[0]
    ab = a.clone()
    for _ in range(k_self_bind):
        ab = _bind(ab, a)
    for _ in range(k_self_bind):
        ab = _unbind(ab, a, method=unbind_method)
    cos_sim = torch.nn.functional.cosine_similarity(ab, a, dim=-1).mean().item()

    sims = []
    recon_every = 10
    all_recon_vectors = []
    recon_steps = []
    
    for m in range(1, k_self_bind + 1):
        cur = a.clone()
        for _ in range(m):
            cur = _bind(cur, a)
        for _ in range(m):
            cur = _unbind(cur, a, method=unbind_method)
        sim_m = torch.nn.functional.cosine_similarity(cur, a, dim=-1).mean().item()
        sims.append(sim_m)
    
    for i, start_vec in enumerate(selected_z):
        recon_vectors_for_this_start = [start_vec.squeeze(0)]
        for m in range(1, k_self_bind + 1):
            cur = start_vec.clone()
            for _ in range(m):
                cur = _bind(cur, start_vec)
            for _ in range(m):
                cur = _unbind(cur, start_vec, method=unbind_method)
            if (m % recon_every == 0) or (m == k_self_bind):
                recon_vectors_for_this_start.append(cur.squeeze(0))
                if i == 0:
                    recon_steps.append(m)
        all_recon_vectors.append(recon_vectors_for_this_start)

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

    try:
        if all_recon_vectors:
            recon_paths = os.path.join(output_dir, f"recon_after_k_binds_{unbind_method}.png")
            
            all_vectors = []
            row_labels = []
            
            for i, recon_vectors_for_start in enumerate(all_recon_vectors):
                for vec in recon_vectors_for_start:
                    all_vectors.append(vec)
                class_label = selected_labels[i] if i < len(selected_labels) else i
                row_labels.append(f"Class {class_label}")
            
            with torch.no_grad():
                imgs = model.decoder(torch.stack(all_vectors, 0))
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'output_activation'):
                    if model.decoder.output_activation == 'sigmoid':
                        imgs = torch.sigmoid(imgs)
                    else:
                        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                else:
                    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                imgs = imgs.cpu()
            
            C, h, w = imgs.shape[1], imgs.shape[-2], imgs.shape[-1]
            n_rows = len(all_recon_vectors)
            n_cols = len(all_recon_vectors[0]) if all_recon_vectors else 1
            
            canvas = torch.zeros(C, n_rows * h, n_cols * w)
            
            img_idx = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if img_idx < len(imgs):
                        canvas[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = imgs[img_idx]
                        img_idx += 1
            
            plt.figure(figsize=(max(12, n_cols * 1.5), max(6, n_rows * 2)))
            if C == 1:
                plt.imshow(canvas.squeeze(0), cmap="gray")
            else:
                plt.imshow(canvas.permute(1, 2, 0))
            
            col_labels = [f"m=0"] + [f"m={m}" for m in recon_steps]
            plt.xticks([])
            plt.yticks([])
            
            class_info = ", ".join([f"Class {label}" for label in selected_labels[:len(all_recon_vectors)]])
            plt.title(f"Reconstructions after bind+unbind m times\nRows: {class_info}")
            plt.tight_layout()
            plt.savefig(recon_paths, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        recon_paths = None

    return {
        "binding_k_self_similarity": cos_sim,
        "similarity_after_k_binds_plot_path": path_bind_curve,
        "recon_after_k_binds_plot_path": recon_paths,
    }


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
    valid_labels = []
    for k in range(10):
        if len(by_label[k]) > 0:
            m = torch.stack(by_label[k], 0).mean(0)
            class_vecs.append(m)
            valid_labels.append(k)
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

    class_names = _resolve_class_names(dataset_name or "", 10)
    
    all_correct = 0
    total_tests = 0
    all_plots = []
    
    for test_item_idx in range(len(class_vecs)):
        test_container_idx = (test_item_idx + 1) % len(class_vecs)
        
        item_vec = class_vecs[test_item_idx].unsqueeze(0)
        container_vec = class_vecs[test_container_idx].unsqueeze(0)
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

        memory = _bind(a, item_vec) + _bind(c, container_vec)
        recovered_item = _unbind(memory, a, method=unbind_method).squeeze(0)
        if normalize_vectors:
            recovered_item = torch.nn.functional.normalize(recovered_item, p=2, dim=-1)

        sims = torch.nn.functional.cosine_similarity(
            recovered_item.unsqueeze(0), class_vecs, dim=-1
        )
        best = int(torch.argmax(sims).item())
        is_correct = 1.0 if best == test_item_idx else 0.0
        all_correct += is_correct
        total_tests += 1
        
        if test_item_idx < 3:
            all_plots.append({
                'sims': sims.detach().cpu().numpy(),
                'expected_idx': test_item_idx,
                'predicted_idx': best,
                'expected_label': class_names[valid_labels[test_item_idx]],
                'predicted_label': class_names[valid_labels[best]],
                'correct': is_correct
            })

    avg_accuracy = all_correct / max(1, total_tests)

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        tag = (dataset_name or "dataset").lower()
        path = os.path.join(output_dir, f"hrr_{tag}_all_classes_{unbind_method}.png")
        
        n_plots = len(all_plots)
        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
            if n_plots == 1:
                axes = [axes]
                
            for i, plot_info in enumerate(all_plots):
                ax = axes[i]
                heights = plot_info['sims']
                colors = ["C0"] * len(heights)
                colors[plot_info['expected_idx']] = "green"
                
                bars = ax.bar(np.arange(len(valid_labels)), heights, alpha=0.8, color=colors)
                try:
                    bars[plot_info['predicted_idx']].set_edgecolor("red")
                    bars[plot_info['predicted_idx']].set_linewidth(2.0)
                except Exception:
                    pass
                    
                labels_subset = [class_names[valid_labels[j]] for j in range(len(valid_labels))]
                ax.set_xticks(np.arange(len(labels_subset)))
                ax.set_xticklabels(labels_subset, rotation=30, ha="right")
                ax.set_ylabel("cosine sim")
                
                status = "✓" if plot_info['correct'] else "✗"
                ax.set_title(f"{status} Expected: {plot_info['expected_label']}\nPredicted: {plot_info['predicted_label']}")
                
            plt.suptitle(f"HRR Decode Test - All Classes (Avg Acc: {avg_accuracy:.2f})")
            plt.tight_layout()
            plt.savefig(path, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception:
        path = None

    return {"hrr_fashion_object_acc": float(avg_accuracy), "hrr_fashion_plot": path}


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
    if normalize_vectors:
        item_memory = torch.nn.functional.normalize(item_memory, p=2, dim=-1)

    accuracies = {}
    for k in k_range:
        if k > n_items:
            continue
        
        trial_accuracies = []
        for _ in range(n_trials):
            indices = torch.randperm(n_items, device=device)[:k]
            chosen_vectors = item_memory[indices]
            
            bundle = torch.sum(chosen_vectors, dim=0) / k
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
                roles = torch.nn.functional.normalize(roles, p=2, dim=-1)
                roles = _fft_make_unitary(roles)

            bound_pairs = _bind(roles, fillers)
            bundle = torch.sum(bound_pairs, dim=0) / k

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
    # Check for both latent_dim and z_dim attributes
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

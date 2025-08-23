import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    note: unbinding with pseudo-inverse is less exact but should work perfectly for "unitary" vectors (plate 1995) 
    """
    if method == "pseudo":
        return _bind(ab, vsa_invert(b))
    elif method == "deconv":
        Fab = torch.fft.fft(ab, dim=-1)
        Fb = torch.fft.fft(b, dim=-1)
        denom = torch.clamp(torch.abs(Fb) ** 2, min=1e-8)
        rec = torch.fft.ifft(Fab * torch.conj(Fb) / denom, dim=-1).real
        return rec

def _fft_make_unitary(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Fx = torch.fft.fft(x, dim=-1)
    mags = torch.abs(Fx)
    Fx_unit = Fx / torch.clamp(mags, min=eps)
    return torch.fft.ifft(Fx_unit, dim=-1).real



def test_fourier_properties(model, loader, device, output_dir, k_self_bind: int = 50, unbind_method: str = "pseudo"):
    try:
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(loader))
            x = x.to(device)
            out = model(x)
            
            if isinstance(out, (tuple, list)):
                if len(out) == 4 and isinstance(out[0], tuple):
                    (z_mean, z_param2), (q_z, p_z), z, x_recon = out
                elif len(out) == 4:
                    x_recon, q_z, p_z, mu = out
                    if getattr(model, "distribution", None) == "clifford":
                        z = q_z.rsample()
                    else:
                        z = mu
                else:
                    z = out[-1]
            else:
                z = out
    except Exception:
        return {
            "fourier_frac_within_0p05": 0.0,
            "fourier_max_dev": 999.0,
            "fourier_mean_dev": 999.0,
            "fourier_phase_std": 0.0,
            "binding_unbinding_cosine": 0.0,
            "binding_magnitude_mean_dev": 999.0,
            "fft_spectrum_plot_path": None,
            "fourier_mean_magnitude": 0.0,
            "fourier_magnitude_std": 0.0,
            "fourier_flatness_mse": 999.0,
            "binding_k_self_similarity": 0.0,
            "similarity_after_k_binds_plot_path": None,
        }

    Fz = torch.fft.fft(z, dim=-1)
    mags = torch.abs(Fz)
    phases = torch.angle(Fz)
    target = 1.0
    dev = torch.abs(mags - target)
    mean_mag = mags.mean().item()
    std_mag = mags.std().item()
    mean_dev = dev.mean().item()
    max_dev = dev.max().item()
    frac_within = _unit_magnitude_fraction(Fz, tol=0.05)
    # deprecated metrics removed per spec

    a = z[:1]
    ab = a.clone()
    for _ in range(k_self_bind):
        ab = _bind(ab, a)
    for _ in range(k_self_bind):
        ab = _unbind(ab, a, method=unbind_method)
    cos_sim = torch.nn.functional.cosine_similarity(ab, a, dim=-1).mean().item()

    # sim curve over m = 1..k_self_bind
    sims = []
    for m in range(1, k_self_bind + 1):
        cur = a.clone()
        for _ in range(m):
            cur = _bind(cur, a)
        for _ in range(m):
            cur = _unbind(cur, a, method=unbind_method)
        sim_m = torch.nn.functional.cosine_similarity(cur, a, dim=-1).mean().item()
        sims.append(sim_m)

    def safe_hist(ax, data, title, target_line=None):
        data_flat = data.ravel()
        data_min, data_max = data_flat.min(), data_flat.max()
        data_range = data_max - data_min
        
        if data_range < 1e-10:
            ax.axhline(
                y=1.0,
                color="blue",
                alpha=0.7,
                linewidth=3,
                label=f"Constant ≈ {data_min:.6f}",
            )
            ax.set_ylim(0, 2)
            ax.legend()
        elif data_range < 1e-6:
            unique_vals = np.unique(data_flat)
            if len(unique_vals) <= 10:
                counts = [np.sum(data_flat == val) for val in unique_vals]
                ax.bar(
                    unique_vals,
                    counts,
                    alpha=0.7,
                    width=data_range / max(1, len(unique_vals) - 1),
                )
            else:
                ax.scatter(range(len(data_flat[:100])), data_flat[:100], alpha=0.7, s=2)
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Value")
        else:
            max_bins = min(50, max(3, int(np.sqrt(len(data_flat)))))
            bins = max_bins
            success = False
            
            for attempt_bins in [bins, bins // 2, bins // 4, 5, 3]:
                try:
                    ax.hist(data_flat, bins=attempt_bins, density=True, alpha=0.7)
                    success = True
                    break
                except (ValueError, np.linalg.LinAlgError):
                    continue
            
            if not success:
                if len(data_flat) > 1000:
                    indices = np.linspace(0, len(data_flat) - 1, 1000, dtype=int)
                    ax.plot(indices, data_flat[indices], "o", alpha=0.5, markersize=1)
                else:
                    ax.plot(data_flat, "o", alpha=0.7, markersize=2)
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")
        
        if target_line is not None and data_range > 1e-10:
            try:
                ax.axvline(
                    x=target_line,
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    label=f"Target={target_line}",
                )
                ax.legend()
            except:
                pass
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    path = None
    path_bind_curve = None
    try:
        avg_mag = mags.mean(dim=0).detach().cpu().numpy()
        n = avg_mag.shape[-1]
        uniform = 1.0

        os.makedirs(output_dir, exist_ok=True)

        # optional: omit heavy Fourier visualizations per cleanup request
        # similarity after k binds curve
        path_bind_curve = os.path.join(output_dir, "similarity_after_k_binds.png")
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

        # removed bundling superposition diagnostic per spec
    except Exception as e:
        print(f"Warning: Failed to plot fourier magnitude spectrum: {e}")

    return {
        "fourier_frac_within_0p05": frac_within,
        # trimmed metrics
        "binding_k_self_similarity": cos_sim,
        "fourier_mean_magnitude": mean_mag,
        "fourier_magnitude_std": std_mag,
        "fourier_mean_dev": mean_dev,
        "fourier_max_dev": max_dev,
        "fft_spectrum_plot_path": path,
        "similarity_after_k_binds_plot_path": path_bind_curve,
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

    # finalize means
    # average exactly 10 per class if available
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
    """Evaluate using cosine for spherical dists, euclidean for normal."""
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
    return torch.flip(a, dims=[-1])


@torch.no_grad()
def test_vsa_operations(
    model,
    loader,
    device,
    output_dir,
    n_test_pairs: int = 50,
    unbind_method: str = "pseudo",
    unitary_keys: bool = False,
    normalize_vectors: bool = False,
):
    """
    Reduced VSA test: only bind/unbind fidelity. Returns metric and plot path.
    """
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
        if normalize_vectors:
            a = torch.nn.functional.normalize(a, p=2, dim=-1)
            b = torch.nn.functional.normalize(b, p=2, dim=-1)
        bound = _bind(a, b)
        if normalize_vectors:
            bound = torch.nn.functional.normalize(bound, p=2, dim=-1)
        recovered = _unbind(bound, a, method=unbind_method)
        if normalize_vectors:
            recovered = torch.nn.functional.normalize(recovered, p=2, dim=-1)
        sim = torch.nn.functional.cosine_similarity(recovered, value.unsqueeze(0), dim=-1).item()
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
            plt.title("Bind-Unbind Fidelity")
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
def test_hrr_mark_ate_fish(
    model,
    loader,
    device,
    output_dir,
    unbind_method: str = "pseudo",
    unitary_keys: bool = False,
    normalize_vectors: bool = False,
):
    """
    HRR sentence test: build memory for "mark ate the fish" as
    M = agent⊛mark + verb⊛eat + object⊛fish (+ frame_id). Taken from Plate, 1995.
    Decode the object via unbinding with the object role; report cleanup accuracy
    returns {hrr_sentence_object_acc, hrr_sentence_plot}
    """
    model.eval()

    # collect a small pool of latent vectors
    latents = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu = _extract_latent_mu(model, x)
            latents.append(mu.detach())
            if len(torch.cat(latents, 0)) >= 512:
                break
    if not latents:
        return {"hrr_sentence_object_acc": 0.0, "hrr_sentence_plot": None}

    Z = torch.cat(latents, 0)
    dist_type = getattr(model, "distribution", "normal")
    if dist_type == "powerspherical" or normalize_vectors:
        Z = torch.nn.functional.normalize(Z, p=2, dim=-1)

    d = Z.shape[-1]

    # choose fillers (candidates) and roles
    rng = np.random.default_rng(0)
    idx = rng.choice(Z.shape[0], size=12, replace=False)
    mark, john, paul, person = Z[idx[0]], Z[idx[1]], Z[idx[2]], Z[idx[3]]
    the_fish, the_bread, food = Z[idx[4]], Z[idx[5]], Z[idx[6]]
    eat, see, state = Z[idx[7]], Z[idx[8]], Z[idx[9]]

    # role keys
    agent_role, object_role, verb_role = Z[idx[10]], Z[idx[11]], Z[idx[0]]
    if unitary_keys:
        agent_role = _fft_make_unitary(agent_role)
        object_role = _fft_make_unitary(object_role)
        verb_role = _fft_make_unitary(verb_role)

    # construct sentence memory
    a = agent_role.unsqueeze(0)
    o = object_role.unsqueeze(0)
    v = verb_role.unsqueeze(0)
    m = mark.unsqueeze(0)
    f = the_fish.unsqueeze(0)
    e = eat.unsqueeze(0)
    if normalize_vectors:
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        o = torch.nn.functional.normalize(o, p=2, dim=-1)
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        m = torch.nn.functional.normalize(m, p=2, dim=-1)
        f = torch.nn.functional.normalize(f, p=2, dim=-1)
        e = torch.nn.functional.normalize(e, p=2, dim=-1)
    memory = _bind(a, m) + _bind(o, f) + _bind(v, e)
    if normalize_vectors:
        memory = torch.nn.functional.normalize(memory, p=2, dim=-1)

    # decode the object
    recovered_object = _unbind(memory, o, method=unbind_method).squeeze(0)
    if normalize_vectors:
        recovered_object = torch.nn.functional.normalize(recovered_object, p=2, dim=-1)

    # cleanup among candidate objects
    candidates = torch.stack([the_fish, the_bread, food, person, john, paul], 0)
    sims = torch.nn.functional.cosine_similarity(
        recovered_object.unsqueeze(0), candidates, dim=-1
    )
    best = int(torch.argmax(sims).item())
    is_correct = 1.0 if best == 0 else 0.0

    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"hrr_mark_ate_fish_{unbind_method}.png")
        labels = ["fish", "bread", "food", "person", "john", "paul"]
        plt.figure(figsize=(6, 3))
        plt.bar(np.arange(len(labels)), sims.detach().cpu().numpy(), alpha=0.8)
        plt.xticks(np.arange(len(labels)), labels, rotation=0)
        plt.ylabel("cosine sim")
        plt.title("Decode object from HRR: mark ate the fish")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        path = None

    return {"hrr_sentence_object_acc": float(is_correct), "hrr_sentence_plot": path}

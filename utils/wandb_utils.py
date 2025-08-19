import os
import math
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
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1).real


def _unbind(ab: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    Fb = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(torch.fft.fft(ab, dim=-1) * torch.conj(Fb), dim=-1).real


def test_fourier_properties(model, loader, device, output_dir, k_self_bind: int = 5):
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
                    if getattr(model, 'distribution', None) == 'clifford':
                        z = q_z.rsample()
                    else:
                        z = mu
                else:
                    z = out[-1]
            else:
                z = out
    except Exception as e:
        return {
            'fourier_frac_within_0p05': 0.0,
            'fourier_max_dev': 999.0,
            'fourier_mean_dev': 999.0,
            'fourier_phase_std': 0.0,
            'binding_unbinding_cosine': 0.0,
            'binding_magnitude_mean_dev': 999.0,
            'fft_spectrum_plot_path': None,
            'fourier_mean_magnitude': 0.0,
            'fourier_magnitude_std': 0.0,
            'fourier_flatness_mse': 999.0,
            'binding_k_self_similarity': 0.0,
            'similarity_after_k_binds_plot_path': None,
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
    phase_std = phases.std().item()
    flat_mse = F.mse_loss(mags.mean(dim=0), torch.full_like(mags.mean(dim=0), target)).item()

    a = z[:1]
    ab = a.clone()
    for _ in range(k_self_bind):
        ab = _bind(ab, a)
    for _ in range(k_self_bind):
        ab = _unbind(ab, a)
    cos_sim = torch.nn.functional.cosine_similarity(ab, a, dim=-1).mean().item()

    # sim curve over m = 1..k_self_bind
    sims = []
    for m in range(1, k_self_bind + 1):
        cur = a.clone()
        for _ in range(m):
            cur = _bind(cur, a)
        for _ in range(m):
            cur = _unbind(cur, a)
        sim_m = torch.nn.functional.cosine_similarity(cur, a, dim=-1).mean().item()
        sims.append(sim_m)


    def safe_hist(ax, data, title, target_line=None):
        data_flat = data.ravel()
        data_min, data_max = data_flat.min(), data_flat.max()
        data_range = data_max - data_min
        
        if data_range < 1e-10:
            ax.axhline(y=1.0, color='blue', alpha=0.7, linewidth=3, label=f'Constant ≈ {data_min:.6f}')
            ax.set_ylim(0, 2)
            ax.legend()
        elif data_range < 1e-6:
            unique_vals = np.unique(data_flat)
            if len(unique_vals) <= 10:
                counts = [np.sum(data_flat == val) for val in unique_vals]
                ax.bar(unique_vals, counts, alpha=0.7, width=data_range/max(1, len(unique_vals)-1))
            else:
                ax.scatter(range(len(data_flat[:100])), data_flat[:100], alpha=0.7, s=2)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Value')
        else:
            max_bins = min(50, max(3, int(np.sqrt(len(data_flat)))))
            bins = max_bins
            success = False
            
            for attempt_bins in [bins, bins//2, bins//4, 5, 3]:
                try:
                    ax.hist(data_flat, bins=attempt_bins, density=True, alpha=0.7)
                    success = True
                    break
                except (ValueError, np.linalg.LinAlgError):
                    continue
            
            if not success:
                if len(data_flat) > 1000:
                    indices = np.linspace(0, len(data_flat)-1, 1000, dtype=int)
                    ax.plot(indices, data_flat[indices], 'o', alpha=0.5, markersize=1)
                else:
                    ax.plot(data_flat, 'o', alpha=0.7, markersize=2)
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
        
        if target_line is not None and data_range > 1e-10:
            try:
                ax.axvline(x=target_line, color='r', linestyle='--', linewidth=2, label=f'Target={target_line}')
                ax.legend()
            except:
                pass
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    path = None
    path_avg = None
    path_bind_curve = None
    try:
        avg_mag = mags.mean(dim=0).detach().cpu().numpy()
        n = avg_mag.shape[-1]
        uniform = 1.0 / math.sqrt(n)

        os.makedirs(output_dir, exist_ok=True)

        try:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            safe_hist(axes[0, 0], mags.cpu().numpy(), '|F(z)|', target)
            safe_hist(axes[0, 1], phases.cpu().numpy(), 'angle(F(z))')
            safe_hist(axes[1, 0], torch.abs(torch.fft.fft(ab, dim=-1)).cpu().numpy(), '|F(bind)|', target)
            try:
                axes[1, 1].imshow(dev.cpu().numpy(), aspect='auto', cmap='viridis')
            except Exception:
                dev_flat = dev.cpu().numpy().ravel()
                axes[1, 1].plot(dev_flat[:min(100, len(dev_flat))], 'o', markersize=2)
                axes[1, 1].set_xlabel('Sample')
                axes[1, 1].set_ylabel('Deviation')
            axes[1, 1].set_title('Deviation |F|-1')
            path = os.path.join(output_dir, 'fourier_analysis.png')
            plt.tight_layout()
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Fourier prop plot saved: {path}")
        except Exception:
            pass

        # new avg magnitude spectrum
        path_avg = os.path.join(output_dir, 'fourier_avg_spectrum.png')
        plt.figure(figsize=(11, 6))
        plt.plot(avg_mag, label='Average |FFT(z)|')
        plt.axhline(y=uniform, color='r', linestyle='--', label=f'Uniform value (1/√{n})')
        plt.title('Average FFT Magnitude Spectrum of Encoded Vectors')
        plt.xlabel('Frequency Index')
        plt.ylabel('Magnitude')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(path_avg, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Fourier average spectrum saved: {path_avg}")
        # similarity after k binds curve
        path_bind_curve = os.path.join(output_dir, 'similarity_after_k_binds.png')
        plt.figure(figsize=(7, 4))
        xs = np.arange(1, k_self_bind + 1)
        plt.plot(xs, sims, marker='o')
        plt.ylim(0.0, 1.05)
        plt.xlabel('m (bind m times then unbind m times)')
        plt.ylabel('Cosine similarity to original')
        plt.title('Similarity After K Binds')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path_bind_curve, dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot fourier magnitude spectrum: {e}")

    return {
        'fourier_frac_within_0p05': frac_within,
        'fourier_phase_std': phase_std,
        'fourier_flatness_mse': flat_mse,
        'binding_k_self_similarity': cos_sim,
        'fourier_mean_magnitude': mean_mag,
        'fourier_magnitude_std': std_mag,
        'fourier_mean_dev': mean_dev,
        'fourier_max_dev': max_dev,
        'fft_spectrum_plot_path': path,
        'fft_avg_spectrum_plot_path': path_avg,
        'similarity_after_k_binds_plot_path': path_bind_curve,
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
            wandb.watch(model, log='gradients', log_freq=100)

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
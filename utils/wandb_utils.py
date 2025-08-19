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


def _conj_symmetry_fraction(F: torch.Tensor, atol: float = 1e-3) -> float:
    # checking F[k] == conj(F[N-k])
    N = F.shape[-1]
    F_rev_conj = torch.conj(torch.flip(F, dims=(-1,)))
    mask = torch.ones(N, dtype=torch.bool, device=F.device)
    mask[0] = False
    if N % 2 == 0:
        mask[N // 2] = False
    diff = torch.abs(F[..., mask] - F_rev_conj[..., mask])
    return float((diff <= atol).float().mean().item())


def _unit_magnitude_fraction(F: torch.Tensor, tol: float = 0.05) -> float:
    mags = torch.abs(F)
    return float((torch.abs(mags - 1.0) < tol).float().mean().item())


def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1).real


def _unbind(ab: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    Fb = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(torch.fft.fft(ab, dim=-1) * torch.conj(Fb), dim=-1).real


def test_fourier_properties(model, loader, device, output_dir, mode: str | None = None):
    """
    checking that mean ~ |F(z)| ≈ 1
    """
    if mode is None:
        mode = os.getenv("FOURIER_LOGGING", "full").lower()
    mode = mode if mode in {"off", "minimal", "full"} else "full"
    if mode == "off":
        return {}
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
            'fourier_conj_symmetry_frac': 0.0,
            'fourier_phase_std': 0.0,
            'binding_unbinding_cosine': 0.0,
            'binding_magnitude_mean_dev': 999.0,
            'fft_spectrum_plot_path': None,
            'fourier_mean_magnitude': 0.0,
            'fourier_magnitude_std': 0.0,
            'fourier_flatness_mse': 999.0,
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
    conj_frac = _conj_symmetry_fraction(Fz, atol=1e-3)
    phase_std = phases.std().item()
    flat_mse = F.mse_loss(mags.mean(dim=0), torch.full_like(mags.mean(dim=0), target)).item()

    # binding/unbinding check
    a = z[:1]
    b = z[1:2] if z.shape[0] > 1 else z[:1].roll(1, dims=-1)
    ab = _bind(a, b)
    a_hat = _unbind(ab, b)
    cos_sim = torch.nn.functional.cosine_similarity(a_hat, a, dim=-1).mean().item()
    mags_bind = torch.abs(torch.fft.fft(ab, dim=-1))
    dev_bind = torch.abs(mags_bind - target).mean().item()

    # unitary_ok = (frac_within > 0.95) and (max_dev < 0.2) and (conj_frac > 0.99) and (cos_sim > 0.98)

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
    if mode == "full":
        try:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            safe_hist(axes[0, 0], mags.cpu().numpy(), '|F(z)|', target)
            safe_hist(axes[0, 1], phases.cpu().numpy(), 'angle(F(z))')
            safe_hist(axes[1, 0], torch.abs(torch.fft.fft(ab, dim=-1)).cpu().numpy(), '|F(bind)|', target)
            
            try:
                axes[1, 1].imshow(dev.cpu().numpy(), aspect='auto', cmap='viridis')
            except:
                dev_flat = dev.cpu().numpy().ravel()
                axes[1, 1].plot(dev_flat[:min(100, len(dev_flat))], 'o', markersize=2)
                axes[1, 1].set_xlabel('Sample')
                axes[1, 1].set_ylabel('Deviation')
            axes[1, 1].set_title('Deviation |F|-1')
            
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 'fourier_analysis.png')
            plt.tight_layout()
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Fourier prop plot saved: {path}")
        except Exception as e:
            print(f"Warning: Failed to plot fourier analysis : {e}")
            if 'fig' in locals():
                plt.close(fig)

    if mode == "minimal":
        return {
            'fourier_frac_within_0p05': frac_within,
            'fourier_conj_symmetry_frac': conj_frac,
        }
    else:  # full
        return {
            'fourier_frac_within_0p05': frac_within,
            'fourier_max_dev': max_dev,
            'fourier_mean_dev': mean_dev,
            'fourier_conj_symmetry_frac': conj_frac,
            'fourier_phase_std': phase_std,
            'binding_unbinding_cosine': cos_sim,
            'binding_magnitude_mean_dev': dev_bind,
            'fft_spectrum_plot_path': path,
            'fourier_mean_magnitude': mean_mag,
            'fourier_magnitude_std': std_mag,
            'fourier_flatness_mse': flat_mse,
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
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as tu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_pairwise_bind_bundle_decode,
    test_cross_class_bind_unbind,
    compute_class_means,
    evaluate_mean_vector_cosine,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
)
from mnist.mlp_vae import MLPVAE, vae_loss, compute_test_metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


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


def plot_reconstructions(model, loader, device, filepath):
    """decode 8 test images and show originals vs reconstructions."""
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
        plt.title("Top: Original | Bottom: Reconstructed")
        plt.axis("off")
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()
    return filepath


def plot_interpolations(model, loader, device, filepath, steps=10):
    """interpolate between two test images from different classes."""
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        idx1 = (y == y[0]).nonzero(as_tuple=True)[0][0].item()
        idx2 = (y != y[0]).nonzero(as_tuple=True)[0][0].item()
        x1 = x[idx1].unsqueeze(0).to(device)
        x2 = x[idx2].unsqueeze(0).to(device)
        z1, _ = model.encode(x1.view(1, -1))
        z2, _ = model.encode(x2.view(1, -1))
        alphas = torch.linspace(0, 1, steps, device=device)
        interp_z = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            interp_z.append(F.normalize(z, p=2, dim=-1))
        interp_z = torch.cat(interp_z, dim=0)
        x_recon = torch.sigmoid(model.decoder(interp_z)).view(-1, 1, 28, 28)
        grid = tu.make_grid(x_recon, nrow=steps, pad_value=0.5)
        plt.figure(figsize=(12, 2))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.title("Latent Space Interpolation (vMF-VAE)")
        plt.axis("off")
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()
    return filepath


def plot_latent_space(model, loader, device, filepath, n_plot=1000):
    """t-sne visualization of latent space."""
    X_z, y = encode_dataset(model, loader, device)
    X_z, y = X_z[:n_plot], y[:n_plot]
    print(f"running t-sne on {n_plot} points...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    z_tsne = tsne.fit_transform(X_z)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap=plt.get_cmap("tab10", 10), s=10, alpha=0.8)
    plt.title("t-SNE Latent Space (vMF-VAE)")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    return filepath


def perform_knn_evaluation(model, train_loader, test_loader, device, n_samples_list):
    X_train_full, y_train_full = encode_dataset(model, train_loader, device)
    X_test, y_test = encode_dataset(model, test_loader, device)
    results = {}
    for n_samples in n_samples_list:
        if n_samples > len(X_train_full):
            continue
        indices = np.random.choice(len(X_train_full), n_samples, replace=False)
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train_full[indices], y_train_full[indices])
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results[f"knn_acc_{n_samples}"] = float(accuracy)
        results[f"knn_f1_{n_samples}"] = float(f1)
        print(f"  knn acc w/ {n_samples}: {accuracy:.4f}, f1: {f1:.4f}")
    return results


def run(args):
    script_start_time = time.time()
    timing_results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), BinarizeWithRandomThreshold()]
    )
    full_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    train_eval_loader = DataLoader(train_dataset, batch_size=1024)
    test_eval_loader = DataLoader(test_dataset, batch_size=1024)

    knn_samples = [100, 600, 1000]
    final_results = []
    logger = WandbLogger(args)

    for d_manifold in args.d_dims:
        print(f"\n{'='*30}\n== vmf d={d_manifold} ==\n{'='*30}")
        model_z_dim = d_manifold + 1  # vmf in R^(d+1)
        agg_results = {s: [] for s in knn_samples}
        agg_f1 = {s: [] for s in knn_samples}
        agg_metrics = {"ll": [], "entropy": [], "recon": [], "kl": []}
        agg_mvc = []

        for run_idx in range(args.n_runs):
            print(f"\n--- run {run_idx+1}/{args.n_runs} ---")
            run_start_time = time.time()

            if logger.use:
                logger.start_run(f"vmf-d{d_manifold}-run{run_idx+1}", args)

            model = MLPVAE(
                h_dim=args.h_dim, z_dim=model_z_dim, distribution="vmf"
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            best_val_loss = float("inf")
            patience_counter = 0
            model_path = f"best_model_vmf_d{d_manifold}_run{run_idx}.pt"
            train_start_time = time.time()

            for epoch in range(args.epochs):
                model.train()
                beta = min(1.0, (epoch + 1) / max(1, args.warmup_epochs))
                for x_mb, _ in train_loader:
                    optimizer.zero_grad()
                    vae_loss(model, x_mb.to(device), beta=beta).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for x_mb, _ in val_loader:
                        total_val_loss += vae_loss(model, x_mb.to(device), beta=1.0).item()
                avg_val_loss = total_val_loss / len(val_loader)

                if logger.use:
                    logger.log_metrics({"epoch": epoch, "val_loss": avg_val_loss, "kl_beta": beta})

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1

                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"early stopping at epoch {epoch+1}")
                    break

            train_time = time.time() - train_start_time
            print(f"training time: {train_time:.2f}s")

            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                eval_start_time = time.time()

                test_metrics = compute_test_metrics(model, test_eval_loader, device)
                for m in ["ll", "entropy", "recon", "kl"]:
                    agg_metrics[m].append(test_metrics[m])
                print(f"  LL={test_metrics['ll']:.2f}  L[q]={test_metrics['entropy']:.2f}  "
                      f"RE={test_metrics['recon']:.2f}  KL={test_metrics['kl']:.2f}")

                knn_results = perform_knn_evaluation(
                    model, train_eval_loader, test_eval_loader, device, knn_samples
                )
                for n in knn_samples:
                    if f"knn_acc_{n}" in knn_results:
                        agg_results[n].append(knn_results[f"knn_acc_{n}"])
                    if f"knn_f1_{n}" in knn_results:
                        agg_f1[n].append(knn_results[f"knn_f1_{n}"])

                if logger.use:
                    knn_metrics = {k: v for k, v in knn_results.items() if k.startswith("knn_")}
                    logger.log_metrics({
                        **knn_metrics,
                        "test/ll": test_metrics["ll"],
                        "test/entropy": test_metrics["entropy"],
                        "test/recon": test_metrics["recon"],
                        "test/kl": test_metrics["kl"],
                        "final_val_loss": best_val_loss,
                    })

                if not args.simple:
                    vis_dir = f"visualizations/d_{d_manifold}/vmf"
                    os.makedirs(vis_dir, exist_ok=True)
                    test_subset = torch.utils.data.Subset(
                        test_dataset, list(range(min(500, len(test_dataset))))
                    )
                    test_subset_loader = DataLoader(test_subset, batch_size=64)

                    self_bind = test_self_binding(
                        model, test_subset_loader, device, vis_dir, unbind_method="*"
                    )

                    deconv_dir = f"visualizations/d_{d_manifold}/vmf/deconv"
                    os.makedirs(deconv_dir, exist_ok=True)
                    self_bind_deconv = test_self_binding(
                        model, test_subset_loader, device, deconv_dir, unbind_method="†"
                    )

                    with torch.no_grad():
                        latents = []
                        for x, _ in test_subset_loader:
                            z_mean, _ = model.encode(x.to(device).view(-1, 784))
                            latents.append(z_mean.detach())
                        item_memory = torch.cat(latents, 0)[:500]

                    bundle_cap_raw = vsa_bundle_capacity(
                        d=item_memory.shape[-1],
                        n_items=500,
                        k_range=list(range(5, 51, 5)),
                        n_trials=3,
                        normalize=True,
                        device=device,
                        plot=True,
                        save_dir=vis_dir,
                        item_memory=item_memory,
                    )

                    rf_variants = [
                        (True, "*", vis_dir, "role_filler_capacity"),
                        (False, "*", vis_dir, "role_filler_no_random_keys"),
                        (True, "†", deconv_dir, "role_filler_capacity_deconv"),
                        (False, "†", deconv_dir, "role_filler_no_random_keys_deconv"),
                    ]
                    rf_results = {}
                    for bind_rand, ubmethod, save_d, rf_name in rf_variants:
                        rf_res = vsa_binding_unbinding(
                            d=item_memory.shape[-1],
                            n_items=500,
                            k_range=list(range(2, 21, 2)),
                            n_trials=3,
                            normalize=True,
                            device=device,
                            plot=True,
                            unbind_method=ubmethod,
                            save_dir=save_d,
                            item_memory=item_memory,
                            bind_with_random=bind_rand,
                        )
                        rf_results[rf_name] = rf_res
                        default_plot = os.path.join(save_d, "role_filler_capacity.png")
                        variant_plot = os.path.join(save_d, f"{rf_name}.png")
                        if os.path.exists(default_plot) and rf_name != "role_filler_capacity":
                            os.rename(default_plot, variant_plot)

                    pairwise_result = test_pairwise_bind_bundle_decode(
                        model, test_subset_loader, device, vis_dir,
                        class_names=[str(i) for i in range(10)],
                        img_shape=(1, 28, 28),
                        n_classes=10,
                    )
                    pairwise_bind_bundle_path = pairwise_result.get("pairwise_bind_bundle_path")

                    # cross-class bind/unbind test (6 vs 9)
                    cross_class_result = test_cross_class_bind_unbind(
                        model, test_subset_loader, device, vis_dir,
                        img_shape=(1, 28, 28),
                        class_a=6, class_b=9,
                    )

                    # reconstruction and interpolation plots
                    recon_path = plot_reconstructions(
                        model, test_eval_loader, device,
                        os.path.join(vis_dir, "reconstructions.png"),
                    )
                    interp_path = plot_interpolations(
                        model, test_eval_loader, device,
                        os.path.join(vis_dir, "interpolations.png"),
                    )

                    # t-sne
                    tsne_path = plot_latent_space(
                        model, test_eval_loader, device,
                        os.path.join(vis_dir, "tsne.png"),
                    )

                    # mean vector cosine
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
                    print(f"  mean vector cosine acc: {mean_vector_acc:.4f}")
                    agg_mvc.append(float(mean_vector_acc))

                    if logger.use:
                        sb_metrics = {
                            f"self_binding/{k}": v
                            for k, v in self_bind.items()
                            if isinstance(v, (int, float, bool))
                        }
                        sb_deconv_metrics = {
                            f"self_binding_deconv/{k}": v
                            for k, v in self_bind_deconv.items()
                            if isinstance(v, (int, float, bool))
                        }
                        logger.log_metrics({
                            **sb_metrics,
                            **sb_deconv_metrics,
                            "mean_vector_cosine_acc": float(mean_vector_acc),
                        })

                        images_to_log = {
                            "Reconstructions": recon_path,
                            "Interpolations": interp_path,
                            "Latent_tSNE": tsne_path,
                        }
                        sp = self_bind.get("similarity_after_k_binds_plot_path")
                        if sp:
                            images_to_log["Self_Binding_*"] = sp
                        bc_plot = os.path.join(vis_dir, "bundle_capacity.png")
                        if os.path.exists(bc_plot):
                            images_to_log["Bundle_Capacity"] = bc_plot
                        rf_plot = os.path.join(vis_dir, "role_filler_capacity.png")
                        if os.path.exists(rf_plot):
                            images_to_log["Role_Filler_Capacity"] = rf_plot
                        sp_d = self_bind_deconv.get("similarity_after_k_binds_plot_path")
                        if sp_d:
                            images_to_log["Self_Binding_†"] = sp_d
                        for rf_name in ["role_filler_no_random_keys", "role_filler_capacity_deconv", "role_filler_no_random_keys_deconv"]:
                            save_d = deconv_dir if "deconv" in rf_name else vis_dir
                            rf_plot = os.path.join(save_d, f"{rf_name}.png")
                            if os.path.exists(rf_plot):
                                images_to_log[rf_name] = rf_plot
                        if pairwise_bind_bundle_path and os.path.exists(pairwise_bind_bundle_path):
                            images_to_log["Pairwise_Bind_Bundle_Decode"] = pairwise_bind_bundle_path
                        if images_to_log:
                            logger.log_images(images_to_log)

                        logger.log_summary({**knn_metrics, **sb_metrics, **sb_deconv_metrics,
                                            "mean_vector_cosine_acc": float(mean_vector_acc),
                                            "test/ll": test_metrics["ll"],
                                            "test/entropy": test_metrics["entropy"],
                                            "test/recon": test_metrics["recon"],
                                            "test/kl": test_metrics["kl"]})
                else:
                    if logger.use:
                        logger.log_summary({**knn_metrics,
                                            "test/ll": test_metrics["ll"],
                                            "test/entropy": test_metrics["entropy"],
                                            "test/recon": test_metrics["recon"],
                                            "test/kl": test_metrics["kl"]})

                if logger.use:
                    logger.finish_run()

                eval_time = time.time() - eval_start_time
                run_time = time.time() - run_start_time
                timing_results[f"vmf_d{d_manifold}_run{run_idx+1}"] = {
                    "train_time_s": train_time,
                    "eval_time_s": eval_time,
                    "total_run_time_s": run_time,
                }
                print(f"eval time: {eval_time:.2f}s, total: {run_time:.2f}s")
                os.remove(model_path)

        row = {"d": d_manifold}
        for n in knn_samples:
            accs = agg_results[n]
            f1s = agg_f1[n]
            row[f"vMF_acc_{n}"] = (
                f"{np.mean(accs)*100:.1f}±{np.std(accs)*100:.1f}" if accs else "N/A"
            )
            row[f"vMF_f1_{n}"] = (
                f"{np.mean(f1s)*100:.1f}±{np.std(f1s)*100:.1f}" if f1s else "N/A"
            )
        row["vMF_mvc"] = (
            f"{np.mean(agg_mvc)*100:.1f}±{np.std(agg_mvc)*100:.1f}" if agg_mvc else "N/A"
        )
        for m in ["ll", "entropy", "recon", "kl"]:
            vals = agg_metrics[m]
            row[f"vMF_{m}"] = (
                f"{np.mean(vals):.2f}±{np.std(vals):.2f}" if vals else "N/A"
            )
        final_results.append(row)

    if final_results:
        try:
            import pandas as pd
            df = pd.DataFrame(final_results).set_index("d")
            print("\n" + "=" * 25 + " all metrics " + "=" * 25)
            print(df.to_string())
            df.to_csv("mnist_vmf_results.csv")
        except ImportError:
            with open("mnist_vmf_results.json", "w") as f:
                json.dump(final_results, f, indent=2)

    script_total_time = time.time() - script_start_time
    timing_results["total_script_time_s"] = script_total_time
    with open("mnist_vmf_timing.json", "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\ntotal script time: {script_total_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vMF-VAE on MNIST — minimal table runner")

    parser.add_argument("--d_dims", type=int, nargs="+", default=[2, 5, 10, 20, 40],
                        help="latent manifold dimensions")
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_runs", type=int, default=1,
                        help="runs to average for table")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mnist-vmf")
    parser.add_argument("--simple", action="store_true",
                        help="only run knn/f1 eval, skip all VSA tests")

    args = parser.parse_args()
    run(args)

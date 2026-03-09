import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import sys
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
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
    elbo_results = []
    logger = WandbLogger(args)

    for d_manifold in args.d_dims:
        print(f"\n{'='*30}\n== vmf d={d_manifold} ==\n{'='*30}")
        model_z_dim = d_manifold + 1  # vmf in R^(d+1)
        agg_results = {s: [] for s in knn_samples}
        agg_metrics = {"ll": [], "entropy": [], "recon": [], "kl": []}

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

                vis_dir = f"visualizations/d_{d_manifold}/vmf"
                os.makedirs(vis_dir, exist_ok=True)
                test_subset = torch.utils.data.Subset(
                    test_dataset, list(range(min(500, len(test_dataset))))
                )
                test_subset_loader = DataLoader(test_subset, batch_size=64)

                self_bind = test_self_binding(
                    model, test_subset_loader, device, vis_dir, unbind_method="*"
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

                role_filler_raw = vsa_binding_unbinding(
                    d=item_memory.shape[-1],
                    n_items=500,
                    k_range=list(range(2, 21, 2)),
                    n_trials=3,
                    normalize=True,
                    device=device,
                    plot=True,
                    unbind_method="*",
                    save_dir=vis_dir,
                    item_memory=item_memory,
                    bind_with_random=True,
                )

                if logger.use:
                    knn_metrics = {k: v for k, v in knn_results.items() if k.startswith("knn_")}
                    sb_metrics = {
                        f"self_binding/{k}": v
                        for k, v in self_bind.items()
                        if isinstance(v, (int, float, bool))
                    }
                    logger.log_metrics({
                        **knn_metrics,
                        **sb_metrics,
                        "test/ll": test_metrics["ll"],
                        "test/entropy": test_metrics["entropy"],
                        "test/recon": test_metrics["recon"],
                        "test/kl": test_metrics["kl"],
                        "final_val_loss": best_val_loss,
                    })

                    images_to_log = {}
                    sp = self_bind.get("similarity_after_k_binds_plot_path")
                    if sp:
                        images_to_log["Self_Binding_*"] = sp
                    bc_plot = os.path.join(vis_dir, "bundle_capacity.png")
                    if os.path.exists(bc_plot):
                        images_to_log["Bundle_Capacity"] = bc_plot
                    rf_plot = os.path.join(vis_dir, "role_filler_capacity.png")
                    if os.path.exists(rf_plot):
                        images_to_log["Role_Filler_Capacity"] = rf_plot
                    if images_to_log:
                        logger.log_images(images_to_log)

                    logger.log_summary({**knn_metrics, **sb_metrics,
                                        "test/ll": test_metrics["ll"],
                                        "test/entropy": test_metrics["entropy"],
                                        "test/recon": test_metrics["recon"],
                                        "test/kl": test_metrics["kl"]})
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

        row_data = {"d": d_manifold}
        for n in knn_samples:
            accs = agg_results[n]
            row_data[f"vMF_{n}"] = (
                f"{np.mean(accs)*100:.1f}±{np.std(accs)*100:.1f}" if accs else "N/A"
            )
        final_results.append(row_data)

        elbo_row = {"d": d_manifold}
        for m in ["ll", "entropy", "recon", "kl"]:
            vals = agg_metrics[m]
            elbo_row[f"vMF_{m}"] = (
                f"{np.mean(vals):.2f}±{np.std(vals):.2f}" if vals else "N/A"
            )
        elbo_results.append(elbo_row)

    if final_results:
        try:
            import pandas as pd
            df = pd.DataFrame(final_results).set_index("d")
            print("\n" + "=" * 25 + " vmf knn results (%) " + "=" * 25)
            print(df.to_string())
            df.to_csv("mnist_vmf_knn_results.csv")
            if elbo_results:
                df_elbo = pd.DataFrame(elbo_results).set_index("d")
                print("\n" + "=" * 25 + " vmf elbo metrics " + "=" * 25)
                print(df_elbo.to_string())
                df_elbo.to_csv("mnist_vmf_elbo_results.csv")
        except ImportError:
            for row in final_results:
                print(row)
            with open("mnist_vmf_knn_results.json", "w") as f:
                json.dump(final_results, f, indent=2)

    with open("mnist_vmf_elbo_raw.json", "w") as f:
        json.dump(elbo_results, f, indent=2)

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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_runs", type=int, default=20,
                        help="runs to average for table")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mnist-vmf")

    args = parser.parse_args()
    run(args)

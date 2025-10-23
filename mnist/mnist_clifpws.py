import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.utils as tu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import (
    WandbLogger,
    test_self_binding,
    test_cross_class_bind_unbind,
    compute_class_means,
    evaluate_mean_vector_cosine,
    plot_clifford_manifold_visualization,
    plot_powerspherical_manifold_visualization,
    plot_gaussian_manifold_visualization,
)
from utils.vsa import (
    test_bundle_capacity as vsa_bundle_capacity,
    test_binding_unbinding_pairs as vsa_binding_unbinding,
    test_per_class_bundle_capacity_two_items,
    test_binding_unbinding_with_self_binding,
)
from mnist.mlp_vae import MLPVAE, vae_loss


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
    metric = (
        "cosine"
        if model.distribution in ["powerspherical", "clifford"]
        else "euclidean"
    )

    for n_samples in n_samples_list:
        indices = np.random.choice(len(X_train_full), n_samples, replace=False)
        X_train_sample, y_train_sample = X_train_full[indices], y_train_full[indices]

        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train_sample, y_train_sample)

        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results[f"knn_acc_{n_samples}"] = float(accuracy)
        results[f"knn_f1_{n_samples}"] = float(f1)
        print(f"  knn acc w/ {n_samples} for train, test: {accuracy:.4f}, f1: {f1:.4f}")

    return results


def plot_reconstructions(model, loader, device, filepath):
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
        plt.title("top: original images | bottom: reconstructed images")
        plt.axis("off")
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()
    return filepath


def plot_interpolations(model, loader, device, filepath, steps=10):
    print("generating interpolations...")
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        idx1 = (y == y[0]).nonzero(as_tuple=True)[0][0].item()
        idx2 = (y != y[0]).nonzero(as_tuple=True)[0][0].item()
        x1, x2 = x[idx1].unsqueeze(0).to(device), x[idx2].unsqueeze(0).to(device)

        z_mean1, _ = model.encode(x1.view(1, -1))
        z_mean2, _ = model.encode(x2.view(1, -1))

        interp_z = []
        alphas = torch.linspace(0, 1, steps, device=device)

        if model.distribution == "clifford":
            delta = z_mean2 - z_mean1
            delta_wrapped = (delta + math.pi) % (2 * math.pi) - math.pi
            interp_angles = z_mean1 + alphas.view(-1, 1) * delta_wrapped

            n = 2 * model.z_dim
            theta_s = torch.zeros(steps, n, device=device, dtype=z_mean1.dtype)
            theta_s[..., 1 : model.z_dim] = interp_angles[..., 1:]
            theta_s[..., -model.z_dim + 1 :] = -torch.flip(
                interp_angles[..., 1:], dims=(-1,)
            )
            samples_complex = torch.exp(1j * theta_s)
            interp_z = torch.fft.ifft(samples_complex, dim=-1, norm="ortho").real.to(
                torch.float32
            )

        elif model.distribution in ["powerspherical", "vmf"]:
            for alpha in alphas:
                z = (1 - alpha) * z_mean1 + alpha * z_mean2
                interp_z.append(torch.nn.functional.normalize(z, p=2, dim=-1))
            interp_z = torch.cat(interp_z, dim=0)
        else:
            for alpha in alphas:
                interp_z.append((1 - alpha) * z_mean1 + alpha * z_mean2)
            interp_z = torch.cat(interp_z, dim=0)

        x_recon_interp = torch.sigmoid(model.decoder(interp_z)).view(-1, 1, 28, 28)

        grid = tu.make_grid(x_recon_interp, nrow=steps, pad_value=0.5)
        plt.figure(figsize=(12, 2))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.title(f"latent space interpolation ({model.distribution.upper()})")
        plt.axis("off")
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()
    return filepath


def plot_latent_space(model, loader, device, filepath, n_plot=1000):
    print(f"generating t-sne plot for {n_plot} points...")
    X_z, y = encode_dataset(model, loader, device)

    # reduce memory usage
    X_z, y = X_z[:n_plot], y[:n_plot]

    # use single perplexity to avoid memory issues
    perplexity = 30
    print(f"running t-sne with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    z_tsne = tsne.fit_transform(X_z)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        z_tsne[:, 0],
        z_tsne[:, 1],
        c=y,
        cmap=plt.get_cmap("tab10", 10),
        s=10,
        alpha=0.8,
    )
    plt.title(f"t-sne latent space for {model.distribution.upper()}-vae")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    return filepath


def plot_pca_analysis(model, loader, device, filepath, n_plot=1000):
    """pca analysis: scatter + explained variance."""
    print(f"pca analysis for {n_plot} points...")
    X_z, y = encode_dataset(model, loader, device)
    X_z, y = X_z[:n_plot], y[:n_plot]

    pca = PCA(n_components=min(50, X_z.shape[1]))
    Z_pca = pca.fit_transform(X_z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sc = ax1.scatter(
        Z_pca[:, 0], Z_pca[:, 1], c=y, cmap=plt.get_cmap("tab10", 10), s=10, alpha=0.8
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax1.set_title(f"PCA of Latent Space (μ) - {model.distribution.upper()}")
    plt.colorbar(sc, ax=ax1, ticks=np.unique(y))

    n_components = min(20, len(pca.explained_variance_ratio_))
    ax2.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title(
        f"PCA Explained Variance\n(Total: {pca.explained_variance_ratio_.sum():.1%})"
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    return filepath


def run(args):
    script_start_time = time.time()
    timing_results = {}

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # data loading w/ dynamic binarization
    transform = transforms.Compose(
        [transforms.ToTensor(), BinarizeWithRandomThreshold()]
    )
    full_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    train_eval_loader = DataLoader(train_dataset, batch_size=512, num_workers=0)
    test_eval_loader = DataLoader(test_dataset, batch_size=512, num_workers=0)

    final_results = []
    distributions_to_test = ["normal", "powerspherical", "clifford"]
    knn_samples = [100, 600, 1000]
    logger = WandbLogger(args)

    for mdim in args.d_dims:
        print(f"\n{'='*30}\n==d = {mdim} ==\n{'='*30}")

        agg_results = {
            dist: {s: [] for s in knn_samples} for dist in distributions_to_test
        }

        for dist in distributions_to_test:
            # dist on sphere S^d is embedded in R^(d+1)
            if dist in ["powerspherical"]:
                model_z_dim = mdim + 1
            else:  # normal (R^d), clifford (T^d)
                model_z_dim = mdim

            if dist == "clifford" and mdim < 2:
                continue

            print(
                f"\n--- Testing {dist.upper()}-VAE with d={mdim} (model z_dim={model_z_dim}) ---"
            )

            for run in range(args.n_runs):
                print(f"\n--- Run {run+1}/{args.n_runs} ---")
                run_start_time = time.time()

                # wandb setup
                if logger.use:
                    logger.start_run(f"{dist}-d{mdim}-run{run+1}", args)

                l2_norm = args.l2_norm if dist == "normal" else False
                model = MLPVAE(
                    h_dim=args.h_dim, z_dim=model_z_dim, distribution=dist, l2_normalize=l2_norm
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                # training
                best_val_loss = float("inf")
                patience_counter = 0
                model_path = f"best_model_{dist}_d{mdim}_run{run}.pt"
                train_start_time = time.time()

                for epoch in range(args.epochs):
                    model.train()
                    beta = min(
                        1.0, (epoch + 1) / max(1, args.warmup_epochs)
                    )  # kl annealing
                    total_train_loss = 0
                    for x_mb, _ in train_loader:
                        optimizer.zero_grad()
                        loss = vae_loss(model, x_mb.to(device), beta=beta)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        total_train_loss += loss.item()

                    # val
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for x_mb, _ in val_loader:
                            total_val_loss += vae_loss(
                                model, x_mb.to(device), beta=1.0
                            ).item()

                    avg_val_loss = total_val_loss / len(val_loader)

                    if logger.use:
                        logger.log_metrics(
                            {
                                "epoch": epoch,
                                "train_loss": total_train_loss / len(train_loader),
                                "val_loss": avg_val_loss,
                                "kl_beta": beta,
                            }
                        )

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), model_path)
                    else:
                        patience_counter += 1

                    if args.patience > 0 and patience_counter >= args.patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break

                train_time = time.time() - train_start_time
                print(f"training time for {dist}-d{mdim}-run{run+1}: {train_time:.2f}s")

                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))

                    eval_start_time = time.time()
                    knn_results = perform_knn_evaluation(
                        model, train_eval_loader, test_eval_loader, device, knn_samples
                    )
                    for n_samples in knn_samples:
                        if f"knn_acc_{n_samples}" in knn_results:
                            agg_results[dist][n_samples].append(
                                knn_results[f"knn_acc_{n_samples}"]
                            )

                    test_subset = torch.utils.data.Subset(
                        test_dataset, list(range(min(1000, len(test_dataset))))
                    )
                    test_subset_loader = DataLoader(test_subset, batch_size=64)

                    # self-binding tests with both pseudo-inverse (*) and deconvolution (†)
                    fourier_pseudo = test_self_binding(
                        model,
                        test_subset_loader,
                        device,
                        f"visualizations/d_{mdim}/{dist}",
                        unbind_method="*",
                    )
                    fourier_deconv = test_self_binding(
                        model,
                        test_subset_loader,
                        device,
                        f"visualizations/d_{mdim}/{dist}",
                        unbind_method="†",
                    )

                    # cross-class binding tests with both methods
                    cross_class_pseudo = test_cross_class_bind_unbind(
                        model,
                        test_subset_loader,
                        device,
                        f"visualizations/d_{mdim}/{dist}",
                        unbind_method="*",
                    )
                    cross_class_deconv = test_cross_class_bind_unbind(
                        model,
                        test_subset_loader,
                        device,
                        f"visualizations/d_{mdim}/{dist}",
                        unbind_method="†",
                    )

                    vis_dir = f"visualizations/d_{mdim}/{dist}"
                    os.makedirs(vis_dir, exist_ok=True)

                    # prepare latent memory for vsa tests (reduced from 1000 to 500 for memory)
                    normalize_vectors = True
                    latents = []
                    labels_list = []
                    images_list = []
                    for x, y in test_eval_loader:
                        z_mean, _ = model.encode(x.to(device).view(-1, 784))
                        latents.append(z_mean.detach())
                        labels_list.append(y)
                        images_list.append(x.cpu())
                        if len(torch.cat(latents, 0)) >= 500:
                            break
                    item_memory = torch.cat(latents, 0)[:500]
                    item_labels = torch.cat(labels_list, 0)[:500].to(device)
                    item_images = torch.cat(images_list, 0)[:500]

                    # test 1: 2-items-per-class bundle capacity (no braiding)
                    print(f"running 2-items-per-class test ({dist}, no braiding)...")
                    two_per_class_res = test_per_class_bundle_capacity_two_items(
                        d=item_memory.shape[-1],
                        n_items=500,
                        n_classes=10,
                        items_per_class=2,
                        n_trials=10,
                        normalize=normalize_vectors,
                        device=device,
                        plot=True,
                        save_dir=vis_dir,
                        item_memory=item_memory,
                        labels=item_labels,
                        item_images=item_images,
                        use_braiding=False,
                    )

                    # test 2: classical bundle capacity (no braiding)
                    print(f"running classical bundle capacity ({dist}, no braiding)...")
                    bundle_cap_raw = vsa_bundle_capacity(
                        d=item_memory.shape[-1],
                        n_items=500,
                        k_range=list(range(5, 31, 5)),
                        n_trials=10,
                        normalize=normalize_vectors,
                        device=device,
                        plot=True,
                        save_dir=vis_dir,
                        item_memory=item_memory,
                        use_braiding=False,
                    )

                    # test 2b: classical bundle capacity (with braiding)
                    bundle_cap_raw_braid = {}
                    if not args.no_braid:
                        print(
                            f"running classical bundle capacity ({dist}, WITH braiding)..."
                        )
                        bundle_cap_raw_braid = vsa_bundle_capacity(
                            d=item_memory.shape[-1],
                            n_items=500,
                            k_range=list(range(5, 31, 5)),
                            n_trials=10,
                            normalize=normalize_vectors,
                            device=device,
                            plot=True,
                            save_dir=os.path.join(vis_dir, "braided"),
                            item_memory=item_memory.clone(),
                            use_braiding=True,
                        )
                    bundle_cap_res = {
                        "bundle_capacity_plot": os.path.join(
                            vis_dir, "bundle_capacity.png"
                        ),
                        "bundle_capacity_accuracies": {
                            k: acc
                            for k, acc in zip(
                                bundle_cap_raw["k"], bundle_cap_raw["accuracy"]
                            )
                        },
                    }

                    # test 3: bind-bundle-unbind (no braiding)
                    print(f"running bind-bundle-unbind test ({dist}, no braiding)...")
                    unbind_bundled_raw = vsa_binding_unbinding(
                        d=item_memory.shape[-1],
                        n_items=500,
                        k_range=list(range(5, 21, 5)),
                        n_trials=10,
                        normalize=normalize_vectors,
                        device=device,
                        plot=True,
                        save_dir=vis_dir,
                        item_memory=item_memory,
                        use_braiding=False,
                    )
                    unbind_bundled_res_inv = {
                        "unbind_bundled_plot": os.path.join(
                            vis_dir, "unbind_bundled_pairs_inv.png"
                        ),
                        "unbind_bundled_accuracies": {
                            k: acc
                            for k, acc in zip(
                                unbind_bundled_raw["k"], unbind_bundled_raw["accuracy"]
                            )
                        },
                    }

                    # test 3b: bind-bundle-unbind (WITH braiding)
                    unbind_bundled_raw_braid = {}
                    unbind_bundled_res_inv_braid = {}
                    if not args.no_braid:
                        print(f"running bind-bundle-unbind test ({dist}, WITH braiding)...")
                        unbind_bundled_raw_braid = vsa_binding_unbinding(
                            d=item_memory.shape[-1],
                            n_items=500,
                            k_range=list(range(5, 21, 5)),
                            n_trials=10,
                            normalize=normalize_vectors,
                            device=device,
                            plot=True,
                            save_dir=os.path.join(vis_dir, "braided"),
                            item_memory=item_memory.clone(),
                            use_braiding=True,
                        )
                        unbind_bundled_res_inv_braid = {
                            "unbind_bundled_plot_braid": os.path.join(
                                vis_dir, "braided", "unbind_bundled_pairs_inv_braided.png"
                            ),
                            "unbind_bundled_accuracies_braid": {
                                k: acc
                                for k, acc in zip(
                                    unbind_bundled_raw_braid["k"],
                                    unbind_bundled_raw_braid["accuracy"],
                                )
                            },
                        }

                    vis_dir = f"visualizations/d_{mdim}/{dist}"
                    os.makedirs(vis_dir, exist_ok=True)

                    recon_path = plot_reconstructions(
                        model,
                        test_eval_loader,
                        device,
                        os.path.join(vis_dir, "reconstructions.png"),
                    )
                    tsne_path = plot_latent_space(
                        model,
                        test_eval_loader,
                        device,
                        os.path.join(vis_dir, "tsne.png"),
                    )
                    pca_path = plot_pca_analysis(
                        model,
                        test_eval_loader,
                        device,
                        os.path.join(vis_dir, "pca.png"),
                    )
                    interp_path = plot_interpolations(
                        model,
                        test_eval_loader,
                        device,
                        os.path.join(vis_dir, "interpolations.png"),
                    )

                    # manifold-specific visualizations
                    if dist == "clifford" and mdim >= 2:
                        cliff_viz = plot_clifford_manifold_visualization(
                            model, device, vis_dir, n_grid=16, dims=(0, 1)
                        )
                    elif dist == "powerspherical" and mdim >= 2:
                        pow_viz = plot_powerspherical_manifold_visualization(
                            model, device, vis_dir, n_samples=1000, dims=(0, 1)
                        )
                    elif dist == "normal" and mdim >= 2:
                        gauss_viz = plot_gaussian_manifold_visualization(
                            model, device, vis_dir, n_samples=1000, dims=(0, 1)
                        )

                    if logger.use:
                        images_to_log = {
                            "Reconstructions": recon_path,
                            "Latent t-SNE": tsne_path,
                            "Latent PCA": pca_path,
                            "Interpolations": interp_path,
                        }

                        # add self-binding tests results
                        for tag, fr in {
                            "*": fourier_pseudo,
                            "†": fourier_deconv,
                        }.items():
                            if fr.get("similarity_after_k_binds_plot_path"):
                                images_to_log[f"Similarity_After_K_Binds_{tag}"] = (
                                    fr["similarity_after_k_binds_plot_path"]
                                )
                            if fr.get("recon_after_k_binds_plot_path"):
                                images_to_log[f"Recon_After_K_Binds_{tag}"] = fr[
                                    "recon_after_k_binds_plot_path"
                                ]

                        # add vsa bundle capacity tests
                        if bundle_cap_res.get("bundle_capacity_plot"):
                            images_to_log["Bundle_Capacity"] = bundle_cap_res[
                                "bundle_capacity_plot"
                            ]

                        # add braided bundle capacity plot
                        bundle_braid_plot = os.path.join(
                            vis_dir, "braided", "bundle_capacity.png"
                        )
                        if os.path.exists(bundle_braid_plot):
                            images_to_log["Bundle_Capacity_BRAIDED"] = (
                                bundle_braid_plot
                            )

                        # add two-per-class test
                        two_per_class_plot = os.path.join(
                            vis_dir, "bundle_two_per_class_similarity.png"
                        )
                        if os.path.exists(two_per_class_plot):
                            images_to_log["Bundle_Two_Per_Class_Similarity"] = (
                                two_per_class_plot
                            )

                        # add unbind bundled tests
                        if unbind_bundled_res_inv.get("unbind_bundled_plot"):
                            images_to_log["Unbind_Bundled_Inv"] = (
                                unbind_bundled_res_inv["unbind_bundled_plot"]
                            )
                        if unbind_bundled_res_inv_braid.get(
                            "unbind_bundled_plot_braid"
                        ):
                            images_to_log["Unbind_Bundled_Inv_BRAIDED"] = (
                                unbind_bundled_res_inv_braid[
                                    "unbind_bundled_plot_braid"
                                ]
                            )

                        # add cross-class binding tests
                        if cross_class_pseudo.get(
                            "cross_class_bind_unbind_plot_path"
                        ):
                            images_to_log["Cross_Class_Binding_Pseudo"] = (
                                cross_class_pseudo[
                                    "cross_class_bind_unbind_plot_path"
                                ]
                            )
                        if cross_class_deconv.get(
                            "cross_class_bind_unbind_plot_path"
                        ):
                            images_to_log["Cross_Class_Binding_Deconv"] = (
                                cross_class_deconv[
                                    "cross_class_bind_unbind_plot_path"
                                ]
                            )

                        # add manifold-specific visualizations to wandb
                        if dist == "clifford" and mdim >= 2 and cliff_viz:
                            images_to_log["Clifford_Manifold"] = cliff_viz
                        elif dist == "powerspherical" and mdim >= 2 and pow_viz:
                            images_to_log["PowerSpherical_Manifold"] = pow_viz
                        elif dist == "normal" and mdim >= 2 and gauss_viz:
                            images_to_log["Gaussian_Manifold"] = gauss_viz

                        logger.log_images(images_to_log)

                    if logger.use:
                        knn_metrics = {
                            k: v for k, v in knn_results.items() if k.startswith("knn_")
                        }

                        # collect fourier metrics from both pseudo-inverse and deconvolution
                        fourier_metrics = {}
                        fourier_metrics.update(
                            {
                                f"pseudo/{k}": v
                                for k, v in fourier_pseudo.items()
                                if isinstance(v, (int, float, bool))
                            }
                        )
                        fourier_metrics.update(
                            {
                                f"deconv/{k}": v
                                for k, v in fourier_deconv.items()
                                if isinstance(v, (int, float, bool))
                            }
                        )

                        # compute braiding metrics
                        braiding_metrics = {}

                        # model latents metrics (no braiding vs braiding)
                        if bundle_cap_raw and bundle_cap_raw_braid:
                            for k_val, acc_no, acc_yes in zip(
                                bundle_cap_raw["k"],
                                bundle_cap_raw["accuracy"],
                                bundle_cap_raw_braid["accuracy"],
                            ):
                                braiding_metrics[
                                    f"{dist}/bundle_acc_k{k_val}_no_braid"
                                ] = acc_no
                                braiding_metrics[
                                    f"{dist}/bundle_acc_k{k_val}_braid"
                                ] = acc_yes
                                braiding_metrics[
                                    f"{dist}/bundle_acc_k{k_val}_braid_delta"
                                ] = (acc_yes - acc_no)

                        # bind-bundle-unbind (no braiding vs braiding)
                        if unbind_bundled_raw and unbind_bundled_raw_braid:
                            for k_val, acc_no, acc_yes in zip(
                                unbind_bundled_raw["k"],
                                unbind_bundled_raw["accuracy"],
                                unbind_bundled_raw_braid["accuracy"],
                            ):
                                braiding_metrics[
                                    f"{dist}/unbind_bundled_acc_k{k_val}_no_braid"
                                ] = acc_no
                                braiding_metrics[
                                    f"{dist}/unbind_bundled_acc_k{k_val}_braid"
                                ] = acc_yes
                                braiding_metrics[
                                    f"{dist}/unbind_bundled_acc_k{k_val}_braid_delta"
                                ] = (acc_yes - acc_no)

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

                        logger.log_metrics(
                            {
                                **knn_metrics,
                                **fourier_metrics,
                                **braiding_metrics,
                                "mean_vector_cosine_acc": float(mean_vector_acc),
                                "final_val_loss": best_val_loss,
                                "cross_class_bind_unbind_similarity_pseudo": cross_class_pseudo.get(
                                    "cross_class_bind_unbind_similarity", 0.0
                                ),
                                "cross_class_bind_unbind_similarity_deconv": cross_class_deconv.get(
                                    "cross_class_bind_unbind_similarity", 0.0
                                ),
                            }
                        )

                        summary_metrics = {
                            **knn_metrics,
                            "final_val_loss": best_val_loss,
                            **fourier_metrics,
                            **braiding_metrics,
                            "mean_vector_cosine_acc": float(mean_vector_acc),
                            "cross_class_bind_unbind_similarity_pseudo": cross_class_pseudo.get(
                                "cross_class_bind_unbind_similarity", 0.0
                            ),
                            "cross_class_bind_unbind_similarity_deconv": cross_class_deconv.get(
                                "cross_class_bind_unbind_similarity", 0.0
                            ),
                        }
                        logger.log_summary(summary_metrics)
                        logger.finish_run()

                    eval_time = time.time() - eval_start_time
                    run_time = time.time() - run_start_time

                    # store timing info
                    timing_key = f"{dist}_d{mdim}_run{run+1}"
                    timing_results[timing_key] = {
                        "train_time_s": train_time,
                        "eval_time_s": eval_time,
                        "total_run_time_s": run_time,
                    }
                    print(
                        f"eval time: {eval_time:.2f}s, total run time: {run_time:.2f}s"
                    )

                    os.remove(model_path)

        row_data = {"d": mdim}
        for dist in distributions_to_test:
            if not any(agg_results[dist].values()):
                continue

            for n_samples in knn_samples:
                accuracies = agg_results[dist][n_samples]
                if accuracies:
                    mean_acc, std_acc = (
                        np.mean(accuracies) * 100,
                        np.std(accuracies) * 100,
                    )
                    row_data[f"{dist.upper()}_{n_samples}"] = (
                        f"{mean_acc:.1f}±{std_acc:.1f}"
                    )
                else:
                    row_data[f"{dist.upper()}_{n_samples}"] = "N/A"
        final_results.append(row_data)

    if final_results:
        import pandas as pd

        df = pd.DataFrame(final_results).set_index("d")
        print("\n" + "=" * 25 + " results (knn acc %) " + "=" * 25)
        print(df.to_string())
        df.to_csv("mnist_vae_knn_results.csv")
    else:
        print("no results were generated.")

    # save timing results
    script_total_time = time.time() - script_start_time
    timing_results["total_script_time_s"] = script_total_time
    with open("mnist_clifpws_timing.json", "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\ntotal script execution time: {script_total_time:.2f}s")
    print(f"timing results saved to mnist_clifpws_timing.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VAE experiments on MNIST, contrasting clifford/gaussian/powerspherical"
    )

    parser.add_argument(
        "--d_dims",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 40, 80],
        help="Latent manifold dimensions to test",
    )
    parser.add_argument("--h_dim", type=int, default=128, help="Hidden layer size")

    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=100, help="KL annealing warmup epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--l2_norm",
        action="store_true",
        help="L2 normalize Gaussian VAE latents (artificial hypersphere)",
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        default=20,
        help="Number of runs (original paper param is 20)",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mnist-svae-experiments",
        help="W&B project name",
    )
    parser.add_argument(
        "--no_braid",
        action="store_true",
        help="skip braiding tests",
    )

    args = parser.parse_args()
    run(args)

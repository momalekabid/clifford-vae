# evaluations on the val-selected checkpoint, run once on the test set. every test uses the mean
# code (models.mean_code); k-NN and vsa also run on one posterior sample per image (knn_sampled,
# vsa_sampled: the codes the submission's vsa tests used). scalars go to metrics.json; raw arrays for figures go to run_dir/raw/
# (figures.py never touches a model, so a plotting bug can't take down a run).
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from . import models as M
from . import vsa as V
from .data import GpuImages

KNN_BUDGETS = [100, 600, 1000]
KNN_K = 5
FID_SAMPLES = 10000
RECOVERY_DEPTHS = [1, 5, 10, 20, 30, 40]


def to_uint8(model, x):
    """decoder output or input batch -> uint8 (N, C, H, W)."""
    if model.arch == "mlp":
        x = x.view(len(x), 1, 28, 28)
    else:
        x = (x.clamp(-1, 1) + 1) / 2
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


# ---------------------------------------------------------------- likelihood


@torch.no_grad()
def test_elbo(model, images, seed, bs=1000):
    torch.manual_seed(seed)
    rec = kl = 0.0
    for idx in images.batches(bs):
        _, r, k = model.loss(images.get(idx), 1.0)
        rec += r.item() * len(idx)
        kl += k.item() * len(idx)
    return {"recon": rec / images.n, "kl": kl / images.n, "neg_elbo": (rec + kl) / images.n}


# ---------------------------------------------------------------- k-NN


def knn(model, pool, pool_y, test, test_y, seed, codes=M.mean_codes):
    """5-NN fit on n random labelled pool codes (seeded), scored on the full test set, with both
    euclidean and cosine distance. acc_n is the paper's metric: euclidean for N (davidson et al.
    2018), cosine for the rest. celeba: y is the 40 attribute matrix, accuracy is the mean over
    attributes."""
    zp = codes(model, pool).cpu().numpy()
    zt = codes(model, test).cpu().numpy()
    rng = np.random.RandomState(seed)
    primary = "euc" if model.prior == "N" else "cos"
    out = {"primary": primary, "n_test": int(len(zt))}
    for n in KNN_BUDGETS:
        idx = rng.choice(len(zp), n, replace=False)
        for name, metric in (("euc", "euclidean"), ("cos", "cosine")):
            pred = KNeighborsClassifier(KNN_K, metric=metric).fit(zp[idx], pool_y[idx]).predict(zt)
            acc = (pred == test_y).mean() if test_y.ndim == 2 else accuracy_score(test_y, pred)
            out[f"{name}_acc_{n}"] = float(acc)
        out[f"acc_{n}"] = out[f"{primary}_acc_{n}"]
    return out


# ---------------------------------------------------------------- fid


@torch.no_grad()
def fid(model, test, seed, n=FID_SAMPLES, bs=500, tries=6):
    """decoded prior samples vs the full test set, torchmetrics inception-v3 (2048 features). when
    several runs share a gpu, inception can run out of memory: then wait, halve the batch and retry."""
    for t in range(tries):
        try:
            return _fid(model, test, seed, n, bs)
        except torch.OutOfMemoryError:
            if t == tries - 1:
                raise
            torch.cuda.empty_cache()
            bs = max(50, bs // 2)
            print(f"fid: out of gpu memory, retrying with batch {bs} in {60 * (t + 1)}s", flush=True)
            time.sleep(60 * (t + 1))


def _fid(model, test, seed, n, bs):
    from torchmetrics.image.fid import FrechetInceptionDistance

    dev = next(model.parameters()).device
    f = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    rgb = lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x  # noqa: E731
    for idx in test.batches(bs):
        f.update(rgb((test.get(idx) + 1) / 2), real=True)
    g = torch.Generator().manual_seed(seed + 777)
    for i in range(0, n, bs):
        z = M.prior_samples(model, min(bs, n - i), g)
        f.update(rgb((model.decoder(z).clamp(-1, 1) + 1) / 2), real=False)
    return {"fid": float(f.compute().item()), "n_fake": n, "n_real": int(test.n)}


# ---------------------------------------------------------------- vsa


@torch.no_grad()
def vsa(model, test, groups, seed, n=V.N_ITEMS, codes=M.mean_codes):
    """all vsa tests on the mean codes of the first 1000 test images (item dimension = 2d for
    clifford). returns the results and the codes (saved for offline tests like role_filler_cross)."""
    items = codes(model, test)[:n].cpu()
    g = None if groups is None else groups[:n]
    res = V.evaluate(items, g, seed)
    u = V.normalize(items)
    res["mean_pairwise_cos"] = float(((u @ u.T).sum() - len(u)) / (len(u) * (len(u) - 1)))
    return res, items


# ---------------------------------------------------------------- raw dumps for figures


@torch.no_grad()
def raw_recon(model, test, n=10, seed=0):
    """first n test images, decoded from the mean code and from one posterior sample."""
    x = test.get(torch.arange(n, device=test.x.device))
    torch.manual_seed(seed)
    return {"x": to_uint8(model, x), "mean": to_uint8(model, model.decoder(M.mean_code(model, x))),
            "sample": to_uint8(model, model.decoder(model.sample_code(x)))}


@torch.no_grad()
def raw_recovery(model, test, groups, seed, depths=RECOVERY_DEPTHS, n_targets=3):
    """the paper's bind/unbind recovery figure: a target code is bound with m partner codes
    (other test images, unit norm) and unbound in reverse, then decoded. one row per target
    (targets from different groups), one column per depth, for both unbind methods."""
    codes = M.mean_codes(model, test)[:V.N_ITEMS]
    g = torch.Generator().manual_seed(seed)
    grp = np.arange(len(codes)) if groups is None else np.asarray(groups[:len(codes)])
    targets = [int(np.nonzero(grp == c)[0][0]) for c in np.unique(grp)[:n_targets]]
    out = {"depths": np.array(depths), "targets": np.array(targets)}
    for method in ("inv", "dagger"):
        rows = []
        for t in targets:
            others = torch.tensor([i for i in range(len(codes)) if i != t])
            partners = V.normalize(codes[others[torch.randperm(len(others), generator=g)[:max(depths)]]])
            row = [codes[t]]
            for m in depths:
                s = codes[t:t + 1]
                for i in range(m):
                    s = V.bind(s, partners[i:i + 1])
                for i in range(m - 1, -1, -1):
                    s = V.unbind(s, partners[i:i + 1], method)
                row.append(s[0])
            rows.append(to_uint8(model, model.decoder(torch.stack(row))))
        out[method] = np.stack(rows)
    return out


@torch.no_grad()
def raw_prior_samples(model, seed, n=64):
    return {"x": to_uint8(model, model.decoder(M.prior_samples(model, n, torch.Generator().manual_seed(seed))))}


@torch.no_grad()
def raw_manifold(model, test, y, n=2000, n_grid=12, n_anchors=3):
    """mnist manifold figures. latent: the encoder's mean output for the first n test images (clifford:
    the raw angles, N: mu, N_L2 / S / V: the unit direction) with labels. grid: decoded images over the
    prior's own 2-d latent space at d=2 (N: a gaussian-quantile grid, N_L2: the circle, S / V: a
    longitude x latitude grid on the sphere, C: its one circle). anchors (clifford, d >= 3): angles 1
    and 2 swept around real test images, the other angles held at that image's mean angles."""
    from scipy.stats import norm

    from .dists import clifford_embed

    dev = next(model.parameters()).device
    x = test.get(torch.arange(min(n, test.n), device=dev))
    lat = model.encode_mean(x)
    out = {"latent": lat.float().cpu().numpy(), "y": np.asarray(y[:len(x)])}
    p, d = model.prior, lat.shape[1]
    a = -np.pi + 2 * np.pi * torch.arange(n_grid, device=dev) / n_grid
    z = None
    if p == "N" and d == 2:
        q = torch.tensor(norm.ppf(np.linspace(0.05, 0.95, n_grid)), device=dev, dtype=torch.float32)
        z = torch.stack(torch.meshgrid(q, q.flip(0), indexing="xy"), -1).reshape(-1, 2)
        out["grid_shape"] = np.array([n_grid, n_grid])
    elif p == "N_L2" and d == 2:
        z = torch.stack([torch.cos(a), torch.sin(a)], -1)
        out["grid_shape"] = np.array([1, n_grid])
    elif p in ("S", "V") and d == 3:
        lon, la = a, torch.linspace(-1.2, 1.2, n_grid // 2, device=dev)
        LA, LO = torch.meshgrid(la.flip(0), lon, indexing="ij")
        z = torch.stack([torch.cos(LA) * torch.cos(LO), torch.cos(LA) * torch.sin(LO), torch.sin(LA)], -1).reshape(-1, 3)
        out["grid_shape"] = np.array([n_grid // 2, n_grid])
    elif p in M.CLIFFORD and d == 2:
        A = torch.zeros(n_grid, 2, device=dev)
        A[:, 1] = a
        z = clifford_embed(A)
        out["grid_shape"] = np.array([1, n_grid])
    if z is not None:
        out["grid"] = to_uint8(model, model.decoder(z))
    if p in M.CLIFFORD and d >= 3:
        yy = out["y"]
        idx = [int(np.nonzero(yy == c)[0][0]) for c in np.unique(yy)[:n_anchors]]
        grids = []
        for i in idx:
            A = lat[i].repeat(n_grid * n_grid, 1)
            A[:, 1], A[:, 2] = [t.reshape(-1) for t in torch.meshgrid(a, a, indexing="ij")]
            grids.append(to_uint8(model, model.decoder(clifford_embed(A))))
        out["anchors"], out["anchor_x"] = np.stack(grids), to_uint8(model, x[idx])
        out["anchor_grid"] = np.array(n_grid)
    return out


@torch.no_grad()
def raw_traversal(model, n_grid=12, dims=(1, 2)):
    """clifford only: decode a grid over two angles, the others at 0. angle 0 is unused by
    clifford_embed, so the sweep uses angles 1 and 2 (the submission swept 0 and 1). d=2 has one
    circle (angle 1), swept as a single row. -pi and pi are the same angle, so the sweep stops
    one step short of pi."""
    d = model.z_dim if model.arch == "mlp" else model.latent_dim
    dev = next(model.parameters()).device
    a = -np.pi + 2 * np.pi * torch.arange(n_grid, device=dev) / n_grid
    dims = dims if d > max(dims) else dims[:1]
    rows = n_grid if len(dims) == 2 else 1
    A = torch.zeros(rows * n_grid, d, device=dev)
    if rows == 1:
        A[:, dims[0]] = a
    else:
        A[:, dims[0]], A[:, dims[1]] = [t.reshape(-1) for t in torch.meshgrid(a, a, indexing="ij")]
    from .dists import clifford_embed

    return {"x": to_uint8(model, model.decoder(clifford_embed(A))), "n_grid": np.array(n_grid), "rows": np.array(rows),
            "dims": np.array(dims)}


def save_raw(rd, name, arrays):
    os.makedirs(os.path.join(rd, "raw"), exist_ok=True)
    tmp = os.path.join(rd, "raw", f"{name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(rd, "raw", f"{name}.npz"))


# ---------------------------------------------------------------- cost


@torch.no_grad()
def latency(model, images, dev, n=2000, bs=500, reps=3):
    """encode / decode ms per 1000 images, synchronised."""
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    x = images.get(torch.arange(min(n, images.n), device=images.x.device))
    M.mean_code(model, x[:bs])
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        z = torch.cat([M.mean_code(model, x[i:i + bs]) for i in range(0, len(x), bs)])
    sync()
    enc = (time.perf_counter() - t) / reps / len(x) * 1e6
    t = time.perf_counter()
    for _ in range(reps):
        for i in range(0, len(z), bs):
            model.decoder(z[i:i + bs])
    sync()
    return {"encode_ms_per_1k": enc, "decode_ms_per_1k": (time.perf_counter() - t) / reps / len(z) * 1e6}


# ---------------------------------------------------------------- everything


def run_all(model, data, rd, seed, dev, skip=()):
    """every evaluation for one trained model. returns the metrics dict; raw arrays go to rd/raw."""
    model.eval()
    te = GpuImages(data.test, dev, data.arch)
    out, timer = {}, time.time()

    def tick(name):
        print(f"eval {name}: {time.time() - timer:.0f}s", flush=True)

    out["test_elbo"] = test_elbo(model, te, seed)
    out["latency"] = latency(model, te, dev)
    tick("elbo")
    if "knn" not in skip and data.knn_pool is not None:
        pool = GpuImages(data.knn_pool, dev, data.arch)
        out["knn"] = knn(model, pool, data.knn_pool.y, te, data.test.y, seed)
        tick("knn")
    if data.name == "shapes3d":
        from .disentangle import composition

        out["composition"] = composition(model, data, dev, seed)
        tick("composition")
        skip = set(skip) | {"vsa", "fid"}
    if "evals" in data.meta:  # a dataset's own tests (omniglot.py)
        out.update(data.meta["evals"](model, data, dev, seed))
        tick(data.name)
        skip = set(skip) | data.meta.get("skip", set())
    if "vsa" not in skip:
        out["vsa"], items = vsa(model, te, data.test.groups, seed)
        save_raw(rd, "codes", {"codes": items.half().numpy(), "y": data.test.y[:len(items)],
                               "groups": data.test.groups[:len(items)]})
        tick("vsa")
    if "sampled" not in skip:
        samp = lambda m, im: M.sampled_codes(m, im, seed=seed)  # noqa: E731
        if data.knn_pool is not None:
            pool = GpuImages(data.knn_pool, dev, data.arch)
            out["knn_sampled"] = knn(model, pool, data.knn_pool.y, te, data.test.y, seed, samp)
        if data.name != "shapes3d":
            out["vsa_sampled"], _ = vsa(model, te, data.test.groups, seed, codes=samp)
        tick("sampled")
    if "fid" not in skip and data.arch == "cnn":
        out["fid"] = fid(model, te, seed)
        tick("fid")
    if "raw" not in skip:
        save_raw(rd, "recon", raw_recon(model, te, seed=seed))
        save_raw(rd, "recovery", raw_recovery(model, te, data.test.groups, seed))
        save_raw(rd, "prior_samples", raw_prior_samples(model, seed))
        if model.prior in M.CLIFFORD:
            save_raw(rd, "traversal", raw_traversal(model))
        if data.name == "mnist":
            save_raw(rd, "manifold", raw_manifold(model, te, data.test.y))
        tick("raw")
    return out

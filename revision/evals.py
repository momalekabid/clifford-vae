# evaluations run on the best (val-selected) checkpoint, once, on the test set.
# every evaluation uses the deterministic mean code (models.mean_codes); for clifford that
# is the 2d unitary vector, never raw angles.
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from revision import models as M
from revision.data import CACHE, GpuImages
from utils.vsa import bind, bundle, normalize_vectors, unbind, unitary_init
from utils.vsa_schlegel import evaluate_items

KNN_BUDGETS = [100, 600, 1000]
KNN_K = 5
VSA_ITEMS = 1000
FID_SAMPLES = 10000


# ---------------------------------------------------------------- elbo on test


@torch.no_grad()
def test_elbo(model, images, bs, seed):
    torch.manual_seed(seed)
    tot = {"recon": 0.0, "kl": 0.0}
    for idx in images.batches(bs):
        x = images.get(idx)
        _, rec, kl, _ = M.loss(model, x, 1.0)
        tot["recon"] += rec.item() * len(idx)
        tot["kl"] += kl.item() * len(idx)
    out = {k: v / images.n for k, v in tot.items()}
    out["neg_elbo"] = out["recon"] + out["kl"]
    return out


@torch.no_grad()
def recon_error(model, images, bs=500):
    """mean absolute error per pixel on the [0, 1] scale (comparable across datasets)."""
    err, n = 0.0, 0
    for idx in images.batches(bs):
        x = images.get(idx)
        xr = model(x)[0] if model.arch == "cnn" else torch.sigmoid(model(x)[3])
        if model.arch == "cnn":
            err += ((xr - x).abs() / 2).mean(dim=(1, 2, 3)).sum().item()
        else:
            err += (xr - x).abs().mean(dim=1).sum().item()
        n += len(idx)
    return err / n


# ---------------------------------------------------------------- k-NN


def knn(model, pool, pool_y, test, test_y, seed):
    """fit a 5-NN on n labelled pool images (random, seeded), score the FULL test set.
    euclidean for gaussian priors, cosine for S / V / C."""
    metric = "euclidean" if model.prior_name in M.EUCLIDEAN_PRIORS else "cosine"
    zp = M.mean_codes(model, pool).cpu().numpy()
    zt = M.mean_codes(model, test).cpu().numpy()
    rng = np.random.RandomState(seed)
    out = {"metric": metric, "n_test": int(len(zt))}
    for n in KNN_BUDGETS:
        idx = rng.choice(len(zp), n, replace=False)
        clf = KNeighborsClassifier(n_neighbors=KNN_K, metric=metric).fit(zp[idx], pool_y[idx])
        pred = clf.predict(zt)
        out[f"acc_{n}"] = float(accuracy_score(test_y, pred))
        out[f"f1_macro_{n}"] = float(f1_score(test_y, pred, average="macro"))
    return out


# ---------------------------------------------------------------- fid


@torch.no_grad()
def generation_fid(model, test, seed, n=FID_SAMPLES, bs=500):
    """decoded prior samples vs the full test set, torchmetrics inception-v3 (2048 features)."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    dev = next(model.parameters()).device
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    for idx in test.batches(bs):
        x = (test.get(idx) + 1) / 2
        fid.update(x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x, real=True)
    g = torch.Generator()
    g.manual_seed(seed + 777)
    done = 0
    while done < n:
        b = min(bs, n - done)
        img = (M.decode(model, M.prior_samples(model, b, g)).clamp(-1, 1) + 1) / 2
        fid.update(img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img, real=False)
        done += b
    return {"fid": float(fid.compute().item()), "n_fake": n, "n_real": int(test.n)}


# ---------------------------------------------------------------- vsa


def vsa(model, test, seed):
    """schlegel et al. (2022) sec 3.1-3.3 on the mean codes of the first 1000 test images.
    the item-memory dimension is recorded (2d for clifford); references are built at that
    dimension in revision/aggregate.py."""
    items = M.mean_codes(model, test)[:VSA_ITEMS].cpu()
    res = evaluate_items(items, seed=seed)
    # how much the codes share a common direction (affects cosine-based tests)
    unit = normalize_vectors(items)
    res["mean_pairwise_cos"] = float(((unit @ unit.T).sum() - len(unit)) / (len(unit) * (len(unit) - 1)))
    res["norm_of_mean_code"] = float(unit.mean(0).norm())
    return res, items


# ---------------------------------------------------------------- slot query (downstream task)


def slot_query(model, pool, pool_y, test, test_y, seed, slots=(1, 2, 3, 4, 6, 8), n_queries=500):
    """end-to-end symbolic query: store K images in one vector as sum_i r_i * c_i with random
    unitary role vectors r_i ("slot i"), unbind one role, and classify what comes back with the
    5-NN fit on 1000 labelled codes. no clean-up memory. K=1 is bind + unbind of a single code."""
    metric = "euclidean" if model.prior_name in M.EUCLIDEAN_PRIORS else "cosine"
    zp = M.mean_codes(model, pool).cpu()
    zt = M.mean_codes(model, test).cpu()
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(zp), 1000, replace=False)
    clf = KNeighborsClassifier(n_neighbors=KNN_K, metric=metric).fit(zp[idx].numpy(), pool_y[idx])
    D = zt.shape[1]
    g = torch.Generator()
    g.manual_seed(seed + 2)
    out = {"slots": list(slots), "accuracy": [], "clean_accuracy": float(accuracy_score(test_y[:2000], clf.predict(zt[:2000].numpy())))}
    for K in slots:
        torch.manual_seed(seed + 100 + K)
        roles = unitary_init(K, D)
        preds, want = [], []
        for _ in range(n_queries):
            pick = torch.randperm(len(zt), generator=g)[:K]
            mem = bundle(bind(roles, zt[pick]), normalize=True)
            j = int(torch.randint(0, K, (1,), generator=g))
            rec = unbind(mem.unsqueeze(0), roles[j : j + 1], method="inv")
            if model.prior_name in ("N_L2", "S", "V", "C"):
                rec = normalize_vectors(rec)
            preds.append(rec[0].numpy())
            want.append(test_y[pick[j]])
        out["accuracy"].append(float(accuracy_score(want, clf.predict(np.stack(preds)))))
    return out


# ---------------------------------------------------------------- shapes2d compositionality


class AttrNet(nn.Module):
    """small cnn predicting (shape, colour) of a 32x32 shapes2d image in [-1, 1]."""

    def __init__(self, n_shapes=4, n_colours=5):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.shape = nn.Linear(128, n_shapes)
        self.colour = nn.Linear(128, n_colours)

    def forward(self, x):
        h = self.f(x)
        return self.shape(h), self.colour(h)


ATTR_PATH = os.path.join(CACHE, "shapes2d_attrnet.pt")


def train_attr_classifier(dev, epochs=8, seed=0):
    """trained once on the full 20-pair grid (never seen by any vae). saved to the cache."""
    from revision.data import shapes2d_arrays

    if os.path.exists(ATTR_PATH):
        net = AttrNet().to(dev)
        net.load_state_dict(torch.load(ATTR_PATH, map_location=dev)["state"])
        return net.eval()
    torch.manual_seed(seed)
    a = shapes2d_arrays()
    tr, te = GpuImages(a["xcl"], dev, "cnn"), GpuImages(a["xclt"], dev, "cnn")
    ytr, yte = torch.from_numpy(a["lcl"]).to(dev), torch.from_numpy(a["lclt"]).to(dev)
    net = AttrNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    g = torch.Generator()
    g.manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for idx in tr.batches(256, g, shuffle=True):
            s, c = net(tr.get(idx))
            l = F.cross_entropy(s, ytr[idx, 0]) + F.cross_entropy(c, ytr[idx, 1])
            opt.zero_grad()
            l.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        s, c = net(te.get(torch.arange(te.n, device=dev)))
        acc = {"shape": (s.argmax(1) == yte[:, 0]).float().mean().item(),
               "colour": (c.argmax(1) == yte[:, 1]).float().mean().item()}
    tmp = ATTR_PATH + ".tmp"
    torch.save({"state": net.state_dict(), "test_acc": acc}, tmp)
    os.replace(tmp, ATTR_PATH)
    return net


@torch.no_grad()
def composition(model, seen, unseen, seen_lab, unseen_lab, dev, bs=500):
    """reconstruction on held-out shape-colour pairs vs seen pairs, plus whether the
    reconstruction still has the right shape and colour (attribute classifier)."""
    net = train_attr_classifier(dev)
    out = {}
    for name, imgs, lab in [("seen", seen, seen_lab), ("unseen", unseen, unseen_lab)]:
        lab = torch.from_numpy(lab).to(dev)
        err, ok_s, ok_c, ok_b, ok_in = 0.0, 0, 0, 0, 0
        for idx in imgs.batches(bs):
            x = imgs.get(idx)
            xr = model(x)[0]
            err += ((xr - x).abs() / 2).mean(dim=(1, 2, 3)).sum().item()
            s, c = net(xr.clamp(-1, 1))
            hs, hc = s.argmax(1) == lab[idx, 0], c.argmax(1) == lab[idx, 1]
            ok_s += hs.sum().item()
            ok_c += hc.sum().item()
            ok_b += (hs & hc).sum().item()
            s0, c0 = net(x)
            ok_in += ((s0.argmax(1) == lab[idx, 0]) & (c0.argmax(1) == lab[idx, 1])).sum().item()
        n = imgs.n
        out[name] = {"recon_mae": err / n, "shape_acc": ok_s / n, "colour_acc": ok_c / n,
                     "both_acc": ok_b / n, "classifier_acc_on_inputs": ok_in / n}
    out["recon_gap"] = out["unseen"]["recon_mae"] / out["seen"]["recon_mae"]
    return out


# ---------------------------------------------------------------- cost


@torch.no_grad()
def latency(model, images, dev, n=2000, bs=500, reps=3):
    """encode / decode time per 1000 images (ms), synchronised."""
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    idx = torch.arange(min(n, images.n), device=images.x.device)
    x = images.get(idx)
    model.mean_code(x[:bs])  # warm-up
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        codes = torch.cat([model.mean_code(x[i : i + bs]) for i in range(0, len(x), bs)])
    sync()
    enc = (time.perf_counter() - t) / reps / len(x) * 1000 * 1000
    z = codes  # mean codes are valid decoder inputs for every prior
    t = time.perf_counter()
    for _ in range(reps):
        for i in range(0, len(z), bs):
            model.decoder(z[i : i + bs])
    sync()
    dec = (time.perf_counter() - t) / reps / len(z) * 1000 * 1000
    return {"encode_ms_per_1k": enc, "decode_ms_per_1k": dec}

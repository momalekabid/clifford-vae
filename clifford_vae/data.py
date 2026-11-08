# datasets, cached once as uint8 numpy arrays, then held on the device and batched with a seeded
# permutation. splits: train (fit), val (checkpoint selection, a fixed 10% of train, seed 0, same
# for every run), test (touched once, at the end).
#
#   mnist         mlp, 28px, binarised (val/test once with a fixed seed, train dynamically)
#   fashionmnist  cnn, 32px grey
#   cifar10       cnn, 32px rgb
#   celeba        cnn, 64px rgb (center crop 140, resize 64), official splits; labels are the
#                 40 binary attributes, groups (for role-filler) are identities
#   shapes3d      cnn, 32px rgb (downsampled), 6 factors, held-out hue x shape pairs (train_disentangle.py)
#   omniglot      lives in omniglot.py (train_omniglot.py); registered here so load() finds it
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from .common import DATA

CACHE = os.path.join(DATA, "cache")
SPLIT_SEED = 0
VAL_FRAC = 0.1


@dataclass
class Split:
    x: np.ndarray                      # uint8 (N, C, H, W)
    y: Optional[np.ndarray] = None     # class labels (N,) or attribute matrix (N, A)
    groups: Optional[np.ndarray] = None  # role-filler: roles and fillers never share a group
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Data:
    name: str
    arch: str
    channels: int
    img_size: int
    train: Split
    val: Split
    test: Split
    knn_pool: Optional[Split] = None   # labelled images the k-NN is fit on
    meta: dict = field(default_factory=dict)


def _sub(sp, idx):
    return Split(sp.x[idx], None if sp.y is None else sp.y[idx], None if sp.groups is None else sp.groups[idx],
                 {k: v[idx] for k, v in sp.extra.items()})


def _val_split(sp, frac=VAL_FRAC):
    idx = np.random.RandomState(SPLIT_SEED).permutation(len(sp.x))
    nv = int(round(frac * len(sp.x)))
    return _sub(sp, np.sort(idx[nv:])), _sub(sp, np.sort(idx[:nv]))


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.npz")
    if not os.path.exists(path):
        arrays = build()
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _torchvision(name, size=None):
    from PIL import Image
    from torchvision import datasets

    cls = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST}[name]
    root = os.path.join(DATA, "torchvision")

    def conv(ds):
        xs, ys = [], []
        for img, lab in ds:
            if size is not None and img.size != (size, size):
                img = img.resize((size, size), Image.BILINEAR)
            xs.append(np.asarray(img, dtype=np.uint8)[None])
            ys.append(lab)
        return np.stack(xs), np.asarray(ys, np.int64)

    def build():
        (xtr, ytr), (xte, yte) = conv(cls(root, train=True, download=True)), conv(cls(root, train=False, download=True))
        return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}

    return _cached(name if size is None else f"{name}{size}", build)


def _hf_parquet(repo, split, config=None):
    """dataframe of one split from a huggingface parquet mirror (the official hosts are slow)."""
    import pandas as pd

    cfg = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet"))
    urls = cfg[config or list(cfg)[0]][split]
    return pd.concat([pd.read_parquet(io.BytesIO(urllib.request.urlopen(u).read())) for u in urls], ignore_index=True)


def _decode_images(col, crop=None, size=None):
    from PIL import Image

    xs = []
    for rec in col:
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        if crop:
            w, h = img.size
            l, t = (w - crop) // 2, (h - crop) // 2
            img = img.crop((l, t, l + crop, t + crop))
        if size and img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        xs.append(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
    return np.stack(xs)


def load_mnist():
    a = _torchvision("mnist")
    train, val = _val_split(Split(a["xtr"], a["ytr"], a["ytr"]))
    rng = np.random.RandomState(12345)
    test = Split(a["xte"], a["yte"], a["yte"])
    for sp in (val, test, train):
        # val/test binarised once; the train copy is only the k-NN pool (training re-binarises)
        sp.extra["binary"] = (rng.rand(*sp.x.shape) < sp.x / 255.0).astype(np.uint8)
    return Data("mnist", "mlp", 1, 28, train, val, test, knn_pool=train)


def load_fashionmnist():
    a = _torchvision("fashionmnist", size=32)
    train, val = _val_split(Split(a["xtr"], a["ytr"], a["ytr"]))
    return Data("fashionmnist", "cnn", 1, 32, train, val, Split(a["xte"], a["yte"], a["yte"]), knn_pool=train)


def load_cifar10():
    def build():
        out = {}
        for split, key in (("train", "tr"), ("test", "te")):
            df = _hf_parquet("uoft-cs/cifar10", split)
            out["x" + key] = _decode_images(df["img"])
            out["y" + key] = df["label"].to_numpy().astype(np.int64)
        assert len(out["xtr"]) == 50000 and len(out["xte"]) == 10000
        return out

    a = _cached("cifar10", build)
    train, val = _val_split(Split(a["xtr"], a["ytr"], a["ytr"]))
    return Data("cifar10", "cnn", 3, 32, train, val, Split(a["xte"], a["yte"], a["yte"]), knn_pool=train)


def load_celeba():
    def build():
        out = {}
        for split, key in (("train", "tr"), ("valid", "va"), ("test", "te")):
            df = _hf_parquet("flwrlabs/celeba", split, "img_align+identity+attr")
            attrs = [c for c in df.columns if c not in ("image", "celeb_id")]
            out["x" + key] = _decode_images(df["image"], crop=140, size=64)
            out["y" + key] = df[attrs].to_numpy().astype(np.int64)
            out["g" + key] = df["celeb_id"].to_numpy().astype(np.int64)
        out["attrs"] = np.array(attrs)
        return out

    a = _cached("celeba64", build)
    sp = {k: Split(a["x" + k], a["y" + k], a["g" + k]) for k in ("tr", "va", "te")}
    return Data("celeba", "cnn", 3, 64, sp["tr"], sp["va"], sp["te"], knn_pool=sp["tr"],
                meta={"attrs": a["attrs"].tolist()})


SHAPES3D_FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
SHAPES3D_SIZES = [10, 10, 10, 8, 4, 15]
# reviewer 1's test: held-out object-hue x shape pairs. every hue misses exactly one shape (hue h
# misses shape h mod 4), so 10 of the 40 pairs are never seen, and every hue and every shape still is.
HELD_OUT = [(h, h % 4) for h in range(10)]


def shapes3d_arrays():
    """deepmind 3d shapes (480k images, a full grid over the 6 factors), downsampled 64 -> 32px
    (2x2 mean) so the 32px cnn applies. labels are factor indices."""
    def build():
        import h5py

        raw = os.path.join(DATA, "3dshapes.h5")
        if not os.path.exists(raw):
            os.makedirs(DATA, exist_ok=True)
            urllib.request.urlretrieve("https://storage.googleapis.com/3d-shapes/3dshapes.h5", raw + ".tmp")
            os.replace(raw + ".tmp", raw)
        with h5py.File(raw, "r") as f:
            n = len(f["images"])
            x = np.empty((n, 3, 32, 32), np.uint8)
            for i in range(0, n, 20000):
                b = f["images"][i:i + 20000].astype(np.float32)
                x[i:i + 20000] = b.reshape(-1, 32, 2, 32, 2, 3).mean((2, 4)).round().astype(np.uint8).transpose(0, 3, 1, 2)
            v = f["labels"][:]
        lab = np.stack([np.unique(v[:, j], return_inverse=True)[1] for j in range(v.shape[1])], 1).astype(np.int64)
        return {"x": x, "labels": lab}

    return _cached("shapes3d32", build)


def load_shapes3d(n_train=100000, n_test=10000):
    """train / val / test on seen pairs, plus test images of the held-out pairs (meta["unseen"]).
    y = pair index (object_hue * 4 + shape); extra["labels"] = all 6 factor indices."""
    a = shapes3d_arrays()
    lab = a["labels"]
    held = np.zeros(len(lab), bool)
    for h, sh in HELD_OUT:
        held |= (lab[:, 2] == h) & (lab[:, 4] == sh)
    rng = np.random.RandomState(2026)
    seen, unseen = rng.permutation(np.nonzero(~held)[0]), rng.permutation(np.nonzero(held)[0])
    pair = lab[:, 2] * 4 + lab[:, 4]
    mk = lambda i: Split(a["x"][i], pair[i], pair[i], {"labels": lab[i]})  # noqa: E731
    nv = n_train // 10
    return Data("shapes3d", "cnn", 3, 32, mk(np.sort(seen[:n_train])), mk(np.sort(seen[n_train:n_train + nv])),
                mk(np.sort(seen[n_train + nv:n_train + nv + n_test])), knn_pool=None,
                meta={"unseen": mk(np.sort(unseen[:n_test])), "all": a, "held_out": HELD_OUT})


def _omniglot():
    from .omniglot import load_omniglot

    return load_omniglot()


LOADERS = {"mnist": load_mnist, "fashionmnist": load_fashionmnist, "cifar10": load_cifar10, "celeba": load_celeba,
           "shapes3d": load_shapes3d, "omniglot": _omniglot}


def load(name: str) -> Data:
    d = LOADERS[name]()
    n = int(os.environ.get("CVAE_SUBSET", "0"))
    if n:
        # quick checks: truncate every split to n images
        for k in ("train", "val", "test", "knn_pool"):
            sp = getattr(d, k)
            if sp is not None:
                setattr(d, k, _sub(sp, np.arange(min(n, len(sp.x)))))
    return d


class GpuImages:
    """uint8 images on the device, converted per batch. cnn: [-1, 1]. mlp: [0, 1] flat, binary
    if a fixed binarisation exists, else bernoulli-resampled per call when dynamic=True."""

    def __init__(self, sp: Split, dev, arch: str):
        self.x = torch.from_numpy(sp.x).to(dev)
        b = sp.extra.get("binary")
        self.binary = None if b is None else torch.from_numpy(b).to(dev)
        self.arch, self.n = arch, len(sp.x)

    def get(self, idx, gen=None, dynamic=False):
        if self.arch == "mlp":
            if dynamic:
                p = self.x[idx].float().flatten(1) / 255.0
                return (torch.rand(p.shape, generator=gen).to(p.device) < p).float()
            if self.binary is not None:
                return self.binary[idx].float().flatten(1)
            return self.x[idx].float().flatten(1) / 255.0
        return self.x[idx].float() / 127.5 - 1.0

    def batches(self, bs, gen=None, shuffle=False):
        order = torch.randperm(self.n, generator=gen) if shuffle else torch.arange(self.n)
        for i in range(0, self.n, bs):
            yield order[i:i + bs].to(self.x.device)

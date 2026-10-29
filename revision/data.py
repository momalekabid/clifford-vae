# datasets for the revision. everything is cached once as uint8 numpy arrays, then held
# on the gpu as tensors and batched with a seeded permutation (fast, deterministic).
#
# splits: train (fit the vae), val (early stopping / checkpoint selection, never test),
# test (touched once, at the end). val is a fixed 10% of the training set (seed 0,
# identical for every run and prior). stl-10: the vae trains on the 100k unlabelled
# images (5% held out as val); k-NN uses the 5k labelled train images and the 8k test set.
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from revision.common import REPO

DATA_ROOT = os.environ.get("REVISION_DATA", os.path.join(REPO, "data"))
CACHE = os.path.join(DATA_ROOT, "cache")
SPLIT_SEED = 0
VAL_FRAC = 0.1


@dataclass
class Split:
    x: np.ndarray  # uint8, (N, C, H, W)
    y: Optional[np.ndarray] = None  # int labels (or None)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Data:
    name: str
    arch: str  # "mlp" or "cnn"
    channels: int
    img_size: int
    train: Split
    val: Split
    test: Split
    knn_pool: Optional[Split] = None  # labelled images the k-NN is fit on
    extra_tests: Dict[str, Split] = field(default_factory=dict)  # e.g. shapes2d unseen pairs
    meta: dict = field(default_factory=dict)


def _val_split(x, y, frac=VAL_FRAC):
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(x))
    nv = int(round(frac * len(x)))
    vi, ti = np.sort(idx[:nv]), np.sort(idx[nv:])
    return Split(x[ti], None if y is None else y[ti]), Split(x[vi], None if y is None else y[vi])


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.npz")
    if not os.path.exists(path):
        arrays = build()
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _tv_to_uint8(ds, size=None):
    from PIL import Image

    xs, ys = [], []
    for img, label in ds:
        if size is not None and img.size != (size, size):
            # bilinear with antialias, same as transforms.Resize on a PIL image
            img = img.resize((size, size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.uint8)
        a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
        xs.append(a)
        ys.append(label)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def _torchvision(name, size=None):
    from torchvision import datasets

    root = os.path.join(DATA_ROOT, "torchvision")
    cls = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[name]

    def build():
        xtr, ytr = _tv_to_uint8(cls(root, train=True, download=True), size)
        xte, yte = _tv_to_uint8(cls(root, train=False, download=True), size)
        return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}

    return _cached(name if size is None else f"{name}{size}", build)


def load_mnist():
    a = _torchvision("mnist")
    train, val = _val_split(a["xtr"], a["ytr"])
    # val / test binarised once with a fixed seed (training batches are re-binarised each step)
    rng = np.random.RandomState(12345)
    val.extra["binary"] = (rng.rand(*val.x.shape) < val.x / 255.0).astype(np.uint8)
    test = Split(a["xte"], a["yte"])
    test.extra["binary"] = (rng.rand(*test.x.shape) < test.x / 255.0).astype(np.uint8)
    # a fixed binarisation of the training split, used only as the k-NN pool
    pool = Split(train.x, train.y, {"binary": (rng.rand(*train.x.shape) < train.x / 255.0).astype(np.uint8)})
    return Data("mnist", "mlp", 1, 28, train, val, test, knn_pool=pool)


def load_fashionmnist():
    a = _torchvision("fashionmnist", size=32)
    train, val = _val_split(a["xtr"], a["ytr"])
    return Data("fashionmnist", "cnn", 1, 32, train, val, Split(a["xte"], a["yte"]), knn_pool=train)


def load_cifar10():
    a = _torchvision("cifar10")
    train, val = _val_split(a["xtr"], a["ytr"])
    return Data("cifar10", "cnn", 3, 32, train, val, Split(a["xte"], a["yte"]), knn_pool=train)


def load_stl10():
    from torchvision import datasets

    root = os.path.join(DATA_ROOT, "torchvision")

    def build():
        out = {}
        for split, key in [("unlabeled", "xun"), ("train", "xtr"), ("test", "xte")]:
            x, y = _tv_to_uint8(datasets.STL10(root, split=split, download=True), size=64)
            out[key] = x
            if split != "unlabeled":
                out["y" + key[1:]] = y
        return out

    a = _cached("stl10_64", build)
    train, val = _val_split(a["xun"], None, frac=0.05)
    return Data("stl10", "cnn", 3, 64, train, val, Split(a["xte"], a["yte"]),
                knn_pool=Split(a["xtr"], a["ytr"]))


# ---------------------------------------------------------------------------
# shapes2d: shape x colour with held-out combinations (reviewer 1's compositionality test).
# 4 shapes x 5 colours = 20 pairs. 4 pairs never appear in training, one per shape, each with
# a different colour; magenta is seen with every shape. every shape and every colour is still
# seen in training. size, position and rotation vary everywhere.
# ---------------------------------------------------------------------------

SHAPES = ["circle", "square", "triangle", "star"]
COLOURS = {
    "red": (255, 40, 40),
    "green": (40, 220, 40),
    "blue": (60, 90, 255),
    "yellow": (240, 230, 40),
    "magenta": (230, 40, 230),
}
COLOUR_NAMES = list(COLOURS)
HELD_OUT = [("circle", "blue"), ("square", "red"), ("triangle", "yellow"), ("star", "green")]
SHAPES2D_SIZE = 32
_SS = 4  # supersampling for anti-aliasing


def _draw(shape, colour, r, cx, cy, rot):
    from PIL import Image, ImageDraw

    S = SHAPES2D_SIZE * _SS
    im = Image.new("RGB", (S, S), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    r, cx, cy = r * _SS, cx * _SS, cy * _SS
    if shape == "circle":
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
    else:
        if shape == "square":
            angs = rot + np.pi / 4 + np.arange(4) * np.pi / 2
            rad = np.full(4, r)
        elif shape == "triangle":
            angs = rot + np.arange(3) * 2 * np.pi / 3
            rad = np.full(3, r)
        else:  # 5-point star
            angs = rot + np.arange(10) * np.pi / 5
            rad = np.where(np.arange(10) % 2 == 0, r, 0.45 * r)
        pts = [(cx + q * np.cos(a), cy + q * np.sin(a)) for a, q in zip(angs, rad)]
        dr.polygon(pts, fill=colour)
    im = im.resize((SHAPES2D_SIZE, SHAPES2D_SIZE), Image.BOX)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def _render(pairs, n, rng):
    """n images, pairs drawn uniformly from `pairs` (list of (shape_idx, colour_idx))."""
    xs = np.zeros((n, 3, SHAPES2D_SIZE, SHAPES2D_SIZE), np.uint8)
    lab = np.zeros((n, 2), np.int64)
    for i in range(n):
        s, c = pairs[rng.randint(len(pairs))]
        r = rng.uniform(5.0, 10.0)
        cx, cy = rng.uniform(r + 1, SHAPES2D_SIZE - 1 - r, size=2)
        rot = rng.uniform(0, 2 * np.pi)
        xs[i] = _draw(SHAPES[s], COLOURS[COLOUR_NAMES[c]], r, cx, cy, rot)
        lab[i] = (s, c)
    return xs, lab


def shapes2d_arrays():
    def build():
        held = {(SHAPES.index(s), COLOUR_NAMES.index(c)) for s, c in HELD_OUT}
        allp = [(s, c) for s in range(len(SHAPES)) for c in range(len(COLOURS))]
        seen = [p for p in allp if p not in held]
        rng = np.random.RandomState(2026)
        out = {}
        out["xtr"], out["ltr"] = _render(seen, 50000, rng)
        out["xva"], out["lva"] = _render(seen, 5000, rng)
        out["xse"], out["lse"] = _render(seen, 5000, rng)  # test, seen pairs
        out["xun"], out["lun"] = _render(sorted(held), 5000, rng)  # test, unseen pairs
        # full grid (all 20 pairs) for the attribute classifier only, never shown to a vae
        out["xcl"], out["lcl"] = _render(allp, 40000, rng)
        out["xclt"], out["lclt"] = _render(allp, 4000, rng)
        return out

    return _cached("shapes2d", build)


def load_shapes2d():
    a = shapes2d_arrays()
    tr = Split(a["xtr"], a["ltr"][:, 0], {"labels": a["ltr"]})
    va = Split(a["xva"], a["lva"][:, 0], {"labels": a["lva"]})
    te = Split(a["xse"], a["lse"][:, 0], {"labels": a["lse"]})
    un = Split(a["xun"], a["lun"][:, 0], {"labels": a["lun"]})
    return Data("shapes2d", "cnn", 3, 32, tr, va, te, knn_pool=tr, extra_tests={"unseen": un},
                meta={"held_out": HELD_OUT, "shapes": SHAPES, "colours": COLOUR_NAMES})


LOADERS = {
    "mnist": load_mnist,
    "fashionmnist": load_fashionmnist,
    "cifar10": load_cifar10,
    "stl10": load_stl10,
    "shapes2d": load_shapes2d,
}


def _shrink(sp: Optional[Split], n: int) -> Optional[Split]:
    if sp is None:
        return None
    return Split(sp.x[:n], None if sp.y is None else sp.y[:n], {k: v[:n] for k, v in sp.extra.items()})


def load(name: str) -> Data:
    d = LOADERS[name]()
    n = int(os.environ.get("REVISION_SMOKE", "0"))
    if n:
        # smoke tests only: truncate every split to n images
        d.train, d.val, d.test = _shrink(d.train, n), _shrink(d.val, n), _shrink(d.test, n)
        d.knn_pool = _shrink(d.knn_pool, n)
        d.extra_tests = {k: _shrink(v, n) for k, v in d.extra_tests.items()}
    return d


# ---------------------------------------------------------------------------
# gpu-side batching
# ---------------------------------------------------------------------------


class GpuImages:
    """uint8 images on the device, converted per batch. cnn: [-1, 1]. mlp: [0, 1] flat."""

    def __init__(self, x: np.ndarray, dev, arch: str, binary: Optional[np.ndarray] = None):
        self.x = torch.from_numpy(x).to(dev)
        self.binary = None if binary is None else torch.from_numpy(binary).to(dev)
        self.arch = arch
        self.n = len(x)

    def get(self, idx, gen: Optional[torch.Generator] = None, dynamic_binarize: bool = False):
        if self.arch == "mlp":
            if self.binary is not None and not dynamic_binarize:
                return self.binary[idx].float().flatten(1)
            p = self.x[idx].float().flatten(1) / 255.0
            if dynamic_binarize:
                u = torch.rand(p.shape, generator=gen, device="cpu").to(p.device)
                return (u < p).float()
            return p
        return self.x[idx].float() / 127.5 - 1.0

    def batches(self, bs: int, gen: Optional[torch.Generator] = None, shuffle: bool = False):
        order = torch.randperm(self.n, generator=gen) if shuffle else torch.arange(self.n)
        for i in range(0, self.n, bs):
            yield order[i : i + bs].to(self.x.device)
